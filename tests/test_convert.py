import gzip
import io
import json
import zipfile
from pathlib import Path

from invoice_parser.convert import convert, detect_type, iter_sources
from invoice_parser.convert.types import Source
from invoice_parser.tabular.blocks import guess_header, split_blocks


def _src(path: Path) -> Source:
    return Source(path.name, path)


def test_csv_with_preamble_splits_into_blocks(tmp_path):
    p = tmp_path / "estimate.csv"
    p.write_text(
        "Estimate summary\nUpfront cost,Monthly cost,Currency\n0,100.5,USD\n\n\n"
        "Detailed Estimate\nRegion,Service,Monthly,Currency\nUS West (Oregon),AWS Config,23.03,USD\n"
        "US West (Oregon),S3 Standard,11.03,USD\n"
    )
    converted = convert(_src(p))
    assert converted.kind == "tables"
    blocks = split_blocks(converted.grids[0])
    assert len(blocks) == 2
    detail = blocks[1]
    assert "Detailed Estimate" in detail.context
    assert detail.first_row_number == 7
    assert guess_header(detail.rows) == 0


def test_json_records_become_a_table_with_context(tmp_path):
    p = tmp_path / "cost.json"
    p.write_text(
        json.dumps(
            {"provider": "AWS", "currency": "USD", "line_items": [{"service": "S3", "cost": 1.5}]}
        )
    )
    converted = convert(_src(p))
    grid = converted.grids[0]
    assert grid.rows == [["service", "cost"], ["S3", "1.5"]]
    assert "provider: AWS" in grid.context


def test_plain_text_and_unknown_binary(tmp_path):
    t = tmp_path / "main.tf"
    t.write_text('resource "aws_instance" "api" {\n  instance_type = "t3.large"\n}\n')
    assert convert(_src(t)).kind == "text"
    b = tmp_path / "blob.bin"
    b.write_bytes(bytes(range(256)) * 20)
    out = convert(_src(b))
    assert out.kind == "unreadable" and out.note


def test_type_is_detected_from_content_not_extension(tmp_path):
    p = tmp_path / "report.dat"
    p.write_bytes(b"%PDF-1.7 fake")
    assert detect_type(p) == "pdf"


def test_archives_expand_and_truncated_zip_is_salvaged(tmp_path):
    src_dir, work = tmp_path / "in", tmp_path / "work"
    src_dir.mkdir()
    work.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.csv", "x,y\n1,2\n" * 50)
        zf.writestr("nested/b.csv.gz", gzip.compress(b"p,q\n3,4\n"))
        zf.writestr("__MACOSX/._a.csv", "junk")
    good = buf.getvalue()
    (src_dir / "good.zip").write_bytes(good)
    (src_dir / "cut.zip").write_bytes(
        good[: good.index(b"PK\x01\x02")]
    )  # drop the central directory
    sources = list(iter_sources(src_dir, work))
    by_file = {}
    for s in sources:
        by_file.setdefault(s.display_path, []).append(s)
    assert [s.member for s in by_file["good.zip"]] == ["a.csv", "nested/b.csv.gz!/b.csv"]
    assert [s.member for s in by_file["cut.zip"]] == ["a.csv", "nested/b.csv.gz!/b.csv"]
    assert any("truncated" in w for w in by_file["cut.zip"][0].warnings)
    assert all(convert(s).kind == "tables" for s in sources)
    assert len({s.document_id for s in sources}) == len(sources)


def test_an_archive_that_unpacks_too_large_is_reported_not_unpacked(tmp_path, monkeypatch):
    import gzip
    import zipfile

    from invoice_parser.convert import sources

    bomb = tmp_path / "in" / "bomb.zip"
    bomb.parent.mkdir()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.csv", b"0" * 5_000_000)  # 5 MB that compresses to a few KB
    assert bomb.stat().st_size < 50_000
    nested = tmp_path / "in" / "nested.zip"
    with zipfile.ZipFile(nested, "w") as zf:  # the same thing one level down
        zf.writestr("inner.csv.gz", gzip.compress(b"0" * 5_000_000))
    many = tmp_path / "in" / "many.zip"
    with zipfile.ZipFile(many, "w") as zf:
        for i in range(30):
            zf.writestr(f"f{i}.csv", "a,b\n1,2\n")
    fine = tmp_path / "in" / "fine.zip"
    with zipfile.ZipFile(fine, "w") as zf:
        zf.writestr("ok.csv", "a,b\n1,2\n")

    monkeypatch.setattr(sources, "MAX_UNPACKED_BYTES", 1_000_000)
    monkeypatch.setattr(sources, "MAX_MEMBERS", 20)
    work = tmp_path / "work"
    work.mkdir()
    found = {s.display_path: s for s in sources.iter_sources(bomb.parent, work)}
    assert "more than" in found["bomb.zip"].error and "more than" in found["nested.zip"].error
    assert "files inside" in found["many.zip"].error
    assert found["fine.zip"].error is None and found["fine.zip"].member == "ok.csv"
    written = sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
    assert written < 3_500_000  # nothing was allowed to run past its limit
