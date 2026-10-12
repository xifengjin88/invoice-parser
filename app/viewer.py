"""Side-by-side viewer: a source file next to what the parser extracted from it.

    uv run --extra viewer streamlit run app/viewer.py

Reads results from an output directory (default `out/`) and the originals from the input
directory (default `cost-estimate-inputs/`). Nothing here calls the model.
"""

from __future__ import annotations

import json
import logging
import tempfile
from decimal import Decimal
from pathlib import Path

import polars as pl
import streamlit as st

from invoice_parser.convert import convert, iter_sources
from invoice_parser.convert.text import line_boxes, page_words
from invoice_parser.convert.types import Source
from invoice_parser.schema import COST_FIELDS, DocumentRecord
from invoice_parser.validate import DOCUMENT, LINE_ITEMS

logging.getLogger("pdfminer").setLevel(logging.ERROR)

RENDER_SCALE = 2.0
STATUS_ICON = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "➖"}
MAIN_COLUMNS = [
    "source_page", "source_row", "service_name", "region_id", "sku_meter", "charge_description",
    "consumed_quantity", "consumed_unit", "contracted_unit_price", "billed_cost",
    "billing_currency", "charge_category", "instance_type", "sub_account_id", "match_level",
    "is_rollup",
]  # fmt: skip
MAX_TABLE_ROWS = 5000

st.set_page_config(page_title="Invoice Parser Viewer", layout="wide")


# --- loading ------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def list_documents(out_dir: str) -> list[dict]:
    docs = []
    for path in sorted(Path(out_dir).glob(f"*/{DOCUMENT}")):
        record = DocumentRecord.model_validate_json(path.read_text())
        name = record.source_file + (f" › {record.source_member}" if record.source_member else "")
        docs.append(
            {
                "dir": str(path.parent),
                "name": name,
                "status": record.validation.status,
                "mtime": path.stat().st_mtime,
            }
        )
    return docs


@st.cache_data(show_spinner=False)
def load_record(doc_dir: str, mtime: float) -> DocumentRecord:
    return DocumentRecord.model_validate_json((Path(doc_dir) / DOCUMENT).read_text())


@st.cache_data(show_spinner=False)
def load_rows(doc_dir: str, mtime: float) -> pl.DataFrame:
    path = Path(doc_dir) / LINE_ITEMS
    if not path.is_file():
        return pl.DataFrame()
    return pl.scan_parquet(path).head(200_000).collect()


@st.cache_data(show_spinner=False)
def load_summary(doc_dir: str, mtime: float) -> dict:
    """Counts and totals over the whole result, however large."""
    path = Path(doc_dir) / LINE_ITEMS
    if not path.is_file():
        return {"rows": 0, "live": 0, "total": None}
    cost = pl.coalesce([pl.col(c) for c in COST_FIELDS])
    is_live = ~pl.col("is_rollup").fill_null(False)
    out = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            is_live.sum().alias("live"),
            cost.filter(is_live).sum().alias("total"),
        )
        .collect()
        .row(0, named=True)
    )
    return out if out["rows"] else {"rows": 0, "live": 0, "total": None}


@st.cache_data(show_spinner=False)
def load_tags(doc_dir: str, mtime: float) -> dict[tuple[int, int], list[dict]]:
    path = Path(doc_dir) / "tags.jsonl"
    tags: dict[tuple[int, int], list[dict]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            tags.setdefault((entry["page"], entry["line"]), []).append(entry["tag"])
    return tags


@st.cache_resource(show_spinner=False)
def _workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="invoice-viewer-"))


@st.cache_resource(show_spinner="Locating the source file…")
def locate_source(input_dir: str, source_file: str, document_id: str) -> Source | None:
    """The original file on disk; archive members are unpacked to a temporary folder."""
    path = Path(input_dir) / source_file
    if not path.exists():
        return None
    for source in iter_sources(path, _workdir()):
        source.display_path = source_file
        if source.document_id == document_id:
            return source
    return None


@st.cache_resource(show_spinner="Reading the source file…")
def load_converted(local_path: str, name: str):
    return convert(Source(name, Path(local_path)))


@st.cache_data(show_spinner=False)
def pdf_page_count(local_path: str) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(local_path)
    try:
        return len(pdf)
    finally:
        pdf.close()


@st.cache_data(show_spinner="Rendering page…")
def render_page(local_path: str, page: int):
    """Page image plus the pixel box of every text line, in the parser's line order."""
    import pdfplumber
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(local_path)
    try:
        image = pdf[page - 1].render(scale=RENDER_SCALE).to_pil().convert("RGB")
    finally:
        pdf.close()
    with pdfplumber.open(local_path) as plumber:
        boxes = line_boxes(page_words(plumber.pages[page - 1]))
    return image, [tuple(v * RENDER_SCALE for v in box) for box in boxes]


def highlight(image, boxes, extracted: set[int], rollups: set[int], selected: int | None):
    """Tint the lines that produced rows, on a transparent layer so the text stays readable."""
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    pad = 3
    for number, (x0, top, x1, bottom) in enumerate(boxes, start=1):
        rect = (x0 - pad, top - pad, x1 + pad, bottom + pad)
        if number == selected:
            draw.rectangle(rect, fill=(255, 196, 0, 90), outline=(214, 120, 0, 255), width=3)
        elif number in extracted:
            draw.rectangle(rect, fill=(0, 150, 136, 46))
        elif number in rollups:
            draw.rectangle(rect, fill=(120, 120, 120, 40))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


@st.cache_data(show_spinner=False, ttl=10)
def list_runs(base: str) -> list[dict]:
    """Result folders under `base`, each labelled with the model that produced it."""
    runs = []
    for folder in sorted(p for p in Path(base).iterdir() if p.is_dir()):
        first = next(iter(sorted(folder.glob(f"*/{DOCUMENT}"))), None)
        if first is None:
            continue
        models: set[str] = set()
        for path in folder.glob(f"*/{DOCUMENT}"):
            usage = json.loads(path.read_text()).get("llm_usage") or {}
            if usage.get("model"):
                models.add(f"{usage.get('provider', '?')} · {usage['model']}")
        runs.append({"dir": str(folder), "name": folder.name, "model": ", ".join(sorted(models))})
    return runs


def run_label(run: dict) -> str:
    return f"{run['name']}  ({run['model'] or 'model not recorded'})"


SAME_ROW = ["source_page", "source_row"]
NOT_COMPARED = {
    "line_item_id", "document_id", "source_file", "source_member", "source_text",
    "schema_version", "extraction_method", "derived_fields",
}  # fmt: skip


def _keyed(df: pl.DataFrame) -> dict[tuple, dict]:
    """Rows by the source line they came from (and their order on it, for lines giving several)."""
    out: dict[tuple, dict] = {}
    seen: dict[tuple, int] = {}
    for row in df.to_dicts():
        place = tuple(row.get(k) for k in SAME_ROW)
        seen[place] = seen.get(place, 0) + 1
        out[(*place, seen[place])] = row
    return out


def _same(a, b) -> bool:
    if a in (None, "") and b in (None, ""):
        return True
    if isinstance(a, Decimal) and isinstance(b, Decimal):
        return a == b
    return str(a) == str(b)


def compare_rows(a: pl.DataFrame, b: pl.DataFrame) -> tuple[list[dict], list[dict], int]:
    """Rows found by only one run, fields on which shared rows disagree, and rows that agree."""
    ka, kb = _keyed(a), _keyed(b)
    brief = [
        "charge_description",
        "billed_cost",
        "billing_currency",
        "charge_category",
        "is_rollup",
    ]
    one_sided = [
        {
            "only in": side,
            "page": key[0],
            "line": key[1],
            **{f: str(row.get(f) or "") for f in brief},
        }
        for side, mine, other in (("A", ka, kb), ("B", kb, ka))
        for key, row in mine.items()
        if key not in other
    ]
    differing, agree = [], 0
    for key in ka.keys() & kb.keys():
        fields = [
            f for f in ka[key] if f not in NOT_COMPARED and not _same(ka[key][f], kb[key].get(f))
        ]
        agree += not fields
        differing += [
            {
                "page": key[0], "line": key[1],
                "row": str(ka[key].get("charge_description") or "")[:60],
                "field": f, "A": str(ka[key][f] or ""), "B": str(kb[key].get(f) or ""),
            }
            for f in fields
        ]  # fmt: skip
    order = lambda r: (r["page"] or 0, r["line"] or 0)  # noqa: E731
    return sorted(one_sided, key=order), sorted(differing, key=order), agree


def run_facts(record: DocumentRecord, summary: dict, raw: dict) -> dict:
    usage = (raw.get("llm_usage") or {}).get("total") or {}
    recon = next((c for c in record.validation.checks if c.name == "reconciliation"), None)
    match = (recon.metrics or {}).get("match") if recon else None
    return {
        "status": f"{STATUS_ICON.get(record.validation.status, '')} {record.validation.status}",
        "line items": summary["live"],
        "rollup rows": summary["rows"] - summary["live"],
        "extracted total": money(summary["total"] if summary["live"] else None),
        "total match": f"{match:.2%}" if match is not None else "—",
        "model calls": int(usage.get("calls", 0)),
        "from saved answers": int(usage.get("cached", 0)),
        "input tokens": int(usage.get("prompt_tokens", 0)),
        "output tokens": int(usage.get("completion_tokens", 0)),
        "model seconds": round(usage.get("seconds", 0)),
    }


# --- small helpers ------------------------------------------------------------------------


def money(value) -> str:
    return "—" if value is None else f"{Decimal(str(value)):,.2f}"


def display_frame(df: pl.DataFrame, columns: list[str] | None) -> pl.DataFrame:
    if columns:
        df = df.select([c for c in columns if c in df.columns])
    casts = [pl.col(c).cast(pl.Float64) for c, t in df.schema.items() if t.is_decimal()]
    lists = [pl.col(c).list.join(", ") for c, t in df.schema.items() if isinstance(t, pl.List)]
    return df.with_columns(casts + lists)


def document_totals(record: DocumentRecord) -> list:
    keep = ("document", "pre_tax", "table")
    return [
        t
        for t in record.stated_totals
        if t.scope in keep or (t.origin == "summary" and t.scope not in ("tax", "credits"))
    ]


# --- sidebar ------------------------------------------------------------------------------

with st.sidebar:
    st.header("Invoice Parser Viewer")
    runs = list_runs(".")
    if not runs:
        st.warning("No results here yet. Run `invoice-parser parse <files> -o out` first.")
        st.stop()
    run = st.selectbox(
        "Run",
        runs,
        format_func=run_label,
        help="Each results folder is one run. Parse with a different `-o` folder per model "
        "(for example `-o out-claude --provider claude`) to get one entry each.",
    )
    out_dir = run["dir"]
    input_dir = st.text_input("Source files folder", "cost-estimate-inputs")
    docs = list_documents(out_dir)
    if st.button("Reload results"):
        st.cache_data.clear()
        st.rerun()
    if not docs:
        st.warning(f"No parsed documents in `{out_dir}/`. Run `invoice-parser parse` first.")
        st.stop()
    statuses = sorted({d["status"] for d in docs})
    shown = st.multiselect("Status", statuses, default=statuses)
    docs = [d for d in docs if d["status"] in shown] or docs
    doc = st.selectbox(
        "Document",
        docs,
        format_func=lambda d: f"{STATUS_ICON.get(d['status'], '')} {d['name']}",
    )
    others = [
        {**r, "doc": d}
        for r in runs
        if r["dir"] != out_dir
        for d in list_documents(r["dir"])
        if d["name"] == doc["name"]
    ]
    other = st.selectbox(
        "Compare with",
        [None, *others],
        format_func=lambda r: "nothing" if r is None else run_label(r),
        help="Another run that parsed this same document.",
    )
    hide_rollups = st.checkbox("Hide rollup rows (summaries that restate detail)", value=True)
    all_columns = st.checkbox("Show every column", value=False)

record = load_record(doc["dir"], doc["mtime"])
rows = load_rows(doc["dir"], doc["mtime"])
tags = load_tags(doc["dir"], doc["mtime"])
live = rows.filter(~pl.col("is_rollup").fill_null(False)) if rows.height else rows

# --- header -------------------------------------------------------------------------------

st.subheader(doc["name"])
bits = [record.doc_type or "not parsed", ", ".join(record.providers) or "provider unknown"]
if record.extraction_method:
    bits.append(record.extraction_method.replace("_", " "))
if record.description:
    bits.append(record.description)
st.caption(" · ".join(bits))

summary = load_summary(doc["dir"], doc["mtime"])
extracted_total = summary["total"] if summary["live"] else None
totals = document_totals(record)
stated = totals[0] if totals else None
recon = next((c for c in record.validation.checks if c.name == "reconciliation"), None)
if recon and recon.examples:
    best = recon.examples[0]
    stated_value, stated_label = (
        Decimal(best["stated"]),
        f"{best['stated_label']} ({best['currency']})",
    )
elif stated:
    stated_value, stated_label = Decimal(stated.amount), f"{stated.label} ({stated.currency})"
else:
    stated_value, stated_label = None, "none stated"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Validation", f"{STATUS_ICON.get(record.validation.status, '')} {record.validation.status}"
)
m2.metric(
    "Line items",
    f"{summary['live']:,}",
    help=f"{summary['rows'] - summary['live']:,} rollup rows excluded",
)
m3.metric("Extracted total", money(extracted_total))
m4.metric("Stated total", money(stated_value), help=stated_label)
if extracted_total is not None and stated_value:
    diff = Decimal(str(extracted_total)) - stated_value
    match = max(0.0, 1 - abs(float(diff / stated_value)))
    m5.metric("Total match", f"{match:.2%}", f"{money(diff)} difference", delta_color="off")
else:
    m5.metric("Total match", "—")

with st.expander(
    "Validator checks, warnings and stated totals", expanded=record.validation.status == "fail"
):
    for check in record.validation.checks:
        st.markdown(f"{STATUS_ICON.get(check.status, '')} **{check.name}** — {check.summary}")
        if check.examples and check.status in ("warn", "fail"):
            st.dataframe(
                pl.DataFrame(
                    [
                        {k: str(v) for k, v in e.items()}
                        if isinstance(e, dict)
                        else {"example": str(e)}
                        for e in check.examples
                    ]
                ),
                hide_index=True,
            )
    if record.error:
        st.error(record.error)
    for warning in record.warnings:
        st.warning(warning)
    if record.stated_totals:
        st.markdown("**Totals the document states**")
        st.dataframe(
            pl.DataFrame([t.model_dump() for t in record.stated_totals], infer_schema_length=None),
            hide_index=True,
        )

if other:
    theirs = other["doc"]
    their_rows = load_rows(theirs["dir"], theirs["mtime"])
    one_sided, differing, agree = compare_rows(rows, their_rows)
    with st.expander(f"Compare: A = {run_label(run)}   vs   B = {run_label(other)}", expanded=True):
        facts = {
            "A": run_facts(record, summary, json.loads(Path(doc["dir"], DOCUMENT).read_text())),
            "B": run_facts(
                load_record(theirs["dir"], theirs["mtime"]),
                load_summary(theirs["dir"], theirs["mtime"]),
                json.loads(Path(theirs["dir"], DOCUMENT).read_text()),
            ),
        }
        st.dataframe(
            pl.DataFrame(
                [{"": k, "A": str(facts["A"][k]), "B": str(facts["B"][k])} for k in facts["A"]]
            ),
            hide_index=True,
        )
        st.markdown(
            f"**{agree}** rows agree on every field · "
            f"**{len({(d['page'], d['line']) for d in differing})}** rows differ · "
            f"**{sum(r['only in'] == 'A' for r in one_sided)}** only in A · "
            f"**{sum(r['only in'] == 'B' for r in one_sided)}** only in B"
        )
        if one_sided:
            st.markdown("**Rows only one run produced**")
            st.dataframe(pl.DataFrame(one_sided, infer_schema_length=None), hide_index=True)
        if differing:
            st.markdown("**Same source line, different values**")
            fields = sorted({d["field"] for d in differing})
            chosen = st.multiselect("Fields", fields, default=fields)
            st.dataframe(
                pl.DataFrame(
                    [d for d in differing if d["field"] in chosen] or differing[:0],
                    infer_schema_length=None,
                ),
                hide_index=True,
            )

source = locate_source(input_dir, record.source_file, record.document_id)
left, right = st.columns(2, gap="medium")
is_text = bool(record.freeform) or (rows.height and rows["source_page"].null_count() < rows.height)

# --- free-form documents: page beside its rows --------------------------------------------

if is_text:
    table = rows.filter(~pl.col("is_rollup").fill_null(False)) if hide_rollups else rows
    pages_with_rows = (
        sorted(table["source_page"].drop_nulls().unique().to_list()) if table.height else []
    )
    is_pdf = source is not None and record.file_type == "pdf"
    converted = load_converted(str(source.local_path), source.name) if source else None
    n_pages = (
        pdf_page_count(str(source.local_path)) if is_pdf
        else len(converted.text.pages) if converted and converted.text
        else (max(pages_with_rows) if pages_with_rows else 1)
    )  # fmt: skip

    with left:
        c1, c2 = st.columns([1, 2])
        page = c1.number_input(
            "Page",
            1,
            max(n_pages, 1),
            value=pages_with_rows[0] if pages_with_rows else 1,
            key=f"page-{doc['dir']}",
        )
        c2.caption(
            f"{n_pages} page(s); extracted rows on {len(pages_with_rows)} of them. "
            "Green: extracted line. Grey: rollup. Orange: the row selected on the right."
        )

    page_rows = table.filter(pl.col("source_page") == page) if table.height else table
    with right:
        st.markdown(f"**Extracted from page {page}** — {page_rows.height} row(s)")
        event = st.dataframe(
            display_frame(page_rows, None if all_columns else MAIN_COLUMNS[1:]),
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=520,
            key=f"rows-{doc['dir']}-{page}",
        )
        picked = event.selection.rows[0] if event.selection.rows else None
        selected_line = None
        if picked is not None and picked < page_rows.height:
            row = page_rows.row(picked, named=True)
            selected_line = row["source_row"]
            st.markdown("**Selected row**")
            st.code(row["source_text"] or "", language=None)
            filled = {
                k: (str(v) if isinstance(v, Decimal) else v)
                for k, v in row.items()
                if v not in (None, [], "")
            }
            d1, d2 = st.columns(2)
            d1.caption("Every filled field")
            d1.json(json.loads(json.dumps(filled, default=str)), expanded=False)
            d2.caption("What the model said about this line")
            d2.json(tags.get((page, selected_line), []), expanded=False)

    with left:
        on_page = rows.filter(pl.col("source_page") == page) if rows.height else rows
        kept = (
            set(on_page.filter(~pl.col("is_rollup").fill_null(False))["source_row"].to_list())
            if on_page.height
            else set()
        )
        rolled = (set(on_page["source_row"].to_list()) - kept) if on_page.height else set()
        if is_pdf:
            image, boxes = render_page(str(source.local_path), int(page))
            st.image(highlight(image, boxes, kept, rolled, selected_line), width="stretch")
        elif converted and converted.text and page <= len(converted.text.pages):
            lines = converted.text.pages[page - 1]
            marks = {n: "●" for n in kept} | {n: "○" for n in rolled}
            if selected_line:
                marks[selected_line] = "▶"
            st.code(
                "\n".join(f"{marks.get(n, ' ')} {n:>3}  {text}" for n, text in enumerate(lines, 1)),
                language=None,
            )
            st.caption("● extracted line · ○ rollup · ▶ selected")
        else:
            st.info(
                f"Source file not found under `{input_dir}/`, so only the extracted rows are shown."
            )
        with st.expander("Page text as the parser saw it"):
            if converted and converted.text and page <= len(converted.text.pages):
                st.code(
                    "\n".join(
                        f"{n:>3}  {t}" for n, t in enumerate(converted.text.pages[page - 1], 1)
                    ),
                    language=None,
                )

# --- tabular documents: raw table beside its rows -----------------------------------------

else:
    table = live if hide_rollups else rows
    with left:
        st.markdown("**Source file**")
        if source is None:
            st.info(f"Source file not found under `{input_dir}/`.")
        else:
            converted = load_converted(str(source.local_path), source.name)
            if converted.lazy_tables:
                lazy = converted.lazy_tables[0]
                st.caption(
                    f"Large file, streamed: {len(lazy.columns)} columns. First {len(lazy.preview)} rows shown."
                )
                width = len(lazy.columns)
                body = [(r + [""] * width)[:width] for r in lazy.preview]
                st.dataframe(pl.DataFrame(body, schema=lazy.columns, orient="row"), height=560)
            elif converted.grids:
                names = [g.name or f"table {i + 1}" for i, g in enumerate(converted.grids)]
                pick = st.selectbox(
                    "Sheet / table", range(len(names)), format_func=lambda i: names[i]
                )
                grid = converted.grids[pick]
                width = max(len(r) for r in grid.rows)
                body = [r + [""] * (width - len(r)) for r in grid.rows[:MAX_TABLE_ROWS]]
                st.caption(
                    f"{len(grid.rows):,} rows as they appear in the file (row 1 is the first line)."
                )
                st.dataframe(
                    pl.DataFrame(
                        body, schema=[f"col {i + 1}" for i in range(width)], orient="row"
                    ).with_row_index("row", 1),
                    hide_index=True,
                    height=560,
                )
            else:
                st.info(converted.note or "Nothing readable in this file.")
    with right:
        st.markdown(
            f"**Extracted rows** — {table.height:,}"
            + (" (first 200,000 loaded)" if rows.height >= 200_000 else "")
        )
        st.dataframe(
            display_frame(
                table.head(MAX_TABLE_ROWS),
                None if all_columns else MAIN_COLUMNS[1:] + ["sku_id", "usage_type", "resource_id"],
            ),
            hide_index=True,
            height=560,
        )
        if table.height > MAX_TABLE_ROWS:
            st.caption(
                f"Showing the first {MAX_TABLE_ROWS:,} rows. `source_row` is the row number in the source file."
            )
        with st.expander("How each table was mapped"):
            st.json(record.tables, expanded=False)
