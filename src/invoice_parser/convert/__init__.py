"""Convert any file into tables, numbered text, or page images. No interpretation happens here."""

from __future__ import annotations

import json
from pathlib import Path

from . import tables, text
from .sources import iter_sources
from .types import Converted, Grid, LazyTable, PageImage, Source, TextDoc

__all__ = [
    "Converted",
    "Grid",
    "LazyTable",
    "PageImage",
    "Source",
    "TextDoc",
    "convert",
    "detect_type",
    "iter_sources",
]

# Files larger than this are not loaded whole as JSON or text.
MAX_INLINE_BYTES = 60 * 1024 * 1024

_EXT_TYPES = {
    "csv": "csv", "tsv": "tsv", "tab": "tsv", "psv": "csv", "dat": "csv",
    "xlsx": "xlsx", "xlsm": "xlsx", "xls": "xls", "xlsb": "xlsb", "ods": "ods",
    "parquet": "parquet", "pq": "parquet",
    "json": "json", "jsonl": "jsonl", "ndjson": "jsonl",
    "pdf": "pdf",
    "png": "image", "jpg": "image", "jpeg": "image", "tif": "image", "tiff": "image",
    "webp": "image", "gif": "image", "bmp": "image",
    "docx": "docx", "pptx": "pptx",
    "html": "html", "htm": "html",
    "yaml": "text", "yml": "text", "tf": "text", "tfvars": "text", "hcl": "text", "txt": "text",
    "md": "text", "log": "text", "xml": "text", "toml": "text", "ini": "text", "template": "text",
}  # fmt: skip

_MAGIC = [
    (b"%PDF", "pdf"),
    (b"PAR1", "parquet"),
    (b"\x89PNG", "image"),
    (b"\xff\xd8\xff", "image"),
    (b"GIF8", "image"),
    (b"II*\x00", "image"),
    (b"MM\x00*", "image"),
    (b"\xd0\xcf\x11\xe0", "xls"),
]


def detect_type(path: Path, name: str | None = None) -> str:
    """File type from magic bytes first, then extension, then content."""
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return "unknown"
    ext = (name or path.name).rsplit(".", 1)[-1].lower() if "." in (name or path.name) else ""
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind if not (kind == "xls" and ext in ("doc", "ppt", "msg")) else "unknown"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head.startswith(b"PK\x03\x04"):
        if ext in ("xlsx", "xlsm", "ods", "docx", "pptx", "xlsb"):
            return _EXT_TYPES.get(ext, "xlsx")
        return "unknown"
    if ext in _EXT_TYPES:
        return _EXT_TYPES[ext]
    if not head:
        return "empty"
    sample = tables.decode_bytes(head)
    printable = sum(c.isprintable() or c in "\r\n\t" for c in sample) / max(len(sample), 1)
    if printable < 0.9:
        return "unknown"
    stripped = sample.lstrip()
    if stripped.startswith(("{", "[")):
        return "json"
    if stripped.lower().startswith(("<!doctype html", "<html")):
        return "html"
    return "csv" if tables.looks_delimited(sample) else "text"


def _convert_json(path: Path, file_type: str) -> Converted:
    raw = tables.decode_bytes(path.read_bytes())
    try:
        if file_type == "jsonl":
            data = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
        else:
            data = json.loads(raw)
    except json.JSONDecodeError:
        return text.convert_plain_text(raw, file_type)
    grids = tables.json_to_grids(data)
    if grids:
        return Converted("tables", file_type, "json-records", grids=grids)
    return text.convert_plain_text(json.dumps(data, indent=1, ensure_ascii=False), file_type)


def convert(source: Source) -> Converted:
    """Never raises: a file that cannot be read comes back as kind 'unreadable' with the reason."""
    if source.error:
        return Converted("unreadable", "unknown", "none", note=source.error)
    path = source.local_path
    file_type = detect_type(path, source.name)
    try:
        if file_type == "empty" or path.stat().st_size == 0:
            return Converted("empty", file_type, "none", note="file is empty")
        if file_type in ("csv", "tsv"):
            return tables.convert_delimited(path, file_type)
        if file_type in ("xlsx", "xls", "xlsb", "ods"):
            return tables.convert_spreadsheet(path, file_type)
        if file_type == "parquet":
            return tables.convert_parquet(path, file_type)
        if file_type == "pdf":
            return text.convert_pdf(path, file_type)
        if file_type == "image":
            return text.convert_image(path, file_type)
        if file_type == "docx":
            return text.convert_docx(path, file_type)
        if file_type == "pptx":
            return text.convert_pptx(path, file_type)
        if file_type in ("json", "jsonl", "html", "text"):
            if path.stat().st_size > MAX_INLINE_BYTES:
                return Converted(
                    "unreadable", file_type, "none", note="text file too large to load; not tabular"
                )
            if file_type in ("json", "jsonl"):
                return _convert_json(path, file_type)
            raw = tables.decode_bytes(path.read_bytes())
            if file_type == "html":
                return text.convert_html(raw, file_type)
            if (
                file_type == "text"
                and source.name.lower().endswith(".txt")
                and tables.looks_delimited(raw)
            ):
                return tables.convert_delimited(path, "csv")
            return text.convert_plain_text(raw, file_type)
        return Converted("unreadable", file_type, "none", note="unrecognized binary file type")
    except Exception as exc:
        return Converted(
            "unreadable", file_type, "none", note=f"{type(exc).__name__} while reading: {exc}"
        )
