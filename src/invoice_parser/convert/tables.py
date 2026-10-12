"""Tabular inputs: delimited text, spreadsheets, Parquet, JSON records."""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

from .types import Converted, Grid, LazyTable

# Above this size a delimited file is streamed instead of loaded.
LAZY_BYTES = 25 * 1024 * 1024
DELIMITERS = [",", "\t", ";", "|"]


def decode_bytes(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def sniff_delimiter(sample: str) -> tuple[str, float]:
    """Best delimiter and the share of non-empty lines that agree on its column count."""
    lines = [ln for ln in sample.splitlines() if ln.strip()][:200]
    best, best_score = ",", 0.0
    for delim in DELIMITERS:
        try:
            widths = [len(r) for r in csv.reader(lines, delimiter=delim)]
        except csv.Error:
            continue
        wide = [w for w in widths if w > 1]
        if not wide:
            continue
        mode = max(set(wide), key=wide.count)
        score = wide.count(mode) / len(widths) * min(mode, 4) / 4
        if score > best_score:
            best, best_score = delim, score
    return best, best_score


def cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()


def convert_delimited(path: Path, file_type: str) -> Converted:
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = decode_bytes(fh.read(256 * 1024))
    delim, score = sniff_delimiter(head)
    if file_type == "tsv":
        delim = "\t"
    if size > LAZY_BYTES:
        lines = head.splitlines()[:-1][:60]
        preview = list(csv.reader(lines, delimiter=delim))
        if not preview:
            return Converted("empty", file_type, "delimited", note="no rows")
        columns = dedupe_headers(preview[0])
        table = LazyTable(path, "csv", columns, preview[1:], separator=delim)
        return Converted("tables", file_type, "delimited-stream", lazy_tables=[table])
    text = decode_bytes(path.read_bytes())
    rows = [
        [c.strip() for c in row]
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    ]
    if not any(any(r) for r in rows):
        return Converted("empty", file_type, "delimited", note="no rows")
    return Converted("tables", file_type, "delimited", grids=[Grid(rows)])


def looks_delimited(text: str) -> bool:
    _, score = sniff_delimiter(text[:65536])
    return score >= 0.7


def convert_spreadsheet(path: Path, file_type: str) -> Converted:
    from python_calamine import CalamineWorkbook

    workbook = CalamineWorkbook.from_path(str(path))
    grids, warnings = [], []
    for name in workbook.sheet_names:
        try:
            raw = workbook.get_sheet_by_name(name).to_python(skip_empty_area=False)
        except Exception as exc:
            warnings.append(f"sheet {name!r} unreadable: {exc}")
            continue
        rows = [[cell_to_str(c) for c in row] for row in raw]
        if any(any(r) for r in rows):
            grids.append(Grid(rows, name=name))
    if not grids:
        return Converted(
            "empty", file_type, "calamine", note="no non-empty sheets", warnings=warnings
        )
    return Converted("tables", file_type, "calamine", grids=grids, warnings=warnings)


def convert_parquet(path: Path, file_type: str) -> Converted:
    lf = pl.scan_parquet(path)
    columns = lf.collect_schema().names()
    head = lf.head(50).select(pl.all().cast(pl.String, strict=False)).collect()
    preview = [[cell_to_str(c) for c in row] for row in head.rows()]
    table = LazyTable(path, "parquet", columns, preview)
    return Converted("tables", file_type, "parquet", lazy_tables=[table])


def dedupe_headers(header: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, name in enumerate(header):
        name = (name or "").strip().lstrip("﻿") or f"column_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
    return out


def _flatten(obj: Any, prefix: str = "", out: dict[str, str] | None = None) -> dict[str, str]:
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}{k}.", out)
    else:
        out[prefix[:-1]] = cell_to_str(obj)
    return out


def _record_arrays(obj: Any, path: str = "") -> list[tuple[str, list[dict]]]:
    found = []
    if isinstance(obj, list) and len(obj) >= 1 and all(isinstance(x, dict) for x in obj):
        found.append((path, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += _record_arrays(v, f"{path}.{k}" if path else str(k))
    return found


def records_to_grid(records: list[dict], name: str | None, context: str | None) -> Grid:
    flat = [_flatten(r) for r in records]
    header: list[str] = []
    for r in flat:
        for k in r:
            if k not in header:
                header.append(k)
    rows = [header] + [[r.get(k, "") for k in header] for r in flat]
    return Grid(rows, name=name, context=context)


def json_to_grids(data: Any) -> list[Grid]:
    """Arrays of objects become tables; scalar siblings become context for the mapper."""
    arrays = _record_arrays(data)
    if not arrays:
        return []
    context = None
    if isinstance(data, dict):
        scalars = {
            k: v
            for k, v in _flatten({k: v for k, v in data.items() if not isinstance(v, list)}).items()
        }
        if scalars:
            context = "\n".join(f"{k}: {v}" for k, v in list(scalars.items())[:60])
    return [records_to_grid(recs, path or None, context) for path, recs in arrays]
