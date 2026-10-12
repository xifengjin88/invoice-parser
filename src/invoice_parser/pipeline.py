"""Orchestration: file -> convert -> tabular or free-form path -> validate -> write."""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import polars as pl

from .convert import Source, convert, iter_sources
from .freeform import parse_text
from .llm import LLM
from .schema import DocumentRecord, empty_frame
from .tabular import parse_tables
from .validate import DOCUMENT, LINE_ITEMS, load_document, validate_document

log = logging.getLogger("invoice_parser")

RUN_REPORT = "run_report.json"
WORK_DIR = ".work"
CACHE_DIR = ".cache"


def process_source(
    source: Source, out_dir: Path, llm: LLM | None, workers: int = 8
) -> DocumentRecord:
    """Parse one document. Never raises: failures are recorded on the document record."""
    doc_dir = out_dir / source.document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    parquet = doc_dir / LINE_ITEMS
    parquet.unlink(missing_ok=True)
    record = DocumentRecord(
        document_id=source.document_id,
        source_file=source.display_path,
        source_member=source.member,
        size_bytes=source.size,
        warnings=list(source.warnings),
    )
    started = time.time()
    if llm is not None and hasattr(llm, "reset_usage"):
        llm.reset_usage()
    log.info("%s: reading", source.name)
    try:
        converted = convert(source)
        if converted.text is not None:
            shape = f"{len(converted.text.pages)} page(s), {converted.text.n_lines} lines"
            if converted.images:
                shape += f", {len(converted.images)} page image(s) to transcribe"
        else:
            shape = f"{len(converted.grids) + len(converted.lazy_tables)} table source(s)"
        log.info("%s: %s %s, %s", source.name, converted.file_type, converted.kind, shape)
        record.file_type = converted.file_type
        record.converter = converted.converter
        record.warnings += converted.warnings
        if converted.kind in ("unreadable", "empty"):
            record.doc_type = "unreadable"
            record.warnings.append(converted.note or "file could not be read")
        elif converted.kind == "tables":
            parse_tables(llm, converted, source, record, parquet)
        elif llm is None:
            record.error = "free-form document needs a model endpoint, and none is configured"
        else:
            parse_text(llm, converted, source, record, parquet, workers)
    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"[:600]
        record.warnings.append(traceback.format_exc(limit=4)[-800:])
    if not parquet.is_file():
        empty_frame().write_parquet(parquet)
    record.line_item_count = pl.scan_parquet(parquet).select(pl.len()).collect().item()
    if llm is not None:
        record.llm_usage = llm.usage()
    log.info("%s: validating %d line item(s)", source.name, record.line_item_count)
    record.validation = validate_document(record, parquet)
    record.elapsed_seconds = round(time.time() - started, 2)
    (doc_dir / DOCUMENT).write_text(record.model_dump_json(indent=1))
    return record


def run(
    path: Path,
    out_dir: Path,
    llm: LLM | None,
    workers: int = 8,
    resume: bool = False,
    on_document: Callable[[DocumentRecord], None] | None = None,
    only: str | None = None,
) -> list[DocumentRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / WORK_DIR
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    records: list[DocumentRecord] = []
    try:
        for source in iter_sources(path, work):
            label = f"{source.display_path}!{source.member or ''}"
            if only and only.lower() not in label.lower():
                continue
            done = out_dir / source.document_id / DOCUMENT
            if resume and done.is_file() and not load_document(done.parent).error:
                record = load_document(done.parent)
            else:
                record = process_source(source, out_dir, llm, workers)
            if source.member and work in source.local_path.parents:
                source.local_path.unlink(missing_ok=True)
            records.append(record)
            if on_document:
                on_document(record)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    write_report(out_dir, records)
    return records


def summarize(record: DocumentRecord) -> dict[str, Any]:
    v = record.validation
    checks = {c.name: c.status for c in v.checks}
    recon = next((c for c in v.checks if c.name == "reconciliation"), None)
    return {
        "document_id": record.document_id,
        "source": record.source_file + (f"!{record.source_member}" if record.source_member else ""),
        "status": v.status,
        "doc_type": record.doc_type,
        "providers": record.providers,
        "extraction_method": record.extraction_method,
        "line_items": record.line_item_count,
        "match_levels": v.match_levels,
        "checks": checks,
        "reconciliation": recon.summary if recon and recon.status != "skip" else None,
        "total_match": recon.metrics.get("match") if recon else None,
        "problems": [f"{c.name}: {c.summary}" for c in v.checks if c.status in ("warn", "fail")],
        "error": record.error,
        "seconds": record.elapsed_seconds,
        "model_calls": (record.llm_usage.get("total") or {}).get("calls", 0),
    }


def write_report(out_dir: Path, records: list[DocumentRecord]) -> dict[str, Any]:
    docs = [summarize(r) for r in records]
    by_status: dict[str, int] = {}
    levels: dict[str, int] = {}
    for d in docs:
        by_status[d["status"]] = by_status.get(d["status"], 0) + 1
        for level, count in d["match_levels"].items():
            levels[level] = levels.get(level, 0) + count
    report = {
        "documents": len(docs),
        "by_status": by_status,
        "line_items": sum(d["line_items"] for d in docs),
        "match_levels": levels,
        "docs": docs,
    }
    (out_dir / RUN_REPORT).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def iter_documents(out_dir: Path) -> Iterator[DocumentRecord]:
    for doc_json in sorted(out_dir.glob(f"*/{DOCUMENT}")):
        yield load_document(doc_json.parent)
