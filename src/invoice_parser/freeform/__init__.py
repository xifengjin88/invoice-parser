"""Free-form path: describe the document, tag its lines, rebuild the hierarchy, write rows."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from ..convert.types import Converted, Source, TextDoc
from ..enrich import enrich
from ..llm import LLM
from ..schema import (
    POLARS_SCHEMA,
    SCHEMA_VERSION,
    DocumentRecord,
    ExtractionMethod,
    RecordType,
    conform,
)
from ..values import parse_datetime
from .describe import describe_rows
from .resources import list_resources, resource_rows
from .tagging import DocMeta, FreeDocType, read_meta, tag_document, transcribe
from .walk import (
    Walker,
    combine_restated,
    currency_twins,
    dedupe_totals,
    ground_meta_totals,
    line_currency,
    mark_rollups,
    missed_lines,
)

__all__ = ["parse_text", "rows_to_frame"]

log = logging.getLogger("invoice_parser")


def rows_to_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    clean = [
        {
            k: (v.value if isinstance(v, StrEnum) else v)
            for k, v in row.items()
            if k in POLARS_SCHEMA
        }
        for row in rows
    ]
    for row in clean:
        row.setdefault("derived_fields", [])
    return (
        pl.from_dicts(clean, schema=POLARS_SCHEMA) if clean else pl.DataFrame(schema=POLARS_SCHEMA)
    )


def _base(record: DocumentRecord, method: str, meta: DocMeta) -> dict[str, Any]:
    """Fields every row of a document shares."""
    return {
        "document_id": record.document_id,
        "source_file": record.source_file,
        "source_member": record.source_member,
        "extraction_method": method,
        "schema_version": SCHEMA_VERSION,
        "record_type": meta.record_type.value,
        "provider": meta.provider,
        "invoice_id": meta.invoice_id,
        "billing_account_id": meta.billing_account_id,
        "billing_account_name": meta.billing_account_name,
        "billing_period_start": parse_datetime(meta.billing_period_start),
        "billing_period_end": parse_datetime(meta.billing_period_end),
    }


def parse_text(
    llm: LLM,
    converted: Converted,
    source: Source,
    record: DocumentRecord,
    out_path: Path,
    workers: int,
) -> None:
    """Fill `record` and write line items for a text document. Raises on model failure."""
    doc: TextDoc = converted.text  # type: ignore[assignment]
    method = ExtractionMethod.LLM_LINE_TAGS
    if converted.images:
        log.info("%s: transcribing %d page image(s)", source.name, len(converted.images))
        pages = transcribe(llm, converted.images, workers)
        for page_no, lines in pages.items():
            doc.pages[page_no - 1] = lines
        record.freeform["pages_transcribed"] = len(pages)
        if len(pages) >= len(doc.pages):
            doc.origin = "ocr"
        method = ExtractionMethod.OCR_LLM_LINE_TAGS
    if doc.n_lines == 0:
        record.doc_type = "unreadable"
        record.warnings.append("no text could be read from the document")
        rows_to_frame([]).write_parquet(out_path)
        return

    log.info("%s: asking the model what the document is", source.name)
    meta: DocMeta = read_meta(llm, doc, source.name)
    log.info(
        "%s: %s from %s, currency %s, %d stated total(s): %s",
        source.name, meta.doc_type.value, meta.provider or "unknown provider",
        meta.currency or "?", len(meta.totals), meta.summary[:100],
    )  # fmt: skip
    record.doc_type = meta.doc_type.value
    record.description = meta.summary
    record.language = meta.language
    record.providers = [meta.provider.lower()] if meta.provider else []
    record.extraction_method = method.value
    record.metadata = meta.model_dump(mode="json", exclude={"totals", "summary", "doc_type"})

    totals, dropped = ground_meta_totals(meta, doc)
    if dropped:
        record.warnings.append(
            f"{dropped} document total(s) named by the model were not found in the text"
        )
    # Only a document the model calls unrelated to cost is skipped. Its yes/no on whether there
    # are lines to extract is not relied on: tagging finds out, and it is wrong too often.
    if meta.doc_type == FreeDocType.NOT_COST_RELATED:
        record.stated_totals = dedupe_totals(totals)
        record.warnings.append("document has no charge, price or resource lines to extract")
        rows_to_frame([]).write_parquet(out_path)
        return

    if meta.doc_type == FreeDocType.INFRA_DEFINITION or meta.record_type == RecordType.RESOURCE:
        # No charges to tag line by line: the unit is a resource, which spans a block.
        found, failures = list_resources(llm, doc, meta, source.name, workers)
        for failure in failures:
            record.warnings.append(f"listing resources failed for a part of the file ({failure})")
        with out_path.with_name("tags.jsonl").open("w") as fh:
            for line, res in found:
                entry = {"page": line.page, "line": line.number, "text": line.text}
                entry["tag"] = {
                    "kind": "resource",
                    **res.model_dump(mode="json", exclude_none=True),
                }
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        rows, stats = resource_rows(doc, _base(record, method.value, meta), found)
        for i, row in enumerate(rows):
            row["line_item_id"] = (
                f"{record.document_id}:p{row['source_page']}:l{row['source_row']}:{i}"
            )
        log.info("%s: %d resource(s)", source.name, len(rows))
        record.stated_totals = dedupe_totals(totals)
        record.freeform.update(stats)
        record.freeform["chunks_failed"] = len(failures)
        record.freeform["lines"] = doc.n_lines
        record.freeform["pages"] = len(doc.pages)
        enrich(conform(rows_to_frame(rows).lazy())).collect().write_parquet(
            out_path, compression="zstd"
        )
        return

    pairs, failures = tag_document(llm, doc, meta, source.name, workers)
    for failure in failures:
        record.warnings.append(f"tagging failed for a part of the document ({failure})")
    # Kept beside the output so a wrong row can be traced to what the model said about its line.
    with out_path.with_name("tags.jsonl").open("w") as fh:
        for line, tag in pairs:
            entry = {"page": line.page, "line": line.number, "text": line.text}
            entry["tag"] = tag.model_dump(mode="json", exclude_none=True)
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    base = _base(record, method.value, meta)
    result = Walker(doc, meta, base, totals).run(pairs)
    log.info(
        "%s: %d tagged line(s) -> %d line item(s); %d/%d cited values found in source",
        source.name, len(pairs), len(result.rows),
        result.stats["values_grounded"], result.stats["values_claimed"],
    )  # fmt: skip
    rows = result.rows
    for i, row in enumerate(rows):
        row["line_item_id"] = f"{record.document_id}:p{row['source_page']}:l{row['source_row']}:{i}"
    totals = dedupe_totals(totals + result.stated_totals)
    totals = dedupe_totals(totals + currency_twins(totals, doc))
    rollups = mark_rollups(rows, totals, line_currency(rows, meta.currency))
    # Wording is read before repeats are folded, so the model is asked about each charge's own
    # text and a change in how rows are combined never changes what the model is asked.
    record.freeform["described"] = describe_rows(llm, rows, meta.record_type.value, workers)
    rows, rollups["restated"] = combine_restated(rows)

    record.stated_totals = totals
    record.freeform.update(result.stats)
    record.freeform["chunks_failed"] = len(failures)
    record.freeform["rollups"] = rollups
    record.freeform["coverage"] = missed_lines(doc, pairs, rows, totals)
    record.freeform["lines"] = doc.n_lines
    record.freeform["pages"] = len(doc.pages)

    frame = enrich(conform(rows_to_frame(rows).lazy())).collect()
    frame.write_parquet(out_path, compression="zstd")
