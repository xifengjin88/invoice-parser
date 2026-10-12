"""Tabular path: find tables, get a mapping plan (seed or model), apply it to every row."""

from __future__ import annotations

import logging
from decimal import Decimal
from itertools import combinations
from pathlib import Path

import polars as pl

from ..convert.tables import dedupe_headers
from ..convert.types import Converted, Source
from ..llm import LLM
from ..schema import (
    DocType,
    DocumentRecord,
    ExtractionMethod,
    Granularity,
    RecordType,
    empty_frame,
)
from .apply import ApplyResult, Provenance, apply_plan
from .blocks import Block, lazy_block, split_blocks
from .plan import TablePlan, plan_block
from .seeds import seed_plan

__all__ = ["parse_tables"]

log = logging.getLogger("invoice_parser")

MAX_TABLES_COMPARED = 12


def _flag_restating_tables(record: DocumentRecord, parts: list[Path]) -> None:
    """A workbook often gives the same charges twice: a summary sheet and the sheets it sums.

    A table whose rows add up to the same amount as a group of other, more detailed tables is
    their summary. Its rows are flagged as rollups so the charges are counted once.
    """
    by_part = {
        part: t
        for part, t in zip(parts, [t for t in record.tables if "rows_out" in t], strict=True)
    }
    tables = [
        (part, t, Decimal(t["kept_sum"]))
        for part, t in by_part.items()
        if t.get("kept_sum") and Decimal(t["kept_sum"]) != 0 and t.get("rows_out")
    ]
    if not 2 <= len(tables) <= MAX_TABLES_COMPARED:
        return
    flagged: set[Path] = set()
    for part, table, amount in sorted(tables, key=lambda x: x[1]["rows_out"]):
        others = [x for x in tables if x[0] != part and x[0] not in flagged]
        wanted = {amount}
        if table.get("table_total"):
            wanted.add(Decimal(table["table_total"]))
        match = None
        for size in range(1, len(others) + 1):
            for group in combinations(others, size):
                together = sum((x[2] for x in group), Decimal(0))
                detail = sum(x[1]["rows_out"] for x in group)
                close = any(
                    abs(together - w) <= max(Decimal("0.05"), abs(w) * Decimal("0.0005"))
                    for w in wanted
                )
                if close and detail > table["rows_out"]:
                    match = group
                    break
            if match:
                break
        if not match:
            # Close but not equal: nothing proves these are the same charges, and nothing
            # rules it out. Left for the validator to raise as a doubt.
            near = [
                x[1]["index"]
                for x in others
                if abs(x[2] - amount) <= abs(amount) * Decimal("0.05")
                and x[1]["rows_out"] > table["rows_out"]
            ]
            if near:
                table["nearly_equals"] = near
            continue
        flagged.add(part)
        tmp = part.with_suffix(".rollup.parquet")
        pl.scan_parquet(part).with_columns(pl.lit(True).alias("is_rollup")).sink_parquet(
            tmp, compression="zstd"
        )
        tmp.replace(part)
        table["rollup_of"] = [x[1]["index"] for x in match]
        table.setdefault("notes", []).append(
            f"adds up to the same amount as table(s) {', '.join(str(i) for i in table['rollup_of'])}, "
            "which hold more detail; flagged as a rollup of them"
        )


MAX_BLOCKS = 40
_GRANULARITY_ORDER = [g.value for g in Granularity]


def _doc_type(plans: list[TablePlan]) -> str:
    types = {p.record_type for p in plans}
    if not plans:
        return DocType.NOT_COST_RELATED
    if types == {RecordType.RESOURCE}:
        return DocType.INFRA_DEFINITION
    if RecordType.CHARGE not in types:
        return DocType.ESTIMATE
    finest = min(
        _GRANULARITY_ORDER.index(p.granularity) for p in plans if p.record_type == RecordType.CHARGE
    )
    return (
        DocType.USAGE_EXPORT
        if finest <= _GRANULARITY_ORDER.index(Granularity.USAGE_TYPE)
        else DocType.COST_SUMMARY
    )


def parse_tables(
    llm: LLM | None, converted: Converted, source: Source, record: DocumentRecord, out_path: Path
) -> None:
    """Fill `record` and write line items for a tabular document."""
    blocks: list[Block] = [lazy_block(t) for t in converted.lazy_tables]
    for grid in converted.grids:
        blocks += split_blocks(grid)
    if len(blocks) > MAX_BLOCKS:
        record.warnings.append(f"{len(blocks)} tables found; only the first {MAX_BLOCKS} were read")
        blocks = blocks[:MAX_BLOCKS]

    parts: list[Path] = []
    used: list[TablePlan] = []
    methods: set[str] = set()
    for index, block in enumerate(blocks):
        entry: dict = {"index": index, "name": block.name, "first_row": block.first_row_number}
        record.tables.append(entry)
        header = block.lazy.columns if block.lazy else dedupe_headers(block.rows[0])
        plan = seed_plan(header)
        warnings: list[str] = []
        if plan is None:
            if llm is None:
                entry["skipped"] = "no known format matched and no model endpoint is configured"
                continue
            log.info(
                "%s: table %d/%d (%s): asking the model for a column mapping",
                source.name, index + 1, len(blocks), block.name or "unnamed",
            )  # fmt: skip
            try:
                plan, warnings = plan_block(llm, block, source.name)
            except Exception as exc:
                entry["skipped"] = f"mapping failed: {type(exc).__name__}: {exc}"[:400]
                record.warnings.append(f"table {index} ({block.name or 'unnamed'}): mapping failed")
                continue
        method = (
            ExtractionMethod.KNOWN_FORMAT
            if plan.source.startswith("seed")
            else ExtractionMethod.LLM_COLUMN_MAP
        )
        entry["plan"] = plan.model_dump(mode="json")
        # Columns that map onto the schema are data, whatever the model's overall verdict.
        if not (plan.field_map or plan.melt):
            entry["skipped"] = "not cost data: " + plan.summary
            continue
        part = out_path.with_name(f"part-{index}.parquet")
        prov = Provenance(
            record.document_id, record.source_file, record.source_member, index, method.value
        )
        log.info(
            "%s: table %d/%d: applying %s (%d field(s) mapped%s)",
            source.name, index + 1, len(blocks), plan.source, len(plan.field_map),
            ", pivoted" if plan.melt else "",
        )  # fmt: skip
        try:
            result: ApplyResult = apply_plan(block, plan, part, prov)
            log.info(
                "%s: table %d/%d: %d rows in, %d line item(s) out",
                source.name, index + 1, len(blocks), result.rows_in, result.rows_out,
            )  # fmt: skip
        except Exception as exc:
            entry["skipped"] = f"apply failed: {type(exc).__name__}: {exc}"[:400]
            record.warnings.append(
                f"table {index} ({block.name or 'unnamed'}): could not apply mapping"
            )
            part.unlink(missing_ok=True)
            continue
        entry.update(
            rows_in=result.rows_in, rows_empty=result.rows_empty, rows_total=result.rows_total,
            rows_out=result.rows_out, parse_failures=result.parse_failures,
            source_sums=result.source_sums, attribute_columns=result.attribute_columns,
            melted=plan.melt is not None, warnings=warnings + result.warnings,
            cost_mapped=result.cost_mapped, number_columns_used=result.number_columns_used,
            number_columns_unused=result.number_columns_unused,
            totals_sum=result.totals_sum, kept_sum=result.kept_sum,
            table_total=result.table_total, notes=result.notes,
            record_type=str(plan.record_type),
        )  # fmt: skip
        record.stated_totals += result.stated_totals
        parts.append(part)
        used.append(plan)
        methods.add(method.value)

    _flag_restating_tables(record, parts)
    if parts:
        if len(parts) == 1:
            parts[0].replace(out_path)
        else:
            pl.concat([pl.scan_parquet(p) for p in parts]).sink_parquet(
                out_path, compression="zstd"
            )
            for p in parts:
                p.unlink(missing_ok=True)
    else:
        empty_frame().write_parquet(out_path)

    record.doc_type = _doc_type(used)
    record.extraction_method = ",".join(sorted(methods)) or None
    record.description = (
        "; ".join(dict.fromkeys(p.summary for p in used if p.summary))[:500] or None
    )
    if used:
        finest = min(_GRANULARITY_ORDER.index(p.granularity) for p in used)
        record.granularity = _GRANULARITY_ORDER[finest]
        providers = (
            pl.scan_parquet(out_path).select(pl.col("provider").drop_nulls().unique()).collect()
        )
        record.providers = sorted(providers.get_column("provider").to_list())
    elif (
        all("no model endpoint" in (t.get("skipped") or "") for t in record.tables)
        and record.tables
    ):
        record.doc_type = None
        record.warnings.append("tables were not mapped because no model endpoint is configured")
