"""The mapping plan: what the model decides about a table. Code applies it to every row."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, Field

from ..convert.tables import dedupe_headers
from ..llm import LLM
from ..schema import DECIMAL_FIELDS, MAPPABLE_FIELDS, Granularity, RecordType, field_guide
from .blocks import Block, describe, guess_header

ATTRIBUTE, IGNORE, PIVOT = "attribute", "ignore", "pivot_value"

MappableField = StrEnum("MappableField", {f: f for f in MAPPABLE_FIELDS})
# What one source column is used for: a schema field, or one of three other roles.
ColumnUse = StrEnum("ColumnUse", {u: u for u in [*MAPPABLE_FIELDS, ATTRIBUTE, IGNORE, PIVOT]})


class ColumnRole(BaseModel):
    column: str
    use: ColumnUse


class Constant(BaseModel):
    field: MappableField
    value: str


class Melt(BaseModel):
    """For pivoted tables: what the pivot_value columns' headers and cells are."""

    label_field: MappableField = Field(description="what the pivoted column headers are")
    value_field: MappableField = Field(description="what the cell values are")
    label_strip: str | None = Field(
        None, description="literal suffix to remove from headers, e.g. ($)"
    )


class TotalRows(BaseModel):
    column: str
    values: list[str]


class TablePlanOut(BaseModel):
    """Model-facing shape. The model walks the table's real columns one by one, so it cannot
    invent a mapping for a field the table has no column for."""

    is_cost_data: bool
    summary: str
    provider: str | None = None
    header_row: int
    record_type: RecordType
    granularity: Granularity
    columns: list[ColumnRole]
    pivot: Melt | None
    constants: list[Constant]
    total_rows: list[TotalRows]


class TablePlan(BaseModel):
    """Resolved plan. `field_map` values are candidate columns; the first present one wins."""

    is_cost_data: bool = True
    summary: str = ""
    header_row: int = 0
    record_type: str = RecordType.CHARGE
    granularity: str = Granularity.SERVICE
    field_map: dict[str, list[str]] = Field(default_factory=dict)
    constants: dict[str, str] = Field(default_factory=dict)
    melt: Melt | None = None
    melt_columns: list[str] = Field(default_factory=list)
    ignore_columns: list[str] = Field(default_factory=list)
    total_rows: dict[str, list[str]] = Field(default_factory=dict)
    interval_fields: dict[str, str] = Field(default_factory=dict)  # field -> start|end of "a/b"
    source: str = "llm"  # llm | seed:<name>


SYSTEM = f"""You map tables from cloud cost documents onto a fixed schema. You never copy data
values; you only say what each source column is. Code applies your answer to every row.

Go through the table's columns in the order shown and give one entry per column, saying what
it is used for. A column's use is exactly one of:
- one of the schema fields below, when the column clearly holds that thing;
- attribute, when it holds something useful that no schema field fits (it is kept as is);
- ignore, when it is a row total, a percentage of total, or merely repeats another column;
- pivot_value, when the column's header is itself data, such as a service name, a month or a
  date, and its cells are amounts. See pivot below.

Schema fields:
{field_guide()}

Rules:
- Decide from the column's name and its sample values together. A column of dates is a date
  field, never a provider or a currency. If nothing fits, the answer is attribute.
- Use each schema field for at most one column. When several columns could be the same field,
  give the field to the best one and mark the rest attribute.
- Costs: billed_cost is what was charged for the row before tax. If the table has only one
  cost column, it is billed_cost. For estimates and quotes use billed_cost for the monthly
  amount.
- header_row is the index (as shown) of the row holding the column names.
- pivot: required when any column is pivot_value, otherwise null. label_field says what the
  pivoted headers are (service_name, charge_period_start, ...); value_field says what the
  cells are (usually billed_cost). The column holding the row labels is mapped normally.
- constants are values true for every row that the table does not carry as a column but that
  the file name or nearby text states (provider, billing_currency, billing_account_id,
  billing_period_start as YYYY-MM-DD, and so on). Do not guess.
- provider is aws, azure, gcp, oci or another short name, or null if unknown.
- total_rows: rows that are totals or subtotals of other rows, given as the column and the
  exact cell values that mark them (for example 'Total', 'Grand Total', 'InvoiceTotal'). Look
  at each column's list of distinct values to find them.
- record_type: charge for billed or metered cost, estimate for quotes, forecasts and pricing
  calculator output, resource for inventories of machines or resources to be priced.
- granularity: the finest level a row describes (resource > sku > usage_type > service > account).
- is_cost_data is false only when the table holds no costs, usage, prices or resources at all,
  for example a legend, KPI tiles, or a schedule of cumulative totals. A list of machines or
  resources with no prices is still data: its record_type is resource."""


def fingerprint(header: list[str]) -> str:
    return hashlib.sha1(json.dumps(header, ensure_ascii=False).encode()).hexdigest()[:16]


def resolve(out: TablePlanOut, header: list[str]) -> tuple[TablePlan, list[str]]:
    """Check the model's plan against the real header; drop what does not exist."""
    warnings: list[str] = []
    by_lower = {h.strip().lower(): h for h in header}

    def real(col: str) -> str | None:
        return col if col in header else by_lower.get(col.strip().lower())

    plan = TablePlan(
        is_cost_data=out.is_cost_data,
        summary=out.summary,
        header_row=out.header_row,
        record_type=out.record_type.value,
        granularity=out.granularity.value,
    )
    seen: set[str] = set()
    for role in out.columns:
        col = real(role.column)
        if col is None:
            warnings.append(f"plan names a column that does not exist: {role.column!r}")
            continue
        if col in seen:
            continue  # the first answer for a column stands
        seen.add(col)
        use = role.use.value
        if use == ATTRIBUTE:
            continue
        if use == IGNORE:
            plan.ignore_columns.append(col)
        elif use == PIVOT:
            plan.melt_columns.append(col)
        elif use in plan.field_map:
            warnings.append(
                f"{col!r} and {plan.field_map[use][0]!r} were both called {use}; "
                f"{col!r} kept as an attribute"
            )
        else:
            plan.field_map[use] = [col]
    for c in out.constants:
        if c.value.strip() and c.field.value not in plan.field_map:
            plan.constants[c.field.value] = c.value.strip()
    if out.provider and "provider" not in plan.field_map:
        plan.constants.setdefault("provider", out.provider)
    for t in out.total_rows:
        col = real(t.column)
        if col and t.values:
            plan.total_rows.setdefault(col, []).extend(t.values)
    if plan.melt_columns:
        usable = (
            out.pivot is not None
            and out.pivot.value_field.value in DECIMAL_FIELDS
            and out.pivot.label_field != out.pivot.value_field
        )
        if usable:
            plan.melt = out.pivot
            # The pivot supplies these two fields; a column mapped to either would clash.
            for name in (out.pivot.label_field.value, out.pivot.value_field.value):
                plan.field_map.pop(name, None)
        else:
            warnings.append(
                "columns were marked pivot_value without a usable pivot; kept as attributes"
            )
            plan.melt_columns = []
    return plan, warnings


def plan_block(llm: LLM, block: Block, file_name: str) -> tuple[TablePlan, list[str]]:
    """Ask the model for a plan; re-ask once if it picks a different header row."""
    header_row = 0 if block.lazy else guess_header(block.rows)
    for _ in range(2):
        user = (
            f"File: {file_name}\n"
            + (f"Sheet / table name: {block.name}\n" if block.name else "")
            + "\n"
            + describe(block, header_row)
        )
        out = llm.json(
            system=SYSTEM, user=user, schema=TablePlanOut, tag="table_plan", max_tokens=8000
        )
        chosen = out.header_row if 0 <= out.header_row < len(block.rows) - 1 else header_row
        if chosen == header_row or block.lazy:
            break
        header_row = chosen
    plan, warnings = resolve(out, dedupe_headers(block.rows[header_row]))
    plan.header_row = header_row
    return plan, warnings
