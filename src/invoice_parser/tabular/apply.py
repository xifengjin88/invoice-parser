"""Apply a mapping plan to every row of a table, streaming, and write canonical Parquet."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from ..convert.tables import dedupe_headers
from ..enrich import enrich
from ..schema import (
    COST_FIELDS,
    DATETIME,
    DATETIME_FIELDS,
    DECIMAL,
    DECIMAL_FIELDS,
    FIELD_BY_NAME,
    SCHEMA_VERSION,
    StatedTotal,
    conform,
)
from ..values import (
    datetime_expr,
    detect_datetime_format,
    detect_number_style,
    looks_like_dates,
    number_expr,
    parse_datetime,
    parse_number,
)
from .blocks import Block
from .plan import TablePlan

SAMPLE_ROWS = 3000
_INTERVAL = re.compile(r"^\d{4}-\d\d-\d\dT[^/]+/\d{4}-\d\d-\d\dT")


@dataclass
class Provenance:
    document_id: str
    source_file: str
    source_member: str | None
    table_index: int
    extraction_method: str


@dataclass
class ApplyResult:
    rows_in: int = 0
    rows_empty: int = 0
    rows_total: int = 0  # rows dropped because the plan marked them as totals
    rows_out: int = 0
    parse_failures: dict[str, int] = field(default_factory=dict)
    source_sums: dict[str, str] = field(default_factory=dict)
    stated_totals: list[StatedTotal] = field(default_factory=list)
    attribute_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # decisions taken, for the record
    # What the mapping check in the validator needs: evidence about the plan's own claims.
    cost_mapped: bool = False
    number_columns_used: int = 0
    number_columns_unused: list[str] = field(default_factory=list)  # hold amounts, mapped nowhere
    totals_sum: str | None = None  # the rows set aside as totals, added up
    table_total: str | None = None  # what the table itself says its rows add up to
    kept_sum: str | None = None  # the rows kept as line items, added up


def _nz(name: str) -> pl.Expr:
    c = pl.col(name).cast(pl.String).str.strip_chars()
    return pl.when(c == "").then(None).otherwise(c)


def _source_frame(block: Block, header: list[str], header_row: int) -> pl.LazyFrame:
    """All-string frame with the plan's column names and a `__row` original row number."""
    if block.lazy is not None:
        table = block.lazy
        if table.fmt == "parquet":
            lf = pl.scan_parquet(table.path).select(pl.all().cast(pl.String, strict=False))
            return lf.rename(
                dict(zip(lf.collect_schema().names(), header, strict=True))
            ).with_row_index("__row", 1)
        lf = pl.scan_csv(
            table.path,
            separator=table.separator,
            has_header=False,
            skip_rows=1,
            new_columns=header,
            infer_schema=False,
            truncate_ragged_lines=True,
            encoding="utf8-lossy",
        )
        return lf.with_row_index("__row", 2)
    data = block.rows[header_row + 1 :]
    columns = {name: [(r[j] if j < len(r) else "") for r in data] for j, name in enumerate(header)}
    df = pl.DataFrame(columns, schema={name: pl.String for name in header})
    first = block.first_row_number + header_row + 1
    return df.lazy().with_row_index("__row", first)


def json_attributes(columns: list[str]) -> pl.Expr:
    """JSON object of the non-empty values in `columns`, built without leaving Polars."""
    if not columns:
        return pl.lit(None, pl.String)
    pieces = []
    for name in columns:
        v = pl.col(name).cast(pl.String).str.strip_chars()
        escaped = (
            v.str.replace_all("\\", "\\\\", literal=True)
            .str.replace_all('"', '\\"', literal=True)
            .str.replace_all(r"[\x00-\x1f]", " ")
        )
        key = json.dumps(name, ensure_ascii=False) + ':"'
        pieces.append(
            pl.when(v.is_not_null() & (v != "")).then(
                pl.concat_str([pl.lit(key), escaped, pl.lit('"')])
            )
        )
    body = pl.concat_str(pieces, separator=",", ignore_nulls=True)
    return pl.when(body != "").then(pl.concat_str([pl.lit("{"), body, pl.lit("}")]))


def _typed(
    field_name: str, raw: pl.Expr, sample: list[str], plan: TablePlan
) -> tuple[pl.Expr | None, str | None]:
    """Typed expression for a mapped field, or (None, reason) when the column does not fit it."""
    if field_name not in DATETIME_FIELDS and looks_like_dates(sample):
        return None, "values are dates"
    if field_name in DECIMAL_FIELDS:
        if sample and sum(parse_number(s) is not None for s in sample) < 0.5 * len(sample):
            return None, "values are not numeric"
        return number_expr(raw, detect_number_style(sample)), None
    if field_name in DATETIME_FIELDS:
        part = plan.interval_fields.get(field_name)
        if part is None and sample and all(_INTERVAL.match(s) for s in sample[:20]):
            part = "start" if field_name.endswith("start") else "end"
        probe = [s.split("/")[0 if part == "start" else 1] for s in sample] if part else sample
        fmt = detect_datetime_format(probe)
        if fmt is None and probe and parse_datetime(probe[0]) is None:
            return None, "values are not dates"
        return datetime_expr(raw, fmt, part), None
    return raw, None


def _constant(field_name: str, value: str) -> pl.Expr | None:
    if field_name in DECIMAL_FIELDS:
        number = parse_number(value)
        return None if number is None else pl.lit(number, DECIMAL)
    if field_name in DATETIME_FIELDS:
        when = parse_datetime(value)
        return None if when is None else pl.lit(when, DATETIME)
    return pl.lit(value, pl.String)


def apply_plan(block: Block, plan: TablePlan, out_path: Path, prov: Provenance) -> ApplyResult:
    result = ApplyResult()
    header = block.lazy.columns if block.lazy else dedupe_headers(block.rows[plan.header_row])
    source = _source_frame(block, header, plan.header_row)

    def sample_of(col: str) -> list[str]:
        s = sample_df.get_column(col).str.strip_chars()
        return s.filter(s.is_not_null() & (s != "")).head(400).to_list()

    # Rows to drop: fully empty, and totals named by the plan.
    non_empty = pl.any_horizontal([_nz(c).is_not_null() for c in header])
    is_total = pl.lit(False)
    for col, values in plan.total_rows.items():
        if col in header:
            is_total = is_total | _nz(col).str.to_lowercase().is_in(
                [v.strip().lower() for v in values]
            )
    is_total = is_total.fill_null(False)
    sample_df = source.filter(non_empty & ~is_total).head(SAMPLE_ROWS).collect()

    field_map = {f: [c for c in cols if c in header] for f, cols in plan.field_map.items()}
    field_map = {f: cols for f, cols in field_map.items() if cols and f in FIELD_BY_NAME}
    mapped_cols = {c for cols in field_map.values() for c in cols}
    ignored = set(plan.ignore_columns)

    melt = plan.melt
    melt_cols: list[str] = []
    label_expr = value_expr = None
    if melt is not None:
        melt_cols = [c for c in plan.melt_columns if c in header and c not in mapped_cols]
        if not melt_cols:
            result.warnings.append("pivot requested but no pivot columns exist; ignored")
            melt = None
        elif looks_like_dates([v for c in melt_cols[:20] for v in sample_of(c)[:20]]):
            result.warnings.append(
                f"pivot over {', '.join(repr(c) for c in melt_cols[:3])} requested but the "
                "values there are dates, not amounts; ignored"
            )
            melt, melt_cols = None, []

    exprs: dict[str, pl.Expr] = {}
    checks: dict[str, tuple[pl.Expr, pl.Expr]] = {}  # field -> (raw, typed) for failure counts
    for field_name, cols in field_map.items():
        raw = pl.coalesce([_nz(c) for c in cols])
        sample = [v for c in cols for v in sample_of(c)][:400]
        typed, problem = _typed(field_name, raw, sample, plan)
        if typed is None:
            result.warnings.append(
                f"{cols[0]!r} was mapped to {field_name} but {problem}; kept as attribute"
            )
            mapped_cols -= set(cols)
            continue
        exprs[field_name] = typed
        if field_name in DECIMAL_FIELDS or field_name in DATETIME_FIELDS:
            checks[field_name] = (raw, typed)

    derived: list[str] = []
    for field_name, value in plan.constants.items():
        if field_name in exprs or field_name not in FIELD_BY_NAME:
            continue
        const = _constant(field_name, value)
        if const is None:
            result.warnings.append(f"constant {field_name}={value!r} could not be parsed; dropped")
            continue
        exprs[field_name] = const
        if not plan.source.startswith("seed"):
            derived.append(field_name)

    attribute_cols = [
        c for c in header if c not in mapped_cols and c not in ignored and c not in melt_cols
    ]
    result.attribute_columns = attribute_cols
    result.cost_mapped = melt is not None or any(f in exprs for f in COST_FIELDS)
    result.number_columns_used = len(melt_cols) + sum(
        len(field_map[f]) for f in exprs if f in DECIMAL_FIELDS and f in field_map
    )
    for col in [*attribute_cols, *(c for c in header if c in ignored and c not in mapped_cols)]:
        values = sample_of(col)
        numbers = [n for n in (parse_number(v) for v in values) if n is not None]
        # Amounts, as opposed to counts and identifiers, are mostly written with a fraction.
        if (
            values
            and len(numbers) >= 0.8 * len(values)
            and sum(n != n.to_integral_value() for n in numbers) >= 0.5 * len(numbers)
            and not looks_like_dates(values)
        ):
            result.number_columns_unused.append(col)

    # Pass 1 over the source: row accounting and parse failures.
    kept = non_empty & ~is_total
    stats = [
        pl.len().alias("rows_in"),
        (~non_empty).sum().alias("rows_empty"),
        (non_empty & is_total).sum().alias("rows_total"),
    ]
    for field_name, (raw, typed) in checks.items():
        stats.append((kept & raw.is_not_null() & typed.is_null()).sum().alias(f"fail:{field_name}"))
        if field_name in COST_FIELDS:
            stats.append(typed.filter(kept).sum().alias(f"sum:{field_name}"))
    row = source.select(stats).collect().row(0, named=True)
    result.rows_in, result.rows_empty, result.rows_total = (
        row["rows_in"],
        row["rows_empty"],
        row["rows_total"],
    )
    result.parse_failures = {k[5:]: v for k, v in row.items() if k.startswith("fail:") and v}
    result.source_sums = {
        k[4:]: str(v) for k, v in row.items() if k.startswith("sum:") and v is not None
    }

    grand = stated_sum = None
    if plan.total_rows and not melt:
        cost_field = next((f for f in COST_FIELDS if f in exprs), None)
        if cost_field:
            label_cols = [c for c in plan.total_rows if c in header]
            totals = (
                source.filter(non_empty & is_total)
                .select(
                    pl.col("__row"),
                    exprs[cost_field].alias("amount"),
                    (
                        pl.coalesce([_nz(c) for c in field_map[cost_field]])
                        if cost_field in field_map
                        else pl.lit(None, pl.String)
                    ).alias("raw"),
                    *[_nz(c).alias(f"l:{c}") for c in label_cols],
                )
                .head(2000)
                .collect()
            )
            # A repeated heading row can sit among the totals; a month or a year printed
            # there is not an amount.
            found = [
                t
                for t in totals.iter_rows(named=True)
                if t["amount"] is not None and not (t["raw"] and looks_like_dates([t["raw"]]))
            ]
            # A grand total is at least as large as any subtotal, so the largest of the total
            # rows stands for the table; the others are subtotals of parts of it.
            grand = max(found, key=lambda t: abs(t["amount"]), default=None)
            stated_sum = sum(t["amount"] for t in found) if found else None
            for t in found:
                label = next((t[f"l:{c}"] for c in label_cols if t[f"l:{c}"]), "total")
                result.stated_totals.append(
                    StatedTotal(
                        label=label,
                        amount=str(t["amount"]),
                        scope="table" if t is grand else "section",
                        source_row=t["__row"],
                        origin="table",
                    )
                )

    frame = source.filter(kept)
    is_rollup = pl.lit(False)
    if grand is not None:
        # The column that marks total rows also says what kind every other row is. When two or
        # more of those kinds each add up to the table's own total, they are the same charges
        # listed more than once (per account and again per payer, say). The most detailed
        # listing is kept and the others flagged, so nothing is counted twice.
        target = abs(grand["amount"])
        slack = max(0.05, float(target) * 0.0005)
        for col in label_cols:
            groups = (
                frame.group_by(_nz(col).alias("kind"))
                .agg(pl.len().alias("n"), exprs[cost_field].sum().alias("sum"))
                .collect()
            )
            whole = [
                g
                for g in groups.iter_rows(named=True)
                if g["kind"] is not None
                and g["sum"] is not None
                and abs(abs(float(g["sum"])) - float(target)) <= slack
            ]
            if len(whole) > 1:
                whole.sort(key=lambda g: g["n"], reverse=True)
                repeats = [g["kind"] for g in whole[1:]]
                is_rollup = _nz(col).is_in(repeats).fill_null(False)
                result.notes.append(
                    f"{col}: {', '.join(repr(g['kind']) for g in whole)} each add up to the "
                    f"table total; kept {whole[0]['kind']!r} ({whole[0]['n']:,} rows) and "
                    f"flagged the rest as rollups"
                )
                break
    known_format = plan.source.startswith("seed")
    cost_name = None if melt is not None else next((f for f in COST_FIELDS if f in exprs), None)
    if grand is not None:
        result.table_total = str(grand["amount"])
    elif cost_name and not known_format:
        # No total row was named. A row that equals all the others added together is one
        # all the same, whatever it is labelled: arithmetic, not wording, decides.
        amounts = (
            source.filter(kept)
            .select(pl.col("__row"), exprs[cost_name].cast(pl.Float64).alias("amount"))
            .filter(pl.col("amount").is_not_null())
        )
        count, whole = amounts.select(pl.len(), pl.col("amount").sum()).collect().row(0)
        if count >= 3 and whole:
            half = whole / 2
            hits = (
                amounts.filter((pl.col("amount") - half).abs() <= max(0.005, abs(half) * 1e-6))
                .head(2)
                .collect()
            )
            if hits.height == 1:
                row_no = hits["__row"][0]
                is_rollup = pl.col("__row") == row_no
                result.table_total = str(parse_number(f"{hits['amount'][0]:.6f}"))
                result.stated_totals.append(
                    StatedTotal(
                        label="row that adds up all the others",
                        amount=result.table_total,
                        scope="table",
                        source_row=row_no,
                        origin="table",
                    )
                )
                result.notes.append(
                    f"row {row_no} equals the sum of every other row, so it was taken as the "
                    "table's total and flagged as a rollup"
                )
    if cost_name and not known_format and result.kept_sum is None:
        live = source.select(exprs[cost_name].filter(kept & ~is_rollup).sum()).collect().item()
        result.kept_sum = None if live is None else str(live)
    if plan.total_rows and result.rows_total:
        # A row is a total only if it adds up what the other rows say. Both sides are summed
        # here so the validator can test that claim.
        if melt is not None:
            style = detect_number_style([v for c in melt_cols[:20] for v in sample_of(c)])
            amount = pl.sum_horizontal([number_expr(_nz(c), style).fill_null(0) for c in melt_cols])
        else:
            cost = next((f for f in COST_FIELDS if f in exprs), None)
            amount = exprs[cost] if cost else None
        if amount is not None:
            sums = (
                source.select(
                    amount.filter(non_empty & is_total).sum().alias("totals"),
                    amount.filter(kept & ~is_rollup).sum().alias("kept"),
                )
                .collect()
                .row(0, named=True)
            )
            if melt is None:
                # Only total rows that print an amount make a claim that can be tested.
                sums["totals"] = stated_sum
            result.totals_sum = None if sums["totals"] is None else str(sums["totals"])
            result.kept_sum = None if sums["kept"] is None else str(sums["kept"])
    item_suffix = pl.lit("")
    if melt is not None:
        frame = frame.unpivot(
            on=melt_cols,
            index=[c for c in header if c not in melt_cols] + ["__row"],
            variable_name="__label",
            value_name="__value",
        ).filter(pl.col("__value").is_not_null() & (pl.col("__value").str.strip_chars() != ""))
        labels = pl.col("__label")
        if melt.label_strip:
            labels = labels.str.replace(re.escape(melt.label_strip) + r"\s*$", "").str.strip_chars()
        cleaned = [
            c.replace(melt.label_strip, "").strip() if melt.label_strip else c for c in melt_cols
        ]
        label_expr, problem = _typed(melt.label_field.value, labels, cleaned, plan)
        if label_expr is None:
            result.warnings.append(
                f"pivot headers do not fit {melt.label_field.value} ({problem}); kept as attribute"
            )
            label_expr = None
        value_expr = number_expr(
            pl.col("__value"),
            detect_number_style([v for c in melt_cols[:20] for v in sample_of(c)]),
        )
        if label_expr is not None:
            exprs[melt.label_field.value] = label_expr
        exprs[melt.value_field.value] = value_expr
        item_suffix = pl.concat_str([pl.lit(":"), pl.col("__label")])

    attributes = json_attributes(attribute_cols)
    if melt is not None and label_expr is None:
        attributes = pl.concat_str(
            [pl.lit('{"pivot_label":"'), pl.col("__label").str.replace_all('"', "'"), pl.lit('"}')]
        )

    out = frame.select(
        *[e.alias(name) for name, e in exprs.items()],
        attributes.alias("attributes"),
        pl.concat_str(
            [
                pl.lit(f"{prov.document_id}:{prov.table_index}:"),
                pl.col("__row").cast(pl.String),
                item_suffix,
            ]
        ).alias("line_item_id"),
        pl.lit(prov.document_id).alias("document_id"),
        pl.lit(prov.source_file).alias("source_file"),
        pl.lit(prov.source_member if prov.source_member else block.name, pl.String).alias(
            "source_member"
        ),
        pl.col("__row").cast(pl.Int64).alias("source_row"),
        pl.lit(prov.extraction_method).alias("extraction_method"),
        pl.lit(SCHEMA_VERSION).alias("schema_version"),
        pl.lit(plan.record_type).alias("record_type"),
        pl.lit(plan.granularity).alias("granularity"),
        pl.lit(block.name, pl.String).alias("section"),
        is_rollup.alias("is_rollup"),
        pl.lit(derived, pl.List(pl.String)).alias("derived_fields"),
    )
    enrich(conform(out)).sink_parquet(out_path, compression="zstd")
    result.rows_out = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    return result
