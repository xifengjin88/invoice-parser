"""Validator: how far a parsed document can be trusted, from the output files alone.

Reads `document.json` and `line_items.parquet`. Each check returns pass, warn, fail or skip
with counts and examples; a document's status is its worst check.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from ..freeform.walk import whole_document_totals
from ..schema import (
    COLUMNS,
    COST_FIELDS,
    ENUMS,
    FIELDS,
    POLARS_SCHEMA,
    CheckResult,
    DocType,
    DocumentRecord,
    MatchLevel,
    ValidationReport,
)
from ..values import find_numbers

__all__ = ["load_document", "validate_dir", "validate_document"]

LINE_ITEMS = "line_items.parquet"
DOCUMENT = "document.json"
_ORDER = {"skip": 0, "pass": 1, "warn": 2, "fail": 3}
_COST_BEARING = {
    DocType.INVOICE, DocType.BILL_DETAIL, DocType.USAGE_EXPORT, DocType.COST_SUMMARY,
    DocType.ESTIMATE, DocType.INFRA_DEFINITION,
}  # fmt: skip


def _check(
    name: str, status: str, summary: str, examples: list[Any] | None = None, **metrics: Any
) -> CheckResult:
    return CheckResult(
        name=name, status=status, summary=summary, metrics=metrics, examples=(examples or [])[:10]
    )


# A check scoring at least GOOD passes; between FAIR and GOOD it warns; below FAIR it fails.
GOOD = 0.90
FAIR = 0.70


def _rate_status(rate: float, pass_at: float = GOOD, warn_at: float = FAIR) -> str:
    return "pass" if rate >= pass_at else "warn" if rate >= warn_at else "fail"


def _tol(target: Decimal, rel: str) -> Decimal:
    return max(Decimal("0.02"), abs(target) * Decimal(rel))


def check_schema(lf: pl.LazyFrame, n: int) -> CheckResult:
    schema = lf.collect_schema()
    problems: list[str] = []
    if schema.names() != COLUMNS:
        missing = [c for c in COLUMNS if c not in schema.names()]
        extra = [c for c in schema.names() if c not in COLUMNS]
        problems.append(f"columns differ (missing {missing}, extra {extra})")
        return _check("schema", "fail", "; ".join(problems), problems)
    for name, dtype in POLARS_SCHEMA.items():
        if schema[name] != dtype:
            problems.append(f"{name} is {schema[name]}, expected {dtype}")
    if n == 0:
        status = "fail" if problems else "pass"
        return _check(
            "schema", status, "; ".join(problems) or "schema conforms (no rows)", problems
        )
    exprs = [
        pl.col("line_item_id").null_count().alias("null_ids"),
        (pl.len() - pl.col("line_item_id").n_unique()).alias("duplicate_ids"),
        pl.col("document_id").n_unique().alias("document_ids"),
        pl.col("record_type").null_count().alias("null_record_type"),
        (pl.col("billing_period_start") > pl.col("billing_period_end"))
        .sum()
        .alias("billing_period_reversed"),
        (pl.col("charge_period_start") > pl.col("charge_period_end"))
        .sum()
        .alias("charge_period_reversed"),
        (
            pl.col("billing_currency").is_not_null()
            & ~pl.col("billing_currency").str.contains(r"^[A-Z]{3}$")
        )
        .sum()
        .alias("bad_currency"),
    ]
    for field, enum in ENUMS.items():
        allowed = [e.value for e in enum]
        exprs.append(
            (pl.col(field).is_not_null() & ~pl.col(field).is_in(allowed))
            .sum()
            .alias(f"bad_{field}")
        )
    row = lf.select(exprs).collect().row(0, named=True)
    if row["document_ids"] > 1:
        problems.append(f"{row['document_ids']} different document ids")
    for key, value in row.items():
        if key != "document_ids" and value:
            problems.append(f"{key}: {value} row(s)")
    return _check(
        "schema", "fail" if problems else "pass",
        "; ".join(problems) or f"{len(COLUMNS)} columns, types, enums and ids conform", problems,
    )  # fmt: skip


def check_row_accounting(record: DocumentRecord, lf: pl.LazyFrame) -> CheckResult:
    tables = [t for t in record.tables if "rows_in" in t]
    if not tables:
        return _check("row_accounting", "skip", "not a tabular document")
    problems: list[str] = []
    rows_in = sum(t["rows_in"] for t in tables)
    failures = 0
    for t in tables:
        accounted = t["rows_out"] + t["rows_empty"] + t["rows_total"]
        if not t.get("melted") and accounted != t["rows_in"]:
            problems.append(
                f"table {t['index']}: {t['rows_in']} rows in, {accounted} accounted for"
            )
        if t.get("rollup_of"):
            continue  # a summary of other tables: its rows are not what gets counted
        for field, count in (t.get("parse_failures") or {}).items():
            failures += count
            problems.append(f"table {t['index']}: {count} value(s) of {field} could not be parsed")
    sums_checked = 0
    if not any(t.get("melted") for t in tables):
        expected: dict[str, Decimal] = {}
        for t in tables:
            for field, value in (t.get("source_sums") or {}).items():
                expected[field] = expected.get(field, Decimal(0)) + Decimal(value)
        if expected:
            got = (
                lf.select([pl.col(f).sum().alias(f) for f in expected]).collect().row(0, named=True)
            )
            for field, want in expected.items():
                sums_checked += 1
                have = got[field] or Decimal(0)
                if abs(have - want) > Decimal("0.000001"):
                    problems.append(f"{field}: source sums to {want}, output sums to {have}")
    hard = [p for p in problems if "could not be parsed" not in p]
    status = "fail" if hard or failures > 0.01 * max(rows_in, 1) else "warn" if failures else "pass"
    summary = (
        "; ".join(problems[:4])
        or f"{rows_in} source rows all accounted for; {sums_checked} cost sum(s) match"
    )
    return _check(
        "row_accounting", status, summary, problems, rows_in=rows_in, parse_failures=failures
    )


def check_table_mapping(record: DocumentRecord) -> CheckResult:
    """Whether the way each table's columns were read holds up against the table itself.

    Row accounting proves no row was lost; it cannot tell a sensible mapping from a wrong one.
    This tests the mapping's own claims with evidence from the source: rows called totals must
    add up the others, a column read as amounts must hold amounts, and columns that plainly
    hold amounts should not be left out.
    """
    tables = [t for t in record.tables if "cost_mapped" in t]
    if not tables:
        return _check("table_mapping", "skip", "no table mapping to test")
    failed: list[str] = []
    doubts: list[str] = []
    any_cost = any(t["cost_mapped"] for t in tables)
    for t in tables:
        where = f"table {t['index']}"
        # A table whose rows add up to the total it states has its cost column confirmed.
        confirmed = (
            t.get("table_total") is not None
            and t.get("kept_sum") is not None
            and abs(Decimal(t["table_total"]) - Decimal(t["kept_sum"]))
            <= _tol(Decimal(t["table_total"]), "0.005")
        )
        if t.get("nearly_equals"):
            doubts.append(
                f"{where} adds up to nearly the same amount as table(s) "
                f"{', '.join(str(i) for i in t['nearly_equals'])}; they may list the same "
                "charges twice"
            )
        unused = t.get("number_columns_unused") or []
        shown = ", ".join(repr(c) for c in unused[:3]) + (" …" if len(unused) > 3 else "")
        if not t["cost_mapped"] and unused:
            problem = f"{where}: no cost column was mapped although {len(unused)} column(s) hold amounts ({shown})"
            # With costs read from another table of the same file, this one is more likely
            # an analysis beside the costs than the costs themselves.
            (doubts if any_cost or t.get("record_type") == "resource" else failed).append(problem)
        elif (
            len(unused) >= 3
            and not confirmed
            and len(unused) > t.get("number_columns_used", 0)
            # A known export format is mapped by its specification, not by judgement.
            and not str((t.get("plan") or {}).get("source", "")).startswith("seed")
        ):
            doubts.append(
                f"{where}: {len(unused)} column(s) of amounts were left unused ({shown}), "
                f"more than the {t.get('number_columns_used', 0)} that were read"
            )
        if t.get("totals_sum") is not None and t.get("kept_sum") is not None:
            totals, kept = Decimal(t["totals_sum"]), Decimal(t["kept_sum"])
            # Totals restate the rows beneath them a whole number of times: once for a single
            # grand total, more where subtotals and a grand total are both present.
            times = (totals / kept) if kept else Decimal(0)
            whole = times.to_integral_value()
            if not (kept and whole >= 1 and abs(times - whole) <= Decimal("0.01")):
                problem = (
                    f"{where}: {t['rows_total']} row(s) were set aside as totals, but they add up "
                    f"to {totals:,.2f} while the remaining rows add up to {kept:,.2f}"
                )
                # Rows wrongly set aside are charges gone missing; how much decides how bad.
                heavy = abs(totals) > abs(kept) * Decimal("0.1")
                (failed if heavy else doubts).append(problem)
        for warning in t.get("warnings") or []:
            if "kept as attribute" in warning or "ignored" in warning or "dropped" in warning:
                doubts.append(f"{where}: {warning}")
    if failed:
        return _check("table_mapping", "fail", "; ".join(failed[:3]), failed + doubts)
    if doubts:
        return _check("table_mapping", "warn", "; ".join(doubts[:3]), doubts)
    return _check(
        "table_mapping", "pass", f"{len(tables)} table mapping(s) consistent with the source"
    )


def check_grounding(record: DocumentRecord, lf: pl.LazyFrame) -> CheckResult:
    stats = record.freeform
    if "values_claimed" not in stats:
        return _check("grounding", "skip", "not a free-form document")
    claimed, grounded = stats["values_claimed"], stats["values_grounded"]
    rate = grounded / claimed if claimed else 1.0
    # Independent re-check: every output amount must appear in the source line stored beside it.
    rows = (
        lf.filter(pl.col("billed_cost").is_not_null() & pl.col("source_text").is_not_null())
        .select("line_item_id", "billed_cost", "source_text")
        .collect()
    )
    unsupported = [
        {"line_item_id": i, "billed_cost": str(c), "source_text": t[:140]}
        for i, c, t in rows.iter_rows()
        if not any(abs(found) == abs(c) for found in find_numbers(t))
    ]
    status = "fail" if unsupported else _rate_status(rate)
    summary = f"{rate:.1%} grounded: {grounded} of {claimed} model-cited values found in their source lines"
    if unsupported:
        summary += f"; {len(unsupported)} output amount(s) not present in their source text"
    examples = unsupported or stats.get("values_rejected", [])
    return _check(
        "grounding", status, summary, examples, rate=round(rate, 4), unsupported=len(unsupported)
    )


def check_missed_lines(record: DocumentRecord) -> CheckResult:
    coverage = record.freeform.get("coverage")
    if not coverage:
        return _check("missed_lines", "skip", "not a free-form document")
    money, missed = coverage["money_lines"], coverage["missed"]
    failed_chunks = record.freeform.get("chunks_failed", 0)
    rate = 1 - (missed / money if money else 0.0)
    status = _rate_status(rate)
    if failed_chunks:
        status = "fail"
    summary = f"{rate:.1%} covered: {money - missed} of {money} money-bearing lines accounted for"
    if failed_chunks:
        summary += f"; {failed_chunks} part(s) of the document failed to tag"
    return _check("missed_lines", status, summary, coverage.get("examples"), rate=round(rate, 4))


def check_subtotals(record: DocumentRecord) -> CheckResult:
    stats = record.freeform
    checked = stats.get("subtotals_checked")
    if not checked:
        return _check("subtotals", "skip", "no heading subtotals to compare")
    ok = stats["subtotals_ok"]
    rate = ok / checked
    return _check(
        # Diagnostic only: it shows where inside the document sums disagree. Whether the
        # document as a whole is right is the reconciliation check's call.
        "subtotals", "pass" if rate >= GOOD else "warn",
        f"{rate:.1%} consistent: {ok} of {checked} heading subtotals equal the sum of their lines",
        stats.get("subtotal_mismatches"), rate=round(rate, 4),
    )  # fmt: skip


def _reconcile_tables(record: DocumentRecord) -> CheckResult | None:
    """Each table against the total it states for itself. A workbook's sheets are separate
    tables, and no single one of their totals need describe the whole file."""
    pairs = [
        (t, Decimal(t["table_total"]), Decimal(t["kept_sum"]))
        for t in record.tables
        if t.get("table_total") is not None and t.get("kept_sum") is not None
    ]
    if not pairs:
        return None
    results = []
    for t, stated, have in pairs:
        diff = abs(have - stated)
        rel = float(diff / abs(stated)) if stated else (0.0 if diff == 0 else 1.0)
        results.append(
            {
                "table": t["index"], "stated_label": f"table {t['index']} total",
                "stated": str(stated), "currency": None, "computed": str(have),
                "difference": str(diff), "relative": round(rel, 5),
                "match": round(max(0.0, 1.0 - rel), 5),
            }
        )  # fmt: skip
    worst = min(results, key=lambda r: r["match"])
    agree = sum(r["match"] >= GOOD for r in results)
    summary = (
        f"{worst['match']:.2%} match: table {worst['table']} rows sum to "
        f"{Decimal(worst['computed']):,.2f} vs its stated total {Decimal(worst['stated']):,.2f}"
        + (f"; {agree} of {len(results)} table totals agree" if len(results) > 1 else "")
    )
    results.sort(key=lambda r: r["match"])
    return _check(
        "reconciliation", _rate_status(worst["match"]), summary, results,
        relative=worst["relative"], match=worst["match"],
    )  # fmt: skip


def check_reconciliation(record: DocumentRecord, lf: pl.LazyFrame, n: int) -> CheckResult:
    if n and (by_table := _reconcile_tables(record)) is not None:
        return by_table
    doc_totals = whole_document_totals(record.stated_totals)
    if not doc_totals or n == 0:
        return _check("reconciliation", "skip", "document states no total to reconcile against")
    cost = pl.coalesce([pl.col(f) for f in COST_FIELDS])
    live = lf.filter(
        ~pl.col("is_rollup").fill_null(False) & pl.col("record_type").is_in(["charge", "estimate"])
    )
    by_currency = (
        live.group_by("billing_currency")
        .agg(
            cost.sum().alias("incl"),
            cost.filter(pl.col("charge_category").ne_missing("tax")).sum().alias("ex_tax"),
        )
        .collect()
    )
    sums = {r["billing_currency"]: r for r in by_currency.iter_rows(named=True)}
    # Items are compared with a total in their own currency. With more than one currency
    # among the items, the one most rows use is the document's line currency.
    counts = dict(live.group_by("billing_currency").len().collect().iter_rows())
    main = max(counts, key=counts.get) if counts else None
    if len(sums) > 1:
        sums = {main: sums[main]}
    best: dict[str, Any] | None = None
    for total in doc_totals:
        target = Decimal(total.amount)
        row = sums.get(total.currency) or (
            next(iter(sums.values())) if len(sums) == 1 and not total.currency else None
        )
        if row is None:
            continue
        for basis in ("incl", "ex_tax"):
            have = row[basis] or Decimal(0)
            diff = abs(have - target)
            rel = float(diff / abs(target)) if target else (0.0 if diff == 0 else 1.0)
            cand = {
                "stated_label": total.label, "stated": str(target), "currency": total.currency,
                "computed": str(have), "basis": basis, "difference": str(diff), "relative": round(rel, 5),
                "within": diff <= _tol(target, "0.005"),
            }  # fmt: skip
            if best is None or rel < best["relative"]:
                best = cand
    if best is None:
        return _check(
            "reconciliation", "warn", "stated totals are in a different currency from the line items",
            [t.model_dump() for t in doc_totals],
        )  # fmt: skip
    match = max(0.0, 1.0 - best["relative"])
    best["match"] = round(match, 5)
    summary = (
        f"{match:.2%} match: line items sum to {Decimal(best['computed']):,.2f} vs stated "
        f"{Decimal(best['stated']):,.2f} {best['currency'] or ''} ({best['stated_label']!r})"
    )
    return _check(
        "reconciliation", _rate_status(match), summary, [best],
        relative=best["relative"], match=round(match, 5),
    )  # fmt: skip


def check_arithmetic(lf: pl.LazyFrame) -> CheckResult:
    price = pl.coalesce(pl.col("contracted_unit_price"), pl.col("list_unit_price")).cast(pl.Float64)
    qty = pl.coalesce(pl.col("pricing_quantity"), pl.col("consumed_quantity")).cast(pl.Float64)
    cost = pl.coalesce(pl.col("contracted_cost"), pl.col("billed_cost"), pl.col("list_cost")).cast(
        pl.Float64
    )
    rows = lf.filter(
        price.is_not_null()
        & qty.is_not_null()
        & cost.is_not_null()
        & (pl.col("charge_category").eq_missing("usage") | pl.col("charge_category").is_null())
    )
    expected = price * qty
    ok = (expected - cost).abs() <= pl.max_horizontal(pl.lit(0.011), cost.abs() * 0.02)
    out = rows.select(pl.len().alias("n"), ok.sum().alias("ok")).collect().row(0, named=True)
    if not out["n"]:
        return _check("arithmetic", "skip", "no rows carry quantity, unit price and cost together")
    rate = out["ok"] / out["n"]
    examples = (
        rows.filter(~ok).select("line_item_id", "charge_description", qty.alias("quantity"), price.alias("unit_price"), cost.alias("cost"))
        .head(5).collect().to_dicts()
    )  # fmt: skip
    # Tiered and free-tier pricing legitimately break quantity x price, so this never fails a document.
    return _check(
        "arithmetic", "pass" if rate >= FAIR else "warn",
        f"{rate:.1%} consistent: quantity x unit price matches cost on {out['ok']} of {out['n']} rows", examples, rate=round(rate, 4),
    )  # fmt: skip


def check_domain(lf: pl.LazyFrame, n: int) -> CheckResult:
    if n == 0:
        return _check("domain", "skip", "no rows")
    day = pl.duration(days=1)
    outside = (pl.col("charge_period_start") < pl.col("billing_period_start") - day) | (
        pl.col("charge_period_start") > pl.col("billing_period_end") + day
    )
    count = lf.select(outside.sum()).collect().item()
    if count:
        return _check("domain", "warn", f"{count} row(s) are dated outside the billing period")
    return _check("domain", "pass", "dates are plausible")


def check_coverage(
    record: DocumentRecord, lf: pl.LazyFrame, n: int, report: ValidationReport
) -> CheckResult:
    if n == 0:
        if record.doc_type in _COST_BEARING:
            return _check(
                "coverage",
                "fail",
                f"document is a {record.doc_type} but no line items were extracted",
            )
        if record.doc_type is None:
            return _check("coverage", "fail", "document was not parsed")
        return _check("coverage", "skip", f"no line items expected ({record.doc_type})")
    fill = (
        lf.select(
            [
                pl.col(f.name).is_not_null().mean().alias(f.name)
                for f in FIELDS
                if f.dtype != pl.List(pl.String)
            ]
        )
        .collect()
        .row(0, named=True)
    )
    report.fill_rates = {k: round(v, 4) for k, v in fill.items() if v}
    levels = lf.group_by("match_level").len().collect()
    report.match_levels = {
        r["match_level"] or "none": r["len"] for r in levels.iter_rows(named=True)
    }
    best = next((lvl.value for lvl in MatchLevel if report.match_levels.get(lvl.value)), "none")
    share = report.match_levels.get(best, 0) / n
    return _check(
        "coverage",
        "pass",
        f"{n} line items; best catalog match level '{best}' on {share:.0%} of rows",
        best_level=best,
    )


def validate_document(record: DocumentRecord, parquet_path: Path) -> ValidationReport:
    report = ValidationReport()
    if record.error:
        report.checks.append(_check("parse", "fail", record.error))
        report.status = "fail"
        return report
    if not parquet_path.is_file():
        report.checks.append(_check("parse", "fail", "line_items.parquet is missing"))
        report.status = "fail"
        return report
    lf = pl.scan_parquet(parquet_path)
    n = lf.select(pl.len()).collect().item()
    checks = [check_schema(lf, n)]
    if checks[0].status != "fail":
        checks += [
            check_row_accounting(record, lf),
            check_table_mapping(record),
            check_grounding(record, lf),
            check_missed_lines(record),
            check_subtotals(record),
            check_reconciliation(record, lf, n),
            check_arithmetic(lf),
            check_domain(lf, n),
            check_coverage(record, lf, n, report),
        ]
    report.checks = checks
    report.status = max((c.status for c in checks), key=_ORDER.__getitem__)
    return report


def load_document(doc_dir: Path) -> DocumentRecord:
    return DocumentRecord.model_validate_json((doc_dir / DOCUMENT).read_text())


def validate_dir(out_dir: Path) -> list[DocumentRecord]:
    """Re-validate every document under an output directory, rewriting each document.json."""
    records = []
    for doc_json in sorted(out_dir.glob(f"*/{DOCUMENT}")):
        record = load_document(doc_json.parent)
        record.validation = validate_document(record, doc_json.parent / LINE_ITEMS)
        doc_json.write_text(record.model_dump_json(indent=1))
        records.append(record)
    return records
