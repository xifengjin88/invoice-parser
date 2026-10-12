from decimal import Decimal
from pathlib import Path

import polars as pl

from invoice_parser.convert.types import Source
from invoice_parser.llm import FakeLLM
from invoice_parser.pipeline import process_source
from invoice_parser.schema import COLUMNS
from invoice_parser.validate import LINE_ITEMS


def _run(tmp_path: Path, name: str, text: str, plan: dict | None):
    path = tmp_path / name
    path.write_text(text)
    llm = FakeLLM({"table_plan": lambda system, user: plan}) if plan else None
    record = process_source(Source(name, path), tmp_path / "out", llm)
    frame = pl.read_parquet(tmp_path / "out" / record.document_id / LINE_ITEMS)
    return record, frame, llm


def _status(record, name):
    return next(c for c in record.validation.checks if c.name == name).status


PLAN = {
    "is_cost_data": True, "summary": "legacy AWS bill", "provider": "aws", "header_row": 0,
    "record_type": "charge", "granularity": "usage_type",
    "columns": [
        {"column": "RecordType", "use": "charge_category"},
        {"column": "ProductName", "use": "service_name"},
        {"column": "UsageType", "use": "usage_type"},
        {"column": "ItemDescription", "use": "charge_description"},
        {"column": "UsageQuantity", "use": "consumed_quantity"},
        {"column": "CurrencyCode", "use": "billing_currency"},
        {"column": "CostBeforeTax", "use": "billed_cost"},
        {"column": "Owner", "use": "attribute"},
        {"column": "NoSuchColumn", "use": "sku_id"},
    ],
    "constants": [{"field": "billing_period_start", "value": "2026-04-01"}],
    "pivot": None,
    "total_rows": [{"column": "RecordType", "values": ["InvoiceTotal"]}],
}  # fmt: skip

CSV = (
    "RecordType,ProductName,UsageType,ItemDescription,UsageQuantity,CurrencyCode,CostBeforeTax,Owner\n"
    'LineItem,Amazon Elastic Compute Cloud,USW2-BoxUsage:c5a.4xlarge,"$1.352 per On Demand Windows c5a.4xlarge Instance Hour",700,USD,"946.40",team-a\n'
    "LineItem,Amazon Simple Storage Service,EUC1-TimedStorage-ByteHrs,$0.023 per GB,5.048e-05,USD,0.00,\n"
    ",,,,,,,\n"
    "InvoiceTotal,,,Total,,USD,946.40,\n"
)


def test_model_plan_is_applied_by_code_with_full_row_accounting(tmp_path):
    record, frame, llm = _run(tmp_path, "export.csv", CSV, PLAN)
    assert frame.columns == COLUMNS
    assert frame.height == 2
    assert frame["billed_cost"].sum() == Decimal("946.40")
    first = frame.row(0, named=True)
    assert first["provider"] == "aws" and first["extraction_method"] == "llm_column_map"
    assert first["source_row"] == 2
    # derived by code, and labelled as such
    # only what the plan mapped is filled: code does not read meaning out of the wording
    assert first["region_id"] is None and first["instance_type"] is None
    assert first["usage_type"] == "USW2-BoxUsage:c5a.4xlarge"
    assert sorted(first["derived_fields"]) == ["billing_period_start", "provider"]
    assert first["match_level"] == "usage_type"
    assert first["attributes"] == '{"Owner":"team-a"}'
    table = record.tables[0]
    assert (table["rows_in"], table["rows_out"], table["rows_empty"], table["rows_total"]) == (
        4,
        2,
        1,
        1,
    )
    assert any("NoSuchColumn" in w for w in table["warnings"])
    assert record.stated_totals[0].amount == "946.4000000000"
    assert _status(record, "row_accounting") == "pass"
    assert _status(record, "reconciliation") == "pass"
    assert record.validation.status == "pass"
    # the model saw a sample, never produced a value
    assert len(llm.calls) == 1


def test_pivoted_table_is_melted(tmp_path):
    csv = (
        '"Service","EC2-Instances($)","S3($)","Total costs($)"\n'
        '"Service total","30","3","33"\n'
        '"2026-07-01","10","1","11"\n'
        '"2026-07-02","20","2","22"\n'
    )
    plan = {
        "is_cost_data": True, "summary": "daily cost by service", "provider": "aws", "header_row": 0,
        "record_type": "charge", "granularity": "service",
        "columns": [
            {"column": "Service", "use": "charge_period_start"},
            {"column": "EC2-Instances($)", "use": "pivot_value"},
            {"column": "S3($)", "use": "pivot_value"},
            {"column": "Total costs($)", "use": "ignore"},
        ],
        "constants": [{"field": "billing_currency", "value": "USD"}],
        "pivot": {"label_field": "service_name", "value_field": "billed_cost", "label_strip": "($)"},
        "total_rows": [{"column": "Service", "values": ["Service total"]}],
    }  # fmt: skip
    record, frame, _ = _run(tmp_path, "ce.csv", csv, plan)
    assert frame.height == 4
    assert sorted(frame["service_name"].unique()) == ["EC2-Instances", "S3"]
    assert frame["billed_cost"].sum() == Decimal("33")
    assert frame["charge_period_start"].null_count() == 0
    assert frame["line_item_id"].n_unique() == 4
    assert record.doc_type == "cost_summary"
    assert record.validation.status == "pass"


def test_known_format_needs_no_model(tmp_path):
    csv = (
        "BilledCost,BillingCurrency,ChargeCategory,ChargePeriodStart,ServiceName,SkuId,ProviderName\n"
        "1.25,USD,Usage,2026-06-01T00:00:00+00:00,Amazon MemoryDB,FR8PTRK3KEFVGG6Y,AWS\n"
    )
    record, frame, _ = _run(tmp_path, "focus.csv", csv, None)
    assert record.extraction_method == "known_format"
    assert frame["match_level"][0] == "sku" and frame["charge_category"][0] == "usage"


def test_table_the_model_rejects_yields_no_rows(tmp_path):
    plan = dict(PLAN, is_cost_data=False, columns=[], total_rows=[], constants=[])
    record, frame, _ = _run(tmp_path, "kpi.csv", "Tile,Value\nSavings,10%\nHappy,yes\n", plan)
    assert frame.height == 0 and record.doc_type == "not_cost_related"
    assert "not cost data" in record.tables[0]["skipped"]


def test_column_that_does_not_fit_its_field_is_kept_as_attribute(tmp_path):
    columns = [
        {"column": "ProductName", "use": "service_name"},
        {"column": "Owner", "use": "billed_cost"},
    ]
    plan = dict(PLAN, columns=columns, total_rows=[])
    record, frame, _ = _run(tmp_path, "bad.csv", CSV, plan)
    assert frame["billed_cost"].null_count() == frame.height
    assert any("not numeric" in w for w in record.tables[0]["warnings"])
    assert '"Owner":"team-a"' in frame["attributes"][0]


def test_table_mapping_check_tests_the_mappings_own_claims():
    from types import SimpleNamespace

    from invoice_parser.validate import check_table_mapping
    from invoice_parser.values import looks_like_dates

    assert looks_like_dates(["2026-07-01", "2026-08-01"]) and looks_like_dates(["Jul 2026"])
    # numbers are never dates, however date-like: a compact date, a year, a small range
    assert not looks_like_dates(["20260701"]) and not looks_like_dates(["2026", "6396", "16.1"])
    assert not looks_like_dates(["BigQuery", "m5.large"])

    def status(**table):
        base = {"index": 0, "cost_mapped": True, "rows_total": 0, "plan": {"source": "llm"}}
        return check_table_mapping(SimpleNamespace(tables=[{**base, **table}])).status

    assert status() == "pass"
    # a real grand total, and subtotals plus a grand total, restate the rows a whole number of times
    assert status(rows_total=1, totals_sum="100.00", kept_sum="100.00") == "pass"
    assert status(rows_total=4, totals_sum="300.00", kept_sum="100.00") == "pass"
    # data rows mistaken for totals do not
    assert status(rows_total=5, totals_sum="48315.70", kept_sum="56561.84") == "fail"
    assert status(rows_total=11, totals_sum="0", kept_sum="476869.88") == "warn"
    # amounts present but none read as cost
    assert status(cost_mapped=False, number_columns_unused=["Monthly cost"]) == "fail"
    assert (
        status(cost_mapped=False, number_columns_unused=["RAM GB"], record_type="resource")
        == "warn"
    )
    many = ["a", "b", "c"]
    assert status(number_columns_unused=many, number_columns_used=1) == "warn"
    assert (
        status(number_columns_unused=many, number_columns_used=1, plan={"source": "seed:cur"})
        == "pass"
    )
    assert (
        status(
            warnings=["'Service' was mapped to billed_cost but values are dates; kept as attribute"]
        )
        == "warn"
    )


def _plan(columns: dict[str, str], totals: dict[str, list[str]] | None = None) -> dict:
    return {
        "is_cost_data": True, "summary": "test table", "provider": "aws", "header_row": 0,
        "record_type": "charge", "granularity": "service",
        "columns": [{"column": c, "use": u} for c, u in columns.items()],
        "constants": [], "pivot": None,
        "total_rows": [{"column": c, "values": v} for c, v in (totals or {}).items()],
    }  # fmt: skip


def test_a_row_that_adds_up_all_the_others_is_the_total_even_unlabelled(tmp_path):
    text = "Name,Monthly\nvm-a,60.00\nvm-b,30.00\nvm-c,10.00\nAll machines,100.00\n"
    plan = _plan({"Name": "resource_name", "Monthly": "billed_cost"})  # names no total row
    record, frame, _ = _run(tmp_path, "machines.csv", text, plan)
    flagged = frame.filter(pl.col("is_rollup"))
    assert flagged["resource_name"].to_list() == ["All machines"]
    assert frame.filter(~pl.col("is_rollup"))["billed_cost"].sum() == Decimal("100.00")
    assert [(t.amount, t.scope) for t in record.stated_totals] == [("100.000000", "table")]
    assert _status(record, "reconciliation") == "pass"

    # no row equals the rest: nothing is taken for a total
    text = "Name,Monthly\nvm-a,60.00\nvm-b,30.00\nvm-c,20.00\n"
    (tmp_path / "b").mkdir()
    record, frame, _ = _run(tmp_path / "b", "machines.csv", text, plan)
    assert not frame["is_rollup"].any() and record.stated_totals == []


def test_the_same_charges_listed_twice_in_one_table_are_counted_once(tmp_path):
    text = (
        "RecordType,Account,Product,Cost\n"
        "LinkedLineItem,111,Compute,40.00\n"
        "LinkedLineItem,111,Storage,10.00\n"
        "LinkedLineItem,222,Compute,50.00\n"
        "PayerLineItem,999,Compute,90.00\n"
        "PayerLineItem,999,Storage,10.00\n"
        "InvoiceTotal,,,100.00\n"
    )
    plan = _plan(
        {"RecordType": "attribute", "Account": "sub_account_id", "Product": "service_name",
         "Cost": "billed_cost"},
        {"RecordType": ["InvoiceTotal"]},
    )  # fmt: skip
    record, frame, _ = _run(tmp_path, "statement.csv", text, plan)
    live = frame.filter(~pl.col("is_rollup"))
    assert live.height == 3 and live["billed_cost"].sum() == Decimal("100.00")
    assert set(live["sub_account_id"]) == {"111", "222"}  # the more detailed listing is kept
    assert _status(record, "reconciliation") == "pass"


def test_a_summary_table_that_restates_detail_tables_is_a_rollup(tmp_path):
    text = (
        "Category,Monthly\nCompute,100.00\nStorage,50.00\n"
        "\n\n"
        "Machine,Cost\nvm-a,60.00\nvm-b,30.00\nvm-c,10.00\n"
        "\n\n"
        "Disk,Cost\ndisk-a,20.00\ndisk-b,20.00\ndisk-c,10.00\n"
    )
    plans = {
        "Category": _plan({"Category": "service_name", "Monthly": "billed_cost"}),
        "Machine": _plan({"Machine": "resource_name", "Cost": "billed_cost"}),
        "Disk": _plan({"Disk": "resource_name", "Cost": "billed_cost"}),
    }
    path = tmp_path / "workbook.csv"
    path.write_text(text)
    llm = FakeLLM({"table_plan": lambda s, u: next(p for key, p in plans.items() if key in u)})
    record = process_source(Source(path.name, path), tmp_path / "out", llm)
    frame = pl.read_parquet(tmp_path / "out" / record.document_id / LINE_ITEMS)

    assert len([t for t in record.tables if "rows_out" in t]) == 3
    assert record.tables[0]["rollup_of"] == [1, 2]
    flagged = frame.filter(pl.col("is_rollup"))
    assert sorted(flagged["service_name"].to_list()) == ["Compute", "Storage"]
    assert frame.filter(~pl.col("is_rollup"))["billed_cost"].sum() == Decimal("150.00")
