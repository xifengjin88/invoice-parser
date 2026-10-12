from decimal import Decimal
from pathlib import Path

import polars as pl

from invoice_parser.convert.types import Source
from invoice_parser.llm import FakeLLM
from invoice_parser.pipeline import process_source
from invoice_parser.validate import LINE_ITEMS

BILL = """\
Acme Cloud Invoice
Account number: 111122223333
Total amount due USD 130.00
Summary
  Charges   USD 120.00
  Tax   USD 10.00
Detail for Linked Account
dev-sandbox (444455556666)   USD 130.00
 Elastic Compute Cloud   USD 100.00
 US West (Oregon)   USD 100.00
  Amazon Elastic Compute Cloud running Linux/UNIX   USD 100.00
    $0.0416 per On Demand Linux t3.medium Instance Hour   1,000 Hrs   USD 41.60
    $0.0832 per On Demand Linux t3.large Instance Hour   1,000 Hrs   USD 83.20
    t3.large Linux instance usage covered by Compute Savings Plans   297.115 Hrs   (USD 24.80)
 Simple Storage Service   USD 20.00
 Tax   USD 10.00
"""

META = {
    "doc_type": "invoice", "has_extractable_lines": True, "summary": "test invoice", "provider": "AWS",
    "language": "en", "record_type": "charge", "currency": "USD", "invoice_id": "INV-1",
    "billing_period_start": "2026-06-01", "billing_period_end": "2026-06-30",
    "billing_account_id": "111122223333",
    "totals": [
        {"label": "Total amount due", "amount": "130.00", "currency": "USD", "scope": "grand_total"},
        {"label": "Invented", "amount": "999.99", "currency": "USD", "scope": "grand_total"},
    ],
}  # fmt: skip

TAGS = [
    {"line": 4, "kind": "section", "label": "Summary"},
    {"line": 5, "kind": "item", "label": "Charges", "amount": "120.00", "charge": "usage"},
    {"line": 6, "kind": "item", "label": "Tax", "amount": "10.00", "charge": "tax"},
    {"line": 7, "kind": "section", "label": "Detail for Linked Account"},
    {"line": 8, "kind": "account", "label": "dev-sandbox (444455556666)", "account_id": "444455556666", "amount": "130.00"},
    {"line": 9, "kind": "service", "label": "Elastic Compute Cloud", "amount": "100.00"},
    {"line": 10, "kind": "region", "label": "US West (Oregon)", "amount": "100.00"},
    # the model mislabels this sub-heading as an item; code repairs it from the indentation
    {"line": 11, "kind": "item", "label": "Amazon Elastic Compute Cloud running Linux/UNIX", "amount": "100.00"},
    {"line": 12, "kind": "item", "label": "$0.0416 per On Demand Linux t3.medium Instance Hour",
     "amount": "41.60", "quantity": "1,000", "unit": "Hrs", "unit_price": "0.0416", "charge": "usage"},
    # hallucinated amount: the line says 83.20
    {"line": 13, "kind": "item", "label": "$0.0832 per On Demand Linux t3.large Instance Hour",
     "amount": "88.20", "quantity": "1,000", "unit": "Hrs", "charge": "usage"},
    {"line": 14, "kind": "item", "label": "t3.large Linux instance usage covered by Compute Savings Plans",
     "amount": "24.80", "quantity": "297.115", "unit": "Hrs", "charge": "discount"},
    # a heading with a subtotal and nothing beneath it becomes a line item itself
    {"line": 15, "kind": "service", "label": "Simple Storage Service", "amount": "20.00"},
    {"line": 16, "kind": "item", "label": "Tax", "amount": "10.00", "charge": "tax"},
]  # fmt: skip


def _run(
    tmp_path: Path,
    tags: list[dict],
    text: str = BILL,
    meta: dict = META,
    name: str = "bill.txt",
    describe=None,
    resources=None,
):
    path = tmp_path / name
    path.write_text(text)
    llm = FakeLLM(
        {
            "doc_meta": lambda s, u: meta,
            "line_tags": lambda s, u: {"rows": tags},
            "describe": describe or (lambda s, u: {"rows": []}),
            "resources": lambda s, u: {"resources": resources or []},
        }
    )
    record = process_source(Source(name, path), tmp_path / "out", llm)
    frame = pl.read_parquet(tmp_path / "out" / record.document_id / LINE_ITEMS)
    return record, frame


def _check(record, name):
    return next(c for c in record.validation.checks if c.name == name)


def test_hierarchy_grounding_and_rollups(tmp_path):
    record, frame = _run(tmp_path, TAGS)
    detail = frame.filter(~pl.col("is_rollup"))
    summary = frame.filter(pl.col("is_rollup"))
    assert summary["section"].unique().to_list() == ["Summary"]
    # t3.medium, t3.large (amount rejected, so no cost), savings-plan line, S3 leaf, tax
    assert detail.height == 5 and detail["billed_cost"].null_count() == 1

    medium = detail.row(0, named=True)
    assert medium["service_name"] == "Elastic Compute Cloud"
    # stated values only: no region code or instance type is guessed from the wording by code
    assert medium["region_name"] == "US West (Oregon)" and medium["region_id"] is None
    assert medium["sku_meter"] == "Amazon Elastic Compute Cloud running Linux/UNIX"
    assert (
        medium["sub_account_id"] == "444455556666" and medium["sub_account_name"] == "dev-sandbox"
    )
    assert medium["billing_account_id"] == "111122223333" and medium["invoice_id"] == "INV-1"
    assert medium["instance_type"] is None and medium["consumed_quantity"] == Decimal("1000")
    assert medium["billed_cost"] == Decimal("41.60") and medium["source_row"] == 12
    assert medium["match_level"] == "description"

    # sign comes from the source text "(USD 24.80)", not from the model's "24.80"
    covered = detail.filter(pl.col("charge_category") == "discount").row(0, named=True)
    assert covered["billed_cost"] == Decimal("-24.80")

    s3 = detail.filter(pl.col("service_name") == "Simple Storage Service").row(0, named=True)
    assert s3["billed_cost"] == Decimal("20.00") and s3["granularity"] == "service"
    assert s3["region_name"] is None  # the new service heading cleared the old region
    tax = detail.filter(pl.col("charge_category") == "tax").row(0, named=True)
    assert tax["service_name"] is None  # outdented line left the service heading

    assert record.freeform["leaf_headings"] >= 1
    assert record.freeform["values_rejected"][0]["claimed"] == "88.20"
    assert [t.amount for t in record.stated_totals if t.scope == "document"] == ["130.00"]
    assert any("not found in the text" in w for w in record.warnings)

    # the validator notices the damage the dropped line caused
    assert (
        _check(record, "grounding").status == "fail"
        or _check(record, "grounding").metrics["rate"] < 0.99
    )
    assert _check(record, "subtotals").status in ("warn", "fail")
    assert _check(record, "reconciliation").status == "fail"
    assert record.validation.status == "fail"


def test_correct_tags_reconcile_and_pass(tmp_path):
    tags = [dict(t) for t in TAGS]
    tags[9]["amount"] = "83.20"
    meta = dict(META, totals=META["totals"][:1])
    record, frame = _run(tmp_path, tags, meta=meta)
    detail = frame.filter(~pl.col("is_rollup"))
    assert detail["billed_cost"].sum() == Decimal("130.00")
    assert record.freeform["subtotals_checked"] == record.freeform["subtotals_ok"] >= 3
    assert _check(record, "reconciliation").status == "pass"
    assert _check(record, "grounding").status == "pass"
    assert record.validation.status == "pass", [
        (c.name, c.status, c.summary) for c in record.validation.checks
    ]


def test_infrastructure_definition_yields_resource_records(tmp_path):
    text = 'provider "aws" {\n  region = "us-west-2"\n}\nresource "aws_instance" "api" {\n  instance_type = "t3.xlarge"\n  count = 3\n}\n'
    meta = {
        "doc_type": "infra_definition", "has_extractable_lines": True, "summary": "terraform",
        "provider": "aws", "record_type": "resource", "totals": [],
    }  # fmt: skip
    resources = [{
        "line": 3,  # miscounted: the resource starts on line 4
        "label": "aws_instance.api", "resource_type": "aws_instance", "resource_name": "api",
        "attrs": [
            {"k": "instance_type", "v": "t3.xlarge"}, {"k": "count", "v": "3"},
            {"k": "region", "v": "us-west-2"},  # written in the provider block, not this one
            {"k": "operating_system", "v": "Windows"},  # nowhere in the file: dropped
        ],
    }]  # fmt: skip
    record, frame = _run(tmp_path, [], text=text, meta=meta, name="main.tf", resources=resources)
    row = frame.row(0, named=True)
    assert frame.height == 1  # one row per resource, not one per line of its block
    assert row["record_type"] == "resource" and row["granularity"] == "resource"
    assert row["resource_type"] == "aws_instance" and row["source_row"] == 4
    assert row["derived_fields"] == ["region_name"]  # resolved from elsewhere in the file
    assert row["instance_type"] == "t3.xlarge" and row["region_name"] == "us-west-2"
    assert row["consumed_quantity"] == Decimal("3") and row["operating_system"] is None
    assert record.freeform["attrs_dropped"] == 1
    # three of four claimed values were in the file; the invented one shows up as a warning
    assert _check(record, "grounding").status == "warn"
    assert all(
        c.status in ("pass", "skip") for c in record.validation.checks if c.name != "grounding"
    )


def test_non_cost_document_is_recorded_with_a_reason(tmp_path):
    meta = {
        "doc_type": "not_cost_related",
        "has_extractable_lines": False,
        "summary": "a memo",
        "record_type": "charge",
        "totals": [],
    }
    record, frame = _run(
        tmp_path, [], text="Hello team,\nlunch is at noon.\n", meta=meta, name="memo.txt"
    )
    assert frame.height == 0 and record.doc_type == "not_cost_related"
    assert record.validation.status in ("pass", "skip")


def test_tags_are_anchored_by_text_when_the_model_miscounts_lines():
    from invoice_parser.freeform.tagging import Line, Tag, anchor_tags

    texts = [
        "Compute   USD 30.00",
        "  West region   USD 30.00",
        "    Small instance hours   100 Hrs   USD 10.00",
        "    Large instance hours   50 Hrs   USD 20.00",
        "Storage   USD 5.00",
        "  West region   USD 5.00",
    ]
    lines = [Line(1, i + 1, t) for i, t in enumerate(texts)]
    tags = [
        Tag(line=1, kind="service", label="Compute", amount="30.00"),
        Tag(line=2, kind="region", label="West region", amount="30.00"),
        Tag(line=4, kind="item", label="Small instance hours", amount="10.00"),  # really line 3
        Tag(line=6, kind="item", label="Large instance hours", amount="20.00"),  # really line 4
        Tag(line=3, kind="service", label="Storage", amount="5.00"),  # really line 5
        Tag(line=2, kind="region", label="West region", amount="5.00"),  # really line 6
    ]
    pairs, moved = anchor_tags(lines, tags)
    assert [line.number for line, _ in pairs] == [1, 2, 3, 4, 5, 6]
    assert moved == 4


def test_pricing_attributes_are_read_from_wording_and_checked(tmp_path):
    seen = {}

    def describe(system, user):
        seen["entries"] = user.splitlines()[1:]
        ids = {ln.split("description: ")[-1]: int(ln.split(" | ")[0][3:]) for ln in seen["entries"]}
        medium = next(i for d, i in ids.items() if "t3.medium" in d)
        covered = next(i for d, i in ids.items() if "covered by" in d)
        return {
            "rows": [
                {
                    "id": medium, "product_family": "compute", "charge_frequency": "usage_based",
                    "instance_type": "t3.medium", "operating_system": "Linux",
                    "purchase_option": "on_demand", "purchase_option_wording": "On Demand",
                    "pricing_unit": "Instance Hour", "unit_price": "0.0416",
                    "commitment_term": "3 Year",  # not in the wording: rejected
                    "discount_type": "savings_plan",  # not a discount row: ignored
                },
                {
                    "id": covered, "discount_type": "savings_plan",
                    "purchase_option": "reserved", "purchase_option_wording": "Reserved Instance",
                },
            ]
        }  # fmt: skip

    tags = [dict(t) for t in TAGS]
    tags[9]["amount"] = "83.20"
    tags[8] = {k: v for k, v in tags[8].items() if k != "unit_price"}
    meta = dict(META, totals=META["totals"][:1])
    record, frame = _run(tmp_path, tags, meta=meta, describe=describe)
    detail = frame.filter(~pl.col("is_rollup"))
    medium = detail.filter(pl.col("charge_description").str.contains("t3.medium")).row(
        0, named=True
    )
    assert medium["instance_type"] == "t3.medium" and medium["operating_system"] == "Linux"
    assert medium["purchase_option"] == "on_demand" and medium["pricing_unit"] == "Instance Hour"
    assert medium["contracted_unit_price"] == Decimal("0.0416")
    assert medium["product_family"] == "compute" and medium["charge_frequency"] == "usage_based"
    assert medium["commitment_term"] is None and medium["discount_type"] is None
    assert "product_family" in medium["derived_fields"]
    assert medium["match_level"] == "attributes"
    covered = detail.filter(pl.col("charge_category") == "discount").row(0, named=True)
    # the category was claimed with wording the entry does not contain, so it is not accepted
    assert covered["purchase_option"] is None and covered["discount_type"] == "savings_plan"
    assert record.freeform["described"]["values_rejected"] == 2
    # rollup rows are not sent, and each distinct wording is sent once
    assert not any(ln.endswith("description: Charges") for ln in seen["entries"])
    assert len(seen["entries"]) == record.freeform["described"]["wordings"]
    assert record.schema_version == "0.2" and frame.width == 75


def test_a_charge_printed_twice_becomes_one_row():
    from invoice_parser.freeform.walk import combine_restated

    def row(page, line, section, service, description, amount, rollup, **extra):
        return {
            "source_page": page, "source_row": line, "section": section, "is_rollup": rollup,
            "service_name": service, "charge_description": description,
            "billed_cost": Decimal(amount), "billing_currency": "USD", **extra,
        }  # fmt: skip

    rows = [
        row(
            1,
            5,
            "Summary",
            "Compute Cloud / Compute Cloud",
            "Charges / Frais",
            "93.05",
            False,
            charge_category="usage",
        ),
        row(1, 6, "Summary", "Storage Service", "Charges", "11.20", False),
        row(1, 7, "Summary", "Queue Service", "Tax", "0.50", False),
        # the same three charges again, worded a little differently, with the account this time
        row(
            4,
            2,
            "By account",
            "Compute Cloud",
            "Charges / Frais",
            "93.05",
            True,
            sub_account_id="111122223333",
        ),
        row(
            4,
            3,
            "By account",
            "Storage Service",
            "Charges",
            "11.20",
            True,
            sub_account_id="111122223333",
        ),
        # same amount but a different service: not the same charge
        row(4, 4, "By account", "Notification Service", "Tax", "0.50", True),
        # an aggregate with no twin stays as a flagged rollup
        row(4, 9, "By account", None, "Account total", "104.75", True),
    ]
    out, stats = combine_restated(rows)
    assert stats == {"combined": 2, "rollups_left": 2}
    assert len(out) == 5
    compute = out[0]
    assert compute["sub_account_id"] == "111122223333" and compute["charge_category"] == "usage"
    assert "p4 l2 (By account)" in compute["attributes"]
    assert [r["charge_description"] for r in out if r["is_rollup"]] == ["Tax", "Account total"]
