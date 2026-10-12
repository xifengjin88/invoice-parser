"""The arithmetic rules that decide what is a charge and what only restates one.

Each test is a small made-up document showing one rule, with the labels a model would give.
No model is called: the rules are code, and these are what keep them from regressing.
"""

from decimal import Decimal

from invoice_parser.convert.types import TextDoc
from invoice_parser.freeform.tagging import DocMeta, Line, Tag
from invoice_parser.freeform.walk import (
    Walker,
    mark_rollups,
    net_of_tax_totals,
    whole_document_totals,
)
from invoice_parser.schema import StatedTotal

META = DocMeta.model_validate(
    {
        "doc_type": "invoice", "has_extractable_lines": True, "summary": "", "provider": "aws",
        "language": "en", "record_type": "charge", "currency": "USD", "totals": [],
    }
)  # fmt: skip


def total(amount: str, scope: str = "document", label: str = "Total") -> StatedTotal:
    return StatedTotal(label=label, amount=amount, currency="USD", scope=scope, origin="summary")


def walk(pages: list[list[tuple[str, dict]]], totals: list[StatedTotal]):
    """Run the hierarchy and rollup steps over pages of (line text, tag) pairs."""
    doc = TextDoc(pages=[[text for text, _ in page] for page in pages], origin="text")
    pairs = [
        (Line(p + 1, n + 1, text), Tag.model_validate({"line": n + 1, **tag}))
        for p, page in enumerate(pages)
        for n, (text, tag) in enumerate(page)
        if tag
    ]
    result = Walker(doc, META, {"record_type": "charge", "provider": "aws"}, totals).run(pairs)
    mark_rollups(result.rows, [*totals, *result.stated_totals], "USD")
    return result.rows


def live(rows) -> Decimal:
    return sum((r["billed_cost"] or 0 for r in rows if not r.get("is_rollup")), Decimal(0))


def item(label, amount, charge="usage", **more):
    return {"kind": "item", "label": label, "amount": amount, "charge": charge, **more}


def test_a_title_that_recurs_with_each_block_is_a_section_not_a_page_header():
    # Every page opens with the same title, which is what a running header looks like. But
    # another title always follows before it comes round again, so it heads a real block: the
    # per-account summary, which restates the detail under it.
    def page(account: str, charge: str, half: str, tax: str, both: str):
        return [
            ("Summary for Account", {"kind": "section", "label": "Summary for Account"}),
            (f"  {account}   ${both}", {"kind": "account", "label": account, "amount": both}),
            (f"    Charges   ${charge}", item("Charges", charge)),
            (f"    Tax   ${tax}", item("Tax", tax, "tax")),
            ("Detail for Account", {"kind": "section", "label": "Detail for Account"}),
            (f"  Compute   ${both}", {"kind": "service", "label": "Compute", "amount": both}),
            (f"    Instance hours   ${half}", item("Instance hours", half)),
            (f"    Volume storage   ${half}", item("Volume storage", half)),
            (f"    Tax   ${tax}", item("Tax", tax, "tax")),
        ]

    pages = [
        page("alpha (111111111111)", "100.00", "50.00", "10.00", "110.00"),
        page("beta (222222222222)", "200.00", "100.00", "20.00", "220.00"),
        page("gamma (333333333333)", "50.00", "25.00", "5.00", "55.00"),
        page("delta (444444444444)", "40.00", "20.00", "4.00", "44.00"),
    ]
    rows = walk(pages, [total("429.00")])
    assert live(rows) == Decimal("429.00")
    kept = {r["section"] for r in rows if not r["is_rollup"]}
    assert kept == {"Detail for Account"}
    assert {r["section"] for r in rows if r["is_rollup"]} == {"Summary for Account"}


def test_a_title_repeated_with_nothing_between_is_a_page_header():
    def page(n: int):
        return [
            ("Charges by service", {"kind": "section", "label": "Charges by service"}),
            (f" Service {n}   USD 10.00", {"kind": "service", "label": f"Service {n}", "amount": "10.00"}),
            ("   Usage   USD 10.00", item("Usage", "10.00")),
        ]  # fmt: skip

    rows = walk([page(n) for n in range(4)], [total("40.00")])
    assert live(rows) == Decimal("40.00") and {r["section"] for r in rows} == {"Charges by service"}


def test_a_total_whose_following_lines_add_up_to_it_heads_its_own_block():
    # The charges are already net of the savings. The savings block lists them again, so
    # counting it would take them off twice.
    pages = [
        [
            ("Charges by service", {"kind": "section", "label": "Charges by service"}),
            (" Compute   USD 70.00", {"kind": "service", "label": "Compute", "amount": "70.00"}),
            ("   On-demand hours   USD 70.00", item("On-demand hours", "70.00")),
            (" Storage   USD 30.00", {"kind": "service", "label": "Storage", "amount": "30.00"}),
            ("   GB-months   USD 30.00", item("GB-months", "30.00")),
            (
                "Savings (2)   Total savings (USD 25.00)",
                {"kind": "total", "label": "Savings (2)", "amount": "(25.00)", "total_scope": "credits"},
            ),
            (" Compute   (USD 20.00)", {"kind": "service", "label": "Compute", "amount": "(20.00)"}),
            (" Storage   (USD 5.00)", {"kind": "service", "label": "Storage", "amount": "(5.00)"}),
        ]
    ]  # fmt: skip
    rows = walk(pages, [total("100.00")])
    assert live(rows) == Decimal("100.00")
    assert sorted(r["billed_cost"] for r in rows if r["is_rollup"]) == [-20, -5]


def test_one_line_repeating_a_total_does_not_start_a_block():
    # A wrapped amount labelled as a total, followed by the single charge it belongs to.
    pages = [
        [
            ("Charges", {"kind": "section", "label": "Charges"}),
            (" Compute   USD 60.00", {"kind": "service", "label": "Compute", "amount": "60.00"}),
            ("   Hours   USD 60.00", item("Hours", "60.00")),
            (" Credits   (USD 20.00)", {"kind": "service", "label": "Credits", "amount": "(20.00)"}),
            (
                "                              (USD 20.00)",
                {"kind": "total", "label": "", "amount": "(20.00)", "total_scope": "credits"},
            ),
            ("   Promotional credit   (USD 20.00)", item("Promotional credit", "(20.00)", "credit")),
        ]
    ]  # fmt: skip
    rows = walk(pages, [total("40.00")])
    assert live(rows) == Decimal("40.00") and {r["section"] for r in rows} == {"Charges"}


def test_a_name_in_the_page_footer_is_not_part_of_the_charge_above_it():
    pages = [
        [
            ("Detail", {"kind": "section", "label": "Detail"}),
            ("Inspector   USD 49.25", {"kind": "service", "label": "Inspector", "amount": "49.25"}),
            ("   Charges   USD 43.58", item("Charges", "43.58")),
            (
                "                                   Example Cloud Canada, Inc.",
                {"kind": "seller", "label": "Example Cloud Canada, Inc."},
            ),
        ],
        [("   Tax   USD 5.67", item("Tax", "5.67", "tax"))],
    ]
    rows = walk(pages, [total("49.25")])
    assert sorted(r["billed_cost"] for r in rows if not r["is_rollup"]) == [
        Decimal("5.67"),
        Decimal("43.58"),
    ]


def test_a_charge_keeps_its_amount_when_the_line_beneath_only_describes_it():
    pages = [
        [
            ("Detail", {"kind": "section", "label": "Detail"}),
            (" Messaging   USD 25.00", {"kind": "service", "label": "Messaging", "amount": "25.00"}),
            ("  Outbound message fees   USD 25.00", item("Outbound message fees", "25.00")),
            # no amount on this line, only how much was used
            ("     Messages sent to one country   48,347 Messages",
             {"kind": "item", "label": "Messages sent to one country", "quantity": "48,347", "unit": "Messages"}),
        ]
    ]  # fmt: skip
    rows = walk(pages, [total("25.00")])
    assert live(rows) == Decimal("25.00")


def test_an_amount_the_model_left_out_is_read_when_the_line_prints_only_one_value():
    pages = [
        [
            ("Detail", {"kind": "section", "label": "Detail"}),
            # the same value twice: as a quantity of dollars and as the cost
            ("   Message fees   2,417.35 Dollar   USD 2,417.35", {"kind": "item", "label": "Message fees"}),
            # two different values: no way to tell which is the cost, so it is not guessed
            ("   Requests   USD 0.40   USD 12.00", {"kind": "item", "label": "Requests", "quantity": "3"}),
        ]
    ]  # fmt: skip
    rows = walk(pages, [])
    assert [r["billed_cost"] for r in rows if r["billed_cost"] is not None] == [Decimal("2417.35")]


def test_the_total_less_its_stated_tax_is_a_total_too():
    totals = [total("120.00"), total("20.00", "tax", "Total tax")]
    (derived,) = net_of_tax_totals(totals)
    assert (derived.amount, derived.scope, derived.origin) == ("100.00", "pre_tax", "derived")
    assert net_of_tax_totals([total("120.00")]) == []
    assert net_of_tax_totals([total("120.00"), total("0.00", "tax")]) == []
    # charges listed before tax reconcile against it, and the tax table is the restatement
    rows = [
        {"section": "Charges", "billed_cost": Decimal("60"), "charge_category": "usage"},
        {"section": "Charges", "billed_cost": Decimal("40"), "charge_category": "usage"},
        {"section": "Taxes by service", "billed_cost": Decimal("120"), "charge_category": None},
    ]
    mark_rollups(rows, totals, "USD")
    assert [r.get("is_rollup", False) for r in rows] == [False, False, True]


def test_totals_that_add_up_to_the_document_total_are_its_parts():
    totals = [
        total("120.00"),
        total("100.00", "pre_tax", "Seller A total"),
        total("20.00", "pre_tax", "Seller B total"),
    ]
    assert [t.amount for t in whole_document_totals(totals)] == ["120.00"]
    # a pre-tax total that stands alone still speaks for the document
    alone = [total("120.00"), total("100.00", "pre_tax")]
    assert [t.amount for t in whole_document_totals(alone)] == ["120.00", "100.00"]

    # so the smaller seller's block is charges of its own, not a restatement of anything
    rows = (
        [{"section": "Seller A", "billed_cost": Decimal("25"), "charge_category": "usage"}] * 4
        + [{"section": "Seller B", "billed_cost": Decimal("20"), "charge_category": "usage"}]
    )  # fmt: skip
    rows = [dict(r) for r in rows]
    mark_rollups(rows, totals, "USD")
    assert not any(r.get("is_rollup") for r in rows)


def test_the_sections_kept_are_the_ones_that_bring_the_sum_closest_to_the_total():
    def rows_of(section, *amounts):
        return [
            {"section": section, "billed_cost": Decimal(a), "charge_category": "usage"}
            for a in amounts
        ]

    rows = (
        rows_of("Detail", "30", "30", "30")  # most detailed: always kept
        + rows_of("One-off charges", "10")  # brings 90 up to the stated 100: kept
        + rows_of("Highlights", "55")  # a figure from the cover page: makes it worse
        + rows_of("Summary", "60", "40")  # alone equals the total: restates everything
    )
    mark_rollups(rows, [total("100.00")], "USD")
    flagged = {r["section"] for r in rows if r.get("is_rollup")}
    assert flagged == {"Highlights", "Summary"}
    assert live(rows) == Decimal("100")
