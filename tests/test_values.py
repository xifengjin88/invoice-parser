from decimal import Decimal

import polars as pl

from invoice_parser.values import (
    datetime_expr,
    detect_datetime_format,
    detect_number_style,
    find_numbers,
    ground_number,
    number_expr,
    parse_number,
)


def test_parse_number_handles_currency_signs_and_notation():
    assert parse_number("$1,234.50") == Decimal("1234.50")
    assert parse_number("(12.00)") == Decimal("-12.00")
    assert parse_number("-USD 0.40") == Decimal("-0.40")
    assert parse_number("5.048e-05") == Decimal("0.00005048")
    assert parse_number("$1,00", "eu") == Decimal("1.00")
    assert parse_number("abc") is None and parse_number("") is None


def test_detect_number_style():
    assert detect_number_style(["$1,00", "$2,50"]) == "eu"
    assert detect_number_style(["1,234.50", "12.5"]) == "us"


def test_find_numbers_ignores_digits_inside_instance_names():
    line = "$0.0416 per On Demand Linux t3.medium Instance Hour  7,130.033 Hrs  USD 296.61"
    found = find_numbers(line)
    assert Decimal("296.61") in found and Decimal("7130.033") in found
    assert Decimal("3") not in found


def test_ground_number_takes_sign_from_source_and_rejects_absent_values():
    assert ground_number("122.06", ["covered by plan (USD 122.06)"]) == (Decimal("-122.06"), 0)
    assert ground_number("999.99", ["USD 122.06"]) == (None, None)


def test_number_expr_matches_scalar_parser():
    values = ["$1,234.50", "(12.00)", "-USD 0.40", "5.048e-05", "", None, "abc", "1e3"]
    out = pl.DataFrame({"a": values}).select(number_expr(pl.col("a"))).to_series().to_list()
    assert out[:4] == [Decimal("1234.5"), Decimal("-12"), Decimal("-0.4"), Decimal("0.00005048")]
    assert out[4:7] == [None, None, None] and out[7] == Decimal("1000")


def test_datetime_formats_and_intervals():
    for sample in [
        "2026-07-01T00:00:00+00:00",
        "2026/04/01 00:00:00",
        "2026-03",
        "Jul 2026",
        "2026-04-01T00:00:00Z",
    ]:
        fmt = detect_datetime_format([sample])
        assert fmt is not None, sample
        assert pl.select(datetime_expr(pl.lit(sample), fmt)).item() is not None
    interval = "2026-04-01T00:00:00Z/2026-04-01T01:00:00Z"
    end = pl.select(datetime_expr(pl.lit(interval), "%Y-%m-%dT%H:%M:%SZ", "end")).item()
    assert end.hour == 1
