"""Number and date parsing, both for single strings and as Polars expressions."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

import polars as pl
from dateutil import parser as dateutil_parser

from .schema import DATETIME, DECIMAL

# A plain space is never a thousands separator: '#1 149.178 Hrs' is two numbers.
_NUM = r"\d{1,3}(?:[,  ]\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+,\d+|\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?|\.\d+"
_CUR = r"[$€£¥₹]|[A-Z]{3}(?=\s?[-\d(.])"
NUMBER_RE = re.compile(
    rf"(?<![\w.])(?P<neg>[-−–]\s?)?(?P<open>\()?\s?(?:(?P<cur>{_CUR})\s?)?(?P<neg2>[-−–]\s?)?"
    rf"(?P<num>{_NUM})(?!\d)(?P<close>\))?"
)
_EU_RE = re.compile(r"^\d{1,3}(?:\.\d{3})*,\d{1,2}$")
_US_GROUPED_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")

# A number that is written as money: a currency mark, or two decimals with thousands grouping.
MONEY_RE = re.compile(
    r"(?:[$€£¥₹]|\b[A-Z]{3})\s?-?\(?\d[\d,]*\.\d{2}\b|\b\d{1,3}(?:,\d{3})*\.\d{2}\b"
)

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}


def _digits_to_decimal(num: str, style: str) -> Decimal | None:
    num = re.sub(r"[\u00a0\u202f]", "", num)
    if style == "eu":
        num = num.replace(".", "").replace(",", ".")
    else:
        num = num.replace(",", "")
    try:
        return Decimal(num)
    except InvalidOperation:
        return None


def detect_number_style(samples: list[str]) -> str:
    """Return 'eu' when values use a decimal comma (1.234,56), else 'us'."""
    eu = us = 0
    for s in samples:
        m = NUMBER_RE.search(s or "")
        if not m:
            continue
        num = m.group("num")
        if _US_GROUPED_RE.match(num) or re.fullmatch(r"\d+\.\d+", num):
            us += 1
        elif _EU_RE.match(num):
            eu += 1
    return "eu" if eu > us else "us"


def parse_number(text: str | None, style: str = "us") -> Decimal | None:
    """Parse a single numeric cell such as '$1,234.50', '(12.00)', '-USD 0.40', '5e-05'."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    m = NUMBER_RE.search(text)
    if not m:
        return None
    value = _digits_to_decimal(m.group("num"), style)
    if value is None:
        return None
    negative = bool(m.group("neg") or m.group("neg2")) or bool(m.group("open") and m.group("close"))
    return -value if negative else value


def find_numbers(line: str) -> list[Decimal]:
    """Every number in a line of text, signed, under both decimal conventions when ambiguous."""
    out: list[Decimal] = []
    for m in NUMBER_RE.finditer(line or ""):
        num = m.group("num")
        negative = bool(m.group("neg") or m.group("neg2")) or bool(
            m.group("open") and m.group("close")
        )
        styles = ["us"]
        if "," in num and not _US_GROUPED_RE.match(num):
            styles = ["eu"]
        elif _EU_RE.match(num) or re.fullmatch(r"\d{1,3}(?:\.\d{3})+", num):
            styles = ["us", "eu"]
        for style in styles:
            value = _digits_to_decimal(num, style)
            if value is not None:
                out.append(-value if negative else value)
    return out


def ground_number(claimed: str | None, lines: list[str]) -> tuple[Decimal | None, int | None]:
    """Return (value, index of the line containing it) if any line states the claimed number.

    The sign is taken from the source text, not from the claim.
    """
    want = parse_number(claimed)
    if want is None and claimed:
        want = parse_number(claimed, "eu")
    if want is None:
        return None, None
    for i, line in enumerate(lines):
        for found in find_numbers(line):
            if abs(found) == abs(want):
                return found, i
    return None, None


_CODE = r"USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|SGD|BRL|MXN|CHF|SEK|NZD|ZAR|KRW"


def currency_of(text: str, amount: Decimal | None) -> str | None:
    """The currency written right beside an amount on a line.

    A line can mention several currencies ('1 USD = 94.73 INR   INR 12,464,104.54'); the one
    that belongs to a given number is the one touching it.
    """
    if amount is None:
        return None
    for m in NUMBER_RE.finditer(text or ""):
        value = _digits_to_decimal(m.group("num"), "us")
        if value is None or abs(value) != abs(amount):
            continue
        if m.group("cur"):
            cur = m.group("cur").strip()
            return CURRENCY_SYMBOLS.get(cur, cur)
        after = re.match(rf"\s?({_CODE})\b", text[m.end() :])
        if after:
            return after.group(1)
    return None


def find_currency(text: str) -> str | None:
    m = re.search(
        r"\b(USD|EUR|GBP|INR|CAD|AUD|JPY|CNY|SGD|BRL|MXN|CHF|SEK|NZD|ZAR|KRW)\b", text or ""
    )
    if m:
        return m.group(1)
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in (text or ""):
            return code
    return None


# --- Polars expressions -------------------------------------------------------------------


def number_expr(col: pl.Expr, style: str = "us") -> pl.Expr:
    """String column to Decimal: strips currency marks and grouping; parentheses mean negative."""
    s = col.cast(pl.String).str.strip_chars()
    negative = s.str.contains(r"^\(.*\)$") | s.str.contains(r"^[^\d(]*[-−–]")
    body = s.str.replace_all(r"[^\d.,eE+]", "").str.replace(r"^[eE+]+", "")
    if style == "eu":
        body = body.str.replace_all(r"\.", "").str.replace_all(",", ".")
    else:
        body = body.str.replace_all(",", "")
    # Scientific notation (5.048e-05) must keep its exponent sign, which the strip above removed.
    sci = s.str.contains(r"\d[eE][-+]?\d")
    plain = body.cast(DECIMAL, strict=False)
    via_float = (
        s.str.replace_all(r"[^\d.eE+\-]", "")
        .cast(pl.Float64, strict=False)
        .cast(DECIMAL, strict=False)
    )
    value = pl.when(sci).then(via_float).otherwise(pl.coalesce(plain, via_float))
    return pl.when(negative & ~sci).then(-value.abs()).otherwise(value)


_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%.fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%:z",
    "%Y-%m-%dT%H:%M:%S%.f%:z",
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y%m%d",
    "%Y-%m",
    "%Y/%m",
    "%b %Y",
    "%B %Y",
    "%b-%y",
]


def _try_format(samples: list[str], fmt: str) -> bool:
    try:
        s = pl.Series(samples, dtype=pl.String)
        if fmt in ("%Y-%m", "%Y/%m"):
            s = s + fmt[2] + "01"
            fmt = fmt + fmt[2] + "%d"
        elif fmt in ("%b %Y", "%B %Y", "%b-%y"):
            s = "01 " + s
            fmt = "%d " + fmt
        parsed = s.str.to_datetime(
            fmt, strict=False, time_zone=None if "z" not in fmt.lower() else "UTC"
        )
        return parsed.null_count() <= len(samples) // 10
    except Exception:
        return False


def detect_datetime_format(samples: list[str]) -> str | None:
    samples = [s.strip() for s in samples if s and s.strip()][:50]
    if not samples:
        return None
    for fmt in _DATE_FORMATS:
        if _try_format(samples, fmt):
            return fmt
    return None


def datetime_expr(col: pl.Expr, fmt: str | None, part: str | None = None) -> pl.Expr:
    """String column to naive UTC datetime. `part` picks a side of a 'start/end' interval."""
    s = col.cast(pl.String).str.strip_chars()
    if part in ("start", "end"):
        s = s.str.split("/").list.get(0 if part == "start" else 1, null_on_oob=True)
    if fmt is None:
        return (
            s.str.to_datetime(strict=False, time_zone="UTC")
            .dt.replace_time_zone(None)
            .cast(DATETIME)
        )
    if fmt in ("%Y-%m", "%Y/%m"):
        s = s + fmt[2] + "01"
        fmt = fmt + fmt[2] + "%d"
    elif fmt in ("%b %Y", "%B %Y", "%b-%y"):
        s = pl.lit("01 ") + s
        fmt = "%d " + fmt
    if "z" in fmt.lower():
        return (
            s.str.to_datetime(fmt, strict=False, time_zone="UTC")
            .dt.replace_time_zone(None)
            .cast(DATETIME)
        )
    return s.str.to_datetime(fmt, strict=False).cast(DATETIME)


_PLAIN_NUMBER = re.compile(r"[-+(]?[\d.,\s]+\)?$")
_YEAR_OR_MONTH = re.compile(r"(?:19|20)\d\d|[^\W\d_]{3}")


def looks_like_dates(samples: list[str]) -> bool:
    """True when most of the values are written as dates. A bare number is never one, however
    much it resembles a compact date."""
    probe = [s.strip() for s in samples[:60] if s and s.strip()]
    if not probe:
        return False
    dates = sum(
        1
        for s in probe
        if not _PLAIN_NUMBER.match(s)
        and _YEAR_OR_MONTH.search(s)
        and detect_datetime_format([s]) is not None
    )
    return dates >= 0.8 * len(probe)


def parse_datetime(text: str | None) -> datetime | None:
    """Lenient single-value date parsing for document metadata."""
    if not text or not str(text).strip():
        return None
    text = str(text).strip()
    fmt = detect_datetime_format([text])
    if fmt:
        try:
            value = pl.select(datetime_expr(pl.lit(text), fmt)).item()
            if value is not None:
                return value
        except Exception:
            pass
    try:
        value = dateutil_parser.parse(text, fuzzy=True, default=datetime(2000, 1, 1))
    except (ValueError, OverflowError):
        return None
    return value.replace(tzinfo=None) if value.tzinfo else value
