"""Put stated values into the schema's vocabulary and rate each row's usefulness for matching.

Nothing here reads meaning out of free text or knows any provider's naming: that is the
model's job (column mapping for tables, the wording pass for documents), with code checking
each value against the source. This step only normalizes values that are already in place.
"""

from __future__ import annotations

import polars as pl

from ..schema import ChargeCategory, MatchLevel, PurchaseOption
from . import data

_S = pl.String


def _nz(name: str) -> pl.Expr:
    c = pl.col(name).cast(_S).str.strip_chars()
    return pl.when(c == "").then(None).otherwise(c)


def normalize_charge_category(raw: pl.Expr) -> pl.Expr:
    r = raw.str.to_lowercase().str.replace_all(r"[\s_-]", "")
    return (
        pl.when(r.is_null())
        .then(None)
        .when(
            r.str.contains(r"^(discountedusage|savingsplancoveredusage|usage|consumption|metered)")
        )
        .then(pl.lit(ChargeCategory.USAGE))
        .when(r.str.contains(r"tax|vat|gst|hst"))
        .then(pl.lit(ChargeCategory.TAX))
        .when(r.str.contains(r"credit"))
        .then(pl.lit(ChargeCategory.CREDIT))
        .when(r.str.contains(r"discount|negation|edp|privaterate|covered"))
        .then(pl.lit(ChargeCategory.DISCOUNT))
        .when(r.str.contains(r"refund|adjust|rounding|correction"))
        .then(pl.lit(ChargeCategory.ADJUSTMENT))
        .when(r.str.contains(r"fee|purchase|upfront|recurring|subscription|support|commitment"))
        .then(pl.lit(ChargeCategory.PURCHASE))
        .when(r.str.contains(r"usage|charge"))
        .then(pl.lit(ChargeCategory.USAGE))
        .otherwise(None)
    )


def normalize_purchase_option(raw: pl.Expr) -> pl.Expr:
    r = raw.str.to_lowercase().str.replace_all(r"[\s_-]", "")
    return (
        pl.when(r.is_null())
        .then(None)
        .when(r.str.contains(r"spot|preemptible|dynamic"))
        .then(pl.lit(PurchaseOption.SPOT))
        .when(r.str.contains(r"savingsplan"))
        .then(pl.lit(PurchaseOption.SAVINGS_PLAN))
        .when(r.str.contains(r"reserv|^ri$|upfront"))
        .then(pl.lit(PurchaseOption.RESERVED))
        .when(r.str.contains(r"committeduse|^cud|committed"))
        .then(pl.lit(PurchaseOption.COMMITTED_USE))
        .when(r.str.contains(r"ondemand|payasyougo|payg|standard"))
        .then(pl.lit(PurchaseOption.ON_DEMAND))
        .when(r.str.contains(r"free"))
        .then(pl.lit(PurchaseOption.FREE_TIER))
        .otherwise(pl.lit(PurchaseOption.OTHER))
    )


def normalize_term(raw: pl.Expr) -> pl.Expr:
    """'1 Year', '3yr', '36 months' -> '1yr', '3yr', '36mo'. Anything else is kept as written."""
    r = raw.str.to_lowercase()
    years = r.str.extract(r"(\d+)\s*-?\s*(?:y|yr|yrs|year|years)\b", 1)
    months = r.str.extract(r"(\d+)\s*-?\s*(?:m|mo|mos|month|months)\b", 1)
    days = r.str.extract(r"(\d+)\s*-?\s*(?:d|day|days)\b", 1)
    return (
        pl.when(years.is_not_null()).then(years + "yr")
        .when(months.is_not_null()).then(months + "mo")
        .when(days.is_not_null()).then(days + "d")
        .otherwise(raw)
    )  # fmt: skip


def enrich(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Expects a frame that already has every canonical column."""
    provider = _nz("provider").str.to_lowercase().replace(data.PROVIDER_ALIASES)
    raw_category = _nz("charge_category")
    raw_purchase = _nz("purchase_option")
    lf = lf.with_columns(
        provider=provider,
        charge_category=normalize_charge_category(raw_category),
        charge_subcategory=pl.coalesce(
            _nz("charge_subcategory"),
            pl.when(~raw_category.str.to_lowercase().is_in([c.value for c in ChargeCategory])).then(
                raw_category
            ),
        ),
        purchase_option=normalize_purchase_option(raw_purchase),
        commitment_term=normalize_term(_nz("commitment_term")),
        charge_frequency=_nz("charge_frequency").str.to_lowercase().str.replace_all(r"[\s-]+", "_"),
    )

    has_service = pl.col("service_code").is_not_null() | pl.col("service_name").is_not_null()
    has_region = pl.col("region_id").is_not_null() | pl.col("region_name").is_not_null()
    has_price_hint = (
        pl.col("contracted_unit_price").is_not_null()
        | pl.col("list_unit_price").is_not_null()
        | pl.col("consumed_unit").is_not_null()
        | pl.col("pricing_unit").is_not_null()
    )
    match_level = (
        pl.when(pl.col("sku_id").is_not_null() | pl.col("sku_price_id").is_not_null())
        .then(pl.lit(MatchLevel.SKU))
        .when(has_service & pl.col("usage_type").is_not_null())
        .then(pl.lit(MatchLevel.USAGE_TYPE))
        .when(has_service & has_region & pl.col("instance_type").is_not_null())
        .then(pl.lit(MatchLevel.ATTRIBUTES))
        .when(has_service & pl.col("charge_description").is_not_null() & has_price_hint)
        .then(pl.lit(MatchLevel.DESCRIPTION))
        .when(has_service)
        .then(pl.lit(MatchLevel.SERVICE))
        .otherwise(pl.lit(MatchLevel.NONE))
    )
    return lf.with_columns(
        derived_fields=pl.col("derived_fields").fill_null([]),
        match_level=match_level,
    )
