"""What a consumer reads: line items as small typed objects, grouped by what they are for.

The parser stores every line item as one row of a wide table (see `schema.py`). That table is
complete but it is the parser's vocabulary. This module is the reader's: an `Item` says which
service, where, what product, how much of it, what it cost and how it is paid for, with the
source line it came from. Nothing here is computed or guessed; every value is one the stored
row already holds, and a value the document did not state is `None`.

    for item in doc.usage:
        item.service.name, item.product.instance_type, item.quantity, item.cost
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, overload

import polars as pl

from .schema import COST_FIELDS, empty_frame

# What an item is. The three lists on a document are built from these.
USAGE_KINDS = ("usage", "resource")
COMMITMENT_KINDS = ("commitment",)
ADJUSTMENT_KINDS = ("discount", "credit", "tax", "fee", "adjustment")
KINDS = (*USAGE_KINDS, *COMMITMENT_KINDS, *ADJUSTMENT_KINDS)


@dataclass(frozen=True, slots=True)
class Money:
    amount: Decimal
    currency: str | None = None

    def __str__(self) -> str:
        # Two decimals as money is written, more only when the amount needs them (a unit
        # price of 0.0416 must not read as 0.04).
        places = max(2, -self.amount.as_tuple().exponent)
        return f"{self.amount:,.{places}f} {self.currency or ''}".strip()


@dataclass(frozen=True, slots=True)
class Quantity:
    amount: Decimal
    unit: str | None = None

    def __str__(self) -> str:
        return f"{self.amount:,f} {self.unit or ''}".strip()


@dataclass(frozen=True, slots=True)
class Period:
    start: datetime | None = None
    end: datetime | None = None


@dataclass(frozen=True, slots=True)
class Service:
    """`name` is how the document words it. `code` is the provider's own identifier and is
    present only when the document gives one."""

    name: str | None = None
    code: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class Region:
    """`name` as the document words it ("US West (Oregon)"); `code` only when stated
    ("us-west-2")."""

    name: str | None = None
    code: str | None = None
    zone: str | None = None


@dataclass(frozen=True, slots=True)
class Product:
    """What exactly is being charged for: the keys a price list is looked up by."""

    sku: str | None = None
    usage_type: str | None = None
    operation: str | None = None
    family: str | None = None
    meter: str | None = None
    instance_type: str | None = None
    operating_system: str | None = None
    database_engine: str | None = None
    deployment: str | None = None
    storage_type: str | None = None
    tenancy: str | None = None
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Pricing:
    """How the item is paid for today. `model` is one of on_demand, reserved, savings_plan,
    spot, committed_use, free_tier, other."""

    model: str | None = None
    term: str | None = None
    payment: str | None = None
    offering_class: str | None = None
    commitment_id: str | None = None
    commitment_type: str | None = None


@dataclass(frozen=True, slots=True)
class Account:
    id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class Resource:
    id: str | None = None
    name: str | None = None
    type: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Source:
    """Where in the input the item was read from."""

    file: str
    member: str | None = None
    page: int | None = None
    row: int | None = None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class Item:
    """One line item.

    kind         usage, resource, commitment, discount, credit, tax, fee or adjustment
    detail       how precisely it can be matched to a price list, best first:
                 sku, usage_type, attributes, description, service, none
    inferred     names of stored fields that were worked out from context (a file-wide
                 setting, a variable's default) and not written on the item's own line
    """

    id: str
    kind: str
    provider: str | None
    service: Service
    region: Region
    product: Product
    quantity: Quantity | None
    cost: Money | None
    list_cost: Money | None
    unit_price: Money | None
    pricing: Pricing
    period: Period
    account: Account
    resource: Resource
    description: str | None
    detail: str
    source: Source
    inferred: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A plain nested dict. Amounts are strings and dates ISO text, so it is JSON-ready."""
        return _plain(asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _trim(value: Decimal | None) -> Decimal | None:
    """Drop the trailing zeros the stored fixed-point column carries (207.3600000000)."""
    if value is None:
        return None
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value.normalize()


def _json_object(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return {"value": text}
    return value if isinstance(value, dict) else {"value": value}


def kind_expr() -> pl.Expr:
    """What each stored row is, as one word. The only place that decides it."""
    category = pl.col("charge_category")
    committed = (
        pl.col("purchase_option").is_in(["reserved", "savings_plan", "committed_use"])
        | pl.col("commitment_term").is_not_null()
        | pl.col("commitment_discount_type").is_not_null()
    ).fill_null(False)
    return (
        pl.when(pl.col("record_type") == "resource")
        .then(pl.lit("resource"))
        .when(category.is_in(["tax", "credit", "discount", "adjustment"]))
        .then(category)
        .when((category == "purchase") & committed)
        .then(pl.lit("commitment"))
        .when(category == "purchase")
        .then(pl.lit("fee"))
        .otherwise(pl.lit("usage"))
        .alias("kind")
    )


_COST = pl.coalesce([pl.col(f) for f in COST_FIELDS])

# The short table: what matching to a price list needs, under the names used on `Item`.
PRICING_VIEW: dict[str, pl.Expr] = {
    "kind": pl.col("kind"),
    "provider": pl.col("provider"),
    "service_name": pl.col("service_name"),
    "service_code": pl.col("service_code"),
    "region_name": pl.col("region_name"),
    "region_code": pl.col("region_id"),
    "sku": pl.coalesce(["sku_id", "sku_price_id"]),
    "usage_type": pl.col("usage_type"),
    "instance_type": pl.col("instance_type"),
    "operating_system": pl.col("operating_system"),
    "database_engine": pl.col("database_engine"),
    "deployment": pl.col("deployment_option"),
    "storage_type": pl.col("storage_type"),
    "description": pl.col("charge_description"),
    "quantity": pl.coalesce(["consumed_quantity", "pricing_quantity"]),
    "unit": pl.coalesce(["consumed_unit", "pricing_unit"]),
    "cost": _COST,
    "currency": pl.col("billing_currency"),
    "unit_price": pl.coalesce(["contracted_unit_price", "list_unit_price"]),
    "pricing_model": pl.col("purchase_option"),
    "commitment_term": pl.col("commitment_term"),
    "commitment_payment": pl.col("commitment_payment_option"),
    "period_start": pl.coalesce(["charge_period_start", "billing_period_start"]),
    "period_end": pl.coalesce(["charge_period_end", "billing_period_end"]),
    "account_id": pl.coalesce(["sub_account_id", "billing_account_id"]),
    "account_name": pl.coalesce(["sub_account_name", "billing_account_name"]),
    "resource_id": pl.col("resource_id"),
    "resource_type": pl.col("resource_type"),
    "extra": pl.col("attributes"),
    "detail": pl.col("match_level"),
    "source_file": pl.col("source_file"),
    "source_page": pl.col("source_page"),
    "source_row": pl.col("source_row"),
    "item_id": pl.col("line_item_id"),
}


def item_from_row(row: dict[str, Any]) -> Item:
    """Build an `Item` from one stored row (with its `kind` column)."""
    get = row.get
    currency = get("billing_currency")

    def money(value: Decimal | None) -> Money | None:
        return None if value is None else Money(_trim(value), currency)

    amount = next((get(f) for f in COST_FIELDS if get(f) is not None), None)
    count = get("consumed_quantity") if get("consumed_quantity") is not None else None
    unit = get("consumed_unit")
    if count is None and get("pricing_quantity") is not None:
        count, unit = get("pricing_quantity"), get("pricing_unit")
    price = get("contracted_unit_price")
    if price is None:
        price = get("list_unit_price")
    return Item(
        id=get("line_item_id") or "",
        kind=get("kind") or "usage",
        provider=get("provider"),
        service=Service(get("service_name"), get("service_code"), get("service_category")),
        region=Region(get("region_name"), get("region_id"), get("availability_zone")),
        product=Product(
            sku=get("sku_id") or get("sku_price_id"),
            usage_type=get("usage_type"),
            operation=get("operation"),
            family=get("product_family"),
            meter=get("sku_meter"),
            instance_type=get("instance_type"),
            operating_system=get("operating_system"),
            database_engine=get("database_engine"),
            deployment=get("deployment_option"),
            storage_type=get("storage_type"),
            tenancy=get("tenancy"),
            license=get("license_model"),
            extra=_json_object(get("attributes")),
        ),
        quantity=None if count is None else Quantity(_trim(count), unit),
        cost=money(amount),
        list_cost=money(get("list_cost")),
        unit_price=money(price),
        pricing=Pricing(
            model=get("purchase_option"),
            term=get("commitment_term"),
            payment=get("commitment_payment_option"),
            offering_class=get("commitment_offering_class"),
            commitment_id=get("commitment_discount_id"),
            commitment_type=get("commitment_discount_type"),
        ),
        period=Period(
            get("charge_period_start") or get("billing_period_start"),
            get("charge_period_end") or get("billing_period_end"),
        ),
        account=Account(
            get("sub_account_id") or get("billing_account_id"),
            get("sub_account_name") or get("billing_account_name"),
        ),
        resource=Resource(
            get("resource_id"),
            get("resource_name"),
            get("resource_type"),
            _json_object(get("tags")),
        ),
        description=get("charge_description"),
        detail=get("match_level") or "none",
        source=Source(
            get("source_file") or "",
            get("source_member"),
            get("source_page"),
            get("source_row"),
            get("source_text"),
        ),
        inferred=tuple(get("derived_fields") or ()),
    )


class Items:
    """A read-only list of line items backed by the stored rows.

    It behaves like a list (iterate it, take its length, index or slice it) but reads from
    disk as needed, so a million-row export costs nothing until it is used. Rows that only
    summarise other rows are never included.
    """

    def __init__(self, paths: list[Path], kinds: tuple[str, ...] = KINDS) -> None:
        self._paths = [p for p in paths if p.is_file()]
        self._kinds = kinds

    def _scan(self) -> pl.LazyFrame:
        if not self._paths:
            return empty_frame().lazy().with_columns(kind_expr()).head(0)
        frames = [pl.scan_parquet(p) for p in self._paths]
        frame = frames[0] if len(frames) == 1 else pl.concat(frames)
        return (
            frame.filter(~pl.col("is_rollup").fill_null(False))
            .with_columns(kind_expr())
            .filter(pl.col("kind").is_in(list(self._kinds)))
        )

    def __iter__(self) -> Iterator[Item]:
        for row in self._scan().collect().iter_rows(named=True):
            yield item_from_row(row)

    def __len__(self) -> int:
        return self._scan().select(pl.len()).collect().item()

    def __bool__(self) -> bool:
        return len(self) > 0

    @overload
    def __getitem__(self, index: int) -> Item: ...
    @overload
    def __getitem__(self, index: slice) -> list[Item]: ...
    def __getitem__(self, index: int | slice) -> Item | list[Item]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            rows = self._scan().slice(start, max(stop - start, 0)).collect()
            return [item_from_row(r) for r in rows.iter_rows(named=True)][::step]
        size = len(self)
        if not -size <= index < size:
            raise IndexError("item index out of range")
        row = self._scan().slice(index % size, 1).collect().row(0, named=True)
        return item_from_row(row)

    def of_kind(self, *kinds: str) -> Items:
        """The same items narrowed to some kinds, for example `doc.adjustments.of_kind("tax")`."""
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown kind(s) {sorted(unknown)}; choose from {', '.join(KINDS)}")
        return Items(self._paths, tuple(k for k in self._kinds if k in kinds))

    def total(self) -> dict[str | None, Decimal]:
        """What the items cost, added up per currency."""
        sums = (
            self._scan()
            .group_by("billing_currency")
            .agg(_COST.sum().alias("cost"), _COST.count().alias("priced"))
            .filter(pl.col("priced") > 0)  # items with no cost at all add up to nothing, not 0
            .collect()
        )
        return {c: _trim(v) for c, v, _ in sums.iter_rows()}

    def to_frame(self, view: str = "pricing") -> pl.DataFrame:
        """A table of the items. `view="pricing"` is the short one, with the columns a price
        lookup needs; `view="full"` is every stored column."""
        if view == "full":
            return self._scan().collect()
        if view != "pricing":
            raise ValueError("view must be 'pricing' or 'full'")
        return self._scan().select([e.alias(n) for n, e in PRICING_VIEW.items()]).collect()

    def to_dicts(self) -> list[dict[str, Any]]:
        """The items as plain nested dicts, ready for JSON."""
        return [item.to_dict() for item in self]

    def __repr__(self) -> str:
        return f"<Items {len(self):,} of kind {'/'.join(self._kinds)}>"
