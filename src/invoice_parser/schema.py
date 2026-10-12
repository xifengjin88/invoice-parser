"""Canonical schema: one line-item shape for every input, plus the document record."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.2"

DECIMAL = pl.Decimal(38, 10)
DATETIME = pl.Datetime("us")


class RecordType(StrEnum):
    CHARGE = "charge"
    ESTIMATE = "estimate"
    RESOURCE = "resource"


class Granularity(StrEnum):
    RESOURCE = "resource"
    SKU = "sku"
    USAGE_TYPE = "usage_type"
    SERVICE = "service"
    ACCOUNT = "account"
    DOCUMENT = "document"


class ChargeCategory(StrEnum):
    USAGE = "usage"
    PURCHASE = "purchase"
    TAX = "tax"
    CREDIT = "credit"
    DISCOUNT = "discount"
    ADJUSTMENT = "adjustment"


class PurchaseOption(StrEnum):
    ON_DEMAND = "on_demand"
    RESERVED = "reserved"
    SAVINGS_PLAN = "savings_plan"
    SPOT = "spot"
    COMMITTED_USE = "committed_use"
    FREE_TIER = "free_tier"
    OTHER = "other"


class ExtractionMethod(StrEnum):
    KNOWN_FORMAT = "known_format"
    LLM_COLUMN_MAP = "llm_column_map"
    LLM_LINE_TAGS = "llm_line_tags"
    OCR_LLM_LINE_TAGS = "ocr_llm_line_tags"


class MatchLevel(StrEnum):
    SKU = "sku"
    USAGE_TYPE = "usage_type"
    ATTRIBUTES = "attributes"
    DESCRIPTION = "description"
    SERVICE = "service"
    NONE = "none"


class DocType(StrEnum):
    INVOICE = "invoice"
    BILL_DETAIL = "bill_detail"
    USAGE_EXPORT = "usage_export"
    COST_SUMMARY = "cost_summary"
    ESTIMATE = "estimate"
    INFRA_DEFINITION = "infra_definition"
    ANALYSIS = "analysis"
    NOT_COST_RELATED = "not_cost_related"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: Any
    group: str
    description: str = ""
    # Fields the model may target when mapping columns or tagging lines.
    mappable: bool = True


S, D, T = pl.String, DECIMAL, DATETIME

FIELDS: tuple[FieldSpec, ...] = (
    # Provenance
    FieldSpec("line_item_id", S, "provenance", mappable=False),
    FieldSpec("document_id", S, "provenance", mappable=False),
    FieldSpec("source_file", S, "provenance", mappable=False),
    FieldSpec("source_member", S, "provenance", mappable=False),
    FieldSpec("source_page", pl.Int64, "provenance", mappable=False),
    FieldSpec("source_row", pl.Int64, "provenance", mappable=False),
    FieldSpec("source_text", S, "provenance", mappable=False),
    FieldSpec("extraction_method", S, "provenance", mappable=False),
    FieldSpec("schema_version", S, "provenance", mappable=False),
    # Kind
    FieldSpec("record_type", S, "kind", mappable=False),
    FieldSpec("granularity", S, "kind", mappable=False),
    FieldSpec("section", S, "kind", mappable=False),
    FieldSpec("is_rollup", pl.Boolean, "kind", mappable=False),
    # Parties
    FieldSpec("provider", S, "parties", "cloud provider: aws, azure, gcp, oci, or a name"),
    FieldSpec("publisher", S, "parties", "who makes the product, e.g. a marketplace vendor"),
    FieldSpec("invoice_issuer", S, "parties", "legal entity issuing the bill / seller of record"),
    FieldSpec("billing_account_id", S, "parties", "payer / management / billing account id"),
    FieldSpec("billing_account_name", S, "parties"),
    FieldSpec("sub_account_id", S, "parties", "linked / usage account, subscription or project id"),
    FieldSpec("sub_account_name", S, "parties", "linked account, subscription or project name"),
    FieldSpec("invoice_id", S, "parties"),
    # Time
    FieldSpec("billing_period_start", T, "time", "start of the invoice / billing month"),
    FieldSpec("billing_period_end", T, "time"),
    FieldSpec("charge_period_start", T, "time", "start of the usage interval for this row"),
    FieldSpec("charge_period_end", T, "time"),
    # Catalog keys
    FieldSpec("service_name", S, "catalog", "service or product name as stated"),
    FieldSpec("service_code", S, "catalog", "provider service code, e.g. AmazonEC2"),
    FieldSpec("service_category", S, "catalog", "e.g. Compute, Storage, Databases"),
    FieldSpec("region_id", S, "catalog", "region code, e.g. us-west-2, eastus"),
    FieldSpec("region_name", S, "catalog", "region as stated, e.g. US West (Oregon)"),
    FieldSpec("availability_zone", S, "catalog"),
    FieldSpec("sku_id", S, "catalog", "provider SKU identifier"),
    FieldSpec("sku_price_id", S, "catalog", "rate code / price id / meter id"),
    FieldSpec("sku_meter", S, "catalog", "meter name or usage group"),
    FieldSpec("product_family", S, "catalog", "e.g. Compute Instance, Storage, Data Transfer"),
    FieldSpec("usage_type", S, "catalog", "provider usage type, e.g. USW2-BoxUsage:c5.large"),
    FieldSpec("operation", S, "catalog", "provider operation, e.g. RunInstances:0002"),
    # Pricing attributes
    FieldSpec("instance_type", S, "pricing", "e.g. t3.medium, Standard_D4s_v5, n2-standard-4"),
    FieldSpec("instance_family", S, "pricing"),
    FieldSpec("operating_system", S, "pricing"),
    FieldSpec("tenancy", S, "pricing"),
    FieldSpec("license_model", S, "pricing"),
    FieldSpec("database_engine", S, "pricing"),
    FieldSpec("deployment_option", S, "pricing", "e.g. Single-AZ, Multi-AZ"),
    FieldSpec("storage_type", S, "pricing", "volume type, storage class or tier"),
    FieldSpec("purchase_option", S, "pricing", "on demand, reserved, savings plan, spot ..."),
    FieldSpec("commitment_discount_id", S, "pricing", "reservation or savings plan id / ARN"),
    FieldSpec(
        "commitment_discount_type", S, "pricing", "kind of commitment: reservation, savings plan"
    ),
    FieldSpec("commitment_term", S, "pricing", "length of a reservation or commitment: 1yr, 3yr"),
    FieldSpec(
        "commitment_payment_option", S, "pricing", "no upfront, partial upfront, all upfront"
    ),
    FieldSpec("commitment_offering_class", S, "pricing", "standard, convertible"),
    FieldSpec("attributes", S, "pricing", mappable=False),
    # Resource
    FieldSpec("resource_id", S, "resource"),
    FieldSpec("resource_name", S, "resource"),
    FieldSpec("resource_type", S, "resource"),
    FieldSpec("tags", S, "resource", "resource tags or labels"),
    # Charge
    FieldSpec("charge_category", S, "charge", "kind of charge: usage, tax, credit, fee ..."),
    FieldSpec("charge_subcategory", S, "charge", "finer charge kind as stated"),
    FieldSpec("charge_frequency", S, "charge", "usage based, recurring or one time"),
    FieldSpec(
        "discount_type", S, "charge", "kind of discount: reservation, savings plan, enterprise ..."
    ),
    FieldSpec("charge_description", S, "charge", "line description / rate text"),
    # Amounts
    FieldSpec("consumed_quantity", D, "amounts", "usage amount"),
    FieldSpec("consumed_unit", S, "amounts", "unit of the usage amount, e.g. Hrs, GB-Mo"),
    FieldSpec("pricing_quantity", D, "amounts", "quantity in pricing units"),
    FieldSpec("pricing_unit", S, "amounts"),
    FieldSpec("list_unit_price", D, "amounts", "public / on-demand rate per unit"),
    FieldSpec("contracted_unit_price", D, "amounts", "negotiated or effective rate per unit"),
    FieldSpec("list_cost", D, "amounts", "cost at public rates"),
    FieldSpec("contracted_cost", D, "amounts", "cost at negotiated rates"),
    FieldSpec("billed_cost", D, "amounts", "amount charged for this row, before tax"),
    FieldSpec("effective_cost", D, "amounts", "amortized cost after discounts and commitments"),
    FieldSpec("tax_amount", D, "amounts", "tax on this row"),
    FieldSpec("billing_currency", S, "amounts", "ISO currency code"),
    # Trust
    FieldSpec("derived_fields", pl.List(pl.String), "trust", mappable=False),
    FieldSpec("match_level", S, "trust", mappable=False),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
COLUMNS: list[str] = [f.name for f in FIELDS]
POLARS_SCHEMA: dict[str, Any] = {f.name: f.dtype for f in FIELDS}
MAPPABLE_FIELDS: list[str] = [f.name for f in FIELDS if f.mappable]
DECIMAL_FIELDS: list[str] = [f.name for f in FIELDS if f.dtype == DECIMAL]
DATETIME_FIELDS: list[str] = [f.name for f in FIELDS if f.dtype == DATETIME]
COST_FIELDS: list[str] = ["billed_cost", "effective_cost", "contracted_cost", "list_cost"]

ENUMS: dict[str, type[StrEnum]] = {
    "record_type": RecordType,
    "granularity": Granularity,
    "charge_category": ChargeCategory,
    "purchase_option": PurchaseOption,
    "extraction_method": ExtractionMethod,
    "match_level": MatchLevel,
}


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=POLARS_SCHEMA)


def conform(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Add missing canonical columns as nulls, cast, and order columns."""
    have = set(df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns)
    exprs = [
        (
            pl.col(f.name).cast(f.dtype, strict=False) if f.name in have else pl.lit(None, f.dtype)
        ).alias(f.name)
        for f in FIELDS
    ]
    return df.select(exprs)


def field_guide(names: list[str] | None = None) -> str:
    """Compact field list for prompts."""
    rows = []
    for f in FIELDS:
        if not f.mappable or (names is not None and f.name not in names):
            continue
        rows.append(f"- {f.name}: {f.description}" if f.description else f"- {f.name}")
    return "\n".join(rows)


class StatedTotal(BaseModel):
    """A total the document itself states, used for reconciliation."""

    label: str
    amount: str
    currency: str | None = None
    scope: str = "document"  # document | pre_tax | tax | credits | account | service | section
    scope_key: str | None = None  # account id or service name when scope is narrower
    section: str | None = None
    source_page: int | None = None
    source_row: int | None = None
    origin: str = "line"  # line: a tagged line | summary: read from the document's opening/closing text | table


class CheckResult(BaseModel):
    name: str
    status: str  # pass | warn | fail | skip
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    examples: list[Any] = Field(default_factory=list)


class ValidationReport(BaseModel):
    status: str = "skip"
    checks: list[CheckResult] = Field(default_factory=list)
    match_levels: dict[str, int] = Field(default_factory=dict)
    fill_rates: dict[str, float] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    document_id: str
    source_file: str
    source_member: str | None = None
    file_type: str | None = None
    size_bytes: int | None = None
    converter: str | None = None
    doc_type: str | None = None
    providers: list[str] = Field(default_factory=list)
    language: str | None = None
    description: str | None = None
    granularity: str | None = None
    extraction_method: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stated_totals: list[StatedTotal] = Field(default_factory=list)
    line_item_count: int = 0
    tables: list[dict[str, Any]] = Field(default_factory=list)
    freeform: dict[str, Any] = Field(default_factory=dict)
    llm_usage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float | None = None
    validation: ValidationReport = Field(default_factory=ValidationReport)
