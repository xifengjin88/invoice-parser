"""Read the pricing attributes out of each charge's wording.

Tagging settles which lines are charges and what they cost. This second pass looks only at
the wording of each distinct charge ("$0.0416 per On Demand Linux t3.medium Instance Hour",
"Reserved VM Instance, Standard_B2ats_v2, UK South, 1 Year") and pulls out what a price is
keyed on. Every value must be a phrase copied from that wording; code checks it is there.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..llm import LLM, Truncated
from ..schema import ChargeCategory, PurchaseOption, RecordType
from ..values import ground_number

log = logging.getLogger("invoice_parser")

BATCH = 25


class ProductFamily(StrEnum):
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    DATA_TRANSFER = "data_transfer"
    NETWORKING = "networking"
    REQUESTS = "requests"
    MONITORING = "monitoring"
    SECURITY = "security"
    AI = "ai"
    SUPPORT = "support"
    LICENSE = "license"
    OTHER = "other"


class DiscountType(StrEnum):
    RESERVATION = "reservation"
    SAVINGS_PLAN = "savings_plan"
    COMMITTED_USE = "committed_use"
    ENTERPRISE_AGREEMENT = "enterprise_agreement"
    PRIVATE_RATE = "private_rate"
    FREE_TIER = "free_tier"
    BUNDLED = "bundled"
    CREDIT = "credit"
    PROMOTION = "promotion"
    OTHER = "other"


class ChargeFrequency(StrEnum):
    USAGE_BASED = "usage_based"
    RECURRING = "recurring"
    ONE_TIME = "one_time"


def _require_every_field(schema: dict[str, Any]) -> None:
    """Make the model write every field for every entry, null included.

    Left optional, a small model answers a few fields and silently skips the rest; which few
    changes from run to run. Having to write each one makes it consider each one.
    """
    schema["required"] = list(schema["properties"])


class Described(BaseModel):
    model_config = ConfigDict(json_schema_extra=_require_every_field)

    id: int
    product_family: ProductFamily | None = None
    charge_frequency: ChargeFrequency | None = None
    discount_type: DiscountType | None = None
    instance_type: str | None = None
    operating_system: str | None = None
    database_engine: str | None = None
    deployment_option: str | None = None
    tenancy: str | None = None
    license_model: str | None = None
    storage_type: str | None = None
    purchase_option: PurchaseOption | None = None
    purchase_option_wording: str | None = None
    commitment_term: str | None = None
    commitment_payment_option: str | None = None
    commitment_offering_class: str | None = None
    region: str | None = None
    pricing_unit: str | None = None
    unit_price: str | None = None


class DescribedList(BaseModel):
    rows: list[Described]


# Described field -> canonical column, for the values that must be copied from the wording.
COPIED = {
    "instance_type": "instance_type",
    "operating_system": "operating_system",
    "database_engine": "database_engine",
    "deployment_option": "deployment_option",
    "tenancy": "tenancy",
    "license_model": "license_model",
    "storage_type": "storage_type",
    "commitment_term": "commitment_term",
    "commitment_payment_option": "commitment_payment_option",
    "commitment_offering_class": "commitment_offering_class",
    "region": "region_name",
    "pricing_unit": "pricing_unit",
}
# Classifications the wording implies but does not spell out; recorded as derived.
JUDGED = ("product_family", "charge_frequency", "discount_type")

SYSTEM = """You read the wording of cloud charges and pull out what each one's price depends on.
Each entry gives a charge's service, group and description as printed on a bill, invoice or
estimate, from any cloud provider. Return one result per entry, with the same id.

Write every field for every entry. Copy these as short phrases exactly as they appear in the
entry, and write null when the entry does not state one. Never infer a value from general
knowledge.
- instance_type: the machine, node or instance size (t3.medium, db.r5.4xlarge,
  Standard_B2ats_v2, n2-standard-4).
- operating_system: Linux, Windows, RHEL, SUSE and so on.
- database_engine: MySQL, PostgreSQL, Aurora PostgreSQL, SQL Server, Redis and so on.
- deployment_option: Single-AZ, Multi-AZ, zone-redundant and so on.
- tenancy: shared, dedicated, host.
- license_model: license included, bring your own license and so on.
- storage_type: the volume type, storage class, tier or redundancy (gp3, Standard, Archive, LRS).
- purchase_option_wording: the words in the entry that say how the usage is bought (on
  demand, spot, reserved, covered by a savings plan, committed use, free allowance).
- commitment_term: the length of a reservation or commitment (1 Year, 3yr, 36 months).
- commitment_payment_option: No Upfront, Partial Upfront, All Upfront, Monthly.
- commitment_offering_class: Standard, Convertible and so on.
- region: the region or location named in the entry.
- pricing_unit: what one unit of the price is (Hour, GB-month, 10,000 requests, vCPU-Hour).
- unit_price: the rate per unit, digits only, when the entry prints one.

Always answer these four for every entry, using null only where noted:
- product_family: the kind of thing charged for.
- charge_frequency: usage_based for metered usage, recurring for a periodic fee or commitment
  instalment, one_time for a single purchase or upfront payment.
- purchase_option: the category that purchase_option_wording means; a reservation or reserved
  capacity is reserved. Null when the entry does not say how the usage is bought.
- discount_type: what kind of discount, credit or free allowance the entry is. Null when the
  entry is not a discount, credit or allowance."""


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row.get("service_name") or "",
        row.get("sku_meter") or "",
        row.get("region_name") or "",
        row.get("charge_description") or "",
    )


def _entry(i: int, key: tuple[str, str, str, str]) -> str:
    service, group, region, description = key
    parts = [f"id {i}"]
    if service:
        parts.append(f"service: {service}")
    if group and group != description:
        parts.append(f"group: {group}")
    if region:
        parts.append(f"region: {region}")
    parts.append(f"description: {description}")
    return " | ".join(parts)


def _ask(llm: LLM, batch: list[tuple[int, tuple[str, str, str, str]]]) -> list[Described]:
    user = "Entries:\n" + "\n".join(_entry(i, key) for i, key in batch)
    try:
        out = llm.json(
            system=SYSTEM, user=user, schema=DescribedList, tag="describe", max_tokens=6000
        )
    except Truncated:
        if len(batch) < 4:
            raise
        mid = len(batch) // 2
        return _ask(llm, batch[:mid]) + _ask(llm, batch[mid:])
    return out.rows


def _stated(value: str, text: str) -> bool:
    return " ".join(value.lower().split()) in text


def describe_rows(
    llm: LLM, rows: list[dict[str, Any]], record_type: str, workers: int
) -> dict[str, Any]:
    """Fill pricing attributes on charge rows from their own wording. Returns statistics."""
    stats: dict[str, Any] = {"wordings": 0, "values_filled": 0, "values_rejected": 0, "failed": 0}
    if record_type == RecordType.RESOURCE:
        return stats
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("is_rollup") and row.get("charge_description"):
            by_key.setdefault(_key(row), []).append(row)
    keys = list(by_key)
    stats["wordings"] = len(keys)
    if not keys:
        return stats
    batches = [list(enumerate(keys))[i : i + BATCH] for i in range(0, len(keys), BATCH)]
    log.info("reading pricing attributes from %d distinct charge wordings", len(keys))

    def one(batch: list[tuple[int, tuple[str, str, str, str]]]) -> list[Described]:
        try:
            return _ask(llm, batch)
        except Exception as exc:
            stats["failed"] += len(batch)
            log.info("attribute reading failed for %d wordings: %s", len(batch), str(exc)[:160])
            return []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [d for part in pool.map(one, batches) for d in part]

    for described in results:
        if not 0 <= described.id < len(keys):
            continue
        key = keys[described.id]
        text = " ".join(" ".join(key).lower().split())
        for row in by_key[key]:
            derived = row.setdefault("derived_fields", [])
            for source, target in COPIED.items():
                value = getattr(described, source)
                if not value or not value.strip() or row.get(target):
                    continue
                if not _stated(value, text):
                    stats["values_rejected"] += 1
                    continue
                row[target] = value.strip()
                stats["values_filled"] += 1
            if described.unit_price and row.get("contracted_unit_price") is None:
                price, _ = ground_number(described.unit_price, [key[3]])
                if price is not None and row.get("billed_cost") != price:
                    row["contracted_unit_price"] = abs(price)
                    stats["values_filled"] += 1
            wording = (described.purchase_option_wording or "").strip()
            if described.purchase_option and not row.get("purchase_option"):
                if wording and not _stated(wording, text):
                    stats["values_rejected"] += 1  # backed by words the entry does not contain
                else:
                    row["purchase_option"] = described.purchase_option.value
                    if not wording:
                        derived.append("purchase_option")  # a judgement with nothing quoted
                    stats["values_filled"] += 1
            for name in JUDGED:
                value = getattr(described, name)
                if value is None or row.get(name):
                    continue
                if name == "discount_type" and row.get("charge_category") not in (
                    ChargeCategory.DISCOUNT, ChargeCategory.CREDIT,
                ):  # fmt: skip
                    continue
                row[name] = value.value
                derived.append(name)
                stats["values_filled"] += 1
    return stats
