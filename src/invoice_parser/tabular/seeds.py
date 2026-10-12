"""Checked-in plans for formats whose columns are a published standard. No model call needed."""

from __future__ import annotations

import re

from ..schema import Granularity, RecordType
from .plan import TablePlan

# canonical field -> candidate CUR columns, best first
_CUR: dict[str, list[str]] = {
    "billing_account_id": ["bill/PayerAccountId"],
    "sub_account_id": ["lineItem/UsageAccountId"],
    "sub_account_name": ["lineItem/UsageAccountName"],
    "invoice_id": ["bill/InvoiceId"],
    "invoice_issuer": ["bill/InvoicingEntity", "lineItem/LegalEntity"],
    "publisher": ["bill/BillingEntity"],
    "billing_period_start": ["bill/BillingPeriodStartDate"],
    "billing_period_end": ["bill/BillingPeriodEndDate"],
    "charge_period_start": ["lineItem/UsageStartDate", "identity/TimeInterval"],
    "charge_period_end": ["lineItem/UsageEndDate", "identity/TimeInterval"],
    "service_code": ["lineItem/ProductCode", "product/servicecode"],
    "service_name": ["product/ProductName", "product/servicename"],
    "product_family": ["product/productFamily"],
    "region_id": ["product/regionCode", "product/region"],
    "region_name": ["product/location"],
    "availability_zone": ["lineItem/AvailabilityZone"],
    "sku_id": ["product/sku"],
    "sku_price_id": ["pricing/RateCode", "pricing/RateId"],
    "usage_type": ["lineItem/UsageType"],
    "operation": ["lineItem/Operation"],
    "instance_type": ["product/instanceType"],
    "instance_family": ["product/instanceTypeFamily"],
    "operating_system": ["product/operatingSystem"],
    "tenancy": ["product/tenancy"],
    "license_model": ["product/licenseModel"],
    "database_engine": ["product/databaseEngine"],
    "deployment_option": ["product/deploymentOption"],
    "storage_type": ["product/volumeApiName", "product/storageClass", "product/volumeType"],
    "purchase_option": ["pricing/term"],
    "commitment_discount_id": ["reservation/ReservationARN", "savingsPlan/SavingsPlanARN"],
    "commitment_discount_type": ["savingsPlan/OfferingType", "reservation/SubscriptionId"],
    "commitment_term": ["pricing/LeaseContractLength", "savingsPlan/PurchaseTerm"],
    "commitment_payment_option": ["pricing/PurchaseOption", "savingsPlan/PaymentOption"],
    "commitment_offering_class": ["pricing/OfferingClass"],
    "resource_id": ["lineItem/ResourceId"],
    "charge_category": ["lineItem/LineItemType"],
    "charge_description": ["lineItem/LineItemDescription"],
    "consumed_quantity": ["lineItem/UsageAmount"],
    "consumed_unit": ["pricing/unit"],
    "list_unit_price": ["pricing/publicOnDemandRate"],
    "list_cost": ["pricing/publicOnDemandCost"],
    "contracted_unit_price": ["lineItem/UnblendedRate"],
    "billed_cost": ["lineItem/UnblendedCost"],
    "effective_cost": ["lineItem/NetUnblendedCost"],
    "billing_currency": ["lineItem/CurrencyCode"],
}

# FOCUS column -> canonical field
_FOCUS: dict[str, str] = {
    "ProviderName": "provider", "PublisherName": "publisher", "InvoiceIssuerName": "invoice_issuer",
    "BillingAccountId": "billing_account_id", "BillingAccountName": "billing_account_name",
    "SubAccountId": "sub_account_id", "SubAccountName": "sub_account_name", "InvoiceId": "invoice_id",
    "BillingPeriodStart": "billing_period_start", "BillingPeriodEnd": "billing_period_end",
    "ChargePeriodStart": "charge_period_start", "ChargePeriodEnd": "charge_period_end",
    "ServiceName": "service_name", "ServiceCategory": "service_category",
    "x_ServiceCode": "service_code", "RegionId": "region_id", "RegionName": "region_name",
    "AvailabilityZone": "availability_zone", "SkuId": "sku_id", "SkuPriceId": "sku_price_id",
    "SkuMeter": "sku_meter", "x_UsageType": "usage_type", "x_Operation": "operation",
    "PricingCategory": "purchase_option", "CommitmentDiscountId": "commitment_discount_id",
    "CommitmentDiscountType": "commitment_discount_type", "ResourceId": "resource_id",
    "ChargeFrequency": "charge_frequency",
    "ResourceName": "resource_name", "ResourceType": "resource_type", "Tags": "tags",
    "ChargeCategory": "charge_category", "ChargeClass": "charge_subcategory",
    "ChargeDescription": "charge_description", "ConsumedQuantity": "consumed_quantity",
    "ConsumedUnit": "consumed_unit", "PricingQuantity": "pricing_quantity",
    "PricingUnit": "pricing_unit", "ListUnitPrice": "list_unit_price",
    "ContractedUnitPrice": "contracted_unit_price", "ListCost": "list_cost",
    "ContractedCost": "contracted_cost", "BilledCost": "billed_cost",
    "EffectiveCost": "effective_cost", "BillingCurrency": "billing_currency",
}  # fmt: skip


def _snake(cur_name: str) -> str:
    """CUR 2.0 spelling of a legacy column: lineItem/UsageType -> line_item_usage_type."""
    return "_".join(
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", part).lower() for part in cur_name.split("/")
    )


def _cur_plan(header: list[str]) -> TablePlan | None:
    have = set(header)
    legacy = "lineItem/UnblendedCost" in have and "lineItem/UsageType" in have
    modern = "line_item_unblended_cost" in have and "line_item_usage_type" in have
    if not (legacy or modern):
        return None
    spell = (lambda c: c) if legacy else _snake
    field_map = {
        field: cols
        for field, candidates in _CUR.items()
        if (cols := [spell(c) for c in candidates if spell(c) in have])
    }
    intervals = {}
    interval_col = spell("identity/TimeInterval")
    for field, part in (("charge_period_start", "start"), ("charge_period_end", "end")):
        if field_map.get(field) == [interval_col]:
            intervals[field] = part
    resource_col = spell("lineItem/ResourceId")
    return TablePlan(
        summary="AWS Cost and Usage Report",
        record_type=RecordType.CHARGE,
        granularity=Granularity.RESOURCE if resource_col in have else Granularity.SKU,
        field_map=field_map,
        constants={"provider": "aws"},
        interval_fields=intervals,
        source="seed:aws_cur",
    )


def _focus_plan(header: list[str]) -> TablePlan | None:
    have = set(header)
    if not {"BilledCost", "ChargeCategory", "ServiceName", "ChargePeriodStart"} <= have:
        return None
    return TablePlan(
        summary="FOCUS cost and usage export",
        record_type=RecordType.CHARGE,
        granularity=Granularity.RESOURCE if "ResourceId" in have else Granularity.SKU,
        field_map={field: [col] for col, field in _FOCUS.items() if col in have},
        source="seed:focus",
    )


def seed_plan(header: list[str]) -> TablePlan | None:
    return _cur_plan(header) or _focus_plan(header)
