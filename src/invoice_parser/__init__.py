"""Cloud cost document parser: any file in, the same line items out.

from invoice_parser import parse

result = await parse("bill.pdf")
for doc in result:
    for item in doc.usage:
        print(item.service.name, item.product.instance_type, item.quantity, item.cost)
"""

from .items import (
    Account,
    Item,
    Items,
    Money,
    Period,
    Pricing,
    Product,
    Quantity,
    Region,
    Resource,
    Service,
    Source,
)
from .schema import SCHEMA_VERSION
from .sdk import FileInput, ParsedDocument, Parser, ParseResult, parse, parse_sync

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "Account",
    "FileInput",
    "Item",
    "Items",
    "Money",
    "ParseResult",
    "ParsedDocument",
    "Parser",
    "Period",
    "Pricing",
    "Product",
    "Quantity",
    "Region",
    "Resource",
    "Service",
    "Source",
    "__version__",
    "parse",
    "parse_sync",
]
