"""Names the same provider goes by, mapped to the schema's short form."""

from __future__ import annotations

PROVIDER_ALIASES: dict[str, str] = {
    "aws": "aws", "amazon web services": "aws", "amazon": "aws",
    "amazon web services, inc.": "aws", "amazon web services inc": "aws",
    "azure": "azure", "microsoft": "azure", "microsoft azure": "azure",
    "gcp": "gcp", "google": "gcp", "google cloud": "gcp", "google cloud platform": "gcp",
    "oci": "oci", "oracle": "oci", "oracle cloud": "oci", "oracle cloud infrastructure": "oci",
}  # fmt: skip
