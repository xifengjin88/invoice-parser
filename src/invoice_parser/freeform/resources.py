"""Infrastructure definitions and inventories: one row per resource.

A bill is read line by line, because each charge is a line. A resource is not: it is a block
(a Terraform resource, a CloudFormation entry, a row of an inventory written as text) whose
sizing sits on the lines inside it. So the model is asked for resources, each with the line it
starts on and its pricing attributes, and code checks every attribute against that block.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field

from ..convert.types import TextDoc
from ..llm import LLM, Truncated
from ..schema import Granularity, RecordType
from ..values import parse_number
from .tagging import Attr, DocMeta, Line, _tokens, describe_meta

log = logging.getLogger("invoice_parser")

PART_LINES = 120
OVERLAP = 25
SEARCH = 10  # how far from the cited line to look for the resource's own line

# attribute keys that are canonical columns
ATTR_FIELDS = {
    "instance_type": "instance_type",
    "operating_system": "operating_system",
    "region": "region_name",
    "availability_zone": "availability_zone",
    "storage_type": "storage_type",
    "database_engine": "database_engine",
    "deployment_option": "deployment_option",
    "tenancy": "tenancy",
    "license_model": "license_model",
    "service": "service_name",
}


class Resource(BaseModel):
    line: int = Field(description="number of the line where the resource starts")
    label: str = Field(description="the resource as the file names it")
    resource_type: str | None = None
    resource_name: str | None = None
    attrs: list[Attr] = Field(default_factory=list)


class ResourceList(BaseModel):
    resources: list[Resource]


SYSTEM = """You read infrastructure definitions and inventories (Terraform, CloudFormation,
Kubernetes manifests, lists of machines, architecture descriptions) and list the resources that
would be billed by a cloud provider.

Return one entry per resource, not one per line. A resource is a whole block or record: all the
lines that configure it belong to its single entry.
- line: the number of the line where the resource starts.
- label: how the file names it, for example its type and name.
- resource_type: the kind of resource as written (aws_db_instance, AWS::EC2::Instance, ...).
- resource_name: its name or identifier as written.
- attrs: what its price depends on, as key/value pairs with values copied from the file. Use
  these keys when they apply: instance_type (the machine, node or instance size, whatever the
  file calls that setting), count (how many), region, availability_zone, operating_system,
  database_engine, deployment_option, storage_type, storage_gb, tenancy, license_model,
  service. Add other sizing settings under their own names. When a value refers to a variable
  or parameter defined elsewhere in the lines you can see, give the value it resolves to.

Leave out things that are not billed on their own: providers, variables, outputs, data lookups,
IAM roles and policies, security group rules, tags. Do not invent resources or values."""


def _parts(doc: TextDoc) -> list[list[Line]]:
    flat = [
        Line(p + 1, n + 1, text) for p, page in enumerate(doc.pages) for n, text in enumerate(page)
    ]
    step = PART_LINES - OVERLAP
    return [flat[i : i + PART_LINES] for i in range(0, max(len(flat) - OVERLAP, 1), step)]


def _ask(llm: LLM, part: list[Line], header: str) -> list[tuple[Line, Resource]]:
    user = header + "\n\nLines:\n" + "\n".join(f"{i + 1}: {ln.text}" for i, ln in enumerate(part))
    try:
        out = llm.json(
            system=SYSTEM, user=user, schema=ResourceList, tag="resources", max_tokens=8000
        )
    except Truncated:
        if len(part) < 20:
            raise
        mid = len(part) // 2
        return _ask(llm, part[:mid], header) + _ask(llm, part[mid:], header)
    found = []
    for res in out.resources:
        words = set(
            _tokens(f"{res.resource_type or ''} {res.resource_name or ''}") or _tokens(res.label)
        )
        cited = res.line - 1
        best, best_score = None, 0.0
        # The cited number is a hint; the line that actually names the resource decides.
        for i in range(max(0, cited - SEARCH), min(len(part), cited + SEARCH + 1)):
            have = set(_tokens(part[i].text))
            score = (len(words & have) / len(words) if words else 0.0) - 0.02 * abs(i - cited)
            if score > best_score:
                best, best_score = i, score
        if best is None and 0 <= cited < len(part):
            best = cited
        if best is not None:
            found.append((part[best], res))
    return found


def list_resources(
    llm: LLM, doc: TextDoc, meta: DocMeta, file_name: str, workers: int
) -> tuple[list[tuple[Line, Resource]], list[str]]:
    header = describe_meta(meta, file_name)
    failures: list[str] = []

    def one(part: list[Line]) -> list[tuple[Line, Resource]]:
        try:
            return _ask(llm, part, header)
        except Exception as exc:
            first = part[0]
            failures.append(
                f"page {first.page} line {first.number}: {type(exc).__name__}: {exc}"[:300]
            )
            return []

    parts = _parts(doc)
    log.info("%s: listing resources in %d part(s)", file_name, len(parts))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = [pair for part in pool.map(one, parts) for pair in part]
    # Parts overlap, so a resource near a boundary can be reported twice: keep the fuller one.
    by_line: dict[tuple[int, int], tuple[Line, Resource]] = {}
    for line, res in results:
        key = (line.page, line.number)
        if key not in by_line or len(res.attrs) > len(by_line[key][1].attrs):
            by_line[key] = (line, res)
    return [by_line[k] for k in sorted(by_line)], failures


def resource_rows(
    doc: TextDoc, base: dict[str, Any], found: list[tuple[Line, Resource]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per resource. An attribute is kept if its value is written in the resource's own
    block; if it is written only elsewhere in the file (a variable's default), it is kept and
    marked as derived; if it is nowhere in the file, it is dropped."""
    flat = [(p + 1, n + 1, t) for p, page in enumerate(doc.pages) for n, t in enumerate(page)]
    index = {(p, n): i for i, (p, n, _) in enumerate(flat)}
    whole = " ".join(" ".join(t.lower().split()) for _, _, t in flat)
    starts = [index[(line.page, line.number)] for line, _ in found]
    rows: list[dict[str, Any]] = []
    claimed = grounded = dropped = 0
    rejected: list[dict[str, Any]] = []
    for k, (line, res) in enumerate(found):
        end = starts[k + 1] if k + 1 < len(starts) else len(flat)
        block = " ".join(" ".join(t.lower().split()) for _, _, t in flat[starts[k] : end])
        row = dict(base)
        derived: list[str] = []
        row.update(
            source_page=line.page,
            source_row=line.number,
            source_text=line.text.strip(),
            record_type=RecordType.RESOURCE,
            granularity=Granularity.RESOURCE,
            charge_description=res.label.strip() or None,
            resource_type=res.resource_type,
            resource_name=res.resource_name,
            is_rollup=False,
        )
        extra: dict[str, str] = {}
        for attr in res.attrs:
            key, value = attr.k.strip().lower().replace(" ", "_"), attr.v.strip()
            if not key or not value:
                continue
            claimed += 1
            needle = " ".join(value.lower().split())
            if needle in block:
                indirect = False
            elif needle in whole:
                indirect = True
            else:
                dropped += 1
                if len(rejected) < 25:
                    rejected.append(
                        {
                            "page": line.page,
                            "line": line.number,
                            "field": key,
                            "claimed": value,
                            "text": line.text[:160],
                        }
                    )
                continue
            grounded += 1
            target = ATTR_FIELDS.get(key)
            if key in ("count", "quantity") and parse_number(value) is not None:
                target = "consumed_quantity"
                row[target] = parse_number(value)
            elif target and not row.get(target):
                row[target] = value
            else:
                target = None
                extra[key] = value
            if indirect and target:
                derived.append(target)
        if extra:
            row["attributes"] = json.dumps(extra, ensure_ascii=False)
        row["derived_fields"] = derived
        rows.append(row)
    stats = {
        "resources": len(rows),
        "values_claimed": claimed,
        "values_grounded": grounded,
        "values_rejected": rejected,
        "attrs_dropped": dropped,
    }
    return rows, stats
