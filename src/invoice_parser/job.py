"""One workflow step: parse what is at a local path, and leave results and a status behind.

    invoice-parser job /data/in/bill.pdf --out /data/out/job-1/0

Written for an orchestrator such as Argo Workflows. The step reads local files and writes local
files; the orchestrator moves them. It writes, into the output folder:

    <document id>/line_items.parquet, document.json     one folder per document found
    summary.json                                         what was parsed and how well
    status                                               pass, warn or fail (worst document)

Exit code 0 means the step did its work, whatever the validation status. Exit code 1 means at
least one document could not be processed at all (for example the model was unreachable), which
is worth a retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .sdk import Parser

log = logging.getLogger("invoice_parser")

ENV_CACHE = "INVOICE_PARSER_CACHE"


def open_cache(location: str | None) -> Any:
    """A folder path, or a redis:// URL so that pods on different nodes share saved answers."""
    if not location:
        return None
    if location.startswith(("redis://", "rediss://")):
        import redis  # only needed when a Redis cache is asked for

        return redis.Redis.from_url(location)
    return Path(location)


async def run_job(
    input_path: Path,
    out: Path,
    *,
    cache: str | None = None,
    workers: int = 8,
    offline: bool = False,
    use_model: bool = True,
    provider: str | None = None,
    model: str | None = None,
    status_file: Path | None = None,
) -> int:
    out.mkdir(parents=True, exist_ok=True)
    parser = Parser(
        cache=open_cache(cache or os.environ.get(ENV_CACHE)),
        workers=workers,
        offline=offline,
        use_model=use_model,
        provider=provider,
        model=model,
    )
    result = await parser.parse(input_path, output_dir=out)
    summary = result.summary()
    summary["input"] = str(input_path)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for target in filter(None, [out / "status", status_file]):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.status)
    for doc in result:
        log.info("%s: %s, %d line item(s)", doc.name, doc.status, doc.record.line_item_count)
    broken = [doc.name for doc in result if doc.error]
    if broken:
        log.error("could not process: %s", "; ".join(broken))
    return 1 if broken else 0


def run(input_path: Path, out: Path, **options: Any) -> int:
    return asyncio.run(run_job(input_path, out, **options))
