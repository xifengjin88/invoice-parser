"""The public interface: a file goes in, line items in the common schema come out.

    from invoice_parser import Parser

    parser = Parser()                              # settings from the environment
    result = await parser.parse("bill.pdf")        # a path
    result = await parser.parse(upload.stream, name="bill.pdf")   # an open file
    result = await parser.parse(data, name="export.csv")          # bytes

    for doc in result:
        print(doc.name, doc.status, doc.total_match)
        rows = doc.line_items()                    # a Polars DataFrame

Everything else in this package is internal and may change; this module is the contract.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import shutil
import tempfile
import threading
import weakref
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO

import polars as pl

from .convert import iter_sources
from .items import ADJUSTMENT_KINDS, COMMITMENT_KINDS, USAGE_KINDS, Items, Money, Period
from .llm import LLM, NotConfigured, create_llm
from .pipeline import process_source, summarize
from .schema import SCHEMA_VERSION, DocumentRecord, empty_frame
from .validate import LINE_ITEMS
from .values import parse_datetime

__all__ = ["FileInput", "ParseResult", "ParsedDocument", "Parser", "parse", "parse_sync"]

log = logging.getLogger("invoice_parser")

_ORDER = {"skip": 0, "pass": 1, "warn": 2, "fail": 3}
_CHUNK = 1 << 20


@dataclass
class FileInput:
    """A file given as content plus the name it should be known by.

    `data` is bytes or an open binary file (sync or async `read`). The name is used to
    recognize the file type and is what the document is called in the results.
    """

    data: bytes | bytearray | memoryview | BinaryIO | Any
    name: str = "upload"


# Anything `Parser.parse` accepts.
Source = str | os.PathLike | bytes | bytearray | memoryview | FileInput | BinaryIO | Any


class _Folder:
    """A temporary results folder, removed once nothing refers to it any more."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(path), True)

    def remove(self) -> None:
        self._finalizer()


_DOC_KINDS = {
    "invoice": "bill",
    "bill_detail": "bill",
    "usage_export": "usage_export",
    "cost_summary": "cost_summary",
    "estimate": "estimate",
    "infra_definition": "infrastructure",
    "unreadable": "unreadable",
}
_CONFIDENCE = {"pass": "high", "warn": "medium", "fail": "low"}


class ParsedDocument:
    """One document found in the input, with its line items and how far to trust them."""

    def __init__(self, record: DocumentRecord, parquet: Path) -> None:
        self.record = record
        self.path = parquet
        self._keeper: _Folder | None = None  # keeps a temporary folder alive while in use

    @property
    def id(self) -> str:
        return self.record.document_id

    @property
    def name(self) -> str:
        """The input's name, plus the path inside it for a member of an archive."""
        member = self.record.source_member
        return self.record.source_file + (f" › {member}" if member else "")

    @property
    def status(self) -> str:
        """pass, warn or fail: the worst of the validation checks."""
        return self.record.validation.status

    @property
    def ok(self) -> bool:
        """True unless validation failed or the document could not be parsed."""
        return self.status != "fail" and not self.record.error

    @property
    def doc_type(self) -> str | None:
        return self.record.doc_type

    @property
    def providers(self) -> list[str]:
        return self.record.providers

    @property
    def error(self) -> str | None:
        return self.record.error

    @property
    def total_match(self) -> float | None:
        """How closely the line items add up to the total the document states, 0 to 1.
        None when the document states no total."""
        for check in self.record.validation.checks:
            if check.name == "reconciliation":
                return check.metrics.get("match")
        return None

    @property
    def problems(self) -> list[str]:
        """The checks that warned or failed, as readable sentences."""
        return [
            f"{c.name}: {c.summary}"
            for c in self.record.validation.checks
            if c.status in ("warn", "fail")
        ]

    # --- the reader's view ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        """What sort of document this is: bill, usage_export, cost_summary, estimate,
        infrastructure, other (a memo, a slide deck, something with no costs), or unreadable
        (a damaged file, an unknown format, an archive past the unpacking limits)."""
        return _DOC_KINDS.get(self.record.doc_type or "", "other")

    @property
    def provider(self) -> str | None:
        """The cloud provider, when the document concerns exactly one."""
        found = self.record.providers
        return found[0] if len(found) == 1 else self.record.metadata.get("provider")

    @property
    def currency(self) -> str | None:
        """The currency the line items are in."""
        stated = self.record.metadata.get("currency")
        if stated or not self.path.is_file():
            return stated
        counts = self.scan().group_by("billing_currency").len().sort("len", descending=True)
        top = counts.filter(pl.col("billing_currency").is_not_null()).head(1).collect()
        return top["billing_currency"][0] if top.height else None

    @property
    def period(self) -> Period:
        """The time the document covers, as it states or as its line items span."""
        meta = self.record.metadata
        start = parse_datetime(meta.get("billing_period_start"))
        end = parse_datetime(meta.get("billing_period_end"))
        if (start is None or end is None) and self.path.is_file():
            first = pl.coalesce(["billing_period_start", "charge_period_start"]).min()
            last = pl.coalesce(["billing_period_end", "charge_period_end"]).max()
            lo, hi = self.scan().select(first.alias("lo"), last.alias("hi")).collect().row(0)
            start, end = start or lo, end or hi
        return Period(start, end)

    @property
    def total(self) -> Money | None:
        """The total the document states for itself, if it states one."""
        totals = [t for t in self.record.stated_totals if t.scope in ("document", "table")]
        currency = self.currency
        totals.sort(key=lambda t: (t.scope != "document", t.currency != currency))
        if not totals:
            return None
        return Money(Decimal(totals[0].amount), totals[0].currency or currency)

    @property
    def confidence(self) -> str:
        """How far to trust the line items: high, medium or low, from the validation checks.
        `unknown` when there was nothing to check (a document with no costs)."""
        if self.record.error or self.record.doc_type == "unreadable":
            return "low"
        if self.status == "pass" and not self.record.line_item_count:
            return "unknown"  # a memo or a slide deck: nothing was extracted, nothing checked
        return _CONFIDENCE.get(self.status, "unknown")

    @property
    def issues(self) -> list[str]:
        """Why confidence is not high, as readable sentences."""
        found = [self.record.error] if self.record.error else []
        if self.record.doc_type == "unreadable":
            found += self.record.warnings  # why the file could not be read
        return found + self.problems

    @property
    def items(self) -> Items:
        """Every line item, in document order."""
        return Items([self.path])

    @property
    def usage(self) -> Items:
        """What is being run: the items to price. For an infrastructure definition these are
        its resources."""
        return Items([self.path], USAGE_KINDS)

    @property
    def commitments(self) -> Items:
        """Reservations, savings plans and other commitments already bought."""
        return Items([self.path], COMMITMENT_KINDS)

    @property
    def adjustments(self) -> Items:
        """Discounts, credits, tax and fees: what changes the bill without being usage."""
        return Items([self.path], ADJUSTMENT_KINDS)

    def to_frame(self, view: str = "pricing") -> pl.DataFrame:
        """The line items as a table. See `Items.to_frame`."""
        return self.items.to_frame(view)

    def to_dicts(self) -> list[dict[str, Any]]:
        """The line items as plain nested dicts, ready for JSON."""
        return self.items.to_dicts()

    # --- the stored table -------------------------------------------------------------------

    def scan(self, include_rollups: bool = False) -> pl.LazyFrame:
        """The line items as a lazy frame, for results too large to hold in memory."""
        if not self.path.is_file():
            return empty_frame().lazy()
        frame = pl.scan_parquet(self.path)
        if include_rollups:
            return frame
        return frame.filter(~pl.col("is_rollup").fill_null(False))

    def line_items(self, include_rollups: bool = False) -> pl.DataFrame:
        """The line items. Rollup rows restate other rows and are left out unless asked for."""
        return self.scan(include_rollups).collect()

    def rows(self, include_rollups: bool = False) -> Iterator[dict[str, Any]]:
        """The line items as plain dicts, one at a time."""
        yield from self.line_items(include_rollups).iter_rows(named=True)

    def to_arrow(self, include_rollups: bool = False):
        return self.line_items(include_rollups).to_arrow()

    def summary(self) -> dict[str, Any]:
        """A small JSON-ready description: status, counts, match levels, problems."""
        return summarize(self.record)

    def __repr__(self) -> str:
        return (
            f"<ParsedDocument {self.name!r} {self.kind}, {self.record.line_item_count} rows, "
            f"confidence {self.confidence}>"
        )


class ParseResult:
    """Everything parsed from one call. Iterate it for the documents.

    When no output folder was given, the row files live in a temporary folder. It is removed
    by `close()`, on leaving a `with` block, or once the result and its documents are no longer
    referenced. Call `save()` to keep the files.
    """

    def __init__(
        self, documents: list[ParsedDocument], folder: Path, keeper: _Folder | None
    ) -> None:
        self.documents = documents
        self.folder = folder
        self._keeper = keeper

    def __iter__(self) -> Iterator[ParsedDocument]:
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> ParsedDocument:
        return self.documents[index]

    @property
    def status(self) -> str:
        """The worst status among the documents."""
        return max((d.status for d in self.documents), key=_ORDER.__getitem__, default="skip")

    @property
    def ok(self) -> bool:
        return bool(self.documents) and all(d.ok for d in self.documents)

    def _paths(self) -> list[Path]:
        return [d.path for d in self.documents]

    @property
    def items(self) -> Items:
        """Every line item of every document."""
        return Items(self._paths())

    @property
    def usage(self) -> Items:
        """What is being run, across all documents: the items to price."""
        return Items(self._paths(), USAGE_KINDS)

    @property
    def commitments(self) -> Items:
        """Commitments already bought, across all documents."""
        return Items(self._paths(), COMMITMENT_KINDS)

    @property
    def adjustments(self) -> Items:
        """Discounts, credits, tax and fees, across all documents."""
        return Items(self._paths(), ADJUSTMENT_KINDS)

    @property
    def confidence(self) -> str:
        """The lowest confidence among the documents."""
        order = ["low", "medium", "unknown", "high"]
        found = [d.confidence for d in self.documents]
        return min(found, key=order.index) if found else "unknown"

    def to_frame(self, view: str = "pricing") -> pl.DataFrame:
        """The line items of every document as one table. See `Items.to_frame`."""
        return self.items.to_frame(view)

    def to_dicts(self) -> list[dict[str, Any]]:
        """The line items of every document as plain nested dicts."""
        return self.items.to_dicts()

    def scan(self, include_rollups: bool = False) -> pl.LazyFrame:
        """Line items of every document together, lazily."""
        frames = [d.scan(include_rollups) for d in self.documents if d.path.is_file()]
        return pl.concat(frames) if frames else empty_frame().lazy()

    def line_items(self, include_rollups: bool = False) -> pl.DataFrame:
        """Line items of every document together. `document_id` says which is which."""
        return self.scan(include_rollups).collect()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "documents": [d.summary() for d in self.documents],
            "line_items": sum(d.record.line_item_count for d in self.documents),
        }

    def save(self, folder: str | os.PathLike) -> Path:
        """Copy the result files (one folder per document) somewhere permanent."""
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        for doc in self.documents:
            src = doc.path.parent
            if src.is_dir() and src.resolve() != (target / src.name).resolve():
                shutil.copytree(src, target / src.name, dirs_exist_ok=True)
        return target

    def close(self) -> None:
        """Remove the temporary folder now. Does nothing when an output folder was given."""
        if self._keeper is not None:
            self._keeper.remove()

    def __enter__(self) -> ParseResult:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ParseResult {len(self.documents)} document(s), {self.status}>"


def _safe_name(name: str | None) -> str:
    base = re.split(r"[\\/]", (name or "").strip())[-1]
    base = "".join(c if c.isalnum() or c in "._- ()[]" else "_" for c in base).strip(" .")
    return base[-150:] or "upload"


def _name_of(obj: Any) -> str | None:
    for attr in ("filename", "name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str) and value and not value.startswith("<"):
            return value
    return None


async def _write(data: Any, dest: Path) -> None:
    """Copy bytes or an open file (sync or async) to disk without holding it all in memory."""
    if isinstance(data, bytes | bytearray | memoryview):
        await asyncio.to_thread(dest.write_bytes, bytes(data))
        return
    reader = getattr(data, "stream", data)  # upload wrappers that keep the file on `.stream`
    read = getattr(reader, "read", None)
    if read is None:
        raise TypeError(f"cannot read a file from {type(data).__name__}")
    with dest.open("wb") as out:
        while True:
            chunk = read(_CHUNK)
            if inspect.isawaitable(chunk):
                chunk = await chunk
            if not chunk:
                break
            out.write(chunk.encode() if isinstance(chunk, str) else chunk)


class Parser:
    """Parses cloud cost files. Create one and reuse it; it holds the model connection.

    Settings left out are read from the environment (and a local `.env`).

    provider
        Which model service answers: "vllm" (a model you host behind an OpenAI-compatible
        endpoint; the default), "openai" or "claude". Default: INVOICE_PARSER_PROVIDER.
    model
        The model's name at that provider. Default: INVOICE_PARSER_MODEL, else the provider's
        own default (openai has none and needs one).
    api_key, base_url
        The provider's credential and endpoint. Pass them when they come from somewhere other
        than the environment, such as a secrets manager; a value passed here always wins.
        Left out, they are read from the environment: INVOICE_PARSER_LLM_KEY and
        INVOICE_PARSER_LLM_URL for vllm, OPENAI_API_KEY for openai, ANTHROPIC_API_KEY for
        claude. `base_url` is required for vllm and optional for the other two.
    cache
        Where model answers are saved so the same question is never paid for twice: a folder
        path, or any object with `get(key)` and `set(key, value)` such as a Redis client.
    workers
        Model calls made at once for a single document.
    offline
        Use saved answers only and never call the model.
    use_model
        False to parse only formats that need no model (AWS CUR and FOCUS exports).
    max_unpacked_gb, max_archive_members
        How much one input may unpack to, archives nested inside it included (default 10 GB
        and 10,000 files, or INVOICE_PARSER_MAX_UNPACKED_GB and
        INVOICE_PARSER_MAX_ARCHIVE_MEMBERS). An archive past either limit is reported as
        unreadable.
    fetch
        Optional: turns your own reference (an upload id, a storage key) into a file, so that
        `parse_ref` can be called with it. May be sync or async, and returns anything `parse`
        accepts.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: Any = None,
        workers: int = 8,
        offline: bool = False,
        use_model: bool = True,
        fetch: Callable[[Any], Source | Awaitable[Source]] | None = None,
        llm: LLM | None = None,
        max_unpacked_gb: float | None = None,
        max_archive_members: int | None = None,
        **provider_options: Any,
    ) -> None:
        self.workers = workers
        self._limits = {
            "max_bytes": None if max_unpacked_gb is None else int(max_unpacked_gb * 2**30),
            "max_members": max_archive_members,
        }
        self.fetch = fetch
        self._llm: LLM | None = llm
        if llm is None and use_model:
            try:
                self._llm = create_llm(
                    provider,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    cache=cache,
                    offline=offline,
                    **provider_options,
                )
            except NotConfigured as exc:
                # Nothing was set up at all: formats that need no model still parse. Anything
                # the caller asked for by name that cannot be had is an error, not a fallback,
                # and so is every other failure (an unknown provider, a missing credential).
                if provider or model or base_url or api_key:
                    raise
                log.warning(
                    "no model configured (%s); only formats that need no model will parse", exc
                )
        # One parse at a time per Parser: model usage is tracked per document, and a large
        # file needs the memory to itself. Parallelism happens inside a document.
        self._lock = threading.Lock()

    # --- async --------------------------------------------------------------------------

    async def parse(
        self,
        source: Source,
        *,
        name: str | None = None,
        output_dir: str | os.PathLike | None = None,
    ) -> ParseResult:
        """Parse one file. It may hold several documents (an archive, a workbook).

        source
            A path, bytes, an open binary file, or a `FileInput`. A folder path parses every
            file in it.
        name
            The file's name, when the source does not carry one. It helps recognize the type.
        output_dir
            Keep the result files here. Without it they go to a temporary folder owned by the
            result.
        """
        return await self.parse_many(
            [source if name is None else _named(source, name)], output_dir=output_dir
        )

    async def parse_many(
        self,
        sources: Iterable[Source],
        *,
        output_dir: str | os.PathLike | None = None,
    ) -> ParseResult:
        """Parse several files into one result."""
        temporary = output_dir is None
        out = (
            Path(tempfile.mkdtemp(prefix="invoice-parser-out-")) if temporary else Path(output_dir)
        )
        out.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="invoice-parser-in-"))
        try:
            paths: list[Path] = []
            for i, source in enumerate(sources):
                paths.append(await self._stage(source, staging / str(i)))
            documents = await asyncio.to_thread(self._run, paths, out, staging / "work")
        except BaseException:
            if temporary:
                shutil.rmtree(out, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        keeper = _Folder(out) if temporary else None
        for doc in documents:
            doc._keeper = keeper
        return ParseResult(documents, out, keeper)

    async def parse_ref(
        self, ref: Any, *, output_dir: str | os.PathLike | None = None
    ) -> ParseResult:
        """Parse a file identified by your own reference, using the `fetch` given at creation."""
        if self.fetch is None:
            raise RuntimeError("parse_ref needs a `fetch` function; pass one to Parser(...)")
        fetched = self.fetch(ref)
        if inspect.isawaitable(fetched):
            fetched = await fetched
        sources = fetched if isinstance(fetched, list | tuple) else [fetched]
        return await self.parse_many(sources, output_dir=output_dir)

    # --- sync twins ---------------------------------------------------------------------

    def parse_sync(
        self,
        source: Source,
        *,
        name: str | None = None,
        output_dir: str | os.PathLike | None = None,
    ) -> ParseResult:
        """`parse` for code that is not async. Do not call from inside a running event loop."""
        return asyncio.run(self.parse(source, name=name, output_dir=output_dir))

    def parse_many_sync(
        self, sources: Iterable[Source], *, output_dir: str | os.PathLike | None = None
    ) -> ParseResult:
        return asyncio.run(self.parse_many(sources, output_dir=output_dir))

    # --- internals ----------------------------------------------------------------------

    async def _stage(self, source: Source, folder: Path) -> Path:
        """Give every kind of input a path on disk. Paths are used where they are."""
        if isinstance(source, str | os.PathLike):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"no such file or folder: {path}")
            return path
        if isinstance(source, FileInput):
            data, name = source.data, source.name
        else:
            data, name = source, _name_of(source)
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / _safe_name(name)
        await _write(data, dest)
        return dest

    def _run(self, paths: list[Path], out: Path, work: Path) -> list[ParsedDocument]:
        work.mkdir(parents=True, exist_ok=True)
        documents: list[ParsedDocument] = []
        seen: set[str] = set()
        with self._lock:
            for number, path in enumerate(paths, start=1):
                for source in iter_sources(path, work, **self._limits):
                    if source.document_id in seen:
                        # Two inputs with the same file name: keep both, told apart by order.
                        source.display_path = f"{number}-{source.display_path}"
                    seen.add(source.document_id)
                    record = process_source(source, out, self._llm, self.workers)
                    if source.member and work in source.local_path.parents:
                        source.local_path.unlink(missing_ok=True)
                    documents.append(ParsedDocument(record, out / record.document_id / LINE_ITEMS))
        return documents


def _named(source: Source, name: str) -> Source:
    if isinstance(source, str | os.PathLike):
        return source  # a path already has its name
    if isinstance(source, FileInput):
        return FileInput(source.data, name)
    return FileInput(source, name)


async def parse(source: Source, *, name: str | None = None, **settings: Any) -> ParseResult:
    """Parse one file with settings from the environment. For repeated use, keep a `Parser`."""
    return await Parser(**settings).parse(source, name=name)


def parse_sync(source: Source, *, name: str | None = None, **settings: Any) -> ParseResult:
    return Parser(**settings).parse_sync(source, name=name)
