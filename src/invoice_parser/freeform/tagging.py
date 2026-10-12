"""Model calls for free-form documents: document metadata, page transcription, line tagging."""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from ..convert.types import PageImage, TextDoc
from ..llm import LLM, Truncated
from ..schema import ChargeCategory, RecordType
from ..values import MONEY_RE, ground_number

log = logging.getLogger("invoice_parser")

CHUNK_LINES = 40
CONTEXT_LINES = 5
MIN_SPLIT_LINES = 8
# A part is re-tagged in halves when more than this share of its money lines went wrong.
RETRY_DEFECT_RATE = 0.10
MAX_RETRY_DEPTH = 2


class FreeDocType(StrEnum):
    INVOICE = "invoice"
    BILL_DETAIL = "bill_detail"
    COST_SUMMARY = "cost_summary"
    ESTIMATE = "estimate"
    INFRA_DEFINITION = "infra_definition"
    ANALYSIS = "analysis"
    NOT_COST_RELATED = "not_cost_related"


class TotalScope(StrEnum):
    GRAND_TOTAL = "grand_total"
    PRE_TAX = "pre_tax"
    TAX = "tax"
    CREDITS = "credits"
    SUBTOTAL = "subtotal"


class MetaTotal(BaseModel):
    label: str
    amount: str
    currency: str | None = None
    scope: TotalScope


class DocMeta(BaseModel):
    doc_type: FreeDocType
    has_extractable_lines: bool = Field(
        description="true if the document lists charges, costs, prices, or resources to be priced"
    )
    summary: str
    provider: str | None = None
    language: str | None = None
    record_type: RecordType
    currency: str | None = Field(None, description="ISO code of the currency line amounts are in")
    invoice_id: str | None = None
    invoice_date: str | None = None
    billing_period_start: str | None = None
    billing_period_end: str | None = None
    billing_account_id: str | None = None
    billing_account_name: str | None = None
    invoice_issuer: str | None = None
    totals: list[MetaTotal] = Field(default_factory=list)


META_SYSTEM = """You read the beginning and end of a document that may be a cloud bill, invoice,
cost report, price estimate, or infrastructure definition, and describe it.

- doc_type: invoice (a legal invoice or statement), bill_detail (an itemized bill with usage
  lines), cost_summary (totals by service or period), estimate (quote, forecast, pricing
  calculator output), infra_definition (Terraform, CloudFormation, VM or resource lists),
  analysis (a memo, deck or study that discusses costs), not_cost_related.
- provider: aws, azure, gcp, oci or another short name; null if unclear or mixed.
- record_type: charge for billed costs, estimate for quotes and forecasts, resource for
  infrastructure definitions and inventories.
- Dates as YYYY-MM-DD. Copy ids and amounts exactly as written. Use null when not stated.
- currency: the currency the itemized line amounts are in. If the document shows a local-currency
  total and itemizes in USD, answer USD.
- totals: every document-level total the text states (amount due, total for this invoice, total
  pre-tax, total tax, total credits), each with its own currency. Not per-service subtotals."""


class Kind(StrEnum):
    SECTION = "section"
    SELLER = "seller"
    ACCOUNT = "account"
    SERVICE = "service"
    REGION = "region"
    GROUP = "group"
    ITEM = "item"
    TOTAL = "total"


class Attr(BaseModel):
    k: str
    v: str


class Tag(BaseModel):
    line: int
    kind: Kind
    label: str
    account_id: str | None = None
    amount: str | None = None
    currency: str | None = None
    quantity: str | None = None
    unit: str | None = None
    unit_price: str | None = None
    charge: ChargeCategory | None = None
    total_scope: TotalScope | None = None
    attrs: list[Attr] | None = None


class TagList(BaseModel):
    rows: list[Tag]


TAG_SYSTEM = """You label lines of a cloud cost document. You get numbered lines from one part of
the document and return one entry per meaningful line. Code reads the values back out of the
lines you cite, so cite the right line and copy numbers exactly as written.

Return an entry for each line that is one of these kinds. The kind says what the line names,
not where it sits: code works out which lines belong under which from indentation and sums.
- section: a title that opens a major part of the document, such as "Summary", "Detail for
  Linked Account", "Charges by service", "Taxes by service", "Invoice Summary".
- seller: a line naming the legal entity or marketplace seller that charges are billed by.
- account: a line naming an account, subscription or project that charges belong to. Put the
  identifier in account_id and the name in label.
- service: a line naming a cloud service or product, for example a compute, storage or
  database service. It is a service whether or not a breakdown follows it.
- region: a line naming a region or location.
- group: a line naming any other category of charges: a usage type, meter, resource group or
  product family.
- item: a specific charge: a rate or usage description with its amount, a credit, discount,
  tax or fee line, or an estimate line. In an infrastructure definition or inventory, each
  resource to be priced is one item.
- total: a line stating a total for the whole document, an account or a section, such as "Total
  for this invoice", "Total pre-tax", "Account 123 total". Set total_scope.

Skip everything else: addresses, page headers and footers repeated on every page, legal text,
payment and bank details, payment history, explanatory notes, trend or "highest spend" tiles,
and exchange-rate notes. A row of column headings ("Description   Usage Quantity   Amount") is
never a heading of any kind: skip it.

Three or more spaces inside a line separate its columns.

Field rules:
- line: the number of the line carrying the entry's amount, or its first line if it has none.
- label: the name or description as written, copied from the line, without the quantity or
  amount columns. If the text wraps onto a second line, join it.
- When a line prints an amount, always give it in amount, whatever the kind.
- amount: the money on that line, as written. If a line shows one charge in two currencies, use
  the document's line currency given below and put that code in currency.
- quantity, unit, unit_price: only when printed on the line, as written, each in its own
  field and not repeated inside label. Columns can come in any order; decide which number is
  which from the column headings and the wording. Never calculate a value: if the line does
  not print a unit price, leave unit_price out.
- charge (items only): usage for metered usage and ordinary charges; tax; credit; discount
  (including usage covered by savings plans or reservations shown as a negative); purchase for
  fees, subscriptions, support, upfront or recurring commitment fees; adjustment for refunds
  and rounding.
- attrs (items only, optional): pricing attributes the text states outright, as key/value pairs.
  Use these keys when they apply: instance_type, operating_system, region, resource_type,
  resource_name, resource_id, storage_type, storage_gb, database_engine, deployment_option,
  tenancy, license_model, count, term. For infrastructure definitions give resource_type (for
  example aws_instance, AWS::RDS::DBInstance) and every sizing attribute stated.
- Keep the entries in line order."""


@dataclass
class Line:
    page: int  # 1-based
    number: int  # 1-based within the page
    text: str


@dataclass
class Chunk:
    lines: list[Line]
    before: list[Line]


def chunk_lines(doc: TextDoc, size: int = CHUNK_LINES) -> list[Chunk]:
    flat = [
        Line(p + 1, n + 1, text) for p, page in enumerate(doc.pages) for n, text in enumerate(page)
    ]
    chunks = []
    for start in range(0, len(flat), size):
        chunks.append(
            Chunk(flat[start : start + size], flat[max(0, start - CONTEXT_LINES) : start])
        )
    return chunks


def describe_meta(meta: DocMeta, file_name: str) -> str:
    bits = [f"File: {file_name}", f"Document: {meta.doc_type.value}, {meta.summary}"]
    if meta.provider:
        bits.append(f"Provider: {meta.provider}")
    if meta.currency:
        bits.append(f"Line currency: {meta.currency}")
    return "\n".join(bits)


def read_meta(llm: LLM, doc: TextDoc, file_name: str) -> DocMeta:
    flat = [text for page in doc.pages for text in page]
    head, tail = flat[:110], (flat[-45:] if len(flat) > 155 else flat[110:155])
    user = f"File: {file_name}\nPages: {len(doc.pages)}\n\nBeginning of document:\n" + "\n".join(
        head
    )
    if tail:
        user += "\n\nEnd of document:\n" + "\n".join(tail)
    return llm.json(system=META_SYSTEM, user=user, schema=DocMeta, tag="doc_meta", max_tokens=2500)


OCR_SYSTEM = """Transcribe the page image to plain text, exactly as printed. One printed line per
output line. Keep the reading order. For tables, put each table row on one line with columns
separated by " | ". Indent sub-items with spaces as they are indented on the page. Do not add
commentary, do not summarize, and do not correct or recompute any number."""


def transcribe(llm: LLM, images: list[PageImage], workers: int) -> dict[int, list[str]]:
    from ..convert.text import to_lines

    def one(image: PageImage) -> tuple[int, list[str]]:
        text = llm.text(
            system=OCR_SYSTEM,
            user="Transcribe this page.",
            tag="ocr",
            max_tokens=6000,
            images=[image.png],
        )
        return image.page, to_lines(text.replace("```", ""))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, images))


def count_defects(lines: list[Line], pairs: list[tuple[Line, Tag]]) -> tuple[int, int]:
    """(defects, money lines): money lines nothing tagged, plus tagged ones given no amount."""
    tagged = {(ln.page, ln.number): tag for ln, tag in pairs}
    money = defects = 0
    for line in lines:
        if not MONEY_RE.search(line.text):
            continue
        money += 1
        tag = tagged.get((line.page, line.number))
        if tag is None or (tag.kind != Kind.SECTION and not tag.amount):
            defects += 1
    return defects, money


def _tag_chunk(llm: LLM, chunk: Chunk, header: str, depth: int = 0) -> list[tuple[Line, Tag]]:
    user = header + "\n\n"
    if chunk.before:
        user += "Lines just before this part (context only, do not label):\n"
        user += "\n".join(f"  {ln.text}" for ln in chunk.before) + "\n\n"
    user += "Lines to label:\n" + "\n".join(
        f"{i + 1}: {ln.text}" for i, ln in enumerate(chunk.lines)
    )
    try:
        out = llm.json(
            system=TAG_SYSTEM, user=user, schema=TagList, tag="line_tags", max_tokens=8000
        )
    except Truncated:
        if len(chunk.lines) < MIN_SPLIT_LINES:
            raise
        mid = len(chunk.lines) // 2
        first = Chunk(chunk.lines[:mid], chunk.before)
        second = Chunk(chunk.lines[mid:], chunk.lines[max(0, mid - CONTEXT_LINES) : mid])
        return _tag_chunk(llm, first, header, depth) + _tag_chunk(llm, second, header, depth)
    pairs, moved = anchor_tags(chunk.lines, out.rows)
    if moved:
        log.debug("re-anchored %d of %d tags by their text", moved, len(out.rows))
    # Self-check: if the model skipped or half-read too many money lines, a smaller part
    # usually reads cleanly. Keep whichever attempt left fewer defects.
    defects, money = count_defects(chunk.lines, pairs)
    if (
        depth < MAX_RETRY_DEPTH
        and len(chunk.lines) >= 2 * MIN_SPLIT_LINES
        and defects > RETRY_DEFECT_RATE * max(money, 1)
    ):
        mid = len(chunk.lines) // 2
        first = Chunk(chunk.lines[:mid], chunk.before)
        second = Chunk(chunk.lines[mid:], chunk.lines[max(0, mid - CONTEXT_LINES) : mid])
        retried = _tag_chunk(llm, first, header, depth + 1) + _tag_chunk(
            llm, second, header, depth + 1
        )
        retry_defects, _ = count_defects(chunk.lines, retried)
        log.debug(
            "part from page %d: %d/%d money lines wrong, %d after re-tagging in halves",
            chunk.lines[0].page, defects, money, retry_defects,
        )  # fmt: skip
        if retry_defects < defects:
            return retried
    return pairs


_WORD = re.compile(r"[^\W_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def anchor_tags(lines: list[Line], tags: list[Tag]) -> tuple[list[tuple[Line, Tag]], int]:
    """Attach each tag to the line its label and amount actually appear on.

    Models miscount line numbers, more so deep into a chunk, so the cited number is only a
    hint: the label's words and the amount decide, with distance and order as tie-breakers.
    """
    line_tokens = [set(_tokens(ln.text)) for ln in lines]
    pairs: list[tuple[Line, Tag]] = []
    moved, cursor = 0, -1
    for tag in tags:
        cited = tag.line - 1
        label = _tokens(tag.label)[:10]
        best, best_score = None, 0.0
        for i, line in enumerate(lines):
            overlap = sum(t in line_tokens[i] for t in label) / len(label) if label else 0.0
            has_amount = bool(tag.amount) and ground_number(tag.amount, [line.text])[0] is not None
            if overlap < 0.6 and not (has_amount and overlap >= 0.34):
                continue
            if tag.amount and not has_amount and overlap < 0.9:
                continue
            score = 2 * overlap + has_amount - 0.03 * abs(i - cited) - (0.5 if i <= cursor else 0)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            if not 0 <= cited < len(lines):
                continue
            best = cited
        moved += best != cited
        cursor = max(cursor, best)
        pairs.append((lines[best], tag))
    return pairs, moved


def tag_document(
    llm: LLM, doc: TextDoc, meta: DocMeta, file_name: str, workers: int
) -> tuple[list[tuple[Line, Tag]], list[str]]:
    """Tag every chunk in parallel. Returns tags in document order plus per-chunk failures."""
    header = describe_meta(meta, file_name)
    chunks = chunk_lines(doc)
    failures: list[str] = []
    done = 0
    lock = threading.Lock()
    started = time.time()
    log.info("%s: tagging %d part(s), up to %d at a time", file_name, len(chunks), max(1, workers))

    def one(chunk: Chunk) -> list[tuple[Line, Tag]]:
        nonlocal done
        first = chunk.lines[0]
        try:
            pairs = _tag_chunk(llm, chunk, header)
            with lock:
                done += 1
                log.info(
                    "%s: part %d/%d done (from page %d, %d lines -> %d tags) at %.0fs",
                    file_name, done, len(chunks), first.page, len(chunk.lines), len(pairs),
                    time.time() - started,
                )  # fmt: skip
            return pairs
        except Exception as exc:
            with lock:
                done += 1
                log.info(
                    "%s: part %d/%d FAILED (from page %d): %s: %s",
                    file_name, done, len(chunks), first.page, type(exc).__name__, str(exc)[:160],
                )  # fmt: skip
            failures.append(
                f"page {first.page} line {first.number}: {type(exc).__name__}: {exc}"[:300]
            )
            return []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, chunks))
    pairs = [pair for chunk_pairs in results for pair in chunk_pairs]
    pairs.sort(key=lambda p: (p[0].page, p[0].number))
    return pairs, failures
