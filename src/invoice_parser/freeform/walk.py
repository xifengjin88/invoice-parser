"""Turn tagged lines into line items.

Three sources of evidence, in order of trust:
1. Geometry: a line indented under another belongs to it.
2. Arithmetic: a line whose amount equals the sum of the lines after it is their heading.
3. The model: what each heading is (a service, a region, an account ...) and, only where the
   first two say nothing, which lines group together.

Every amount and quantity is read back from the cited source line, so nothing the model wrote
reaches the output unverified.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Any

from ..convert.types import TextDoc
from ..schema import ChargeCategory, Granularity, RecordType, StatedTotal
from ..values import (
    MONEY_RE,
    currency_of,
    find_currency,
    ground_number,
    parse_number,
)
from .tagging import DocMeta, Kind, Line, Tag, TotalScope

RECORD_LINES = 8  # how far below its first line one charge's figures may sit
MAX_LOOKAHEAD = 400
_ACCOUNT_ID = re.compile(
    r"\b(\d{12}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.IGNORECASE
)

# attrs keys that are canonical columns
ATTR_FIELDS = {
    "instance_type": "instance_type",
    "operating_system": "operating_system",
    "os": "operating_system",
    "region": "region_name",
    "resource_type": "resource_type",
    "resource_name": "resource_name",
    "resource_id": "resource_id",
    "storage_type": "storage_type",
    "database_engine": "database_engine",
    "engine": "database_engine",
    "deployment_option": "deployment_option",
    "tenancy": "tenancy",
    "license_model": "license_model",
    "usage_type": "usage_type",
    "operation": "operation",
    "sku": "sku_id",
    "sku_id": "sku_id",
    "service": "service_name",
}

_SCOPES = {
    TotalScope.GRAND_TOTAL: "document",
    TotalScope.PRE_TAX: "pre_tax",
    TotalScope.TAX: "tax",
    TotalScope.CREDITS: "credits",
    TotalScope.SUBTOTAL: "section",
}


def tolerance(amount: Decimal) -> Decimal:
    return max(Decimal("0.02"), abs(amount) * Decimal("0.005"))


def close_enough(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= tolerance(b)


def _exact(a: Decimal, b: Decimal) -> bool:
    """Equal up to rounding of printed subtotals."""
    return abs(a - b) <= max(Decimal("0.02"), abs(b) * Decimal("0.0005"))


def _indent(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


@dataclass
class Node:
    line: Line  # the line the tag is anchored to
    where: Line  # the line its amount was found on
    tag: Tag
    kind: Kind
    label: str
    amount: Decimal | None
    indent: int
    children: list[Node] = field(default_factory=list)
    maybe_total: bool = False
    restated: bool = False  # the same charges appear again, in more detail, right after

    def leaves(self) -> int:
        return sum(c.leaves() for c in self.children) if self.children else 1

    def priced_below(self) -> bool:
        """Whether any line beneath this one carries an amount."""
        return any(c.amount is not None or c.priced_below() for c in self.children)

    def leaf_sum(self) -> Decimal:
        if not self.children:
            return self.amount or Decimal(0)
        return sum((c.leaf_sum() for c in self.children), Decimal(0))

    def subtotal(self) -> Decimal:
        return self.amount if self.amount is not None else self.leaf_sum()


@dataclass
class Section:
    label: str | None
    roots: list[Node] = field(default_factory=list)
    account: tuple[str | None, str | None] = (None, None)  # carried in from earlier sections
    seller: str | None = None


@dataclass
class WalkResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    stated_totals: list[StatedTotal] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def page_furniture(
    doc: TextDoc, edge: int = 3, min_pages: int = 3, min_share: float = 0.0
) -> set[str]:
    """Lines repeated word for word at the top or bottom of several pages: running headers,
    footers and repeated column headings. They are page layout, not content."""
    at_edge: dict[str, set[int]] = {}
    inside: dict[str, int] = {}
    for p, page in enumerate(doc.pages):
        for n, text in enumerate(page):
            key = " ".join(text.split())
            if n < edge or n >= len(page) - edge:
                at_edge.setdefault(key, set()).add(p)
            else:
                inside[key] = inside.get(key, 0) + 1
    return {
        key
        for key, pages in at_edge.items()
        if key
        and len(pages) >= max(min_pages, min_share * len(doc.pages))
        and len(pages) >= 2 * inside.get(key, 0)
    }


def unit_from(quantity: str | None) -> str | None:
    """The unit when the model wrote it inside the quantity ('185.246 Hrs')."""
    match = re.match(r"^\s*-?[\d.,]+\s*([^\W\d_].{0,30})$", quantity or "")
    return match.group(1).strip() if match else None


def propagate_sections(
    doc: TextDoc, pairs: list[tuple[Line, Tag]]
) -> tuple[list[tuple[Line, Tag]], int]:
    """A line the model called a section title is one wherever it recurs word for word.

    Section titles carry no amount, so a missed one is invisible to the money checks, yet it
    moves a whole block of charges into the wrong section. Only titles tagged that way at
    least twice, and never tagged as anything else, are extended to their other occurrences.
    """
    as_section: dict[str, Tag] = {}
    counts: dict[str, int] = {}
    other: set[str] = set()
    tagged = set()
    for line, tag in pairs:
        key = " ".join(line.text.split())
        tagged.add((line.page, line.number))
        if tag.kind == Kind.SECTION:
            as_section[key] = tag
            counts[key] = counts.get(key, 0) + 1
        else:
            other.add(key)
    titles = {k for k, n in counts.items() if n >= 2 and k not in other}
    if not titles:
        return pairs, 0
    added = []
    for p, page in enumerate(doc.pages):
        for n, text in enumerate(page):
            key = " ".join(text.split())
            if key in titles and (p + 1, n + 1) not in tagged and not MONEY_RE.search(text):
                added.append((Line(p + 1, n + 1, text), as_section[key].model_copy()))
    merged = sorted(pairs + added, key=lambda pt: (pt[0].page, pt[0].number))
    return merged, len(added)


def split_account(label: str, account_id: str | None) -> tuple[str | None, str | None]:
    found = _ACCOUNT_ID.search(account_id or "") or _ACCOUNT_ID.search(label)
    ident = found.group(1) if found else (account_id or None)
    name = (
        re.sub(r"[\s(:#-]*" + re.escape(ident) + r"[\s)]*", " ", label).strip() if ident else label
    )
    return ident, (name or None)


class Walker:
    def __init__(
        self,
        doc: TextDoc,
        meta: DocMeta,
        base: dict[str, Any],
        document_totals: list[StatedTotal] | None = None,
    ) -> None:
        self.document_totals = [t for t in document_totals or [] if is_document_total(t)]
        self.tagged: set[tuple[int, int]] = set()
        self.doc = doc
        self.meta = meta
        self.base = base
        self.full_text = "\n".join(t for page in doc.pages for t in page).lower()
        self.furniture = page_furniture(doc)
        # A section title may legitimately open several pages; only one on nearly every page
        # is a running header.
        self.running_headers = page_furniture(doc, min_share=0.8)
        self.totals_kept_as_charges = self.restatements = self.amounts_read_from_line = 0
        self.detached = 0
        self.furniture_skipped = self.kinds_harmonized = 0
        self.out = WalkResult()
        self.claimed = self.grounded = 0
        self.rejected: list[dict[str, Any]] = []
        self.attrs_dropped = 0
        self.nested_by_sum = self.nested_by_kind = self.leaf_headings = self.duplicates = 0
        self.subtotals: list[dict[str, Any]] = []
        self.kinds: dict[str, int] = {}

    # --- grounding ---------------------------------------------------------------------
    def _window(self, line: Line) -> list[Line]:
        """Where a tag's values may be read.

        Its own line when that line prints a money value. Otherwise the figures are laid out
        elsewhere: on the lines after it up to the next tagged line (a description, then its
        numbers further down, possibly over a page break), or on the line before it.
        """
        window = [line]
        if MONEY_RE.search(line.text):
            return window  # the line carries its own figures
        page, number = line.page, line.number
        while len(window) <= RECORD_LINES:
            number += 1
            if number > len(self.doc.pages[page - 1]):
                page, number = page + 1, 1
                if page > len(self.doc.pages) or not self.doc.pages[page - 1]:
                    break
            if (page, number) in self.tagged:
                break
            text = self.doc.pages[page - 1][number - 1]
            if " ".join(text.split()) not in self.furniture:
                window.append(Line(page, number, text))
        if line.number > 1:
            # Also when that line has a tag of its own: a wrapped tail then reads the same
            # amount as its first line and is recognized as a repeat of it.
            before = self.doc.pages[line.page - 1][line.number - 2]
            window.append(Line(line.page, line.number - 1, before))
        return window

    def ground(self, claimed: str | None, line: Line, what: str) -> tuple[Decimal | None, Line]:
        """The value as the source states it, and the line it was found on."""
        if claimed is None or not str(claimed).strip():
            return None, line
        self.claimed += 1
        window = self._window(line)
        value, idx = ground_number(str(claimed), [ln.text for ln in window])
        if value is None:
            if len(self.rejected) < 25:
                self.rejected.append(
                    {
                        "page": line.page,
                        "line": line.number,
                        "field": what,
                        "claimed": claimed,
                        "text": line.text[:160],
                    }  # fmt: skip
                )
            return None, line
        self.grounded += 1
        return value, window[idx]

    # --- structure ---------------------------------------------------------------------
    def build(self, pairs: list[tuple[Line, Tag]]) -> list[Section]:
        """Indentation tree per section. Same-indent grouping is settled afterwards."""
        sections = [Section(None)]
        stack: list[Node] = []
        account: tuple[str | None, str | None] = (None, None)
        seller: str | None = None
        last_scope: dict[Kind, str] = {}
        claimed_spots: dict[tuple[int, int, Decimal], Node] = {}
        # A title at the top of nearly every page is a running header only if it sometimes
        # repeats with no other title in between (the same block carrying on over the page).
        # One that always alternates with other titles heads a block that itself recurs,
        # such as a summary printed once per account.
        titles = [" ".join(ln.text.split()) for ln, tg in pairs if tg.kind == Kind.SECTION]
        carried_over = {a for a, b in zip(titles, titles[1:], strict=False) if a == b}
        opened: set[str] = set()
        for at, (line, tag) in enumerate(pairs):
            self.kinds[tag.kind.value] = self.kinds.get(tag.kind.value, 0) + 1
            label = tag.label.strip()
            text_key = " ".join(line.text.split())
            if tag.kind == Kind.SECTION:
                # A running header opens its section once; after that it is page layout.
                running = (
                    text_key in self.running_headers
                    and text_key in carried_over
                    and text_key in opened
                )
                opened.add(text_key)
                if running or not re.search(r"[^\W\d_]", label):
                    self.furniture_skipped += 1  # a running header or a bare page number
                    continue
                sections.append(Section(label, account=account, seller=seller))
                stack = []
                continue
            if text_key in self.furniture:
                self.furniture_skipped += 1
                continue
            maybe_total = False
            if tag.kind == Kind.TOTAL:
                if tag.total_scope not in (None, TotalScope.SUBTOTAL):
                    self._total(line, tag, sections[-1], account[0])
                    after = pairs[at + 1] if at + 1 < len(pairs) else None
                    below = (
                        after is not None
                        and after[0].page == line.page
                        and after[1].kind not in (Kind.SECTION, Kind.TOTAL)
                        and _indent(after[0].text) > _indent(line.text)
                    )
                    if (tag.total_scope == TotalScope.GRAND_TOTAL and below) or self._heads(
                        pairs, at
                    ):
                        # A total is the title of a block when it is printed above an indented
                        # list, or when the lines after it add up to it: charges per seller,
                        # a recap per account, the savings already taken off. As a block of
                        # its own it can be weighed against the others. A total that closes
                        # what came before it (tax at the foot of a service) is neither.
                        sections.append(Section(label, account=account, seller=seller))
                        stack = []
                    continue
                # "Subtotal" is a judgement the arithmetic can check: the line joins the tree
                # as a charge and is removed again only if it restates the lines around it.
                maybe_total = True
            amount, where = self.ground(tag.amount, line, "amount")
            if amount is None and not tag.amount:
                # The model named the line but left its amount out. If the line prints exactly
                # one money value there is no doubt which it is.
                money = {parse_number(m) for m in MONEY_RE.findall(line.text)} - {None}
                if len(money) == 1:  # one value, or the same one printed more than once
                    amount = money.pop()
                    self.amounts_read_from_line += amount is not None
            if maybe_total and amount is None:
                continue
            if tag.kind in (Kind.ACCOUNT, Kind.SELLER):
                key = (
                    label
                    if tag.kind == Kind.SELLER
                    else (split_account(label, tag.account_id)[0] or label)
                )
                repeated = last_scope.get(tag.kind) == key
                last_scope[tag.kind] = key
                if tag.kind == Kind.ACCOUNT:
                    account = split_account(label, tag.account_id)
                else:
                    seller = label
                if repeated and amount is None:
                    continue  # a page header restating the current account or seller
            if amount is not None:
                # One printed amount is one charge. A second tag reading the same amount off
                # the same line is a wrapped continuation of the description, or a repeat.
                spot = (where.page, where.number, amount)
                first = claimed_spots.get(spot)
                if first is not None:
                    if line.number == first.line.number + 1 and line.page == first.line.page:
                        first.label = f"{first.label} {label}".strip()
                    self.duplicates += 1
                    continue
            kind = Kind.ITEM if maybe_total else tag.kind
            if (
                kind == Kind.ITEM
                and amount is None
                and self.meta.record_type != RecordType.RESOURCE
                and not tag.attrs
                and not tag.quantity
            ):
                continue  # a charge line with nothing verifiable on it adds only confusion
            node = Node(
                line, where, tag, kind, label, amount, _indent(line.text), maybe_total=maybe_total
            )
            if amount is not None:
                claimed_spots[spot] = node
            # A line naming who is billed or who sells, with no amount of its own, is never part
            # of a charge's breakdown however far right it is printed (a centred page footer).
            names_party = amount is None and tag.kind in (Kind.ACCOUNT, Kind.SELLER)
            while stack and (
                stack[-1].indent >= node.indent or (names_party and stack[-1].kind == Kind.ITEM)
            ):
                stack.pop()
            (stack[-1].children if stack else sections[-1].roots).append(node)
            stack.append(node)
        return [s for s in sections if s.roots]

    def _heads(self, pairs: list[tuple[Line, Tag]], at: int, reach: int = 1500) -> bool:
        """Whether the total at `at` adds up the lines that follow it: the first line after
        it and the later ones of the same kind and indent, up to the next total or title."""
        whole = parse_number(pairs[at][1].amount or "")
        if not whole or at + 1 >= len(pairs):
            return False
        first_line, first = pairs[at + 1]
        if first.kind in (Kind.SECTION, Kind.TOTAL):
            return False
        indent, running, lines = _indent(first_line.text), Decimal(0), 0
        for line, tag in pairs[at + 1 : at + 1 + reach]:
            if tag.kind == Kind.SECTION or (
                tag.kind == Kind.TOTAL and tag.total_scope not in (None, TotalScope.SUBTOTAL)
            ):
                break
            if tag.kind != first.kind or _indent(line.text) != indent:
                continue
            running += abs(parse_number(tag.amount or "") or Decimal(0))
            lines += 1
            if _exact(running, abs(whole)):
                # One line repeating the amount is a wrapped line or a restatement, not a
                # list being added up.
                return lines > 1
            if running > abs(whole) + tolerance(whole):
                break
        return False

    def prune_totals(
        self, siblings: list[Node], parent: Node | None, section: Section
    ) -> list[Node]:
        """Drop lines tagged as subtotals that really do restate their surroundings: the amount
        of the heading above them, the line just before them, or everything before them."""
        kept: list[Node] = []
        for node in siblings:
            node.children = self.prune_totals(node.children, node, section)
            if node.maybe_total and not node.children and node.amount is not None:
                before = sum((k.subtotal() for k in kept), Decimal(0))
                known = any(
                    (t.source_page, t.source_row) == (node.where.page, node.where.number)
                    or (node.amount != 0 and Decimal(t.amount) == node.amount)
                    for t in self.document_totals
                )
                restates = known or (
                    (
                        parent is not None
                        and parent.amount is not None
                        and _exact(node.amount, parent.amount)
                    )
                    or (kept and node.amount != 0 and _exact(node.amount, kept[-1].subtotal()))
                    or (len(kept) > 1 and node.amount != 0 and _exact(node.amount, before))
                )
                if restates:
                    text = self.doc.pages[node.where.page - 1][node.where.number - 1]
                    self.out.stated_totals.append(
                        StatedTotal(
                            label=node.label[:120],
                            amount=str(node.amount),
                            currency=currency_of(text, node.amount) or self.meta.currency,
                            scope="section",
                            scope_key=section.account[0],
                            section=section.label,
                            source_page=node.where.page,
                            source_row=node.where.number,
                        )  # fmt: skip
                    )
                    continue
                self.totals_kept_as_charges += 1
            kept.append(node)
        return kept

    def nest(self, siblings: list[Node]) -> list[Node]:
        """Group lines that share an indent level.

        A line followed by lines adding up to its amount is their heading. Where a heading has
        no usable amount, the model's kinds decide: it owns what follows until the next heading
        of its own kind. Works from the last line backwards, so that by the time a line is
        considered, the lines after it have already been grouped.
        """
        out: list[Node] = []
        for node in reversed(siblings):
            node.children = self.nest(node.children)
            if (
                node.amount
                and node.amount > 0
                and node.children
                and all(c.subtotal() >= 0 for c in node.children)
                and node.leaf_sum() > node.amount * Decimal("1.5") + tolerance(node.amount)
            ):
                # Parts cannot add up to more than their whole. Lines indented under this one
                # that far exceed its amount are not its breakdown (a heading between them was
                # missed); they stand beside it.
                out = node.children + out
                node.children = []
                self.detached += 1
            complete = node.children and (
                node.amount is None or _exact(node.leaf_sum(), node.amount)
            )
            heading = node.kind != Kind.ITEM
            end = None
            if complete and node.amount:
                self._flag_restatement(node, out)
            if not complete and node.amount:
                running = node.leaf_sum() if node.children else Decimal(0)
                for j, follower in enumerate(out[:MAX_LOOKAHEAD]):
                    if follower.indent < node.indent:
                        break  # a line further left is never part of this one
                    running += follower.subtotal()
                    if _exact(running, node.amount):
                        # Small amounts add up by coincidence, so a matching sum alone is not
                        # enough: the followers must have breakdowns of their own, or the
                        # model must see a heading followed by a different kind of line.
                        structural = any(f.children for f in out[: j + 1])
                        if structural or (heading and out[0].kind != node.kind):
                            end = j + 1
                            self.nested_by_sum += 1
                        break
            if end is None and not complete and heading and not node.children:
                # No exact sum (a line was missed or misread). The model says this is a
                # heading, so it owns what follows until the next heading of its own kind,
                # and never more than its stated amount can pay for.
                budget = node.amount if node.amount and node.amount > 0 else None
                running, j = Decimal(0), 0
                # With an amount of its own, a heading followed directly by a plain charge
                # line is more likely a charge itself; it needs another heading under it.
                opens = bool(out) and (budget is None or out[0].kind != Kind.ITEM)
                while opens and j < len(out):
                    follower = out[j]
                    if follower.kind in (node.kind, Kind.ACCOUNT, Kind.SELLER):
                        break
                    if follower.indent < node.indent:
                        break
                    part = follower.subtotal()
                    if budget is not None and (
                        part < 0 or running + part > budget + tolerance(budget)
                    ):
                        break
                    running += part
                    j += 1
                if j:
                    end = j
                    self.nested_by_kind += 1
            if end:
                node.children = node.children + out[:end]
                out = out[end:]
            out.insert(0, node)
        return out

    def harmonize(self, roots: list[Node]) -> None:
        """Headings at the same depth of one section are the same kind of thing. Where enough
        of them agree on a kind, the rest take it too."""
        levels: dict[int, list[Node]] = {}

        def collect(nodes: list[Node], depth: int) -> None:
            for node in nodes:
                if node.children:
                    levels.setdefault(depth, []).append(node)
                    collect(node.children, depth + 1)

        collect(roots, 0)
        # Most lines in agreement first. A path names one service, one region, one account and
        # one seller, so once a depth has claimed one of those kinds no other depth may;
        # sub-groups can repeat at any depth.
        shares: list[tuple[int, int, Kind]] = []  # (lines agreeing, depth, kind)
        for depth, nodes in levels.items():
            if len(nodes) < 3:
                continue
            counts: dict[Kind, int] = {}
            for node in nodes:
                kind = Kind.GROUP if node.kind == Kind.ITEM else node.kind
                counts[kind] = counts.get(kind, 0) + 1
            shares += [
                (count, depth, kind) for kind, count in counts.items() if count >= 0.4 * len(nodes)
            ]
        decided: dict[int, Kind] = {}
        taken: set[Kind] = set()
        for _, depth, kind in sorted(shares, key=lambda x: -x[0]):
            if depth in decided or kind in taken:
                continue
            decided[depth] = kind
            if kind != Kind.GROUP:
                taken.add(kind)
        # Only lines the model left generic are renamed. One it named outright keeps its name:
        # if it sits at an odd depth, the structure around it is what is off, not the name.
        for depth, kind in decided.items():
            for node in levels[depth]:
                if node.kind in (Kind.ITEM, Kind.GROUP) and node.kind != kind:
                    node.kind = kind
                    self.kinds_harmonized += 1

    def _flag_restatement(self, node: Node, followers: list[Node]) -> None:
        """A heading already fully broken down, then followed by other lines that add up to
        the very same amount, is being shown twice: once in brief and once in detail. Keep the
        more detailed of the two."""
        running = Decimal(0)
        for j, follower in enumerate(followers[:MAX_LOOKAHEAD]):
            if follower.restated:
                continue
            running += follower.subtotal()
            if _exact(running, node.amount):
                again = [f for f in followers[: j + 1] if not f.restated]
                # Neighbours with equal amounts are common; a restatement is several lines
                # of a different kind adding up to this one.
                if len(again) >= 2 and again[0].kind != node.kind:
                    if node.leaves() <= sum(f.leaves() for f in again):
                        node.restated = True
                    else:
                        for f in again:
                            f.restated = True
                    self.restatements += 1
                return

    # --- rows ---------------------------------------------------------------------------
    def _context(self, path: list[Node], section: Section) -> dict[str, Any]:
        """Name each ancestor level from the model's kinds; a repeated kind is a sub-group."""
        ctx: dict[str, Any] = {
            "section": section.label,
            "invoice_issuer": section.seller or self.meta.invoice_issuer,
            "sub_account_id": section.account[0],
            "sub_account_name": section.account[1],
        }
        groups: list[str] = []
        seen: set[Kind] = set()
        for node in path:
            kind = node.kind if node.kind != Kind.ITEM and node.kind not in seen else Kind.GROUP
            seen.add(kind)
            if kind == Kind.SELLER:
                ctx["invoice_issuer"] = node.label
            elif kind == Kind.ACCOUNT:
                ctx["sub_account_id"], ctx["sub_account_name"] = split_account(
                    node.label, node.tag.account_id
                )
            elif kind == Kind.SERVICE:
                ctx["service_name"] = node.label
            elif kind == Kind.REGION:
                ctx["region_name"] = node.label
            else:
                groups.append(node.label)
        if groups:
            ctx["sku_meter"] = groups[-1]
            if len(groups) > 1:
                ctx["__groups"] = groups
        return ctx

    def _leaf_row(self, node: Node, path: list[Node], section: Section) -> dict[str, Any] | None:
        tag, where = node.tag, node.where
        heading = node.kind != Kind.ITEM
        # A childless heading is itself the charge: it names its own level.
        ctx = self._context([*path, node] if heading else path, section)
        groups = ctx.pop("__groups", None)
        text = self.doc.pages[where.page - 1][where.number - 1]
        quantity, _ = self.ground(tag.quantity, where, "quantity")
        unit_price, _ = self.ground(tag.unit_price, where, "unit_price")
        row = dict(self.base)
        row.update(ctx)
        row.update(
            source_page=where.page,
            source_row=where.number,
            source_text=text.strip(),
            charge_description=node.label or None,
            billed_cost=node.amount,
            billing_currency=(
                currency_of(text, node.amount)
                or tag.currency
                or find_currency(text)
                or self.meta.currency
            ),
            consumed_quantity=quantity,
            consumed_unit=(tag.unit or unit_from(tag.quantity)),
            contracted_unit_price=unit_price,
            charge_category=(tag.charge.value if tag.charge else None),
            is_rollup=False,
        )
        extra: dict[str, Any] = {"group_path": " > ".join(groups)} if groups else {}
        for attr in tag.attrs or []:
            key, value = attr.k.strip().lower().replace(" ", "_"), attr.v.strip()
            if not key or not value:
                continue
            if value.lower() not in self.full_text:
                self.attrs_dropped += 1
                continue
            target = ATTR_FIELDS.get(key)
            if target and not row.get(target):
                row[target] = value
            elif (
                key in ("count", "quantity")
                and row["consumed_quantity"] is None
                and parse_number(value) is not None
            ):
                row["consumed_quantity"] = parse_number(value)
            else:
                extra[key] = value
        if extra:
            row["attributes"] = json.dumps(extra, ensure_ascii=False)
        row["granularity"] = self._granularity(row, heading)
        usable = (
            node.amount is not None
            or self.meta.record_type == RecordType.RESOURCE
            or tag.attrs
            or quantity is not None
        )
        if heading:
            self.leaf_headings += 1
        return row if usable else None

    @staticmethod
    def _granularity(row: dict[str, Any], heading: bool) -> str:
        if row.get("resource_id") or row.get("resource_type") or row.get("resource_name"):
            return Granularity.RESOURCE
        priced = (
            row.get("consumed_quantity") is not None or row.get("contracted_unit_price") is not None
        )
        if priced or (row.get("sku_meter") and not heading):
            return Granularity.USAGE_TYPE
        if row.get("service_name"):
            return Granularity.SERVICE
        if row.get("sub_account_id"):
            return Granularity.ACCOUNT
        return Granularity.DOCUMENT

    def _emit(self, node: Node, path: list[Node], section: Section) -> None:
        if node.children and node.amount is not None and not node.priced_below():
            # Everything beneath describes this line without pricing it: the amount printed
            # here is the charge.
            node.children = []
        if not node.children:
            row = self._leaf_row(node, path, section)
            if row is not None:
                row["is_rollup"] = node.restated or any(p.restated for p in path)
                self.out.rows.append(row)
            return
        if node.amount is not None:
            computed = node.leaf_sum()
            self.subtotals.append(
                {
                    "ok": close_enough(computed, node.amount),
                    "kind": node.kind.value,
                    "label": node.label[:80],
                    "stated": str(node.amount),
                    "computed": str(computed),
                    "page": node.line.page,
                    "line": node.line.number,
                }  # fmt: skip
            )
        for child in node.children:
            self._emit(child, [*path, node], section)

    def _total(self, line: Line, tag: Tag, section: Section, account_id: str | None) -> None:
        amount, where = self.ground(tag.amount, line, "total")
        if amount is None:
            return
        scope = _SCOPES.get(tag.total_scope, "section")
        text = self.doc.pages[where.page - 1][where.number - 1]
        self.out.stated_totals.append(
            StatedTotal(
                label=tag.label.strip()[:120],
                amount=str(amount),
                currency=currency_of(text, amount)
                or tag.currency
                or find_currency(text)
                or self.meta.currency,
                scope=scope,
                scope_key=account_id if scope == "section" else None,
                section=section.label,
                source_page=where.page,
                source_row=where.number,
            )
        )

    def run(self, pairs: list[tuple[Line, Tag]]) -> WalkResult:
        pairs, sections_added = propagate_sections(self.doc, pairs)
        self.tagged = {(line.page, line.number) for line, _ in pairs}
        for section in self.build(pairs):
            section.roots = self.prune_totals(section.roots, None, section)
            section.roots = self.nest(section.roots)
            self.harmonize(section.roots)
            for root in section.roots:
                self._emit(root, [], section)
        self.out.rows.sort(key=lambda r: (r["source_page"], r["source_row"]))
        bad = [s for s in self.subtotals if not s["ok"]]
        self.out.stats = {
            "tags": self.kinds,
            "nested_by_sum": self.nested_by_sum,
            "nested_by_kind": self.nested_by_kind,
            "leaf_headings": self.leaf_headings,
            "page_furniture_skipped": self.furniture_skipped,
            "kinds_harmonized": self.kinds_harmonized,
            "totals_kept_as_charges": self.totals_kept_as_charges,
            "restatements_flagged": self.restatements,
            "breakdowns_detached": self.detached,
            "amounts_read_from_line": self.amounts_read_from_line,
            "section_titles_added": sections_added,
            "duplicate_tags_dropped": self.duplicates,
            "values_claimed": self.claimed,
            "values_grounded": self.grounded,
            "values_rejected": self.rejected,
            "attrs_dropped": self.attrs_dropped,
            "subtotals_checked": len(self.subtotals),
            "subtotals_ok": len(self.subtotals) - len(bad),
            "subtotal_mismatches": bad[:25],
        }
        return self.out


def ground_meta_totals(meta: DocMeta, doc: TextDoc) -> tuple[list[StatedTotal], int]:
    """Keep only the document totals whose amount really appears in the text."""
    out, dropped = [], 0
    for total in meta.totals:
        hit = None
        for p, page in enumerate(doc.pages):
            value, idx = ground_number(total.amount, page)
            if value is not None:
                hit = (value, p + 1, idx + 1, page[idx])
                break
        if hit is None:
            dropped += 1
            continue
        scope = {
            TotalScope.GRAND_TOTAL: "document", TotalScope.PRE_TAX: "pre_tax", TotalScope.TAX: "tax",
            TotalScope.CREDITS: "credits", TotalScope.SUBTOTAL: "section",
        }[total.scope]  # fmt: skip
        out.append(
            StatedTotal(
                label=total.label[:120],
                amount=str(hit[0]),
                currency=currency_of(hit[3], hit[0])
                or total.currency
                or find_currency(hit[3])
                or meta.currency,
                origin="summary",
                scope=scope,
                source_page=hit[1],
                source_row=hit[2],
            )  # fmt: skip
        )
    return out, dropped


def dedupe_totals(totals: list[StatedTotal]) -> list[StatedTotal]:
    seen, out = set(), []
    for t in totals:
        key = (t.scope, Decimal(t.amount), t.currency, t.scope_key)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def currency_twins(totals: list[StatedTotal], doc: TextDoc) -> list[StatedTotal]:
    """A document total restated in a second currency on the same line.

    Bills often total in the paying currency while itemizing in another. Where a line prints
    the stated total next to an amount in a different currency, that amount is the same total
    in that currency, which is what line items in that currency should add up to.
    """
    twins: list[StatedTotal] = []
    for total in totals:
        if not is_document_total(total) or not total.currency:
            continue
        amount = Decimal(total.amount)
        for p, page in enumerate(doc.pages):
            for n, text in enumerate(page):
                if currency_of(text, amount) != total.currency:
                    continue
                for printed in MONEY_RE.findall(text):  # money as printed, not "1 USD = ..."
                    other = parse_number(printed)
                    cur = currency_of(text, other)
                    if other and cur and cur != total.currency:
                        twins.append(
                            StatedTotal(
                                label=f"{total.label} (in {cur})"[:120],
                                amount=str(other),
                                currency=cur,
                                scope=total.scope,
                                origin="summary",
                                source_page=p + 1,
                                source_row=n + 1,
                            )  # fmt: skip
                        )
    return twins


_NOT_MERGED = {"line_item_id", "source_page", "source_row", "source_text", "section", "is_rollup"}


SAME_WORDING = 0.8  # share of the shorter wording's words that the longer one must contain


def _charge_words(row: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(row.get(k) or "")
        for k in ("service_name", "region_name", "sku_meter", "charge_description")
    )
    return set(re.sub(r"[^\w]+", " ", text.lower()).split())


def combine_restated(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fold a charge that the document prints twice into one row.

    Rollup sections restate the detail. Where a rollup row is the very same charge as a kept
    row (same amount and currency, and wording that agrees), it is not a second line item: the
    kept row absorbs whatever the copy knows that it does not, records where else it was
    printed, and the copy is dropped. Rollup rows with no twin are aggregates of several
    charges and stay flagged.
    """
    kept: dict[tuple, list[tuple[set[str], dict[str, Any]]]] = {}
    for row in rows:
        if not row.get("is_rollup") and row.get("billed_cost") is not None:
            key = (row["billed_cost"], row.get("billing_currency"))
            kept.setdefault(key, []).append((_charge_words(row), row))
    used: set[int] = set()
    out: list[dict[str, Any]] = []
    merged = 0
    for row in rows:
        target = None
        if row.get("is_rollup") and row.get("billed_cost") is not None:
            words = _charge_words(row)
            best = 0.0
            for other, candidate in kept.get((row["billed_cost"], row.get("billing_currency")), []):
                if id(candidate) in used or not words or not other:
                    continue
                # The same charge is often worded a little differently the second time (a
                # name given in two languages, a longer label), so one wording must contain
                # most of the other rather than equal it.
                share = len(words & other) / min(len(words), len(other))
                if share >= SAME_WORDING and share > best:
                    best, target = share, candidate
        if target is None:
            out.append(row)
            continue
        used.add(id(target))
        for name, value in row.items():
            if (
                name not in _NOT_MERGED
                and value not in (None, "", [])
                and target.get(name) in (None, "", [])
            ):
                target[name] = value
        place = f"p{row.get('source_page')} l{row.get('source_row')}"
        if row.get("section"):
            place += f" ({row['section']})"
        extra = json.loads(target["attributes"]) if target.get("attributes") else {}
        extra["also_printed_at"] = "; ".join(filter(None, [extra.get("also_printed_at"), place]))
        target["attributes"] = json.dumps(extra, ensure_ascii=False)
        merged += 1
    left = sum(1 for r in out if r.get("is_rollup"))
    return out, {"combined": merged, "rollups_left": left}


def line_currency(rows: list[dict[str, Any]], fallback: str | None) -> str | None:
    """The currency most line items are in: what the items are actually priced in, which is
    not always the currency the document is paid in."""
    counts: dict[str, int] = {}
    for row in rows:
        cur = row.get("billing_currency")
        if cur and row.get("billed_cost") is not None:
            counts[cur] = counts.get(cur, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else fallback


def is_document_total(total: StatedTotal) -> bool:
    """Totals for the whole document: tagged as such, or read from its opening or closing text
    (which is only ever asked for document-level totals)."""
    if total.scope in ("document", "pre_tax", "table"):
        return True
    return total.origin == "summary" and total.scope not in ("tax", "credits")


def net_of_tax_totals(totals: list[StatedTotal]) -> list[StatedTotal]:
    """The document total with each stated tax total taken off.

    Some bills list their charges before tax and the tax in a table of its own. Their line
    items then add up to a figure the bill never prints but plainly implies.
    """
    out: list[StatedTotal] = []
    seen: set[tuple[str, str | None]] = set()
    for whole in totals:
        if whole.scope != "document":
            continue
        for tax in totals:
            if tax.scope != "tax" or tax.currency != whole.currency:
                continue
            gross, levy = Decimal(whole.amount), Decimal(tax.amount)
            if not 0 < levy < gross:
                continue
            key = (str(gross - levy), whole.currency)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                StatedTotal(
                    label=f"{whole.label} less {tax.label}"[:120],
                    amount=str(gross - levy),
                    currency=whole.currency,
                    scope="pre_tax",
                    origin="derived",
                )
            )
    return out


def whole_document_totals(totals: list[StatedTotal]) -> list[StatedTotal]:
    """The totals that speak for the whole document.

    Totals that together add up to the document total are its parts (one per seller, say),
    not totals of the document, however they are labelled. The figure implied by taking the
    stated tax off the total is included.
    """
    docs = [t for t in totals if is_document_total(t)]
    derived = net_of_tax_totals(totals)
    wholes = [t for t in [*docs, *derived] if t.scope == "document" or t.origin == "derived"]
    parts = [t for t in docs if t.scope != "document"][:8]
    demoted: set[int] = set()
    for size in range(2, len(parts) + 1):
        for group in combinations(parts, size):
            together = sum((Decimal(t.amount) for t in group), Decimal(0))
            if any(
                w.currency == group[0].currency
                and abs(together - Decimal(w.amount))
                <= max(Decimal("0.05"), abs(Decimal(w.amount)) * Decimal("0.001"))
                for w in wholes
            ):
                demoted.update(id(t) for t in group)
    return [t for t in docs if id(t) not in demoted] + derived


MAX_SECTIONS_SEARCHED = 12


def mark_rollups(
    rows: list[dict[str, Any]], totals: list[StatedTotal], currency: str | None
) -> dict[str, Any]:
    """When several sections restate the same charges, keep the combination that agrees with
    the document total and flag the others as rollups."""
    sections: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("is_rollup"):  # already known to restate something inside its section
            sections.setdefault(row.get("section"), []).append(row)
    info: dict[str, Any] = {"sections": {str(k): len(v) for k, v in sections.items()}}
    if len(sections) < 2:
        return info
    targets = [
        Decimal(t.amount) for t in whole_document_totals(totals)
        if not t.currency or not currency or t.currency == currency
    ]  # fmt: skip
    if not targets:
        info["note"] = "several sections but no document total in the line currency; none flagged"
        return info

    def sums(names: Iterable[str | None]) -> tuple[Decimal, Decimal]:
        costs = [
            (r.get("billed_cost") or Decimal(0), r.get("charge_category"))
            for n in names
            for r in sections[n]
        ]
        incl = sum((c for c, _ in costs), Decimal(0))
        return incl, sum((c for c, cat in costs if cat != ChargeCategory.TAX), Decimal(0))

    def gap(names: Iterable[str | None]) -> Decimal:
        return min(abs(s - t) for t in targets for s in sums(names))

    def near(name: str | None) -> bool:
        return any(
            abs(s - t) <= max(Decimal("0.05"), abs(t) * Decimal("0.15"))
            for t in targets
            for s in sums([name])
        )

    restating = [name for name in sections if near(name)]
    if not restating:
        info["note"] = "no single section agrees with the document total; none flagged"
        return info
    # The most detailed section is always kept. A section that alone adds up to the document
    # total restates everything. Of the rest, the combination that brings the sum closest to
    # the total is kept: a section in it holds charges found nowhere else, one left out
    # repeats or summarises what is already counted.
    primary = max(sections, key=lambda name: len(sections[name]))
    rollups = [n for n in sections if n != primary and n in restating]
    free = [n for n in sections if n != primary and n not in restating]
    if primary in restating and free:
        if len(free) <= MAX_SECTIONS_SEARCHED:
            choices = (c for size in range(len(free) + 1) for c in combinations(free, size))
            keep = min(choices, key=lambda c: (gap([primary, *c]), len(c)))
        else:
            keep = tuple(n for n in free if gap([primary, n]) < gap([primary]))
        rollups += [n for n in free if n not in keep]
    for name in rollups:
        for row in sections[name]:
            row["is_rollup"] = True
    info["primary_section"] = primary
    info["rollup_sections"] = [str(n) for n in rollups]
    return info


def missed_lines(
    doc: TextDoc,
    pairs: list[tuple[Line, Tag]],
    rows: list[dict[str, Any]],
    totals: list[StatedTotal],
) -> dict[str, Any]:
    """Lines that look like they carry money but that nothing tagged."""
    covered = {(line.page, line.number) for line, _ in pairs}
    covered |= {(r["source_page"], r["source_row"]) for r in rows}
    covered |= {(t.source_page, t.source_row) for t in totals}
    money, missed = 0, []
    for p, page in enumerate(doc.pages):
        for n, text in enumerate(page):
            if MONEY_RE.search(text):
                money += 1
                if (p + 1, n + 1) not in covered:
                    missed.append({"page": p + 1, "line": n + 1, "text": text.strip()[:140]})
    return {"money_lines": money, "missed": len(missed), "examples": missed[:25]}
