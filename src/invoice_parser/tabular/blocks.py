"""Split a raw grid into table blocks and describe them compactly for the model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..convert.tables import dedupe_headers
from ..convert.types import Grid, LazyTable

MAX_CONTEXT_LINES = 14
DISTINCT_SCAN_ROWS = 50_000
MAX_DISTINCT_SHOWN = 12
_NUMERIC = re.compile(r"^[\s$€£¥₹(+-]*\d[\d,.\s]*%?\)?$")


@dataclass
class Block:
    """A candidate table: rows[0] is the first row of the block, not necessarily the header."""

    rows: list[list[str]]
    first_row_number: int  # original 1-based row number of rows[0]
    name: str | None = None
    context: str = ""
    lazy: LazyTable | None = None
    tail: list[list[str]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return max((len(r) for r in self.rows), default=0)


def _trim(row: list[str]) -> list[str]:
    end = len(row)
    while end and not row[end - 1]:
        end -= 1
    return row[:end]


def _filled(row: list[str]) -> int:
    return sum(1 for c in row if c)


def is_numeric(cell: str) -> bool:
    return bool(_NUMERIC.match(cell or ""))


def split_blocks(grid: Grid) -> list[Block]:
    """Blank rows separate segments; segments shaped like a table become blocks, the rest context."""
    rows = [_trim(r) for r in grid.rows]
    segments: list[tuple[int, list[list[str]]]] = []
    start = None
    for i, row in enumerate(rows + [[]]):
        if row and start is None:
            start = i
        elif not row and start is not None:
            segments.append((start, rows[start:i]))
            start = None

    blocks: list[Block] = []
    context: list[str] = [grid.context] if grid.context else []
    prev_end = None
    for seg_start, seg in segments:
        # Title lines sitting directly on top of a table belong to its context, not its rows.
        while len(seg) > 2 and _filled(seg[0]) <= 1 and _filled(seg[1]) >= 2:
            context.append(" ".join(c for c in seg[0] if c))
            seg, seg_start = seg[1:], seg_start + 1
        last = blocks[-1] if blocks else None
        # A short run after one blank row that still looks like data (totals, a second group)
        # stays with the table above it.
        continues = (
            last is not None
            and prev_end is not None
            and seg_start - prev_end == 1
            and _filled(seg[0]) >= 2
            and max(len(r) for r in seg) <= last.width
            and any(is_numeric(c) for c in seg[0])  # first row is data, not a new header
        )
        tabular = len(seg) >= 2 and sum(1 for r in seg if _filled(r) >= 2) >= 2
        if not tabular and not continues:
            context += [" | ".join(c for c in r if c) for r in seg]
            continue
        if continues:
            last.rows += [[]] + seg  # keep row numbering aligned
        else:
            blocks.append(
                Block(
                    rows=seg,
                    first_row_number=grid.first_row_number + seg_start,
                    name=grid.name,
                    context="\n".join(context[-MAX_CONTEXT_LINES:]),
                )
            )
        prev_end = seg_start + len(seg)
    return blocks


def lazy_block(table: LazyTable) -> Block:
    return Block(
        rows=[table.columns] + table.preview, first_row_number=1, name=table.name, lazy=table
    )


def guess_header(rows: list[list[str]]) -> int:
    """Index of the most header-like row among the first few: wide, textual, followed by data."""
    width = max((len(r) for r in rows), default=0)
    for i, row in enumerate(rows[:12]):
        cells = [c for c in row if c]
        if len(cells) < max(2, width // 2) or i + 1 >= len(rows):
            continue
        if any(is_numeric(c) for c in cells) or len(set(cells)) < len(cells) * 0.8:
            continue
        return i
    return 0


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def describe(block: Block, header_row: int, max_rows: int = 12) -> str:
    """Prompt text for a block: raw top rows, per-column samples, and the last rows."""
    rows = block.rows
    wide = block.width > 24
    cell_w, shown_cols = (18, 14) if wide else (36, 24)
    parts: list[str] = []
    if block.context:
        parts.append("Text near the table:\n" + block.context)
    top = rows[: max(header_row + 4, 6)]
    parts.append(
        "First rows as they appear (row index: cells):\n"
        + "\n".join(
            f"{i}: "
            + " | ".join(_clip(c, cell_w) for c in r[:shown_cols])
            + (" | …" if len(r) > shown_cols else "")
            for i, r in enumerate(top)
        )
    )
    header = dedupe_headers(rows[header_row]) if rows else []
    data = [r for r in rows[header_row + 1 :] if r]
    lines = []
    for j, name in enumerate(header):
        counts: dict[str, int] = {}
        for r in data[:DISTINCT_SCAN_ROWS]:
            v = r[j] if j < len(r) else ""
            if v:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            shown = "(empty)"
        elif len(counts) <= MAX_DISTINCT_SHOWN and sum(counts.values()) > len(counts):
            # A column with few distinct values is a category: show them all, so markers of
            # total rows that only appear late in the table are visible.
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])
            shown = "all values: " + ", ".join(f"{_clip(v, 40)!r} x{n}" for v, n in ranked)
        else:
            shown = ", ".join(repr(_clip(v, 40)) for v in list(counts)[: 3 if wide else 5])
            shown += f" … ({len(counts)}+ distinct)" if len(counts) > 5 else ""
        lines.append(f"- {_clip(name, 60)!r}: {shown}")
    parts.append(
        f"Columns if row {header_row} is the header ({len(header)} columns), with sample values:\n"
        + "\n".join(lines)
    )
    tail = block.tail or (data[-4:] if len(data) > max_rows else [])
    if tail:
        parts.append(
            "Last rows:\n"
            + "\n".join(" | ".join(_clip(c, cell_w) for c in r[:shown_cols]) for r in tail)
        )
    n = "unknown (large file)" if block.lazy else str(len(data))
    parts.append(f"Data rows: {n}")
    return "\n\n".join(parts)
