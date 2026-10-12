"""What a file becomes before any interpretation: tables, numbered text lines, or page images."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Source:
    """One document to parse: a file on disk, possibly extracted from an archive."""

    display_path: str  # path relative to the input root
    local_path: Path
    member: str | None = None  # path inside the archive chain, if any
    warnings: list[str] = field(default_factory=list)
    error: str | None = None  # set when the file could not even be opened

    @property
    def name(self) -> str:
        return (self.member or self.display_path).rsplit("/", 1)[-1]

    @property
    def document_id(self) -> str:
        key = f"{self.display_path}!{self.member or ''}"
        slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-").lower()[:60].strip("-")
        return f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"

    @property
    def size(self) -> int:
        try:
            return self.local_path.stat().st_size
        except OSError:
            return 0


@dataclass
class Grid:
    """A small table source held in memory: every cell a string, rows as they appear."""

    rows: list[list[str]]
    name: str | None = None  # sheet name, JSON key ...
    context: str | None = None  # surrounding non-tabular text (preamble, sibling JSON keys)
    first_row_number: int = 1  # original row number of rows[0]


@dataclass
class LazyTable:
    """A large single-table file that must be streamed, header on the first row."""

    path: Path
    fmt: str  # csv | parquet
    columns: list[str]
    preview: list[list[str]]
    separator: str = ","
    name: str | None = None


@dataclass
class TextDoc:
    """Text as pages of lines. Line numbers are 1-based within a page."""

    pages: list[list[str]]
    origin: str  # pdf_text | ocr | text | office | html

    @property
    def n_lines(self) -> int:
        return sum(len(p) for p in self.pages)


@dataclass
class PageImage:
    page: int  # 1-based
    png: bytes


@dataclass
class Converted:
    kind: str  # tables | text | images | unreadable | empty
    file_type: str
    converter: str
    grids: list[Grid] = field(default_factory=list)
    lazy_tables: list[LazyTable] = field(default_factory=list)
    text: TextDoc | None = None
    images: list[PageImage] = field(default_factory=list)  # pages needing transcription
    note: str | None = None
    warnings: list[str] = field(default_factory=list)
