"""Enumerate documents under a path, unpacking archives (including truncated zips) safely."""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import tarfile
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

from .types import Source

MAX_DEPTH = 4
# An archive is untrusted input: a few kilobytes can expand to fill a disk. One input file,
# with everything nested inside it, may unpack to at most this much.
MAX_UNPACKED_BYTES = int(float(os.environ.get("INVOICE_PARSER_MAX_UNPACKED_GB", "10")) * 2**30)
MAX_MEMBERS = int(os.environ.get("INVOICE_PARSER_MAX_ARCHIVE_MEMBERS", "10000"))
OFFICE_EXTS = {".xlsx", ".xlsm", ".docx", ".pptx", ".ods", ".odt", ".odp"}
_SKIP_NAMES = {".DS_Store", "Thumbs.db"}


class ArchiveTooLarge(ValueError):
    pass


class _Budget:
    """What one input file may still unpack, shared by every archive nested inside it."""

    def __init__(self, size: int = 0, members: int = 0) -> None:
        self.max_bytes = self.bytes = size or MAX_UNPACKED_BYTES
        self.max_members = self.members = members or MAX_MEMBERS

    def member(self) -> None:
        self.members -= 1
        if self.members < 0:
            raise ArchiveTooLarge(f"more than {self.max_members:,} files inside")

    def write(self, dst, chunk: bytes) -> None:
        self.bytes -= len(chunk)
        if self.bytes < 0:
            raise ArchiveTooLarge(
                f"unpacks to more than {self.max_bytes / 2**30:g} GB "
                "(raise the limit if this is expected)"
            )
        dst.write(chunk)

    def copy(self, src, dst) -> None:
        while chunk := src.read(1 << 20):
            self.write(dst, chunk)


def _skip(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return (
        not parts[-1]
        or parts[-1] in _SKIP_NAMES
        or parts[-1].startswith("._")
        or "__MACOSX" in parts
    )


def _magic(path: Path, n: int = 8) -> bytes:
    try:
        with path.open("rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def archive_kind(path: Path) -> str | None:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] in OFFICE_EXTS:
        return None
    head = _magic(path)
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    try:
        if tarfile.is_tarfile(path):
            return "tar"
    except OSError:
        pass
    return None


def _dest(workdir: Path, key: str, name: str) -> Path:
    """A collision-free extraction path that cannot escape workdir (member names are untrusted)."""
    safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in name.rsplit("/", 1)[-1])[-120:]
    folder = workdir / hashlib.sha1(key.encode()).hexdigest()[:12]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (safe or "member")


def _salvage_zip(
    path: Path, workdir: Path, key: str, budget: _Budget
) -> tuple[list[tuple[str, Path]], str | None]:
    """Recover members from a zip whose central directory is missing (truncated download).

    Walks local file headers and inflates each deflate stream until it ends on its own.
    Returns the members and the name of the one that was cut short, if any.
    """
    out: list[tuple[str, Path]] = []
    cut: str | None = None
    with path.open("rb") as fh:
        while True:
            header = fh.read(30)
            if len(header) < 30 or header[:4] != b"PK\x03\x04":
                break
            _, flags, method, _, _, _, csize, _, name_len, extra_len = struct.unpack(
                "<HHHHHIIIHH", header[4:]
            )
            name = fh.read(name_len).decode("utf-8", "replace")
            fh.read(extra_len)
            if method not in (0, 8):
                break
            budget.member()
            dest = _dest(workdir, f"{key}!{name}", name)
            complete = False
            with dest.open("wb") as dst:
                if method == 0:
                    left = csize
                    while left and (chunk := fh.read(min(left, 1 << 20))):
                        budget.write(dst, chunk)
                        left -= len(chunk)
                    complete = True
                else:
                    inflater = zlib.decompressobj(-15)
                    while not inflater.eof:
                        chunk = fh.read(1 << 16)
                        if not chunk:
                            break
                        # Inflate in bounded steps: one small chunk can expand enormously.
                        while chunk:
                            budget.write(dst, inflater.decompress(chunk, 1 << 20))
                            chunk = inflater.unconsumed_tail
                    complete = inflater.eof
                    if complete:
                        fh.seek(-len(inflater.unused_data), 1)
            if not name.endswith("/") and dest.stat().st_size:
                out.append((name, dest))
            if not complete:
                cut = name
                break
            if flags & 0x08:  # data descriptor follows; find the next local header
                tail = fh.read(24)
                nxt = max(tail.find(b"PK\x03\x04"), tail.find(b"PK\x01\x02"))
                if nxt < 0:
                    break
                fh.seek(nxt - len(tail), 1)
    return out, cut


def _expand(source: Source, workdir: Path, depth: int, budget: _Budget) -> Iterator[Source]:
    kind = archive_kind(source.local_path) if depth < MAX_DEPTH else None
    if kind is None:
        yield source
        return
    key = f"{source.display_path}!{source.member or ''}"
    inner_prefix = f"{source.member}!/" if source.member else ""
    members: list[tuple[str, Path]] = []
    warnings: list[str] = []
    try:
        if kind == "zip":
            try:
                with zipfile.ZipFile(source.local_path) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or _skip(info.filename):
                            continue
                        budget.member()
                        dest = _dest(workdir, f"{key}!{info.filename}", info.filename)
                        with zf.open(info) as src, dest.open("wb") as dst:
                            budget.copy(src, dst)
                        members.append((info.filename, dest))
            except zipfile.BadZipFile:
                salvaged, cut = _salvage_zip(source.local_path, workdir, key, budget)
                members = [(n, p) for n, p in salvaged if not _skip(n)]
                warnings.append(
                    f"zip archive is truncated or corrupt; salvaged {len(members)} member(s)"
                    + (f"; {cut} is cut short" if cut else "")
                )
        elif kind == "gzip":
            inner = (
                source.name[:-3] if source.name.lower().endswith(".gz") else source.name + ".out"
            )
            dest = _dest(workdir, f"{key}!{inner}", inner)
            try:
                budget.member()
                with gzip.open(source.local_path, "rb") as src, dest.open("wb") as dst:
                    budget.copy(src, dst)
            except (EOFError, zlib.error):
                if not dest.stat().st_size:
                    raise
                warnings.append("gzip stream is cut short; the last rows are missing")
            members.append((inner, dest))
        elif kind == "tar":
            with tarfile.open(source.local_path) as tf:
                for info in tf:
                    if not info.isfile() or _skip(info.name):
                        continue
                    budget.member()
                    dest = _dest(workdir, f"{key}!{info.name}", info.name)
                    src = tf.extractfile(info)
                    if src is None:
                        continue
                    with src, dest.open("wb") as dst:
                        budget.copy(src, dst)
                    members.append((info.name, dest))
    except Exception as exc:  # unreadable archive: report it as a document
        source.error = f"could not unpack {kind} archive: {exc}"
        yield source
        return
    if not members:
        source.error = f"{kind} archive contains no readable files"
        source.warnings += warnings
        yield source
        return
    for name, dest in members:
        child = Source(
            display_path=source.display_path,
            local_path=dest,
            member=inner_prefix + name,
            warnings=list(warnings),
        )
        yield from _expand(child, workdir, depth + 1, budget)


def iter_sources(
    path: Path, workdir: Path, max_bytes: int | None = None, max_members: int | None = None
) -> Iterator[Source]:
    """Yield every document under `path` (a file or directory), archives expanded. The limits
    are per input file; left out, the module defaults apply."""
    path = path.resolve()
    root = path if path.is_dir() else path.parent
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        rel = file.relative_to(root).as_posix()
        if _skip(rel) or any(part.startswith(".") for part in file.relative_to(root).parts):
            continue
        yield from _expand(
            Source(display_path=rel, local_path=file),
            workdir,
            0,
            _Budget(max_bytes or 0, max_members or 0),
        )
