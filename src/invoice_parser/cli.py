"""Command line: inspect, parse, validate, report."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer

from .convert import convert, iter_sources
from .llm import LLMError, NotConfigured, create_llm
from .pipeline import CACHE_DIR, iter_documents, run, summarize, write_report
from .schema import DocumentRecord
from .validate import validate_dir

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Cloud cost document parser.")

_MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "-   "}


def _quiet() -> None:
    for name in ("pdfminer", "pdfplumber", "httpx", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _progress(level: int) -> None:
    """Progress lines go to stderr so the result lines on stdout stay clean."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger = logging.getLogger("invoice_parser")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def _line(record: DocumentRecord) -> str:
    s = summarize(record)
    best = next(
        iter(
            sorted(
                s["match_levels"],
                key=["sku", "usage_type", "attributes", "description", "service", "none"].index,
            )
        ),
        "-",
    )
    source = s["source"] if len(s["source"]) <= 62 else "…" + s["source"][-61:]
    match = f"{s['total_match']:.1%}" if s.get("total_match") is not None else "-"
    return (
        f"{_MARK[s['status']]} {(s['doc_type'] or '?'):16} {','.join(s['providers']) or '-':6} "
        f"{s['line_items']:>8} {best:11} total {match:>6} {s['seconds'] or 0:>6.1f}s  {source}"
    )


def _print_problems(record: DocumentRecord) -> None:
    for check in record.validation.checks:
        if check.status in ("warn", "fail"):
            typer.echo(f"       {check.status}: {check.name}: {check.summary}")
    if record.error:
        typer.echo(f"       error: {record.error}")


@app.command()
def inspect(path: Path) -> None:
    """Show what each file converts to, without calling the model."""
    _quiet()
    work = Path(tempfile.mkdtemp(prefix="invoice-parser-"))
    try:
        for source in iter_sources(path, work):
            c = convert(source)
            if c.lazy_tables:
                detail = f"streamed table, {len(c.lazy_tables[0].columns)} columns"
            elif c.grids:
                detail = ", ".join(
                    f"{g.name or 'table'} {len(g.rows)}x{max(map(len, g.rows))}" for g in c.grids
                )
            elif c.text:
                detail = f"{len(c.text.pages)} pages, {c.text.n_lines} lines, {len(c.images)} to transcribe"
            else:
                detail = c.note or ""
            name = source.display_path + (f"!{source.member}" if source.member else "")
            typer.echo(f"{c.kind:10} {c.file_type:8} {name}\n           {detail[:150]}")
            for warning in source.warnings + c.warnings:
                typer.echo(f"           note: {warning}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.command()
def parse(
    path: Path,
    out: Annotated[Path, typer.Option("-o", "--out", help="output directory")] = Path("out"),
    workers: Annotated[int, typer.Option(help="parallel model calls per document")] = 8,
    resume: Annotated[
        bool, typer.Option(help="skip documents already parsed without error")
    ] = False,
    only: Annotated[
        str | None, typer.Option(help="only files whose path contains this text")
    ] = None,
    provider: Annotated[
        str | None, typer.Option(help="model service: vllm, openai or claude")
    ] = None,
    model: Annotated[str | None, typer.Option(help="model name at that provider")] = None,
    no_model: Annotated[
        bool, typer.Option("--no-model", help="known tabular formats only")
    ] = False,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="reuse saved model answers only; never call the model"),
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="print failing checks")] = False,
    quiet: Annotated[bool, typer.Option("-q", "--quiet", help="no progress lines")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="also log every model call")] = False,
) -> None:
    """Parse a file or directory into canonical line items and validate the result."""
    _quiet()
    if not quiet:
        _progress(logging.DEBUG if debug else logging.INFO)
    llm = None
    if not no_model:
        try:
            llm = create_llm(provider, model=model, cache=out / CACHE_DIR / "llm", offline=offline)
            logging.getLogger("invoice_parser").info(
                "offline: saved model answers only, no model calls"
                if offline
                else f"model: {llm.provider}, {llm.model}"
            )
        except NotConfigured as exc:
            if provider or model:
                raise typer.BadParameter(str(exc)) from exc
            typer.echo(
                f"No model configured: {exc}\nContinuing with known tabular formats only.\n",
                err=True,
            )
        except LLMError as exc:  # an unknown provider, a missing model name or credential
            raise typer.BadParameter(str(exc)) from exc

    def show(record: DocumentRecord) -> None:
        typer.echo(_line(record))
        if verbose:
            _print_problems(record)

    records = run(path, out, llm, workers=workers, resume=resume, on_document=show, only=only)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.validation.status] = counts.get(r.validation.status, 0) + 1
    typer.echo(
        f"\n{len(records)} documents: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    typer.echo(f"Results in {out}/  (summary: {out}/run_report.json)")


@app.command()
def job(
    input_path: Annotated[Path, typer.Argument(help="a local file or folder to parse")],
    out: Annotated[Path, typer.Option("-o", "--out", help="output folder")] = Path("out"),
    cache: Annotated[
        str | None,
        typer.Option(help="folder or redis:// URL for saved model answers"),
    ] = None,
    workers: Annotated[int, typer.Option(help="parallel model calls per document")] = 8,
    offline: Annotated[
        bool, typer.Option("--offline", help="saved model answers only; never call the model")
    ] = False,
    no_model: Annotated[
        bool, typer.Option("--no-model", help="only formats that need no model")
    ] = False,
    provider: Annotated[
        str | None, typer.Option(help="model service: vllm, openai or claude")
    ] = None,
    model: Annotated[str | None, typer.Option(help="model name at that provider")] = None,
    status_file: Annotated[
        Path | None, typer.Option(help="also write the status (pass, warn, fail) to this file")
    ] = None,
) -> None:
    """Run as one step of a workflow: local files in, results and a status file out."""
    from .job import run

    _quiet()
    _progress(logging.INFO)
    code = run(
        input_path,
        out,
        cache=cache,
        workers=workers,
        offline=offline,
        use_model=not no_model,
        provider=provider,
        model=model,
        status_file=status_file,
    )
    raise typer.Exit(code)


@app.command()
def validate(out: Path = typer.Argument(Path("out"))) -> None:
    """Re-run the validator over an output directory."""
    records = validate_dir(out)
    write_report(out, records)
    for record in records:
        typer.echo(_line(record))
        _print_problems(record)
    if any(r.validation.status == "fail" for r in records):
        raise typer.Exit(1)


@app.command()
def report(out: Path = typer.Argument(Path("out"))) -> None:
    """Corpus summary: status, document type, line items and catalog match level per document."""
    records = list(iter_documents(out))
    summary = write_report(out, records)
    for record in records:
        typer.echo(_line(record))
    typer.echo(f"\n{summary['documents']} documents, {summary['line_items']:,} line items")
    typer.echo(
        "status:      " + ", ".join(f"{k} {v}" for k, v in sorted(summary["by_status"].items()))
    )
    total = sum(summary["match_levels"].values()) or 1
    typer.echo(
        "match level: "
        + ", ".join(
            f"{k} {v:,} ({v / total:.1%})"
            for k, v in sorted(summary["match_levels"].items(), key=lambda kv: -kv[1])
        )
    )


def main() -> None:
    app()
