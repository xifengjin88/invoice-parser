# invoice-parser

Turns any cloud cost file (PDF bill, invoice, CSV or spreadsheet export, usage report,
Terraform, and so on) into one common line-item table, with a validation report saying how far
to trust it.

An open-weight model running on a Modal GPU decides what things mean. Code reads every value
back from the source file and checks the arithmetic. [`docs/guide.ipynb`](docs/guide.ipynb)
walks through the results.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync
```

## Deploy the model (once)

The model runs in your own Modal account. Pick any long random string as the key:

```
KEY=$(openssl rand -hex 32)
uv run --extra deploy modal secret create invoice-parser-llm VLLM_API_KEY=$KEY
uv run --extra deploy modal deploy modal_app/serve.py
echo $KEY
```

The deploy prints a URL ending in `.modal.run`. Put it and the key in a `.env` file in the
project folder:

```
INVOICE_PARSER_LLM_URL=https://<your-url>.modal.run
INVOICE_PARSER_LLM_KEY=<the key>
```

Redeploy only when `modal_app/serve.py` changes. Changes to the parser itself take effect on
the next run with no redeploy.

### Cost

Modal bills per second while the GPU container is up: during cold start, while working, and for
the idle window after the last request (2 minutes by default). It costs nothing while scaled to
zero. To stop it immediately:

```
uv run --extra deploy modal app stop invoice-parser-llm
```

Model answers are cached in `out/.cache/llm/`. Re-running a file is free unless the file, the
prompts or the model changed.

## Choose the model

The parser talks to one model service at a time, and the choice is a setting. Nothing else in
the code changes.

| Provider | What it is | Needs |
|---|---|---|
| `vllm` (default) | An open-weight model you host behind an OpenAI-compatible endpoint, such as `modal_app/serve.py` | `INVOICE_PARSER_LLM_URL`, `INVOICE_PARSER_LLM_KEY` |
| `openai` | The OpenAI API | `OPENAI_API_KEY`, and a model name |
| `claude` | The Claude API | `ANTHROPIC_API_KEY` (model defaults to `claude-opus-5-5`) |

Set it in `.env`:

```
INVOICE_PARSER_PROVIDER=claude
INVOICE_PARSER_MODEL=claude-opus-5-5      # optional for claude and vllm, required for openai
```

or per run:

```
uv run invoice-parser parse inputs/bill.pdf -o out -v --provider claude
uv run invoice-parser parse inputs/bill.pdf -o out -v --provider openai --model <name>
```

or in code:

```python
Parser(provider="claude")
Parser(provider="openai", model="<name>")
Parser(provider="vllm", base_url="https://…", api_key="…")
```

Every setting can also be passed in code, which is how to supply a key fetched at run time
(from AWS Secrets Manager, say) without putting it in the environment. A value passed in code
always wins over the environment:

```python
Parser(provider="claude", api_key=fetch_secret("anthropic-key"), model="claude-haiku-5-5")
```

| In code | Environment |
|---|---|
| `provider` | `INVOICE_PARSER_PROVIDER` |
| `model` | `INVOICE_PARSER_MODEL` |
| `api_key` | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `INVOICE_PARSER_LLM_KEY` for vllm |
| `base_url` | `INVOICE_PARSER_LLM_URL` (required for vllm, optional otherwise) |
| `effort` (claude) | `INVOICE_PARSER_EFFORT` |
| `max_unpacked_gb`, `max_archive_members` | `INVOICE_PARSER_MAX_UNPACKED_GB`, `INVOICE_PARSER_MAX_ARCHIVE_MEMBERS` |

Saved answers are kept apart per provider and model, so switching never reuses another
model's answers, and switching back costs nothing.

For Claude, two settings are on by default: reasoning effort `low` (set `INVOICE_PARSER_EFFORT`
or `Parser(provider="claude", effort="medium")` to change it), and server-side fallback, which
re-runs a request on another Claude model if a safety classifier declines it
(`Parser(provider="claude", fallbacks=False)` turns it off).

A hosted provider receives the text of the files you parse. With `vllm` it stays on
infrastructure you run.

To add a provider of your own, subclass `invoice_parser.llm.BaseLLM`, implement `_send`, and
call `register_provider("name", factory)`.

## Parse files

Point it at a file or a folder:

```
# one file
uv run invoice-parser parse inputs/some-bill.pdf -o out -v

# a whole folder
uv run invoice-parser parse inputs -o out -v

# only files whose path contains some text
uv run invoice-parser parse inputs -o out -v --only azure
```

Useful options:

| Option | Effect |
|---|---|
| `-v` | Print the checks that warned or failed under each file |
| `--only TEXT` | Only files whose path contains TEXT |
| `--resume` | Skip files already parsed without error |
| `--workers N` | Parallel model calls per document (default 8) |
| `--no-model` | Do not call the model. Only AWS CUR and FOCUS exports parse this way |
| `--debug` | Also log every model call with timing and token counts |
| `-q` | No progress lines |

The first request after the GPU has been idle waits several minutes for a cold start. Progress
lines show it waiting.

### Reading the status line

```
ok   bill_detail      aws         642 attributes  total  99.9%  367.8s  aws-bill.pdf
```

Status, document type, provider, line-item count, best catalog match level, how closely the
line items add up to the total the document states, time taken, file.

A check passes at 90% or better, warns between 70% and 90%, and fails below 70%.

## Output

Each document gets a folder under `out/`:

| File | Contents |
|---|---|
| `line_items.parquet` | The extracted rows in the common schema (75 columns) |
| `document.json` | Document type, stated totals, how tables were mapped, every validator check |
| `tags.jsonl` | For PDFs and text: what the model said about each source line |

`out/run_report.json` summarizes the last run.

Rows with `is_rollup` set are summary sections that restate the detail. Filter them out before
summing.

Other commands:

```
uv run invoice-parser inspect inputs     # what each file converts to; no model
uv run invoice-parser validate out                      # re-run the validator; no model
uv run invoice-parser report out                        # summary across all parsed documents
```

## Use it from code

A file goes in; line items come out, grouped by what you would do with them. The full,
runnable walkthrough is the notebook [`docs/guide.ipynb`](docs/guide.ipynb).

```python
from invoice_parser import parse

result = await parse("bill.pdf")  # a path, bytes, or an open file

for doc in result:  # one file can hold several documents
    print(doc.kind, doc.provider, doc.total, doc.confidence)

    for item in doc.usage:  # what is being run: the items to price
        print(
            item.service.name,
            item.region.name,
            item.product.instance_type,
            item.quantity,
            item.cost,
            item.pricing.model,
        )

    doc.commitments  # reservations and savings plans already bought
    doc.adjustments  # discounts, credits, tax, fees
```

| On a document | Meaning |
|---|---|
| `kind` | `bill`, `usage_export`, `cost_summary`, `estimate`, `infrastructure`, `other`, `unreadable` |
| `provider`, `currency`, `period`, `total` | As the document states them |
| `confidence` | `high`, `medium`, `low`, from the parser's own checks; `issues` says why |
| `usage`, `commitments`, `adjustments`, `items` | The line items. Summary rows are never included |
| `to_frame()` | The items as a short table (34 columns); `to_frame("full")` for all 75 |
| `to_dicts()` | The items as plain nested dicts, ready for JSON |

Each item groups its fields the way a price lookup uses them: `service`, `region`, `product`,
`quantity`, `cost`, `pricing`, `period`, `account`, `resource`, `source`. A value the document
did not state is `None`; nothing is guessed. `item.detail` says how precisely the item can be
matched to a price list (`sku`, `usage_type`, `attributes`, `description`, `service`, `none`).

Ways to hand over a file:

| You have | Call |
|---|---|
| A path to a file or folder | `await parse(path)` |
| Bytes | `await parse(data, name="file.csv")` |
| An open file, sync or async | `await parse(fileobj, name="file.pdf")` |
| No event loop | `parse_sync(...)` |
| Many files, or your own settings | `parser = Parser(...)`, then `await parser.parse_many([a, b, c])` |
| Your own reference, such as an upload id | `Parser(fetch=my_lookup)`, then `await parser.parse_ref(upload_id)` |

`fetch` is any function, sync or async, that turns your reference into something `parse`
accepts. The parser itself knows nothing about where files are stored.

Settings, all optional and otherwise read from the environment:

```python
Parser(
    provider="claude",  # or "openai", "vllm"; see "Choose the model"
    model="…",
    api_key="…",  # for example, fetched from a secrets manager
    cache=redis_client,  # a folder path, or anything with get(key) / set(key, value)
    workers=8,  # model calls at once within a document
    offline=False,  # True: saved answers only, never call the model
    use_model=True,  # False: only formats that need no model
)
```

Create one `Parser` and reuse it. It runs one parse at a time; parallelism happens inside a
document. Result files go to a temporary folder that is removed when the result is closed or no
longer used; pass `output_dir=` to keep them, or call `result.save(folder)`.

The stored table is still available for anything the item view leaves out: `doc.line_items()`
returns it as a Polars DataFrame, and `doc.record` is the full document record with every
validation check.

To rebuild the guide after changing the interface:

```
uv run --extra docs jupyter execute docs/guide.ipynb --inplace
```

## Run it as a workflow

`deploy/` holds what an orchestrator needs. Argo Workflows runs the steps; the model stays on
whatever endpoint the Secret names (Modal today).

```
docker build -f deploy/Dockerfile -t invoice-parser:dev .

kubectl apply -f deploy/argo/volume.example.yaml      # adjust storage class first
kubectl apply -f deploy/argo/secret.yaml              # copy of secret.example.yaml, filled in
kubectl apply -f deploy/argo/workflow-template.yaml

argo submit --from workflowtemplate/invoice-parse -p input-path=/data/in/job-1 --watch
```

The workflow lists the files under `input-path`, starts one `parse` pod per file (four at a
time), combines their summaries, and hands off to a `match-prices` placeholder. Each step reads
and writes local files on a volume mounted at `/data`; results land in
`/data/out/<job-id>/`.

A single step is one command, which is also how to try it without a cluster:

```
uv run invoice-parser job path/to/file-or-folder --out results/
```

It writes one folder per document, `summary.json`, and a `status` file (pass, warn or fail).
Exit code 0 means the step did its work, whatever the validation status; exit code 1 means a
document could not be processed at all, which the workflow retries.

## Viewer

A Streamlit app that shows each source file next to what was extracted from it. It never calls
the model.

```
uv run --extra viewer streamlit run app/viewer.py
```

It opens in your browser. Pick a document in the sidebar.

- For PDFs and text files, the page is on the left and the rows from that page on the right.
  Green marks lines that became rows, grey marks rollups. Click a row to highlight its source
  line and see all its fields and what the model said about that line.
- For CSVs and spreadsheets, the raw table is on the left and the extracted rows on the right.
- The header shows validation status, the extracted and stated totals, and how closely they
  match. An expander lists every check.

Click "Reload results" after parsing more files. The sidebar lets you point it at a different
results or source folder.

## Development

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests use a stand-in for the model and need no network or GPU.

## Project layout

```
src/invoice_parser/
  schema.py        canonical fields and the document record
  convert/         any file to tables, numbered text lines or page images
  llm.py           the only code that calls the model; the three providers; disk cache
  items.py         the reader's view: Item and the usage / commitments / adjustments lists
  sdk.py           the public interface: parse, Parser, ParseResult, ParsedDocument
  tabular/         table blocks, column mapping, streaming apply
  freeform/        line tagging, hierarchy rebuild, wording pass
  enrich/          normalizes stated values into the schema's vocabulary
  validate/        the checks
  pipeline.py      one document end to end; batch runs
  cli.py           the command line
app/viewer.py      Streamlit viewer
docs/guide.ipynb   runnable walkthrough of the results
examples/          a small made-up export the guide and tests run on
modal_app/serve.py vLLM on a Modal GPU
```

Input folders (`inputs/`, `cost-estimate-inputs/`) and result folders (`out/`, `out-*/`) hold
billing data and are git-ignored.
