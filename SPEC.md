# Spec: Cloud Cost Document Parser

Status: approved 2026-10-11. Schema version: 0.1.

## Objective

Turn any cloud-cost-related file into one common line-item schema, with as much
detail as the file contains, so that a later step can match each line against
our SKU pricing catalog database.

- **Input:** any file type (PDF, CSV, spreadsheet, archive, image, JSON, text,
  slides, infrastructure code), any provider, any layout, wording or language.
- **Output:** the same record shape every time, at whatever granularity the
  file supports, plus a validation report saying how far to trust it.
- **User:** anyone holding cloud cost files they need in one shape.
- **Step two (out of scope here, but drives the schema):** map each line item
  to the pricing catalog, as granularly as the keys allow.

### Assumptions (correct these before build)

1. Non-bill inputs that describe infrastructure (Terraform, CloudFormation,
   VM size lists, Pricing Calculator estimates) are in scope. They produce
   records with no billed cost (`record_type` = `resource` or `estimate`).
2. Files may be sent to a model endpoint in our own Modal account, and nowhere
   else.
3. Output is files on disk (Parquet + JSON). Loading into a database is step two.
4. It runs as a batch CLI on a laptop (this one has 8 GB RAM), so large files
   must stream.
5. English prompts; documents may be in any language.
6. Field names follow FOCUS 1.2 (the open billing standard) in snake_case,
   extended with pricing-match attributes.

## Design

### Principle

The model decides *what things mean*. Code moves every number. The validator
proves it.

- In tables, the model never sees more than a sample and never emits a value;
  it emits a column mapping that code applies to all rows.
- In free-form documents, the model tags numbered source lines; code reads the
  values back out of those lines and builds the hierarchy.
- Anything inferred rather than read is listed in `derived_fields`.

### Pipeline

```
file ─► convert ─► classify ─┬─► tabular path ──┐
        (any type            │   (column map)   ├─► enrich ─► write ─► validate
         to text/tables)     └─► free-form path ┘
                                 (line tagging)
```

| Module | Responsibility | Depends on |
|---|---|---|
| `schema` | Canonical line item, document record, enums, Arrow schema | none |
| `convert` | Any file to `TextDoc` (numbered lines per page) or `TableBlock`s; unpack archives recursively; rasterize scans | none |
| `llm` | Client for an OpenAI-compatible endpoint; schema-constrained JSON; retries; on-disk cache; fake client for tests | none |
| `modal_app` | vLLM serving the open-weight model on a Modal GPU, scale to zero | none |
| `classify` | Document type, provider, language, expected granularity | convert, llm |
| `tabular` | Find tables in a grid, get a mapping plan from the model (or a known-format seed), apply it by streaming | schema, convert, llm |
| `freeform` | Page-parallel line tagging, hierarchy walk, document metadata and stated totals | schema, convert, llm |
| `enrich` | Deterministic derivations: region codes, usage-type parsing, rate-description parsing, service aliases | schema |
| `validate` | Checks below; per-document and corpus-level report | schema |
| `pipeline`, `cli` | Orchestration, batch runs, resumability | all |

Build order: schema, validate → convert → tabular (seeds + fake model) →
modal_app, llm → tabular (live) → freeform → enrich → vision path → batch report.

### Convert: supported inputs

| Kind | Types | Becomes |
|---|---|---|
| Delimited | csv, tsv, txt with delimiters, any of these gzipped | `TableBlock`s (lazy, all columns as strings) |
| Spreadsheet | xlsx, xlsm, xls, ods | one grid per sheet, then `TableBlock`s |
| Columnar | parquet | `TableBlock` |
| Structured text | json, jsonl, yaml, tf, CloudFormation | table if it is an array of records, otherwise `TextDoc` |
| PDF with text | pdf | `TextDoc`, layout-preserving, numbered lines per page |
| PDF scan, image | pdf without text layer, png, jpg, tiff, webp | page images, transcribed by the vision model to a `TextDoc` |
| Office | docx, pptx, html | `TextDoc` (embedded tables become `TableBlock`s) |
| Archive | zip, tar, gz | each member processed as its own document |
| Unknown | anything else | sniff magic bytes; try as text; otherwise record as unreadable with the reason |

A file never crashes the run. The worst outcome is a document record saying
why nothing was extracted.

### Tabular path

1. Split each grid into table blocks (blank-line and column-count boundaries),
   so preamble blocks like "Estimate summary" separate from the detail table.
2. Known formats (AWS CUR in both spellings, FOCUS) use a checked-in seed mapping and
   need no model call. Model responses are cached on disk by prompt, so re-running a file
   costs nothing. (Plans are not shared across files by header alone, because a plan also
   carries per-file constants such as the billing period.)
3. Otherwise show the model the header and about 30 sample rows. It returns a
   plan:
   - `field_map`: canonical field to source column
   - `attribute_columns`: source columns kept as pricing attributes
   - `melt`: for pivoted tables (services or months as columns), which columns
     unpivot and what their labels mean
   - `constants`: values true for every row (provider, currency)
   - `row_filters`: total and subtotal rows to drop, recorded as stated totals
4. Code applies the plan with Polars in streaming mode and writes Parquet.
   Unmapped non-empty columns go to `attributes` so nothing is dropped.

### Free-form path

1. Text is split into pages of numbered lines. One call on the first and last
   pages extracts document metadata and stated totals.
2. Pages are tagged in parallel. For each meaningful line the model returns the
   line number, a `kind`, and field values:
   - header kinds set context: `seller`, `account`, `service`, `region`, `group`
   - `item` is a charge or resource line
   - `total` is a stated total or subtotal
3. Code walks the tagged lines in order with a context stack (a header replaces
   its own level and clears everything set after it; a repeated page header is
   a no-op), so items inherit account, service and region across page breaks.
4. Code re-reads each amount and quantity from the referenced source line. A
   value the line does not contain is rejected and counted.
5. Sections that restate the same charges (for example "Detail for Consolidated
   Bill" and then each linked account) are tagged with a `section`, and rollup
   lines are flagged `is_rollup` so sums never double count.

### Canonical line item (schema 0.1)

Every field except provenance is nullable. Same columns for every input.

| Group | Fields |
|---|---|
| Provenance | `line_item_id`, `document_id`, `source_file`, `source_member`, `source_page`, `source_row`, `source_text`, `extraction_method`, `schema_version` |
| Kind | `record_type` (charge, estimate, resource), `granularity` (resource, sku, usage_type, service, account, document), `section`, `is_rollup` |
| Parties | `provider`, `publisher`, `invoice_issuer`, `billing_account_id`, `billing_account_name`, `sub_account_id`, `sub_account_name`, `invoice_id` |
| Time | `billing_period_start`, `billing_period_end`, `charge_period_start`, `charge_period_end` |
| Catalog keys | `service_name` (as stated), `service_code`, `service_category`, `region_id`, `region_name`, `availability_zone`, `sku_id`, `sku_price_id`, `sku_meter`, `product_family`, `usage_type`, `operation` |
| Pricing attributes | `instance_type`, `instance_family`, `operating_system`, `tenancy`, `license_model`, `database_engine`, `deployment_option`, `storage_type`, `purchase_option`, `commitment_discount_id`, `commitment_discount_type`, `attributes` (JSON, everything else) |
| Resource | `resource_id`, `resource_name`, `resource_type`, `tags` (JSON) |
| Charge | `charge_category` (usage, purchase, tax, credit, discount, adjustment), `charge_subcategory`, `charge_description` |
| Amounts | `consumed_quantity`, `consumed_unit`, `pricing_quantity`, `pricing_unit`, `list_unit_price`, `contracted_unit_price`, `list_cost`, `contracted_cost`, `billed_cost`, `effective_cost`, `tax_amount`, `billing_currency` |
| Trust | `derived_fields` (list), `match_level` |

`match_level` states the best catalog match the line can support, computed by
the validator:

| Level | Keys present |
|---|---|
| `sku` | `sku_id` or `sku_price_id` |
| `usage_type` | `service_code` + `usage_type` (+ `operation`) |
| `attributes` | service + `region_id` + `instance_type` or equivalent attributes + unit |
| `description` | service + region + a rate description to match by text |
| `service` | service only |
| `none` | nothing usable |

### Document record (`document.json`)

Document id, source path and member, file type, converter used, `doc_type`
(invoice, bill_detail, usage_export, cost_summary, estimate,
infra_definition, analysis, not_cost_related, unreadable), providers,
language, invoice metadata (ids, dates, accounts, currency, exchange rate),
stated totals with their scope, line item count, model calls and tokens, the
mapping plan used, warnings, and the validation report.

### Validator

| Check | What it proves |
|---|---|
| Schema | Columns, types and enum values conform; ids unique; dates ordered; currencies are ISO codes |
| Row accounting (tabular) | rows out + rows filtered = rows in; mapped cost column sums match the source to the cent; count of values that failed to parse |
| Grounding (free-form) | Every amount, quantity and description appears in its cited source line |
| Missed lines (free-form) | Source lines containing a money amount that no record or total accounts for |
| Reconciliation | Non-rollup line items sum to stated totals at each scope (document, account, service), within tolerance |
| Arithmetic | quantity × unit price ≈ cost where all three exist |
| Domain | Region ids valid for the provider; instance types well-formed; dates inside the billing period |
| Coverage | Fill rate per field; distribution of `match_level` |

Each check returns pass, warn or fail with counts and examples. A document's
status is its worst check. `report` aggregates across a run.

## Tech Stack

- Python 3.13, managed with uv (`pyproject.toml`, `uv.lock`), `src/` layout.
- pydantic 2 (models, JSON Schema for constrained output), polars (streaming
  apply, Parquet), pyarrow.
- pdfplumber (layout text), pypdfium2 (rasterizing), python-calamine
  (spreadsheets), python-docx, python-pptx, pyyaml.
- openai client against the vLLM endpoint; typer CLI; pytest; ruff.
- Modal + vLLM. Model is a config value, default `Qwen/Qwen3-VL-8B-Instruct`
  (Apache-2.0, text and vision in one model). `Qwen/Qwen3-VL-32B-Instruct` is
  the upgrade if validator scores on the corpus fall short of the targets.

## Commands

```
Setup:        uv sync
Deploy model: uv run --extra deploy modal deploy modal_app/serve.py   # run by a person, not the agent
Inspect:      uv run invoice-parser inspect <path>            # convert only, no model
Parse:        uv run invoice-parser parse <file-or-dir> -o out/ [--workers 8] [--resume] [--only TEXT] [--no-model] [-v]
Validate:     uv run invoice-parser validate out/
Report:       uv run invoice-parser report out/               # corpus summary table
Test:         uv run pytest
Lint:         uv run ruff check . && uv run ruff format --check .
```

Endpoint settings come from `INVOICE_PARSER_LLM_URL`, `INVOICE_PARSER_LLM_KEY`
and `INVOICE_PARSER_MODEL` (or `.env`).

## Project Structure

```
SPEC.md
pyproject.toml
src/invoice_parser/
  schema.py            canonical models, enums, Arrow schema
  convert/             one module per input kind, plus dispatch and archives
  llm.py               client, cache, fake
  classify.py
  tabular/             blocks, plan model, seeds/, apply
  freeform/            tagging, hierarchy walk, metadata
  enrich/              regions, usage types, rate descriptions, service aliases
  validate/            checks, report
  pipeline.py
  cli.py
modal_app/serve.py     vLLM service
tests/                 unit tests and small synthetic fixtures
cost-estimate-inputs/  real inputs (git-ignored)
out/                   results (git-ignored)
```

Output layout per document:

```
out/<document_id>/line_items.parquet
out/<document_id>/document.json
out/run_report.json
```

## Code Style

Typed, small pure functions, pydantic at boundaries, no model call outside
`llm.py`.

```python
class TaggedLine(BaseModel):
    """One source line as tagged by the model. Values are re-read from the line by code."""

    line: int
    kind: LineKind
    label: str | None = None
    amount: Decimal | None = None


def ground_amount(tagged: TaggedLine, source_line: str) -> Decimal | None:
    """Return the amount only if the source line actually contains it."""
    if tagged.amount is None:
        return None
    return tagged.amount if tagged.amount in parse_amounts(source_line) else None
```

- Money is `Decimal` in Python and `decimal(38, 10)` in Parquet, never float.
- Enums are lowercase snake_case strings.
- ruff defaults, line length 100.

## Testing Strategy

- **Unit (pytest, no network, no GPU):** number and date parsing, block
  splitting, plan application including melt, hierarchy walk, grounding, every
  validator check. Model calls use the fake client with recorded responses.
- **Fixtures:** small synthetic files modelled on each real layout, with no
  customer data, checked in under `tests/fixtures/`.
- **Corpus run (manual, needs the endpoint):** `parse` then `report` over
  `cost-estimate-inputs/`. The validator is the acceptance test.

## Boundaries

- **Always:** keep every value traceable to file, page and line or row; list
  inferred fields in `derived_fields`; stream large files; bump
  `schema_version` on any schema change.
- **Ask first:** changing GPU size or model (costs money); adding
  dependencies beyond the list above; changing canonical field names; sending
  file contents anywhere other than our Modal endpoint.
- **Never:** run Modal commands from the agent session (the user deploys); modify or delete input files; commit `cost-estimate-inputs/` or
  `out/` to git; let the model emit numbers for tabular rows; invent a SKU that
  the file does not state (derivation from the catalog is step two).

## Success Criteria

1. `parse cost-estimate-inputs/` completes with no crash, and every file and
   archive member has a `document.json` with either line items or a stated
   reason for none.
2. Tabular documents: row accounting passes for 100% of them, with cost sums
   equal to the source to the cent.
3. Free-form documents: at least 99% of extracted amounts are grounded, and
   under 2% of money-bearing lines are unaccounted for.
4. At least 90% of documents that state a grand total reconcile within 0.5%;
   the rest are flagged, not silently wrong.
5. The 1 GB usage report (938,644 rows) parses on this 8 GB laptop in under
   10 minutes with one model call or none.
6. The 294-page invoice parses in under 10 minutes of wall time.
7. `uv run pytest` passes with no network.
8. `report` shows the `match_level` distribution per document, which is the
   input to step two.

## Open Questions

1. **Catalog tables.** What are the tables and key columns in the pricing
   catalog (for example AWS price-list style `sku` / `rateCode` /
   `usagetype` / `operation`, or something normalized of our own)? The catalog
   keys and pricing attributes above are a best guess until I see the DDL.
2. **Infrastructure files.** Confirm assumption 1: should Terraform,
   CloudFormation and VM size lists become `resource` records to be priced?
3. **Rollups.** When a file has both a summary and its detail, keep both
   (flagged) or keep only the most detailed? The spec keeps both.
4. **Model budget.** Is a larger GPU acceptable if the 8B model misses the
   targets?
