import asyncio
import gc
import io
import zipfile
from decimal import Decimal

import pytest

from invoice_parser import FileInput, Parser
from invoice_parser.llm import FakeLLM, make_cache

FOCUS = (
    b"BilledCost,BillingCurrency,ChargeCategory,ChargePeriodStart,ServiceName,SkuId,ProviderName\n"
    b"1.25,USD,Usage,2026-06-01T00:00:00+00:00,Amazon MemoryDB,FR8PTRK3KEFVGG6Y,AWS\n"
    b"2.75,USD,Usage,2026-06-01T00:00:00+00:00,Amazon S3,AB12CD34EF56GH78,AWS\n"
)


def run(coro):
    return asyncio.run(coro)


def test_bytes_with_a_name():
    parser = Parser(use_model=False)
    with run(parser.parse(FOCUS, name="export.csv")) as result:
        assert len(result) == 1 and result.ok and result.status == "pass"
        doc = result[0]
        assert doc.name == "export.csv" and doc.providers == ["aws"]
        assert doc.line_items()["billed_cost"].sum() == Decimal("4.00")
        assert [r["sku_id"] for r in doc.rows()] == ["FR8PTRK3KEFVGG6Y", "AB12CD34EF56GH78"]
        assert result.summary()["line_items"] == 2
        folder = result.folder
        assert folder.is_dir()
    assert not folder.exists()  # the temporary folder goes with the result


def test_open_files_sync_and_async_and_upload_wrappers():
    class AsyncFile:  # the shape of an async upload object
        filename = "async.csv"

        def __init__(self, data):
            self._buf = io.BytesIO(data)

        async def read(self, n=-1):
            return self._buf.read(n)

    class Wrapper:  # the shape of a framework upload: content on `.stream`
        filename = "wrapped.csv"
        stream = io.BytesIO(FOCUS)

    plain = io.BytesIO(FOCUS)
    plain.name = "/tmp/somewhere/plain.csv"
    parser = Parser(use_model=False)
    result = run(
        parser.parse_many([plain, AsyncFile(FOCUS), Wrapper(), FileInput(FOCUS, "given.csv")])
    )
    assert [d.name for d in result] == ["plain.csv", "async.csv", "wrapped.csv", "given.csv"]
    assert result.line_items().height == 8 and result.line_items()["document_id"].n_unique() == 4
    result.close()


def test_path_archive_and_output_dir(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a/one.csv", FOCUS)
        zf.writestr("b/two.csv", FOCUS)
    out = tmp_path / "kept"
    result = Parser(use_model=False).parse_sync(archive, output_dir=out)
    assert [d.name for d in result] == ["bundle.zip › a/one.csv", "bundle.zip › b/two.csv"]
    result.close()  # an explicit output folder is never removed
    assert all((out / d.id / "line_items.parquet").is_file() for d in result)
    assert all((out / d.id / "document.json").is_file() for d in result)


def test_same_name_twice_and_files_outlive_the_result_object():
    parser = Parser(use_model=False)
    docs = run(parser.parse_many([FileInput(FOCUS, "x.csv"), FileInput(FOCUS, "x.csv")])).documents
    gc.collect()  # the result object is gone; its documents still need their files
    assert len({d.id for d in docs}) == 2
    assert [d.line_items().height for d in docs] == [2, 2]
    path = docs[0].path
    del docs
    gc.collect()
    assert not path.exists()


def test_parse_ref_uses_your_own_lookup():
    store = {"upload-42": FileInput(FOCUS, "from-store.csv")}

    async def fetch(upload_id):
        return store[upload_id]

    result = run(Parser(use_model=False, fetch=fetch).parse_ref("upload-42"))
    assert result[0].name == "from-store.csv" and result.ok
    with pytest.raises(RuntimeError):
        run(Parser(use_model=False).parse_ref("upload-42"))


def test_a_file_needing_the_model_reports_why_when_there_is_none():
    result = Parser(use_model=False).parse_sync(b"Service,Cost\nCompute,5\n", name="summary.csv")
    doc = result[0]
    assert not doc.ok and doc.line_items().height == 0
    assert any("no model endpoint" in w for w in doc.record.warnings)


def test_model_path_through_the_public_interface():
    plan = {
        "is_cost_data": True, "summary": "cost by service", "provider": "gcp", "header_row": 0,
        "record_type": "charge", "granularity": "service",
        "columns": [{"column": "Service", "use": "service_name"}, {"column": "Cost", "use": "billed_cost"}],
        "pivot": None, "constants": [], "total_rows": [],
    }  # fmt: skip
    parser = Parser(llm=FakeLLM({"table_plan": lambda s, u: plan}))
    doc = parser.parse_sync(b"Service,Cost\nCompute,5\nStorage,7\n", name="summary.csv")[0]
    assert doc.ok and doc.providers == ["gcp"]
    assert doc.line_items()["billed_cost"].sum() == Decimal("12")


def test_any_get_set_store_can_hold_saved_answers():
    class Store(dict):  # stands in for a Redis client
        def set(self, key, value):
            self[key] = value.encode()

    store = Store()
    cache = make_cache(store)
    assert cache.get("k") is None
    cache.set("k", '{"content": "x"}')
    assert cache.get("k") == '{"content": "x"}' and next(iter(store)).startswith("invoice-parser:")


def test_reader_view_groups_items_by_what_they_are_for():
    from decimal import Decimal
    from pathlib import Path

    from invoice_parser import Item, Parser

    sample = Path(__file__).parent.parent / "examples" / "sample-focus-export.csv"
    result = Parser(use_model=False).parse_sync(sample)
    doc = result[0]
    assert (doc.kind, doc.provider, doc.currency, doc.confidence) == (
        "usage_export", "aws", "USD", "high",
    )  # fmt: skip
    assert doc.issues == [] and doc.period.start.year == 2026 and doc.total is None

    assert (len(doc.items), len(doc.usage), len(doc.commitments), len(doc.adjustments)) == (
        9, 5, 1, 3,
    )  # fmt: skip
    assert [i.kind for i in doc.adjustments] == ["credit", "tax", "fee"]
    assert len(doc.adjustments.of_kind("tax")) == 1
    assert doc.usage.total() == {"USD": Decimal("412.80")}

    item = doc.usage[0]
    assert isinstance(item, Item)
    assert (item.service.name, item.service.code) == ("Amazon Elastic Compute Cloud", "AmazonEC2")
    assert (item.region.name, item.region.code) == ("US East (N. Virginia)", "us-east-1")
    assert item.product.usage_type == "BoxUsage:m5.large" and item.product.instance_type is None
    assert (item.quantity.amount, item.quantity.unit) == (Decimal("2160"), "Hours")
    assert (item.cost.amount, item.cost.currency) == (Decimal("207.36"), "USD")
    assert item.pricing.model == "on_demand" and item.detail == "sku"
    assert item.account.id == "444455556666" and item.source.row == 2
    assert doc.usage[-1].service.code == "AmazonS3" and len(doc.usage[1:3]) == 2

    plan = doc.commitments[0]
    assert plan.pricing.commitment_id == "sp-0123456789abcdef" and plan.cost.amount == 108

    frame = doc.to_frame()
    assert frame.height == 9 and {"kind", "service_name", "cost", "detail"} <= set(frame.columns)
    assert len(frame.columns) < 40 < len(doc.to_frame("full").columns)
    assert doc.to_dicts()[0]["cost"] == {"amount": "207.36", "currency": "USD"}
    assert len(result.usage) == 5 and result.confidence == "high"


def test_a_wrong_setting_is_an_error_but_no_setting_is_not(monkeypatch, tmp_path):
    import pytest

    from invoice_parser import Parser
    from invoice_parser.llm import LLMError

    for name in ("INVOICE_PARSER_PROVIDER", "INVOICE_PARSER_MODEL", "INVOICE_PARSER_LLM_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here

    assert Parser()._llm is None  # nothing set up: known formats still parse
    with pytest.raises(LLMError, match="unknown provider"):
        Parser(provider="cluade")
    with pytest.raises(LLMError, match="model name"):
        Parser(provider="openai", api_key="k")
    with pytest.raises(LLMError, match="not set"):
        Parser(provider="vllm")  # asked for by name, so its missing endpoint is an error


def test_every_setting_can_be_passed_without_touching_the_environment(monkeypatch, tmp_path):
    import zipfile

    from invoice_parser import Parser

    for name in (
        "INVOICE_PARSER_PROVIDER", "INVOICE_PARSER_MODEL", "INVOICE_PARSER_LLM_URL",
        "INVOICE_PARSER_LLM_KEY", "INVOICE_PARSER_EFFORT", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here

    claude = Parser(
        provider="claude", api_key="from-secrets", model="claude-haiku-5-5", effort="medium"
    )
    assert claude._llm._client.api_key == "from-secrets"
    assert (claude._llm.model, claude._llm.effort) == ("claude-haiku-5-5", "medium")

    openai = Parser(provider="openai", api_key="from-secrets", model="some-model")
    assert openai._llm._client.api_key == "from-secrets"

    hosted = Parser(provider="vllm", base_url="https://gpu.example", api_key="from-secrets")
    assert str(hosted._llm._client.base_url).startswith("https://gpu.example/v1")
    assert hosted._llm._client.api_key == "from-secrets"

    # what is passed wins over the environment
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("INVOICE_PARSER_MODEL", "env-model")
    both = Parser(provider="claude", api_key="passed", model="claude-haiku-5-5")
    assert (both._llm._client.api_key, both._llm.model) == ("passed", "claude-haiku-5-5")

    # the unpacking limits too
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(5):
            zf.writestr(f"f{i}.csv", "a,b\n1,2\n")
    result = Parser(use_model=False, max_archive_members=3).parse_sync(archive)
    doc = result[0]
    assert (doc.kind, doc.confidence) == ("unreadable", "low")
    assert any("more than 3 files inside" in issue for issue in doc.issues)
