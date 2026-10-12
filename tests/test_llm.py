"""The provider clients, exercised against mock HTTP servers: no network, no keys."""

import json

import anthropic
import httpx2
import openai
import pytest

from invoice_parser.freeform.describe import DescribedList
from invoice_parser.freeform.resources import ResourceList
from invoice_parser.freeform.tagging import DocMeta, TagList
from invoice_parser.llm import (
    ClaudeClient,
    LLMError,
    OpenAIClient,
    Truncated,
    VLLMClient,
    create_llm,
    register_provider,
    strict_schema,
)
from invoice_parser.tabular.plan import TablePlanOut

ANSWER = {"rows": [{"line": 1, "kind": "item", "label": "Compute", "amount": "5.00"}]}
SCHEMAS = [TagList, DocMeta, TablePlanOut, DescribedList, ResourceList]


def claude_with(handler, **options) -> ClaudeClient:
    client = ClaudeClient(api_key="test", **options)
    client._client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return client


def claude_reply(body, text, stop_reason="end_turn"):
    return httpx2.Response(
        200,
        json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 3},
        },
    )  # fmt: skip


def openai_with(handler, cls=OpenAIClient, **options):
    client = cls(**options)
    client._client = openai.OpenAI(
        api_key="test",
        base_url="http://mock.invalid/v1",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return client


def openai_reply(text, finish_reason="stop", refusal=None):
    return httpx2.Response(
        200,
        json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{
                "index": 0, "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text, "refusal": refusal},
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        },
    )  # fmt: skip


def test_claude_request_and_reply():
    seen = {}

    def handler(request):
        seen["headers"], seen["body"] = dict(request.headers), json.loads(request.content)
        return claude_reply(seen["body"], json.dumps(ANSWER))

    client = claude_with(handler)
    out = client.json(
        system="SYS", user="USER", schema=TagList, tag="line_tags", max_tokens=8000, images=[b"png"]
    )
    assert out.rows[0].label == "Compute" and out.rows[0].amount == "5.00"
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 16000
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert [b["type"] for b in body["messages"][0]["content"]] == ["image", "text"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body and "thinking" not in body
    # a declined request is re-run on a fallback model by the server
    assert body["fallbacks"] == "default"
    assert seen["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    usage = client.usage()
    assert usage["provider"] == "claude" and usage["total"]["prompt_tokens"] == 14


def test_claude_accepts_every_schema_the_parser_uses():
    sent = []

    def handler(request):
        body = json.loads(request.content)
        sent.append(body["output_config"]["format"]["schema"])
        return claude_reply(body, "{}")

    client = claude_with(handler)
    for schema in SCHEMAS:
        with pytest.raises(LLMError):  # "{}" matches none of them; the request itself is fine
            client.json(system="S", user="U", schema=schema, tag="t")
    assert len(sent) == len(SCHEMAS)
    assert all(s["additionalProperties"] is False for s in sent)


def test_claude_truncation_refusal_and_models_without_fallback():
    def cut(request):
        return claude_reply(json.loads(request.content), '{"rows": [', stop_reason="max_tokens")

    with pytest.raises(Truncated):
        claude_with(cut).json(system="S", user="U", schema=TagList, tag="t")

    def refuse(request):
        return claude_reply(json.loads(request.content), "", stop_reason="refusal")

    with pytest.raises(LLMError, match="declined"):
        claude_with(refuse).text(system="S", user="U", tag="t")

    seen = {}

    def handler(request):
        seen["headers"], seen["body"] = dict(request.headers), json.loads(request.content)
        return claude_reply(seen["body"], "hello")

    client = claude_with(handler, model="claude-haiku-5-5", effort=None)
    assert client.text(system="S", user="U", tag="t") == "hello"
    assert "fallbacks" not in seen["body"] and "anthropic-beta" not in seen["headers"]
    assert "output_config" not in seen["body"]


def test_openai_request_uses_strict_schema():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return openai_reply(json.dumps(ANSWER))

    client = openai_with(handler, model="some-model", api_key="test")
    out = client.json(system="SYS", user="USER", schema=TagList, tag="t", max_tokens=8000)
    assert out.rows[0].label == "Compute"
    body = seen["body"]
    assert body["max_completion_tokens"] == 16000
    assert "max_tokens" not in body and "temperature" not in body
    fmt = body["response_format"]["json_schema"]
    assert fmt["strict"] is True and fmt["schema"]["additionalProperties"] is False
    tag = fmt["schema"]["$defs"]["Tag"]
    assert set(tag["required"]) == set(tag["properties"]) and tag["additionalProperties"] is False

    def refuse(request):
        return openai_reply(None, refusal="no")

    with pytest.raises(LLMError, match="declined"):
        openai_with(refuse, model="m", api_key="test").text(system="S", user="U", tag="t")

    def cut(request):
        return openai_reply('{"rows": [', finish_reason="length")

    with pytest.raises(Truncated):
        openai_with(cut, model="m", api_key="test").json(
            system="S", user="U", schema=TagList, tag="t"
        )


def test_strict_schema_closes_every_object():
    for schema in SCHEMAS:

        def check(node):
            if isinstance(node, dict):
                if "properties" in node:
                    assert node["additionalProperties"] is False
                    assert set(node["required"]) == set(node["properties"])
                assert "default" not in node
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)

        check(strict_schema(schema))


def test_vllm_request_is_unchanged_and_retries_server_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx2.Response(503, json={"error": "starting"})
        return openai_reply(json.dumps(ANSWER))

    client = openai_with(handler, cls=VLLMClient, base_url="http://gpu.example", api_key="k")
    out = client.json(system="SYS", user="USER", schema=TagList, tag="t", max_tokens=8000)
    assert out.rows[0].label == "Compute" and len(calls) == 2
    body = calls[1]
    assert body["temperature"] == 0 and body["max_tokens"] == 8000
    assert "strict" not in body["response_format"]["json_schema"]


def test_saved_answers_are_kept_apart_per_provider_and_reused(tmp_path):
    hits = {"claude": 0, "openai": 0}

    def claude_handler(request):
        hits["claude"] += 1
        return claude_reply(json.loads(request.content), "from claude")

    def openai_handler(request):
        hits["openai"] += 1
        return openai_reply("from openai")

    a = claude_with(claude_handler, cache=tmp_path)
    b = openai_with(openai_handler, model="m", api_key="test", cache=tmp_path)
    for _ in range(2):
        assert a.text(system="S", user="U", tag="t") == "from claude"
        assert b.text(system="S", user="U", tag="t") == "from openai"
    assert hits == {"claude": 1, "openai": 1}
    assert a.usage()["total"]["cached"] == 1

    a.offline = True
    assert a.text(system="S", user="U", tag="t") == "from claude"
    with pytest.raises(LLMError, match="offline"):
        a.text(system="S", user="something new", tag="t")


def test_choosing_a_provider(monkeypatch):
    for name in ("INVOICE_PARSER_PROVIDER", "INVOICE_PARSER_MODEL", "INVOICE_PARSER_LLM_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir("/")  # no .env here
    assert create_llm("claude", api_key="k").provider == "claude"
    assert create_llm("claude", api_key="k", model="claude-haiku-5-5").model == "claude-haiku-5-5"
    assert create_llm("openai", api_key="k", model="m").provider == "openai"
    assert create_llm("vllm", base_url="http://gpu.example", api_key="k").provider == "vllm"
    with pytest.raises(LLMError, match="model name"):
        create_llm("openai", api_key="k")
    with pytest.raises(LLMError, match="not set"):
        create_llm("vllm")
    with pytest.raises(LLMError, match="unknown provider"):
        create_llm("nope")
    monkeypatch.setenv("INVOICE_PARSER_PROVIDER", "claude")
    assert create_llm(api_key="k").provider == "claude"

    class Mine:
        provider, model, offline = "mine", "x", False

    register_provider("mine", lambda **kw: Mine())
    assert create_llm("mine").provider == "mine"


def test_an_idle_client_holds_no_thread(monkeypatch):
    import threading
    import time as real_time

    monkeypatch.setattr("invoice_parser.llm.HEARTBEAT_SECONDS", 0.05)
    before = threading.active_count()
    clients = [
        openai_with(lambda r: openai_reply("ok"), model="m", api_key="test") for _ in range(5)
    ]
    assert threading.active_count() == before  # creating a client starts nothing
    for client in clients:
        assert client.text(system="S", user="U", tag="t") == "ok"
    deadline = real_time.time() + 3
    while threading.active_count() > before and real_time.time() < deadline:
        real_time.sleep(0.05)
    assert threading.active_count() == before  # and the notice threads end once idle
    assert clients[0].text(system="S", user="again", tag="t") == "ok"  # and start again on use
