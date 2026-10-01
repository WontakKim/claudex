"""Integration guards for prompt counting through registered Responses adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

import claudex.relay.endpoints as relay_endpoints
import claudex.server as server
import claudex.translate.context_overflow as context_overflow
from claudex import compaction
from claudex.config import GatewayConfig, OpenAICompatibleProvider
from claudex.providers.backends import ResponsesBackend
from claudex.providers.codex_client import CodexClient
from claudex.providers.grok_client import GrokClient
from claudex.providers.kimi_client import KimiClient
from claudex.providers.openai_compatible_client import OpenAICompatibleClient
from claudex.upstream_errors import UpstreamError


@pytest.fixture(params=[
    pytest.param((CodexClient, "codex", "gpt-5.5", None, False), id="codex-standard"),
    pytest.param((CodexClient, "codex", "gpt-5.5", "fast", True), id="codex-fast-supported"),
    pytest.param((CodexClient, "codex", "gpt-5.5", "fast", False), id="codex-fast-unsupported"),
    pytest.param((GrokClient, "grok", "grok-4.5", None, False), id="grok-thinking"),
    pytest.param((GrokClient, "grok", "grok-composer-2.5-fast", None, False), id="grok-non-thinking"),
    pytest.param((OpenAICompatibleClient, "custom", "gpt-5.5", None, False), id="custom-responses"),
])
def counting_gateway(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[SimpleNamespace]:
    client_class, provider, model, tier, supports_fast = request.param
    provider_config = OpenAICompatibleProvider(
        wire_api="responses", base_url="https://models.example/v1", api_key="test-key"
    )
    config = GatewayConfig(
        model_map={"opus": f"{provider}:{model}"},
        custom_providers={"custom": provider_config},
        compaction_model="claude:claude-sonnet-5",
        codex_service_tier=tier,
        reasoning_effort_override="max",
    )
    counted_prompts: list[dict[str, Any]] = []
    overflow_prompts: list[dict[str, Any]] = []
    sent_payloads: list[dict[str, Any]] = []
    catalog_calls: list[str] = []
    tier_calls: list[str] = []
    estimate_tokens = context_overflow.estimate_overflow_prompt_tokens

    def record_count(body: dict[str, Any], **kwargs: Any) -> str:
        counted_prompts.append(deepcopy(body))
        return json.dumps(body, **kwargs)

    def record_overflow(body: dict[str, Any]) -> int:
        overflow_prompts.append(deepcopy(body))
        return estimate_tokens(body)

    async def context_window(self: Any, target: str) -> int:
        catalog_calls.append(target)
        return 1_000_000

    async def supports_fast_tier(self: Any, target: str) -> bool:
        tier_calls.append(target)
        return supports_fast

    async def stream_responses(
        self: Any, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        sent_payloads.append(deepcopy(payload))
        yield {"type": "response.created", "response": {"id": "resp_counting", "model": model}}
        raise UpstreamError(400, json.dumps({"error": {
            "code": "context_length_exceeded", "message": "Context window exceeded."
        }}))

    def reject_network(request: httpx.Request) -> httpx.Response:
        pytest.fail("Prompt-counting integration must not perform network I/O")

    async def reject_auth(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Prompt-counting integration must not read or refresh credentials")

    monkeypatch.setattr(relay_endpoints, "json", SimpleNamespace(
        loads=json.loads, dumps=record_count, JSONDecodeError=json.JSONDecodeError
    ))
    monkeypatch.setattr(context_overflow, "estimate_overflow_prompt_tokens", record_overflow)
    monkeypatch.setattr(client_class, "context_window", context_window)
    monkeypatch.setattr(client_class, "stream_responses", stream_responses)
    monkeypatch.setattr(CodexClient, "supports_fast_tier", supports_fast_tier)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(reject_network))
    auth = SimpleNamespace(get_credentials=reject_auth)
    codex = CodexClient(auth, http_client)
    kimi = KimiClient(auth, http_client)
    grok = GrokClient(auth, http_client)
    custom = OpenAICompatibleClient("custom", provider_config, http_client)
    app = server.create_app(config)
    app.state.config = config
    app.state.http_client = http_client
    app.state.compaction_last_reroute = None
    app.state.route_backends = server._assemble_route_backends(
        app, config, codex, kimi, grok, {"custom": custom}
    )
    assert {name for name, backend in app.state.route_backends.items()
            if isinstance(backend, ResponsesBackend)} == {"codex", "grok", "custom"}
    assert type(app.state.route_backends[provider].transport) is client_class
    client = TestClient(app)
    try:
        yield SimpleNamespace(
            client=client, provider=provider, model=model, tier=tier,
            supports_fast=supports_fast, counted=counted_prompts,
            overflow=overflow_prompts, sent=sent_payloads,
            catalog_calls=catalog_calls, tier_calls=tier_calls,
        )
    finally:
        client.close()
        asyncio.run(http_client.aclose())


def _toolsearch_compaction_request() -> dict[str, Any]:
    return {
        "model": "claude-opus-4-6", "max_tokens": 64,
        "system": "Use the available tools to inspect the project.",
        "tools": [
            {"name": "ToolSearch", "input_schema": {"type": "object"}},
            {"name": "DeferredToolPlaceholder", "defer_loading": True,
             "description": "Reserved placeholder that keeps deferred tool loading active; never call this tool.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "lookup", "defer_loading": True,
             "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}},
        ],
        "messages": [
            {"role": "user", "content": "Find the lookup tool."},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "search", "name": "ToolSearch", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "search", "content": [
                    {"type": "tool_reference", "tool_name": "lookup"}
                ]}
            ]},
            {"role": "system", "content": [
                {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": {
                    "name": "inspect", "description": "Inspect a project file.",
                    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }}}
            ]},
            {"role": "assistant", "content": "The project tools are ready."},
            {"role": "user", "content": (
                f"{compaction.SIGNAL_A_PREFIX}\n{compaction.SIGNAL_A_MARKER} of this session."
            )},
        ],
    }


def _assert_counted_prompts_match(gateway: SimpleNamespace) -> None:
    body = _toolsearch_compaction_request()
    headers = {"anthropic-beta": "inline-tools-2026-09-15"}
    counted = gateway.client.post("/v1/messages/count_tokens", json=body, headers=headers)
    response = gateway.client.post("/v1/messages", json=body, headers=headers)
    assert counted.status_code == 200, counted.text
    assert response.status_code == 400, response.text
    assert "prompt is too long" in response.json()["error"]["message"]
    [count_prompt] = gateway.counted
    compaction_prompt, send_overflow_prompt = gateway.overflow
    [payload] = gateway.sent
    sent_prompt = {key: payload[key] for key in ("instructions", "input", "tools") if key in payload}
    assert count_prompt == compaction_prompt == send_overflow_prompt == sent_prompt, (
        "Counted prompt content diverged across count_tokens, compaction, and send/overflow"
    )
    assert set(count_prompt) == {"instructions", "input", "tools"}
    assert [tool["name"] for tool in count_prompt["tools"]] == ["ToolSearch", "lookup", "inspect"]
    assert "tool_reference" in json.dumps(count_prompt["input"])
    assert "tool_addition" in json.dumps(count_prompt["input"])
    assert counted.json()["input_tokens"] == max(len(json.dumps(count_prompt, ensure_ascii=False)) // 4, 1)
    assert gateway.catalog_calls == [gateway.model]
    assert gateway.client.app.state.compaction_last_reroute is None
    assert gateway.tier_calls == ([gateway.model] if gateway.provider == "codex" and gateway.tier == "fast" else [])
    assert payload.get("service_tier") == ("priority" if gateway.provider == "codex" and gateway.supports_fast else None)
    if gateway.provider == "grok":
        if gateway.model == "grok-4.5":
            assert payload["reasoning"]["effort"] == "high"
        else:
            assert "reasoning" not in payload


def test_registered_responses_adapters_count_the_same_toolsearch_prompt(
    counting_gateway: SimpleNamespace,
) -> None:
    _assert_counted_prompts_match(counting_gateway)


@pytest.mark.parametrize("adapter,field", [
    pytest.param("send", "instructions", id="send-instructions-drift"),
    pytest.param("probe", "input", id="probe-input-drift"),
    pytest.param("send", "tools", id="send-tools-drift"),
])
def test_prompt_counting_guard_detects_adapter_drift(
    counting_gateway: SimpleNamespace, adapter: str, field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backends = counting_gateway.client.app.state.route_backends
    backend = backends[counting_gateway.provider]

    def change_prompt(payload: dict[str, Any]) -> dict[str, Any]:
        changed = deepcopy(payload)
        if field == "instructions":
            changed[field] += "\nAdapter-only instruction."
        elif field == "input":
            changed[field].append({"role": "user", "content": [
                {"type": "input_text", "text": "Adapter-only input."}
            ]})
        else:
            changed[field][0]["description"] = "Adapter-only description."
        return changed

    async def divergent_send(payload: dict[str, Any], model: str) -> dict[str, Any]:
        return change_prompt(await backend.adapt_payload(payload, model))

    def divergent_probe(payload: dict[str, Any], model: str) -> dict[str, Any]:
        return change_prompt(backend.adapt_probe_payload(payload, model))

    replacement = replace(backend, **{
        "adapt_payload" if adapter == "send" else "adapt_probe_payload":
            divergent_send if adapter == "send" else divergent_probe
    })
    monkeypatch.setitem(backends, counting_gateway.provider, replacement)
    with pytest.raises(AssertionError, match="Counted prompt content diverged"):
        _assert_counted_prompts_match(counting_gateway)
