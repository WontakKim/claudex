"""Tests for the Grok Responses client and its payload sanitizer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from claudex_gateway.grok_auth import GrokCredentials
from claudex_gateway.grok_client import (
    GROK_MODELS_URL,
    GROK_RESPONSES_URL,
    GrokClient,
    GrokUpstreamError,
    sanitize_grok_payload,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "grok-4.5",
        "instructions": "",
        "input": [],
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "session-1",
    }
    payload.update(overrides)
    return payload


class TestSanitizeGrokPayload:
    def test_drops_unsupported_fields(self) -> None:
        payload = _payload(
            previous_response_id="resp_1",
            prompt_cache_retention="24h",
            safety_identifier="safe",
            stream_options={"include_usage": True},
            stop=["END"],
        )

        sanitized = sanitize_grok_payload(payload, "grok-4.5")

        for field in (
            "previous_response_id",
            "prompt_cache_retention",
            "safety_identifier",
            "stream_options",
            "stop",
        ):
            assert field not in sanitized
        # Everything else passes through untouched.
        assert sanitized["store"] is False
        assert sanitized["include"] == ["reasoning.encrypted_content"]
        assert sanitized["prompt_cache_key"] == "session-1"

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_thinking_model_keeps_clamped_effort(self, effort: str, expected: str) -> None:
        sanitized = sanitize_grok_payload(
            _payload(reasoning={"effort": effort, "summary": "auto"}), "grok-4.5"
        )
        assert sanitized["reasoning"] == {"effort": expected, "summary": "auto"}

    @pytest.mark.parametrize(
        "model", ["grok-composer-2.5-fast", "grok-build-0.1", "grok-9-unreleased"]
    )
    def test_non_thinking_model_drops_reasoning(self, model: str) -> None:
        assert "reasoning" not in sanitize_grok_payload(_payload(), model)

    @pytest.mark.parametrize(
        "model",
        ["grok-4.5", "grok-4.3", "grok-3-mini", "grok-3-mini-fast", "grok-4.20-multi-agent-0309"],
    )
    def test_registry_thinking_models_keep_reasoning(self, model: str) -> None:
        assert "reasoning" in sanitize_grok_payload(_payload(), model)


class _FakeAuthManager:
    def __init__(self) -> None:
        self.force_refresh_calls = 0

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        if force_refresh:
            self.force_refresh_calls += 1
        return GrokCredentials(access_token="grok-token-1", email=None)


def _sse(events: list[dict[str, Any]]) -> bytes:
    chunks = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return chunks + b"data: [DONE]\n\n"


async def _collect(client: GrokClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event async for event in client.stream_responses(payload, "session-1")]


def test_stream_responses_sends_grok_headers_and_parses_events() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=b'ignored-comment: hi\n'
            + _sse([{"type": "response.created", "response": {"id": "r1"}}]),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await _collect(GrokClient(_FakeAuthManager(), http_client), {"model": "grok-4.5"})

    events = asyncio.run(scenario())

    assert events == [{"type": "response.created", "response": {"id": "r1"}}]
    (request,) = captured
    assert str(request.url) == GROK_RESPONSES_URL
    assert request.headers["authorization"] == "Bearer grok-token-1"
    assert request.headers["x-xai-token-auth"] == "xai-grok-cli"
    assert request.headers["x-grok-client-version"]
    assert request.headers["user-agent"].startswith("xai-grok-workspace/")
    assert request.headers["x-grok-conv-id"] == "session-1"
    assert request.headers["accept"] == "text/event-stream"


def test_stream_responses_retries_once_with_fresh_credentials_on_401() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, text="expired")
        return httpx.Response(200, content=_sse([{"type": "response.created", "response": {}}]))

    auth_manager = _FakeAuthManager()

    async def scenario() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await _collect(GrokClient(auth_manager, http_client), {})

    events = asyncio.run(scenario())

    assert len(events) == 1
    assert calls["n"] == 2
    assert auth_manager.force_refresh_calls == 1


def test_stream_responses_raises_upstream_error_on_non_401() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = GrokClient(_FakeAuthManager(), http_client)
            async for _event in client.stream_responses({}, "session-1"):
                pass

    with pytest.raises(GrokUpstreamError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 500
    assert exc_info.value.body == "boom"


def test_list_models_returns_catalog_ids() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == GROK_MODELS_URL
        assert request.headers["accept"] == "application/json"
        assert request.headers["authorization"] == "Bearer grok-token-1"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "grok-4.5", "object": "model"},
                    {"id": "grok-4.3", "object": "model"},
                    {"no_id": True},
                ],
            },
        )

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await GrokClient(_FakeAuthManager(), http_client).list_models()

    assert asyncio.run(scenario()) == ["grok-4.5", "grok-4.3"]


def test_list_models_raises_on_upstream_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="expired")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await GrokClient(_FakeAuthManager(), http_client).list_models()

    with pytest.raises(GrokUpstreamError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 401
