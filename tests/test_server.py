"""Integration tests for the gateway HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import struct
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.testclient import TestClient

import claudex_gateway.server as server
from claudex_gateway import compaction
from claudex_gateway.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex_gateway.config import GatewayConfig
from claudex_gateway.kimi_auth import KimiCredentials
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.grok_auth import GrokCredentials
from claudex_gateway.grok_client import GrokClient, GrokUpstreamError
from claudex_gateway.translate import translate_claude_request_to_codex
from claudex_gateway.translate.codex_to_claude import estimate_overflow_prompt_tokens


class AvailableCodexAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            is_api_key=False,
            account_id="account",
            access_token="codex",
            email="codex@example.com",
        )


class MissingCodexAuthManager(AvailableCodexAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> SimpleNamespace:
        raise server.CodexAuthError("missing Codex credentials")


class FakeCodexClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None


class AvailableKimiAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        return KimiCredentials(
            access_token="kimi-token", device_id="device-1", account="kimi-user-1"
        )


class MissingKimiAuthManager(AvailableKimiAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        raise server.KimiAuthError("no Kimi credentials; run `kimi login` first")


class FakeKimiClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class AvailableGrokAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        return GrokCredentials(access_token="grok-token", email="user@example.com")


class MissingGrokAuthManager(AvailableGrokAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        raise server.GrokAuthError("no Grok credentials; run `grok login` first")


class FakeGrokClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None


def _create_test_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: GatewayConfig | None = None,
    codex_auth: type = AvailableCodexAuthManager,
    codex_client: type = FakeCodexClient,
    kimi_auth: type = AvailableKimiAuthManager,
    kimi_client: type = FakeKimiClient,
    grok_auth: type = AvailableGrokAuthManager,
    grok_client: type = FakeGrokClient,
    base_url: str = "http://testserver",
) -> TestClient:
    monkeypatch.setattr(server, "CodexAuthManager", codex_auth)
    monkeypatch.setattr(server, "CodexClient", codex_client)
    monkeypatch.setattr(server, "KimiAuthManager", kimi_auth)
    monkeypatch.setattr(server, "KimiClient", kimi_client)
    monkeypatch.setattr(server, "GrokAuthManager", grok_auth)
    monkeypatch.setattr(server, "GrokClient", grok_client)
    return TestClient(server.create_app(config or GatewayConfig()), base_url=base_url)


def test_messages_routes_enforce_local_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(monkeypatch, config=config) as client:
        messages = client.post("/v1/messages", json={"messages": []})
        count_tokens = client.post("/v1/messages/count_tokens", json={"messages": []})

    for response in (messages, count_tokens):
        assert response.status_code == 401
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "authentication_error"


def test_removed_responses_direction_routes_return_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        assert client.post("/v1/responses", json={"input": "Hello"}).status_code == 404
        assert client.get("/v1/models").status_code == 404


def test_health_reports_ok_with_codex_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "providers": {
            "codex": {
                "status": "ok",
                "auth_mode": "chatgpt",
                "account": "account",
                "email": "codex@example.com",
            },
            "kimi": {"status": "ok", "required": False, "account": "kimi-user-1"},
            "grok": {
                "status": "ok",
                "required": False,
                "auth_mode": "oauth",
                "account": "user@example.com",
            },
        },
    }


def test_health_reports_error_without_codex_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch, codex_auth=MissingCodexAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["codex"]["status"] == "error"


def test_health_stays_ok_without_kimi_credentials_when_map_has_no_kimi_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch, kimi_auth=MissingKimiAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is False


def test_health_reports_error_without_kimi_credentials_when_map_routes_to_kimi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "kimi:k2.5"})
    with _create_test_client(
        monkeypatch, config=config, kimi_auth=MissingKimiAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is True


def test_health_stays_ok_without_grok_credentials_when_map_has_no_grok_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch, grok_auth=MissingGrokAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is False


def test_health_reports_error_without_grok_credentials_when_map_routes_to_grok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "grok:grok-4.5"})
    with _create_test_client(
        monkeypatch, config=config, grok_auth=MissingGrokAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is True


def _upstream_error(status_code: int, error: dict) -> CodexUpstreamError:
    return CodexUpstreamError(status_code, json.dumps({"error": error}))


# Mirrors the Claude Code client's own overflow-message parser: it extracts
# the actual/limit token counts and trims exactly `actual - limit` leading
# tokens before retrying compaction.
_CLIENT_OVERFLOW_NUMBERS_RE = re.compile(
    r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", re.IGNORECASE
)


def _overflow_error_body(message: str) -> str:
    return json.dumps({"error": {"code": "context_length_exceeded", "message": message}})


def test_context_overflow_http_error_is_rewritten_for_claude_compaction() -> None:
    status_code, body = server._upstream_error_to_claude(
        _upstream_error(
            400,
            {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "Your input exceeds the context window of this model.",
            },
        )
    )
    assert status_code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == (
        "prompt is too long: Your input exceeds the context window of this model."
    )


def test_context_overflow_forces_400_regardless_of_upstream_status() -> None:
    status_code, body = server._upstream_error_to_claude(
        _upstream_error(413, {"message": "Request exceeds the maximum context length."})
    )
    assert status_code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("prompt is too long: ")


def test_non_overflow_error_passes_through_unchanged() -> None:
    status_code, body = server._upstream_error_to_claude(
        _upstream_error(429, {"type": "rate_limit_error", "message": "slow down"})
    )
    assert status_code == 429
    assert body["error"] == {"type": "rate_limit_error", "message": "slow down"}


# --- /v1/messages routing: mapped models -> Codex, everything else -> Anthropic ---


class StubCodexClient:
    """Records translated payloads, then fails with a recognizable error."""

    def __init__(
        self,
        context_window: int | None = None,
        error: CodexUpstreamError | None = None,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.context_window_calls: list[str] = []
        self._context_window = context_window
        self._error = error or CodexUpstreamError(503, "stub codex upstream")

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return self._context_window

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        self.payloads.append(payload)
        raise self._error
        yield  # unreachable; makes this an async generator like the real client


def _gateway(
    config: GatewayConfig,
    anthropic_handler,
    kimi_handler=None,
    kimi_auth: Any | None = None,
    grok_client: Any | None = None,
    codex_context_window: int | None = None,
    codex_error: CodexUpstreamError | None = None,
) -> tuple[TestClient, StubCodexClient]:
    app = server.create_app(config)
    # The lifespan requires real Codex credentials, so set the state directly
    # instead of entering the TestClient context manager.
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    stub = StubCodexClient(codex_context_window, codex_error)
    app.state.codex_client = stub
    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(anthropic_handler)
    )
    if kimi_handler is not None or kimi_auth is not None:
        app.state.kimi_client = KimiClient(
            kimi_auth or AvailableKimiAuthManager(),
            httpx.AsyncClient(
                transport=httpx.MockTransport(
                    kimi_handler or (lambda request: httpx.Response(500))
                )
            ),
        )
    if grok_client is not None:
        app.state.grok_client = grok_client
    return TestClient(app), stub


def _message_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_passthrough_forwards_unmapped_model_to_anthropic() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"id": "msg_1", "model": "claude-fable-5"},
            headers={"request-id": "req_abc"},
        )

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(config, handler)

    response = client.post(
        "/v1/messages?beta=true",
        content=json.dumps(_message_body("claude-fable-5")),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-oat01-test",
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "anthropic-version": "2023-06-01",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"id": "msg_1", "model": "claude-fable-5"}
    assert response.headers["request-id"] == "req_abc"
    assert stub.payloads == []
    (upstream,) = captured
    assert str(upstream.url) == "https://api.anthropic.com/v1/messages?beta=true"
    assert upstream.headers["host"] == "api.anthropic.com"
    assert upstream.headers["authorization"] == "Bearer sk-ant-oat01-test"
    assert upstream.headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
    assert json.loads(upstream.content) == _message_body("claude-fable-5")


def test_passthrough_relays_streaming_response_verbatim() -> None:
    sse_body = b'event: message_start\ndata: {"type": "message_start"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}
        )

    config = GatewayConfig()
    client, _ = _gateway(config, handler)

    response = client.post("/v1/messages", json=_message_body("claude-fable-5"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.content == sse_body


def test_passthrough_returns_502_when_anthropic_is_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    config = GatewayConfig()
    client, _ = _gateway(config, handler)

    response = client.post("/v1/messages", json=_message_body("claude-fable-5"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


class _AbortingByteStream(httpx.AsyncByteStream):
    """Yields one chunk, then dies like a reset upstream connection."""

    def __init__(self, first_chunk: bytes) -> None:
        self._first_chunk = first_chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first_chunk
        raise httpx.ReadError("connection reset")


def test_passthrough_sse_abort_emits_error_event() -> None:
    first_event = b'event: message_start\ndata: {"type": "message_start"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_AbortingByteStream(first_event),
            headers={"content-type": "text/event-stream"},
        )

    client, _ = _gateway(GatewayConfig(), handler)

    response = client.post("/v1/messages", json=_message_body("claude-fable-5"))

    assert response.status_code == 200
    assert response.content.startswith(first_event)
    tail = response.content[len(first_event) :].decode()
    assert "event: error" in tail
    payload = json.loads(tail.rsplit("data: ", 1)[1].strip())
    assert payload["error"]["type"] == "api_error"
    assert "connection reset" in payload["error"]["message"]


def test_passthrough_json_abort_truncates_without_error_event() -> None:
    partial_body = b'{"id": "msg_1", '

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_AbortingByteStream(partial_body),
            headers={"content-type": "application/json"},
        )

    client, _ = _gateway(GatewayConfig(), handler)

    response = client.post("/v1/messages", json=_message_body("claude-fable-5"))

    assert response.status_code == 200
    assert response.content == partial_body


def test_mapped_model_routes_to_codex() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={})

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(config, handler)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "stub codex upstream"
    assert stub.payloads[0]["model"] == "gpt-5.6-sol"
    assert captured == []


def test_codex_pre_stream_overflow_uses_catalog_context_window() -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(
        config,
        _failing_anthropic_handler,
        codex_context_window=272000,
        codex_error=CodexUpstreamError(
            400,
            _overflow_error_body("Your input exceeds the context window of this model."),
        ),
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 400
    match = _CLIENT_OVERFLOW_NUMBERS_RE.search(response.json()["error"]["message"])
    assert match is not None
    actual, limit = int(match.group(1)), int(match.group(2))
    assert limit == 272000
    assert actual > limit


def test_pre_stream_overflow_without_catalog_window_falls_back_to_legacy_message() -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(
        config,
        _failing_anthropic_handler,
        codex_error=CodexUpstreamError(
            400,
            _overflow_error_body("Your input exceeds the context window of this model."),
        ),
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "prompt is too long: Your input exceeds the context window of this model."
    )
    assert _CLIENT_OVERFLOW_NUMBERS_RE.search(response.json()["error"]["message"]) is None


class MidStreamOverflowCodexClient(FakeCodexClient):
    """Yields one event, then fails with an overflow-shaped upstream error."""

    async def context_window(self, model: str) -> int | None:
        return 272000

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"id": "resp_1", "model": payload["model"]}}
        raise CodexUpstreamError(
            400,
            _overflow_error_body("Your input exceeds the context window of this model."),
        )


def test_mid_stream_overflow_error_carries_catalog_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    body = _message_body("claude-opus-4-6")
    body["stream"] = True
    with _create_test_client(
        monkeypatch, config=config, codex_client=MidStreamOverflowCodexClient
    ) as client:
        response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    events = [event for event in response.text.split("\n\n") if event]
    error_event = next(event for event in events if event.startswith("event: error"))
    payload = json.loads(error_event.split("data: ", 1)[1])
    match = _CLIENT_OVERFLOW_NUMBERS_RE.search(payload["error"]["message"])
    assert match is not None
    actual, limit = int(match.group(1)), int(match.group(2))
    assert limit == 272000
    assert actual > limit


class RecordingMidStreamOverflowCodexClient(FakeCodexClient):
    """Yields one event, then fails with an overflow-shaped upstream error.

    Also records every model passed to `context_window`, so callers can
    prove it was resolved from the mapped `RouteTarget.model`.
    """

    def __init__(self) -> None:
        self.context_window_calls: list[str] = []

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return 272000

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"id": "resp_1", "model": payload["model"]}}
        raise CodexUpstreamError(
            400,
            _overflow_error_body("Your input exceeds the context window of this model."),
        )


def test_non_streaming_mid_stream_overflow_reports_numbers() -> None:
    # T-5's mid-stream overflow coverage only exercised the streaming
    # translation path (_translate_claude_sse); a non-streaming mapped
    # request goes through _aggregate_claude_response's own exception
    # handler instead, which must synthesize the same numeric pair.
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    app = server.create_app(config)
    # The lifespan requires real Codex credentials, so set the state
    # directly instead of entering the TestClient context manager.
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    stub = RecordingMidStreamOverflowCodexClient()
    app.state.codex_client = stub
    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_failing_anthropic_handler)
    )
    client = TestClient(app)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 400
    payload = response.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    match = _CLIENT_OVERFLOW_NUMBERS_RE.search(payload["error"]["message"])
    assert match is not None
    actual, limit = int(match.group(1)), int(match.group(2))
    assert limit == 272000
    assert actual > limit
    assert stub.context_window_calls == ["gpt-5.6-sol"]


def test_context_window_cache_is_reused_across_requests() -> None:
    # Proves the relay reuses the lifespan-owned client and its
    # ContextWindowCache instead of constructing a fresh one per request:
    # two mapped requests through the same app fetch the catalog exactly
    # once (within the cache's TTL) while still hitting the upstream model
    # endpoint once per request.
    calls = {"catalog": 0, "responses": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(CODEX_MODELS_URL):
            calls["catalog"] += 1
            return httpx.Response(
                200, json={"models": [{"slug": "gpt-5.6-sol", "context_window": 272000}]}
            )
        if str(request.url) == CODEX_RESPONSES_URL:
            calls["responses"] += 1
            return httpx.Response(
                200,
                content=(
                    b'data: {"type": "response.created", "response": {"id": "resp_1"}}\n\n'
                    b'data: {"type": "response.completed", '
                    b'"response": {"id": "resp_1", "usage": {}, "output": []}}\n\n'
                    b"data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"unexpected request to {request.url}")

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    app = server.create_app(config)
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    app.state.codex_client = CodexClient(
        AvailableCodexAuthManager(), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_failing_anthropic_handler)
    )
    client = TestClient(app)

    first = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))
    second = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["catalog"] == 1
    assert calls["responses"] == 2


def test_mapped_request_without_max_tokens_is_rejected() -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(config, lambda request: httpx.Response(200, json={}))

    body = _message_body("claude-opus-4-6")
    del body["max_tokens"]
    response = client.post("/v1/messages", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in response.json()["error"]["message"]
    assert stub.payloads == []


def test_mapped_request_with_unsupported_document_is_rejected() -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(config, lambda request: httpx.Response(200, json={}))

    body = _message_body("claude-opus-4-6")
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "url", "url": "https://example.com/a.pdf"},
                }
            ],
        }
    ]
    response = client.post("/v1/messages", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "application/pdf" in response.json()["error"]["message"]
    assert stub.payloads == []


def test_invalid_json_passes_through_to_anthropic() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error"}}
        )

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(config, handler)

    response = client.post(
        "/v1/messages", content=b"not-json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 400
    assert captured[0].content == b"not-json"
    assert stub.payloads == []


def test_count_tokens_passes_through_for_unmapped_model() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"input_tokens": 1234})

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(config, handler)

    response = client.post(
        "/v1/messages/count_tokens", json=_message_body("claude-fable-5")
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 1234}
    assert str(captured[0].url) == "https://api.anthropic.com/v1/messages/count_tokens"


def test_count_tokens_estimates_locally_for_mapped_model() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"input_tokens": 1234})

    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(config, handler)

    response = client.post(
        "/v1/messages/count_tokens", json=_message_body("claude-opus-4-6")
    )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0
    assert captured == []


# --- /v1/messages routing: kimi-mapped models relay to the Kimi backend ---


def _kimi_config() -> GatewayConfig:
    return GatewayConfig(model_map={"opus": "kimi:k2.5"})


def _failing_anthropic_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": "unexpected anthropic passthrough"})


def test_kimi_mapped_model_relays_with_model_rewrite() -> None:
    captured: list[httpx.Request] = []

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"id": "msg_1", "type": "message", "model": "k2.5", "content": []},
            headers={"request-id": "req_kimi"},
        )

    client, stub = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post(
        "/v1/messages",
        content=json.dumps(_message_body("claude-opus-4-6")),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-oat01-test",
            "x-api-key": "sk-ant-key",
            "anthropic-beta": "claude-code-20250219",
            "anthropic-version": "2023-06-01",
        },
    )

    assert response.status_code == 200
    assert response.json()["id"] == "msg_1"
    # The gateway reports the requested Claude model, not the Kimi target.
    assert response.json()["model"] == "claude-opus-4-6"
    assert response.headers["request-id"] == "req_kimi"
    assert stub.payloads == []
    (upstream,) = captured
    assert str(upstream.url) == "https://api.kimi.com/coding/v1/messages?beta=true"
    assert upstream.headers["authorization"] == "Bearer kimi-token"
    assert "x-api-key" not in upstream.headers
    assert upstream.headers["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20"
    assert upstream.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(upstream.content)["model"] == "k2.5"


def test_kimi_oauth_beta_is_not_duplicated() -> None:
    seen: dict[str, str] = {}

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        seen["beta"] = request.headers["anthropic-beta"]
        return httpx.Response(200, json={"type": "message", "model": "k2.5"})

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    client.post(
        "/v1/messages",
        json=_message_body("claude-opus-4-6"),
        headers={"anthropic-beta": "oauth-2025-04-20,claude-code-20250219"},
    )

    assert seen["beta"] == "oauth-2025-04-20,claude-code-20250219"


def test_kimi_stream_rewrites_only_message_start() -> None:
    sse_body = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg_1","model":"k2.5"}}\n'
        b"\n"
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
        b'event: message_stop\n'
        b'data: {"type":"message_stop"}\n'
        b"\n"
    )

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}
        )

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    body = _message_body("claude-opus-4-6")
    body["stream"] = True
    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    lines = response.content.decode().split("\n")
    assert lines[0] == "event: message_start"
    message_start = json.loads(lines[1].removeprefix("data: "))
    assert message_start["message"]["model"] == "claude-opus-4-6"
    # Every line outside message_start is relayed byte-identical.
    assert (
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}'
        in lines
    )
    assert 'data: {"type":"message_stop"}' in lines


def test_kimi_stream_yields_complete_sse_events() -> None:
    sse_body = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"model":"k3"}}\n'
        b"\n"
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
        b'event: message_stop\n'
        b'data: {"type":"message_stop"}\n'
    )

    async def scenario() -> list[bytes]:
        response = httpx.Response(200, content=sse_body)
        return [
            chunk
            async for chunk in server._rewrite_kimi_sse(response, "claude-fable-5")
        ]

    chunks = asyncio.run(scenario())

    assert len(chunks) == 3
    assert chunks[0].endswith(b"\n\n")
    assert b'"model": "claude-fable-5"' in chunks[0]
    assert chunks[1] == (
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
    )
    assert chunks[2] == b'event: message_stop\ndata: {"type":"message_stop"}\n'


def test_kimi_stream_cancellation_interrupts_buffered_events() -> None:
    closed = False
    sse_body = b"".join(
        f'event: content_block_delta\ndata: {{"index":{index}}}\n\n'.encode()
        for index in range(20)
    )

    class BufferedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse_body

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def scenario() -> list[bytes]:
        response = httpx.Response(200, stream=BufferedStream())
        chunks: list[bytes] = []
        first_chunk_seen = asyncio.Event()

        async def consume() -> None:
            async for chunk in server._rewrite_kimi_sse(response, "claude-fable-5"):
                chunks.append(chunk)
                first_chunk_seen.set()

        task = asyncio.create_task(consume())
        await first_chunk_seen.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return chunks

    chunks = asyncio.run(scenario())

    assert chunks == [b'event: content_block_delta\ndata: {"index":0}\n\n']
    assert closed is True


def test_kimi_stream_client_reset_stops_closed_socket_writes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream_closed = threading.Event()

    class BurstStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"".join(
                f'event: content_block_delta\ndata: {{"index":{index}}}\n\n'.encode()
                for index in range(20)
            )

        async def aclose(self) -> None:
            upstream_closed.set()

    class BurstKimiClient:
        async def send_messages(
            self, _body: bytes, _headers: dict[str, str]
        ) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BurstStream(),
            )

    config = GatewayConfig(model_map={"fable": "kimi:k3"})
    app = server.create_app(config)
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    app.state.kimi_client = BurstKimiClient()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    uvicorn_server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off", access_log=False)
    )

    def run_server() -> None:
        asyncio.run(uvicorn_server.serve(sockets=[listener]))

    server_thread = threading.Thread(target=run_server, daemon=True)
    caplog.set_level(logging.WARNING, logger="asyncio")
    try:
        server_thread.start()
        startup_deadline = time.monotonic() + 2
        while not uvicorn_server.started and time.monotonic() < startup_deadline:
            time.sleep(0.001)
        assert uvicorn_server.started

        body = json.dumps(
            {
                "model": "claude-fable-5",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode()
        request = (
            b"POST /v1/messages HTTP/1.1\r\n"
            + f"Host: 127.0.0.1:{port}\r\n".encode()
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            client.sendall(request)
            client.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_LINGER,
                struct.pack("ii", 1, 0),
            )

        assert upstream_closed.wait(2)
    finally:
        uvicorn_server.should_exit = True
        server_thread.join(timeout=2)
        listener.close()

    assert not server_thread.is_alive()
    assert not any(
        record.name == "asyncio"
        and record.getMessage() == "socket.send() raised exception."
        for record in caplog.records
    )


def test_kimi_missing_credentials_return_401_with_login_guidance() -> None:
    client, _ = _gateway(
        _kimi_config(), _failing_anthropic_handler, kimi_auth=MissingKimiAuthManager()
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert "kimi login" in response.json()["error"]["message"]


def test_kimi_anthropic_shaped_error_is_relayed_with_status() -> None:
    error_body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=error_body)

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 429
    assert response.json() == error_body


def test_kimi_overflow_shaped_error_is_forwarded_verbatim() -> None:
    """Kimi's relay is untouched by design: no numeric overflow rewrite applies."""
    error_body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Your input exceeds the context window of this model.",
        },
    }

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=error_body)

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 400
    assert response.json() == error_body
    assert "prompt is too long" not in response.json()["error"]["message"]


def test_kimi_persistent_401_blames_gateway_credentials() -> None:
    attempts: list[str] = []

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers["authorization"])
        return httpx.Response(401, json={"error": {"message": "token expired"}})

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    # One retry with force-refreshed credentials, then a gateway-side 401 so
    # Claude Code does not re-auth its own Anthropic session.
    assert len(attempts) == 2
    assert response.status_code == 401
    message = response.json()["error"]["message"]
    assert "token expired" in message
    assert "kimi login" in message


def test_kimi_unreachable_returns_502() -> None:
    def kimi_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert "Kimi" in response.json()["error"]["message"]


def test_kimi_route_skips_codex_only_validation() -> None:
    # Kimi is the Messages authority for its own contract, exactly like the
    # Anthropic passthrough; a missing max_tokens is Kimi's call to reject.
    captured: list[httpx.Request] = []

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"type": "message", "model": "k2.5"})

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    body = _message_body("claude-opus-4-6")
    del body["max_tokens"]
    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    assert len(captured) == 1


def test_count_tokens_forwards_to_kimi_for_kimi_model() -> None:
    captured: list[httpx.Request] = []

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"input_tokens": 42})

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post(
        "/v1/messages/count_tokens", json=_message_body("claude-opus-4-6")
    )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    (upstream,) = captured
    assert str(upstream.url) == (
        "https://api.kimi.com/coding/v1/messages/count_tokens?beta=true"
    )
    assert json.loads(upstream.content)["model"] == "k2.5"


def test_count_tokens_falls_back_to_estimate_when_kimi_fails() -> None:
    def kimi_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    client, _ = _gateway(_kimi_config(), _failing_anthropic_handler, kimi_handler)

    response = client.post(
        "/v1/messages/count_tokens", json=_message_body("claude-opus-4-6")
    )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


# --- /v1/messages routing: grok-mapped models go to the Grok Responses backend ---


class StubGrokClient:
    """Records translated payloads and replays a scripted Responses stream."""

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        context_window: int | None = None,
        error: GrokUpstreamError | None = None,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.events = events
        self._context_window = context_window
        self._error = error or GrokUpstreamError(503, "stub grok upstream")

    async def context_window(self, model: str) -> int | None:
        return self._context_window

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        self.payloads.append(payload)
        if self.events is None:
            raise self._error
        for event in self.events:
            yield event


def _grok_config(target: str = "grok-4.5", **kwargs: Any) -> GatewayConfig:
    return GatewayConfig(model_map={"opus": f"grok:{target}"}, **kwargs)


def test_grok_mapped_model_streams_translated_response() -> None:
    stub = StubGrokClient(
        [
            {"type": "response.created", "response": {"id": "resp_1", "model": "grok-4.5"}},
            {"type": "response.output_text.delta", "delta": "hello"},
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {"input_tokens": 1, "output_tokens": 1}},
            },
        ]
    )
    client, codex_stub = _gateway(_grok_config(), _failing_anthropic_handler, grok_client=stub)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 200
    # The gateway reports the requested Claude model, not the Grok target.
    assert response.json()["model"] == "claude-opus-4-6"
    assert codex_stub.payloads == []
    (payload,) = stub.payloads
    assert payload["model"] == "grok-4.5"
    # No thinking block in the request: the derived medium survives clamping.
    assert payload["reasoning"]["effort"] == "medium"


def test_grok_route_strips_reasoning_for_non_thinking_model() -> None:
    stub = StubGrokClient()
    client, _ = _gateway(
        _grok_config("grok-composer-2.5-fast"), _failing_anthropic_handler, grok_client=stub
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    (payload,) = stub.payloads
    assert "reasoning" not in payload


def test_grok_route_clamps_effort_for_thinking_model() -> None:
    stub = StubGrokClient()
    config = _grok_config("grok-4.5", reasoning_effort_override="max")
    client, _ = _gateway(config, _failing_anthropic_handler, grok_client=stub)

    client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    (payload,) = stub.payloads
    assert payload["reasoning"]["effort"] == "high"


def test_grok_pre_stream_overflow_uses_catalog_context_window() -> None:
    # Each provider resolves its overflow limit from its own catalog: Codex's
    # 272000 test elsewhere must not leak into Grok's 500000 expectation.
    stub = StubGrokClient(
        context_window=500000,
        error=GrokUpstreamError(
            400,
            _overflow_error_body("Your input exceeds the context window of this model."),
        ),
    )
    client, _ = _gateway(_grok_config(), _failing_anthropic_handler, grok_client=stub)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 400
    match = _CLIENT_OVERFLOW_NUMBERS_RE.search(response.json()["error"]["message"])
    assert match is not None
    actual, limit = int(match.group(1)), int(match.group(2))
    assert limit == 500000
    assert actual > limit


# --- /v1/messages routing: compaction reroute trigger (T-5) ---------------

_COMPACTION_RAW_TARGET = "claude:claude-sonnet-5"
_COMPACTION_CANONICAL_TARGET = "claude-sonnet-5"

# A genuine Anthropic credential, so build_reroute_headers has something to
# forward; local_token is unset on every config below, so it never collides
# with _require_local_token's own check.
_ANTHROPIC_CREDENTIAL_HEADERS = {"x-api-key": "sk-ant-real-key"}


class _TrackedByteStream(httpx.AsyncByteStream):
    """Yields `chunks`, optionally raising `error` afterward; records aclose()."""

    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True


def _compaction_signal_text() -> str:
    return (
        f"{compaction.SIGNAL_A_PREFIX} Some filler context so the message resembles a "
        f"real conversation. {compaction.SIGNAL_A_MARKER} some more filler detail."
    )


def _compaction_body(
    model: str, *, max_tokens: int = 16, thinking_block: bool = False
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if thinking_block:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "internal reasoning that must never reach Anthropic",
                        "signature": "sig-from-a-different-backend",
                    },
                    {"type": "text", "text": "visible reply"},
                ],
            }
        )
    messages.append({"role": "user", "content": _compaction_signal_text()})
    return {"model": model, "max_tokens": max_tokens, "messages": messages}


def _compaction_config(model_map: dict[str, str], **kwargs: Any) -> GatewayConfig:
    return GatewayConfig(model_map=model_map, compaction_model=_COMPACTION_RAW_TARGET, **kwargs)


def _pinned_reroute_record_keys() -> set[str]:
    return {
        "outcome",
        "timestamp",
        "target_model",
        "mapped_model",
        "estimated_prompt_tokens",
        "context_window",
        "detail",
    }


def _parse_sse_error(raw: bytes) -> dict[str, Any]:
    """Parse the JSON payload out of a single, parseable `event: error` frame."""
    text = raw.decode()
    assert "event: error" in text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    return json.loads(data_line[len("data:") :].strip())


def test_compaction_reroute_success_returns_anthropic_response_and_records_diagnostics() -> None:
    body = _compaction_body("claude-opus-4-6")
    expected_estimate = estimate_overflow_prompt_tokens(body)
    window = expected_estimate - 1
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream(
        [
            json.dumps(
                {"id": "msg_1", "type": "message", "model": _COMPACTION_CANONICAL_TARGET}
            ).encode()
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, stream=stream, headers={"content-type": "application/json"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "id": "msg_1",
        "type": "message",
        "model": _COMPACTION_CANONICAL_TARGET,
    }
    assert stub.payloads == []  # translation never ran; the reroute short-circuited it
    assert len(captured) == 1
    (upstream,) = captured
    assert str(upstream.url) == "https://api.anthropic.com/v1/messages"
    assert json.loads(upstream.content)["model"] == _COMPACTION_CANONICAL_TARGET
    assert stream.closed is True

    record = client.app.state.compaction_last_reroute
    assert set(record) == _pinned_reroute_record_keys()
    assert record["outcome"] == "rerouted"
    assert record["detail"] is None
    assert record["target_model"] == _COMPACTION_CANONICAL_TARGET
    assert record["mapped_model"] == "codex:gpt-5.1-codex-max"
    assert record["context_window"] == window
    assert record["estimated_prompt_tokens"] == expected_estimate
    assert isinstance(record["timestamp"], str) and record["timestamp"]


def test_compaction_reroute_skipped_without_credentials_falls_back_to_mapped_path() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"unexpected": True})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body)  # no x-api-key, no authorization

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "stub codex upstream"
    # No credential to forward means no Anthropic attempt at all -- there is
    # no httpx.Response to close on this branch, so nothing can leak.
    assert captured == []
    (payload,) = stub.payloads
    assert payload["model"] == "gpt-5.1-codex-max"

    record = client.app.state.compaction_last_reroute
    assert set(record) == _pinned_reroute_record_keys()
    assert record["outcome"] == "skipped_no_credentials"
    assert record["detail"] is None
    assert record["mapped_model"] == "codex:gpt-5.1-codex-max"
    assert record["target_model"] == _COMPACTION_CANONICAL_TARGET
    assert record["context_window"] == window
    assert stub.context_window_calls == ["gpt-5.1-codex-max"]


def test_compaction_reroute_falls_back_on_non_2xx_status() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream([b'{"error": {"message": "invalid x-api-key"}}'])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(401, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert len(captured) == 1  # exactly one Anthropic attempt, no retry
    assert stub.payloads  # fallback ran translation against the mapped backend

    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "http_401"
    assert stream.closed is True


def test_compaction_reroute_falls_back_on_3xx_without_following_redirect() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream([b""])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            307, headers={"location": "https://api.anthropic.com/v1/messages"}, stream=stream
        )

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    # The redirect target is never followed: exactly one outbound request.
    assert len(captured) == 1
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "http_307"
    assert stream.closed is True


def test_compaction_reroute_falls_back_on_connect_error() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "connect_error"


def test_compaction_reroute_falls_back_on_non_connect_transport_error() -> None:
    # httpx.HTTPError is not narrowed to ConnectError: any pre-response
    # transport failure must fall back the same way.
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.WriteError("connection reset while writing the request")

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "connect_error"


def test_compaction_reroute_falls_back_on_read_error() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    stream = _TrackedByteStream([b'{"partial": '], httpx.ReadError("connection reset"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "read_error"
    assert stream.closed is True


def test_compaction_reroute_falls_back_on_invalid_json() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    stream = _TrackedByteStream([b"not valid json"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "invalid_json"
    assert stream.closed is True


def test_compaction_reroute_falls_back_on_nonfinite_json_constants() -> None:
    # json.loads would happily parse NaN into a dict, but Starlette's
    # JSONResponse refuses to serialize it; the reroute must classify such a
    # body as invalid_json and fall back rather than record "rerouted" and
    # then fail the request.
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    stream = _TrackedByteStream([b'{"value": NaN}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "invalid_json"
    assert stream.closed is True


def test_compaction_reroute_falls_back_on_json_scalar_body() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    stream = _TrackedByteStream([b"42"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "invalid_json"
    assert stream.closed is True


def test_compaction_reroute_success_never_calls_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("translation must not run when the reroute succeeds")

    monkeypatch.setattr(server, "translate_claude_request_to_codex", _boom)

    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, json={"id": "msg_1", "type": "message", "model": _COMPACTION_CANONICAL_TARGET}
        )

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert len(captured) == 1
    assert stub.payloads == []


def test_compaction_reroute_fallback_translates_untouched_original_body_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    def recording_translate(
        claude_request: dict[str, Any], upstream_model: str, reasoning_effort_override: str | None
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request, upstream_model, reasoning_effort_override
        )

    monkeypatch.setattr(server, "translate_claude_request_to_codex", recording_translate)

    body = _compaction_body("claude-opus-4-6", thinking_block=True)
    window = estimate_overflow_prompt_tokens(body) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert len(received) == 1
    assert received[0] == body  # the untouched original body, thinking block included
    assert any(
        block.get("type") == "thinking"
        for message in received[0]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
    )
    # One catalog lookup total: the trigger's own lookup is reused by the
    # mapped fallback instead of being repeated.
    assert stub.context_window_calls == ["gpt-5.1-codex-max"]


def test_compaction_trigger_reroutes_grok_mapped_request() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, json={"id": "msg_1", "type": "message", "model": _COMPACTION_CANONICAL_TARGET}
        )

    grok_stub = StubGrokClient(context_window=window)
    config = _compaction_config({"opus": "grok:grok-4.5"})
    client, codex_stub = _gateway(config, handler, grok_client=grok_stub)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert len(captured) == 1
    assert grok_stub.payloads == []
    assert codex_stub.payloads == []
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "rerouted"
    assert record["mapped_model"] == "grok:grok-4.5"
    assert record["target_model"] == _COMPACTION_CANONICAL_TARGET


def test_compaction_trigger_never_engages_for_kimi_mapped_requests() -> None:
    body = _compaction_body("claude-opus-4-6")
    captured: list[httpx.Request] = []

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected anthropic call"})

    kimi_calls: list[httpx.Request] = []

    def kimi_handler(request: httpx.Request) -> httpx.Response:
        kimi_calls.append(request)
        return httpx.Response(
            200, json={"id": "msg_1", "type": "message", "model": "k2.5", "content": []}
        )

    config = _compaction_config({"opus": "kimi:k2.5"})
    client, _ = _gateway(config, anthropic_handler, kimi_handler)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert captured == []
    assert len(kimi_calls) == 1
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_never_engages_for_unmapped_passthrough() -> None:
    body = _compaction_body("claude-fable-5")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_1", "model": "claude-fable-5"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert len(captured) == 1  # the ordinary passthrough call, not a reroute
    assert stub.payloads == []
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_absent_setting_never_runs_signal_a_or_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_calls = {"count": 0}
    estimate_calls = {"count": 0}
    real_is_compaction_request = server.is_compaction_request
    real_estimate = server.estimate_overflow_prompt_tokens

    def counting_is_compaction_request(*args: Any, **kwargs: Any) -> bool:
        signal_calls["count"] += 1
        return real_is_compaction_request(*args, **kwargs)

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(server, "is_compaction_request", counting_is_compaction_request)
    monkeypatch.setattr(server, "estimate_overflow_prompt_tokens", counting_estimate)

    body = _compaction_body("claude-opus-4-6")
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.1-codex-max"})  # compaction_model unset
    client, stub = _gateway(config, _failing_anthropic_handler, codex_context_window=10)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert signal_calls["count"] == 0
    assert estimate_calls["count"] == 0
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_non_signal_body_never_runs_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate_calls = {"count": 0}
    real_estimate = server.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(server, "estimate_overflow_prompt_tokens", counting_estimate)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, _failing_anthropic_handler, codex_context_window=10)

    response = client.post(
        "/v1/messages",
        json=_message_body("claude-opus-4-6"),
        headers=_ANTHROPIC_CREDENTIAL_HEADERS,
    )

    assert response.status_code == 503
    assert estimate_calls["count"] == 0
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_under_window_does_not_reroute() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body) + 1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert captured == []
    assert stub.payloads
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_equal_window_does_not_reroute() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = estimate_overflow_prompt_tokens(body)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert captured == []
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_none_window_does_not_reroute() -> None:
    body = _compaction_body("claude-opus-4-6")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler)  # codex_context_window defaults to None

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert captured == []
    assert client.app.state.compaction_last_reroute is None


@pytest.mark.parametrize("bogus_window", [True, False])
def test_compaction_trigger_rejects_boolean_catalog_window(
    monkeypatch: pytest.MonkeyPatch, bogus_window: bool
) -> None:
    # bool is an int subclass; the trigger's window check must exclude it
    # explicitly rather than trusting a bare isinstance(x, int).
    estimate_calls = {"count": 0}
    real_estimate = server.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(server, "estimate_overflow_prompt_tokens", counting_estimate)

    body = _compaction_body("claude-opus-4-6")
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=bogus_window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert captured == []
    assert estimate_calls["count"] == 0
    assert client.app.state.compaction_last_reroute is None


def test_compaction_reroute_never_reenters_the_model_map_via_the_canonical_target() -> None:
    # The canonical reroute target ("claude-sonnet-5") substring-matches this
    # very map key; the reroute must never re-enter model_map resolution and
    # loop back into the mapped Codex path for its own output.
    body = _compaction_body("claude-sonnet-4-6")
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, json={"id": "msg_1", "type": "message", "model": _COMPACTION_CANONICAL_TARGET}
        )

    config = _compaction_config({"sonnet": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert response.json()["model"] == _COMPACTION_CANONICAL_TARGET
    assert len(captured) == 1
    assert stub.payloads == []


# --- /v1/messages routing: compaction reroute trigger, stream:true (T-6) --
# stream:true shares every pre-commit step with stream:false (T-5 above): an
# HTTP 2xx response is the sole commit boundary, not the first relayed byte.
# Once committed, `_CompactionStreamRelay` owns the upstream response and
# relays `aiter_bytes()` unchanged; these tests exercise both the full
# `/v1/messages` round trip (via `_gateway`/`_TrackedByteStream`) and, for
# the ownership/closure properties, `_CompactionStreamRelay` directly --
# the same style already used above for `_translate_claude_sse`.


def test_compaction_stream_reroute_success_relays_sse_bytes_and_headers() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    expected_estimate = estimate_overflow_prompt_tokens(body)
    window = expected_estimate - 1
    captured: list[httpx.Request] = []
    chunks = [
        b'event: message_start\ndata: {"type": "message_start"}\n\n',
        b'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n',
        b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
    ]
    stream = _TrackedByteStream(chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            stream=stream,
            headers={
                "content-type": "text/event-stream",
                # Deliberately upstream-only headers: none of these may leak
                # into the client-facing response.
                "content-length": "999",
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
                "set-cookie": "sid=abc",
                "content-encoding": "identity",
            },
        )

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.content == b"".join(chunks)  # exact byte ordering, unmodified
    assert stub.payloads == []  # translation never ran; the 2xx already committed
    assert len(captured) == 1
    assert stream.closed is True

    for forbidden in (
        "content-length",
        "content-encoding",
        "transfer-encoding",
        "connection",
        "set-cookie",
    ):
        assert forbidden not in response.headers

    record = client.app.state.compaction_last_reroute
    assert set(record) == _pinned_reroute_record_keys()
    assert record["outcome"] == "rerouted"
    assert record["detail"] is None
    assert record["target_model"] == _COMPACTION_CANONICAL_TARGET
    assert record["mapped_model"] == "codex:gpt-5.1-codex-max"
    assert record["context_window"] == window
    assert record["estimated_prompt_tokens"] == expected_estimate


def test_compaction_stream_reroute_falls_back_on_non_2xx_status_and_closes_response() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream([b'{"error": {"message": "invalid x-api-key"}}'])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(401, stream=stream)

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert len(captured) == 1  # exactly one Anthropic attempt, no retry
    assert stub.payloads  # fallback ran translation against the mapped backend

    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "http_401"
    assert stream.closed is True


def test_compaction_stream_reroute_falls_back_on_3xx_without_following_redirect() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream([b""])

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            307, headers={"location": "https://api.anthropic.com/v1/messages"}, stream=stream
        )

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    # The redirect target is never followed: exactly one outbound request.
    assert len(captured) == 1
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "http_307"
    assert stream.closed is True


def test_compaction_stream_reroute_falls_back_on_connect_error() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "fallback_mapped"
    assert record["detail"] == "connect_error"


def test_compaction_stream_reroute_fallback_translates_untouched_original_body_with_thinking_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, Any]] = []

    def recording_translate(
        claude_request: dict[str, Any], upstream_model: str, reasoning_effort_override: str | None
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request, upstream_model, reasoning_effort_override
        )

    monkeypatch.setattr(server, "translate_claude_request_to_codex", recording_translate)

    body = _compaction_body("claude-opus-4-6", thinking_block=True)
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    assert response.status_code == 503
    assert len(received) == 1
    assert received[0] == body  # untouched original body: stream:true, thinking block included
    assert any(
        block.get("type") == "thinking"
        for message in received[0]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
    )
    # One catalog lookup total: the trigger's own lookup is reused by the
    # mapped fallback instead of being repeated.
    assert stub.context_window_calls == ["gpt-5.1-codex-max"]


def test_compaction_stream_reroute_read_failure_after_first_byte_emits_error_event_and_records_midstream_error() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1
    first_chunk = b'event: message_start\ndata: {"type": "message_start"}\n\n'
    stream = _TrackedByteStream([first_chunk], httpx.ReadError("connection reset"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    # The status line already committed the response; a mid-stream failure
    # can only be reported in-band, never by falling back to the mapped path.
    assert response.status_code == 200
    assert response.content.startswith(first_chunk)
    assert stub.payloads == []
    tail = response.content[len(first_chunk) :]
    assert tail.startswith(b"\n\n")  # forced event boundary before the injected event
    error_payload = _parse_sse_error(tail)
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert isinstance(error_payload["error"]["message"], str)
    assert "connection reset" not in response.content.decode()  # no raw exception text
    assert stream.closed is True

    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "midstream_error"
    assert record["detail"] == "read_error"
    # Every other pinned field from the original "rerouted" record survives.
    assert record["target_model"] == _COMPACTION_CANONICAL_TARGET
    assert record["mapped_model"] == "codex:gpt-5.1-codex-max"
    assert record["context_window"] == window


def test_compaction_stream_reroute_read_failure_before_first_byte_never_invokes_mapped_backend() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1
    stream = _TrackedByteStream([], httpx.ReadError("connection reset before any byte"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    # Locks the commit-point ruling: HTTP 2xx acceptance -- not the first
    # relayed byte -- is the boundary, so the mapped backend must never run
    # even though no chunk ever reached the client.
    assert response.status_code == 200
    assert stub.payloads == []
    error_payload = _parse_sse_error(response.content)
    assert error_payload["type"] == "error"
    assert error_payload["error"]["type"] == "api_error"
    assert stream.closed is True

    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "midstream_error"
    assert record["detail"] == "read_error"


def test_compaction_stream_reroute_first_sse_event_is_error_relayed_unchanged_without_fallback() -> None:
    body = _compaction_body("claude-opus-4-6")
    body["stream"] = True
    window = estimate_overflow_prompt_tokens(body) - 1
    error_chunk = (
        b"event: error\n"
        b'data: {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}\n\n'
    )
    stream = _TrackedByteStream([error_chunk])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    config = _compaction_config({"opus": "codex:gpt-5.1-codex-max"})
    client, stub = _gateway(config, handler, codex_context_window=window)

    response = client.post("/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS)

    # A 2xx status line already commits the reroute; the gateway never peeks
    # at or special-cases the first event to decide whether to fall back.
    assert response.status_code == 200
    assert response.content == error_chunk  # relayed byte-for-byte, unmodified
    assert stub.payloads == []

    record = client.app.state.compaction_last_reroute
    assert record["outcome"] == "rerouted"
    assert record["detail"] is None
    assert stream.closed is True


def test_relay_compaction_stream_explicit_close_closes_upstream_response() -> None:
    stream = _TrackedByteStream([b"event: a\ndata: {}\n\n", b"event: b\ndata: {}\n\n"])
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = SimpleNamespace(compaction_last_reroute=None, compaction_reroute_sequence=0)
    sequence = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        relay = server._CompactionStreamRelay(upstream_response, app_state, sequence)
        first = await anext(relay)
        assert first == b"event: a\ndata: {}\n\n"
        await relay.aclose()

    asyncio.run(scenario())

    assert stream.closed is True
    # An explicit close is neither an httpx.HTTPError nor a cancellation, so
    # the already-committed "rerouted" record is left untouched.
    assert app_state.compaction_last_reroute["outcome"] == "rerouted"


def test_relay_compaction_stream_cancellation_closes_upstream_response() -> None:
    class _HangingByteStream(httpx.AsyncByteStream):
        """Yields one chunk, then hangs until cancelled; records aclose()."""

        def __init__(self, first_chunk: bytes) -> None:
            self._first_chunk = first_chunk
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self._first_chunk
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed = True

    stream = _HangingByteStream(b"event: ping\ndata: {}\n\n")
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = SimpleNamespace(compaction_last_reroute=None, compaction_reroute_sequence=0)
    sequence = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )
    first_chunk_seen = asyncio.Event()

    async def consume() -> None:
        async for _chunk in server._CompactionStreamRelay(upstream_response, app_state, sequence):
            first_chunk_seen.set()

    async def scenario() -> None:
        task = asyncio.create_task(consume())
        await first_chunk_seen.wait()
        await asyncio.sleep(0)  # let the consumer block awaiting the next chunk
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert stream.closed is True
    # asyncio.CancelledError is not an httpx.HTTPError: it must propagate
    # instead of being swallowed, and must not upgrade the diagnostics record.
    assert app_state.compaction_last_reroute["outcome"] == "rerouted"


def test_relay_compaction_stream_stale_failure_does_not_clobber_newer_record() -> None:
    # Request A commits (sequence N); request B then writes its own fresh
    # record (sequence N+1); only afterward does request A's stream fail.
    app_state = SimpleNamespace(compaction_last_reroute=None, compaction_reroute_sequence=0)

    sequence_a = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=100,
        context_window=50,
        detail=None,
    )
    server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.2-codex-max",
        estimated_prompt_tokens=200,
        context_window=90,
        detail=None,
    )
    record_b = dict(app_state.compaction_last_reroute)

    stream = _TrackedByteStream(
        [b"event: message_start\ndata: {}\n\n"], httpx.ReadError("connection reset")
    )
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )

    async def drain() -> None:
        async for _chunk in server._CompactionStreamRelay(
            upstream_response, app_state, sequence_a
        ):
            pass

    asyncio.run(drain())

    # Stale request A's failure must never clobber request B's newer record.
    assert app_state.compaction_last_reroute == record_b


def _committed_relay_state() -> SimpleNamespace:
    app_state = SimpleNamespace(compaction_last_reroute=None, compaction_reroute_sequence=0)
    return app_state


def test_relay_compaction_stream_records_midstream_error_before_terminal_chunk_is_consumed() -> None:
    # The diagnostics upgrade must not depend on the consumer requesting
    # another item after the terminal error chunk: a client that closes right
    # after receiving it must still leave midstream_error behind.
    stream = _TrackedByteStream(
        [b"event: message_start\ndata: {}\n\n"], httpx.ReadError("connection reset")
    )
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = _committed_relay_state()
    sequence = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        relay = server._CompactionStreamRelay(upstream_response, app_state, sequence)
        assert await anext(relay) == b"event: message_start\ndata: {}\n\n"
        terminal = await anext(relay)
        assert terminal.startswith(b"\n\nevent: error\n")
        # No further __anext__: the record must already be upgraded and the
        # upstream response already released.
        assert app_state.compaction_last_reroute["outcome"] == "midstream_error"
        assert app_state.compaction_last_reroute["detail"] == "read_error"
        assert stream.closed is True
        await relay.aclose()

    asyncio.run(scenario())


def test_relay_compaction_stream_aclose_before_first_iteration_closes_upstream() -> None:
    # A generator's finally would never run in this case; the owning
    # iterator must release the upstream response anyway.
    stream = _TrackedByteStream([b"event: a\ndata: {}\n\n"])
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = _committed_relay_state()
    sequence = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        relay = server._CompactionStreamRelay(upstream_response, app_state, sequence)
        await relay.aclose()

    asyncio.run(scenario())

    assert stream.closed is True
    assert app_state.compaction_last_reroute["outcome"] == "rerouted"


def test_owned_streaming_response_closes_iterator_when_send_fails_mid_stream() -> None:
    # A send failure while the iterator sits between chunks must still close
    # the upstream response: Starlette itself never acloses body iterators,
    # so _OwnedStreamingResponse's finally is what releases the stream.
    stream = _TrackedByteStream(
        [b"event: a\ndata: {}\n\n", b"event: b\ndata: {}\n\n"]
    )
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = _committed_relay_state()
    sequence = server._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )
    relay = server._CompactionStreamRelay(upstream_response, app_state, sequence)
    response = server._OwnedStreamingResponse(relay, media_type="text/event-stream")

    body_messages_sent = 0

    async def failing_send(message: dict[str, Any]) -> None:
        nonlocal body_messages_sent
        if message["type"] == "http.response.body":
            body_messages_sent += 1
            if body_messages_sent == 2:
                raise RuntimeError("client went away mid-send")

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="client went away mid-send"):
            await response.stream_response(failing_send)

    asyncio.run(scenario())

    assert stream.closed is True
    # A send failure is not an upstream read failure: the committed record
    # must remain rerouted.
    assert app_state.compaction_last_reroute["outcome"] == "rerouted"


# --- upstream stream ownership -------------------------------------------
# Starlette never closes body iterators, so the gateway's own generators must
# release the Codex stream on every exit: exhaustion, early error returns,
# explicit closure, and cancellation after a client disconnect.


def _finalizing_upstream(flags: dict[str, bool], events: list[dict[str, Any]]):
    async def upstream() -> AsyncIterator[dict[str, Any]]:
        try:
            for event in events:
                yield event
        finally:
            flags["closed"] = True

    return upstream()


_CREATED_EVENT = {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-5.5"}}


def test_closing_the_sse_generator_finalizes_the_upstream_generator() -> None:
    flags = {"closed": False}
    upstream = _finalizing_upstream(
        flags, [_CREATED_EVENT, {"type": "response.output_text.delta", "delta": "hi"}]
    )

    async def scenario() -> None:
        sse = server._translate_claude_sse({"model": "claude-opus-4-6"}, upstream)
        first = await anext(sse)
        assert "message_start" in first
        assert flags["closed"] is False
        await sse.aclose()

    asyncio.run(scenario())
    assert flags["closed"] is True


def test_cancellation_mid_stream_finalizes_the_upstream_generator() -> None:
    flags = {"closed": False}
    first_chunk_seen = asyncio.Event()

    async def hanging_upstream() -> AsyncIterator[dict[str, Any]]:
        try:
            yield _CREATED_EVENT
            await asyncio.Event().wait()  # a quiet upstream: no further events
        finally:
            flags["closed"] = True

    async def consume() -> None:
        async for _chunk in server._translate_claude_sse({}, hanging_upstream()):
            first_chunk_seen.set()

    async def scenario() -> None:
        task = asyncio.create_task(consume())
        await first_chunk_seen.wait()
        await asyncio.sleep(0)  # let the consumer block on the next event
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert flags["closed"] is True


def test_aggregation_error_return_finalizes_the_upstream_generator() -> None:
    flags = {"closed": False}
    upstream = _finalizing_upstream(
        flags,
        [
            _CREATED_EVENT,
            {"type": "error", "error": {"type": "server_error", "message": "boom"}},
            {"type": "response.output_text.delta", "delta": "never read"},
        ],
    )

    response = asyncio.run(server._aggregate_claude_response({}, upstream))

    assert response.status_code == 502
    assert flags["closed"] is True


class ScriptedCodexClient(FakeCodexClient):
    """Yields scripted Codex events and records stream finalization."""

    events: list[dict[str, Any]] = []
    closed = False

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        try:
            for event in type(self).events:
                yield event
        finally:
            type(self).closed = True


def test_non_streaming_request_closes_the_codex_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ScriptedCodexClient.events = [
        _CREATED_EVENT,
        {"type": "response.output_text.delta", "delta": "hello"},
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "usage": {"input_tokens": 1, "output_tokens": 1}},
        },
    ]
    ScriptedCodexClient.closed = False
    client = _create_test_client(
        monkeypatch,
        config=GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"}),
        codex_client=ScriptedCodexClient,
    )
    with client:
        response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 200
    assert response.json()["model"] == "claude-opus-4-6"
    assert ScriptedCodexClient.closed is True


class TimeoutCodexClient(FakeCodexClient):
    """Transport failure before any event arrives (dead upstream connection)."""

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise httpx.ReadTimeout("")


class MidStreamErrorCodexClient(FakeCodexClient):
    """Transport failure after the stream has started."""

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {}}
        raise httpx.ReadError("connection reset")


def test_transport_error_before_first_event_returns_claude_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    with _create_test_client(
        monkeypatch, config=config, codex_client=TimeoutCodexClient
    ) as client:
        response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert "codex backend" in response.json()["error"]["message"]


def test_transport_error_during_aggregation_returns_claude_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    with _create_test_client(
        monkeypatch, config=config, codex_client=MidStreamErrorCodexClient
    ) as client:
        response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


class TestAdminMappingApi:
    """GET/PUT /admin/mapping — runtime map changes persisted to settings.json."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        config = GatewayConfig(
            settings_file=tmp_path / "settings.json", **config_kwargs
        )
        return _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_current_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, model_map={"haiku": "codex:gpt-5.6-luna"}
        ) as client:
            response = client.get("/admin/mapping")

        assert response.status_code == 200
        assert response.json()["model_map"] == {"haiku": "codex:gpt-5.6-luna"}

    def test_put_updates_runtime_and_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )
            # The swap is live for later requests ...
            reread = client.get("/admin/mapping")

        assert response.status_code == 200
        assert response.json()["model_map"] == {"opus": "codex:gpt-5.6-sol"}
        assert reread.json()["model_map"] == {"opus": "codex:gpt-5.6-sol"}
        # ... and persisted without clobbering unrelated settings keys.
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317, "model_map": {"opus": "codex:gpt-5.6-sol"}}

    def test_put_rejected_when_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "codex:gpt-5.6-luna"}')
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )

        assert response.status_code == 409
        assert "CLAUDEX_MODEL_MAP" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/mapping",
                content='{"model_map": {}}',
                headers={"content-type": "text/plain"},
            )
        assert response.status_code == 415

    def test_put_validates_map_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/mapping", json={"model_map": {"": "x"}})
        assert response.status_code == 400
        assert "non-empty strings" in response.json()["error"]["message"]

    def test_put_accepts_and_validates_provider_prefixes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        with self._admin_client(monkeypatch, tmp_path) as client:
            accepted = client.put(
                "/admin/mapping", json={"model_map": {"opus": "kimi:k2.5"}}
            )
            bare_value = client.put(
                "/admin/mapping", json={"model_map": {"opus": "gpt-5.6-sol"}}
            )
            empty_model = client.put(
                "/admin/mapping", json={"model_map": {"opus": "kimi:"}}
            )
            unknown_prefix = client.put(
                "/admin/mapping", json={"model_map": {"opus": "kim:k2.5"}}
            )

        assert accepted.status_code == 200
        assert accepted.json()["model_map"] == {"opus": "kimi:k2.5"}
        assert bare_value.status_code == 400
        assert "no provider prefix" in bare_value.json()["error"]["message"]
        assert empty_model.status_code == 400
        assert "names no model" in empty_model.json()["error"]["message"]
        assert unknown_prefix.status_code == 400
        assert "unknown provider prefix" in unknown_prefix.json()["error"]["message"]

    def test_put_rejects_unknown_and_empty_bodies(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/mapping", json={"bogus": {}}).status_code == 400
            assert client.put("/admin/mapping", json={}).status_code == 400

    def test_admin_refuses_foreign_host_header(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, config=config) as client:
            response = client.get("/admin/mapping")
        assert response.status_code == 403
        assert "DNS-rebinding" in response.json()["error"]["message"]

    def test_admin_requires_local_token_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/mapping").status_code == 401
            response = client.get(
                "/admin/mapping", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200

    def test_get_reports_dashboard_facts_and_env_locks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "codex:gpt-5.6-luna"}')
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/mapping").json()

        assert payload["env_locked"] == {"model_map": "CLAUDEX_MODEL_MAP"}
        assert payload["codex_home"].endswith(".codex")
        assert payload["kimi_code_home"].endswith(".kimi-code")
        assert payload["grok_home"].endswith(".grok")


class TestAdminLogLevel:
    """GET/PUT /admin/log-level — runtime log level applied and persisted."""

    @staticmethod
    def _admin_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        monkeypatch.delenv("CLAUDEX_LOG_LEVEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        return _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_current_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/log-level").json()

        assert payload == {
            "log_level": "info",
            "choices": ["debug", "info", "warning", "error"],
            "env_locked": None,
        }

    def test_put_applies_and_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        saved_levels = {
            name: logging.getLogger(name).level
            for name in server._LOG_LEVEL_LOGGER_NAMES
        }
        try:
            with self._admin_client(monkeypatch, tmp_path) as client:
                response = client.put("/admin/log-level", json={"log_level": "debug"})
                reread = client.get("/admin/log-level")

            assert response.status_code == 200
            assert response.json()["log_level"] == "debug"
            assert reread.json()["log_level"] == "debug"
            assert logging.getLogger().level == logging.DEBUG
            assert logging.getLogger("uvicorn").level == logging.DEBUG
            saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
            assert saved == {"log_level": "debug"}
        finally:
            for name, level in saved_levels.items():
                logging.getLogger(name).setLevel(level)

    def test_put_rejected_when_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        monkeypatch.setenv("CLAUDEX_LOG_LEVEL", "info")
        with _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/log-level", json={"log_level": "debug"})

        assert response.status_code == 409
        assert "CLAUDEX_LOG_LEVEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_validates_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/log-level", json={"log_level": "loud"}).status_code == 400
            assert client.put("/admin/log-level", json={}).status_code == 400


class TestAdminCompactionApi:
    """GET/PUT /admin/compaction — compaction reroute target, mirroring /admin/mapping."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        )

    # --- GET: fresh/configured state, diagnostics schema -------------------

    def test_get_returns_fresh_state_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/compaction").json()

        assert payload == {"model": None, "env_locked": False, "last_reroute": None}

    def test_get_returns_configured_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            payload = client.get("/admin/compaction").json()

        assert payload == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }

    def test_get_reports_last_reroute_with_exact_pinned_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            server._assign_compaction_reroute(
                client.app.state,
                outcome="rerouted",
                target_model="claude-opus-5",
                mapped_model="codex:gpt-5.1-codex-max",
                estimated_prompt_tokens=4096,
                context_window=4000,
                detail=None,
            )
            payload = client.get("/admin/compaction").json()

        last_reroute = payload["last_reroute"]
        assert set(last_reroute) == _pinned_reroute_record_keys()
        assert "sequence" not in last_reroute
        assert last_reroute["outcome"] == "rerouted"
        assert last_reroute["target_model"] == "claude-opus-5"
        assert last_reroute["mapped_model"] == "codex:gpt-5.1-codex-max"
        assert last_reroute["estimated_prompt_tokens"] == 4096
        assert last_reroute["context_window"] == 4000
        assert last_reroute["detail"] is None
        assert isinstance(last_reroute["timestamp"], str) and last_reroute["timestamp"]

    def test_get_reports_env_locked_true_for_nonempty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "claude:claude-env-5")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/compaction").json()

        assert payload["env_locked"] is True

    def test_get_reports_env_locked_true_for_empty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/compaction").json()

        assert payload["env_locked"] is True

    # --- PUT: persistence, hot-swap, disable, enable/disable trigger -------

    def test_put_persists_without_clobbering_unrelated_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/compaction", json={"model": "claude:claude-opus-5"}
            )
            reread = client.get("/admin/compaction")

        assert response.status_code == 200
        assert response.json() == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }
        assert reread.json()["model"] == "claude:claude-opus-5"
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317, "compaction.model": "claude:claude-opus-5"}

    def test_put_hot_swaps_live_config_for_subsequent_requests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            client.put("/admin/compaction", json={"model": "claude:claude-opus-5"})
            assert client.app.state.config.compaction_model == "claude:claude-opus-5"
            reread = client.get("/admin/compaction")

        assert reread.json()["model"] == "claude:claude-opus-5"

    def test_put_disables_and_removes_the_settings_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"compaction.model": "claude:claude-old-5", "port": 1234}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-old-5"
        ) as client:
            response = client.put("/admin/compaction", json={"model": None})
            reread = client.get("/admin/compaction")

        assert response.status_code == 200
        assert response.json() == {"model": None, "env_locked": False, "last_reroute": None}
        assert reread.json()["model"] is None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 1234}
        assert "compaction.model" not in saved

    def test_put_enable_then_disable_changes_the_live_reroute_trigger(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = _compaction_body("claude-opus-4-6")
        window = estimate_overflow_prompt_tokens(body) - 1
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"id": "msg_1", "type": "message", "model": "claude-opus-5"},
            )

        config = GatewayConfig(
            settings_file=tmp_path / "settings.json",
            model_map={"opus": "codex:gpt-5.1-codex-max"},
        )
        client, stub = _gateway(config, handler, codex_context_window=window)
        client.app.state.admin_lock = asyncio.Lock()
        client.base_url = "http://127.0.0.1:8787"

        enable = client.put("/admin/compaction", json={"model": "claude:claude-opus-5"})
        assert enable.status_code == 200

        rerouted = client.post(
            "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
        )
        assert rerouted.status_code == 200
        assert len(captured) == 1
        assert stub.payloads == []

        disable = client.put("/admin/compaction", json={"model": None})
        assert disable.status_code == 200

        not_rerouted = client.post(
            "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
        )
        # Reroute is now off: falls through to the mapped Codex backend
        # (the stub, which always fails), never a second Anthropic attempt.
        assert not_rerouted.status_code == 503
        assert len(captured) == 1
        assert len(stub.payloads) == 1

    # --- PUT: request-shape validation --------------------------------------

    def test_put_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put(
                "/admin/compaction",
                content='{"model": null}',
                headers={"content-type": "text/plain"},
            )
            config_after = client.app.state.config
        assert response.status_code == 415
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_missing_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put("/admin/compaction", json={})
            config_after = client.app.state.config
        assert response.status_code == 400
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_unknown_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put(
                "/admin/compaction",
                json={"model": "claude:claude-opus-5", "bogus": True},
            )
            config_after = client.app.state.config
        assert response.status_code == 400
        assert "bogus" in response.json()["error"]["message"]
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_invalid_string(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/compaction", json={"model": "gpt-5"})
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "claude:" in response.json()["error"]["message"]
        assert not settings_file.exists()
        assert config_after.compaction_model is None

    @pytest.mark.parametrize(
        "value", [True, False, 5, 1.5, ["claude:claude-opus-5"], {"model": "claude:x"}]
    )
    def test_put_rejects_every_non_string_non_null_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Any
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put("/admin/compaction", json={"model": value})
            config_after = client.app.state.config
        assert response.status_code == 400
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    # --- Security/lock: bearer + Host, for both GET and PUT ----------------

    def test_get_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/compaction").status_code == 401

    def test_get_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/compaction", headers={"Authorization": "Bearer wrong"}
            )
        assert response.status_code == 401

    def test_get_succeeds_with_correct_bearer_on_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200

    def test_get_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, config=config) as client:
            response = client.get(
                "/admin/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 403

    def test_put_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put("/admin/compaction", json={"model": None})
            config_after = client.app.state.config
        assert response.status_code == 401
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put(
                "/admin/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer wrong"},
            )
            config_after = client.app.state.config
        assert response.status_code == 401
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_succeeds_with_correct_bearer_on_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put(
                "/admin/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer secret"},
            )
        assert response.status_code == 200

    def test_put_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        with _create_test_client(monkeypatch, config=config) as client:
            response = client.put(
                "/admin/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer secret"},
            )
            config_after = client.app.state.config
        assert response.status_code == 403
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    # --- Environment lock: 409 before the lock, empty value still locks ----

    def test_put_rejected_when_env_locked_with_nonempty_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "claude:claude-env-5")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/compaction", json={"model": "claude:claude-opus-5"}
            )
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_COMPACTION_MODEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_rejected_when_env_locked_with_empty_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/compaction", json={"model": None})
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_COMPACTION_MODEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    # --- Persistence failure: 500, no hot-swap, no false success envelope --

    def test_put_persistence_failure_returns_500_without_mutating_runtime(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise server.ConfigError("disk full")

        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setattr(server, "update_settings_file", boom)
            response = client.put(
                "/admin/compaction", json={"model": "claude:claude-opus-5"}
            )
            config_after = client.app.state.config

        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert "last_reroute" not in body
        assert config_after.compaction_model is None
        assert not (tmp_path / "settings.json").exists()


def test_admin_logs_returns_recent_records(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        logging.getLogger("claudex_gateway.test").warning("hello %s", "world")
        response = client.get("/admin/logs")

    assert response.status_code == 200
    entries = [e for e in response.json()["logs"] if e["message"] == "hello world"]
    assert entries and entries[0]["level"] == "WARNING"
    assert entries[0]["logger"] == "claudex_gateway.test"


def test_admin_logs_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch) as client:
        assert client.get("/admin/logs").status_code == 403


def test_admin_usage_returns_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_claude(http_client: Any) -> dict[str, Any]:
        return {"provider": "claude", "status": "ok", "error": None}

    async def fake_codex(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        return {"provider": "codex", "status": "unavailable", "error": "no creds"}

    async def fake_kimi(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        return {"provider": "kimi", "status": "ok", "error": None}

    async def fake_grok(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        return {"provider": "grok", "status": "ok", "error": None}

    monkeypatch.setattr(server, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(server, "fetch_codex_usage", fake_codex)
    monkeypatch.setattr(server, "fetch_kimi_usage", fake_kimi)
    monkeypatch.setattr(server, "fetch_grok_usage", fake_grok)
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert body["codex"]["status"] == "unavailable"
    assert body["kimi"]["status"] == "ok"
    assert body["grok"]["status"] == "ok"
    assert body["fetched_at"] > 0


def test_admin_usage_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch) as client:
        assert client.get("/admin/usage").status_code == 403


def test_admin_usage_single_provider_skips_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_claude(http_client: Any) -> dict[str, Any]:
        return {"provider": "claude", "status": "ok", "error": None}

    async def codex_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("codex probe ran for ?provider=claude")

    async def kimi_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("kimi probe ran for ?provider=claude")

    async def grok_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("grok probe ran for ?provider=claude")

    monkeypatch.setattr(server, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(server, "fetch_codex_usage", codex_must_not_run)
    monkeypatch.setattr(server, "fetch_kimi_usage", kimi_must_not_run)
    monkeypatch.setattr(server, "fetch_grok_usage", grok_must_not_run)
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "claude"})

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert "codex" not in body
    assert "kimi" not in body
    assert "grok" not in body


def test_admin_usage_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "gemini"})

    assert response.status_code == 400


def _record_reset_keys(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[dict[str, Any]]
) -> list[str]:
    """Stub the consume call, answering `outcomes` in order and logging the keys."""
    keys: list[str] = []
    remaining = list(outcomes)

    async def fake_consume(
        http_client: Any, auth_manager: Any, redeem_request_id: str
    ) -> dict[str, Any]:
        keys.append(redeem_request_id)
        return remaining.pop(0)

    monkeypatch.setattr(server, "consume_codex_reset_credit", fake_consume)
    return keys


def test_admin_reset_credit_returns_the_backend_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = _record_reset_keys(
        monkeypatch, [{"status": "ok", "outcome": "reset", "error": None}]
    )
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.post("/admin/codex/reset-credit", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "outcome": "reset", "error": None}
    assert len(keys) == 1 and keys[0]


def test_admin_reset_credit_reuses_the_key_until_an_attempt_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A timed-out attempt may or may not have burned the credit, so the retry
    # must carry the same idempotency key and let the backend deduplicate;
    # only a settled outcome may mint a fresh one.
    keys = _record_reset_keys(
        monkeypatch,
        [
            {"status": "error", "outcome": None, "error": "timeout"},
            {"status": "unavailable", "outcome": None, "error": "no creds"},
            {"status": "ok", "outcome": "reset", "error": None},
            {"status": "ok", "outcome": "no_credit", "error": None},
        ],
    )
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        for _ in range(4):
            assert client.post("/admin/codex/reset-credit", json={}).status_code == 200

    assert keys[0] == keys[1] == keys[2], "unsettled attempts must retry the same key"
    assert keys[3] != keys[2], "a settled attempt must not reuse its key"


def test_admin_reset_credit_is_guarded_like_every_other_admin_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a guarded request reached the ChatGPT backend")

    monkeypatch.setattr(server, "consume_codex_reset_credit", must_not_run)

    # Foreign Host header (DNS-rebinding guard).
    with _create_test_client(monkeypatch) as client:
        assert client.post("/admin/codex/reset-credit", json={}).status_code == 403
    # A form post would dodge the CORS preflight, so JSON is required.
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        assert client.post("/admin/codex/reset-credit", content="x").status_code == 415
    # And the local bearer token still applies.
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(
        monkeypatch, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        assert client.post("/admin/codex/reset-credit", json={}).status_code == 401


def test_admin_reset_credit_is_never_reachable_by_GET(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spending a credit must not be possible by navigating to a URL.
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a GET spent a reset credit")

    monkeypatch.setattr(server, "consume_codex_reset_credit", must_not_run)
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/admin/codex/reset-credit").status_code == 405


def test_dashboard_usage_merged_into_status_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    # Usage renders inside the Status tab's provider cards: a per-provider
    # body hook, the fetch on entering Status, and no separate tab.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert 'data-t="usage"' not in page
    assert 'id="tab-usage"' not in page
    assert 'id="usage-body-claude"' in page
    assert 'id="usage-body-codex"' in page
    assert 'id="usage-body-kimi"' in page
    assert "/admin/usage" in page
    # The usage hooks sit inside the Status section, not a sibling tab.
    assert page.index('id="tab-status"') < page.index('id="usage-body-codex"')
    assert page.index('id="usage-body-claude"') < page.index('id="tab-log"')
    # Claude leads the provider cards as a Status card like the others.
    assert "<h2>Claude Status" in page
    assert "<h2>Claude Usage" not in page
    assert page.index('id="usage-body-claude"') < page.index('id="codex-stat"')


def test_dashboard_settings_tab_leads_and_holds_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Settings tab is the settings home: it leads the tab bar, opens by
    # default, and holds the Compact Reroute row behind the category rail;
    # the provider status cards live in the Status tab.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert (
        page.index('data-t="settings"')
        < page.index('data-t="status"')
        < page.index('data-t="map"')
        < page.index('data-t="log"')
    )
    assert '<body data-tab="settings">' in page
    assert 'var TAB_NAMES=["settings","status","map","log"]' in page
    # The category rail leads the pane and deep-links its single category.
    assert 'href="#settings/general"' in page
    assert (
        page.index('class="rail-item on"')
        < page.index('id="compaction-card"')
    )
    # The compaction row sits inside the Settings section (before the next
    # sibling section), not in Status where it used to render.
    assert (
        page.index('id="tab-settings"')
        < page.index('id="compaction-card"')
        < page.index('id="tab-map"')
    )

def test_dashboard_plan_and_credits_read_inside_the_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # Plan and credit chips used to crowd the card headings; both now read as
    # part of the card they qualify, so the hooks and the chip style are gone.
    assert "usage-chips" not in page
    assert "uplan" not in page
    for provider in ("Claude", "Codex", "Kimi"):
        assert f"<h2>{provider} Status</h2>" in page

    # The plan always precedes the quota bars it qualifies.
    for provider in ("claude", "codex", "kimi"):
        assert f'id="usage-plan-{provider}"' in page
        assert page.index(f'id="usage-plan-{provider}"') < page.index(
            f'id="usage-body-{provider}"'
        )
    # Codex and Kimi read it inside the login-status box, above the state
    # line. That line is its own element precisely so a health render can
    # rewrite the status without wiping the plan the usage probe put there.
    for provider in ("codex", "kimi"):
        assert (
            page.index(f'id="{provider}-stat"')
            < page.index(f'id="usage-plan-{provider}"')
            < page.index(f'id="{provider}-statline"')
        )
    # Claude has no login-status box, so its plan reads under the description.
    assert page.index('id="usage-plan-claude"') > page.index("<h2>Claude Status</h2>")

    # The credit count is stated by the reset line, which owns the spend
    # button too, so there is one place credits are reported.
    assert 'class="uact"' in page
    assert "리셋 크레딧" in page


def test_dashboard_status_cards_load_as_skeletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # The cards used to render a one-line "확인 중…" that the loaded content
    # then pushed apart. They now ship a skeleton of the same shape instead,
    # so nothing moves when the probes answer.
    assert "확인 중" not in page
    assert 'class="sk"' in page
    # A skeleton carries the text it stands in for, painted transparent, so
    # its line box matches the line that replaces it.
    assert ".sk{" in page and "color:transparent" in page
    # Both status boxes seed one before any request goes out, and the usage
    # bodies are filled synchronously at boot rather than starting empty.
    for provider in ("codex", "kimi"):
        assert f'id="{provider}-statline"><span class="sk">' in page
    assert 'renderUsageProvider(p,null)' in page


def test_dashboard_served_at_root(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Claudex Gateway" in response.text
    assert "/admin/mapping" in response.text


def test_favicon_served_for_browser_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch) as client:
        response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "max-age" in response.headers["cache-control"]
    assert response.text.startswith("<svg")


def test_dashboard_port_has_an_enlarged_invisible_hit_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 14px visible port dot is too small to drag from: the board widens
    # the grab area to the node's full height plus margins without changing
    # the visual. These markers are the whole mechanism, so pin them.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert ".node.src .port::after" in page
    assert "pointer-events:auto" in page


def test_hello_reports_identity_and_auth_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    with _create_test_client(monkeypatch) as client:
        body = client.get("/api/hello").json()

    assert body["hello"] == "claudex-gateway"
    assert body["local_auth_required"] is False
    assert isinstance(body["pid"], int)


def test_hello_reports_auth_required_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(local_token="secret-token-value")
    with _create_test_client(monkeypatch, config=config) as client:
        response = client.get("/api/hello")

    assert response.json()["local_auth_required"] is True
    assert "secret-token-value" not in response.text


def test_dashboard_keeps_the_local_token_in_memory_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # The dashboard bootstraps from the safe hello flag and attaches the
    # bearer token to admin calls from a closure variable only.
    assert "local_auth_required" in page
    assert '"Authorization":"Bearer "+localToken' in page
    # The token must never be persisted or interpolated anywhere durable.
    assert "localStorage" not in page
    assert "sessionStorage" not in page
    assert "document.cookie" not in page


def test_dashboard_has_no_hardcoded_codex_model_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # Catalogs are fetched live; a baked-in model list goes stale and misleads
    # when the catalog request fails. Nodes already in the mapping still render
    # via buildColumns.
    assert "CODEX_FALLBACK" not in page
    assert "gpt-5" not in page
    assert "CATALOG={codex:[],kimi:[],grok:[]}" in page


def test_dashboard_board_shows_only_referenced_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # With several providers a catalog dump is unusable as a board, so target
    # nodes are only what the map references plus what the add-node box
    # stages; the catalogs survive purely as autocomplete for that box.
    assert "concat(Object.values(DIR.mapping),addedTargets)" in page
    assert 'list="add-catalog"' in page
    # All provider catalogs feed it, so the dashboard depends on the Kimi
    # and Grok endpoints too — not just the Codex one.
    assert '"/admin/kimi/models"' in page
    assert '"/admin/grok/models"' in page


def _compaction_section(page: str) -> str:
    """Slice out only the Compaction card's own markup, by its own markers.

    `claude-haiku` must never appear as a curated compaction option, but the
    Router tab's `CLAUDE_KEYS` array legitimately mentions bare "haiku"
    elsewhere in the same document — so absence has to be asserted against
    this scoped slice, never the whole page.
    """
    start = page.index("<!-- compaction-section:start -->")
    end = page.index("<!-- compaction-section:end -->")
    return page[start:end]


def _compaction_apply_fn(page: str) -> str:
    start = page.index("function applyCompaction(){")
    end = page.index(
        'document.getElementById("comp-select").addEventListener("change"', start
    )
    return page[start:end]


def test_dashboard_compaction_section_marker_and_endpoint_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert "<!-- compaction-section:start -->" in page
    assert "<!-- compaction-section:end -->" in page
    section = _compaction_section(page)
    assert 'id="compaction-card"' in section
    assert "/admin/compaction" in page


def test_dashboard_compaction_options_in_pinned_order_without_haiku(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    section = _compaction_section(page)
    assert (
        section.index(">Disabled<")
        < section.index("claude-sonnet-5 (recommended)")
        < section.index(">claude-opus-5<")
        < section.index(">claude-fable-5<")
        < section.index(">Custom<")
    )
    assert "claude-haiku" not in section
    # The literal string still exists elsewhere in the document (see
    # compactionDraftFromModel's own comment), proving this is a scoped
    # assertion and not a whole-page absence check.
    assert "claude-haiku" in page


def test_dashboard_compaction_custom_input_labeled_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    section = _compaction_section(page)
    assert "unverified until first use" in section
    assert 'id="comp-custom-input"' in section


def test_dashboard_compaction_credentials_disclosure_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The card must state which credentials rerouted requests run on, so the
    # user knows their own Claude account is being used.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    section = _compaction_section(page)
    assert "장치에 저장된 Claude 기본 자격증명" in section


def test_dashboard_compaction_fetched_in_parallel_boot_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    boot_start = page.index("function boot(){")
    promise_all = page.index("Promise.all([", boot_start)
    promise_all_end = page.index("]);", promise_all)
    parallel_calls = page[promise_all:promise_all_end]
    assert 'jfetch("/admin/compaction")' in parallel_calls


def test_dashboard_compaction_keeps_configured_custom_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    # A configured model that is not one of the three curated ids renders as
    # Custom with its raw id filled in, instead of being dropped.
    assert "COMPACTION_CURATED_MODELS.indexOf(id)>=0" in page
    assert '{kind:"custom",custom:id}' in page


def test_dashboard_compaction_diagnostics_ui_removed_by_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Settings redesign dropped the last-reroute record from the page:
    # diagnostics stay reachable through GET /admin/compaction only. Guard
    # against the UI quietly returning.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert 'id="comp-diagnostics"' not in page
    assert "renderCompactionDiagnostics" not in page
    assert "아직 재라우팅이 시도되지 않았습니다" not in page


def test_dashboard_compaction_apply_body_matches_pinned_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert "JSON.stringify({model:model})" in page
    apply_fn = _compaction_apply_fn(page)
    # Disabled sends exactly {"model": null}; curated/custom selections carry
    # the "claude:" prefix parse_compaction_model expects.
    assert "model=null" in apply_fn
    assert '"claude:"+raw' in apply_fn
    assert '"claude:"+COMP.draftKind' in apply_fn


def test_dashboard_compaction_custom_submission_is_trimmed_and_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    apply_fn = _compaction_apply_fn(page)
    assert "input.value.trim()" in apply_fn
    assert "if(!raw)return;" in apply_fn


def test_dashboard_compaction_409_branch_refreshes_via_get_not_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    apply_fn = _compaction_apply_fn(page)
    branch_start = apply_fn.index("r.status===409")
    branch_end = apply_fn.index("if(!r.ok){", branch_start)
    locked_branch = apply_fn[branch_start:branch_end]

    # Controls lock and the admin error renders before any GET is issued.
    assert "COMP.locked=true" in locked_branch
    assert "errDetail(r.body)" in locked_branch
    # Current state is only ever adopted from a fresh, authenticated GET —
    # never from the 409 response body itself.
    assert 'jfetch("/admin/compaction")' in locked_branch


def test_dashboard_compaction_409_refresh_failure_stays_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If the post-409 refresh GET itself fails or returns a malformed
    # envelope, its body must not be rendered as state and the card must
    # remain locked.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    apply_fn = _compaction_apply_fn(page)
    branch_start = apply_fn.index("r.status===409")
    branch_end = apply_fn.index("if(!r.ok){", branch_start)
    locked_branch = apply_fn[branch_start:branch_end]

    refresh_start = locked_branch.index('jfetch("/admin/compaction")')
    refresh_branch = locked_branch[refresh_start:]
    # renderCompactionState is guarded on a successful, well-formed envelope.
    guard_at = refresh_branch.index('typeof g.body.env_locked==="boolean"')
    render_at = refresh_branch.index("renderCompactionState(g.body)")
    assert guard_at < render_at
    assert "g.ok" in refresh_branch[:render_at]
    # The failure arm keeps the card locked instead of adopting the body.
    else_arm = refresh_branch[render_at:]
    assert "COMP.locked=true" in else_arm
    assert "renderCompactionState(g.body)" in locked_branch
    assert "r.body.model" not in locked_branch
    assert "r.body.env_locked" not in locked_branch
    assert "r.body.last_reroute" not in locked_branch


class CatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        return ["gpt-5.6-sol", "gpt-5.5"]


class FailingCatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        raise CodexUpstreamError(400, '{"error":{"message":"unsupported client"}}')


class ProbeCodexClient(FakeCodexClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class RejectingCodexClient(FakeCodexClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise CodexUpstreamError(400, '{"error":{"message":"model_not_found"}}')


class CatalogKimiClient(FakeKimiClient):
    async def list_models(self) -> Any:
        return {"data": [{"id": "k2.5"}, {"id": "k3"}]}


class FailingCatalogKimiClient(FakeKimiClient):
    async def list_models(self) -> Any:
        raise KimiUpstreamError(401, '{"error":{"message":"token expired"}}')


class ProbeKimiClient(FakeKimiClient):
    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        # The probe must send the raw model with the prefix already removed.
        assert json.loads(body)["model"] == "k3"
        return httpx.Response(200, json={"type": "message", "model": "k3"})


class RejectingKimiClient(FakeKimiClient):
    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        raise KimiUpstreamError(
            404, '{"type":"error","error":{"type":"not_found_error","message":"model not found"}}'
        )


class ProbeGrokClient(FakeGrokClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class CatalogGrokClient(FakeGrokClient):
    async def list_models(self) -> list[str]:
        return ["grok-4.5", "grok-4.3"]


class FailingCatalogGrokClient(FakeGrokClient):
    async def list_models(self) -> list[str]:
        raise GrokUpstreamError(401, '{"error":{"message":"token expired"}}')


class RejectingGrokClient(FakeGrokClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise GrokUpstreamError(400, '{"error":{"message":"model_not_found"}}')


def test_codex_client_list_models_filters_hidden_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["client_version"] == "0.146.0"
        assert request.headers["Chatgpt-Account-Id"] == "account"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.6-sol", "visibility": "list"},
                    {"slug": "codex-auto-review", "visibility": "hide"},
                    {"slug": "gpt-5.5"},
                ]
            },
        )

    async def run() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(AvailableCodexAuthManager(), http_client).list_models()

    assert asyncio.run(run()) == ["gpt-5.6-sol", "gpt-5.5"]


class TestAdminDashboardApi:
    """Codex catalog proxy and connection-test endpoints behind the admin guard."""

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> TestClient:
        return _create_test_client(
            monkeypatch, base_url="http://127.0.0.1:8787", **kwargs
        )

    def test_codex_models_returns_visible_slugs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, codex_client=CatalogCodexClient) as client:
            response = client.get("/admin/codex/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["gpt-5.6-sol", "gpt-5.5"]}

    def test_codex_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, codex_client=FailingCatalogCodexClient) as client:
            response = client.get("/admin/codex/models")

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "unsupported client"

    def test_codex_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _create_test_client(monkeypatch, codex_client=CatalogCodexClient) as client:
            assert client.get("/admin/codex/models").status_code == 403

    def test_kimi_models_relays_catalog_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The catalog passes through unshaped: the map bypasses model IDs
        # untouched, so the raw backend answer is the preset source.
        with self._client(monkeypatch, kimi_client=CatalogKimiClient) as client:
            response = client.get("/admin/kimi/models")

        assert response.status_code == 200
        assert response.json() == {"data": [{"id": "k2.5"}, {"id": "k3"}]}

    def test_kimi_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, kimi_client=FailingCatalogKimiClient) as client:
            response = client.get("/admin/kimi/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_kimi_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _create_test_client(monkeypatch, kimi_client=CatalogKimiClient) as client:
            assert client.get("/admin/kimi/models").status_code == 403

    def test_grok_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, grok_client=CatalogGrokClient) as client:
            response = client.get("/admin/grok/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["grok-4.5", "grok-4.3"]}

    def test_grok_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, grok_client=FailingCatalogGrokClient) as client:
            response = client.get("/admin/grok/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_grok_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _create_test_client(monkeypatch, grok_client=CatalogGrokClient) as client:
            assert client.get("/admin/grok/models").status_code == 403

    def test_connection_test_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with self._client(monkeypatch, codex_client=ProbeCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "codex:gpt-5.6-luna"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "gpt-5.6-luna"
        assert isinstance(result["latency_ms"], int)

    def test_connection_test_rejects_bare_target(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, codex_client=ProbeCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "gpt-5.6-luna"},
            )

        assert response.status_code == 400
        assert "no provider prefix" in response.json()["error"]["message"]

    def test_connection_test_kimi_target_probes_kimi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, kimi_client=ProbeKimiClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kimi:k3"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "k3"

    def test_connection_test_reports_kimi_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, kimi_client=RejectingKimiClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kimi:k3"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 404
        assert "model not found" in result["detail"]

    def test_connection_test_grok_target_probes_grok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, grok_client=ProbeGrokClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "grok:grok-4.5"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "grok-4.5"

    def test_connection_test_reports_grok_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, grok_client=RejectingGrokClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "grok:grok-nope"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert "model_not_found" in result["detail"]

    def test_connection_test_rejects_unknown_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kim:k3"},
            )

        assert response.status_code == 400
        assert "unknown provider prefix" in response.json()["error"]["message"]

    def test_connection_test_reports_unknown_codex_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, codex_client=RejectingCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "codex:gpt-nope"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert result["detail"] == "model_not_found"

    def test_connection_test_validates_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch) as client:
            empty_target = client.post(
                "/admin/test",
                json={"target": " "},
            )
            wrong_content_type = client.post(
                "/admin/test",
                content='{"target": "codex:x"}',
                headers={"content-type": "text/plain"},
            )

        assert empty_target.status_code == 400
        assert wrong_content_type.status_code == 415
