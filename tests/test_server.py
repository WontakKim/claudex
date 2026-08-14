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
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.testclient import TestClient

import claudex_gateway.admin_api as admin_api
import claudex_gateway.relay as relay
import claudex_gateway.server as server
import claudex_gateway.server_support as server_support
from claudex_gateway import claude_accounts, compaction, paths
from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.claude_account_pool import AccountCooldownTracker
from claudex_gateway.claude_auth import CLAUDE_TOKEN_URL
from claudex_gateway.claude_balanced_router import (
    ClaudeBalancedRouter,
    ClaudeBalancedRuntime,
    ClaudeUsagePollCoordinator,
    UsagePollAccount,
    derive_session_key,
)
from claudex_gateway.claude_pool_runtime_state import (
    ClaudePoolRuntimeStateStore,
    RestoreValidationContext,
)
from claudex_gateway.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex_gateway.config import GatewayConfig, OpenAICompatibleProvider
from claudex_gateway.kimi_auth import KimiCredentials
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.openai_compatible_client import OpenAICompatibleUpstreamError
from claudex_gateway.grok_auth import GrokCredentials
from claudex_gateway.grok_client import GrokClient, GrokUpstreamError
from claudex_gateway.relay import (
    _CompactionStreamRelay,
    _OwnedStreamingResponse,
    _aggregate_claude_response,
    _rewrite_kimi_sse,
    _translate_claude_sse,
    _upstream_error_to_claude,
)
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

    async def supports_fast_tier(self, model: str) -> bool:
        return False


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


_CUSTOM_API_KEY = "sk-custom-secret"


def _custom_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        wire_api="responses",
        base_url="https://models.example/api/v1",
        api_key=_CUSTOM_API_KEY,
    )


class FakeOpenAICompatibleClient:
    def __init__(
        self,
        name: str,
        provider: OpenAICompatibleProvider,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        self.name = name
        self.provider = provider

    async def list_models(self) -> list[str]:
        return ["gpt-5.5"]

    async def context_window(self, model: str) -> int | None:
        return None


class FailingOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def list_models(self) -> list[str]:
        raise OpenAICompatibleUpstreamError(
            503, '{"error":{"message":"catalog unavailable"}}', self.name
        )


# The real HOME this process started with, captured before any test ever
# monkeypatches it -- lets `_create_test_client` tell a still-real HOME
# (a naive test that never isolated it) apart from one a caller already
# isolated itself (e.g. `_balanced_env`, `_admin_client`, under its own
# `tmp_path` subpath), so it isolates HOME without clobbering a caller's
# own choice.
_REAL_HOME = Path.home()


def _create_test_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config: GatewayConfig | None = None,
    codex_auth: type = AvailableCodexAuthManager,
    codex_client: type = FakeCodexClient,
    kimi_auth: type = AvailableKimiAuthManager,
    kimi_client: type = FakeKimiClient,
    grok_auth: type = AvailableGrokAuthManager,
    grok_client: type = FakeGrokClient,
    custom_client: type = FakeOpenAICompatibleClient,
    base_url: str = "http://testserver",
) -> TestClient:
    # T-9's lifespan acquires the process-lifetime claude account pool lease
    # (paths.runtime_dir() == Path.home() / ".claudex") for every routing
    # mode, so a naive lifespan test run against the REAL HOME would contend
    # with a live claudex-gateway daemon holding that lock (G-6). Isolate
    # HOME to this test's own tmp_path before the lifespan ever runs, unless
    # the caller already isolated it to somewhere else first.
    if Path.home() == _REAL_HOME:
        monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(server, "CodexAuthManager", codex_auth)
    monkeypatch.setattr(server, "CodexClient", codex_client)
    monkeypatch.setattr(server, "KimiAuthManager", kimi_auth)
    monkeypatch.setattr(server, "KimiClient", kimi_client)
    monkeypatch.setattr(server, "GrokAuthManager", grok_auth)
    monkeypatch.setattr(server, "GrokClient", grok_client)
    monkeypatch.setattr(server, "OpenAICompatibleClient", custom_client)
    return TestClient(server.create_app(config or GatewayConfig()), base_url=base_url)


def test_route_ownership_matches_surface_modules() -> None:
    app = server.create_app(GatewayConfig())
    routes = [route for route in app.routes if hasattr(route, "endpoint")]
    admin_paths = {"/", "/favicon.ico", "/api/hello", "/health"}
    assert admin_paths <= {route.path for route in routes}
    assert any(route.path.startswith("/admin/") for route in routes)

    for route in routes:
        if route.path in admin_paths or route.path.startswith("/admin/"):
            assert route.endpoint.__module__ == "claudex_gateway.admin_api"
        elif route.path in {"/v1/messages", "/v1/messages/count_tokens"}:
            assert route.endpoint.__module__ == "claudex_gateway.relay"


def test_messages_routes_enforce_local_bearer_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        messages = client.post("/v1/messages", json={"messages": []})
        count_tokens = client.post("/v1/messages/count_tokens", json={"messages": []})

    for response in (messages, count_tokens):
        assert response.status_code == 401
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "authentication_error"


def test_removed_responses_direction_routes_return_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.post("/v1/responses", json={"input": "Hello"}).status_code == 404
        assert client.get("/v1/models").status_code == 404


def test_health_reports_ok_with_codex_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, codex_auth=MissingCodexAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["codex"]["status"] == "error"


def test_health_stays_ok_without_kimi_credentials_when_map_has_no_kimi_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, kimi_auth=MissingKimiAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is False


def test_health_reports_error_without_kimi_credentials_when_map_routes_to_kimi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "kimi:k2.5"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, kimi_auth=MissingKimiAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is True


def test_health_stays_ok_without_grok_credentials_when_map_has_no_grok_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, grok_auth=MissingGrokAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is False


def test_health_reports_error_without_grok_credentials_when_map_routes_to_grok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "grok:grok-4.5"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, grok_auth=MissingGrokAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is True


def test_health_reports_ready_custom_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["providers"]["wrtn"] == {
        "status": "ok",
        "required": True,
    }


def test_health_reports_error_when_required_custom_provider_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=FailingOpenAICompatibleClient,
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["wrtn"]["status"] == "error"
    assert health.json()["providers"]["wrtn"]["required"] is True


def test_health_stays_ready_when_unrequired_custom_provider_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=FailingOpenAICompatibleClient,
    ) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["wrtn"]["status"] == "error"
    assert health.json()["providers"]["wrtn"]["required"] is False


def test_health_without_custom_providers_keeps_builtin_provider_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        health = client.get("/health")

    assert list(health.json()["providers"]) == ["codex", "kimi", "grok"]


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
    status_code, body = _upstream_error_to_claude(
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
    status_code, body = _upstream_error_to_claude(
        _upstream_error(413, {"message": "Request exceeds the maximum context length."})
    )
    assert status_code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("prompt is too long: ")


def test_non_overflow_error_passes_through_unchanged() -> None:
    status_code, body = _upstream_error_to_claude(
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
        supports_fast_tier: bool = False,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.context_window_calls: list[str] = []
        self._context_window = context_window
        self._error = error or CodexUpstreamError(503, "stub codex upstream")
        self._supports_fast_tier = supports_fast_tier

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return self._context_window

    async def supports_fast_tier(self, model: str) -> bool:
        return self._supports_fast_tier

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
    custom_provider_clients: dict[str, Any] | None = None,
    codex_context_window: int | None = None,
    codex_error: CodexUpstreamError | None = None,
    codex_supports_fast_tier: bool = False,
) -> tuple[TestClient, StubCodexClient]:
    app = server.create_app(config)
    # The lifespan requires real Codex credentials, so set the state directly
    # instead of entering the TestClient context manager.
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    stub = StubCodexClient(
        codex_context_window, codex_error, codex_supports_fast_tier
    )
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
    app.state.custom_provider_clients = custom_provider_clients or {}
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


def test_codex_fast_tier_is_sent_for_supported_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(
        model_map={"opus": "codex:gpt-5.6-sol"}, codex_service_tier="fast"
    )
    client, stub = _gateway(
        config,
        _failing_anthropic_handler,
        codex_supports_fast_tier=True,
    )
    caplog.set_level(logging.INFO, logger="claudex_gateway.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert stub.payloads[0]["service_tier"] == "priority"
    assert any(
        record.name == "claudex_gateway.server"
        and record.levelno == logging.INFO
        and "tier=priority" in record.getMessage()
        for record in caplog.records
    )


def test_codex_fast_tier_is_omitted_for_unsupported_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(
        model_map={"opus": "codex:gpt-5.6-sol"}, codex_service_tier="fast"
    )
    client, stub = _gateway(config, _failing_anthropic_handler)
    caplog.set_level(logging.DEBUG, logger="claudex_gateway.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert "service_tier" not in stub.payloads[0]
    assert any(
        record.name == "claudex_gateway.server"
        and record.levelno == logging.DEBUG
        and record.getMessage()
        == "fast tier requested but the codex catalog does not advertise it for gpt-5.6-sol"
        for record in caplog.records
    )
    assert any(
        record.name == "claudex_gateway.server"
        and record.levelno == logging.INFO
        and "tier=standard" in record.getMessage()
        for record in caplog.records
    )


def test_codex_fast_tier_is_omitted_when_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, stub = _gateway(
        config,
        _failing_anthropic_handler,
        codex_supports_fast_tier=True,
    )
    caplog.set_level(logging.INFO, logger="claudex_gateway.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert "service_tier" not in stub.payloads[0]
    assert any(
        record.name == "claudex_gateway.server"
        and record.levelno == logging.INFO
        and "tier=standard" in record.getMessage()
        for record in caplog.records
    )


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    body = _message_body("claude-opus-4-6")
    body["stream"] = True
    with _create_test_client(
        monkeypatch, tmp_path, config=config, codex_client=MidStreamOverflowCodexClient
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
    # ModelCatalogCache instead of constructing a fresh one per request:
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
            async for chunk in _rewrite_kimi_sse(response, "claude-fable-5")
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
            async for chunk in _rewrite_kimi_sse(response, "claude-fable-5"):
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


# --- /v1/messages routing: custom providers use the Responses backend ---


class StubOpenAICompatibleClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.context_window_calls: list[str] = []

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return None

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.payloads.append(payload)
        raise OpenAICompatibleUpstreamError(503, "stub custom upstream", "wrtn")
        yield


def test_custom_provider_route_uses_custom_client_without_builtin_payload_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_stub = StubOpenAICompatibleClient()
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
        reasoning_effort_override="max",
        codex_service_tier="fast",
    )

    def unexpected_grok_sanitizer(payload: dict[str, Any], model: str) -> dict[str, Any]:
        raise AssertionError("Grok sanitizer must not run for a custom provider")

    monkeypatch.setattr(relay, "sanitize_grok_payload", unexpected_grok_sanitizer)
    assert relay.sanitize_grok_payload is unexpected_grok_sanitizer
    client, codex_stub = _gateway(
        config,
        _failing_anthropic_handler,
        custom_provider_clients={"wrtn": custom_stub},
        codex_supports_fast_tier=True,
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert codex_stub.payloads == []
    assert custom_stub.context_window_calls == ["gpt-5.5"]
    (payload,) = custom_stub.payloads
    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"]["effort"] == "max"
    assert "service_tier" not in payload
    assert "max_output_tokens" not in payload


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


def test_grok_route_omits_codex_fast_service_tier() -> None:
    stub = StubGrokClient()
    client, _ = _gateway(
        _grok_config(codex_service_tier="fast"),
        _failing_anthropic_handler,
        grok_client=stub,
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert "service_tier" not in stub.payloads[0]


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

    monkeypatch.setattr(relay, "translate_claude_request_to_codex", _boom)
    assert relay.translate_claude_request_to_codex is _boom

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
        claude_request: dict[str, Any],
        upstream_model: str,
        reasoning_effort_override: str | None,
        *,
        service_tier: str | None = None,
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            reasoning_effort_override,
            service_tier=service_tier,
        )

    monkeypatch.setattr(relay, "translate_claude_request_to_codex", recording_translate)

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
    real_is_compaction_request = relay.is_compaction_request
    real_estimate = relay.estimate_overflow_prompt_tokens

    def counting_is_compaction_request(*args: Any, **kwargs: Any) -> bool:
        signal_calls["count"] += 1
        return real_is_compaction_request(*args, **kwargs)

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(relay, "is_compaction_request", counting_is_compaction_request)
    monkeypatch.setattr(relay, "estimate_overflow_prompt_tokens", counting_estimate)
    assert relay.is_compaction_request is counting_is_compaction_request
    assert relay.estimate_overflow_prompt_tokens is counting_estimate

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
    real_estimate = relay.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(relay, "estimate_overflow_prompt_tokens", counting_estimate)
    assert relay.estimate_overflow_prompt_tokens is counting_estimate

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
    real_estimate = relay.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(relay, "estimate_overflow_prompt_tokens", counting_estimate)
    assert relay.estimate_overflow_prompt_tokens is counting_estimate

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
        claude_request: dict[str, Any],
        upstream_model: str,
        reasoning_effort_override: str | None,
        *,
        service_tier: str | None = None,
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            reasoning_effort_override,
            service_tier=service_tier,
        )

    monkeypatch.setattr(relay, "translate_claude_request_to_codex", recording_translate)

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
    sequence = relay._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        stream_relay = _CompactionStreamRelay(upstream_response, app_state, sequence)
        first = await anext(stream_relay)
        assert first == b"event: a\ndata: {}\n\n"
        await stream_relay.aclose()

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
    sequence = relay._assign_compaction_reroute(
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
        async for _chunk in _CompactionStreamRelay(upstream_response, app_state, sequence):
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

    sequence_a = relay._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=100,
        context_window=50,
        detail=None,
    )
    relay._assign_compaction_reroute(
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
        async for _chunk in _CompactionStreamRelay(
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
    sequence = relay._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        stream_relay = _CompactionStreamRelay(upstream_response, app_state, sequence)
        assert await anext(stream_relay) == b"event: message_start\ndata: {}\n\n"
        terminal = await anext(stream_relay)
        assert terminal.startswith(b"\n\nevent: error\n")
        # No further __anext__: the record must already be upgraded and the
        # upstream response already released.
        assert app_state.compaction_last_reroute["outcome"] == "midstream_error"
        assert app_state.compaction_last_reroute["detail"] == "read_error"
        assert stream.closed is True
        await stream_relay.aclose()

    asyncio.run(scenario())


def test_relay_compaction_stream_aclose_before_first_iteration_closes_upstream() -> None:
    # A generator's finally would never run in this case; the owning
    # iterator must release the upstream response anyway.
    stream = _TrackedByteStream([b"event: a\ndata: {}\n\n"])
    upstream_response = httpx.Response(
        200, stream=stream, headers={"content-type": "text/event-stream"}
    )
    app_state = _committed_relay_state()
    sequence = relay._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )

    async def scenario() -> None:
        stream_relay = _CompactionStreamRelay(upstream_response, app_state, sequence)
        await stream_relay.aclose()

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
    sequence = relay._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=10,
        context_window=5,
        detail=None,
    )
    stream_relay = _CompactionStreamRelay(upstream_response, app_state, sequence)
    response = _OwnedStreamingResponse(stream_relay, media_type="text/event-stream")

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
        sse = _translate_claude_sse({"model": "claude-opus-4-6"}, upstream)
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
        async for _chunk in _translate_claude_sse({}, hanging_upstream()):
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

    response = asyncio.run(_aggregate_claude_response({}, upstream))

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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        monkeypatch, tmp_path,
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, codex_client=TimeoutCodexClient
    ) as client:
        response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    assert "codex backend" in response.json()["error"]["message"]


def test_transport_error_during_aggregation_returns_claude_502(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, codex_client=MidStreamErrorCodexClient
    ) as client:
        response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"


class TestAdminMappingApi:
    """GET/PUT /admin/settings/mapping — runtime map changes persisted to settings.json."""

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
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_current_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, model_map={"haiku": "codex:gpt-5.6-luna"}
        ) as client:
            response = client.get("/admin/settings/mapping")

        assert response.status_code == 200
        assert response.json()["model_map"] == {"haiku": "codex:gpt-5.6-luna"}

    def test_get_includes_custom_provider_metadata_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch,
            tmp_path,
            custom_providers={"wrtn": _custom_provider()},
        ) as client:
            response = client.get("/admin/settings/mapping")

        assert response.status_code == 200
        assert response.json()["custom_providers"] == [
            {
                "name": "wrtn",
                "wire_api": "responses",
                "base_url": "https://models.example/api/v1",
            }
        ]
        assert "api_key" not in response.text
        assert _CUSTOM_API_KEY not in response.text

    def test_get_without_custom_providers_keeps_existing_payload_shape(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.get("/admin/settings/mapping")

        assert "custom_providers" not in response.json()

    def test_put_updates_runtime_and_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )
            # The swap is live for later requests ...
            reread = client.get("/admin/settings/mapping")

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
                "/admin/settings/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )

        assert response.status_code == 409
        assert "CLAUDEX_MODEL_MAP" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/mapping",
                content='{"model_map": {}}',
                headers={"content-type": "text/plain"},
            )
        assert response.status_code == 415

    def test_put_validates_map_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/settings/mapping", json={"model_map": {"": "x"}})
        assert response.status_code == 400
        assert "non-empty strings" in response.json()["error"]["message"]

    def test_put_accepts_and_validates_provider_prefixes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        with self._admin_client(monkeypatch, tmp_path) as client:
            accepted = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kimi:k2.5"}}
            )
            bare_value = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "gpt-5.6-sol"}}
            )
            empty_model = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kimi:"}}
            )
            unknown_prefix = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kim:k2.5"}}
            )

        assert accepted.status_code == 200
        assert accepted.json()["model_map"] == {"opus": "kimi:k2.5"}
        assert bare_value.status_code == 400
        assert "no provider prefix" in bare_value.json()["error"]["message"]
        assert empty_model.status_code == 400
        assert "names no model" in empty_model.json()["error"]["message"]
        assert unknown_prefix.status_code == 400
        assert "unknown provider prefix" in unknown_prefix.json()["error"]["message"]

    def test_put_accepts_custom_provider_prefix_and_rejects_unknown_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        with self._admin_client(
            monkeypatch,
            tmp_path,
            custom_providers={"wrtn": _custom_provider()},
        ) as client:
            accepted = client.put(
                "/admin/settings/mapping",
                json={"model_map": {"opus": "wrtn:gpt-5.5"}},
            )
            rejected = client.put(
                "/admin/settings/mapping",
                json={"model_map": {"opus": "other:gpt-5.5"}},
            )

        assert accepted.status_code == 200
        assert accepted.json()["model_map"] == {"opus": "wrtn:gpt-5.5"}
        assert rejected.status_code == 400
        assert "unknown provider prefix" in rejected.json()["error"]["message"]

    def test_put_rejects_unknown_and_empty_bodies(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/settings/mapping", json={"bogus": {}}).status_code == 400
            assert client.put("/admin/settings/mapping", json={}).status_code == 400

    def test_admin_refuses_foreign_host_header(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.get("/admin/settings/mapping")
        assert response.status_code == 403
        assert "DNS-rebinding" in response.json()["error"]["message"]

    def test_admin_requires_local_token_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/settings/mapping").status_code == 401
            response = client.get(
                "/admin/settings/mapping", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200

    def test_get_reports_dashboard_facts_and_env_locks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "codex:gpt-5.6-luna"}')
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/mapping").json()

        assert payload["env_locked"] == {"model_map": "CLAUDEX_MODEL_MAP"}
        assert payload["codex_home"].endswith(".codex")
        assert payload["kimi_code_home"].endswith(".kimi-code")
        assert payload["grok_home"].endswith(".grok")


class TestAdminLogLevel:
    """GET/PUT /admin/settings/log-level — runtime log level applied and persisted."""

    @staticmethod
    def _admin_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        monkeypatch.delenv("CLAUDEX_LOG_LEVEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_current_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/log-level").json()

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
            for name in admin_api._LOG_LEVEL_LOGGER_NAMES
        }
        try:
            with self._admin_client(monkeypatch, tmp_path) as client:
                response = client.put("/admin/settings/log-level", json={"log_level": "debug"})
                reread = client.get("/admin/settings/log-level")

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
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/settings/log-level", json={"log_level": "debug"})

        assert response.status_code == 409
        assert "CLAUDEX_LOG_LEVEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_validates_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/settings/log-level", json={"log_level": "loud"}).status_code == 400
            assert client.put("/admin/settings/log-level", json={}).status_code == 400


class TestAdminCompactionApi:
    """GET/PUT /admin/settings/compaction — compaction reroute target, mirroring /admin/settings/mapping."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    # --- GET: fresh/configured state, diagnostics schema -------------------

    def test_get_returns_fresh_state_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload == {"model": None, "env_locked": False, "last_reroute": None}

    def test_get_returns_configured_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }

    def test_get_reports_last_reroute_with_exact_pinned_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            relay._assign_compaction_reroute(
                client.app.state,
                outcome="rerouted",
                target_model="claude-opus-5",
                mapped_model="codex:gpt-5.1-codex-max",
                estimated_prompt_tokens=4096,
                context_window=4000,
                detail=None,
            )
            payload = client.get("/admin/settings/compaction").json()

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
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload["env_locked"] is True

    def test_get_reports_env_locked_true_for_empty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload["env_locked"] is True

    # --- PUT: persistence, hot-swap, disable, enable/disable trigger -------

    def test_put_persists_without_clobbering_unrelated_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
            )
            reread = client.get("/admin/settings/compaction")

        assert response.status_code == 200
        assert response.json() == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }
        assert reread.json()["model"] == "claude:claude-opus-5"
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {
            "port": 9317,
            "compaction": {"model": "claude:claude-opus-5"},
        }

    def test_put_hot_swaps_live_config_for_subsequent_requests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            client.put("/admin/settings/compaction", json={"model": "claude:claude-opus-5"})
            assert client.app.state.config.compaction_model == "claude:claude-opus-5"
            reread = client.get("/admin/settings/compaction")

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
            response = client.put("/admin/settings/compaction", json={"model": None})
            reread = client.get("/admin/settings/compaction")

        assert response.status_code == 200
        assert response.json() == {"model": None, "env_locked": False, "last_reroute": None}
        assert reread.json()["model"] is None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 1234}
        assert "compaction" not in saved

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

        enable = client.put("/admin/settings/compaction", json={"model": "claude:claude-opus-5"})
        assert enable.status_code == 200

        rerouted = client.post(
            "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
        )
        assert rerouted.status_code == 200
        assert len(captured) == 1
        assert stub.payloads == []

        disable = client.put("/admin/settings/compaction", json={"model": None})
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
                "/admin/settings/compaction",
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
            response = client.put("/admin/settings/compaction", json={})
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
                "/admin/settings/compaction",
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
            response = client.put("/admin/settings/compaction", json={"model": "gpt-5"})
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
            response = client.put("/admin/settings/compaction", json={"model": value})
            config_after = client.app.state.config
        assert response.status_code == 400
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    # --- Security/lock: bearer + Host, for both GET and PUT ----------------

    def test_get_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/settings/compaction").status_code == 401

    def test_get_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer wrong"}
            )
        assert response.status_code == 401

    def test_get_succeeds_with_correct_bearer_on_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200

    def test_get_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 403

    def test_put_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put("/admin/settings/compaction", json={"model": None})
            config_after = client.app.state.config
        assert response.status_code == 401
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put(
                "/admin/settings/compaction",
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
                "/admin/settings/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer secret"},
            )
        assert response.status_code == 200

    def test_put_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.put(
                "/admin/settings/compaction",
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
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
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
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/settings/compaction", json={"model": None})
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
            raise admin_api.ConfigError("disk full")

        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setattr(admin_api, "update_settings_file", boom)
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
            )
            config_after = client.app.state.config

        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert "last_reroute" not in body
        assert config_after.compaction_model is None
        assert not (tmp_path / "settings.json").exists()


class TestAdminCodexApi:
    """GET/PUT /admin/settings/codex — Codex Fast service tier."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CODEX_SERVICE_TIER", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_default_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": None, "env_locked": False}

    def test_put_fast_persists_and_hot_swaps_live_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )
            config_after = client.app.state.config
            reread = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": "fast", "env_locked": False}
        assert reread.json() == {"service_tier": "fast", "env_locked": False}
        assert config_after.codex_service_tier == "fast"
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "port": 9317,
            "codex": {"service_tier": "fast"},
        }

    def test_put_null_removes_key_and_hot_swaps_live_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"port": 9317, "codex": {"service_tier": "fast"}}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, codex_service_tier="fast"
        ) as client:
            response = client.put(
                "/admin/settings/codex", json={"service_tier": None}
            )
            config_after = client.app.state.config
            reread = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": None, "env_locked": False}
        assert reread.json() == {"service_tier": None, "env_locked": False}
        assert config_after.codex_service_tier is None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317}
        assert "codex" not in saved

    @pytest.mark.parametrize(
        "body",
        [
            {"service_tier": "priority"},
            {"service_tier": 123},
            {},
        ],
    )
    def test_put_rejects_invalid_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        body: dict[str, Any],
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/settings/codex", json=body)

        assert response.status_code == 400
        assert "error" in response.json()
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize("env_value", ["fast", ""])
    def test_env_override_locks_get_and_put(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        env_value: str,
    ) -> None:
        monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", env_value)
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            get_response = client.get("/admin/settings/codex")
            put_response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )

        assert get_response.json() == {"service_tier": None, "env_locked": True}
        assert put_response.status_code == 409
        assert "CLAUDEX_CODEX_SERVICE_TIER" in put_response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_get_and_put_require_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            get_response = client.get("/admin/settings/codex")
            put_response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )

        assert get_response.status_code == 401
        assert put_response.status_code == 401
        assert not (tmp_path / "settings.json").exists()


class TestAdminClaudeAccountApi:
    """GET/PUT /admin/providers/claude/pool/serving — serving-account selection, mirroring
    /admin/settings/compaction. The shared bearer/Host guard paths are exhaustively
    covered by the compaction tests; here one auth check pins the guard is
    wired at all, and the rest exercises the account-specific logic."""

    _ACCOUNT_ID = "0a1b2c3d-4e5f-4678-9abc-def012345678"

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_fresh_state_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload == {"account_id": None, "env_locked": False}

    def test_get_returns_configured_account(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload == {"account_id": self._ACCOUNT_ID, "env_locked": False}

    def test_get_reports_env_locked_even_for_empty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", "")
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload["env_locked"] is True

    def test_put_registered_account_persists_and_hot_swaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            account_id = _register_serving_account()
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": account_id}
            )
            reread = client.get("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"account_id": account_id, "env_locked": False}
        assert reread.json()["account_id"] == account_id
        assert config_after.claude_account_id == account_id
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317, "claude_account": {"id": account_id}}

    def test_put_null_is_rejected_pointing_at_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.id": self._ACCOUNT_ID}), encoding="utf-8"
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            response = client.put("/admin/providers/claude/pool/serving", json={"account_id": None})
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "DELETE" in response.json()["error"]["message"]
        assert config_after.claude_account_id == self._ACCOUNT_ID
        assert "claude_account.id" in json.loads(settings_file.read_text())

    def test_delete_clears_the_setting_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.id": self._ACCOUNT_ID}), encoding="utf-8"
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            response = client.delete("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"account_id": None, "env_locked": False}
        assert config_after.claude_account_id is None
        assert "claude_account" not in json.loads(settings_file.read_text())

    def test_delete_when_already_clear_is_a_no_op_200(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.delete("/admin/providers/claude/pool/serving")

        assert response.status_code == 200
        assert response.json() == {"account_id": None, "env_locked": False}

    def test_put_unregistered_account_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": self._ACCOUNT_ID}
            )
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "no account registered" in response.json()["error"]["message"]
        assert config_after.claude_account_id is None
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize(
        "value", ["not-a-uuid", "0A1B2C3D-4E5F-4678-9ABC-DEF012345678", 1, True, [], {}]
    )
    def test_put_invalid_account_id_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Any
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/providers/claude/pool/serving", json={"account_id": value})

        assert response.status_code == 400

    def test_put_rejects_unknown_and_missing_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            unknown = client.put("/admin/providers/claude/pool/serving", json={"account": "x"})
            missing = client.put("/admin/providers/claude/pool/serving", json={})

        assert unknown.status_code == 400
        assert missing.status_code == 400

    def test_put_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            account_id = _register_serving_account()
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", self._ACCOUNT_ID)
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": account_id}
            )
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.claude_account_id is None

    def test_delete_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", self._ACCOUNT_ID)
            response = client.delete("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.claude_account_id == self._ACCOUNT_ID

    def test_endpoints_require_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            get_response = client.get("/admin/providers/claude/pool/serving")
            put_response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": None}
            )

        assert get_response.status_code == 401
        assert put_response.status_code == 401
        assert not (tmp_path / "settings.json").exists()


class TestAdminClaudeRoutingApi:
    """GET/PUT /admin/providers/claude/pool/routing — the pool policy document."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_defaults_to_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/providers/claude/pool/routing").json()

        assert payload == {"mode": "disabled", "env_locked": False}

    def test_put_fallback_persists_the_policy_document_and_hot_swaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "fallback"}
            )
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"mode": "fallback", "env_locked": False}
        assert config_after.claude_account_routing_mode == "fallback"
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"claude_account": {"routing": {"mode": "fallback"}}}

    def test_put_disabled_removes_the_settings_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.routing": {"mode": "fallback"}}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_routing_mode="fallback"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"mode": "disabled", "env_locked": False}
        assert config_after.claude_account_routing_mode == "disabled"
        assert "claude_account" not in json.loads(settings_file.read_text())

    def test_put_balanced_with_unknown_keys_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # "balanced" is a fully valid mode now (T-10); a future-shaped
        # document still reports the real "unknown keys" reason, not a stale
        # "not implemented" one, and the exact mode-envelope shape is
        # unchanged.
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing",
                json={"mode": "balanced", "balanced": {"window": "session"}},
            )

        assert response.status_code == 400
        assert "unknown keys: balanced" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"mode": "round-robin"},
            {"mode": None},
            {"mode": True},
            {"mode": "fallback", "weights": [1]},
        ],
    )
    def test_put_garbage_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict[str, Any]
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/providers/claude/pool/routing", json=body)

        assert response.status_code == 400, body
        assert not (tmp_path / "settings.json").exists()

    def test_put_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", '{"mode": "fallback"}')
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            locked = client.get("/admin/providers/claude/pool/routing").json()

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ROUTING" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert locked["env_locked"] is True


def test_admin_logs_returns_recent_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        logging.getLogger("claudex_gateway.test").warning("hello %s", "world")
        response = client.get("/admin/logs")

    assert response.status_code == 200
    entries = [e for e in response.json()["logs"] if e["message"] == "hello world"]
    assert entries and entries[0]["level"] == "WARNING"
    assert entries[0]["logger"] == "claudex_gateway.test"


def test_admin_logs_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.get("/admin/logs").status_code == 403


def test_admin_usage_returns_all_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    async def fake_claude(http_client: Any) -> dict[str, Any]:
        calls.append("claude")
        return {"provider": "claude", "status": "ok", "error": None}

    async def fake_codex(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("codex")
        return {"provider": "codex", "status": "unavailable", "error": "no creds"}

    async def fake_kimi(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("kimi")
        return {"provider": "kimi", "status": "ok", "error": None}

    async def fake_grok(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("grok")
        return {"provider": "grok", "status": "ok", "error": None}

    monkeypatch.setattr(admin_api, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(admin_api, "fetch_codex_usage", fake_codex)
    monkeypatch.setattr(admin_api, "fetch_kimi_usage", fake_kimi)
    monkeypatch.setattr(admin_api, "fetch_grok_usage", fake_grok)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert body["codex"]["status"] == "unavailable"
    assert body["kimi"]["status"] == "ok"
    assert body["grok"]["status"] == "ok"
    assert body["fetched_at"] > 0
    assert calls == ["claude", "codex", "kimi", "grok"]


def test_admin_usage_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.get("/admin/usage").status_code == 403


def test_admin_usage_single_provider_skips_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_claude(http_client: Any) -> dict[str, Any]:
        return {"provider": "claude", "status": "ok", "error": None}

    async def codex_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("codex probe ran for ?provider=claude")

    async def kimi_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("kimi probe ran for ?provider=claude")

    async def grok_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("grok probe ran for ?provider=claude")

    monkeypatch.setattr(admin_api, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(admin_api, "fetch_codex_usage", codex_must_not_run)
    monkeypatch.setattr(admin_api, "fetch_kimi_usage", kimi_must_not_run)
    monkeypatch.setattr(admin_api, "fetch_grok_usage", grok_must_not_run)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "claude"})

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert "codex" not in body
    assert "kimi" not in body
    assert "grok" not in body


def test_admin_usage_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
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

    monkeypatch.setattr(admin_api, "consume_codex_reset_credit", fake_consume)
    return keys


def test_admin_reset_credit_returns_the_backend_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keys = _record_reset_keys(
        monkeypatch, [{"status": "ok", "outcome": "reset", "error": None}]
    )
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.post("/admin/providers/codex/reset-credit", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "outcome": "reset", "error": None}
    assert len(keys) == 1 and keys[0]


def test_admin_reset_credit_reuses_the_key_until_an_attempt_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        for _ in range(4):
            assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 200

    assert keys[0] == keys[1] == keys[2], "unsettled attempts must retry the same key"
    assert keys[3] != keys[2], "a settled attempt must not reuse its key"


def test_admin_reset_credit_is_guarded_like_every_other_admin_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a guarded request reached the ChatGPT backend")

    monkeypatch.setattr(admin_api, "consume_codex_reset_credit", must_not_run)

    # Foreign Host header (DNS-rebinding guard).
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 403
    # A form post would dodge the CORS preflight, so JSON is required.
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        assert client.post("/admin/providers/codex/reset-credit", content="x").status_code == 415
    # And the local bearer token still applies.
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(
        monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 401


def test_admin_reset_credit_is_never_reachable_by_GET(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Spending a credit must not be possible by navigating to a URL.
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a GET spent a reset credit")

    monkeypatch.setattr(admin_api, "consume_codex_reset_credit", must_not_run)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/admin/providers/codex/reset-credit").status_code == 405


def test_dashboard_usage_merged_into_status_cards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Usage renders inside the Status tab's provider cards: a per-provider
    # body hook, the fetch on entering Status, and no separate tab.
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings tab is the settings home: it leads the tab bar, opens by
    # default, and holds the Compact Reroute row behind the category rail;
    # the provider status cards live in the Status tab.
    with _create_test_client(monkeypatch, tmp_path) as client:
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


def test_dashboard_optional_providers_hidden_until_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Kimi and Grok are extensions: their Status cards and the Router
    # add-node provider buttons ship hidden and are revealed from /health
    # only when a local login is detected — or when the map already routes
    # to them, where hiding a required-login error would mislead. The gating
    # is cosmetic only; routing, settings.json, and the admin API are
    # untouched.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert '<div class="card provider-hidden" id="card-kimi">' in page
    assert '<div class="card provider-hidden" id="card-grok">' in page
    # Codex is built in and never hides.
    assert 'id="card-codex"' not in page
    assert "function setProviderVisibility(" in page
    assert 'info.status==="ok"||info.required===true' in page
    # The Router provider picker builds optional providers hidden too.
    assert '''(p==="codex"?"":' class="provider-hidden"')''' in page
    # Bulk usage refresh only probes visible cards.
    assert "PROVIDER_VISIBLE[p]!==false" in page


def test_dashboard_plan_and_credits_read_inside_the_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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


def test_dashboard_served_at_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Claudex Gateway" in response.text
    assert "/admin/settings/mapping" in response.text


def test_favicon_served_for_browser_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "max-age" in response.headers["cache-control"]
    assert response.text.startswith("<svg")


def test_dashboard_port_has_an_enlarged_invisible_hit_zone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The 14px visible port dot is too small to drag from: the board widens
    # the grab area to the node's full height plus margins without changing
    # the visual. These markers are the whole mechanism, so pin them.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert ".node.src .port::after" in page
    assert "pointer-events:auto" in page


def test_hello_reports_identity_and_auth_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        body = client.get("/api/hello").json()

    assert body["hello"] == "claudex-gateway"
    assert body["local_auth_required"] is False
    assert isinstance(body["pid"], int)


def test_hello_reports_auth_required_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(local_token="secret-token-value")
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        response = client.get("/api/hello")

    assert response.json()["local_auth_required"] is True
    assert "secret-token-value" not in response.text


def test_dashboard_keeps_the_local_token_in_memory_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    # Catalogs are fetched live; a baked-in model list goes stale and misleads
    # when the catalog request fails. Nodes already in the mapping still render
    # via buildColumns.
    assert "CODEX_FALLBACK" not in page
    assert "gpt-5" not in page
    assert "CATALOG={codex:[],kimi:[],grok:[]}" in page


def test_dashboard_board_shows_only_referenced_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    # With several providers a catalog dump is unusable as a board, so target
    # nodes are only what the map references plus what the add-node box
    # stages; the catalogs survive purely as autocomplete for that box.
    assert "concat(Object.values(DIR.mapping),addedTargets)" in page
    assert 'list="add-catalog"' in page
    # All provider catalogs feed it, so the dashboard depends on the Kimi
    # and Grok endpoints too — not just the Codex one.
    assert '"/admin/providers/kimi/models"' in page
    assert '"/admin/providers/grok/models"' in page


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


def _routing_section(page: str) -> str:
    """Slice the routing card's own markup — "balanced" legitimately appears
    elsewhere in the document (API test prose never, but future copy might),
    so absence is asserted against this scoped slice."""
    start = page.index("<!-- routing-section:start -->")
    end = page.index("<!-- routing-section:end -->")
    return page[start:end]


def test_dashboard_compaction_section_marker_and_endpoint_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert "<!-- compaction-section:start -->" in page
    assert "<!-- compaction-section:end -->" in page
    section = _compaction_section(page)
    assert 'id="compaction-card"' in section
    assert "/admin/settings/compaction" in page


def test_dashboard_compaction_options_in_pinned_order_without_haiku(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    section = _compaction_section(page)
    assert "unverified until first use" in section
    assert 'id="comp-custom-input"' in section


def test_dashboard_compaction_credentials_disclosure_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The card must state which credentials rerouted requests run on, so the
    # user knows their own Claude account is being used.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    section = _compaction_section(page)
    assert "장치에 저장된 Claude 기본 자격증명" in section


def test_dashboard_compaction_fetched_in_parallel_boot_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    boot_start = page.index("function boot(){")
    promise_all = page.index("Promise.all([", boot_start)
    promise_all_end = page.index("]);", promise_all)
    parallel_calls = page[promise_all:promise_all_end]
    assert 'jfetch("/admin/settings/compaction")' in parallel_calls


def test_dashboard_compaction_keeps_configured_custom_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    # A configured model that is not one of the three curated ids renders as
    # Custom with its raw id filled in, instead of being dropped.
    assert "COMPACTION_CURATED_MODELS.indexOf(id)>=0" in page
    assert '{kind:"custom",custom:id}' in page


def test_dashboard_compaction_diagnostics_ui_removed_by_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings redesign dropped the last-reroute record from the page:
    # diagnostics stay reachable through GET /admin/settings/compaction only. Guard
    # against the UI quietly returning.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert 'id="comp-diagnostics"' not in page
    assert "renderCompactionDiagnostics" not in page
    assert "아직 재라우팅이 시도되지 않았습니다" not in page


def test_dashboard_compaction_apply_body_matches_pinned_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert "JSON.stringify({model:model})" in page
    apply_fn = _compaction_apply_fn(page)
    # Disabled sends exactly {"model": null}; curated/custom selections carry
    # the "claude:" prefix parse_compaction_model expects.
    assert "model=null" in apply_fn
    assert '"claude:"+raw' in apply_fn
    assert '"claude:"+COMP.draftKind' in apply_fn


def test_dashboard_compaction_custom_submission_is_trimmed_and_guarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    apply_fn = _compaction_apply_fn(page)
    assert "input.value.trim()" in apply_fn
    assert "if(!raw)return;" in apply_fn


def test_dashboard_compaction_409_branch_refreshes_via_get_not_error_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
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
    assert 'jfetch("/admin/settings/compaction")' in locked_branch


def test_dashboard_compaction_409_refresh_failure_stays_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # If the post-409 refresh GET itself fails or returns a malformed
    # envelope, its body must not be rendered as state and the card must
    # remain locked.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    apply_fn = _compaction_apply_fn(page)
    branch_start = apply_fn.index("r.status===409")
    branch_end = apply_fn.index("if(!r.ok){", branch_start)
    locked_branch = apply_fn[branch_start:branch_end]

    refresh_start = locked_branch.index('jfetch("/admin/settings/compaction")')
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


def _codex_section(page: str) -> str:
    start = page.index("<!-- codex-section:start -->")
    end = page.index("<!-- codex-section:end -->")
    return page[start:end]


def test_dashboard_codex_fast_card_wires_apply_flow_and_env_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    section = _codex_section(page)
    assert page.index('id="compaction-card"') < page.index('id="codex-card"')
    assert 'type="checkbox" id="codex-fast"' in section
    assert 'id="codex-apply"' in section
    assert "~1.5x speed" in section
    assert "~2–2.5x usage burn" in section
    assert "silently stay standard" in section
    assert "CLAUDEX_CODEX_SERVICE_TIER" in page
    assert 'jfetch("/admin/settings/codex")' in page
    assert 'jfetch("/admin/settings/codex",{' in page
    assert 'JSON.stringify({service_tier:CODEX.draft?"fast":null})' in page
    assert "checkbox.disabled=CODEX.locked" in page
    assert "btn.disabled=CODEX.locked||CODEX.draft===(CODEX.serviceTier===\"fast\")" in page


def test_dashboard_codex_fast_fetched_in_parallel_boot_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    boot_start = page.index("function boot(){")
    promise_all = page.index("Promise.all([", boot_start)
    promise_all_end = page.index("]);", promise_all)
    parallel_calls = page[promise_all:promise_all_end]
    assert 'jfetch("/admin/settings/codex")' in parallel_calls


def test_dashboard_settings_rail_switches_categories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings rail is real category switching now: one .scard visible at
    # a time, gated by data-cat on the section, deep-linkable per category.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert '<section id="tab-settings" data-cat="general">' in page
    assert 'href="#settings/general"' in page
    assert 'href="#settings/accounts"' in page
    assert ".scard{display:none}" in page
    assert 'id="scard-general"' in page
    assert 'id="scard-accounts"' in page
    assert "function setSettingsCat(" in page
    # General leads the rail and the card order; the accounts card follows.
    assert page.index('href="#settings/general"') < page.index(
        'href="#settings/accounts"'
    )
    assert page.index('id="scard-general"') < page.index('id="scard-accounts"')
    # A #settings/accounts deep link lands on the category at boot and on
    # hash changes.
    assert 'if(bootTab==="settings"&&bootParts[1])setSettingsCat(bootParts[1])' in page
    assert 'if(parts[0]==="settings")setSettingsCat(parts[1]||"general")' in page


def test_dashboard_accounts_card_mirrors_the_final_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Composition per _design-probes/_accounts-final.html: the local CLI hero
    # leads (the only boxed area), then the 등록 계정 caption with the add
    # button, then dense flat rows that expand independently.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert 'class="lhero"' in page
    assert "로컬 CLI 로그인" in page
    assert "게이트웨이 서빙과 무관" in page
    assert 'id="btn-local-refresh"' in page
    assert 'id="btn-add-account"' in page
    assert (
        page.index('id="local-sec"')
        < page.index('id="acct-count"')
        < page.index('id="acct-list"')
    )
    # Collapsed rows carry status text only (no chips, no mini bars); the
    # right edge is the plan text.
    assert "서빙 중" in page
    assert "재로그인 필요" in page
    assert 'class="plan-txt"' in page
    # Expansion is independent per-row state, never an accordion.
    assert "ACCT.open[id]=!ACCT.open[id]" in page
    # Plan pill mapping: claude_max -> MAX, claude_pro -> PRO, null -> dash.
    assert "function planLabel(" in page
    assert 'replace(/^claude_/,"").toUpperCase()' in page


def test_dashboard_accounts_fetch_paints_registry_before_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The registry GET paints rows immediately; the cache-backed usage GET
    # and the local hero's ambient usage fill in async afterwards. No force
    # parameter exists — the UI shows data age instead.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert 'jfetch("/admin/providers/claude/accounts")' in page
    assert 'jfetch("/admin/providers/claude/pool/usage")' in page
    assert 'jfetch("/admin/usage?provider=claude")' in page
    assert "force" not in page.split('jfetch("/admin/providers/claude/pool/usage")')[1][:200]
    fetch_fn = page[page.index("function fetchAccounts(") :]
    assert fetch_fn.index("renderAcctList()") < fetch_fn.index("fetchAccountUsage()")
    # Data age renders from each result's updated_at.
    assert "function fmtAgo(" in page
    assert "기준" in page


def test_dashboard_serving_selection_reuses_the_singular_admin_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 사용/해제 go through the existing PUT /admin/providers/claude/pool/serving; a 409
    # env-lock renders the lockband and disables the buttons.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert 'jfetch("/admin/providers/claude/pool/serving",{' in page
    assert "JSON.stringify({account_id:accountId})" in page
    assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in page
    assert 'id="acct-lockband"' in page
    assert "#scard-accounts.locked .acctlock{display:block}" in page
    assert "이 계정으로 서빙" in page
    assert "서빙 해제" in page
    # Removal uses the account endpoint; the serving pin guard stays visible in the UI.
    assert 'jfetch("/admin/providers/claude/accounts/"+encodeURIComponent(accountId),{method:"DELETE"})' in page


def test_dashboard_routing_section_wires_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert "<!-- routing-section:start -->" in page
    assert "<!-- routing-section:end -->" in page
    # Policy row sits in General, directly after the compaction row.
    assert page.index('id="compaction-card"') < page.index('id="routing-card"')
    # Boot GET and the apply PUT both target the pool/routing endpoint, and
    # the PUT body is pinned to exactly {"mode": ...}.
    assert 'jfetch("/admin/providers/claude/pool/routing")' in page
    assert 'jfetch("/admin/providers/claude/pool/routing",{' in page
    assert "JSON.stringify({mode:ROUTING.draft})" in page
    section = _routing_section(page)
    assert ">Disabled<" in section
    assert ">Fallback<" in section
    # balanced is a fully valid server-side mode (config.py's
    # VALID_CLAUDE_ACCOUNT_ROUTING_MODES) and is offered here too.
    assert 'value="balanced"' in section
    assert ">Balanced<" in section
    assert "계정별 라우팅 상태 보기" in section
    # The mode-envelope validator adopts a balanced boot GET/apply response
    # instead of rejecting it as malformed.
    assert 'body.mode==="balanced"' in page


def test_dashboard_accounts_surface_pool_usage_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Balanced-mode usage reads are cache-only (T-13): the accounts screen
    # renders pool/status's usage_freshness chip plus each window's
    # pool/usage observation age/source, and a queued manual refresh renders
    # its own indication instead of claiming a completed refresh.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert "ACCT.usageFreshness" in page
    assert "statusResp.body.usage_freshness" in page
    assert 'id="pool-fresh-pill"' in page
    assert "function renderPoolFreshness(" in page
    assert "meta.age_seconds" in page
    assert "meta.source" in page
    assert "u.queued" in page
    assert "대기 중" in page


def test_dashboard_routing_locked_renders_readonly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A 409 flips the local lock; the lockband names the env var and the
    # control disables — same discipline as the compaction card.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert "CLAUDEX_CLAUDE_ACCOUNT_ROUTING" in page
    assert 'id="routing-lock-env"' in page
    assert (
        "#compaction-card.locked .complock,#codex-card.locked .complock,"
        "#routing-card.locked .complock{display:block}" in page
    )
    apply_fn = page[
        page.index("function applyRouting(){") : page.index(
            'document.getElementById("routing-select").addEventListener("change"'
        )
    ]
    assert "r.status===409" in apply_fn
    assert "ROUTING.locked=true" in apply_fn


def test_dashboard_accounts_surface_routing_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Row badges come from pool/status; a failed status GET renders no badges
    # (never stale), and the cooldown row explains itself in the detail pane.
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    assert 'jfetch("/admin/providers/claude/pool/status")' in page
    assert "라우팅 준비" in page
    assert "라우팅 불가" in page
    assert "쿨다운 · " in page
    assert "coolnote" in page
    assert "function fmtCooldownUntil(" in page
    # The accounts card links back to the policy row in General.
    assert (
        '라우팅 정책은 <a class="route-link" href="#settings/general">General</a>에서 설정합니다.'
        in page
    )


def test_dashboard_login_modal_drives_the_login_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        page = client.get("/").text

    # All five login endpoints are wired: start, poll, code, confirm, cancel.
    assert 'jfetch("/admin/providers/claude/login",{\n    method:"POST"' in page.replace(
        "\r\n", "\n"
    ) or 'method:"POST",headers:{"Content-Type":"application/json"},body:"{}"' in page
    assert 'jfetch("/admin/providers/claude/login")' in page
    assert "/admin/providers/claude/login/code" in page
    assert "/admin/providers/claude/login/replace" in page
    assert 'method:"DELETE"' in page
    # 409 login-active attaches to the running session instead of erroring.
    assert 'code==="login-active"' in page
    # The replace confirmation shows exactly the CLI's two choices.
    assert "교체 안 함" in page
    assert 'data-lact="replace"' in page
    assert 'data-lact="decline"' in page
    # The URL opens in a new tab with rel=noopener and an https-only guard.
    assert 'rel="noopener"' in page
    assert "/^https:\\/\\//.test(st.url||" in page
    # Anti-churn: the body re-renders only when the session state changes.
    assert "if(key===LOGIN.lastKey)return" in page
    # Explicit cancel only — neither a backdrop click nor Escape ever closes
    # the login modal (a silent close would leak a live CLI login child).
    assert "closeLoginModal()" in page
    assert "target===this)closeLoginModal()" not in page
    assert 'Escape")closeLoginModal()' not in page


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


class CatalogOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def list_models(self) -> list[str]:
        return ["gpt-5.5", "gemini-3.1-pro"]


class ProbeOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class RejectingOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise OpenAICompatibleUpstreamError(
            400, '{"error":{"message":"model_not_found"}}', self.name
        )


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
    def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> TestClient:
        return _create_test_client(
            monkeypatch, tmp_path, base_url="http://127.0.0.1:8787", **kwargs
        )

    def test_codex_models_returns_visible_slugs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=CatalogCodexClient) as client:
            response = client.get("/admin/providers/codex/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["gpt-5.6-sol", "gpt-5.5"]}

    def test_codex_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=FailingCatalogCodexClient) as client:
            response = client.get("/admin/providers/codex/models")

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "unsupported client"

    def test_codex_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, codex_client=CatalogCodexClient) as client:
            assert client.get("/admin/providers/codex/models").status_code == 403

    def test_kimi_models_relays_catalog_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The catalog passes through unshaped: the map bypasses model IDs
        # untouched, so the raw backend answer is the preset source.
        with self._client(monkeypatch, tmp_path, kimi_client=CatalogKimiClient) as client:
            response = client.get("/admin/providers/kimi/models")

        assert response.status_code == 200
        assert response.json() == {"data": [{"id": "k2.5"}, {"id": "k3"}]}

    def test_kimi_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, kimi_client=FailingCatalogKimiClient) as client:
            response = client.get("/admin/providers/kimi/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_kimi_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, kimi_client=CatalogKimiClient) as client:
            assert client.get("/admin/providers/kimi/models").status_code == 403

    def test_grok_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=CatalogGrokClient) as client:
            response = client.get("/admin/providers/grok/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["grok-4.5", "grok-4.3"]}

    def test_grok_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=FailingCatalogGrokClient) as client:
            response = client.get("/admin/providers/grok/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_grok_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, grok_client=CatalogGrokClient) as client:
            assert client.get("/admin/providers/grok/models").status_code == 403

    def test_custom_provider_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            custom_client=CatalogOpenAICompatibleClient,
        ) as client:
            response = client.get("/admin/providers/custom/wrtn/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["gpt-5.5", "gemini-3.1-pro"]}

    def test_custom_provider_models_returns_404_for_unknown_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(monkeypatch, tmp_path, config=config) as client:
            response = client.get("/admin/providers/custom/other/models")

        assert response.status_code == 404
        assert "not configured" in response.json()["error"]["message"]

    def test_connection_test_ok(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=ProbeCodexClient) as client:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=ProbeCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "gpt-5.6-luna"},
            )

        assert response.status_code == 400
        assert "no provider prefix" in response.json()["error"]["message"]

    def test_connection_test_kimi_target_probes_kimi(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, kimi_client=ProbeKimiClient) as client:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, kimi_client=RejectingKimiClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kimi:k3"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 404
        assert "model not found" in result["detail"]

    def test_connection_test_grok_target_probes_grok(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=ProbeGrokClient) as client:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=RejectingGrokClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "grok:grok-nope"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert "model_not_found" in result["detail"]

    def test_connection_test_custom_target_probes_custom_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            custom_client=ProbeOpenAICompatibleClient,
        ) as client:
            response = client.post("/admin/test", json={"target": "wrtn:gpt-5.5"})

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "gpt-5.5"

    def test_connection_test_reports_custom_provider_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            custom_client=RejectingOpenAICompatibleClient,
        ) as client:
            response = client.post("/admin/test", json={"target": "wrtn:gpt-nope"})

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert result["detail"] == "model_not_found"

    def test_connection_test_rejects_unknown_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kim:k3"},
            )

        assert response.status_code == 400
        assert "unknown provider prefix" in response.json()["error"]["message"]

    def test_connection_test_reports_unknown_codex_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=RejectingCodexClient) as client:
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path) as client:
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


# ---------------------------------------------------------------------------
# Anthropic passthrough served with a registered Claude account
# ---------------------------------------------------------------------------


def _register_serving_account(
    *,
    email: str = "pool@example.com",
    account_uuid: str | None = "serving-account-uuid",
    access_token: str = "pool-access-1",
    refresh_token: str = "pool-refresh-1",
    expires_in_seconds: float = 3600,
) -> str:
    """Register one ready account under the (HOME-isolated) registry.

    Distinct emails register distinct accounts (the registry deduplicates on
    the (email, organizationUuid) pair), which is how pool tests build a
    multi-account chain.
    """
    oauth_account: dict[str, Any] = {"emailAddress": email}
    if account_uuid is not None:
        oauth_account["accountUuid"] = account_uuid
    record = claude_accounts.add_account(
        email=email,
        organization_uuid="org-1",
        organization_name="Example Org",
        credentials_json={
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": (time.time() + expires_in_seconds) * 1000,
                "scopes": ["user:inference", "user:profile"],
            }
        },
        oauth_account_json=oauth_account,
    )
    return record.id


def _claude_code_user_id(account_uuid: str = "client-account-uuid") -> str:
    return json.dumps(
        {
            "device_id": "d" * 64,
            "account_uuid": account_uuid,
            "session_id": "11111111-2222-4333-8444-555555555555",
        },
        separators=(",", ":"),
    )


def _account_body(model: str = "claude-fable-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {"user_id": _claude_code_user_id()}
    return body


def test_account_passthrough_swaps_credentials_and_rewrites_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_1"}, headers={"request-id": "req_1"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post(
        "/v1/messages",
        content=json.dumps(_account_body()),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-oat01-client",
            "x-api-key": "client-api-key",
            "anthropic-beta": "claude-code-20250219",
        },
    )

    assert response.status_code == 200
    assert response.headers["request-id"] == "req_1"
    (upstream,) = captured
    assert upstream.headers["authorization"] == "Bearer pool-access-1"
    assert "x-api-key" not in upstream.headers
    betas = [beta.strip() for beta in upstream.headers["anthropic-beta"].split(",")]
    assert "claude-code-20250219" in betas
    assert "oauth-2025-04-20" in betas
    forwarded = json.loads(upstream.content)
    forwarded_user_id = json.loads(forwarded["metadata"]["user_id"])
    assert forwarded_user_id["account_uuid"] == "serving-account-uuid"
    assert forwarded_user_id["session_id"] == "11111111-2222-4333-8444-555555555555"
    assert forwarded_user_id["device_id"] == "d" * 64
    # Everything except the metadata rewrite is byte-for-byte the client's body.
    assert forwarded["model"] == "claude-fable-5"
    assert forwarded["messages"] == _account_body()["messages"]


def test_account_passthrough_strips_account_uuid_when_capture_recorded_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account(account_uuid=None)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 200
    (upstream,) = captured
    forwarded_user_id = json.loads(json.loads(upstream.content)["metadata"]["user_id"])
    # Forwarding the client's own uuid with a pool token would name another
    # account; with no recorded serving uuid the field is stripped instead.
    assert "account_uuid" not in forwarded_user_id
    assert forwarded_user_id["session_id"] == "11111111-2222-4333-8444-555555555555"


def test_account_passthrough_forwards_non_claude_code_metadata_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)
    body = _message_body("claude-fable-5")
    body["metadata"] = {"user_id": "not-a-json-string"}
    raw = json.dumps(body)

    response = client.post(
        "/v1/messages", content=raw, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    (upstream,) = captured
    assert upstream.content == raw.encode()
    assert upstream.headers["authorization"] == "Bearer pool-access-1"


def test_account_passthrough_refreshes_and_retries_once_on_401(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    api_calls: list[str] = []
    token_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CLAUDE_TOKEN_URL:
            token_calls.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 900,
                },
            )
        api_calls.append(request.headers["authorization"])
        if len(api_calls) == 1:
            return httpx.Response(401, json={"type": "error"})
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 200
    assert api_calls == ["Bearer pool-access-1", "Bearer rotated-access"]
    assert len(token_calls) == 1
    assert token_calls[0]["refresh_token"] == "pool-refresh-1"
    # The rotated single-use refresh token is persisted for the next request.
    persisted = json.loads(
        (
            claude_accounts.paths.accounts_dir("claude") / account_id / "credentials.json"
        ).read_text()
    )["claudeAiOauth"]
    assert persisted["refreshToken"] == "rotated-refresh"


def test_account_passthrough_post_retry_401_reports_the_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CLAUDE_TOKEN_URL:
            return httpx.Response(
                200, json={"access_token": "rotated-access", "expires_in": 900}
            )
        return httpx.Response(401, json={"type": "error"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["type"] == "authentication_error"
    assert "account add" in error["message"]


def test_account_passthrough_unregistered_account_returns_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen without an account")

    client, _ = _gateway(
        GatewayConfig(claude_account_id="0a1b2c3d-4e5f-4678-9abc-def012345678"), handler
    )

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 503
    assert "not registered" in response.json()["error"]["message"]


def test_account_passthrough_invalid_grant_marks_needs_reauth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # An already-expired token forces the proactive refresh before any
    # Anthropic call, and that refresh conclusively fails (invalid_grant).
    account_id = _register_serving_account(expires_in_seconds=-60)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CLAUDE_TOKEN_URL
        return httpx.Response(400, json={"error": "invalid_grant"})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 503
    assert "needs re-authentication" in response.json()["error"]["message"]
    [record] = claude_accounts.load_registry()
    assert record.state == "needs-reauth"


def test_account_passthrough_transient_refresh_failure_returns_503_without_marking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account(expires_in_seconds=-60)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CLAUDE_TOKEN_URL
        return httpx.Response(500, text="token endpoint down")

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 503
    assert "unusable" in response.json()["error"]["message"]
    [record] = claude_accounts.load_registry()
    assert record.state == "ready"


def test_account_passthrough_serves_count_tokens_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"input_tokens": 42})

    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages/count_tokens", json=_account_body())

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    (upstream,) = captured
    assert str(upstream.url) == "https://api.anthropic.com/v1/messages/count_tokens"
    assert upstream.headers["authorization"] == "Bearer pool-access-1"


def test_unset_account_keeps_passthrough_forwarding_client_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A registered account alone must change nothing: only claude_account.id
    # switches the passthrough into managed relay.
    monkeypatch.setenv("HOME", str(tmp_path))
    _register_serving_account()
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(GatewayConfig(), handler)

    response = client.post(
        "/v1/messages",
        content=json.dumps(_account_body()),
        headers={
            "content-type": "application/json",
            "authorization": "Bearer sk-ant-oat01-client",
        },
    )

    assert response.status_code == 200
    (upstream,) = captured
    assert upstream.headers["authorization"] == "Bearer sk-ant-oat01-client"
    forwarded_user_id = json.loads(json.loads(upstream.content)["metadata"]["user_id"])
    assert forwarded_user_id["account_uuid"] == "client-account-uuid"


# ---------------------------------------------------------------------------
# Account pool: ordered fallback on the managed relay
# ---------------------------------------------------------------------------


def _register_pool_accounts(
    monkeypatch: pytest.MonkeyPatch, *, first_expires_in_seconds: float = 3600
) -> tuple[str, str]:
    """Register two ready accounts with deterministic registration order.

    `_now_millis` is replaced with a counter because two registrations can
    land in the same real millisecond, which would leave the createdAt chain
    order to the random UUID tiebreak.
    """
    millis = iter(range(1_700_000_000_000, 1_700_000_000_100))
    monkeypatch.setattr(claude_accounts, "_now_millis", lambda: next(millis))
    first = _register_serving_account(expires_in_seconds=first_expires_in_seconds)
    second = _register_serving_account(
        email="pool2@example.com",
        account_uuid="second-account-uuid",
        access_token="pool-access-2",
        refresh_token="pool-refresh-2",
    )
    return first, second


def _pool_config(serving_account_id: str) -> GatewayConfig:
    """Fallback-mode config: the pool section exercises multi-account routing."""
    return GatewayConfig(
        claude_account_id=serving_account_id, claude_account_routing_mode="fallback"
    )


def _quota_429(marker: str = "Error") -> httpx.Response:
    """The empirically observed OAuth quota rejection: no reset signal at all."""
    return httpx.Response(
        429,
        json={"type": "error", "error": {"type": "rate_limit_error", "message": marker}},
        headers={"x-should-retry": "true"},
    )


def test_routing_disabled_by_default_relays_429_without_touching_other_accounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _quota_429("relayed")

    # No claude_account.routing configured: multi-account routing stays off.
    client, _ = _gateway(GatewayConfig(claude_account_id=first), handler)

    first_response = client.post("/v1/messages", json=_account_body())
    assert first_response.status_code == 429
    assert first_response.json()["error"]["message"] == "relayed"
    assert [call.headers["authorization"] for call in calls] == ["Bearer pool-access-1"]

    # No cooldown bookkeeping either: the next request probes upstream again
    # instead of answering from a synthesized cooldown 429.
    second_response = client.post("/v1/messages", json=_account_body())
    assert second_response.status_code == 429
    assert "retry-after" not in second_response.headers
    assert not client.app.state.claude_account_cooldowns.is_cooling(first)
    assert [call.headers["authorization"] for call in calls] == [
        "Bearer pool-access-1",
        "Bearer pool-access-1",
    ]


def test_pool_fails_over_to_second_account_on_429(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 200
    assert response.json() == {"id": "msg_1"}
    assert [call.headers["authorization"] for call in calls] == [
        "Bearer pool-access-1",
        "Bearer pool-access-2",
    ]
    # The metadata rewrite names each attempt's own account, never a stale one.
    for call, expected_uuid in zip(calls, ["serving-account-uuid", "second-account-uuid"]):
        user_id = json.loads(json.loads(call.content)["metadata"]["user_id"])
        assert user_id["account_uuid"] == expected_uuid


def test_pool_cooldown_persists_across_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)

    assert client.post("/v1/messages", json=_account_body()).status_code == 200
    calls.clear()

    # The first account is cooling down now: the next request goes straight
    # to the second account without probing the rate-limited one again.
    assert client.post("/v1/messages", json=_account_body()).status_code == 200
    assert [call.headers["authorization"] for call in calls] == ["Bearer pool-access-2"]


def test_pool_fails_back_after_cooldown_expiry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    first_recovered = [False]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["authorization"] == "Bearer pool-access-1" and not first_recovered[0]:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)
    now = [1_000.0]
    client.app.state.claude_account_cooldowns = AccountCooldownTracker(clock=lambda: now[0])

    assert client.post("/v1/messages", json=_account_body()).status_code == 200
    calls.clear()
    first_recovered[0] = True

    # Once the cooldown (default 60s — the quota 429 carried no signal)
    # expires, the serving account is preferred again.
    now[0] += 61.0
    assert client.post("/v1/messages", json=_account_body()).status_code == 200
    assert [call.headers["authorization"] for call in calls] == ["Bearer pool-access-1"]


def test_pool_exhausted_replays_final_upstream_429(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        marker = "first" if request.headers["authorization"] == "Bearer pool-access-1" else "second"
        return _quota_429(marker)

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_body())

    # Every account 429'd: the client sees the last real Anthropic rejection,
    # which Claude Code knows how to render.
    assert response.status_code == 429
    assert response.json()["error"]["message"] == "second"
    assert response.headers["x-should-retry"] == "true"


def test_all_cooling_returns_429_with_retry_after_without_contacting_upstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _quota_429()

    client, _ = _gateway(_pool_config(first), handler)

    assert client.post("/v1/messages", json=_account_body()).status_code == 429
    upstream_probes = len(calls)
    assert upstream_probes == 2

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 429
    assert "rate-limited" in response.json()["error"]["message"]
    assert int(response.headers["retry-after"]) >= 1
    assert len(calls) == upstream_probes  # nothing touched upstream this time


def test_pool_429_cooldown_uses_cached_usage_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)
    envelope = {
        "provider": "claude",
        "status": "ok",
        "fable_weekly": {
            "used_percent": 100.0,
            "window_minutes": 10_080,
            "resets_at": time.time() + 7_200.0,
        },
    }
    monkeypatch.setattr(
        client.app.state.claude_account_usage_cache,
        "peek",
        lambda account_id: envelope if account_id == first else None,
    )

    assert client.post("/v1/messages", json=_account_body()).status_code == 200

    remaining = client.app.state.claude_account_cooldowns.remaining_seconds(first)
    assert 7_000.0 < remaining <= 7_200.0


def test_pool_fails_over_when_refresh_requires_reauth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # The serving account's token is already expired, and its refresh
    # conclusively fails — the pool marks it and serves from the next one.
    first, second = _register_pool_accounts(monkeypatch, first_expires_in_seconds=-60)
    api_calls: list[str] = []
    token_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CLAUDE_TOKEN_URL:
            token_calls.append(json.loads(request.content))
            return httpx.Response(400, json={"error": "invalid_grant"})
        api_calls.append(request.headers["authorization"])
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_body())

    assert response.status_code == 200
    assert api_calls == ["Bearer pool-access-2"]
    assert [call["refresh_token"] for call in token_calls] == ["pool-refresh-1"]
    states = {record.id: record.state for record in claude_accounts.load_registry()}
    assert states == {first: "needs-reauth", second: "ready"}


def test_pool_fails_over_on_post_retry_401_and_marks_reauth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, second = _register_pool_accounts(monkeypatch)
    api_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CLAUDE_TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 900,
                },
            )
        api_calls.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer pool-access-2":
            return httpx.Response(200, json={"id": "msg_1"})
        return httpx.Response(401, json={"type": "error"})

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_body())

    # The serving account got its full 401 → refresh → retry-once treatment
    # before the pool moved on and durably marked it.
    assert response.status_code == 200
    assert api_calls == ["Bearer pool-access-1", "Bearer rotated-access", "Bearer pool-access-2"]
    states = {record.id: record.state for record in claude_accounts.load_registry()}
    assert states == {first: "needs-reauth", second: "ready"}


def test_pool_transport_error_returns_502_without_failover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("network down", request=request)

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_body())

    # A transport failure is not account-specific: retrying the next account
    # would double down on a dead network, so it stays terminal.
    assert response.status_code == 502
    assert len(calls) == 1


def test_pool_serves_count_tokens_failover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"input_tokens": 42})

    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages/count_tokens", json=_account_body())

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 42}
    assert [str(call.url) for call in calls] == [
        "https://api.anthropic.com/v1/messages/count_tokens",
        "https://api.anthropic.com/v1/messages/count_tokens",
    ]


def test_pool_single_account_429_replays_upstream_body_and_cools_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _quota_429("single")

    client, _ = _gateway(_pool_config(account_id), handler)

    first_response = client.post("/v1/messages", json=_account_body())
    assert first_response.status_code == 429
    assert first_response.json()["error"]["message"] == "single"
    assert len(calls) == 1

    # In-cooldown requests answer from the tracker without touching upstream.
    second_response = client.post("/v1/messages", json=_account_body())
    assert second_response.status_code == 429
    assert int(second_response.headers["retry-after"]) >= 1
    assert len(calls) == 1


def test_pool_forwards_non_claude_code_body_unchanged_on_failover(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(_pool_config(first), handler)
    body = _message_body("claude-fable-5")
    body["metadata"] = {"user_id": "not-a-json-string"}
    raw = json.dumps(body)

    response = client.post(
        "/v1/messages", content=raw, headers={"content-type": "application/json"}
    )

    assert response.status_code == 200
    assert [call.content for call in calls] == [raw.encode(), raw.encode()]


def test_pool_skips_unregistered_serving_id_when_other_ready_accounts_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _register_serving_account()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    client, _ = _gateway(
        _pool_config("0a1b2c3d-4e5f-4678-9abc-def012345678"), handler
    )

    response = client.post("/v1/messages", json=_account_body())

    # A stale serving selection must not take the whole pool down.
    assert response.status_code == 200
    assert [call.headers["authorization"] for call in calls] == ["Bearer pool-access-1"]


# ---------------------------------------------------------------------------
# Admin claude-accounts surface (dashboard account management)
# ---------------------------------------------------------------------------


_ADMIN_BASE = "http://127.0.0.1:8787"


class TestAdminClaudeAccountsApi:
    @staticmethod
    def _client(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **config_kwargs: Any
    ) -> TestClient:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        return _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(
                settings_file=tmp_path / "settings.json", **config_kwargs
            ),
            base_url=_ADMIN_BASE,
        )

    def test_list_returns_only_registry_rows_without_secrets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()

        with client:
            response = client.get("/admin/providers/claude/accounts")

        assert response.status_code == 200
        payload = response.json()
        # The collection alone: local login, serving pin, and cooldown
        # telemetry each live at their own endpoint now.
        assert set(payload) == {"accounts"}
        [row] = payload["accounts"]
        assert set(row) == {
            "id",
            "email",
            "organizationUuid",
            "organizationName",
            "createdAt",
            "updatedAt",
            "lastAuthenticatedAt",
            "state",
            "accountIncarnationId",
            "upstreamAccountUuid",
            "planType",
            "rateLimitTier",
        }
        assert row["id"] == account_id
        assert row["state"] == "ready"
        # _register_serving_account's oauth-account fixture has no plan
        # fields, so the derived plan metadata degrades to nulls.
        assert row["planType"] is None
        assert row["rateLimitTier"] is None
        # The registry response must never leak credential material.
        assert "accessToken" not in response.text
        assert "refreshToken" not in response.text
        assert "pool-access" not in response.text
        assert "pool-refresh" not in response.text

    def test_pool_status_reports_ready_cooldown_and_unavailable_members(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        millis = iter(range(1_700_000_000_000, 1_700_000_000_100))
        monkeypatch.setattr(claude_accounts, "_now_millis", lambda: next(millis))
        ready_id = _register_serving_account()
        cooling_id = _register_serving_account(
            email="pool2@example.com",
            account_uuid="second-account-uuid",
            access_token="pool-access-2",
            refresh_token="pool-refresh-2",
        )
        dead_id = _register_serving_account(
            email="pool3@example.com",
            account_uuid="third-account-uuid",
            access_token="pool-access-3",
            refresh_token="pool-refresh-3",
        )
        claude_accounts.mark_account_needs_reauth(dead_id)
        client.app.state.claude_account_cooldowns.mark(cooling_id, 120.0)

        with client:
            response = client.get("/admin/providers/claude/pool/status")

        assert response.status_code == 200
        members = {member["account_id"]: member for member in response.json()["members"]}
        assert members[ready_id] == {"account_id": ready_id, "routing_state": "ready"}
        cooling = members[cooling_id]
        assert cooling["routing_state"] == "cooldown"
        # Epoch ms, like every other registry timestamp in the payload.
        expected = (time.time() + 120.0) * 1000
        assert abs(cooling["cooldown_until"] - expected) < 5_000
        assert members[dead_id] == {
            "account_id": dead_id,
            "routing_state": "unavailable",
            "reason": "needs-reauth",
        }

    def test_list_derives_plan_fields_from_the_captured_oauth_account(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        claude_accounts.add_account(
            email="max@example.com",
            organization_uuid="org-max",
            organization_name="Max Org",
            credentials_json={"claudeAiOauth": {"accessToken": "at"}},
            oauth_account_json={
                "emailAddress": "max@example.com",
                "organizationType": "claude_max",
                "organizationRateLimitTier": "default_claude_max_20x",
            },
        )

        with client:
            [row] = client.get("/admin/providers/claude/accounts").json()["accounts"]

        assert row["planType"] == "claude_max"
        assert row["rateLimitTier"] == "default_claude_max_20x"

    def test_local_reports_the_ambient_login_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The dashboard's 로컬 CLI 로그인 hero reads this block: identity and
        # plan metadata from the CLI's own ~/.claude.json — never secrets.
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        client = self._client(monkeypatch, tmp_path)
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        (home / ".claude.json").write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "accountUuid": "11111111-2222-3333-4444-555555555555",
                        "emailAddress": "local@example.com",
                        "organizationName": "Local Org",
                        "organizationType": "claude_max",
                        "organizationRateLimitTier": "default_claude_max_20x",
                    }
                }
            ),
            encoding="utf-8",
        )

        with client:
            payload = client.get("/admin/providers/claude/local").json()

        assert payload["local"] == {
            "accountUuid": "11111111-2222-3333-4444-555555555555",
            "email": "local@example.com",
            "organizationName": "Local Org",
            "planType": "claude_max",
            "rateLimitTier": "default_claude_max_20x",
        }

    def test_local_degrades_to_null_without_a_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        client = self._client(monkeypatch, tmp_path)

        with client:
            payload = client.get("/admin/providers/claude/local").json()

        assert payload == {"local": None}

    def test_usage_serves_from_the_cache_after_the_first_fetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        calls: list[str] = []

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        with client:
            first = client.get("/admin/providers/claude/pool/usage")
            second = client.get("/admin/providers/claude/pool/usage")

        assert first.status_code == 200
        assert first.json()["accounts"][account_id]["status"] == "ok"
        assert second.json()["accounts"][account_id]["status"] == "ok"
        assert calls == [account_id]

    def test_usage_skips_needs_reauth_accounts_without_fetching(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        claude_accounts.mark_account_needs_reauth(account_id)

        async def fail_fetch(http_client: Any, manager: Any) -> None:
            raise AssertionError("needs-reauth accounts must not fetch usage")

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fail_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        result = response.json()["accounts"][account_id]
        assert result["status"] == "unavailable"
        assert "re-authentication" in result["error"]

    def test_usage_unknown_account_filter_is_a_400(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        _register_serving_account()
        with client:
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": "99999999-9999-4999-8999-999999999999"},
            )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/admin/providers/claude/accounts"),
            ("GET", "/admin/providers/claude/pool/usage"),
            ("GET", "/admin/providers/claude/login"),
            ("POST", "/admin/providers/claude/login"),
            ("DELETE", "/admin/providers/claude/login"),
            ("POST", "/admin/providers/claude/login/code"),
            ("POST", "/admin/providers/claude/login/replace"),
        ],
    )
    def test_foreign_host_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        method: str,
        path: str,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url="http://evil.example",
        )
        with client:
            response = client.request(method, path, json={})
        assert response.status_code == 403

    def test_login_post_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # code/replace content-type handling is covered in the lifecycle
        # class — those endpoints check the attempt guard first, so they
        # need an attached session to reach the 415.
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.post(
                "/admin/providers/claude/login",
                content="{}",
                headers={"content-type": "text/plain"},
            )
        assert response.status_code == 415

    def test_login_get_without_a_session_is_idle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            assert client.get("/admin/providers/claude/login").json() == {"status": "idle"}

    def test_login_commands_without_a_session_are_409_stale_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # With no session there is no attempt any command could name — the
        # guard answers before body or state validation ever runs.
        client = self._client(monkeypatch, tmp_path)
        with client:
            code = client.post("/admin/providers/claude/login/code", json={"code": "x"})
            confirm = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
            )
            cancel = client.delete("/admin/providers/claude/login")
        for response in (code, confirm, cancel):
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "stale_login"

    def test_login_post_rejects_a_non_empty_body(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.post("/admin/providers/claude/login", json={"mode": "x"})
        assert response.status_code == 400

    def test_login_post_conflicts_with_a_held_capture_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        from claudex_gateway.claude_login_session import capture_lock_path
        from claudex_gateway.locking import try_file_lock as _try_lock

        handle = _try_lock(capture_lock_path())
        assert handle is not None
        try:
            with client:
                response = client.post("/admin/providers/claude/login", json={})
        finally:
            handle.release()
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "login-locked"

    def test_account_delete_removes_the_registry_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        client.app.state.claude_account_cooldowns.mark(account_id, 120.0)

        with client:
            # Prime the cached auth manager so the delete has one to drop.
            client.app.state.claude_account_auth_managers[account_id] = object()
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert response.status_code == 204
        assert response.content == b""
        assert claude_accounts.list_accounts() == []
        assert account_id not in client.app.state.claude_account_auth_managers
        assert not client.app.state.claude_account_cooldowns.is_cooling(account_id)

    def test_account_delete_unknown_id_is_404(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.delete(
                "/admin/providers/claude/accounts/99999999-9999-4999-8999-999999999999"
            )
        assert response.status_code == 404

    def test_account_delete_serving_account_is_409(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        account_id = _register_serving_account()
        client = self._client(monkeypatch, tmp_path, claude_account_id=account_id)

        with client:
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert response.status_code == 409
        assert "pool/serving" in response.json()["error"]["message"]
        assert [record.id for record in claude_accounts.list_accounts()] == [account_id]

    def test_account_delete_requires_the_admin_guard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        account_id = "99999999-9999-4999-8999-999999999999"
        client = self._client(monkeypatch, tmp_path, local_token="secret")
        with client:
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")
        assert response.status_code == 401


# The atomic cutover removed every pre-reorg admin path with no aliases:
# a request to an old path must 404, never silently hit a moved handler.
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/mapping"),
        ("PUT", "/admin/mapping"),
        ("GET", "/admin/log-level"),
        ("PUT", "/admin/log-level"),
        ("GET", "/admin/compaction"),
        ("PUT", "/admin/compaction"),
        ("GET", "/admin/claude-account"),
        ("PUT", "/admin/claude-account"),
        ("GET", "/admin/claude-accounts"),
        ("GET", "/admin/claude-accounts/usage"),
        ("GET", "/admin/claude-accounts/login"),
        ("POST", "/admin/claude-accounts/login"),
        ("DELETE", "/admin/claude-accounts/login"),
        ("POST", "/admin/claude-accounts/login/code"),
        ("POST", "/admin/claude-accounts/login/confirm"),
        ("GET", "/admin/codex/models"),
        ("POST", "/admin/codex/reset-credit"),
        ("GET", "/admin/kimi/models"),
        ("GET", "/admin/grok/models"),
    ],
)
def test_pre_reorg_admin_paths_are_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str, path: str
) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url=_ADMIN_BASE) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 404


class TestAdminClaudeLoginLifecycle:
    """End-to-end login sessions against the PATH-prepended fake claude.

    Every test runs the client as a context manager: the login driver task
    lives on the lifespan portal's event loop, which only exists while the
    `with` block is open.
    """

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        import sys as _sys

        from fake_claude import prepend_fake_claude

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        monkeypatch.setattr(_sys, "platform", "linux")
        prepend_fake_claude(monkeypatch, tmp_path)
        return _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )

    @staticmethod
    def _poll_until(client: TestClient, predicate: Any, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.get("/admin/providers/claude/login").json()
            if predicate(status):
                return status
            time.sleep(0.05)
        raise AssertionError(f"login status never satisfied the predicate: {status}")

    def test_full_login_flow_with_code_submission(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-url-code")
        client = self._client(monkeypatch, tmp_path)

        with client:
            started = client.post("/admin/providers/claude/login", json={})
            assert started.status_code == 201
            envelope = started.json()
            assert envelope["status"] == "starting"
            attempt = envelope["attempt_id"]
            attempt_header = {"X-Login-Attempt": attempt}

            status = self._poll_until(
                client, lambda s: s["status"] == "awaiting-browser"
            )
            assert status["attempt_id"] == attempt
            assert status["url"].startswith("https://claude.com/cai/oauth/authorize")
            assert status["code_prompt_detected"] is True
            assert status["expires_at"] is not None

            submitted = client.post(
                "/admin/providers/claude/login/code",
                json={"code": "good-code"},
                headers=attempt_header,
            )
            assert submitted.status_code == 200

            status = self._poll_until(client, lambda s: s["status"] == "succeeded")
            assert status["account"]["email"] == "fixture@example.com"

            rows = client.get("/admin/providers/claude/accounts").json()["accounts"]
            assert [row["email"] for row in rows] == ["fixture@example.com"]

    def test_second_login_while_active_is_409_then_cancel_converges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            attempt_header = {"X-Login-Attempt": attempt}
            conflict = client.post("/admin/providers/claude/login", json={})
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "login-active"

            cancelled = client.delete(
                "/admin/providers/claude/login", headers=attempt_header
            )
            assert cancelled.json() == {"status": "cancelling"}
            self._poll_until(client, lambda s: s["status"] == "cancelled")

            # A terminal session clears on DELETE and frees the slot.
            assert client.delete(
                "/admin/providers/claude/login", headers=attempt_header
            ).json() == {"status": "idle"}
            assert client.get("/admin/providers/claude/login").json() == {
                "status": "idle"
            }

    def test_stale_attempt_commands_are_409_stale_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            stale_header = {"X-Login-Attempt": "0" * 32}

            for response in (
                client.get("/admin/providers/claude/login", headers=stale_header),
                client.post(
                    "/admin/providers/claude/login/code",
                    json={"code": "late-code"},
                    headers=stale_header,
                ),
                client.post(
                    "/admin/providers/claude/login/replace",
                    json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
                    headers=stale_header,
                ),
                client.delete("/admin/providers/claude/login", headers=stale_header),
            ):
                assert response.status_code == 409
                assert response.json()["error"]["code"] == "stale_login"

            # A bare GET is discovery: it still answers, exposing the live
            # attempt so a fresh tab can re-attach.
            bare = client.get("/admin/providers/claude/login")
            assert bare.status_code == 200
            assert bare.json()["attempt_id"] == attempt

            # The stale commands drove nothing: the session is still alive.
            client.delete(
                "/admin/providers/claude/login", headers={"X-Login-Attempt": attempt}
            )
            self._poll_until(client, lambda s: s["status"] == "cancelled")

    def test_duplicate_login_confirm_replace_via_the_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-autocomplete")
        client = self._client(monkeypatch, tmp_path)
        original = claude_accounts.add_account(
            "fixture@example.com",
            None,
            None,
            {"claudeAiOauth": {"accessToken": "old-token"}},
            None,
        )

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            status = self._poll_until(
                client, lambda s: s["status"] == "awaiting-replace"
            )
            assert status["email"] == "fixture@example.com"
            assert status["existing_account_id"] == original.id

            mismatched = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
                headers={"X-Login-Attempt": attempt},
            )
            assert mismatched.status_code == 409
            assert "does not match" in mismatched.json()["error"]["message"]

            confirmed = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": original.id},
                headers={"X-Login-Attempt": attempt},
            )
            assert confirmed.status_code == 200
            status = self._poll_until(client, lambda s: s["status"] == "succeeded")
            assert status["account"]["id"] == original.id

        [record] = claude_accounts.load_registry()
        assert record.id == original.id
        assert record.state == "ready"

    def test_attached_login_code_body_validation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Body validation sits behind the attempt guard, so it is exercised
        # through an attached session.
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            attempt_header = {"X-Login-Attempt": attempt}

            wrong_type = client.post(
                "/admin/providers/claude/login/code",
                content="{}",
                headers={**attempt_header, "content-type": "text/plain"},
            )
            assert wrong_type.status_code == 415
            wrong_type = client.post(
                "/admin/providers/claude/login/replace",
                content="{}",
                headers={**attempt_header, "content-type": "text/plain"},
            )
            assert wrong_type.status_code == 415

            for bad_body in (
                {},
                {"code": ""},
                {"code": "  "},
                {"code": "line\nbreak"},
                {"code": 7},
                {"code": "ok", "extra": 1},
            ):
                response = client.post(
                    "/admin/providers/claude/login/code",
                    json=bad_body,
                    headers=attempt_header,
                )
                assert response.status_code == 400, bad_body

            for bad_body in (
                {},
                {"existing_account_id": ""},
                {"existing_account_id": None},
                {"existing_account_id": 7},
                {"replace": True},
                {"existing_account_id": "x", "extra": 1},
            ):
                response = client.post(
                    "/admin/providers/claude/login/replace",
                    json=bad_body,
                    headers=attempt_header,
                )
                assert response.status_code == 400, bad_body

            client.delete("/admin/providers/claude/login", headers=attempt_header)
            self._poll_until(client, lambda s: s["status"] == "cancelled")

    def test_duplicate_login_decline_is_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-autocomplete")
        client = self._client(monkeypatch, tmp_path)
        original = claude_accounts.add_account(
            "fixture@example.com",
            None,
            None,
            {"claudeAiOauth": {"accessToken": "old-token"}},
            None,
        )

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            self._poll_until(client, lambda s: s["status"] == "awaiting-replace")

            declined = client.delete(
                "/admin/providers/claude/login",
                headers={"X-Login-Attempt": attempt},
            )
            assert declined.json() == {"status": "cancelling"}
            self._poll_until(client, lambda s: s["status"] == "cancelled")

        [record] = claude_accounts.load_registry()
        assert record == original


# ---------------------------------------------------------------------------
# Claude account pool lease: acquired at lifespan startup and held for the
# process lifetime, regardless of routing mode (T-9)
# ---------------------------------------------------------------------------


def test_lifespan_holds_the_claude_pool_lease_for_the_process_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lock_path = paths.claude_account_pool_lock()

    with _create_test_client(monkeypatch, tmp_path) as client:
        assert lock_path.is_file()
        # Contended while the lifespan-backed client is open.
        assert server.try_file_lock(lock_path) is None
        assert client.get("/api/hello").status_code == 200

    # Released once the client (and its lifespan) has shut down.
    reacquired = server.try_file_lock(lock_path)
    assert reacquired is not None
    reacquired.release()


def test_lifespan_raises_the_pinned_message_when_the_pool_lock_is_contended(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lock_path = paths.claude_account_pool_lock()

    holder = server.try_file_lock(lock_path)
    assert holder is not None
    try:
        client = _create_test_client(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError) as exc_info:
            with client:
                pass
        assert str(exc_info.value) == (
            "claude account pool is already served by another process "
            "(balanced-router.lock held)"
        )
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# Balanced routing lifecycle: transactional mode changes, restart-preserving
# shutdown, fail-closed dispatch (T-10)
# ---------------------------------------------------------------------------

_BALANCED_ACCOUNT_UUID = "11111111-2222-3333-4444-555555555555"


def _balanced_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate HOME under `tmp_path` and clear the env-lock override, exactly
    like `TestAdminClaudeRoutingApi._admin_client` -- so registry/pool state
    lives under an isolated, per-test `.claudex` directory."""
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _register_balanced_ready_account(
    *, email: str = "balanced@example.com", account_uuid: str | None = _BALANCED_ACCOUNT_UUID
) -> str:
    """Register one ready account. With the default canonical `account_uuid`, it
    carries a valid T-3 profile fingerprint and can enable balanced routing; a
    caller after `account_uuid=None` (or any non-UUID string, e.g.
    `_register_serving_account`'s own default) gets one that cannot.
    """
    return _register_serving_account(email=email, account_uuid=account_uuid)


class TestBalancedRoutingEnable:
    """PUT .../pool/routing {"mode": "balanced"} -- prepares the complete runtime
    (store, epoch, router, T-3 fingerprint verification) while the old mode keeps
    serving, and persists settings only once every check passes (T-10 Steps 1-7).
    """

    def test_put_balanced_enables_with_valid_fingerprints(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            runtime = client.app.state.claude_balanced_runtime
            config_after = client.app.state.config
            # Read while the runtime is still live -- lifespan shutdown resets
            # `status` to "disabled" on `with`-block exit (shutdown_preserving_epoch).
            status_while_active = runtime.status
            epoch_id_while_active = runtime.epoch_id
            router_while_active = runtime.router

        assert response.status_code == 200
        assert response.json() == {"mode": "balanced", "env_locked": False}
        assert config_after.claude_account_routing_mode == "balanced"
        assert status_while_active == "active"
        assert epoch_id_while_active is not None
        assert router_while_active is not None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"claude_account": {"routing": {"mode": "balanced"}}}

    def test_put_balanced_rejects_invalid_account_uuid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        # The default account_uuid ("serving-account-uuid") is not a UUID, so
        # this account carries no T-3 profile fingerprint.
        _register_balanced_ready_account(account_uuid="serving-account-uuid")
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            runtime = client.app.state.claude_balanced_runtime
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "profile_fingerprint" in response.json()["error"]["message"]
        assert config_after.claude_account_routing_mode == "disabled"
        assert runtime.status == "disabled"
        assert not settings_file.exists()

    def test_put_balanced_rejects_store_open_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)

        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated store-open failure")

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "open_", classmethod(_boom))
                response = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
                )
            runtime = client.app.state.claude_balanced_runtime

        assert response.status_code == 500
        assert "could not enable balanced routing" in response.json()["error"]["message"]
        assert runtime.status == "disabled"
        assert not settings_file.exists()

    def test_put_balanced_tears_down_before_settings_persist_on_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Publication ordering (Context): a failure at the persist() commit point
        tears preparation down and leaves the old mode -- and a clean retry
        (store closed, no lingering lock) then succeeds.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("simulated disk-full settings write")

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            with monkeypatch.context() as fault:
                fault.setattr(admin_api, "update_settings_file", _boom)
                failed = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
                )
            runtime = client.app.state.claude_balanced_runtime
            status_after_failure = runtime.status
            config_after_failure = client.app.state.config
            settings_missing_after_failure = not settings_file.exists()

            retried = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            status_after_retry = runtime.status

        assert failed.status_code == 500
        assert status_after_failure == "disabled"
        assert config_after_failure.claude_account_routing_mode == "disabled"
        assert settings_missing_after_failure
        assert retried.status_code == 200
        assert status_after_retry == "active"

    def test_lifespan_activates_persisted_balanced_mode_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.routing": {"mode": "balanced"}}),
            encoding="utf-8",
        )
        config = GatewayConfig.load(settings_file)
        assert config.claude_account_routing_mode == "balanced"

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = client.app.state.claude_balanced_runtime
            payload = client.get("/admin/providers/claude/pool/routing").json()
            status_while_active = runtime.status
            epoch_id_while_active = runtime.epoch_id

        assert status_while_active == "active"
        assert epoch_id_while_active is not None
        assert payload == {"mode": "balanced", "env_locked": False}

    def test_get_reports_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            payload = client.get("/admin/providers/claude/pool/routing").json()

        assert enable.status_code == 200
        assert payload == {"mode": "balanced", "env_locked": False}


def test_balanced_dispatch_fails_closed_when_runtime_not_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-closed dispatch (Step 6): "balanced" persisted/published with no
    active runtime (never prepared -- the lifespan never ran) is the
    *inconsistent* state, reserved for the 503 -- never a silent fallback to
    single-account routing.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("balanced dispatch must never reach the upstream here")

    client, _ = _gateway(
        GatewayConfig(claude_account_routing_mode="balanced"), handler
    )

    response = client.post("/v1/messages", json=_message_body("claude-fable-5"))

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "balanced routing is not active"


class TestBalancedRoutingExit:
    """Intentional balanced -> fallback/disabled exit (T-10 Step 8): drains
    in-flight dispatch, persists+publishes the target mode, invalidates the
    current epoch's pins, and starts a fresh epoch on any later re-entry --
    contrast `test_graceful_shutdown_preserves_epoch_and_pin_for_restart`
    below, where a process shutdown preserves that very same state instead.
    """

    def test_exit_drains_persists_invalidates_pins_and_reentry_gets_new_epoch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            epoch_id_1 = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1

            runtime_db_path = paths.claude_account_pool_runtime_db()

            exited = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            served_after_exit = client.post("/v1/messages", json=_account_body())

            assert exited.status_code == 200
            assert exited.json() == {"mode": "disabled", "env_locked": False}
            assert runtime.status == "disabled"
            # Draining resolved cleanly: the in-flight-created pin's request, and
            # a fresh one after exit (now served single-account, "disabled"), both
            # completed rather than erroring.
            assert served_after_exit.status_code == 200
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert "claude_account" not in saved

            # The exited epoch's pins are gone at the persistence layer too --
            # not just discarded in memory with this runtime instance.
            inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
            try:
                restore_result = inspect_store.restore(
                    RestoreValidationContext(now_utc=time.time())
                )
                epoch_id_after_exit = inspect_store.balanced_epoch_id
            finally:
                inspect_store.close()
            assert epoch_id_after_exit != epoch_id_1
            assert restore_result.pins == {}

            reentered = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert reentered.status_code == 200
            assert runtime.status == "active"
            assert runtime.epoch_id != epoch_id_1
            # T-22 (fix for gap G-7): the admin re-entry path durably mints its
            # OWN fresh epoch, independent of the one `exit_mode` already
            # rotated to above -- it never simply reuses whatever the store
            # happens to hold.
            assert runtime.epoch_id != epoch_id_after_exit
            assert runtime.router is not None
            assert runtime.router.pin_count() == 0

    def test_exit_persistence_failure_aborts_and_leaves_epoch_and_pins_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Crash contract (T-20, fix for gap G-3): a settings-persistence
        failure during exit must abort BEFORE anything balanced-only is
        touched -- the PUT returns 500, the runtime resumes "active", and
        the epoch id and its durable pins are exactly as they were, still
        serving requests under balanced with the same pins.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise admin_api.ConfigError("simulated disk-full settings write")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            epoch_id_before = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1

            with monkeypatch.context() as fault:
                fault.setattr(admin_api, "update_settings_file", _boom)
                failed = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )

            assert failed.status_code == 500
            assert runtime.status == "active"
            assert runtime.epoch_id == epoch_id_before
            assert client.app.state.config.claude_account_routing_mode == "balanced"
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert saved["claude_account"]["routing"] == {"mode": "balanced"}

            # The durable pin the persistence failure was supposed to leave
            # untouched is still there, both in memory and at the durable
            # layer -- a subsequent request keeps being served under
            # balanced, reusing it rather than a fresh placement.
            assert runtime._store is not None
            session_key = derive_session_key(_account_body(), runtime.epoch_seed, "fable")
            assert session_key is not None
            assert runtime.router.get_pin(session_key.digest) is not None
            assert runtime._store.get_pin(session_key.digest) is not None

            served = client.post("/v1/messages", json=_account_body())
            assert served.status_code == 200
            assert runtime.router.pin_count() == 1

    def test_exit_epoch_rotation_failure_after_persistence_surfaces_persistence_degraded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Crash contract (T-20, fix for gap G-3): once `persist()` has
        already committed the target mode, a subsequent epoch-rotation
        failure must NOT roll back to balanced -- the exit still completes
        under the target mode, with the cleanup failure only surfaced as
        persistence_degraded.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom_rotate(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated durable epoch rotation failure")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        caplog.set_level(logging.WARNING, logger="claudex_gateway.claude_balanced_router")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime

            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "rotate_epoch", _boom_rotate)
                exited = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )

            assert exited.status_code == 200
            assert exited.json() == {"mode": "disabled", "env_locked": False}
            assert runtime.status == "disabled"
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert "claude_account" not in saved

        assert any(
            "persistence_degraded" in record.getMessage() for record in caplog.records
        )

    def test_admin_reenable_after_degraded_exit_rotation_mints_a_fresh_epoch_and_drops_the_old_pin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Gap G-7(b) regression: `exit_mode`'s own epoch rotation can fail
        (`persistence_degraded`, exercised above) and leave the runtime DB
        holding the exited epoch's id/seed and its durable pin. A later admin
        re-enable must never resurrect that epoch or pin --
        `prepare_and_publish`'s `entry="admin_enable"` branch durably mints a
        fresh epoch (wiping pins) before it ever restores from the store,
        regardless of what the store still contains.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom_rotate(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated durable epoch rotation failure")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            old_epoch_id = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1
            session_key = derive_session_key(_account_body(), runtime.epoch_seed, "fable")
            assert session_key is not None
            old_pin_digest = session_key.digest

            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "rotate_epoch", _boom_rotate)
                exited = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )
            assert exited.status_code == 200
            assert runtime.status == "disabled"

            # The degraded exit's own rotation never ran -- the runtime DB
            # still durably holds the "exited" epoch id and its pin, exactly
            # the resurrection precondition gap G-7(b) describes.
            runtime_db_path = paths.claude_account_pool_runtime_db()
            inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
            try:
                assert inspect_store.balanced_epoch_id == old_epoch_id
                assert inspect_store.get_pin(old_pin_digest) is not None
            finally:
                inspect_store.close()

            reentered = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert reentered.status_code == 200
            new_runtime = client.app.state.claude_balanced_runtime
            assert new_runtime.status == "active"
            # A durably fresh epoch -- never the exited one the degraded
            # rotation left behind.
            assert new_runtime.epoch_id != old_epoch_id
            assert new_runtime.router is not None
            assert new_runtime.router.pin_count() == 0
            assert new_runtime.router.get_pin(old_pin_digest) is None
            assert new_runtime._store is not None
            assert new_runtime._store.get_pin(old_pin_digest) is None


def test_balanced_request_during_controlled_exit_awaits_and_serves_under_target_mode(
    tmp_path: Path,
) -> None:
    """Step 6/8's spec-gate ruling, exercised directly against
    `ClaudeBalancedRuntime` (the exact primitives
    `relay._passthrough_with_claude_balanced` dispatches through): a request
    arriving while `exit_mode` is draining awaits the transition, then observes
    the ALREADY-published target mode -- never a 503 merely because the
    controlled transition was in flight.
    """
    from claudex_gateway.claude_accounts import AccountRecord
    from claudex_gateway.claude_balanced_router import ClaudeBalancedRuntime

    account_id = "22222222-3333-4444-5555-666666666666"
    accounts_root = tmp_path / "accounts"
    (accounts_root / account_id).mkdir(parents=True)
    (accounts_root / account_id / "oauth-account.json").write_text(
        json.dumps({"accountUuid": _BALANCED_ACCOUNT_UUID}), encoding="utf-8"
    )
    account = AccountRecord(
        id=account_id,
        email="balanced@example.com",
        organization_uuid=None,
        organization_name=None,
        created_at=0,
        updated_at=0,
        last_authenticated_at=0,
        state="ready",
        account_incarnation_id="incarnation-1",
        upstream_account_uuid=_BALANCED_ACCOUNT_UUID,
    )
    published = {"mode": "disabled"}

    async def scenario() -> str:
        runtime = ClaudeBalancedRuntime()
        await runtime.prepare_and_publish(
            accounts=[account],
            accounts_root=accounts_root,
            runtime_db_path=tmp_path / "runtime.sqlite3",
            persist=lambda: published.__setitem__("mode", "balanced"),
            entry="admin_enable",
        )
        assert runtime.status == "active"

        # Simulate one in-flight balanced dispatch holding a slot open.
        assert runtime.begin_request()

        exit_task = asyncio.create_task(
            runtime.exit_mode(
                "fallback", publish=lambda: published.__setitem__("mode", "fallback")
            )
        )
        await asyncio.sleep(0)
        assert runtime.status == "draining"

        async def waiting_dispatch() -> str:
            # Mirrors relay._passthrough_with_claude_balanced's transition-await
            # branch: wait, then re-read the published mode.
            await runtime.wait_for_transition()
            return published["mode"]

        waiter_task = asyncio.create_task(waiting_dispatch())
        await asyncio.sleep(0)
        assert not waiter_task.done()  # still awaiting -- draining hasn't resolved

        runtime.end_request()  # the in-flight request finishes -> drain completes
        target_mode_seen = await asyncio.wait_for(waiter_task, timeout=5.0)
        await asyncio.wait_for(exit_task, timeout=5.0)
        assert runtime.status == "disabled"
        return target_mode_seen

    assert asyncio.run(scenario()) == "fallback"


def test_balanced_graceful_shutdown_preserves_epoch_and_pin_for_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Step 9: two lifespan-backed app instances over the same pool directory.
    Closing the first (settings still "balanced") preserves the epoch id, seed,
    and durable pin for the second's automatic startup restoration -- contrast
    `TestBalancedRoutingExit`, where an INTENTIONAL exit invalidates that very
    same state instead.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
    _register_balanced_ready_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    settings_file = tmp_path / "settings.json"
    first_config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=first_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        enable = client.put(
            "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
        )
        assert enable.status_code == 200
        first_runtime = client.app.state.claude_balanced_runtime
        epoch_id_1 = first_runtime.epoch_id
        seed_1 = first_runtime.epoch_seed

        pinned = client.post("/v1/messages", json=_account_body())
        assert pinned.status_code == 200
        assert first_runtime.router is not None
        assert first_runtime.router.pin_count() == 1

    # The first instance's lifespan has shut down (shutdown_preserving_epoch,
    # then the T-9 lease released) -- a second, independent app instance loads
    # the SAME settings file and pool directory (HOME unchanged) and restores
    # them automatically at startup.
    second_config = GatewayConfig.load(settings_file)
    assert second_config.claude_account_routing_mode == "balanced"
    with _create_test_client(
        monkeypatch, tmp_path, config=second_config, base_url="http://127.0.0.1:8787"
    ) as client:
        second_runtime = client.app.state.claude_balanced_runtime
        assert second_runtime.status == "active"
        assert second_runtime.epoch_id == epoch_id_1
        assert second_runtime.epoch_seed == seed_1
        assert second_runtime.router is not None
        assert second_runtime.router.pin_count() == 1
        payload = client.get("/admin/providers/claude/pool/routing").json()

    assert payload == {"mode": "balanced", "env_locked": False}


# ---------------------------------------------------------------------------
# Balanced serve-path chain runner: commit-at-headers protocol and §6.5
# exhaustion responses (T-11). The Step 3 fallback-regression guard is every
# `test_pool_*`/`test_account_passthrough_*` test above, left byte-for-byte
# unmodified and still green in this same run.
# ---------------------------------------------------------------------------


def _register_balanced_accounts(count: int) -> list[tuple[str, str]]:
    """Register `count` balanced-ready accounts, each with a distinct valid T-3
    fingerprint and a distinct access token, in registration order.
    """
    accounts = []
    for index in range(count):
        access_token = f"balanced-access-{index}"
        account_id = _register_serving_account(
            email=f"balanced{index}@example.com",
            account_uuid=str(uuid.uuid4()),
            access_token=access_token,
            refresh_token=f"balanced-refresh-{index}",
        )
        accounts.append((account_id, access_token))
    return accounts


_USAGE_PROBE_PATH = "/api/oauth/usage"  # usage._CLAUDE_USAGE_URL's path component


def _usage_probe_intercepting_handler(anthropic_handler: Any) -> Any:
    """Wrap `anthropic_handler` so the T-18 background usage poll driver's own
    automatic probe (`GET .../api/oauth/usage`) is answered directly with a
    no-window "ok" envelope, without ever reaching `anthropic_handler` --
    once balanced routing is active, that driver immediately runs its own
    warm-up poll in the background, and most tests here only care about
    `/v1/messages` traffic (call counts, retry/migration sequencing, ...).
    """

    def transport_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == _USAGE_PROBE_PATH:
            return httpx.Response(200, json={})
        return anthropic_handler(request)

    return transport_handler


def _enable_balanced(
    client: TestClient, anthropic_handler: Any, *, intercept_usage_probe: bool = True
) -> Any:
    """Swap in a mock Anthropic transport and enable balanced routing.

    `intercept_usage_probe` (default `True`, T-18) wraps `anthropic_handler`
    with `_usage_probe_intercepting_handler`; pass `False` for the few tests
    that deliberately want to observe or control the driver's own automatic
    probe's HTTP behavior.

    Returns the resulting `ClaudeBalancedRuntime`.
    """
    handler = (
        _usage_probe_intercepting_handler(anthropic_handler)
        if intercept_usage_probe
        else anthropic_handler
    )
    client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
    assert response.status_code == 200, response.text
    return client.app.state.claude_balanced_runtime


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _balanced_body(session_id: str, *, model: str = "claude-fable-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {
        "user_id": json.dumps(
            {
                "device_id": "d" * 64,
                "account_uuid": "client-account-uuid",
                "session_id": session_id,
            },
            separators=(",", ":"),
        )
    }
    return body


_UNPINNABLE_BODY: dict[str, Any] = {
    "model": "claude-fable-5",
    "max_tokens": 16,
    "messages": [],
}


def test_balanced_new_session_pins_and_serves_with_a_durable_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, access_token = _register_balanced_accounts(1)[0]
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    settings_file = tmp_path / "settings.json"
    config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert response.json() == {"id": "msg_1"}
        assert len(calls) == 1
        assert calls[0].headers["authorization"] == f"Bearer {access_token}"
        assert runtime.router.pin_count() == 1

        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        assert session_key is not None
        pin = runtime.router.get_pin(session_key.digest)
        assert pin is not None
        assert pin.account_id == account_id

        store_row = runtime._store.get_pin(session_key.digest)
        assert store_row is not None
        assert store_row.account_id == account_id
        assert store_row.generation == 0


def test_balanced_repeat_request_reuses_the_existing_pin_without_a_new_placement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, access_token = _register_balanced_accounts(1)[0]
    calls: list[httpx.Request] = []
    created_flags: list[bool] = []
    original_place_session = ClaudeBalancedRouter.place_session

    def spy_place_session(self: ClaudeBalancedRouter, **kwargs: Any) -> Any:
        result = original_place_session(self, **kwargs)
        created_flags.append(result.created)
        return result

    monkeypatch.setattr(ClaudeBalancedRouter, "place_session", spy_place_session)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        first = client.post("/v1/messages", json=body)
        second = client.post("/v1/messages", json=body)

        assert first.status_code == 200
        assert second.status_code == 200
        assert runtime.router.pin_count() == 1
        assert created_flags == [True, False]
        assert [call.headers["authorization"] for call in calls] == [
            f"Bearer {access_token}",
            f"Bearer {access_token}",
        ]


def test_balanced_quota_429_migrates_commits_and_cools_down_the_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    accounts = _register_balanced_accounts(2)
    tokens_by_call: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert len(tokens_by_call) == 2
        source_token, target_token = tokens_by_call
        assert source_token != target_token
        source_id = next(aid for aid, token in accounts if f"Bearer {token}" == source_token)
        target_id = next(aid for aid, token in accounts if f"Bearer {token}" == target_token)

        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        pin = runtime.router.get_pin(session_key.digest)
        assert pin is not None
        assert pin.account_id == target_id
        assert pin.generation == 1
        assert runtime.router.migration_outcome_counts.get("committed") == 1
        assert client.app.state.claude_account_cooldowns.is_cooling(source_id)
        assert not client.app.state.claude_account_cooldowns.is_cooling(target_id)


def test_balanced_migration_hop_uses_the_scoring_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    tokens_by_call: list[str] = []
    recorded_digests: list[bytes] = []
    original_pick_account = relay._balanced_pick_account

    def recording_pick_account(*args: Any, **kwargs: Any) -> str:
        recorded_digests.append(kwargs["session_key_digest"])
        return original_pick_account(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        monkeypatch.setattr(relay, "_balanced_pick_account", recording_pick_account)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert len(tokens_by_call) == 2
        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        assert session_key is not None
        assert recorded_digests
        assert all(digest == session_key.scoring_digest for digest in recorded_digests)
        assert all(digest != session_key.digest for digest in recorded_digests)


def test_balanced_migration_target_preheader_failure_tries_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(3)
    tokens_by_call: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) < 3:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert len(tokens_by_call) == 3
        assert len(set(tokens_by_call)) == 3  # three distinct accounts, never repeated
        assert runtime.router.migration_outcome_counts.get("retryable_preheader_failure") == 1
        assert runtime.router.migration_outcome_counts.get("committed") == 1


def test_balanced_post_2xx_midstream_failure_relays_sse_error_and_retains_the_committed_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    accounts = _register_balanced_accounts(2)
    tokens_by_call: list[str] = []
    first_event = b'event: message_start\ndata: {"type": "message_start"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) == 1:
            return _quota_429()
        return httpx.Response(
            200,
            stream=_AbortingByteStream(first_event),
            headers={"content-type": "text/event-stream"},
        )

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert response.content.startswith(first_event)
        assert b"event: error" in response.content

        target_token = tokens_by_call[1]
        target_id = next(aid for aid, token in accounts if f"Bearer {token}" == target_token)
        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        pin = runtime.router.get_pin(session_key.digest)
        assert pin is not None
        assert pin.account_id == target_id
        assert pin.generation == 1


def test_balanced_all_cooling_chain_exhaustion_relays_the_upstream_429_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429("balanced-exhausted")

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 429
        assert response.json()["error"]["message"] == "balanced-exhausted"
        assert response.headers["x-should-retry"] == "true"
        assert client.app.state.claude_account_cooldowns.is_cooling(account_id)


def test_balanced_all_cooling_synthesizes_429_with_retry_after_clamped_over_candidate_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    ready_id, _ready_token = _register_balanced_accounts(1)[0]
    # A second, not-ready ("disabled") account with a much EARLIER cooldown
    # deadline: it must never enter the candidate set `C`, so it can never
    # shorten Retry-After below the ready account's own (longer) deadline.
    denied_id = _register_serving_account(
        email="denied@example.com", account_uuid=str(uuid.uuid4())
    )
    claude_accounts.mark_account_needs_reauth(denied_id)

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)
        tracker = client.app.state.claude_account_cooldowns
        tracker.mark(denied_id, 5.0)

        exhausted = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert exhausted.status_code == 429
        assert tracker.is_cooling(ready_id)

        second = client.post("/v1/messages", json=_balanced_body(_new_session_id()))

        assert second.status_code == 429
        retry_after = int(second.headers["retry-after"])
        # The ready account's own (freshly installed, ~60s default) cooldown drives
        # Retry-After; the excluded, not-ready account's much shorter 5s deadline
        # never shortens it.
        assert retry_after > 10
        assert retry_after <= 61


def test_balanced_returns_the_adjudicated_503_byte_exact_when_the_candidate_set_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen with an empty candidate set")

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)
        claude_accounts.mark_account_needs_reauth(account_id)

        response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))

        assert response.status_code == 503
        assert response.json() == {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "no registered account is eligible for the requested model",
            },
        }
        assert "retry-after" not in response.headers


def test_balanced_non_streaming_request_commits_at_headers_before_body_relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    order: list[str] = []

    original_commit_at_headers = ClaudeBalancedRouter.commit_at_headers

    def spy_commit_at_headers(self: ClaudeBalancedRouter, *args: Any, **kwargs: Any) -> Any:
        result = original_commit_at_headers(self, *args, **kwargs)
        order.append(f"commit:{result[0]}")
        return result

    monkeypatch.setattr(ClaudeBalancedRouter, "commit_at_headers", spy_commit_at_headers)

    class _LoggingByteStream(httpx.AsyncByteStream):
        def __init__(self, body: bytes) -> None:
            self._body = body

        async def __aiter__(self) -> AsyncIterator[bytes]:
            order.append("body_pulled")
            yield self._body

    tokens_by_call: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) == 1:
            return _quota_429()
        return httpx.Response(
            200,
            stream=_LoggingByteStream(b'{"id": "msg_1"}'),
            headers={"content-type": "application/json"},
        )

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())
        body["stream"] = False

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 200
        assert response.json() == {"id": "msg_1"}
        assert order == ["commit:committed", "body_pulled"]


def test_balanced_concurrent_same_session_request_awaits_the_pending_durability_barrier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    upstream_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.headers["authorization"])
        return httpx.Response(200, json={"id": "msg_1"})

    config = GatewayConfig(claude_account_routing_mode="balanced")
    app = server.create_app(config)
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    app.state.codex_client = StubCodexClient()
    app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app.state.claude_account_auth_managers = {}

    async def _unused_fetch(_account_id: str) -> tuple[dict[str, Any], float | None]:
        raise AssertionError("usage cache fetch must not be reached in this test")

    app.state.claude_account_usage_cache = ClaudeAccountUsageCache(_unused_fetch)
    app.state.claude_account_cooldowns = AccountCooldownTracker()
    app.state.claude_balanced_runtime = ClaudeBalancedRuntime()

    gate = asyncio.Event()
    write_pending: list[bool] = []
    original_submit_new_pin_durability = ClaudeBalancedRouter.submit_new_pin_durability

    async def delayed_submit_new_pin_durability(self: ClaudeBalancedRouter, digest: bytes) -> None:
        write_pending.append(True)
        await gate.wait()
        await original_submit_new_pin_durability(self, digest)

    monkeypatch.setattr(
        ClaudeBalancedRouter, "submit_new_pin_durability", delayed_submit_new_pin_durability
    )

    async def scenario() -> tuple[list[int], list[str]]:
        runtime = app.state.claude_balanced_runtime
        await runtime.prepare_and_publish(
            accounts=claude_accounts.list_accounts(),
            accounts_root=paths.accounts_dir("claude"),
            runtime_db_path=paths.claude_account_pool_runtime_db(),
            persist=lambda: None,
            entry="admin_enable",
        )
        body = _balanced_body(_new_session_id())

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as async_client:
            first_task = asyncio.create_task(async_client.post("/v1/messages", json=body))
            for _ in range(200):
                if write_pending:
                    break
                await asyncio.sleep(0)
            assert write_pending  # the creator is blocked on its own durability barrier

            second_task = asyncio.create_task(async_client.post("/v1/messages", json=body))
            for _ in range(50):
                await asyncio.sleep(0)
            # Neither request has reached upstream yet: both await the barrier first.
            assert upstream_calls == []

            gate.set()
            first_response = await asyncio.wait_for(first_task, timeout=5.0)
            second_response = await asyncio.wait_for(second_task, timeout=5.0)
            return [first_response.status_code, second_response.status_code], list(upstream_calls)

    statuses, calls = asyncio.run(scenario())

    assert statuses == [200, 200]
    assert len(calls) == 2


def test_balanced_unpinnable_retry_chain_reuses_one_stateless_digest_and_creates_no_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    digest_calls: list[bytes] = []
    original_derive_stateless_routing_digest = relay.derive_stateless_routing_digest

    def spy_derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
        digest = original_derive_stateless_routing_digest(seed, nonce)
        digest_calls.append(digest)
        return digest

    monkeypatch.setattr(
        relay, "derive_stateless_routing_digest", spy_derive_stateless_routing_digest
    )

    tokens_by_call: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_by_call.append(request.headers["authorization"])
        if len(tokens_by_call) == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post("/v1/messages", json=_UNPINNABLE_BODY)

        assert response.status_code == 200
        assert len(tokens_by_call) == 2
        assert len(set(tokens_by_call)) == 2  # both accounts tried, never repeated
        assert len(digest_calls) == 1  # ONE stateless digest reused for the whole chain
        assert runtime.router.pin_count() == 0


def test_balanced_unpinnable_separate_requests_get_independent_stateless_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)
    digest_calls: list[bytes] = []
    original_derive_stateless_routing_digest = relay.derive_stateless_routing_digest

    def spy_derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
        digest = original_derive_stateless_routing_digest(seed, nonce)
        digest_calls.append(digest)
        return digest

    monkeypatch.setattr(
        relay, "derive_stateless_routing_digest", spy_derive_stateless_routing_digest
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)

        first = client.post("/v1/messages", json=_UNPINNABLE_BODY)
        second = client.post("/v1/messages", json=_UNPINNABLE_BODY)

        assert first.status_code == 200
        assert second.status_code == 200
        assert len(digest_calls) == 2
        assert digest_calls[0] != digest_calls[1]
        assert runtime.router.pin_count() == 0


def test_balanced_count_tokens_follows_the_pin_without_refresh_or_router_state_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, access_token = _register_balanced_accounts(1)[0]
    count_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/count_tokens"):
            count_calls.append(request)
            return httpx.Response(200, json={"input_tokens": 7})
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        pinned = client.post("/v1/messages", json=body)
        assert pinned.status_code == 200

        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        pin_before = runtime.router.get_pin(session_key.digest)
        assert pin_before is not None
        last_seen_before = pin_before.last_seen_monotonic
        pin_count_before = runtime.router.pin_count()
        migration_counts_before = dict(runtime.router.migration_outcome_counts)
        in_flight_before = runtime.router.in_flight_count(account_id)

        counted = client.post("/v1/messages/count_tokens", json=body)

        assert counted.status_code == 200
        assert counted.json() == {"input_tokens": 7}
        assert len(count_calls) == 1
        assert count_calls[0].headers["authorization"] == f"Bearer {access_token}"

        pin_after = runtime.router.get_pin(session_key.digest)
        assert pin_after is not None
        assert pin_after.last_seen_monotonic == last_seen_before
        assert runtime.router.pin_count() == pin_count_before
        assert dict(runtime.router.migration_outcome_counts) == migration_counts_before
        assert runtime.router.in_flight_count(account_id) == in_flight_before


def test_balanced_count_tokens_resolves_the_same_family_pin_as_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _account_id, access_token = _register_balanced_accounts(1)[0]
    count_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/count_tokens"):
            count_calls.append(request)
            return httpx.Response(200, json={"input_tokens": 7})
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())
        pinned = client.post("/v1/messages", json=body)
        assert pinned.status_code == 200

        def fail_fallback_pick(*args: Any, **kwargs: Any) -> str:
            raise AssertionError("fallback pick must not run")

        monkeypatch.setattr(relay, "_balanced_pick_account", fail_fallback_pick)
        assert relay._balanced_pick_account is fail_fallback_pick
        counted = client.post("/v1/messages/count_tokens", json=body)

        assert counted.status_code == 200
        assert len(count_calls) == 1
        assert count_calls[0].headers["authorization"] == f"Bearer {access_token}"


def test_balanced_count_tokens_fallback_uses_the_scoring_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)
    recorded_digests: list[bytes] = []
    original_pick_account = relay._balanced_pick_account

    def recording_pick_account(*args: Any, **kwargs: Any) -> str:
        recorded_digests.append(kwargs["session_key_digest"])
        return original_pick_account(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"input_tokens": 7})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        monkeypatch.setattr(relay, "_balanced_pick_account", recording_pick_account)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages/count_tokens", json=body)

        assert response.status_code == 200
        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        assert session_key is not None
        assert recorded_digests == [session_key.scoring_digest]
        assert recorded_digests[0] != session_key.digest


def test_balanced_count_tokens_never_creates_a_pin_or_a_cooldown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages/count_tokens", json=body)

        # A 429 relays verbatim -- no failover chain, no cooldown bookkeeping --
        # matching the council-pinned "never creates ... cooldowns" rule.
        assert response.status_code == 429
        assert runtime.router.pin_count() == 0
        assert not client.app.state.claude_account_cooldowns.is_cooling(account_id)


# ---------------------------------------------------------------------------
# Durable cooldowns, Fable family gate, capability evidence, removal cleanup
# (T-12, design v2 §6.4/§5.5/§5.7, adjudication G)
# ---------------------------------------------------------------------------


def _ingest_gate_satisfying_fable_observations(router: ClaudeBalancedRouter, account_id: str) -> None:
    """Fresh, REAL readings satisfying every non-status/non-family §6.4 gate
    condition: `fable_weekly` >=99%, `five_hour`/`seven_day` <=70%, all <=15 min
    old, with a valid future Fable reset."""
    router.ingest_observation(
        account_id,
        "fable_weekly",
        used_percent=99.0,
        source="usage_api",
        age_seconds=0.0,
        reset_in_seconds=3600.0,
        reset_identity="fable-reset-1",
    )
    router.ingest_observation(account_id, "five_hour", used_percent=40.0, source="usage_api", age_seconds=0.0)
    router.ingest_observation(account_id, "seven_day", used_percent=40.0, source="usage_api", age_seconds=0.0)


def test_balanced_fable_429_with_fresh_gate_observations_installs_family_cooldown_and_still_serves_sonnet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Fable 429 with three fresh, gate-satisfying observations installs a
    FAMILY-scoped cooldown (not account-wide) -- so a later Sonnet (default
    family) request keeps being served from the very same account.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, access_token = _register_balanced_accounts(1)[0]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["authorization"])
        if len(calls) == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        router = runtime.router
        assert router is not None
        _ingest_gate_satisfying_fable_observations(router, account_id)

        fable_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert fable_response.status_code == 429

        now = time.monotonic()
        assert router.family_cooldown_deadline(account_id, "fable", now=now) is not None
        assert router.account_cooldown_deadline(account_id, now=now) is None

        sonnet_body = _balanced_body(_new_session_id(), model="claude-sonnet-5")
        sonnet_response = client.post("/v1/messages", json=sonnet_body)

        assert sonnet_response.status_code == 200
        assert len(calls) == 2
        assert calls[1] == f"Bearer {access_token}"


def test_balanced_fable_429_with_one_stale_observation_fails_the_family_gate_and_installs_account_wide_cooldown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ambiguous observations (one stale, >15 min old) fail the family gate's
    freshness condition -- the cooldown falls back to account-wide."""
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        router = runtime.router
        assert router is not None
        router.ingest_observation(
            account_id,
            "fable_weekly",
            used_percent=99.0,
            source="usage_api",
            age_seconds=0.0,
            reset_in_seconds=3600.0,
            reset_identity="fable-reset-1",
        )
        # Stale: >15 min old, so the family gate's freshness condition fails.
        router.ingest_observation(account_id, "five_hour", used_percent=40.0, source="usage_api", age_seconds=20 * 60)
        router.ingest_observation(account_id, "seven_day", used_percent=40.0, source="usage_api", age_seconds=0.0)

        response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert response.status_code == 429

        now = time.monotonic()
        assert router.family_cooldown_deadline(account_id, "fable", now=now) is None
        assert router.account_cooldown_deadline(account_id, now=now) is not None


def test_balanced_restart_restores_the_family_cooldown_without_a_repeat_429_burst(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A daemon restart restores durable cooldowns (§5.4/§5.5): the second
    process instance skips the still-cooling account for its own Fable-family
    request WITHOUT ever calling upstream again -- no repeat-429 burst.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["authorization"])
        return _quota_429()

    settings_file = tmp_path / "settings.json"
    first_config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=first_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_usage_probe_intercepting_handler(handler))
        )
        enable = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
        assert enable.status_code == 200
        first_runtime = client.app.state.claude_balanced_runtime
        router = first_runtime.router
        assert router is not None
        _ingest_gate_satisfying_fable_observations(router, account_id)

        first_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert first_response.status_code == 429
        assert len(calls) == 1
        assert router.family_cooldown_deadline(account_id, "fable", now=time.monotonic()) is not None

    # A second, independent process instance over the same pool directory
    # (mirrors `test_balanced_graceful_shutdown_preserves_epoch_and_pin_for_restart`).
    second_config = GatewayConfig.load(settings_file)
    assert second_config.claude_account_routing_mode == "balanced"
    with _create_test_client(
        monkeypatch, tmp_path, config=second_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_usage_probe_intercepting_handler(handler))
        )
        second_runtime = client.app.state.claude_balanced_runtime
        assert second_runtime.status == "active"
        assert second_runtime.router is not None
        assert second_runtime.router.family_cooldown_deadline(
            account_id, "fable", now=time.monotonic()
        ) is not None

        second_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))

        assert second_response.status_code == 429
        assert len(calls) == 1  # no repeat-429 burst: the restored cooldown skipped upstream entirely


def test_balanced_account_removal_clears_its_durable_rows_for_that_incarnation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removing an account while balanced is active runs the router's own
    removal matrix and deletes every durable row of that incarnation (§5.7) --
    verified directly against the persisted store after the process shuts down.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429()

    settings_file = tmp_path / "settings.json"
    config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        router = runtime.router
        assert router is not None

        incarnation = next(
            record.account_incarnation_id
            for record in claude_accounts.list_accounts()
            if record.id == account_id
        )

        # One 429 leaves both a pin (still pointing at the cooling account) and
        # a durable cooldown row -- both durably written and awaited already.
        cooling_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert cooling_response.status_code == 429
        assert router.pin_count() == 1
        assert router.account_cooldown_deadline(account_id, now=time.monotonic()) is not None

        delete_response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert delete_response.status_code == 204
        assert router.pin_count() == 0
        assert router.account_cooldown_deadline(account_id, now=time.monotonic()) is None

    runtime_db_path = paths.claude_account_pool_runtime_db()
    inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
    try:
        restore_result = inspect_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert not any(row.account_incarnation_id == incarnation for row in restore_result.pins.values())
        assert not any(row.account_incarnation_id == incarnation for row in restore_result.cooldowns.values())
        assert not any(
            row.account_incarnation_id == incarnation
            for row in restore_result.usage_observations.values()
        )
        assert not any(
            row.account_incarnation_id == incarnation
            for row in restore_result.capability_evidence.values()
        )
    finally:
        inspect_store.close()



def test_balanced_successful_2xx_records_eligible_capability_evidence_for_the_request_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful balanced 2xx records ELIGIBLE capability evidence for the
    request's own capability key (adjudication G) -- and never implies
    eligibility for a different key.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        router = runtime.router
        assert router is not None
        record = next(r for r in claude_accounts.list_accounts() if r.id == account_id)
        from claudex_gateway.claude_account_profile import load_account_profile_fingerprint

        fingerprint = load_account_profile_fingerprint(paths.accounts_dir("claude") / account_id)
        assert fingerprint is not None

        response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert response.status_code == 200

        assert router.is_capability_eligible(
            account_id,
            "fable",
            account_incarnation_id=record.account_incarnation_id,
            account_profile_fingerprint=fingerprint,
        )
        assert not router.is_capability_eligible(
            account_id,
            "opus",
            account_incarnation_id=record.account_incarnation_id,
            account_profile_fingerprint=fingerprint,
        )


# ---------------------------------------------------------------------------
# Balanced usage poll coordinator, refresh isolation, mode-safe usage API
# (T-13): budget/fairness/cooldown/anti-starvation fake-clock unit tests
# (Step 7), plus HTTP-level balanced/fallback/disabled endpoint tests (Steps
# 4-6, 8-9).
# ---------------------------------------------------------------------------


class _FakeMonotonicClock:
    """A controllable stand-in for `time.monotonic` (and `time.time`) --
    no real sleeping. Reused as both `clock` and `wall_clock` below: the
    coordinator's tests only need consistent, deterministic numbers, not a
    realistic monotonic/wall split.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeUsagePollFetch:
    """Records fetch order and replays per-account queued responses -- the
    coordinator's own fake stand-in for the real per-account usage probe.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses: dict[str, list[tuple[dict[str, Any], float | None]]] = {}

    def queue(self, account_id: str, result: dict[str, Any], retry_after: float | None = None) -> None:
        self.responses.setdefault(account_id, []).append((result, retry_after))

    async def __call__(self, account_id: str) -> tuple[dict[str, Any], float | None]:
        self.calls.append(account_id)
        return self.responses[account_id].pop(0)


def _fake_usage_ok(*, session_percent: float = 10.0, weekly_percent: float = 20.0) -> dict[str, Any]:
    return {
        "provider": "claude",
        "status": "ok",
        "error": None,
        "session": {"used_percent": session_percent, "resets_at": None},
        "weekly": {"used_percent": weekly_percent, "resets_at": None},
        "fable_weekly": None,
    }


def _fake_usage_err(message: str) -> dict[str, Any]:
    return {"provider": "claude", "status": "error", "error": message, "session": None}


class _RecordingUsageObservationStore:
    """Duck-typed stand-in for `ClaudePoolRuntimeStateStore.upsert_usage_observation`
    -- records what the coordinator submits, with none of the real store's
    background-thread persistence timing.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upsert_usage_observation(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _make_usage_poll_coordinator(
    *, clock: _FakeMonotonicClock, fetch: _FakeUsagePollFetch, store: Any = None
) -> tuple[ClaudeUsagePollCoordinator, ClaudeAccountUsageCache, ClaudeBalancedRouter]:
    cache = ClaudeAccountUsageCache(fetch, clock=clock)
    router = ClaudeBalancedRouter(balanced_epoch_id="epoch-1", clock=clock, wall_clock=clock)
    coordinator = ClaudeUsagePollCoordinator(
        cache=cache, router=router, store=store, clock=clock, wall_clock=clock
    )
    return coordinator, cache, router


class TestUsagePollCoordinatorScheduling:
    """Fake-clock coordinator unit tests (T-13 Step 7): budget enforcement,
    missing-window priority, fairness after failure, global cooldown, manual
    coalescing/rate limiting, and full-sweep progress under manual refreshes.
    """

    def test_coordinator_enforces_the_actual_call_budget(self) -> None:
        # Two DIFFERENT accounts, both never observed -- this isolates the
        # coordinator's own 30s call budget from the cache's own (120s,
        # unrelated) per-account TTL: "b" stays due the whole time, so its
        # own due-ness can never explain a budget_wait here.
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok())
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        first = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        second = asyncio.run(coordinator.run_due_poll(["a", "b"]))

        assert first.outcome == "fetched"
        assert first.account_id == "a"
        assert second.outcome == "budget_wait"
        assert fetch.calls == ["a"]

        clock.advance(30.0)
        fetch.queue("b", _fake_usage_ok())
        third = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert third.outcome == "fetched"
        assert third.account_id == "b"
        assert fetch.calls == ["a", "b"]

    def test_missing_window_accounts_are_serviced_before_already_observed_ones(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("warm", _fake_usage_ok())
        coordinator, cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        asyncio.run(cache.get(["warm"]))  # seeds "warm" with a fresh observation
        fetch.queue("cold", _fake_usage_ok())

        tick = asyncio.run(coordinator.run_due_poll(["warm", "cold"]))

        assert tick.outcome == "fetched"
        assert tick.account_id == "cold"

    def test_a_failing_account_does_not_starve_the_rest_of_the_pool(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        for _ in range(4):
            fetch.queue("a", _fake_usage_err("usage API returned 500: boom"))
        fetch.queue("b", _fake_usage_ok())
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        serviced: list[str | None] = []
        for _ in range(4):
            serviced.append(asyncio.run(coordinator.run_due_poll(["a", "b"])).account_id)
            clock.advance(30.0)

        assert "b" in serviced
        assert fetch.calls.count("b") == 1
        assert fetch.calls.count("a") >= 1

    def test_global_cooldown_is_reported_and_blocks_the_whole_tick(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_err("usage API rate-limited (429); try again shortly"), 45.0)
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        first = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert first.outcome == "fetched"  # a's own call opened the cooldown

        clock.advance(30.0)
        second = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert second.outcome == "cooldown"
        assert fetch.calls == ["a"]

    def test_manual_refresh_coalesces_per_account_and_is_globally_rate_limited(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        assert coordinator.request_manual_refresh("a") is True
        assert coordinator.request_manual_refresh("a") is True  # coalesced, not a new enqueue
        assert coordinator.diagnostics().manual_enqueued_count == 1

        assert coordinator.request_manual_refresh("b") is False  # globally rate-limited
        assert coordinator.diagnostics().manual_rate_limited_count == 1

        clock.advance(300.0)
        assert coordinator.request_manual_refresh("b") is True
        assert coordinator.diagnostics().manual_enqueued_count == 2

    def test_manual_refresh_never_fetches_inline(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        assert coordinator.request_manual_refresh("a") is True

        assert fetch.calls == []
        assert coordinator.is_manual_refresh_pending("a") is True

    def test_manual_refresh_only_consumes_a_slot_after_an_automatic_account_is_serviced(
        self,
    ) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("auto", _fake_usage_ok())
        coordinator, cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        asyncio.run(cache.get(["manual_target"]))  # already fresh -- never automatically due
        coordinator.request_manual_refresh("manual_target")

        # Nothing automatic has been serviced by THIS coordinator yet -- the
        # very first tick must claim the one genuinely due account ("auto"),
        # not the pending manual one.
        first = asyncio.run(coordinator.run_due_poll(["auto", "manual_target"]))
        assert first.outcome == "fetched"
        assert first.account_id == "auto"
        assert first.manual is False

        # Now that an automatic account has been serviced, and nothing else
        # is due (both accounts are within their own TTL), the next tick
        # claims the pending manual one -- forcing a fresh fetch despite it.
        fetch.queue("manual_target", _fake_usage_ok())
        clock.advance(30.0)
        second = asyncio.run(coordinator.run_due_poll(["auto", "manual_target"]))
        assert second.outcome == "fetched"
        assert second.account_id == "manual_target"
        assert second.manual is True
        assert coordinator.is_manual_refresh_pending("manual_target") is False

    def test_full_sweep_still_makes_progress_under_repeated_manual_refresh_requests(
        self,
    ) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        accounts = ["a", "b", "c"]
        for account_id in accounts:
            fetch.queue(account_id, _fake_usage_ok())
        coordinator.request_manual_refresh("a")

        # Exactly one tick per account -- within the cache's own 120s TTL, so
        # nothing here is re-fetched merely because it aged out.
        for _ in range(len(accounts)):
            asyncio.run(coordinator.run_due_poll(accounts))
            clock.advance(30.0)
            coordinator.request_manual_refresh("b")  # rate-limited every time; harmless either way

        # The full automatic sweep completed -- every ready account actually
        # got polled, not just whichever one a manual request kept naming.
        assert set(fetch.calls) == set(accounts)
        assert coordinator.diagnostics().fetched_count == len(accounts)

    def test_a_successful_fetch_feeds_the_router_observation_view(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok(session_percent=42.0, weekly_percent=17.0))
        coordinator, _cache, router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        tick = asyncio.run(coordinator.run_due_poll(["a"]))

        assert tick.outcome == "fetched"
        assert router.observations.window_pressure("a", "five_hour", now=clock.now) == 42.0
        assert router.observations.window_pressure("a", "seven_day", now=clock.now) == 17.0

    def test_a_successful_fetch_persists_a_durable_usage_observation_row(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok(session_percent=55.0, weekly_percent=12.0))
        store = _RecordingUsageObservationStore()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch, store=store)
        accounts = {
            "a": UsagePollAccount(
                account_id="a", account_incarnation_id="inc-1", account_profile_fingerprint="fp-1"
            )
        }

        tick = asyncio.run(coordinator.run_due_poll(["a"], accounts=accounts))

        assert tick.outcome == "fetched"
        by_window = {call["window"]: call for call in store.calls}
        assert by_window["five_hour"]["used_percent"] == 55.0
        assert by_window["five_hour"]["account_incarnation_id"] == "inc-1"
        assert by_window["five_hour"]["account_profile_fingerprint"] == "fp-1"
        assert by_window["seven_day"]["used_percent"] == 12.0

    def test_durable_persistence_is_skipped_without_a_profile_fingerprint(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok())
        store = _RecordingUsageObservationStore()
        coordinator, _cache, router = _make_usage_poll_coordinator(clock=clock, fetch=fetch, store=store)
        accounts = {
            "a": UsagePollAccount(
                account_id="a", account_incarnation_id="inc-1", account_profile_fingerprint=None
            )
        }

        asyncio.run(coordinator.run_due_poll(["a"], accounts=accounts))

        assert store.calls == []
        # The router's in-memory observation view is still fed regardless.
        assert router.observations.window_pressure("a", "five_hour", now=clock.now) is not None


class TestBalancedUsageCoordinatorEndpoints:
    """HTTP-level coverage (T-13 Steps 4-6, 8-9): active balanced usage reads
    are cache-only and never call upstream, manual refresh returns `queued`
    without fetching inline, and fallback/disabled mode keeps the pre-existing
    fetch path/envelope and never surfaces a stranded `queued` indication.
    """

    def test_pool_usage_active_balanced_mode_never_calls_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        # T-18: the background driver already warmed this account up with one
        # automatic poll the moment balanced routing activated -- this
        # endpoint's OWN read is still cache-only and adds no second call.
        assert payload["accounts"][account_id]["status"] == "ok"
        assert payload["accounts"][account_id]["windows"] == {}
        assert payload["accounts"][account_id]["queued"] is False
        assert calls == [account_id]

    def test_manual_refresh_returns_queued_without_fetching_in_active_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler)
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": account_id, "refresh": "1"},
            )
            assert runtime.usage_poll_coordinator is not None
            pending = runtime.usage_poll_coordinator.is_manual_refresh_pending(account_id)

        assert response.status_code == 200
        payload = response.json()
        assert payload["queued"] is True
        assert payload["accounts"][account_id]["queued"] is True
        assert pending is True
        # T-18: the background driver's own automatic warm-up poll is the
        # only call recorded -- `?refresh` itself never fetches inline, it
        # only enqueues the manual refresh the driver will service later.
        assert calls == [account_id]

    def test_manual_refresh_is_inert_outside_active_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        with client:
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": account_id, "refresh": "1"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert "queued" not in payload
        assert "queued" not in payload["accounts"][account_id]

    def test_pool_usage_disabled_mode_preserves_the_existing_envelope_and_fetch_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"accounts", "fetched_at"}
        assert payload["accounts"][account_id] == {
            "provider": "claude",
            "status": "ok",
            "error": None,
        }
        assert calls == [account_id]  # an equivalent uncached read DOES hit upstream

    def test_pool_usage_fallback_mode_preserves_the_existing_envelope_and_fetch_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(
                settings_file=tmp_path / "settings.json", claude_account_routing_mode="fallback"
            ),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"accounts", "fetched_at"}
        envelope = payload["accounts"][account_id]
        assert "queued" not in envelope
        assert "windows" not in envelope
        assert calls == [account_id]

    def test_pool_status_usage_freshness_is_none_outside_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        _register_serving_account()

        with client:
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] is None
        assert payload["usage_diagnostics"] is None

    def test_pool_status_reports_degraded_usage_freshness_before_any_observation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "degraded"
        assert payload["usage_diagnostics"]["persistence_degraded"] is False
        # T-18: the background driver already made its one automatic warm-up
        # attempt (the mock transport's no-window "ok" stub, same as every
        # other `_enable_balanced` caller) -- still "degraded" since it
        # carried no actual window data, but no longer the pre-driver "never
        # even tried" baseline of 0.
        assert payload["usage_diagnostics"]["coordinator"]["fetched_count"] == 1

    def test_pool_status_reports_fresh_usage_freshness_once_every_window_is_recent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": {"used_percent": 5.0, "resets_at": None},
                    "fable_weekly": None,
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            # Populate the cache through the ordinary (still-disabled-mode)
            # fetch path first -- the SAME cache instance survives the mode
            # switch below, so this seeds a fresh observation without a live
            # poll coordinator loop.
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "fresh"

    def test_pool_status_reports_partial_usage_freshness_when_a_binding_window_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """G-2 regression: `fresh` requires BOTH binding windows (`session`
        AND `weekly`) present -- an account with only `session` recently
        observed, and `weekly` never observed, must not count as fresh even
        though its one observed window is well within the 5-minute bound.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": None,
                    "fable_weekly": None,
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "partial"

    def test_pool_status_reports_partial_usage_freshness_when_fable_weekly_present_but_weekly_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """G-2 regression: `fable_weekly` is a scoped, Fable-only extra
        window and must never substitute for a missing `weekly` binding
        window -- an account with a fresh `session` and a fresh
        `fable_weekly`, but no `weekly`, still does not count as fresh.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": None,
                    "fable_weekly": {"used_percent": 1.0, "resets_at": None},
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "partial"


# ---------------------------------------------------------------------------
# Usage poll driver (T-18, fix for gap G-1): the background task that
# actually calls `ClaudeUsagePollCoordinator.run_due_poll` while balanced
# routing is active -- no production code path ever did before this.
# ---------------------------------------------------------------------------


def _wait_until(client: TestClient, predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Poll `predicate`, giving the app's background event loop another turn
    (a real `GET /health` round trip) between checks, until it holds or
    `timeout` elapses. Avoids a fixed real-time `sleep` in the test itself
    while staying deterministic: a working driver satisfies `predicate`
    almost immediately, and a genuinely broken/cancelled one fails fast
    instead of hanging until the timeout.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met within timeout")
        client.get("/health")


class TestUsagePollDriver:
    """T-18 (fix for gap G-1): `ClaudeBalancedRuntime` now actually drives
    `usage_poll_coordinator.run_due_poll` itself -- automatically once
    balanced routing activates, cancelled on exit/shutdown strictly before
    the store closes, and eventually draining a queued manual refresh --
    instead of leaving the coordinator dormant outside of tests.
    """

    def test_usage_poll_driver_starts_automatically_and_polls_without_any_manual_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Merely activating balanced routing starts the driver: it performs
        one automatic upstream poll entirely on its own -- this test never
        calls `run_due_poll` or requests a manual refresh itself.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != _USAGE_PROBE_PATH:
                raise AssertionError("no /v1/messages traffic in this test")
            calls.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "five_hour": {"utilization": 42.0, "resets_at": None},
                    "seven_day": {"utilization": 17.0, "resets_at": None},
                },
            )

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler, intercept_usage_probe=False)
            _wait_until(client, lambda: len(calls) >= 1)

            response = client.get("/admin/providers/claude/pool/usage")

        assert calls == [_USAGE_PROBE_PATH]
        payload = response.json()
        assert payload["accounts"][account_id]["status"] == "ok"
        assert payload["accounts"][account_id]["windows"]["session"]["state"] == "fresh"
        assert payload["accounts"][account_id]["windows"]["weekly"]["state"] == "fresh"

    def test_usage_poll_driver_is_cancelled_by_an_intentional_exit_and_polls_no_further(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Step 2: disabling balanced routing (`exit_mode`) cancels+awaits the
        driver strictly before the store closes -- no further automatic poll
        happens afterward, even though a short scheduling budget would
        otherwise have allowed several more within this test's own
        wall-clock time.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={})

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01
            _wait_until(client, lambda: len(calls) >= 1)
            assert calls == [_USAGE_PROBE_PATH]

            disabled = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            assert disabled.status_code == 200
            assert runtime.status == "disabled"
            assert runtime._usage_poll_driver_task is None

            # Several more short-budget ticks would fit in this window if the
            # driver were still running -- none did.
            client.get("/health")
            client.get("/health")
            assert calls == [_USAGE_PROBE_PATH]

    def test_usage_poll_driver_is_cancelled_on_lifespan_shutdown_and_polls_no_further(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Step 2: process shutdown (`shutdown_preserving_epoch`) gives the
        same cancel-before-close guarantee as an intentional exit, via the
        lifespan's own teardown path instead of an explicit disable.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={})

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01
            _wait_until(client, lambda: len(calls) >= 1)
            assert calls == [_USAGE_PROBE_PATH]
            assert runtime.status == "active"

        # The `with` block's exit ran `shutdown_preserving_epoch` to
        # completion -- the driver is cancelled+awaited and the store closed
        # before it returned; no additional automatic poll occurred despite
        # the short scheduling budget.
        assert runtime.status == "disabled"
        assert runtime._usage_poll_driver_task is None
        assert calls == [_USAGE_PROBE_PATH]

    def test_usage_poll_driver_drains_a_pending_manual_refresh_without_an_extra_upstream_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A manual refresh (T-13 Step 5) never fetches inline -- it is only
        ever consumed by THIS driver's own next tick, once every automatic
        candidate has yielded no fetch that tick. Total calls stay exactly
        one per account attempt: the pre-seed, the automatic poll, and the
        one deliberate manual-forced poll -- never a redundant extra one.
        """
        _balanced_env(monkeypatch, tmp_path)
        (auto_id, auto_token), (manual_id, manual_token) = _register_balanced_accounts(2)
        auto_bearer = f"Bearer {auto_token}"
        manual_bearer = f"Bearer {manual_token}"
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != _USAGE_PROBE_PATH:
                raise AssertionError("no /v1/messages traffic in this test")
            calls.append(request.headers["authorization"])
            return httpx.Response(
                200,
                json={
                    "five_hour": {"utilization": 10.0, "resets_at": None},
                    "seven_day": {"utilization": 5.0, "resets_at": None},
                },
            )

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            # Pre-seed the manual-target account as already fresh (still in
            # the pre-existing, ordinary fetch-through mode) -- it is never
            # automatically due once balanced routing activates below.
            seeded = client.get(
                "/admin/providers/claude/pool/usage", params={"account": manual_id}
            )
            assert seeded.json()["accounts"][manual_id]["status"] == "ok"
            assert calls == [manual_bearer]

            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01

            # Tick 1: the only genuinely due account is `auto_id`.
            _wait_until(client, lambda: len(calls) >= 2)
            assert calls == [manual_bearer, auto_bearer]

            runtime.usage_poll_coordinator.request_manual_refresh(manual_id)
            # Tick 2+: nothing automatic is due anymore -- the driver drains
            # the pending manual refresh instead.
            _wait_until(client, lambda: len(calls) >= 3)
            assert runtime.usage_poll_coordinator.is_manual_refresh_pending(manual_id) is False

        assert calls == [manual_bearer, auto_bearer, manual_bearer]
