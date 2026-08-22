"""Tests for message relay routes and relay helpers."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.requests import Request
from starlette.responses import Response
from starlette.testclient import TestClient

import claudex.relay.balanced as relay_balanced
import claudex.relay.openai_backend as relay_openai_backend
import claudex.relay.registered as relay_registered
import claudex.server as server
import claudex.translate.context_overflow as context_overflow
from claudex import compaction, paths
from claudex.claude import accounts as claude_accounts
from claudex.claude.account_usage_cache import ClaudeAccountUsageCache
from claudex.claude.account_pool import AccountCooldownTracker
from claudex.claude.auth import CLAUDE_TOKEN_URL
from claudex.balanced.router import ClaudeBalancedRouter
from claudex.balanced.runtime import ClaudeBalancedRuntime
from claudex.balanced.selection import derive_session_key
from claudex.providers.backends import AnthropicBackend, ResponsesBackend
from claudex.providers.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex.config import GatewayConfig, OpenAICompatibleProvider
from claudex.providers.kimi_auth import KimiCredentials
from claudex.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex.providers.openai_compatible_client import OpenAICompatibleUpstreamError
from claudex.providers.grok_auth import GrokCredentials
from claudex.providers.grok_client import GrokUpstreamError
from claudex.relay.common import _upstream_error_to_claude
from claudex.relay.anthropic_backend import (
    _AnthropicStreamRelay,
    _count_tokens_via_anthropic_backend,
    _relay_via_anthropic_backend,
)
from claudex.relay.kimi import _kimi_error_to_claude, _kimi_request_headers
from claudex.relay.openai_backend import (
    _CompactionStreamRelay,
    _OwnedStreamingResponse,
    _aggregate_claude_response,
    _translate_claude_sse,
)
from claudex.translate import translate_claude_request_to_codex
from claudex.translate.thought_signature import (
    CallSignatureCarrier,
    decode_call_signature_carrier,
    encode_call_signature_carrier,
)


_RELAY_SYMBOL_MANIFEST = {
    "common": {
        "_upstream_error_code",
        "_upstream_error_to_claude",
        "_format_sse",
        "_send_to_anthropic",
        "_relay_anthropic_response",
    },
    "registered": {
        "_claude_account_unavailable",
        "_claude_account_request_headers",
        "_rewrite_metadata_account_uuid",
        "_FailedAttempt",
        "_replay_buffered_response",
        "_attempt_with_account",
        "_passthrough_with_claude_account",
        "_passthrough_with_claude_pool",
    },
    "balanced": {
        "_install_balanced_quota_cooldown",
        "_record_balanced_capability_evidence",
        "_balanced_routing_not_active",
        "_passthrough_with_claude_balanced",
        "_balanced_candidates",
        "_balanced_pick_account",
        "_balanced_eligible_candidate_set",
        "_balanced_all_cooling_response",
        "_passthrough_with_balanced_pool",
        "_serve_balanced_stateless_message",
        "_serve_balanced_pinned_message",
        "_serve_balanced_count_tokens",
    },
    "anthropic_compatible": {
        "_anthropic_compatible_request_headers",
        "_safe_anthropic_error",
        "_safe_upstream_message",
        "_configured_credential_error",
        "_anthropic_compatible_error_to_claude",
    },
    "anthropic_backend": {
        "_rewrite_message_start_data",
        "_rewrite_anthropic_sse",
        "_AnthropicStreamRelay",
        "_AnthropicStreamingResponse",
        "_relay_via_anthropic_backend",
        "_count_tokens_via_anthropic_backend",
    },
    "openai_backend": {
        "_resolve_context_window",
        "_validate_mapped_claude_request",
        "_relay_via_responses_backend",
        "_rfc3339_now",
        "_reject_nonfinite_json",
        "_assign_compaction_reroute",
        "_replace_compaction_reroute_if_current",
        "_reroute_compaction",
        "_CompactionStreamRelay",
        "_OwnedStreamingResponse",
        "_translate_claude_sse",
        "_aggregate_claude_response",
    },
    "kimi": {
        "_kimi_request_headers",
        "_kimi_error_to_claude",
    },
    "endpoints": {
        "_get_route_backend",
        "_route_for_request",
        "_passthrough_to_anthropic",
        "_handle_messages",
        "_handle_count_tokens",
    },
}
_RELAY_MANIFEST_SYMBOLS = set().union(*_RELAY_SYMBOL_MANIFEST.values())


def test_relay_symbol_manifest_has_one_canonical_owner() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src/claudex/relay"
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    definitions_by_module = {
        module_name: {
            node.name
            for node in ast.parse(
                (package_root / f"{module_name}.py").read_text(encoding="utf-8")
            ).body
            if isinstance(node, node_types)
        }
        for module_name in _RELAY_SYMBOL_MANIFEST
    }

    assert len(_RELAY_MANIFEST_SYMBOLS) == 55
    assert definitions_by_module == _RELAY_SYMBOL_MANIFEST
    for symbol in _RELAY_MANIFEST_SYMBOLS:
        owners = [
            module_name
            for module_name, definitions in definitions_by_module.items()
            if symbol in definitions
        ]
        assert owners == [
            next(
                module_name
                for module_name, expected in _RELAY_SYMBOL_MANIFEST.items()
                if symbol in expected
            )
        ]


def test_relay_imports_as_private_package_in_clean_subprocess() -> None:
    moved_symbols = sorted(_RELAY_MANIFEST_SYMBOLS)
    code = f"""
import importlib.util
from pathlib import Path

import claudex.relay as relay
import claudex.relay.endpoints

spec = importlib.util.find_spec("claudex.relay")
assert spec is not None and spec.submodule_search_locations is not None
assert Path(relay.__file__).as_posix().endswith("relay/__init__.py")
assert not [symbol for symbol in {moved_symbols!r} if hasattr(relay, symbol)]
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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

    async def count_tokens(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return httpx.Response(200, json={"input_tokens": 1})

    async def list_models(self) -> Any:
        return {"data": []}


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


def test_context_overflow_http_error_is_rewritten_for_claude_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rewrite_calls: list[int] = []
    real_rewrite = context_overflow.rewrite_context_overflow_message

    def counting_rewrite(*args: Any, **kwargs: Any) -> str | None:
        rewrite_calls.append(1)
        return real_rewrite(*args, **kwargs)

    monkeypatch.setattr(context_overflow, "rewrite_context_overflow_message", counting_rewrite)

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
    assert rewrite_calls == [1]


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
    kimi_backend_client: Any = FakeKimiClient()
    if kimi_handler is not None or kimi_auth is not None:
        kimi_backend_client = KimiClient(
            kimi_auth or AvailableKimiAuthManager(),
            httpx.AsyncClient(
                transport=httpx.MockTransport(
                    kimi_handler or (lambda request: httpx.Response(500))
                )
            ),
        )
    grok_backend_client = grok_client or FakeGrokClient()
    custom_backend_clients = custom_provider_clients or {}
    app.state.kimi_client = kimi_backend_client
    app.state.grok_client = grok_backend_client
    app.state.custom_provider_clients = custom_backend_clients
    app.state.route_backends = server._assemble_route_backends(
        app,
        config,
        stub,
        kimi_backend_client,
        grok_backend_client,
        custom_backend_clients,
    )
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
    caplog.set_level(logging.INFO, logger="claudex.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert stub.payloads[0]["service_tier"] == "priority"
    assert any(
        record.name == "claudex.server"
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
    caplog.set_level(logging.DEBUG, logger="claudex.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert "service_tier" not in stub.payloads[0]
    assert any(
        record.name == "claudex.server"
        and record.levelno == logging.DEBUG
        and record.getMessage()
        == "fast tier requested but the codex catalog does not advertise it for gpt-5.6-sol"
        for record in caplog.records
    )
    assert any(
        record.name == "claudex.server"
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
    caplog.set_level(logging.INFO, logger="claudex.server")

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert "service_tier" not in stub.payloads[0]
    assert any(
        record.name == "claudex.server"
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


def test_codex_pre_stream_overflow_uses_context_window_override() -> None:
    config = GatewayConfig(
        model_map={"opus": "codex:gpt-5.6-sol"},
        context_window_map={"codex:gpt-5.6-sol": 872000},
    )
    client, stub = _gateway(
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
    assert limit == 872000
    assert actual > limit
    assert stub.context_window_calls == []


def test_context_window_override_requires_exact_upstream_slug() -> None:
    config = GatewayConfig(
        model_map={"opus": "codex:gpt-5.6-sol"},
        context_window_map={"codex:gpt-5.6": 872000},
    )
    client, stub = _gateway(
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
    assert int(match.group(2)) == 272000
    assert stub.context_window_calls == ["gpt-5.6-sol"]


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

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
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


def test_streaming_mid_stream_overflow_uses_context_window_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "codex:gpt-5.6-sol"},
        context_window_map={"codex:gpt-5.6-sol": 872000},
    )
    body = _message_body("claude-opus-4-6")
    body["stream"] = True

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        codex_client=RecordingMidStreamOverflowCodexClient,
    ) as client:
        stub = client.app.state.codex_client
        response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    events = [event for event in response.text.split("\n\n") if event]
    error_event = next(event for event in events if event.startswith("event: error"))
    payload = json.loads(error_event.split("data: ", 1)[1])
    match = _CLIENT_OVERFLOW_NUMBERS_RE.search(payload["error"]["message"])
    assert match is not None
    assert int(match.group(2)) == 872000
    assert stub.context_window_calls == []


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
    app.state.route_backends = server._assemble_route_backends(
        app, config, stub, FakeKimiClient(), FakeGrokClient(), {}
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
    app.state.route_backends = server._assemble_route_backends(
        app,
        config,
        app.state.codex_client,
        FakeKimiClient(),
        FakeGrokClient(),
        {},
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


class _TrackingByteStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        error: httpx.HTTPError | None = None,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.close_calls += 1


def _stream_backend(transport: Any | None = None) -> AnthropicBackend:
    async def count_tokens(
        body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return httpx.Response(200, json={"input_tokens": 1})

    return AnthropicBackend(
        transport=transport or FakeKimiClient(),
        header_policy=_kimi_request_headers,
        error_policy=_kimi_error_to_claude,
        token_counter=count_tokens,
    )


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


def test_anthropic_stream_yields_complete_sse_events() -> None:
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

    async def scenario() -> tuple[list[bytes], _TrackingByteStream]:
        stream = _TrackingByteStream([sse_body])
        response = httpx.Response(200, stream=stream)
        chunks = [
            chunk
            async for chunk in _AnthropicStreamRelay(
                response, "claude-fable-5", _stream_backend()
            )
        ]
        return chunks, stream

    chunks, stream = asyncio.run(scenario())

    assert len(chunks) == 3
    assert chunks[0].endswith(b"\n\n")
    assert b'"model": "claude-fable-5"' in chunks[0]
    assert chunks[1] == (
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
    )
    assert chunks[2] == b'event: message_stop\ndata: {"type":"message_stop"}\n'
    assert stream.close_calls == 1


def test_anthropic_stream_early_close_before_first_read_closes_once() -> None:
    async def scenario() -> tuple[bool, int]:
        stream = _TrackingByteStream([b"event: message_stop\n\n"])
        response = httpx.Response(200, stream=stream)
        relay = _AnthropicStreamRelay(
            response, "claude-fable-5", _stream_backend()
        )
        await relay.aclose()
        await relay.aclose()
        return response.is_closed, stream.close_calls

    is_closed, close_calls = asyncio.run(scenario())

    assert is_closed is True
    assert close_calls == 1


def test_anthropic_stream_read_failure_emits_error_and_closes_once() -> None:
    async def scenario() -> tuple[list[bytes], int]:
        stream = _TrackingByteStream(
            [b"event: content_block_delta\n"],
            httpx.ReadError("stream failed"),
        )
        response = httpx.Response(200, stream=stream)
        chunks = [
            chunk
            async for chunk in _AnthropicStreamRelay(
                response, "claude-fable-5", _stream_backend()
            )
        ]
        return chunks, stream.close_calls

    chunks, close_calls = asyncio.run(scenario())

    assert chunks[0] == b"event: content_block_delta\n"
    assert chunks[1].startswith(b"event: error\ndata: ")
    error = json.loads(chunks[1].split(b"data: ", 1)[1])
    assert error["error"]["type"] == "api_error"
    assert "kimi stream aborted" in error["error"]["message"]
    assert close_calls == 1


def test_anthropic_stream_cancellation_interrupts_buffered_events() -> None:
    close_calls = 0
    sse_body = b"".join(
        f'event: content_block_delta\ndata: {{"index":{index}}}\n\n'.encode()
        for index in range(20)
    )

    class BufferedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield sse_body

        async def aclose(self) -> None:
            nonlocal close_calls
            close_calls += 1

    async def scenario() -> list[bytes]:
        response = httpx.Response(200, stream=BufferedStream())
        chunks: list[bytes] = []
        first_chunk_seen = asyncio.Event()

        async def consume() -> None:
            async for chunk in _AnthropicStreamRelay(
                response, "claude-fable-5", _stream_backend()
            ):
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
    assert close_calls == 1


@pytest.mark.parametrize(
    ("upstream_body", "expected_body"),
    [
        (
            b'{"type":"message","model":"native-model"}',
            {"type": "message", "model": "claude-fable-5"},
        ),
        (b"not-json", b"not-json"),
    ],
)
def test_anthropic_non_streaming_response_closes_once(
    upstream_body: bytes, expected_body: dict[str, Any] | bytes
) -> None:
    stream = _TrackingByteStream([upstream_body])
    upstream_response = httpx.Response(
        200,
        headers={
            "content-type": "application/x-native-message",
            "request-id": "req_native",
        },
        stream=stream,
    )
    sent: list[bytes] = []

    class FakeNativeTransport:
        async def send_messages(
            self, body: bytes, headers: dict[str, str]
        ) -> httpx.Response:
            sent.append(body)
            return upstream_response

    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    )
    response = asyncio.run(
        _relay_via_anthropic_backend(
            request,
            _message_body("claude-fable-5"),
            "native-model",
            _stream_backend(FakeNativeTransport()),
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-native-message"
    assert response.headers["request-id"] == "req_native"
    if isinstance(expected_body, dict):
        assert json.loads(response.body) == expected_body
    else:
        assert response.body == expected_body
    assert json.loads(sent[0])["model"] == "native-model"
    assert upstream_response.is_closed is True
    assert stream.close_calls == 1


def test_anthropic_non_streaming_read_failure_propagates_and_closes_once() -> None:
    stream = _TrackingByteStream([], httpx.ReadError("body failed"))
    upstream_response = httpx.Response(200, stream=stream)

    class FakeNativeTransport:
        async def send_messages(
            self, body: bytes, headers: dict[str, str]
        ) -> httpx.Response:
            return upstream_response

    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/messages", "headers": []}
    )
    with pytest.raises(httpx.ReadError, match="body failed"):
        asyncio.run(
            _relay_via_anthropic_backend(
                request,
                _message_body("claude-fable-5"),
                "native-model",
                _stream_backend(FakeNativeTransport()),
            )
        )

    assert upstream_response.is_closed is True
    assert stream.close_calls == 1


def test_anthropic_count_tokens_owns_success_malformed_and_read_failure_responses() -> None:
    success_stream = _TrackingByteStream([b'{"input_tokens":41}'])
    success_response = httpx.Response(200, stream=success_stream)
    malformed_stream = _TrackingByteStream([b"not-json"])
    malformed_response = httpx.Response(200, stream=malformed_stream)
    failed_stream = _TrackingByteStream([], httpx.ReadError("count failed"))
    failed_response = httpx.Response(200, stream=failed_stream)
    responses = [success_response, malformed_response, failed_response]

    async def token_counter(
        body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return responses.pop(0)

    backend = AnthropicBackend(
        transport=FakeKimiClient(),
        header_policy=_kimi_request_headers,
        error_policy=_kimi_error_to_claude,
        token_counter=token_counter,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages/count_tokens",
            "headers": [],
        }
    )

    async def scenario() -> tuple[Response | None, Response | None, Response | None]:
        body = _message_body("claude-fable-5")
        return (
            await _count_tokens_via_anthropic_backend(
                request, body, "native-model", backend
            ),
            await _count_tokens_via_anthropic_backend(
                request, body, "native-model", backend
            ),
            await _count_tokens_via_anthropic_backend(
                request, body, "native-model", backend
            ),
        )

    success, malformed, failed = asyncio.run(scenario())

    assert success is not None
    assert json.loads(success.body) == {"input_tokens": 41}
    assert malformed is None
    assert failed is None
    assert success_stream.close_calls == 1
    assert malformed_stream.close_calls == 1
    assert failed_stream.close_calls == 1
    assert success_response.is_closed is True
    assert malformed_response.is_closed is True
    assert failed_response.is_closed is True


def test_anthropic_count_tokens_transport_error_returns_no_response() -> None:
    async def token_counter(
        body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        raise KimiUpstreamError(503, "counter unavailable")

    backend = AnthropicBackend(
        transport=FakeKimiClient(),
        header_policy=_kimi_request_headers,
        error_policy=_kimi_error_to_claude,
        token_counter=token_counter,
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages/count_tokens",
            "headers": [],
        }
    )

    counted = asyncio.run(
        _count_tokens_via_anthropic_backend(
            request, _message_body("claude-fable-5"), "native-model", backend
        )
    )

    assert counted is None


def test_anthropic_backend_without_token_counter_uses_exact_local_estimate_without_io() -> None:
    class MessagesOnlyTransport:
        async def send_messages(
            self, body: bytes, headers: dict[str, str]
        ) -> httpx.Response:
            raise AssertionError("Messages transport must not handle token counting")

    def header_policy(request: Request) -> dict[str, str]:
        raise AssertionError("Header policy must not run without a token counter")

    def error_policy(exc: Any) -> tuple[int, dict[str, Any]]:
        raise AssertionError(f"Unexpected backend error: {exc!r}")

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Local estimation must not perform upstream I/O")

    config = GatewayConfig(model_map={"opus": "codex:native-model"})
    client, _ = _gateway(config, anthropic_handler)
    client.app.state.route_backends["codex"] = AnthropicBackend(
        transport=MessagesOnlyTransport(),
        header_policy=header_policy,
        error_policy=error_policy,
        token_counter=None,
        catalog_loader=None,
    )
    body = _message_body("claude-opus-4-6")

    response = client.post("/v1/messages/count_tokens", json=body)

    expected = max(len(json.dumps(body, ensure_ascii=False)) // 4, 1)
    assert response.status_code == 200
    assert response.json() == {"input_tokens": expected}


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
    app.state.route_backends = {
        "kimi": _stream_backend(app.state.kimi_client),
    }

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
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.context_window_calls: list[str] = []
        self.events = events

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return None

    async def list_models(self) -> list[str]:
        return []

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.payloads.append(payload)
        if self.events is None:
            raise OpenAICompatibleUpstreamError(503, "stub custom upstream", "wrtn")
        for event in self.events:
            yield event


@pytest.mark.parametrize(
    ("provider", "signature_namespace"),
    [("codex", None), ("wrtn", "wrtn")],
)
def test_responses_backend_translates_then_awaits_adapter_and_uses_bound_transport(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    signature_namespace: str | None,
) -> None:
    upstream_model = "selected-model"
    call_order: list[tuple[Any, ...]] = []
    translated_payload = {
        "model": upstream_model,
        "prompt_cache_key": "session-1",
        "input": [],
    }
    adapted_payload = {**translated_payload, "adapted": True}

    def recording_translate(
        claude_request: dict[str, Any],
        model: str,
        reasoning_effort_override: str | None,
        *,
        service_tier: str | None = None,
        custom_provider: str | None = None,
    ) -> dict[str, Any]:
        call_order.append(("translate", custom_provider))
        assert model == upstream_model
        assert service_tier is None
        return translated_payload

    async def adapt_payload(
        payload: dict[str, Any], model: str
    ) -> dict[str, Any]:
        call_order.append(("adapt", payload, model))
        assert payload is translated_payload
        return adapted_payload

    def adapt_probe_payload(
        payload: dict[str, Any], model: str
    ) -> dict[str, Any]:
        return payload

    class RecordingTransport(StubCodexClient):
        async def stream_responses(
            self, payload: dict[str, Any], session_id: str
        ) -> AsyncIterator[dict[str, Any]]:
            call_order.append(("stream", payload, session_id))
            self.payloads.append(payload)
            raise CodexUpstreamError(503, "selected transport")
            yield

    monkeypatch.setattr(
        relay_openai_backend,
        "translate_claude_request_to_codex",
        recording_translate,
    )
    custom_providers = {"wrtn": _custom_provider()} if provider == "wrtn" else {}
    configured_custom_transport = StubOpenAICompatibleClient()
    custom_transports = (
        {"wrtn": configured_custom_transport} if provider == "wrtn" else {}
    )
    config = GatewayConfig(
        model_map={"opus": f"{provider}:{upstream_model}"},
        context_window_map={f"{provider}:{upstream_model}": 100_000},
        custom_providers=custom_providers,
    )
    client, codex_transport = _gateway(
        config,
        _failing_anthropic_handler,
        custom_provider_clients=custom_transports,
    )
    selected_transport = RecordingTransport()
    client.app.state.route_backends[provider] = ResponsesBackend(
        transport=selected_transport,
        adapt_payload=adapt_payload,
        adapt_probe_payload=adapt_probe_payload,
        signature_namespace=signature_namespace,
    )

    response = client.post(
        "/v1/messages", json=_message_body("claude-opus-4-6")
    )

    assert response.status_code == 503
    assert call_order == [
        ("translate", signature_namespace),
        ("adapt", translated_payload, upstream_model),
        ("stream", adapted_payload, "session-1"),
    ]
    assert selected_transport.payloads == [adapted_payload]
    assert codex_transport.payloads == []
    assert configured_custom_transport.payloads == []


def test_anthropic_wire_dispatch_is_independent_of_provider_name() -> None:
    sent: list[tuple[bytes, dict[str, str]]] = []
    counted: list[tuple[bytes, dict[str, str]]] = []
    header_paths: list[str] = []

    class FakeNativeTransport:
        async def send_messages(
            self, body: bytes, headers: dict[str, str]
        ) -> httpx.Response:
            sent.append((body, headers))
            return httpx.Response(
                200,
                json={"type": "message", "model": "native-model"},
                headers={"request-id": "req_native"},
            )

    def header_policy(request: Any) -> dict[str, str]:
        header_paths.append(request.url.path)
        return {"x-native-policy": "selected"}

    def error_policy(exc: Any) -> tuple[int, dict[str, Any]]:
        raise AssertionError(f"unexpected native backend error: {exc!r}")

    async def token_counter(
        body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        counted.append((body, headers))
        return httpx.Response(200, json={"input_tokens": 73})

    config = GatewayConfig(model_map={"opus": "codex:native-model"})
    client, codex_transport = _gateway(config, _failing_anthropic_handler)
    client.app.state.route_backends["codex"] = AnthropicBackend(
        transport=FakeNativeTransport(),
        header_policy=header_policy,
        error_policy=error_policy,
        token_counter=token_counter,
    )

    message = client.post(
        "/v1/messages", json=_message_body("claude-opus-4-6")
    )
    count = client.post(
        "/v1/messages/count_tokens",
        json=_message_body("claude-opus-4-6"),
    )

    assert message.status_code == 200
    assert message.json()["model"] == "claude-opus-4-6"
    assert message.headers["request-id"] == "req_native"
    assert count.status_code == 200
    assert count.json() == {"input_tokens": 73}
    assert header_paths == ["/v1/messages", "/v1/messages/count_tokens"]
    assert json.loads(sent[0][0])["model"] == "native-model"
    assert sent[0][1] == {"x-native-policy": "selected"}
    assert json.loads(counted[0][0])["model"] == "native-model"
    assert counted[0][1] == {"x-native-policy": "selected"}
    assert codex_transport.payloads == []


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_mapped_route_fails_clearly_when_backend_binding_is_missing(path: str) -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(config, _failing_anthropic_handler)
    del client.app.state.route_backends["codex"]

    with pytest.raises(
        RuntimeError,
        match="route backend registry has no binding for provider 'codex'",
    ):
        client.post(path, json=_message_body("claude-opus-4-6"))


def test_mapped_route_fails_clearly_for_unsupported_backend_binding() -> None:
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.6-sol"})
    client, _ = _gateway(config, _failing_anthropic_handler)
    client.app.state.route_backends["codex"] = object()

    with pytest.raises(
        RuntimeError,
        match=(
            "route backend registry contains an unsupported binding for "
            "provider 'codex': object"
        ),
    ):
        client.post("/v1/messages", json=_message_body("claude-opus-4-6"))


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

    async def unexpected_grok_adapter(
        payload: dict[str, Any], model: str
    ) -> dict[str, Any]:
        raise AssertionError("Grok adapter must not run for a custom provider")

    monkeypatch.setattr(server, "_adapt_grok_payload", unexpected_grok_adapter)
    assert server._adapt_grok_payload is unexpected_grok_adapter
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


def test_context_window_override_is_scoped_to_provider() -> None:
    custom_stub = StubOpenAICompatibleClient()
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        context_window_map={"codex:gpt-5.5": 872000},
        custom_providers={"wrtn": _custom_provider()},
    )
    client, _ = _gateway(
        config,
        _failing_anthropic_handler,
        custom_provider_clients={"wrtn": custom_stub},
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    assert custom_stub.context_window_calls == ["gpt-5.5"]


def _custom_function_call_events(
    call_id: str, thought_signature: str
) -> list[dict[str, Any]]:
    return [
        {
            "type": "response.created",
            "response": {"id": "resp_custom", "model": "gpt-5.5"},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "extra_content": {
                    "google": {"thought_signature": thought_signature}
                },
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": call_id,
                "name": "lookup",
                "arguments": "{}",
                "extra_content": {
                    "google": {"thought_signature": thought_signature}
                },
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_custom",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "output": [],
            },
        },
    ]


def _carrier_thinking_block(
    provider: str, call_id: str, thought_signature: str
) -> dict[str, Any]:
    carrier = encode_call_signature_carrier(provider, call_id, thought_signature)
    assert carrier is not None
    return {"type": "thinking", "thinking": "", "signature": carrier}


def _custom_provider_gateway(custom_stub: StubOpenAICompatibleClient) -> TestClient:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    client, _ = _gateway(
        config,
        _failing_anthropic_handler,
        custom_provider_clients={"wrtn": custom_stub},
    )
    return client


def _message_with_carrier(
    provider: str, call_id: str, thought_signature: str
) -> dict[str, Any]:
    body = _message_body("claude-opus-4-6")
    body["messages"] = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": call_id, "name": "lookup", "input": {}},
                _carrier_thinking_block(provider, call_id, thought_signature),
            ],
        }
    ]
    return body


def test_custom_provider_stream_emits_carrier_thinking_block() -> None:
    call_id = "call_stream"
    thought_signature = "stream-signature+/=한글"
    custom_stub = StubOpenAICompatibleClient(
        _custom_function_call_events(call_id, thought_signature)
    )
    client = _custom_provider_gateway(custom_stub)
    body = _message_body("claude-opus-4-6")
    body["stream"] = True

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    payloads = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in response.text.split("\n\n")
        if frame
    ]
    block_starts = [
        payload
        for payload in payloads
        if payload.get("type") == "content_block_start"
    ]
    assert [payload["content_block"]["type"] for payload in block_starts] == [
        "tool_use",
        "thinking",
    ]
    tool_use_block = block_starts[0]["content_block"]
    assert block_starts[1]["content_block"] == {
        "type": "thinking",
        "thinking": "",
    }
    signature_delta = next(
        payload["delta"]
        for payload in payloads
        if payload.get("delta", {}).get("type") == "signature_delta"
    )
    assert decode_call_signature_carrier(signature_delta["signature"]) == (
        CallSignatureCarrier(
            provider="wrtn",
            call_id=tool_use_block["id"],
            signature=thought_signature,
        )
    )


def test_custom_provider_nonstream_emits_carrier_thinking_block() -> None:
    call_id = "call_nonstream"
    thought_signature = "nonstream-signature+/=한글"
    custom_stub = StubOpenAICompatibleClient(
        _custom_function_call_events(call_id, thought_signature)
    )
    client = _custom_provider_gateway(custom_stub)
    body = _message_body("claude-opus-4-6")
    body["stream"] = False

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200
    message = response.json()
    assert [block["type"] for block in message["content"]] == [
        "tool_use",
        "thinking",
    ]
    tool_use_block, carrier_block = message["content"]
    assert carrier_block["thinking"] == ""
    assert decode_call_signature_carrier(carrier_block["signature"]) == (
        CallSignatureCarrier(
            provider="wrtn",
            call_id=tool_use_block["id"],
            signature=thought_signature,
        )
    )


def test_custom_provider_replay_attaches_extra_content_upstream() -> None:
    call_id = "call_replay"
    thought_signature = "replayed-signature+/=한글"
    custom_stub = StubOpenAICompatibleClient()
    client = _custom_provider_gateway(custom_stub)
    body = _message_with_carrier("wrtn", call_id, thought_signature)

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 503
    function_call = next(
        item for item in custom_stub.payloads[0]["input"] if item["type"] == "function_call"
    )
    assert function_call["extra_content"] == {
        "google": {"thought_signature": thought_signature}
    }


def test_builtin_route_payload_has_no_extra_content_even_with_forged_carrier() -> None:
    call_id = "call_forged"
    body = _message_with_carrier("wrtn", call_id, "forged-signature")
    config = GatewayConfig(model_map={"opus": "codex:gpt-5.5"})
    client, codex_stub = _gateway(config, _failing_anthropic_handler)

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 503
    (payload,) = codex_stub.payloads
    assert all("extra_content" not in item for item in payload["input"])
    assert all(item["type"] != "reasoning" for item in payload["input"])


def test_custom_provider_cross_name_carrier_not_replayed() -> None:
    call_id = "call_cross_name"
    custom_stub = StubOpenAICompatibleClient()
    client = _custom_provider_gateway(custom_stub)
    body = _message_with_carrier(
        "other-provider", call_id, "foreign-signature"
    )

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 503
    function_call = next(
        item for item in custom_stub.payloads[0]["input"] if item["type"] == "function_call"
    )
    assert "extra_content" not in function_call


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
    expected_estimate = context_overflow.estimate_overflow_prompt_tokens(body)
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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

    monkeypatch.setattr(relay_openai_backend, "translate_claude_request_to_codex", _boom)
    assert relay_openai_backend.translate_claude_request_to_codex is _boom

    body = _compaction_body("claude-opus-4-6")
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
        custom_provider: str | None = None,
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            reasoning_effort_override,
            service_tier=service_tier,
            custom_provider=custom_provider,
        )

    monkeypatch.setattr(relay_openai_backend, "translate_claude_request_to_codex", recording_translate)

    body = _compaction_body("claude-opus-4-6", thinking_block=True)
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    real_is_compaction_request = relay_openai_backend.is_compaction_request
    real_estimate = context_overflow.estimate_overflow_prompt_tokens

    def counting_is_compaction_request(*args: Any, **kwargs: Any) -> bool:
        signal_calls["count"] += 1
        return real_is_compaction_request(*args, **kwargs)

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(relay_openai_backend, "is_compaction_request", counting_is_compaction_request)
    monkeypatch.setattr(context_overflow, "estimate_overflow_prompt_tokens", counting_estimate)
    assert relay_openai_backend.is_compaction_request is counting_is_compaction_request
    assert context_overflow.estimate_overflow_prompt_tokens is counting_estimate

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
    real_estimate = context_overflow.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(context_overflow, "estimate_overflow_prompt_tokens", counting_estimate)
    assert context_overflow.estimate_overflow_prompt_tokens is counting_estimate

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) + 1
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


def test_compaction_trigger_uses_context_window_override() -> None:
    body = _compaction_body("claude-opus-4-6")
    estimated_prompt_tokens = context_overflow.estimate_overflow_prompt_tokens(body)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, json={"error": "unexpected"})

    config = _compaction_config(
        {"opus": "codex:gpt-5.1-codex-max"},
        context_window_map={"codex:gpt-5.1-codex-max": estimated_prompt_tokens + 1},
    )
    client, stub = _gateway(
        config,
        handler,
        codex_context_window=estimated_prompt_tokens - 1,
    )

    response = client.post(
        "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
    )

    assert response.status_code == 503
    assert captured == []
    assert stub.payloads
    assert stub.context_window_calls == []
    assert client.app.state.compaction_last_reroute is None


def test_compaction_trigger_equal_window_does_not_reroute() -> None:
    body = _compaction_body("claude-opus-4-6")
    window = context_overflow.estimate_overflow_prompt_tokens(body)
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
    real_estimate = context_overflow.estimate_overflow_prompt_tokens

    def counting_estimate(*args: Any, **kwargs: Any) -> int:
        estimate_calls["count"] += 1
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(context_overflow, "estimate_overflow_prompt_tokens", counting_estimate)
    assert context_overflow.estimate_overflow_prompt_tokens is counting_estimate

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    expected_estimate = context_overflow.estimate_overflow_prompt_tokens(body)
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1

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
        custom_provider: str | None = None,
    ) -> dict[str, Any]:
        # A deep, JSON-round-tripped copy: proves equality without ever
        # aliasing the mutable dict the caller still holds.
        received.append(json.loads(json.dumps(claude_request)))
        return translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            reasoning_effort_override,
            service_tier=service_tier,
            custom_provider=custom_provider,
        )

    monkeypatch.setattr(relay_openai_backend, "translate_claude_request_to_codex", recording_translate)

    body = _compaction_body("claude-opus-4-6", thinking_block=True)
    body["stream"] = True
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1

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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    window = context_overflow.estimate_overflow_prompt_tokens(body) - 1
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
    sequence = relay_openai_backend._assign_compaction_reroute(
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
    sequence = relay_openai_backend._assign_compaction_reroute(
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

    sequence_a = relay_openai_backend._assign_compaction_reroute(
        app_state,
        outcome="rerouted",
        target_model=_COMPACTION_CANONICAL_TARGET,
        mapped_model="codex:gpt-5.1-codex-max",
        estimated_prompt_tokens=100,
        context_window=50,
        detail=None,
    )
    relay_openai_backend._assign_compaction_reroute(
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
    sequence = relay_openai_backend._assign_compaction_reroute(
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
    sequence = relay_openai_backend._assign_compaction_reroute(
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
    sequence = relay_openai_backend._assign_compaction_reroute(
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


def _claude_code_user_id(
    account_uuid: str = "client-account-uuid",
    *,
    session_id: str = "11111111-2222-4333-8444-555555555555",
) -> str:
    return json.dumps(
        {
            "device_id": "d" * 64,
            "account_uuid": account_uuid,
            "session_id": session_id,
        },
        separators=(",", ":"),
    )


def _account_body(model: str = "claude-fable-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {"user_id": _claude_code_user_id()}
    return body


_LEG_RAW_SESSION_UUID = "550E8400-E29B-41D4-A716-446655440000"
_LEG_CANONICAL_SESSION_UUID = "550e8400-e29b-41d4-a716-446655440000"
_LEG_PROMPT_CONTENT = "PROMPT-CONTENT-MUST-NOT-APPEAR-IN-ACCOUNT-LEG-LOG"
_LEG_TOOL_ARGUMENT = "TOOL-ARGUMENT-MUST-NOT-APPEAR-IN-ACCOUNT-LEG-LOG"


def _account_leg_body(
    session_id: str = _LEG_RAW_SESSION_UUID,
) -> dict[str, Any]:
    canonical_session_id = str(uuid.UUID(session_id))
    body = _message_body(
        f"claude-fable-5-{session_id}-{canonical_session_id}"
    )
    body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _LEG_PROMPT_CONTENT},
                {
                    "type": "tool_use",
                    "id": "toolu_fixture",
                    "name": "fixture",
                    "input": {"secret": _LEG_TOOL_ARGUMENT},
                },
            ],
        }
    ]
    body["tools"] = [
        {
            "name": "fixture",
            "description": _LEG_TOOL_ARGUMENT,
            "input_schema": {"type": "object"},
        }
    ]
    body["metadata"] = {"user_id": _claude_code_user_id(session_id=session_id)}
    return body


def _account_leg_envelopes(
    caplog: pytest.LogCaptureFixture,
    *,
    extra_session_literals: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    envelopes = []
    for record in caplog.records:
        if record.name != "claudex.server":
            continue
        message = record.getMessage()
        try:
            envelope = json.loads(message)
        except json.JSONDecodeError:
            continue
        if not isinstance(envelope, dict) or envelope.get("event") != "claude_account_leg":
            continue
        assert message == json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert set(envelope) == {
            "v",
            "event",
            "occurred_at_utc",
            "mode",
            "account_id",
            "model",
            "result",
            "attempt",
            "request_shape",
        }
        assert set(envelope["request_shape"]) == {
            "body_bytes",
            "message_count",
            "tool_count",
            "pin_created",
        }
        for forbidden in (
            _LEG_PROMPT_CONTENT,
            _LEG_TOOL_ARGUMENT,
            _LEG_RAW_SESSION_UUID,
            _LEG_CANONICAL_SESSION_UUID,
            *extra_session_literals,
        ):
            assert forbidden not in message
        envelopes.append(envelope)
    return envelopes


class _RaisingLegLogger:
    def info(self, message: Any, *args: Any, **_kwargs: Any) -> None:
        if not args and isinstance(message, str) and '"event":"claude_account_leg"' in message:
            raise RuntimeError("account-leg info logging failed")

    def warning(self, message: Any, *_args: Any, **_kwargs: Any) -> None:
        if message == "failed to emit Claude account-leg log":
            raise RuntimeError("account-leg warning logging failed")


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


def test_fallback_429_schedules_forced_usage_refresh_without_blocking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    poll_started = asyncio.Event()
    poll_release = asyncio.Event()
    poll_completed = asyncio.Event()
    poll_calls: list[tuple[str, bool]] = []

    class BlockingUsageCache:
        def peek(self, _account_id: str) -> None:
            return None

        async def poll(self, marked_account_id: str, *, force: bool = False) -> Any:
            poll_calls.append((marked_account_id, force))
            poll_started.set()
            await poll_release.wait()
            poll_completed.set()
            return SimpleNamespace(source="fetched", result={"status": "ok"})

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429("fallback-refresh")

    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(account_id)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        client.app.state.claude_account_usage_cache = BlockingUsageCache()

        async def exercise_request() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app),
                base_url="http://testserver",
            ) as async_client:
                response_task = asyncio.create_task(
                    async_client.post("/v1/messages", json=_account_body())
                )
                await asyncio.wait_for(poll_started.wait(), timeout=1.0)
                response = await asyncio.wait_for(response_task, timeout=1.0)
                assert not poll_completed.is_set()
                assert len(client.app.state.claude_usage_refresh_tasks) == 1
                poll_release.set()
                await asyncio.wait_for(poll_completed.wait(), timeout=1.0)
                await asyncio.sleep(0)
                assert client.app.state.claude_usage_refresh_tasks == set()
                return response

        assert client.portal is not None
        response = client.portal.call(exercise_request)

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "fallback-refresh"
    assert poll_calls == [(account_id, True)]


def test_fallback_refresh_scheduling_failure_preserves_429(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    real_create_task = asyncio.create_task

    def fail_task_creation(coroutine, *args, **kwargs):
        if coroutine.cr_code.co_name == "_refresh_usage_after_quota_429":
            raise RuntimeError("simulated scheduling failure")
        return real_create_task(coroutine, *args, **kwargs)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429("scheduling-failure")

    monkeypatch.setattr(
        relay_registered.asyncio, "create_task", fail_task_creation
    )
    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(account_id)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        response = client.post("/v1/messages", json=_account_body())
        assert client.app.state.claude_usage_refresh_tasks == set()

    assert response.status_code == 429
    assert response.json()["error"]["message"] == "scheduling-failure"
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["result"] == "rate_limited"


def test_fallback_refresh_unexpected_exception_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    poll_completed = asyncio.Event()
    poll_calls: list[tuple[str, bool]] = []

    class RaisingUsageCache:
        def peek(self, _account_id: str) -> None:
            return None

        async def poll(self, marked_account_id: str, *, force: bool = False) -> None:
            poll_calls.append((marked_account_id, force))
            poll_completed.set()
            raise RuntimeError("simulated refresh task failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    caplog.set_level(logging.WARNING)
    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(account_id)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        client.app.state.claude_account_usage_cache = RaisingUsageCache()

        async def exercise_request() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app),
                base_url="http://testserver",
            ) as async_client:
                response = await async_client.post(
                    "/v1/messages", json=_account_body()
                )
                await asyncio.wait_for(poll_completed.wait(), timeout=1.0)
                await asyncio.sleep(0)
                return response

        assert client.portal is not None
        response = client.portal.call(exercise_request)

    refresh_warnings = [
        record
        for record in caplog.records
        if record.getMessage()
        == "post-429 Claude usage refresh failed unexpectedly"
    ]
    assert response.status_code == 429
    assert poll_calls == [(account_id, True)]
    assert len(refresh_warnings) == 1
    assert refresh_warnings[0].exc_info is None
    assert "simulated refresh task failure" not in caplog.text


def test_fallback_refresh_cache_error_result_not_double_warned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    poll_completed = asyncio.Event()
    poll_calls: list[tuple[str, bool]] = []
    cache_logger = logging.getLogger("claudex.claude.account_usage_cache")

    class ErrorResultUsageCache:
        def peek(self, _account_id: str) -> None:
            return None

        async def poll(self, marked_account_id: str, *, force: bool = False) -> Any:
            poll_calls.append((marked_account_id, force))
            cache_logger.warning("simulated cache-owned usage refresh failure")
            poll_completed.set()
            return SimpleNamespace(
                source="fetched",
                result={"status": "error", "error": "usage probe failed"},
            )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    caplog.set_level(logging.WARNING)
    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(account_id)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        client.app.state.claude_account_usage_cache = ErrorResultUsageCache()

        async def exercise_request() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app),
                base_url="http://testserver",
            ) as async_client:
                response = await async_client.post(
                    "/v1/messages", json=_account_body()
                )
                await asyncio.wait_for(poll_completed.wait(), timeout=1.0)
                await asyncio.sleep(0)
                return response

        assert client.portal is not None
        response = client.portal.call(exercise_request)

    cache_warnings = [
        record
        for record in caplog.records
        if record.getMessage() == "simulated cache-owned usage refresh failure"
    ]
    relay_warnings = [
        record
        for record in caplog.records
        if record.getMessage()
        == "post-429 Claude usage refresh failed unexpectedly"
    ]
    assert response.status_code == 429
    assert poll_calls == [(account_id, True)]
    assert len(cache_warnings) == 1
    assert relay_warnings == []


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

    client, _ = _gateway(
        _pool_config(first), _usage_probe_intercepting_handler(handler)
    )

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

    client, _ = _gateway(
        _pool_config(first), _usage_probe_intercepting_handler(handler)
    )

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

    client, _ = _gateway(
        _pool_config(first), _usage_probe_intercepting_handler(handler)
    )

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

    client, _ = _gateway(
        _pool_config(account_id), _usage_probe_intercepting_handler(handler)
    )

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

    client, _ = _gateway(
        _pool_config(first), _usage_probe_intercepting_handler(handler)
    )
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


_USAGE_PROBE_PATH = "/api/oauth/usage"  # claude_usage._CLAUDE_USAGE_URL's path component


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


def test_balanced_429_enqueues_manual_usage_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    class RecordingUsagePollCoordinator:
        poll_interval_seconds = 300.0

        def __init__(self, router: ClaudeBalancedRouter) -> None:
            self.router = router
            self.account_ids: list[str] = []
            self.cooldown_installed: list[bool] = []

        async def run_due_poll(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def request_manual_refresh(self, marked_account_id: str) -> bool:
            self.account_ids.append(marked_account_id)
            self.cooldown_installed.append(
                self.router.account_cooldown_deadline(marked_account_id) is not None
            )
            return True

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)
        assert runtime.router is not None
        assert client.portal is not None
        client.portal.call(runtime._stop_usage_poll_driver)
        recording_coordinator = RecordingUsagePollCoordinator(runtime.router)
        runtime.usage_poll_coordinator = recording_coordinator

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert recording_coordinator.account_ids == [account_id]
        assert recording_coordinator.cooldown_installed == [True]


def test_balanced_refresh_enqueue_failure_preserves_evidence_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    class RaisingUsagePollCoordinator:
        poll_interval_seconds = 300.0

        async def run_due_poll(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def request_manual_refresh(self, _account_id: str) -> bool:
            raise RuntimeError("simulated enqueue failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    caplog.set_level(logging.WARNING, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)
        assert client.portal is not None
        client.portal.call(runtime._stop_usage_poll_driver)
        runtime.usage_poll_coordinator = RaisingUsagePollCoordinator()

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        incident_path = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        )
        assert incident_path.read_bytes() == row.evidence.encode() + b"\n"
        assert "failed to enrich Claude 429 incident evidence" not in caplog.text
        assert "simulated enqueue failure" not in caplog.text


def test_balanced_429_refresh_enqueue_coalesces_per_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)
        coordinator = runtime.usage_poll_coordinator
        assert coordinator is not None
        assert client.portal is not None
        client.portal.call(runtime._stop_usage_poll_driver)
        assert coordinator.request_manual_refresh(account_id) is True
        assert coordinator.diagnostics().manual_enqueued_count == 1

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert coordinator.is_manual_refresh_pending(account_id) is True
        assert coordinator.diagnostics().manual_enqueued_count == 1


def test_balanced_429_persists_evidence_json_with_required_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        evidence = json.loads(row.evidence)
        assert set(evidence) == {
            "v",
            "event",
            "occurred_at_utc",
            "mode",
            "account_id",
            "model",
            "cooldown_seconds",
            "cooldown_source",
            "installed_scope",
            "quota_family",
            "family_gate",
            "observed_scope",
            "scope_rationale",
            "upstream",
            "attempt",
            "session_fingerprint",
            "request_shape",
        }
        assert evidence["event"] == "claude_quota_429"
        assert evidence["mode"] == "balanced"
        assert evidence["account_id"] == account_id
        assert evidence["installed_scope"] == "account"
        assert evidence["quota_family"] == "fable"
        assert evidence["observed_scope"] == "unknown"
        assert evidence["scope_rationale"] == "fable_weekly_not_saturated"
        assert evidence["session_fingerprint"]


def test_balanced_429_evidence_and_incident_jsonl_carry_identical_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        incident_path = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        )
        assert incident_path.read_bytes() == row.evidence.encode("utf-8") + b"\n"


def test_fallback_429_writes_incident_record_without_durable_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(first)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        response = client.post("/v1/messages", json=_account_body())

        assert response.status_code == 200
        incident_path = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        )
        records = [json.loads(line) for line in incident_path.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["mode"] == "fallback"
        assert records[0]["installed_scope"] == "account"
        assert records[0]["quota_family"] is None
        assert records[0]["family_gate"] is None
        assert records[0]["scope_rationale"] == "fallback_no_family_gate"
        assert client.app.state.claude_balanced_runtime.router is None


def test_429_evidence_captures_request_id_from_header_and_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "slow down"},
                "request_id": "request-from-body",
            },
            headers={"request-id": "request-from-header"},
        )

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        upstream = json.loads(row.evidence)["upstream"]
        assert upstream["headers"]["request-id"] == "request-from-header"
        assert upstream["request_id"] == "request-from-body"


def test_429_evidence_preserves_first_duplicate_allowlisted_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers=[
                ("request-id", "first-request-id"),
                ("request-id", "second-request-id"),
                ("x-should-retry", "true"),
            ],
        )

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        headers = json.loads(row.evidence)["upstream"]["headers"]
        assert headers == {"request-id": "first-request-id"}
        assert "second-request-id" not in row.evidence


def test_429_evidence_omits_secrets_and_raw_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    credential = "Bearer secret-token-that-must-not-appear"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "rate_limit_error",
                    "message": f"{credential} {_LEG_RAW_SESSION_UUID}",
                }
            },
            headers={
                "authorization": credential,
                "x-api-key": "forbidden-api-key",
                "request-id": "safe-request-id",
            },
        )

    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _account_leg_body()
        body["model"] = "claude-fable-5"

        response = client.post("/v1/messages", json=body)

        assert response.status_code == 429
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        incident = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text()
        combined = row.evidence + incident + caplog.text
        for forbidden in (
            _LEG_RAW_SESSION_UUID,
            _LEG_CANONICAL_SESSION_UUID,
            _LEG_PROMPT_CONTENT,
            _LEG_TOOL_ARGUMENT,
            credential,
            "secret-token-that-must-not-appear",
            "forbidden-api-key",
        ):
            assert forbidden not in combined


def test_429_incident_records_cooldown_source_default_60(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(first)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        response = client.post("/v1/messages", json=_account_body())

        assert response.status_code == 200
        [line] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        record = json.loads(line)
        assert record["cooldown_seconds"] == 60.0
        assert record["cooldown_source"] == "default_60"


def test_429_timestamp_failure_replays_and_records_degraded_incident(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    upstream_returned = False
    response_body = b'{"error":{"type":"rate_limit_error","message":"quota reached"}}'
    original_datetime = relay_registered.datetime
    original_install_callback = relay_balanced._install_balanced_quota_cooldown
    install_calls: list[dict[str, Any]] = []

    class CountingCooldownTracker(AccountCooldownTracker):
        def __init__(self) -> None:
            super().__init__()
            self.mark_calls: list[tuple[str, float]] = []

        def mark(self, marked_account_id: str, seconds: float) -> None:
            self.mark_calls.append((marked_account_id, seconds))
            super().mark(marked_account_id, seconds)

    class FailingIncidentDatetime:
        call_count = 0

        @classmethod
        def now(cls, timezone_value: Any) -> Any:
            assert upstream_returned
            cls.call_count += 1
            if cls.call_count == 1:
                raise RuntimeError("simulated incident timestamp failure")
            return original_datetime.now(timezone_value)

    async def recording_install_callback(
        *args: Any, **kwargs: Any
    ) -> str:
        install_calls.append(kwargs)
        return await original_install_callback(*args, **kwargs)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_returned
        response = httpx.Response(
            429,
            content=response_body,
            headers={
                "request-id": "upstream-request-id",
                "retry-after": "17",
                "x-should-retry": "true",
            },
        )
        upstream_returned = True
        return response

    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)
        cooldown_tracker = CountingCooldownTracker()
        client.app.state.claude_account_cooldowns = cooldown_tracker
        appended_records: list[str] = []
        incident_writer = client.app.state.claude_quota_429_incident_writer
        original_append_record = incident_writer.append_record

        async def recording_append_record(canonical_record: str) -> None:
            appended_records.append(canonical_record)
            await original_append_record(canonical_record)

        monkeypatch.setattr(relay_registered, "datetime", FailingIncidentDatetime)
        monkeypatch.setattr(
            relay_balanced,
            "_install_balanced_quota_cooldown",
            recording_install_callback,
        )
        monkeypatch.setattr(
            incident_writer, "append_record", recording_append_record
        )

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert response.content == response_body
        assert response.headers["request-id"] == "upstream-request-id"
        assert response.headers["retry-after"] == "17"
        assert response.headers["x-should-retry"] == "true"
        assert cooldown_tracker.mark_calls == [(account_id, 17.0)]
        assert cooldown_tracker.is_cooling(account_id)
        assert len(install_calls) == 1
        assert install_calls[0]["account_id"] == account_id
        assert len(appended_records) == 1
        incident = json.loads(appended_records[0])
        mandatory_fields = {
            "v",
            "event",
            "occurred_at_utc",
            "mode",
            "account_id",
            "model",
            "cooldown_seconds",
            "cooldown_source",
            "installed_scope",
            "quota_family",
            "family_gate",
            "observed_scope",
            "scope_rationale",
            "upstream",
            "attempt",
            "session_fingerprint",
            "request_shape",
            "record_degraded",
            "degradation_reason",
        }
        assert mandatory_fields <= incident.keys()
        assert incident["v"] == 1
        assert incident["event"] == "claude_quota_429"
        assert incident["mode"] == "balanced"
        assert incident["cooldown_seconds"] == 17.0
        assert incident["cooldown_source"] == "retry_after"
        assert set(incident["upstream"]) == {
            "body_bytes",
            "body_sha256",
            "error_type",
            "message",
            "request_id",
            "headers",
        }
        assert set(incident["request_shape"]) == {
            "body_bytes",
            "message_count",
            "tool_count",
            "pin_created",
        }
        assert re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z",
            incident["occurred_at_utc"],
        )
        assert incident["record_degraded"] is True
        assert isinstance(incident["degradation_reason"], str)
        assert incident["degradation_reason"]
        incident_path = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        )
        assert incident_path.read_text() == appended_records[0] + "\n"

    assert FailingIncidentDatetime.call_count == 2
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["result"] == "rate_limited"


def test_429_evidence_failure_still_installs_cooldown_and_records_degraded_incident(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    install_evidence: list[str] = []
    original_install = ClaudeBalancedRouter.install_cooldown

    def fail_fingerprint(_seed: bytes, _session_id: str) -> str:
        raise RuntimeError("simulated evidence failure")

    def recording_install(
        self: ClaudeBalancedRouter, **kwargs: Any
    ) -> Any:
        install_evidence.append(kwargs["evidence"])
        return original_install(self, **kwargs)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    monkeypatch.setattr(
        relay_balanced, "observability_session_fingerprint", fail_fingerprint
    )
    monkeypatch.setattr(
        relay_balanced.ClaudeBalancedRouter, "install_cooldown", recording_install
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert install_evidence == [""]
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        assert row.evidence == ""
        [line] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        record = json.loads(line)
        assert record["installed_scope"] == "account"
        assert record["family_gate"]["reason"] == "fable_weekly_not_saturated"
        assert record["scope_rationale"] == "fable_weekly_not_saturated"
        assert record["record_degraded"] is True
        assert record["degradation_reason"] == "evidence_enrichment_failed"
        assert record["session_fingerprint"] is None


def test_balanced_429_finalization_failure_installs_empty_evidence_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    response_body = b'{"error":{"type":"rate_limit_error","message":"quota reached"}}'
    install_calls: list[dict[str, Any]] = []
    original_install_callback = relay_balanced._install_balanced_quota_cooldown
    original_install = ClaudeBalancedRouter.install_cooldown

    async def inject_finalization_failure(*args: Any, **kwargs: Any) -> str:
        kwargs["mark"].record["unknown_nonserializable_extra"] = object()
        return await original_install_callback(*args, **kwargs)

    def recording_install(
        self: ClaudeBalancedRouter, **kwargs: Any
    ) -> Any:
        install_calls.append(dict(kwargs))
        return original_install(self, **kwargs)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=response_body,
            headers={
                "request-id": "upstream-request-id",
                "retry-after": "17",
                "x-should-retry": "true",
            },
        )

    monkeypatch.setattr(
        relay_balanced,
        "_install_balanced_quota_cooldown",
        inject_finalization_failure,
    )
    monkeypatch.setattr(
        ClaudeBalancedRouter, "install_cooldown", recording_install
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert response.content == response_body
        assert response.headers["request-id"] == "upstream-request-id"
        assert response.headers["retry-after"] == "17"
        assert response.headers["x-should-retry"] == "true"
        assert len(install_calls) == 1
        assert install_calls[0]["evidence"] == ""
        row = runtime._store.get_cooldown(account_id, "account", "")
        assert row is not None
        assert row.evidence == ""
        [canonical] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        incident = json.loads(canonical)
        assert {
            "v",
            "event",
            "occurred_at_utc",
            "mode",
            "account_id",
            "model",
            "cooldown_seconds",
            "cooldown_source",
            "installed_scope",
            "quota_family",
            "family_gate",
            "observed_scope",
            "scope_rationale",
            "upstream",
            "attempt",
            "session_fingerprint",
            "request_shape",
            "record_degraded",
            "degradation_reason",
        } <= incident.keys()
        assert incident["installed_scope"] == "account"
        assert incident["quota_family"] == "fable"
        assert incident["family_gate"]["reason"] == (
            "fable_weekly_not_saturated"
        )
        assert incident["observed_scope"] == "unknown"
        assert incident["scope_rationale"] == "fable_weekly_not_saturated"
        assert incident["session_fingerprint"] is None
        assert incident["record_degraded"] is True
        assert incident["degradation_reason"] == "evidence_enrichment_failed"
        assert "unknown_nonserializable_extra" not in incident
        assert set(incident) != {"record_degraded", "degradation_reason"}
        assert canonical == json.dumps(
            incident,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


def test_429_incident_missing_context_records_null_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer pool-access-1":
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(
        relay_registered, "try_begin_account_leg", lambda _tracker, _pin: None
    )
    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(first)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        response = client.post("/v1/messages", json=_account_leg_body())

        assert response.status_code == 200
        [line] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        record = json.loads(line)
        assert record["attempt"] is None
        assert record["request_shape"]["pin_created"] is None
        assert record["record_degraded"] is True
        assert record["degradation_reason"] == "attempt_context_unavailable"


def test_429_incident_session_extraction_failure_uses_conservative_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)

    def fail_session_extraction(_body):
        raise ValueError("session extraction unavailable")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["authorization"] == "Bearer pool-access-1":
            return httpx.Response(
                429,
                json={
                    "request_id": _LEG_RAW_SESSION_UUID,
                    "error": {"message": _LEG_CANONICAL_SESSION_UUID},
                },
                headers={"request-id": _LEG_RAW_SESSION_UUID},
            )
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(
        relay_registered, "extract_session_uuid", fail_session_extraction
    )
    with _create_test_client(
        monkeypatch, tmp_path, config=_pool_config(first)
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )

        response = client.post("/v1/messages", json=_account_leg_body())

        assert response.status_code == 200
        [line] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        record = json.loads(line)
        assert record["model"] is None
        assert record["attempt"] is None
        assert record["upstream"]["request_id"] is None
        assert record["upstream"]["message"] is None
        assert record["upstream"]["headers"] == {}
        assert record["degradation_reason"] == (
            "attempt_and_session_context_unavailable"
        )
        assert _LEG_RAW_SESSION_UUID not in line
        assert _LEG_CANONICAL_SESSION_UUID not in line


def test_balanced_429_callback_failure_replays_and_attempts_degraded_incident(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    async def fail_callback(*_args, **_kwargs) -> str:
        raise RuntimeError("simulated callback entry failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            content=b'{"error":{"type":"rate_limit_error"}}',
            headers={"request-id": "upstream-request-id"},
        )

    monkeypatch.setattr(
        relay_balanced, "_install_balanced_quota_cooldown", fail_callback
    )
    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        assert response.content == b'{"error":{"type":"rate_limit_error"}}'
        assert response.headers["request-id"] == "upstream-request-id"
        assert client.app.state.claude_account_cooldowns.is_cooling(account_id)
        assert runtime._store.get_cooldown(account_id, "account", "") is None
        [line] = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        ).read_text().splitlines()
        record = json.loads(line)
        assert record["record_degraded"] is True
        assert record["degradation_reason"] == "evidence_enrichment_failed"
        envelopes = _account_leg_envelopes(caplog)
        assert len(envelopes) == 1
        assert envelopes[0]["result"] == "rate_limited"


def test_one_mark_one_incident_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429()

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 429
        incident_path = (
            paths.claude_account_pool_dir() / "claude-429-incidents.jsonl"
        )
        lines = incident_path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "claude_quota_429"


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


def test_pinned_touch_schedules_durable_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)
    planned_digests: list[bytes] = []
    scheduled_refreshes: list[tuple[bytes, float, float]] = []

    def plan_pin_durable_last_seen(
        _router: ClaudeBalancedRouter, digest: bytes
    ) -> tuple[float, float]:
        planned_digests.append(digest)
        return 1_000_000.0, 1_000_120.0

    async def refresh_pin_durable_last_seen(
        _router: ClaudeBalancedRouter,
        digest: bytes,
        *,
        last_seen_utc: float,
        expires_at_utc: float,
    ) -> None:
        scheduled_refreshes.append((digest, last_seen_utc, expires_at_utc))

    monkeypatch.setattr(
        ClaudeBalancedRouter,
        "plan_pin_durable_last_seen",
        plan_pin_durable_last_seen,
    )
    monkeypatch.setattr(
        ClaudeBalancedRouter,
        "refresh_pin_durable_last_seen",
        refresh_pin_durable_last_seen,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        runtime = _enable_balanced(client, handler)
        body = _balanced_body(_new_session_id())

        response = client.post("/v1/messages", json=body)

        session_key = derive_session_key(body, runtime.epoch_seed, "fable")
        assert session_key is not None
        assert response.status_code == 200
        assert planned_digests == [session_key.digest]
        assert scheduled_refreshes == [
            (session_key.digest, 1_000_000.0, 1_000_120.0)
        ]
        assert client.app.state.claude_pin_refresh_tasks == set()


def test_pinned_durable_refresh_planning_failure_preserves_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    def fail_plan(_router: ClaudeBalancedRouter, _digest: bytes):
        raise OSError("simulated planning failure")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(
        ClaudeBalancedRouter, "plan_pin_durable_last_seen", fail_plan
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 200
        assert response.json() == {"id": "msg_1"}
        assert client.app.state.claude_pin_refresh_tasks == set()


def test_pinned_durable_refresh_scheduling_failure_preserves_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)
    real_create_task = asyncio.create_task

    def fail_refresh_task(coroutine, *args, **kwargs):
        if coroutine.cr_code.co_name == "_refresh_durable_pin":
            raise RuntimeError("simulated scheduling failure")
        return real_create_task(coroutine, *args, **kwargs)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(relay_balanced.asyncio, "create_task", fail_refresh_task)
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )

        assert response.status_code == 200
        assert response.json() == {"id": "msg_1"}
        assert client.app.state.claude_pin_refresh_tasks == set()


def test_pinned_durable_refresh_exception_warning_is_constant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    async def fail_refresh(
        _router: ClaudeBalancedRouter,
        _digest: bytes,
        *,
        last_seen_utc: float,
        expires_at_utc: float,
    ) -> None:
        raise RuntimeError(
            f"secret path /private/{last_seen_utc}/{expires_at_utc}"
        )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(
        ClaudeBalancedRouter, "refresh_pin_durable_last_seen", fail_refresh
    )
    caplog.set_level(logging.WARNING, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)

        response = client.post(
            "/v1/messages", json=_balanced_body(_new_session_id())
        )
        assert client.portal is not None
        client.portal.call(asyncio.sleep, 0)

        assert response.status_code == 200
        warnings = [
            record
            for record in caplog.records
            if record.getMessage()
            == "balanced durable pin refresh failed unexpectedly"
        ]
        assert len(warnings) == 1
        assert warnings[0].exc_info is None
        assert "secret path" not in caplog.text
        assert client.app.state.claude_pin_refresh_tasks == set()


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

    monkeypatch.setattr(relay_balanced.ClaudeBalancedRouter, "place_session", spy_place_session)

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
    original_pick_account = relay_balanced._balanced_pick_account

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
        monkeypatch.setattr(relay_balanced, "_balanced_pick_account", recording_pick_account)
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


def test_chain_exhausted_real_429_without_retry_after_gets_local_retry_after(
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
        assert int(response.headers["retry-after"]) >= 1
        assert response.headers["x-should-retry"] == "true"
        assert client.app.state.claude_account_cooldowns.is_cooling(account_id)


def test_chain_exhausted_retry_after_matches_earliest_local_unblock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(relay_balanced.time, "monotonic", lambda: 1_000.0)
    records = {
        "account-a": SimpleNamespace(id="account-a", state="ready"),
        "account-b": SimpleNamespace(id="account-b", state="ready"),
    }
    account_offsets = {"account-a": 9.25, "account-b": 1.0}
    family_offsets = {"account-a": 2.0, "account-b": 5.25}
    router = SimpleNamespace(
        account_cooldown_deadline=lambda account_id, *, now: now
        + account_offsets[account_id],
        family_cooldown_deadline=lambda account_id, family, *, now: now
        + family_offsets[account_id],
    )
    upstream = Response(content=b"upstream", status_code=429)

    returned = relay_balanced._balanced_all_cooling_response(
        records, router, family="fable", chain_exhausted_429=upstream
    )

    assert returned is upstream
    assert returned.headers["retry-after"] == "6"


@pytest.mark.parametrize("retry_after", ["", "not-a-delay"])
def test_chain_exhausted_real_429_preserves_upstream_retry_after(
    retry_after: str,
) -> None:
    records = {"account-a": SimpleNamespace(id="account-a", state="ready")}
    router = SimpleNamespace(
        account_cooldown_deadline=lambda account_id, *, now: now + 10.0,
        family_cooldown_deadline=lambda account_id, family, *, now: now + 20.0,
    )
    upstream = Response(
        content=b"upstream",
        status_code=429,
        headers={"ReTrY-AfTeR": retry_after, "request-id": "req_preserved"},
    )
    before_headers = list(upstream.raw_headers)

    returned = relay_balanced._balanced_all_cooling_response(
        records, router, family="fable", chain_exhausted_429=upstream
    )

    assert returned is upstream
    assert returned.headers["retry-after"] == retry_after
    assert list(returned.raw_headers) == before_headers


def test_chain_exhausted_real_429_preserves_body_and_other_headers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    api_calls: list[httpx.Request] = []
    payload = b"\x00upstream-body\xff"

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
        api_calls.append(request)
        if len(api_calls) <= 2:
            return httpx.Response(401, json={"type": "error"})
        return httpx.Response(
            429,
            content=payload,
            headers={
                "content-type": "application/octet-stream",
                "request-id": "req_terminal_429",
                "x-should-retry": "true",
            },
        )

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, handler)

        response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))

        assert response.status_code == 429
        assert response.content == payload
        assert response.headers["retry-after"] == "1"
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["request-id"] == "req_terminal_429"
        assert response.headers["x-should-retry"] == "true"
        assert len(api_calls) == 3


def test_synthetic_all_cooling_response_unchanged(
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
        assert second.json() == {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "every eligible claude account is rate-limited; "
                "retry after the cooldown",
            },
        }
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

    monkeypatch.setattr(relay_balanced.ClaudeBalancedRouter, "commit_at_headers", spy_commit_at_headers)

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
    original_derive_stateless_routing_digest = relay_balanced.derive_stateless_routing_digest

    def spy_derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
        digest = original_derive_stateless_routing_digest(seed, nonce)
        digest_calls.append(digest)
        return digest

    monkeypatch.setattr(
        relay_balanced,
        "derive_stateless_routing_digest",
        spy_derive_stateless_routing_digest,
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
    original_derive_stateless_routing_digest = relay_balanced.derive_stateless_routing_digest

    def spy_derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
        digest = original_derive_stateless_routing_digest(seed, nonce)
        digest_calls.append(digest)
        return digest

    monkeypatch.setattr(
        relay_balanced,
        "derive_stateless_routing_digest",
        spy_derive_stateless_routing_digest,
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

        monkeypatch.setattr(relay_balanced, "_balanced_pick_account", fail_fallback_pick)
        assert relay_balanced._balanced_pick_account is fail_fallback_pick
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
    original_pick_account = relay_balanced._balanced_pick_account

    def recording_pick_account(*args: Any, **kwargs: Any) -> str:
        recorded_digests.append(kwargs["session_key_digest"])
        return original_pick_account(*args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"input_tokens": 7})

    with _create_test_client(
        monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        monkeypatch.setattr(relay_balanced, "_balanced_pick_account", recording_pick_account)
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
        from claudex.claude.account_profile import load_account_profile_fingerprint

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


def test_fallback_attempts_emit_leg_logs_with_ordinals_and_gaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first, _second = _register_pool_accounts(monkeypatch)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(_pool_config(first), handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 200
    envelopes = _account_leg_envelopes(caplog)
    assert [envelope["mode"] for envelope in envelopes] == ["fallback", "fallback"]
    assert [envelope["result"] for envelope in envelopes] == [
        "rate_limited",
        "success",
    ]
    assert [envelope["attempt"]["ordinal"] for envelope in envelopes] == [1, 2]
    assert envelopes[0]["attempt"] == {
        "ordinal": 1,
        "elapsed_ms_since_first": 0,
        "gap_ms_since_previous": None,
    }
    assert envelopes[1]["attempt"]["gap_ms_since_previous"] >= 0
    assert (
        envelopes[1]["attempt"]["elapsed_ms_since_first"]
        == envelopes[1]["attempt"]["gap_ms_since_previous"]
    )
    assert all(
        envelope["request_shape"]["pin_created"] is None
        for envelope in envelopes
    )


@pytest.mark.parametrize("fallback", [False, True])
def test_observability_session_extraction_failure_does_not_block_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fallback: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    if fallback:
        account_id, _second = _register_pool_accounts(monkeypatch)
        config = _pool_config(account_id)
    else:
        account_id = _register_serving_account()
        config = GatewayConfig(claude_account_id=account_id)
    upstream_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        return httpx.Response(200, json={"id": "msg_1"})

    def fail_session_extraction(_body):
        raise ValueError("session observability unavailable")

    monkeypatch.setattr(relay_registered, "extract_session_uuid", fail_session_extraction)
    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(config, handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 200
    assert upstream_calls == 1
    assert _account_leg_envelopes(caplog) == []


def test_observability_clock_failure_preserves_upstream_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    upstream_calls = 0

    class RelaySignal(BaseException):
        pass

    signal = RelaySignal("unexpected relay signal")

    def fail_clock(_tracker, _pin_created):
        raise OSError("clock unavailable")

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        raise signal

    monkeypatch.setattr(relay_registered.AccountLegTracker, "begin_leg", fail_clock)
    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    with pytest.raises(RelaySignal) as raised:
        client.post("/v1/messages", json=_account_leg_body())

    assert raised.value is signal
    assert upstream_calls == 1
    assert _account_leg_envelopes(caplog) == []


def test_401_refresh_resend_stays_single_leg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()
    api_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal api_calls
        if str(request.url) == CLAUDE_TOKEN_URL:
            return httpx.Response(
                200,
                json={
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 900,
                },
            )
        api_calls += 1
        if api_calls == 1:
            return httpx.Response(401, json={"type": "error"})
        return httpx.Response(200, json={"id": "msg_1"})

    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 200
    assert api_calls == 2
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["mode"] == "disabled"
    assert envelopes[0]["result"] == "success"
    assert envelopes[0]["attempt"]["ordinal"] == 1


def test_credential_failure_before_send_still_emits_leg_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account(expires_in_seconds=-60)
    upstream_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream_paths.append(request.url.path)
        assert str(request.url) == CLAUDE_TOKEN_URL
        return httpx.Response(400, json={"error": "invalid_grant"})

    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 503
    assert upstream_paths == ["/v1/oauth/token"]
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["mode"] == "disabled"
    assert envelopes[0]["result"] == "failed"
    assert envelopes[0]["attempt"]["ordinal"] == 1


def test_count_token_attempts_emit_leg_logs_with_null_pin_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 7})
        return httpx.Response(200, json={"id": "msg_1"})

    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)
        pinned_body = _account_leg_body()
        assert client.post("/v1/messages", json=pinned_body).status_code == 200
        caplog.clear()

        pinned_count = client.post(
            "/v1/messages/count_tokens", json=pinned_body
        )
        unpinned_session = _new_session_id()
        unpinned_count = client.post(
            "/v1/messages/count_tokens",
            json=_account_leg_body(unpinned_session),
        )

    assert pinned_count.status_code == 200
    assert unpinned_count.status_code == 200
    envelopes = _account_leg_envelopes(
        caplog, extra_session_literals=(unpinned_session,)
    )
    assert len(envelopes) == 2
    assert all(
        envelope["mode"] == "balanced_count_tokens" for envelope in envelopes
    )
    assert all(envelope["result"] == "success" for envelope in envelopes)
    assert all(envelope["attempt"]["ordinal"] == 1 for envelope in envelopes)
    assert all(
        envelope["request_shape"]["pin_created"] is None
        for envelope in envelopes
    )


def test_count_token_429_emits_rate_limited_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(1)

    def handler(_request: httpx.Request) -> httpx.Response:
        return _quota_429("count-token-rate-limit")

    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)
        response = client.post(
            "/v1/messages/count_tokens", json=_account_leg_body()
        )

    assert response.status_code == 429
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["mode"] == "balanced_count_tokens"
    assert envelopes[0]["result"] == "rate_limited"
    assert envelopes[0]["request_shape"]["pin_created"] is None


def test_pinned_initial_leg_pin_created_true_migration_leg_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)
        response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 200
    envelopes = _account_leg_envelopes(caplog)
    assert [envelope["mode"] for envelope in envelopes] == [
        "balanced_pinned",
        "balanced_pinned",
    ]
    assert [envelope["result"] for envelope in envelopes] == [
        "rate_limited",
        "success",
    ]
    assert [
        envelope["request_shape"]["pin_created"] for envelope in envelopes
    ] == [True, False]
    assert [envelope["attempt"]["ordinal"] for envelope in envelopes] == [1, 2]


def test_stateless_attempts_emit_leg_logs_with_null_pin_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_accounts(2)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _quota_429()
        return httpx.Response(200, json={"id": "msg_1"})

    stateless_body = {
        "model": "claude-fable-5",
        "max_tokens": 16,
        "messages": [
            {"role": "assistant", "content": _LEG_PROMPT_CONTENT}
        ],
        "tools": [
            {
                "name": "fixture",
                "description": _LEG_TOOL_ARGUMENT,
                "input_schema": {"type": "object"},
            }
        ],
    }
    caplog.set_level(logging.INFO, logger="claudex.server")
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=GatewayConfig(),
        base_url="http://127.0.0.1:8787",
    ) as client:
        _enable_balanced(client, handler)
        response = client.post("/v1/messages", json=stateless_body)

    assert response.status_code == 200
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 2
    assert all(envelope["mode"] == "balanced_stateless" for envelope in envelopes)
    assert [envelope["result"] for envelope in envelopes] == [
        "rate_limited",
        "success",
    ]
    assert all(
        envelope["request_shape"]["pin_created"] is None
        for envelope in envelopes
    )


def test_non_2xx_terminal_attempt_emits_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "upstream failed"}})

    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 500
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["mode"] == "disabled"
    assert envelopes[0]["result"] == "failed"


def test_unexpected_exception_emits_exception_result_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    class RelaySignal(BaseException):
        pass

    signal = RelaySignal("unexpected relay signal")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise signal

    caplog.set_level(logging.INFO, logger="claudex.server")
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    with pytest.raises(RelaySignal) as raised:
        client.post("/v1/messages", json=_account_leg_body())

    assert raised.value is signal
    envelopes = _account_leg_envelopes(caplog)
    assert len(envelopes) == 1
    assert envelopes[0]["mode"] == "disabled"
    assert envelopes[0]["result"] == "exception"


def test_leg_log_emission_failure_preserves_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    monkeypatch.setattr(relay_registered, "logger", _RaisingLegLogger())
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    response = client.post("/v1/messages", json=_account_leg_body())

    assert response.status_code == 200
    assert response.json() == {"id": "msg_1"}


def test_leg_log_emission_failure_preserves_escaping_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    account_id = _register_serving_account()

    class RelaySignal(BaseException):
        pass

    signal = RelaySignal("unexpected relay signal")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise signal

    monkeypatch.setattr(relay_registered, "logger", _RaisingLegLogger())
    client, _ = _gateway(GatewayConfig(claude_account_id=account_id), handler)

    with pytest.raises(RelaySignal) as raised:
        client.post("/v1/messages", json=_account_leg_body())

    assert raised.value is signal
