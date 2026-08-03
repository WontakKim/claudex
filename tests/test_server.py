"""Integration tests for the gateway HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
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
from claudex_gateway.codex_client import CodexClient, CodexUpstreamError
from claudex_gateway.config import GatewayConfig
from claudex_gateway.kimi_auth import KimiCredentials
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.xai_auth import XAICredentials
from claudex_gateway.xai_client import XAIClient, XAIUpstreamError


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


class AvailableXAIAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> XAICredentials:
        return XAICredentials(access_token="xai-token", email="user@example.com")


class MissingXAIAuthManager(AvailableXAIAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> XAICredentials:
        raise server.XAIAuthError("no xAI credentials; run `grok login` first")


class FakeXAIClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _create_test_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: GatewayConfig | None = None,
    codex_auth: type = AvailableCodexAuthManager,
    codex_client: type = FakeCodexClient,
    kimi_auth: type = AvailableKimiAuthManager,
    kimi_client: type = FakeKimiClient,
    xai_auth: type = AvailableXAIAuthManager,
    xai_client: type = FakeXAIClient,
    base_url: str = "http://testserver",
) -> TestClient:
    monkeypatch.setattr(server, "CodexAuthManager", codex_auth)
    monkeypatch.setattr(server, "CodexClient", codex_client)
    monkeypatch.setattr(server, "KimiAuthManager", kimi_auth)
    monkeypatch.setattr(server, "KimiClient", kimi_client)
    monkeypatch.setattr(server, "XAIAuthManager", xai_auth)
    monkeypatch.setattr(server, "XAIClient", xai_client)
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
            "xai": {
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


def test_health_stays_ok_without_xai_credentials_when_map_has_no_xai_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _create_test_client(monkeypatch, xai_auth=MissingXAIAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["xai"]["status"] == "error"
    assert health.json()["providers"]["xai"]["required"] is False


def test_health_reports_error_without_xai_credentials_when_map_routes_to_xai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GatewayConfig(model_map={"opus": "xai:grok-4.5"})
    with _create_test_client(
        monkeypatch, config=config, xai_auth=MissingXAIAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["xai"]["status"] == "error"
    assert health.json()["providers"]["xai"]["required"] is True


def _upstream_error(status_code: int, error: dict) -> CodexUpstreamError:
    return CodexUpstreamError(status_code, json.dumps({"error": error}))


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

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        self.payloads.append(payload)
        raise CodexUpstreamError(503, "stub codex upstream")
        yield  # unreachable; makes this an async generator like the real client


def _gateway(
    config: GatewayConfig,
    anthropic_handler,
    kimi_handler=None,
    kimi_auth: Any | None = None,
    xai_client: Any | None = None,
) -> tuple[TestClient, StubCodexClient]:
    app = server.create_app(config)
    # The lifespan requires real Codex credentials, so set the state directly
    # instead of entering the TestClient context manager.
    app.state.config = config
    stub = StubCodexClient()
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
    if xai_client is not None:
        app.state.xai_client = xai_client
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


# --- /v1/messages routing: xai-mapped models go to the xAI Responses backend ---


class StubXAIClient:
    """Records translated payloads and replays a scripted Responses stream."""

    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.events = events

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        self.payloads.append(payload)
        if self.events is None:
            raise XAIUpstreamError(503, "stub xai upstream")
        for event in self.events:
            yield event


def _xai_config(target: str = "grok-4.5", **kwargs: Any) -> GatewayConfig:
    return GatewayConfig(model_map={"opus": f"xai:{target}"}, **kwargs)


def test_xai_mapped_model_streams_translated_response() -> None:
    stub = StubXAIClient(
        [
            {"type": "response.created", "response": {"id": "resp_1", "model": "grok-4.5"}},
            {"type": "response.output_text.delta", "delta": "hello"},
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "usage": {"input_tokens": 1, "output_tokens": 1}},
            },
        ]
    )
    client, codex_stub = _gateway(_xai_config(), _failing_anthropic_handler, xai_client=stub)

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 200
    # The gateway reports the requested Claude model, not the xAI target.
    assert response.json()["model"] == "claude-opus-4-6"
    assert codex_stub.payloads == []
    (payload,) = stub.payloads
    assert payload["model"] == "grok-4.5"
    # No thinking block in the request: the derived medium survives clamping.
    assert payload["reasoning"]["effort"] == "medium"


def test_xai_route_strips_reasoning_for_non_thinking_model() -> None:
    stub = StubXAIClient()
    client, _ = _gateway(
        _xai_config("grok-composer-2.5-fast"), _failing_anthropic_handler, xai_client=stub
    )

    response = client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    assert response.status_code == 503
    (payload,) = stub.payloads
    assert "reasoning" not in payload


def test_xai_route_clamps_effort_for_thinking_model() -> None:
    stub = StubXAIClient()
    config = _xai_config("grok-4.5", reasoning_effort_override="max")
    client, _ = _gateway(config, _failing_anthropic_handler, xai_client=stub)

    client.post("/v1/messages", json=_message_body("claude-opus-4-6"))

    (payload,) = stub.payloads
    assert payload["reasoning"]["effort"] == "high"


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

    async def fake_xai(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        return {"provider": "xai", "status": "ok", "error": None}

    monkeypatch.setattr(server, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(server, "fetch_codex_usage", fake_codex)
    monkeypatch.setattr(server, "fetch_kimi_usage", fake_kimi)
    monkeypatch.setattr(server, "fetch_xai_usage", fake_xai)
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert body["codex"]["status"] == "unavailable"
    assert body["kimi"]["status"] == "ok"
    assert body["xai"]["status"] == "ok"
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

    async def xai_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("xai probe ran for ?provider=claude")

    monkeypatch.setattr(server, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(server, "fetch_codex_usage", codex_must_not_run)
    monkeypatch.setattr(server, "fetch_kimi_usage", kimi_must_not_run)
    monkeypatch.setattr(server, "fetch_xai_usage", xai_must_not_run)
    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "claude"})

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert "codex" not in body
    assert "kimi" not in body
    assert "xai" not in body


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
    # Usage renders inside the General tab's provider cards: a per-provider
    # body hook, the fetch on entering General, and no separate tab.
    with _create_test_client(monkeypatch) as client:
        page = client.get("/").text

    assert 'data-t="usage"' not in page
    assert 'id="tab-usage"' not in page
    assert 'id="usage-body-claude"' in page
    assert 'id="usage-body-codex"' in page
    assert 'id="usage-body-kimi"' in page
    assert "/admin/usage" in page
    # The usage hooks sit inside the General section, not a sibling tab.
    assert page.index('id="tab-general"') < page.index('id="usage-body-codex"')
    assert page.index('id="usage-body-claude"') < page.index('id="tab-log"')
    # Claude leads the provider cards as a Status card like the others.
    assert "<h2>Claude Status" in page
    assert "<h2>Claude Usage" not in page
    assert page.index('id="usage-body-claude"') < page.index('id="codex-stat"')


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
    assert "CATALOG={codex:[],kimi:[],xai:[]}" in page


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
    # and xAI endpoints too — not just the Codex one.
    assert '"/admin/kimi/models"' in page
    assert '"/admin/xai/models"' in page


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


class ProbeXAIClient(FakeXAIClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class CatalogXAIClient(FakeXAIClient):
    async def list_models(self) -> list[str]:
        return ["grok-4.5", "grok-4.3"]


class FailingCatalogXAIClient(FakeXAIClient):
    async def list_models(self) -> list[str]:
        raise XAIUpstreamError(401, '{"error":{"message":"token expired"}}')


class RejectingXAIClient(FakeXAIClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise XAIUpstreamError(400, '{"error":{"message":"model_not_found"}}')


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

    def test_xai_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, xai_client=CatalogXAIClient) as client:
            response = client.get("/admin/xai/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["grok-4.5", "grok-4.3"]}

    def test_xai_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, xai_client=FailingCatalogXAIClient) as client:
            response = client.get("/admin/xai/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_xai_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with _create_test_client(monkeypatch, xai_client=CatalogXAIClient) as client:
            assert client.get("/admin/xai/models").status_code == 403

    def test_connection_test_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with self._client(monkeypatch, codex_client=ProbeCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"source": "haiku", "target": "codex:gpt-5.6-luna"},
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
                json={"source": "haiku", "target": "gpt-5.6-luna"},
            )

        assert response.status_code == 400
        assert "no provider prefix" in response.json()["error"]["message"]

    def test_connection_test_kimi_target_probes_kimi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, kimi_client=ProbeKimiClient) as client:
            response = client.post(
                "/admin/test",
                json={"source": "fable", "target": "kimi:k3"},
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
                json={"source": "fable", "target": "kimi:k3"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 404
        assert "model not found" in result["detail"]

    def test_connection_test_xai_target_probes_xai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, xai_client=ProbeXAIClient) as client:
            response = client.post(
                "/admin/test",
                json={"source": "fable", "target": "xai:grok-4.5"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "grok-4.5"

    def test_connection_test_reports_xai_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, xai_client=RejectingXAIClient) as client:
            response = client.post(
                "/admin/test",
                json={"source": "fable", "target": "xai:grok-nope"},
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
                json={"source": "fable", "target": "kim:k3"},
            )

        assert response.status_code == 400
        assert "unknown provider prefix" in response.json()["error"]["message"]

    def test_connection_test_reports_unknown_codex_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with self._client(monkeypatch, codex_client=RejectingCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"source": "haiku", "target": "codex:gpt-nope"},
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
                json={"source": "a", "target": " "},
            )
            wrong_content_type = client.post(
                "/admin/test",
                content='{"source": "a", "target": "b"}',
                headers={"content-type": "text/plain"},
            )

        assert empty_target.status_code == 400
        assert wrong_content_type.status_code == 415
