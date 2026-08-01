"""Tests for the Kimi relay client's auth overlay and retry behavior."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from claudex_gateway.kimi_auth import KimiCredentials
from claudex_gateway.kimi_client import (
    KIMI_COUNT_TOKENS_URL,
    KIMI_MESSAGES_URL,
    KIMI_MODELS_URL,
    KimiClient,
    KimiUpstreamError,
)


class _FakeAuthManager:
    """Hands out token-1, token-2, ... and records force_refresh flags."""

    def __init__(self) -> None:
        self.calls: list[bool] = []

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        self.calls.append(force_refresh)
        return KimiCredentials(access_token=f"token-{len(self.calls)}", device_id="device-1")


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_send_messages_adds_beta_param_and_bearer_over_forwarded_headers() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        seen["anthropic-beta"] = request.headers["anthropic-beta"]
        seen["body"] = request.content
        return httpx.Response(200, content=b'{"type":"message"}')

    async def scenario() -> bytes:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = KimiClient(_FakeAuthManager(), http_client)
            response = await client.send_messages(
                b'{"model":"k2.5"}', {"anthropic-beta": "interleaved-thinking-2025-05-14"}
            )
            try:
                return await response.aread()
            finally:
                await response.aclose()

    body = _run(scenario())

    assert seen["url"] == f"{KIMI_MESSAGES_URL}?beta=true"
    assert seen["authorization"] == "Bearer token-1"
    assert seen["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert seen["body"] == b'{"model":"k2.5"}'
    assert body == b'{"type":"message"}'


def test_401_forces_refresh_and_retries_once() -> None:
    auth_manager = _FakeAuthManager()
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer token-1":
            return httpx.Response(401, content=b'{"error":"unauthorized"}')
        return httpx.Response(200, content=b"{}")

    async def scenario() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            response = await KimiClient(auth_manager, http_client).send_messages(b"{}", {})
            await response.aclose()
            return response.status_code

    assert _run(scenario()) == 200
    assert auth_manager.calls == [False, True]
    assert attempts == ["Bearer token-1", "Bearer token-2"]


def test_persistent_401_raises_after_single_retry() -> None:
    auth_manager = _FakeAuthManager()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b'{"error":"unauthorized"}')

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiClient(auth_manager, http_client).send_messages(b"{}", {})

    with pytest.raises(KimiUpstreamError) as excinfo:
        _run(scenario())

    assert excinfo.value.status_code == 401
    assert auth_manager.calls == [False, True]


def test_non_200_raises_with_body() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=b'{"type":"error"}')

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiClient(_FakeAuthManager(), http_client).send_messages(b"{}", {})

    with pytest.raises(KimiUpstreamError) as excinfo:
        _run(scenario())

    assert excinfo.value.status_code == 429
    assert excinfo.value.body == '{"type":"error"}'


def test_list_models_relays_catalog_verbatim() -> None:
    # The catalog is the backend's own answer — no reshaping, so newly
    # released models appear without a gateway update.
    catalog = {"data": [{"id": "k2.5"}, {"id": "k3"}]}
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        return httpx.Response(200, json=catalog)

    async def scenario() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await KimiClient(_FakeAuthManager(), http_client).list_models()

    assert _run(scenario()) == catalog
    assert seen["url"] == KIMI_MODELS_URL
    assert seen["authorization"] == "Bearer token-1"


def test_list_models_retries_once_on_401() -> None:
    auth_manager = _FakeAuthManager()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] == "Bearer token-1":
            return httpx.Response(401, content=b'{"error":"unauthorized"}')
        return httpx.Response(200, json={"data": []})

    async def scenario() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await KimiClient(auth_manager, http_client).list_models()

    assert _run(scenario()) == {"data": []}
    assert auth_manager.calls == [False, True]


def test_count_tokens_targets_count_endpoint() -> None:
    seen: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b'{"input_tokens":12}')

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            response = await KimiClient(_FakeAuthManager(), http_client).count_tokens(b"{}", {})
            await response.aclose()

    _run(scenario())

    assert seen["url"] == f"{KIMI_COUNT_TOKENS_URL}?beta=true"
