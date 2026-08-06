"""Tests for the Codex client's model catalog and context-window lookup."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from claudex_gateway.codex_auth import CodexAuthError, CodexCredentials
from claudex_gateway.codex_client import CODEX_MODELS_URL, CodexClient, CodexUpstreamError
from claudex_gateway.context_window_cache import ContextWindowCache


class _FakeAuthManager:
    def __init__(self) -> None:
        self.calls = 0

    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        self.calls += 1
        return CodexCredentials(access_token="codex-token-1", account_id="account-1")


_CATALOG_MODELS: list[dict[str, Any]] = [
    {"slug": "gpt-5.6-sol", "context_window": 272000},
    {"slug": "gpt-5.3-codex-spark", "context_window": 128000},
    {"slug": "gpt-5.4", "context_window": 272000, "max_context_window": 1000000},
    {"slug": "gpt-5.6-hidden", "context_window": 64000, "visibility": "hide"},
    {"slug": "no-window-field"},
    {"slug": "string-window", "context_window": "272000"},
    {"slug": "bool-window", "context_window": True},
    {"slug": "zero-window", "context_window": 0},
    {"slug": "negative-window", "context_window": -1},
    {"slug": "fractional-window", "context_window": 272000.5},
    {"slug": "integral-float-window", "context_window": 272000.0},
]


def _catalog_handler(calls: dict[str, int]) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert str(request.url).startswith(CODEX_MODELS_URL)
        assert request.url.params["client_version"]
        assert request.headers["accept"] == "application/json"
        return httpx.Response(200, json={"models": _CATALOG_MODELS})

    return handler


class _FakeClock:
    """A controllable stand-in for `time.monotonic`, advanced explicitly.

    CodexClient exposes no public clock-injection parameter, so forcing the
    cache's 900s TTL to expire without a real sleep requires replacing the
    client's private `_context_windows` cache with one built from this fake
    clock (see `_codex_client_with_fake_clock` below).
    """

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _codex_client_with_fake_clock(http_client: httpx.AsyncClient, clock: _FakeClock) -> CodexClient:
    client = CodexClient(_FakeAuthManager(), http_client)
    client._context_windows = ContextWindowCache(
        client._fetch_context_windows,
        expected_errors=(CodexAuthError, CodexUpstreamError, httpx.HTTPError),
        clock=clock,
    )
    return client


def _malformed_catalog_response(kind: str) -> httpx.Response:
    """Build a response for each structural catalog-failure variant."""
    if kind == "non_200":
        return httpx.Response(500, text="boom")
    if kind == "invalid_json":
        return httpx.Response(200, content=b"not valid json{")
    if kind == "missing_models_key":
        return httpx.Response(200, json={"unexpected": []})
    if kind == "non_list_models":
        return httpx.Response(200, json={"models": {"not": "a-list"}})
    raise ValueError(f"unknown malformed-catalog kind: {kind}")


def test_context_window_returns_window_for_exact_slug_match() -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("gpt-5.3-codex-spark")

    assert asyncio.run(scenario()) == 128000
    assert calls["n"] == 1


def test_hidden_model_excluded_from_list_but_resolvable_via_context_window() -> None:
    calls = {"n": 0}

    async def scenario() -> tuple[list[str], int | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            models = await client.list_models()
            window = await client.context_window("gpt-5.6-hidden")
            return models, window

    models, window = asyncio.run(scenario())

    assert "gpt-5.6-hidden" not in models
    assert window == 64000


def test_max_context_window_sibling_field_is_ignored() -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("gpt-5.4")

    assert asyncio.run(scenario()) == 272000


@pytest.mark.parametrize(
    "slug",
    ["no-window-field", "string-window", "bool-window", "zero-window", "negative-window", "fractional-window"],
)
def test_invalid_context_window_values_resolve_to_none(slug: str) -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window(slug)

    assert asyncio.run(scenario()) is None


def test_positive_integral_float_window_coerced_to_int() -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("integral-float-window")

    result = asyncio.run(scenario())
    assert result == 272000
    assert isinstance(result, int)


def test_unknown_slug_returns_none() -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("does-not-exist")

    assert asyncio.run(scenario()) is None


def test_structural_failure_after_success_serves_stale_value() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"models": _CATALOG_MODELS})
        return httpx.Response(500, text="boom")

    async def scenario() -> tuple[int | None, int | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            first = await client.context_window("gpt-5.6-sol")
            # CodexClient doesn't expose a clock hook, so force the cache's
            # 900s TTL to be considered expired without a real sleep.
            client._context_windows._snapshot_time = 0.0
            second = await client.context_window("gpt-5.6-sol")
            return first, second

    first, second = asyncio.run(scenario())

    assert first == 272000
    assert second == 272000
    assert calls["n"] == 2


def test_cold_cache_structural_failure_returns_none() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("gpt-5.6-sol")

    assert asyncio.run(scenario()) is None


def test_non_json_decode_failure_degrades_like_structural_failure() -> None:
    # A 200 response whose body fails to decode with a non-JSONDecodeError
    # ValueError (here: undecodable bytes -> UnicodeDecodeError) must behave
    # exactly like any structural catalog failure: cold cache -> None, warm
    # cache -> stale value served.
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 2:
            return httpx.Response(200, json={"models": _CATALOG_MODELS})
        return httpx.Response(
            200, content=b"\xff\xfe\xff", headers={"Content-Type": "application/json"}
        )

    async def scenario() -> tuple[int | None, int | None, int | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            cold = await client.context_window("gpt-5.6-sol")
            # Clear the failure backoff so the next lookup refreshes.
            client._context_windows._failure_time = None
            warm = await client.context_window("gpt-5.6-sol")
            client._context_windows._snapshot_time = 0.0
            stale = await client.context_window("gpt-5.6-sol")
            return cold, warm, stale

    cold, warm, stale = asyncio.run(scenario())

    assert cold is None
    assert warm == 272000
    assert stale == 272000
    assert calls["n"] == 3


@pytest.mark.parametrize(
    "kind", ["non_200", "invalid_json", "missing_models_key", "non_list_models"]
)
def test_codex_stale_window_served_after_failed_refresh(kind: str) -> None:
    # A stale-on-structural-error test run immediately after a successful
    # fetch stays inside the 900s TTL and never exercises a failed refresh,
    # so a fake clock forces the snapshot past its TTL before the second
    # lookup, across every structural catalog-failure variant.
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"models": _CATALOG_MODELS})
        return _malformed_catalog_response(kind)

    async def scenario() -> tuple[int | None, int | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            clock = _FakeClock()
            client = _codex_client_with_fake_clock(http_client, clock)
            first = await client.context_window("gpt-5.6-sol")
            clock.advance(901.0)  # past the cache's 900s TTL
            second = await client.context_window("gpt-5.6-sol")
            return first, second

    first, second = asyncio.run(scenario())

    assert first == 272000
    assert second == 272000
    assert calls["n"] == 2


@pytest.mark.parametrize(
    "kind", ["non_200", "invalid_json", "missing_models_key", "non_list_models"]
)
def test_codex_cold_cache_malformed_catalog_returns_none(kind: str) -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _malformed_catalog_response(kind)

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("gpt-5.6-sol")

    assert asyncio.run(scenario()) is None
    assert calls["n"] == 1


def test_list_models_fetches_fresh_after_context_window_populated_cache() -> None:
    calls = {"n": 0}

    async def scenario() -> tuple[int | None, list[str]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            window = await client.context_window("gpt-5.6-sol")
            models = await client.list_models()
            return window, models

    window, models = asyncio.run(scenario())

    assert window == 272000
    assert "gpt-5.6-sol" in models
    assert "gpt-5.6-hidden" not in models
    assert calls["n"] == 2


def test_list_models_raises_upstream_error_on_non_200() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="expired")

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(_FakeAuthManager(), http_client).list_models()

    with pytest.raises(CodexUpstreamError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 401
