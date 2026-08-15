"""Tests for the Codex client's model catalog and context-window lookup."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from claudex.providers.codex_auth import CodexAuthError, CodexCredentials
from claudex.providers.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex.providers.model_catalog_cache import ModelCatalogCache


class _FakeAuthManager:
    def __init__(self) -> None:
        self.calls = 0

    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        self.calls += 1
        return CodexCredentials(access_token="codex-token-1", account_id="account-1")


_CATALOG_MODELS: list[dict[str, Any]] = [
    {
        "slug": "gpt-5.6-sol",
        "context_window": 272000,
        "service_tiers": [
            {"id": "priority", "name": "Fast", "description": "Faster responses"}
        ],
    },
    {"slug": "gpt-5.3-codex-spark", "context_window": 128000, "service_tiers": []},
    {"slug": "gpt-5.4", "context_window": 272000, "max_context_window": 1000000},
    {"slug": "malformed-tier-list", "context_window": 64000, "service_tiers": [None, "priority"]},
    {"slug": "malformed-tiers", "context_window": 64000, "service_tiers": {"id": "priority"}},
    {"slug": "gpt-5.6-hidden", "context_window": 64000, "visibility": "hide"},
    {"slug": "no-window-field"},
    {
        "slug": "string-window",
        "context_window": "272000",
        "service_tiers": [{"id": "priority"}],
    },
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


def _sse(events: list[dict[str, Any]]) -> bytes:
    chunks = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return chunks + b"data: [DONE]\n\n"


class _HangingSSEByteStream(httpx.AsyncByteStream):
    """Yield one SSE chunk, then wait until the consumer is cancelled."""

    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk
        self.wait_started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._chunk
        self.wait_started.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        pass


async def _collect(client: CodexClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event async for event in client.stream_responses(payload, "session-1")]


class _FakeClock:
    """A controllable stand-in for `time.monotonic`, advanced explicitly.

    CodexClient exposes no public clock-injection parameter, so forcing the
    cache's 900s TTL to expire without a real sleep requires replacing the
    client's private `_catalog_entries` cache with one built from this fake
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
    client._catalog_entries = ModelCatalogCache(
        client._fetch_catalog_entries,
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


def test_supports_fast_tier_from_catalog() -> None:
    calls = {"n": 0}

    async def scenario() -> bool:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_catalog_handler(calls))
        ) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.supports_fast_tier("gpt-5.6-sol")

    assert asyncio.run(scenario()) is True


@pytest.mark.parametrize(
    "slug",
    [
        "gpt-5.3-codex-spark",
        "gpt-5.4",
        "malformed-tier-list",
        "malformed-tiers",
        "does-not-exist",
    ],
)
def test_supports_fast_tier_is_false_when_not_advertised(slug: str) -> None:
    calls = {"n": 0}

    async def scenario() -> bool:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_catalog_handler(calls))
        ) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.supports_fast_tier(slug)

    assert asyncio.run(scenario()) is False


def test_catalog_fetch_serves_context_window_and_fast_tier() -> None:
    calls = {"n": 0}

    async def scenario() -> tuple[int | None, bool]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_catalog_handler(calls))
        ) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            window = await client.context_window("gpt-5.6-sol")
            supports_fast_tier = await client.supports_fast_tier("gpt-5.6-sol")
            return window, supports_fast_tier

    assert asyncio.run(scenario()) == (272000, True)
    assert calls["n"] == 1


def test_invalid_window_model_still_resolves_fast_tier() -> None:
    calls = {"n": 0}

    async def scenario() -> tuple[int | None, bool]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_catalog_handler(calls))
        ) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            window = await client.context_window("string-window")
            supports_fast_tier = await client.supports_fast_tier("string-window")
            return window, supports_fast_tier

    assert asyncio.run(scenario()) == (None, True)


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
            # CodexClient doesn't expose a clock hook, so rewind the snapshot
            # past the TTL without a real sleep. (Setting it to absolute 0.0
            # only reads as expired when monotonic uptime exceeds the TTL —
            # false on a freshly booted CI runner.)
            client._catalog_entries._snapshot_time -= (
                client._catalog_entries._ttl_seconds + 1
            )
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
            client._catalog_entries._failure_time = None
            warm = await client.context_window("gpt-5.6-sol")
            client._catalog_entries._snapshot_time -= (
                client._catalog_entries._ttl_seconds + 1
            )
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


def test_stream_responses_sends_fast_tier_routing_hint() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_sse([{"type": "response.created", "response": {"id": "r1"}}]),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await _collect(
                CodexClient(_FakeAuthManager(), http_client),
                {"model": "gpt-5.6-sol", "service_tier": "priority"},
            )

    events = asyncio.run(scenario())

    assert events == [{"type": "response.created", "response": {"id": "r1"}}]
    (request,) = captured
    assert str(request.url) == CODEX_RESPONSES_URL
    assert request.headers["x-codex-routing-hint"] == (
        "model=gpt-5.6-sol;tier=priority"
    )


def test_stream_responses_propagates_cancellation_after_first_event() -> None:
    upstream_event = {"type": "response.output_text.delta", "delta": "hello"}

    async def scenario() -> None:
        stream = _HangingSSEByteStream(_sse([upstream_event]))
        first_event_seen = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                stream=stream,
                headers={"content-type": "text/event-stream"},
            )

        async def consume(client: CodexClient) -> None:
            async for event in client.stream_responses({"stream": True}, "session-1"):
                assert event == upstream_event
                first_event_seen.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            task = asyncio.create_task(consume(CodexClient(_FakeAuthManager(), http_client)))
            await asyncio.wait_for(first_event_seen.wait(), timeout=1)
            await asyncio.wait_for(stream.wait_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_stream_responses_omits_routing_hint_without_service_tier() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=_sse([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await _collect(CodexClient(_FakeAuthManager(), http_client), {"model": "gpt-5.6-sol"})

    asyncio.run(scenario())

    (request,) = captured
    assert "x-codex-routing-hint" not in request.headers


def test_list_models_raises_upstream_error_on_non_200() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="expired")

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(_FakeAuthManager(), http_client).list_models()

    with pytest.raises(CodexUpstreamError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 401
