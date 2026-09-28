"""Tests for the Codex client's model catalog and context-window lookup."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import claudex.providers.codex_client as codex_module
from claudex.providers.codex_auth import CodexAuthError, CodexCredentials
from claudex.providers.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex.providers.model_catalog_cache import ModelCatalogCache


_REAL_CREATE_SUBPROCESS_EXEC = asyncio.create_subprocess_exec
_REAL_WHICH = shutil.which


class _FakeAuthManager:
    def __init__(self) -> None:
        self.calls = 0

    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        self.calls += 1
        return CodexCredentials(access_token="codex-token-1", account_id="account-1")


@pytest.fixture(autouse=True)
def no_host_codex_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing_codex(*args: Any, **kwargs: Any) -> Any:
        assert args == ("codex", "--version")
        raise FileNotFoundError("codex")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_codex)
    monkeypatch.setattr(shutil, "which", lambda command: None)


@pytest.fixture
def installed_codex_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        assert args == ("codex", "--version")
        return _CodexVersionProcess("codex-cli 0.157.1")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)


class _CodexVersionProcess:
    pid = 12345

    def __init__(
        self, output: str, *, returncode: int = 0, stderr: str = "", hang: bool = False
    ) -> None:
        self.output = output
        self.returncode = returncode
        self.stderr = stderr
        self.hang = hang
        self.killed = False
        self.reaped = False
        self.started = asyncio.Event()
        self.wait = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        if self.hang:
            await self.wait.wait()
        self.reaped = True
        return self.output.encode(), self.stderr.encode()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.wait.set()


def _reject_unexpected_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request: {request.url}")


_CATALOG_MODELS: list[dict[str, Any]] = [
    {"slug": "gpt-6-astra", "context_window": 272000},
    {"slug": "gpt-6-sol", "context_window": 272000},
    {"slug": "gpt-6-luna", "context_window": 272000},
    {"slug": "gpt-6-hidden", "visibility": "hide", "context_window": 64000},
    {
        "slug": "gpt-5.6-sol",
        "context_window": 272000,
        "service_tiers": [
            {"id": "priority", "name": "Fast", "description": "Faster responses"}
        ],
    },
    {"slug": "gpt-5.3-codex-spark", "context_window": 128000, "service_tiers": []},
    {"slug": "gpt-5.4", "context_window": 272000, "max_context_window": 1000000},
    {"slug": "max-only-window", "max_context_window": 872000},
    {"slug": "context-larger-than-max", "context_window": 128000, "max_context_window": 64000},
    {"slug": "valid-context-malformed-max", "context_window": 128000, "max_context_window": True},
    {"slug": "valid-max-malformed-context", "context_window": "272000", "max_context_window": 872000},
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
    {"slug": "bool-max-window", "max_context_window": True},
    {"slug": "string-max-window", "max_context_window": "872000"},
    {"slug": "zero-window", "context_window": 0},
    {"slug": "negative-window", "context_window": -1},
    {"slug": "fractional-window", "context_window": 272000.5},
    {"slug": "integral-float-window", "context_window": 272000.0},
]


def _catalog_handler(calls: dict[str, int]) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert str(request.url).startswith(CODEX_MODELS_URL)
        expected_version = "0.157.1"
        assert request.url.params["client_version"] == expected_version
        assert request.headers["user-agent"].startswith(f"codex-tui/{expected_version} ")
        assert request.headers["user-agent"].endswith(f"(codex-tui; {expected_version})")
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


def test_hidden_model_excluded_from_list_but_resolvable_via_context_window(
    installed_codex_probe: None,
) -> None:
    calls = {"n": 0}

    async def scenario() -> tuple[list[str], int | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            models = await client.list_models()
            window = await client.context_window("gpt-5.6-hidden")
            return models, window

    models, window = asyncio.run(scenario())

    assert "gpt-5.6-hidden" not in models
    assert "gpt-6-hidden" not in models
    assert window == 64000


def test_context_window_uses_larger_catalog_value() -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window("gpt-5.4")

    assert asyncio.run(scenario()) == 1000000


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("max-only-window", 872000),
        ("context-larger-than-max", 128000),
        ("valid-context-malformed-max", 128000),
        ("valid-max-malformed-context", 872000),
    ],
)
def test_context_window_uses_each_valid_catalog_value(
    slug: str, expected: int
) -> None:
    calls = {"n": 0}

    async def scenario() -> int | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_catalog_handler(calls))) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            return await client.context_window(slug)

    assert asyncio.run(scenario()) == expected


@pytest.mark.parametrize(
    "slug",
    [
        "no-window-field",
        "string-window",
        "bool-window",
        "bool-max-window",
        "string-max-window",
        "zero-window",
        "negative-window",
        "fractional-window",
    ],
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


def test_list_models_fetches_fresh_after_context_window_populated_cache(
    installed_codex_probe: None,
) -> None:
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
    assert {"gpt-6-astra", "gpt-6-sol", "gpt-6-luna"}.issubset(models)
    assert "gpt-6-hidden" not in models
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
    assert request.headers["user-agent"].startswith("codex-tui/0.157.1 ")
    assert request.headers["user-agent"].endswith("(codex-tui; 0.157.1)")
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


def test_list_models_raises_upstream_error_on_non_200(
    installed_codex_probe: None,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="expired")

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(_FakeAuthManager(), http_client).list_models()

    with pytest.raises(CodexUpstreamError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 401


def test_missing_codex_returns_visible_presets_without_auth_or_http() -> None:
    auth = _FakeAuthManager()
    calls = {"http": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        raise AssertionError("missing Codex must not fetch suggestion catalog")

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(auth, http_client).list_models()

    assert asyncio.run(scenario()) == [
        "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
        "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
    ]
    assert auth.calls == 0
    assert calls["http"] == 0


@pytest.mark.parametrize(
    ("cli_version", "effective_version"),
    [
        ("0.99.0", "0.157.1"),
        ("0.157.1", "0.157.1"),
        ("0.160.0", "0.160.0"),
        ("0.158.0-beta.1", "0.158.0-beta.1"),
        ("0.157.1-rc.1", "0.157.1"),
    ],
)
def test_installed_codex_uses_numeric_floor_for_catalog_and_user_agent(
    cli_version: str, effective_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[Any, ...]] = []
    process = _CodexVersionProcess(f"codex-cli {cli_version}\n")

    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        commands.append(args)
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"models": _CATALOG_MODELS})

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(_FakeAuthManager(), http_client).list_models()

    models = asyncio.run(scenario())
    assert commands == [("codex", "--version")]
    assert process.reaped
    assert "gpt-6-hidden" not in models
    assert "gpt-6-astra" in models
    (request,) = captured
    assert request.url.params["client_version"] == effective_version
    assert request.headers["user-agent"].startswith(f"codex-tui/{effective_version} ")
    assert request.headers["user-agent"].endswith(f"(codex-tui; {effective_version})")


@pytest.mark.parametrize(
    ("output", "returncode", "message"),
    [
        ("broken", 0, "version"),
        ("codex-cli 0.160.0", 7, "exit"),
    ],
)
def test_broken_installed_codex_is_not_treated_as_missing(
    output: str, returncode: int, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        return _CodexVersionProcess(output, returncode=returncode, stderr="failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    auth = _FakeAuthManager()

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            return await CodexClient(auth, http_client).list_models()

    with pytest.raises(codex_module.CodexDiscoveryError, match=message):
        asyncio.run(scenario())
    assert auth.calls == 0


def test_codex_permission_failure_is_not_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    async def denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("codex is not executable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            return await CodexClient(_FakeAuthManager(), http_client).list_models()

    with pytest.raises(codex_module.CodexDiscoveryError, match="codex"):
        asyncio.run(scenario())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shebang behavior")
def test_present_codex_with_missing_shebang_interpreter_is_not_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = tmp_path / "codex"
    launcher.write_text("#!/nonexistent/codex-interpreter\n", encoding="utf-8")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _REAL_CREATE_SUBPROCESS_EXEC)
    monkeypatch.setattr(shutil, "which", _REAL_WHICH)
    auth = _FakeAuthManager()

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            return await CodexClient(auth, http_client).list_models()

    with pytest.raises(codex_module.CodexDiscoveryError, match="codex"):
        asyncio.run(scenario())
    assert auth.calls == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("cancel", [False, True])
def test_probe_reaps_child_holding_inherited_pipe_after_parent_exits(
    cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InheritedPipeProcess(_CodexVersionProcess):
        pid = 12345

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            # The child still owns the pipe after the launcher is killed.
            asyncio.get_running_loop().call_later(0.1, self.wait.set)

    process = InheritedPipeProcess("", hang=True)
    commands: list[dict[str, Any]] = []
    group_kills: list[int] = []

    async def spawn(*args: Any, **kwargs: Any) -> InheritedPipeProcess:
        commands.append(kwargs)
        return process

    def kill_group(process_group: int, signal_number: int) -> None:
        assert signal_number == signal.SIGKILL
        group_kills.append(process_group)
        process.wait.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", kill_group)
    monkeypatch.setattr(codex_module, "_CODEX_VERSION_PROBE_TIMEOUT", 0.01, raising=False)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            task = asyncio.create_task(CodexClient(_FakeAuthManager(), http_client).list_models())
            await asyncio.wait_for(process.started.wait(), timeout=1)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
            else:
                with pytest.raises(codex_module.CodexDiscoveryError, match="timed out"):
                    await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())
    assert commands[0]["start_new_session"] is True
    assert group_kills == [process.pid]
    assert not process.killed
    assert process.reaped


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_probe_cleanup_is_bounded_when_pipe_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OpenPipeProcess:
        pid = 12345
        returncode: int | None = None
        reaped = False

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.pipe_closed = asyncio.Event()
            self.parent_exited = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            self.started.set()
            await self.pipe_closed.wait()
            return b"", b""

        async def wait(self) -> int:
            await self.parent_exited.wait()
            self.reaped = True
            self.returncode = -9
            return -9

        def kill(self) -> None:
            self.parent_exited.set()

    process = OpenPipeProcess()

    async def spawn(*args: Any, **kwargs: Any) -> OpenPipeProcess:
        return process

    def kill_group(process_group: int, signal_number: int) -> None:
        assert process_group == process.pid
        process.parent_exited.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", kill_group)
    monkeypatch.setattr(codex_module, "_CODEX_VERSION_PROBE_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(codex_module, "_CODEX_VERSION_CLEANUP_TIMEOUT", 0.01)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            with pytest.raises(codex_module.CodexDiscoveryError, match="timed out"):
                await asyncio.wait_for(client.list_models(), timeout=0.2)

    asyncio.run(scenario())
    assert process.reaped


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_probe_cancellation_survives_double_cleanup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StalledProcess:
        pid = 12345

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.never_finishes = asyncio.Event()

        async def communicate(self) -> tuple[bytes, bytes]:
            self.started.set()
            await self.never_finishes.wait()
            return b"", b""

        async def wait(self) -> int:
            await self.never_finishes.wait()
            return -9

    process = StalledProcess()
    group_kills: list[int] = []

    async def spawn(*args: Any, **kwargs: Any) -> StalledProcess:
        return process

    def kill_group(process_group: int, signal_number: int) -> None:
        group_kills.append(process_group)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", kill_group)
    monkeypatch.setattr(codex_module, "_CODEX_VERSION_CLEANUP_TIMEOUT", 0.01)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            task = asyncio.create_task(CodexClient(_FakeAuthManager(), http_client).list_models())
            await asyncio.wait_for(process.started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.2)

    asyncio.run(scenario())
    assert group_kills == [process.pid]


@pytest.mark.parametrize("cancel", [False, True])
def test_codex_probe_timeout_and_cancellation_reap_process(
    cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _CodexVersionProcess("", hang=True)

    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(os, "killpg", lambda process_group, signal_number: process.kill())
    monkeypatch.setattr(codex_module, "_CODEX_VERSION_PROBE_TIMEOUT", 0.01, raising=False)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_reject_unexpected_http)) as http_client:
            task = asyncio.create_task(CodexClient(_FakeAuthManager(), http_client).list_models())
            await asyncio.wait_for(process.started.wait(), timeout=1)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(codex_module.CodexDiscoveryError, match="timed out"):
                    await task

    asyncio.run(scenario())
    assert process.killed
    assert process.reaped


def test_codex_version_cache_refreshes_after_install_and_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    versions = [None, "codex-cli 0.161.0", "codex-cli 0.162.0"]
    commands: list[tuple[Any, ...]] = []
    requested_versions: list[str] = []

    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        commands.append(args)
        version = versions.pop(0)
        if version is None:
            raise FileNotFoundError("codex")
        return _CodexVersionProcess(version)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_versions.append(request.url.params["client_version"])
        assert request.headers["user-agent"].startswith(
            f"codex-tui/{requested_versions[-1]} "
        )
        return httpx.Response(200, json={"models": _CATALOG_MODELS})

    async def scenario() -> list[list[str]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            client._version_discovery._clock = clock
            absent = await client.list_models()
            cached_absent = await client.list_models()
            clock.advance(61)
            installed = await client.list_models()
            cached_installed = await client.list_models()
            clock.advance(61)
            upgraded = await client.list_models()
            return [absent, cached_absent, installed, cached_installed, upgraded]

    results = asyncio.run(scenario())
    assert results[0] == results[1]
    assert results[0] == [
        "gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol",
        "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5",
    ]
    assert results[2] == results[3] == results[4]
    assert requested_versions == ["0.161.0", "0.161.0", "0.162.0"]
    assert commands == [("codex", "--version")] * 3


def test_expired_version_probe_failure_does_not_serve_stale_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    commands = 0
    http_calls = 0

    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        nonlocal commands
        commands += 1
        if commands == 2:
            raise PermissionError("Codex update is broken")
        return _CodexVersionProcess("codex-cli 0.161.0")

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(200, json={"models": _CATALOG_MODELS})

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            client._version_discovery._clock = clock
            await client.list_models()
            clock.advance(61)
            with pytest.raises(codex_module.CodexDiscoveryError, match="codex"):
                await client.list_models()

    asyncio.run(scenario())
    assert commands == 2
    assert http_calls == 1


def test_stream_uses_selected_catalog_identity_after_local_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> _CodexVersionProcess:
        return _CodexVersionProcess("codex-cli 0.161.0")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url).startswith(CODEX_MODELS_URL):
            return httpx.Response(200, json={"models": _CATALOG_MODELS})
        return httpx.Response(200, content=_sse([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            await client.list_models()
            assert await client.context_window("gpt-5.6-sol") == 272000
            await _collect(client, {"model": "gpt-5.6-sol"})

    asyncio.run(scenario())
    assert len(requests) == 3
    assert all(request.url.params["client_version"] == "0.161.0" for request in requests[:2])
    assert all(request.headers["user-agent"].startswith("codex-tui/0.161.0 ") for request in requests)


def test_stream_and_metadata_without_cli_use_bundled_identity_and_live_catalog() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url).startswith(CODEX_MODELS_URL):
            return httpx.Response(200, json={"models": _CATALOG_MODELS})
        return httpx.Response(200, content=_sse([]))

    async def scenario() -> tuple[int | None, bool]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = CodexClient(_FakeAuthManager(), http_client)
            window = await client.context_window("gpt-5.6-sol")
            fast = await client.supports_fast_tier("gpt-5.6-sol")
            await _collect(client, {"model": "gpt-5.6-sol"})
            return window, fast

    assert asyncio.run(scenario()) == (272000, True)
    assert len(requests) == 2
    assert requests[0].url.params["client_version"] == "0.157.1"
    assert all(request.headers["user-agent"].startswith("codex-tui/0.157.1 ") for request in requests)
