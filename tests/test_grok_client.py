"""Tests for the Grok Responses client and its payload sanitizer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import claudex.providers.grok_client as grok_module
from claudex.providers.model_catalog_cache import ModelCatalogCache
from claudex.providers.grok_auth import GrokAuthError, GrokCredentials
from claudex.providers.grok_client import (
    GROK_MODELS_URL,
    GROK_RESPONSES_URL,
    GrokClient,
    GrokUpstreamError,
    sanitize_grok_payload,
)


@pytest.fixture(autouse=True)
def no_host_grok_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing_grok(*args: Any, **kwargs: Any) -> Any:
        assert args == ("grok", "--version")
        raise FileNotFoundError("grok")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_grok)


class _GrokVersionProcess:
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
        self.finished = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        if self.hang:
            await self.finished.wait()
        self.reaped = True
        return self.output.encode(), self.stderr.encode()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.finished.set()


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
            service_tier="priority",
            stream_options={"include_usage": True},
            stop=["END"],
        )

        sanitized = sanitize_grok_payload(payload, "grok-4.5")

        for field in (
            "previous_response_id",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
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
            (" XHIGH ", "high"),
            ("future", "medium"),
        ],
    )
    def test_thinking_model_normalizes_and_maps_effort_with_medium_fallback(
        self, effort: str, expected: str
    ) -> None:
        sanitized = sanitize_grok_payload(
            _payload(reasoning={"effort": effort, "summary": "auto"}), "grok-4.5"
        )
        assert sanitized["reasoning"] == {"effort": expected, "summary": "auto"}

    def test_thinking_effort_update_preserves_nested_reasoning_alias(self) -> None:
        payload = _payload(reasoning={"effort": "xhigh", "summary": "auto"})

        sanitized = sanitize_grok_payload(payload, "grok-4.5")

        assert sanitized is not payload
        assert sanitized["reasoning"] is payload["reasoning"]
        assert payload["reasoning"] == {"effort": "high", "summary": "auto"}

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


class _FakeClock:
    """A controllable stand-in for `time.monotonic`, advanced explicitly.

    GrokClient exposes no public clock-injection parameter, so forcing the
    cache's 900s TTL to expire without a real sleep requires replacing the
    client's private `_context_windows` cache with one built from this fake
    clock (see `TestContextWindow._client_with_fake_clock` below).
    """

    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


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
    assert request.headers["x-grok-client-version"] == "0.2.93"
    assert request.headers["user-agent"] == "xai-grok-workspace/0.2.93"
    assert request.headers["x-grok-conv-id"] == "session-1"
    assert request.headers["accept"] == "text/event-stream"


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

        async def consume(client: GrokClient) -> None:
            async for event in client.stream_responses({"stream": True}, "session-1"):
                assert event == upstream_event
                first_event_seen.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            task = asyncio.create_task(consume(GrokClient(_FakeAuthManager(), http_client)))
            await asyncio.wait_for(first_event_seen.wait(), timeout=1)
            await asyncio.wait_for(stream.wait_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


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


class TestContextWindow:
    @staticmethod
    def _catalog_response(data: list[Any]) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": data})

    @staticmethod
    def _malformed_catalog_response(kind: str) -> httpx.Response:
        """Build a response for each structural catalog-failure variant."""
        if kind == "non_200":
            return httpx.Response(500, text="boom")
        if kind == "invalid_json":
            return httpx.Response(200, content=b"not valid json{")
        if kind == "missing_data_key":
            return httpx.Response(200, json={"object": "list"})
        if kind == "non_list_data":
            return httpx.Response(200, json={"object": "list", "data": {"not": "a-list"}})
        raise ValueError(f"unknown malformed-catalog kind: {kind}")

    @staticmethod
    def _client_with_fake_clock(http_client: httpx.AsyncClient, clock: _FakeClock) -> GrokClient:
        client = GrokClient(_FakeAuthManager(), http_client)
        client._context_windows = ModelCatalogCache(
            client._fetch_context_windows,
            expected_errors=(GrokAuthError, GrokUpstreamError, httpx.HTTPError),
            clock=clock,
        )
        return client

    def test_resolves_exact_id_and_ignores_sibling_field(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response(
                [{"id": "grok-4.5", "context_window": 500000, "auto_compact_threshold_percent": 80}]
            )

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-4.5")

        assert asyncio.run(scenario()) == 500000

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": "grok-4.5"},
            {"id": "grok-4.5", "context_window": "500000"},
            {"id": "grok-4.5", "context_window": True},
            {"id": "grok-4.5", "context_window": 0},
            {"id": "grok-4.5", "context_window": -1},
            {"id": "grok-4.5", "context_window": 500000.5},
        ],
    )
    def test_invalid_context_window_resolves_to_none(self, entry: dict[str, Any]) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response([entry])

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-4.5")

        assert asyncio.run(scenario()) is None

    def test_positive_integral_float_is_coerced_to_int(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response([{"id": "grok-4.5", "context_window": 500000.0}])

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-4.5")

        result = asyncio.run(scenario())
        assert result == 500000
        assert isinstance(result, int)

    def test_unknown_id_resolves_to_none(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response([{"id": "grok-4.5", "context_window": 500000}])

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-unknown")

        assert asyncio.run(scenario()) is None

    def test_cold_cache_structural_failure_returns_none(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-4.5")

        assert asyncio.run(scenario()) is None

    def test_stale_value_served_after_structural_refresh_failure(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return self._catalog_response([{"id": "grok-4.5", "context_window": 500000}])
            return httpx.Response(500, text="boom")

        async def scenario() -> tuple[int | None, int | None]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client = GrokClient(_FakeAuthManager(), http_client)
                first = await client.context_window("grok-4.5")
                # Rewind the snapshot past the TTL without a real sleep (an
                # absolute 0.0 is not reliably expired on low-uptime hosts).
                client._context_windows._snapshot_time -= (
                    client._context_windows._ttl_seconds + 1
                )
                second = await client.context_window("grok-4.5")
                return first, second

        first, second = asyncio.run(scenario())
        assert first == 500000
        assert second == 500000
        assert calls["n"] == 2

    def test_non_json_decode_failure_degrades_like_structural_failure(self) -> None:
        # A 200 response whose body raises a non-JSONDecodeError ValueError
        # (undecodable bytes -> UnicodeDecodeError) must degrade like any
        # structural failure: cold cache -> None, warm cache -> stale value.
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 2:
                return self._catalog_response([{"id": "grok-4.5", "context_window": 500000}])
            return httpx.Response(
                200, content=b"\xff\xfe\xff", headers={"Content-Type": "application/json"}
            )

        async def scenario() -> tuple[int | None, int | None, int | None]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client = GrokClient(_FakeAuthManager(), http_client)
                cold = await client.context_window("grok-4.5")
                client._context_windows._failure_time = None
                warm = await client.context_window("grok-4.5")
                client._context_windows._snapshot_time -= (
                    client._context_windows._ttl_seconds + 1
                )
                stale = await client.context_window("grok-4.5")
                return cold, warm, stale

        cold, warm, stale = asyncio.run(scenario())
        assert cold is None
        assert warm == 500000
        assert stale == 500000
        assert calls["n"] == 3

    @pytest.mark.parametrize(
        "kind", ["non_200", "invalid_json", "missing_data_key", "non_list_data"]
    )
    def test_stale_window_served_after_failed_refresh(self, kind: str) -> None:
        # A stale-on-structural-error test run immediately after a successful
        # fetch stays inside the 900s TTL and never exercises a failed
        # refresh, so a fake clock forces the snapshot past its TTL before
        # the second lookup, across every structural catalog-failure variant.
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return self._catalog_response([{"id": "grok-4.5", "context_window": 500000}])
            return self._malformed_catalog_response(kind)

        async def scenario() -> tuple[int | None, int | None]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                clock = _FakeClock()
                client = self._client_with_fake_clock(http_client, clock)
                first = await client.context_window("grok-4.5")
                clock.advance(901.0)  # past the cache's 900s TTL
                second = await client.context_window("grok-4.5")
                return first, second

        first, second = asyncio.run(scenario())
        assert first == 500000
        assert second == 500000
        assert calls["n"] == 2

    @pytest.mark.parametrize(
        "kind", ["non_200", "invalid_json", "missing_data_key", "non_list_data"]
    )
    def test_cold_cache_malformed_catalog_returns_none(self, kind: str) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return self._malformed_catalog_response(kind)

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await GrokClient(_FakeAuthManager(), http_client).context_window("grok-4.5")

        assert asyncio.run(scenario()) is None
        assert calls["n"] == 1

    def test_list_models_still_fetches_fresh_after_context_window_populates_cache(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return self._catalog_response([{"id": "grok-4.5", "context_window": 500000}])

        async def scenario() -> tuple[int | None, list[str]]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client = GrokClient(_FakeAuthManager(), http_client)
                window = await client.context_window("grok-4.5")
                models = await client.list_models()
                return window, models

        window, models = asyncio.run(scenario())
        assert window == 500000
        assert models == ["grok-4.5"]
        assert calls["n"] == 2

    def test_list_models_still_raises_on_upstream_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        async def scenario() -> None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                await GrokClient(_FakeAuthManager(), http_client).list_models()

        with pytest.raises(GrokUpstreamError) as exc_info:
            asyncio.run(scenario())
        assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    ("cli_version", "effective_version"),
    [("1.0.46", "1.0.46"), ("0.2.9", "0.2.93"), ("0.2.93", "0.2.93")],
)
def test_installed_grok_version_is_used_for_stream_and_catalog_headers(
    cli_version: str, effective_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[Any, ...]] = []
    process = _GrokVersionProcess(f"grok {cli_version} (2765805b9442) [stable]\nignored")

    async def spawn(*args: Any, **kwargs: Any) -> _GrokVersionProcess:
        commands.append(args)
        assert kwargs == {"stdout": asyncio.subprocess.PIPE, "stderr": asyncio.subprocess.PIPE}
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if str(request.url) == GROK_MODELS_URL:
            return httpx.Response(200, json={"data": [{"id": "grok-4.5"}]})
        return httpx.Response(200, content=_sse([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = GrokClient(_FakeAuthManager(), http_client)
            await _collect(client, _payload())
            assert await client.list_models() == ["grok-4.5"]

    asyncio.run(scenario())
    assert commands == [("grok", "--version")]
    assert process.reaped
    assert [str(request.url) for request in captured] == [GROK_RESPONSES_URL, GROK_MODELS_URL]
    for request in captured:
        assert request.headers["x-grok-client-version"] == effective_version
        assert request.headers["user-agent"] == f"xai-grok-workspace/{effective_version}"


def test_missing_grok_uses_bundled_version_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> None:
        discovery = grok_module.GrokVersionDiscovery()
        assert discovery.current_version == "0.2.93"
        assert await discovery.get_version() == "0.2.93"
        assert discovery.current_version == "0.2.93"

    asyncio.run(scenario())
    assert not caplog.records


@pytest.mark.parametrize(("output", "returncode"), [("garbage", 0), ("", 0), ("grok 1.0.46", 7)])
def test_broken_grok_probe_falls_back_and_warns_once(
    output: str,
    returncode: int,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def spawn(*args: Any, **kwargs: Any) -> _GrokVersionProcess:
        return _GrokVersionProcess(output, returncode=returncode, stderr="failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def scenario() -> None:
        discovery = grok_module.GrokVersionDiscovery()
        assert await discovery.get_version() == "0.2.93"
        assert await discovery.get_version() == "0.2.93"

    asyncio.run(scenario())
    assert len(caplog.records) == 1
    assert caplog.records[0].name == grok_module.__name__
    assert caplog.records[0].levelname == "WARNING"


def test_grok_os_error_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("grok is not executable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)
    assert asyncio.run(grok_module.GrokVersionDiscovery().get_version()) == "0.2.93"
    assert len(caplog.records) == 1
    assert "grok is not executable" in caplog.text


@pytest.mark.parametrize("cancel", [False, True])
def test_grok_probe_timeout_and_cancellation_kill_and_reap_process(
    cancel: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    process = _GrokVersionProcess("", hang=True)

    async def spawn(*args: Any, **kwargs: Any) -> _GrokVersionProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(grok_module, "_GROK_VERSION_PROBE_TIMEOUT", 0.01, raising=False)

    async def scenario() -> None:
        discovery = grok_module.GrokVersionDiscovery()
        task = asyncio.create_task(discovery.get_version())
        await asyncio.wait_for(process.started.wait(), timeout=1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert discovery._checked_at is None
        else:
            assert await asyncio.wait_for(task, timeout=1) == "0.2.93"

    asyncio.run(scenario())
    assert process.killed
    assert process.reaped
    assert len(caplog.records) == (0 if cancel else 1)


def test_grok_version_cache_expiry_updates_both_request_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = iter(["1.0.46", "1.0.47", "1.0.48"])
    commands: list[tuple[Any, ...]] = []

    async def spawn(*args: Any, **kwargs: Any) -> _GrokVersionProcess:
        commands.append(args)
        return _GrokVersionProcess(f"grok {next(versions)} (hash) [stable]")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if str(request.url) == GROK_MODELS_URL:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, content=_sse([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = GrokClient(_FakeAuthManager(), http_client)
            clock = _FakeClock()
            client._version_discovery._clock = clock
            await _collect(client, _payload())
            clock.advance(59)
            await client.list_models()
            assert len(commands) == 1
            clock.advance(1)
            await client.list_models()
            assert len(commands) == 2
            clock.advance(60)
            await _collect(client, _payload())
            assert client._version_discovery.current_version == "1.0.48"

    asyncio.run(scenario())
    assert len(commands) == 3
    for request, version in zip(captured, ["1.0.46", "1.0.46", "1.0.47", "1.0.48"], strict=True):
        assert request.headers["x-grok-client-version"] == version
        assert request.headers["user-agent"] == f"xai-grok-workspace/{version}"


def test_concurrent_grok_version_requests_share_one_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _GrokVersionProcess("grok 1.0.46", hang=True)
    commands: list[tuple[Any, ...]] = []

    async def spawn(*args: Any, **kwargs: Any) -> _GrokVersionProcess:
        commands.append(args)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def scenario() -> None:
        discovery = grok_module.GrokVersionDiscovery()
        tasks = [asyncio.create_task(discovery.get_version()) for _ in range(3)]
        await asyncio.wait_for(process.started.wait(), timeout=1)
        process.finished.set()
        assert await asyncio.gather(*tasks) == ["1.0.46"] * 3

    asyncio.run(scenario())
    assert commands == [("grok", "--version")]


@pytest.mark.parametrize("cancel", [False, True])
def test_grok_probe_cleanup_is_bounded_when_pipes_stay_open(
    cancel: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class OpenPipeProcess(_GrokVersionProcess):
        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = OpenPipeProcess("", hang=True)

    async def spawn(*args: Any, **kwargs: Any) -> OpenPipeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(grok_module, "_GROK_VERSION_PROBE_TIMEOUT", 0.01)
    monkeypatch.setattr(grok_module, "_GROK_VERSION_CLEANUP_TIMEOUT", 0.01, raising=False)

    async def scenario() -> None:
        discovery = grok_module.GrokVersionDiscovery()
        task = asyncio.create_task(discovery.get_version())
        await asyncio.wait_for(process.started.wait(), timeout=1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=0.2)
            assert discovery._checked_at is None
        else:
            assert await asyncio.wait_for(task, timeout=0.2) == "0.2.93"
            assert discovery._checked_at is not None

    asyncio.run(scenario())
    assert process.killed
    assert len(caplog.records) == (0 if cancel else 1)
    if not cancel:
        assert caplog.records[0].levelname == "WARNING"
        assert "could not reap its process" in caplog.text
