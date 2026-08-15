"""Tests for the custom OpenAI-compatible Responses client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from claudex.config import OpenAICompatibleProvider
from claudex.providers.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleUpstreamError,
)

_PROVIDER_NAME = "wrtn"
_BASE_URL = "https://model.example/api/v1"
_API_KEY = "sk-custom-secret"


def _provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        wire_api="responses",
        base_url=_BASE_URL,
        api_key=_API_KEY,
    )


def _sse(events: list[dict[str, Any]]) -> bytes:
    chunks = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
    return b": keep-alive\n\n" + chunks + b"data: [DONE]\n\n"


class _TrackedSSEByteStream(httpx.AsyncByteStream):
    """Yield SSE chunks, optionally block afterward, and record closure."""

    def __init__(self, chunks: list[bytes], *, should_hang: bool = False) -> None:
        self._chunks = chunks
        self._should_hang = should_hang
        self.wait_started = asyncio.Event()
        self.is_closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._should_hang:
            self.wait_started.set()
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.is_closed = True


async def _collect(
    client: OpenAICompatibleClient, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    return [event async for event in client.stream_responses(payload, "session-1")]


def test_stream_responses_posts_to_prefixed_url_and_parses_function_call_events() -> None:
    captured: list[httpx.Request] = []
    upstream_events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "function-1",
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup_weather",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "function-1",
            "delta": '{"city":"Seoul"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "function-1",
            "arguments": '{"city":"Seoul"}',
        },
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_sse(upstream_events),
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
            assert _API_KEY not in repr(client)
            return await _collect(client, {"model": "gpt-5.5", "stream": True})

    assert asyncio.run(scenario()) == upstream_events
    (request,) = captured
    assert str(request.url) == f"{_BASE_URL}/responses"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "text/event-stream"
    assert "x-grok-conv-id" not in request.headers


def test_stream_responses_explicit_close_closes_upstream_response_synchronously() -> None:
    upstream_event = {"type": "response.output_text.delta", "delta": "hello"}
    stream = _TrackedSSEByteStream([_sse([upstream_event])])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "text/event-stream"},
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
            events = client.stream_responses({"stream": True}, "session-1")
            assert await anext(events) == upstream_event
            assert stream.is_closed is False
            await events.aclose()
            assert stream.is_closed is True

    asyncio.run(scenario())


def test_stream_responses_cancellation_closes_upstream_response() -> None:
    upstream_event = {"type": "response.output_text.delta", "delta": "hello"}
    stream = _TrackedSSEByteStream([_sse([upstream_event])], should_hang=True)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "text/event-stream"},
        )

    async def consume(client: OpenAICompatibleClient) -> None:
        async for event in client.stream_responses({"stream": True}, "session-1"):
            assert event == upstream_event

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
            task = asyncio.create_task(consume(client))
            await asyncio.wait_for(stream.wait_started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.is_closed is True

    asyncio.run(scenario())


def test_stream_responses_raises_upstream_error_without_retry() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="rate limited")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
            async for _event in client.stream_responses({}, "session-1"):
                pass

    with pytest.raises(OpenAICompatibleUpstreamError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == "rate limited"
    assert "wrtn upstream returned 429" in str(exc_info.value)
    assert _API_KEY not in str(exc_info.value)
    assert _API_KEY not in repr(exc_info.value)
    assert calls["n"] == 1


def test_stream_responses_redacts_api_key_echoed_by_upstream() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid credential: {_API_KEY}")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
            async for _event in client.stream_responses({}, "session-1"):
                pass

    with pytest.raises(OpenAICompatibleUpstreamError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.body == "invalid credential: [REDACTED]"
    assert _API_KEY not in str(exc_info.value)
    assert _API_KEY not in repr(exc_info.value)


def test_list_models_returns_openai_catalog_ids() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "gpt-5.5", "object": "model"},
                    {"id": "gemini-3.1-pro", "object": "model"},
                    {"id": 123},
                    {"no_id": True},
                ],
            },
        )

    async def scenario() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await OpenAICompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            ).list_models()

    assert asyncio.run(scenario()) == ["gpt-5.5", "gemini-3.1-pro"]
    (request,) = captured
    assert str(request.url) == f"{_BASE_URL}/models"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["accept"] == "application/json"


@pytest.mark.parametrize(
    "kind", ["non_200", "invalid_json", "non_object", "missing_data", "non_list_data"]
)
def test_list_models_raises_on_structural_catalog_failure(kind: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if kind == "non_200":
            return httpx.Response(503, text="catalog unavailable")
        if kind == "invalid_json":
            return httpx.Response(200, content=b"not valid json{")
        if kind == "non_object":
            return httpx.Response(200, json=[])
        if kind == "missing_data":
            return httpx.Response(200, json={"object": "list"})
        return httpx.Response(200, json={"data": {"id": "not-a-list"}})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await OpenAICompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            ).list_models()

    with pytest.raises(OpenAICompatibleUpstreamError) as exc_info:
        asyncio.run(scenario())

    expected_status = 503 if kind == "non_200" else 502
    assert exc_info.value.status_code == expected_status
    assert _API_KEY not in str(exc_info.value)
    if kind == "invalid_json":
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestContextWindow:
    @staticmethod
    def _catalog_response(data: list[Any]) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": data})

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            (
                {
                    "id": "gpt-5.5",
                    "context_window": 400000,
                    "limits": {"contextWindow": 200000},
                },
                400000,
            ),
            ({"id": "gpt-5.5", "limits": {"contextWindow": 200000}}, 200000),
            ({"id": "gpt-5.5", "limits": {"contextWindow": 200000.0}}, 200000),
        ],
    )
    def test_reads_top_level_and_nested_catalog_shapes(
        self, entry: dict[str, Any], expected: int
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response([entry])

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await OpenAICompatibleClient(
                    _PROVIDER_NAME, _provider(), http_client
                ).context_window("gpt-5.5")

        assert asyncio.run(scenario()) == expected

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": "gpt-5.5"},
            {"id": "gpt-5.5", "context_window": "400000"},
            {"id": "gpt-5.5", "context_window": True},
            {"id": "gpt-5.5", "context_window": 0},
            {"id": "gpt-5.5", "context_window": -1},
            {"id": "gpt-5.5", "context_window": 400000.5},
            {"id": "gpt-5.5", "limits": None},
            {"id": "gpt-5.5", "limits": {"contextWindow": "200000"}},
            {
                "id": "gpt-5.5",
                "context_window": None,
                "limits": {"contextWindow": 200000},
            },
        ],
    )
    def test_missing_or_invalid_context_window_resolves_to_none(
        self, entry: dict[str, Any]
    ) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return self._catalog_response([entry])

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await OpenAICompatibleClient(
                    _PROVIDER_NAME, _provider(), http_client
                ).context_window("gpt-5.5")

        assert asyncio.run(scenario()) is None

    def test_reuses_fresh_catalog_snapshot(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return self._catalog_response(
                [{"id": "gpt-5.5", "limits": {"contextWindow": 200000}}]
            )

        async def scenario() -> tuple[int | None, int | None]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
                return (
                    await client.context_window("gpt-5.5"),
                    await client.context_window("gpt-5.5"),
                )

        assert asyncio.run(scenario()) == (200000, 200000)
        assert calls["n"] == 1

    def test_cold_cache_structural_failure_returns_none(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list"})

        async def scenario() -> int | None:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                return await OpenAICompatibleClient(
                    _PROVIDER_NAME, _provider(), http_client
                ).context_window("gpt-5.5")

        assert asyncio.run(scenario()) is None

    def test_serves_stale_window_after_refresh_failure(self) -> None:
        calls = {"n": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return self._catalog_response(
                    [{"id": "gpt-5.5", "limits": {"contextWindow": 200000}}]
                )
            return httpx.Response(503, text="catalog unavailable")

        async def scenario() -> tuple[int | None, int | None]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                client = OpenAICompatibleClient(_PROVIDER_NAME, _provider(), http_client)
                first = await client.context_window("gpt-5.5")
                client._context_windows._snapshot_time -= (
                    client._context_windows._ttl_seconds + 1
                )
                second = await client.context_window("gpt-5.5")
                return first, second

        assert asyncio.run(scenario()) == (200000, 200000)
        assert calls["n"] == 2
