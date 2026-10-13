"""Response-lifetime ownership at the downstream ASGI send boundary."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import anyio
import h11
import httpx
import pytest
from starlette.requests import ClientDisconnect, Request
from uvicorn.protocols.http.flow_control import FlowControl
from uvicorn.protocols.http.h11_impl import H11Protocol, RequestResponseCycle

from claudex.config import GatewayConfig, RouteTarget
from claudex.providers.backends import AnthropicBackend, ResponsesBackend
from claudex.providers.client_support import stream_sse_events
from claudex.relay.anthropic_backend import _AnthropicStreamRelay, _AnthropicStreamingResponse
from claudex.relay.common import _relay_anthropic_response
from claudex.relay.openai_backend import (
    _CompactionStreamRelay,
    _OwnedStreamingResponse,
    _relay_via_responses_backend,
)
from claudex.upstream_errors import UpstreamError


_RESPONSE_EVENTS = (
    {"type": "response.created", "response": {"id": "resp_1", "model": "synthetic"}},
    {"type": "response.output_text.delta", "delta": "hello"},
    {"type": "response.completed", "response": {
        "id": "resp_1", "usage": {"input_tokens": 1, "output_tokens": 1},
    }},
)
_OWNERS = ("passthrough", "responses", "native", "compaction")


class TrackedByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.close_calls = 0
        self.chunks_read = 0
        self.should_block_read = False
        self.read_blocked = asyncio.Event()
        self.cleanup_error: Exception | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk
            if self.should_block_read:
                self.read_blocked.set()
                await asyncio.Event().wait()

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.close_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class SyntheticResponsesTransport:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def context_window(self, model: str) -> None:
        return None

    def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        return stream_sse_events(
            self.client, "https://synthetic.invalid/responses", payload, {},
            make_error=UpstreamError,
        )


@asynccontextmanager
async def _stream_case(owner: str):
    chunks = (b"event: ping\ndata: {}\n\n",) * 3
    if owner == "responses":
        chunks = tuple(f"data: {json.dumps(event)}\n\n".encode() for event in _RESPONSE_EVENTS)
    stream = TrackedByteStream(chunks)
    upstream = httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})
    completions: list[int] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: upstream), trust_env=False) as client:
        if owner == "passthrough":
            response = _relay_anthropic_response(
                upstream, on_finished=lambda: completions.append(stream.close_calls)
            )
        elif owner == "responses":
            async def adapt_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
                return payload

            backend = ResponsesBackend(
                transport=SyntheticResponsesTransport(client),
                adapt_payload=adapt_payload,
                adapt_probe_payload=lambda payload, model: payload,
                signature_namespace=None,
                codex_regex_compat=True,
            )
            app = SimpleNamespace(state=SimpleNamespace(
                config=GatewayConfig(), route_backends={"synthetic": backend},
            ))
            request = Request({
                "type": "http", "method": "POST", "path": "/v1/messages", "headers": [], "app": app,
            })
            response = await _relay_via_responses_backend(
                request,
                {"model": "synthetic", "max_tokens": 16, "stream": True,
                 "messages": [{"role": "user", "content": "synthetic input"}]},
                RouteTarget("synthetic", "synthetic"),
            )
            assert stream.chunks_read == 1
            assert not upstream.is_closed
        elif owner == "native":
            backend = AnthropicBackend(
                transport=object(), header_policy=lambda request: {},
                error_policy=lambda error: (502, {"type": "error", "error": {"type": "api_error"}}),
            )
            response = _AnthropicStreamingResponse(_AnthropicStreamRelay(upstream, "synthetic", backend))
        else:
            assert owner == "compaction"
            response = _OwnedStreamingResponse(_CompactionStreamRelay(upstream, SimpleNamespace(), 1))
        case = SimpleNamespace(
            owner=owner, stream=stream, upstream=upstream, response=response, completions=completions,
        )
        try:
            yield case
        finally:
            await response.body_iterator.aclose()
            await upstream.aclose()


def _assert_finished(case: SimpleNamespace) -> None:
    assert case.upstream.is_closed
    assert case.stream.close_calls == 1
    assert case.completions == ([1] if case.owner == "passthrough" else [])


async def _unexpected_receive() -> dict[str, Any]:
    raise AssertionError("ASGI 2.4 does not use a disconnect listener")


@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("send_type", ["http.response.start", "http.response.body"])
@pytest.mark.parametrize("error_type", [BrokenPipeError, RuntimeError])
def test_send_failure_releases_upstream_and_preserves_error(
    owner: str, send_type: str, error_type: type[Exception],
) -> None:
    async def scenario() -> None:
        async with _stream_case(owner) as case:
            failure = error_type("synthetic send failure")

            async def send(message: dict[str, Any]) -> None:
                if message["type"] == send_type:
                    raise failure

            expected = ClientDisconnect if error_type is BrokenPipeError else error_type
            with pytest.raises(expected) as caught:
                await case.response({"type": "http", "asgi": {"spec_version": "2.4"}}, _unexpected_receive, send)
            if expected is ClientDisconnect:
                assert caught.value.__context__ is failure
            else:
                assert caught.value is failure
            assert case.stream.chunks_read == (1 if owner == "responses" or send_type == "http.response.body" else 0)
            _assert_finished(case)
            await case.response.body_iterator.aclose()
            _assert_finished(case)

    asyncio.run(scenario())


@pytest.mark.parametrize("owner", ["passthrough", "responses"])
@pytest.mark.parametrize("send_type", ["http.response.start", "http.response.body"])
def test_send_cancellation_shields_async_upstream_cleanup(owner: str, send_type: str) -> None:
    async def scenario() -> None:
        async with _stream_case(owner) as case:
            is_cancelled = False
            with anyio.CancelScope() as cancel_scope:
                async def send(message: dict[str, Any]) -> None:
                    if message["type"] == send_type:
                        cancel_scope.cancel()
                        await anyio.sleep(0)

                try:
                    await case.response({"type": "http", "asgi": {"spec_version": "2.4"}}, _unexpected_receive, send)
                except asyncio.CancelledError:
                    is_cancelled = True
                    raise
            assert is_cancelled
            _assert_finished(case)

    asyncio.run(scenario())


def test_already_cancelled_response_scope_releases_upstream() -> None:
    async def scenario() -> None:
        async with _stream_case("passthrough") as case:
            with anyio.CancelScope() as cancel_scope:
                cancel_scope.cancel()

                async def send(message: dict[str, Any]) -> None:
                    await anyio.sleep(0)

                async def receive() -> dict[str, Any]:
                    await anyio.sleep(0)
                    return {"type": "http.disconnect"}

                await case.response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            _assert_finished(case)

    asyncio.run(scenario())


@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("should_start_body", [False, True])
def test_explicit_body_close_releases_upstream_once(owner: str, should_start_body: bool) -> None:
    async def scenario() -> None:
        async with _stream_case(owner) as case:
            if should_start_body:
                await anext(case.response.body_iterator)
            await case.response.body_iterator.aclose()
            _assert_finished(case)
            await case.response.body_iterator.aclose()
            _assert_finished(case)
            with pytest.raises(StopAsyncIteration):
                await anext(case.response.body_iterator)

    asyncio.run(scenario())


@pytest.mark.parametrize("owner", _OWNERS)
def test_normal_exhaustion_releases_upstream_once(owner: str) -> None:
    async def scenario() -> None:
        async with _stream_case(owner) as case:
            sent: list[dict[str, Any]] = []

            async def send(message: dict[str, Any]) -> None:
                sent.append(message)

            await case.response({"type": "http", "asgi": {"spec_version": "2.4"}}, _unexpected_receive, send)
            _assert_finished(case)
            assert sent[0]["type"] == "http.response.start"
            assert sent[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
            body = b"".join(message.get("body", b"") for message in sent)
            if owner == "responses":
                assert b"event: message_start" in body
                assert b'"text": "hello"' in body
                assert b"event: message_stop" in body
            else:
                assert body == b"".join(case.stream.chunks)

    asyncio.run(scenario())


def test_send_failure_keeps_primary_error_and_finishes_once_when_cleanup_fails() -> None:
    async def scenario() -> None:
        async with _stream_case("passthrough") as case:
            case.stream.cleanup_error = RuntimeError("synthetic cleanup failure")
            failure = BrokenPipeError("synthetic primary send failure")

            async def send(message: dict[str, Any]) -> None:
                raise failure

            with pytest.raises(ClientDisconnect) as caught:
                await case.response({"type": "http", "asgi": {"spec_version": "2.4"}}, _unexpected_receive, send)
            assert caught.value.__context__ is failure
            _assert_finished(case)
            await case.response.body_iterator.aclose()
            _assert_finished(case)

    asyncio.run(scenario())


def test_direct_close_propagates_cleanup_error_and_finishes_once() -> None:
    async def scenario() -> None:
        async with _stream_case("passthrough") as case:
            failure = RuntimeError("synthetic cleanup failure")
            case.stream.cleanup_error = failure
            with pytest.raises(RuntimeError) as caught:
                await case.response.body_iterator.aclose()
            assert caught.value is failure
            _assert_finished(case)
            await case.response.body_iterator.aclose()
            _assert_finished(case)

    asyncio.run(scenario())


def test_upstream_read_failure_preserves_primary_when_cleanup_fails() -> None:
    async def scenario() -> None:
        failure = ValueError("synthetic upstream read failure")

        class FailingReadStream(TrackedByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                if False:
                    yield b""
                raise failure

        stream = FailingReadStream(())
        stream.cleanup_error = RuntimeError("synthetic cleanup failure")
        upstream = httpx.Response(200, stream=stream)
        completions: list[int] = []
        response = _relay_anthropic_response(upstream, on_finished=lambda: completions.append(stream.close_calls))
        with pytest.raises(ValueError) as caught:
            await anext(response.body_iterator)
        assert caught.value is failure
        assert upstream.is_closed
        assert stream.close_calls == 1
        assert completions == [1]
        await response.body_iterator.aclose()
        assert completions == [1]

    asyncio.run(scenario())


@pytest.mark.parametrize("has_send_error", [False, True])
def test_completion_hook_error_is_visible_without_masking_primary(has_send_error: bool) -> None:
    async def scenario() -> None:
        stream = TrackedByteStream(())
        upstream = httpx.Response(200, stream=stream)
        completions: list[int] = []
        hook_failure = RuntimeError("synthetic completion failure")

        def on_finished() -> None:
            completions.append(stream.close_calls)
            raise hook_failure

        response = _relay_anthropic_response(upstream, on_finished=on_finished)
        if has_send_error:
            failure = BrokenPipeError("synthetic primary send failure")

            async def send(message: dict[str, Any]) -> None:
                raise failure

            with pytest.raises(ClientDisconnect) as caught:
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, _unexpected_receive, send)
            assert caught.value.__context__ is failure
        else:
            with pytest.raises(RuntimeError) as caught:
                await response.body_iterator.aclose()
            assert caught.value is hook_failure
        assert stream.close_calls == 1
        assert completions == [1]
        await response.body_iterator.aclose()
        assert completions == [1]

    asyncio.run(scenario())


def test_shared_sse_client_cancellation_releases_suspending_stream() -> None:
    async def scenario() -> None:
        stream = TrackedByteStream((b'data: {"type": "created"}\n\n',))
        stream.should_block_read = True
        upstream = httpx.Response(200, stream=stream)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: upstream), trust_env=False) as client:
            events = stream_sse_events(client, "https://synthetic.invalid/responses", {}, {}, make_error=UpstreamError)
            assert await anext(events) == {"type": "created"}
            is_cancelled = False

            async def consume() -> None:
                nonlocal is_cancelled
                try:
                    await anext(events)
                except asyncio.CancelledError:
                    is_cancelled = True
                    raise

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(consume)
                await asyncio.wait_for(stream.read_blocked.wait(), timeout=2)
                task_group.cancel_scope.cancel()
            assert is_cancelled
            assert upstream.is_closed
            assert stream.close_calls == 1
            await events.aclose()
            assert stream.close_calls == 1

    asyncio.run(scenario())


def test_shared_sse_client_preserves_compression_framing_and_filtering() -> None:
    async def scenario() -> None:
        wire_body = gzip.compress(
            'event: ignored\r\ndata: {"type": "delta", "text": "café"}\r\n\r\n'
            'data: not-json\n\ndata: [DONE]\n\ndata: {"type": "completed"}\n\n'.encode()
        )
        stream = TrackedByteStream((wire_body[:5], wire_body[5:12], wire_body[12:]))
        upstream = httpx.Response(200, stream=stream, headers={"content-encoding": "gzip"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: upstream), trust_env=False) as client:
            events = [event async for event in stream_sse_events(
                client, "https://synthetic.invalid/responses", {}, {}, make_error=UpstreamError,
            )]
        assert events == [{"type": "delta", "text": "café"}, {"type": "completed"}]
        assert upstream.is_closed
        assert stream.close_calls == 1

    asyncio.run(scenario())


class MemoryTransport:
    """A disposable H11 transport; no sockets or production listener."""

    def __init__(self, should_pause_body: bool) -> None:
        self.should_pause_body = should_pause_body
        self.writes = 0
        self.flow: FlowControl | None = None
        self.is_closed = False

    def write(self, data: bytes) -> None:
        self.writes += 1
        if self.should_pause_body and self.writes == 2:
            self.flow.pause_writing()

    def close(self) -> None:
        self.is_closed = True

    def is_closing(self) -> bool:
        return self.is_closed

    def pause_reading(self) -> None:
        pass

    def resume_reading(self) -> None:
        pass


class MemoryProtocol(SimpleNamespace):
    __hash__ = object.__hash__


class ObservableFlowControl(FlowControl):
    def __init__(self, transport: MemoryTransport) -> None:
        super().__init__(transport)
        self.drain_entered = asyncio.Event()

    async def drain(self) -> None:
        self.drain_entered.set()
        await super().drain()


@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("stage", ["headers", "body", "upstream_read"])
@pytest.mark.parametrize("should_cancel", [False, True], ids=["disconnect", "cancel"])
def test_uvicorn23_disconnect_and_explicit_cancellation_release_upstream(
    owner: str, stage: str, should_cancel: bool,
) -> None:
    async def scenario() -> None:
        async with _stream_case(owner) as case:
            case.stream.should_block_read = stage == "upstream_read"
            transport = MemoryTransport(should_pause_body=stage == "body")
            flow = ObservableFlowControl(transport)
            transport.flow = flow
            if stage == "headers":
                flow.pause_writing()
            connection = h11.Connection(h11.SERVER)
            connection.receive_data(b"POST /v1/messages HTTP/1.1\r\nHost: synthetic.invalid\r\nContent-Length: 0\r\n\r\n")
            assert isinstance(connection.next_event(), h11.Request)
            assert isinstance(connection.next_event(), h11.EndOfMessage)
            logger = logging.getLogger("stream-ownership-test")
            cycle = RequestResponseCycle(
                {"type": "http", "asgi": {"spec_version": "2.3"}, "method": "POST", "path": "/v1/messages", "http_version": "1.1", "headers": []},
                connection, transport, flow, logger, logger, False, [], asyncio.Event(), lambda: None,
            )
            cycle.more_body = False
            protocol = MemoryProtocol(
                connections=set(), logger=logger, client=None, cycle=cycle, conn=connection,
                flow=flow, transport=transport, _unset_keepalive_if_required=lambda: None,
            )
            task = asyncio.create_task(case.response(cycle.scope, cycle.receive, cycle.send))
            barrier = case.stream.read_blocked if stage == "upstream_read" else flow.drain_entered
            try:
                await asyncio.wait_for(barrier.wait(), timeout=2)
                if should_cancel:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    # The installed server wakes receive before resuming drain;
                    # ordinary ASGI 2.3 disconnect does not inject a send OSError.
                    H11Protocol.connection_lost(protocol, None)
                    await asyncio.wait_for(task, timeout=2)
                _assert_finished(case)
                await case.response.body_iterator.aclose()
                _assert_finished(case)
            finally:
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task

    asyncio.run(scenario())
