"""Bounded liveness of the shared Responses-to-Messages streaming relay."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.requests import ClientDisconnect, Request

from claudex.config import GatewayConfig, RouteTarget
from claudex.providers.backends import ResponsesBackend
from claudex.providers.client_support import stream_sse_events
from claudex.upstream_errors import UpstreamAuthError, UpstreamError
import claudex.relay.openai_backend as relay


_CREATED = {"type": "response.created", "response": {"id": "resp_1", "model": "upstream-model"}}
_TEXT = {"type": "response.output_text.delta", "delta": "hello"}
_COMPLETED = {
    "type": "response.completed",
    "response": {"id": "resp_1", "usage": {"input_tokens": 2, "output_tokens": 1}},
}
_PING = 'event: ping\ndata: {"type": "ping"}\n\n'


class ControlledResponsesTransport:
    def __init__(self, *events: dict[str, Any] | Exception | None) -> None:
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)
        self.read_calls = 0
        self.read_tasks = []
        self.read_started = asyncio.Event()
        self.active_reads = 0
        self.cancelled_reads = 0
        self.close_calls = 0
        self.cleanup_error = None

    async def context_window(self, model: str) -> None:
        return None

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        try:
            while True:
                self.read_calls += 1
                self.read_tasks.append(asyncio.current_task())
                self.active_reads += 1
                self.read_started.set()
                try:
                    event = await self.events.get()
                except asyncio.CancelledError:
                    self.cancelled_reads += 1
                    raise
                finally:
                    self.active_reads -= 1
                if event is None:
                    return
                if isinstance(event, Exception):
                    raise event
                yield event
        finally:
            self.close_calls += 1
            if self.cleanup_error is not None:
                raise self.cleanup_error


async def _response(transport, *, provider="codex", should_stream=True, **request_fields):
    async def adapt_payload(payload, model):
        return payload

    backend = ResponsesBackend(
        transport=transport,
        adapt_payload=adapt_payload,
        adapt_probe_payload=lambda payload, model: payload,
        signature_namespace=None if provider == "codex" else provider,
        codex_regex_compat=provider == "codex",
    )
    app = SimpleNamespace(state=SimpleNamespace(
        config=GatewayConfig(), route_backends={provider: backend},
    ))
    request = Request({
        "type": "http", "method": "POST", "path": "/v1/messages", "headers": [], "app": app,
    })
    return await relay._relay_via_responses_backend(
        request,
        {"model": "claude-requested", "max_tokens": 16, "stream": should_stream,
         "messages": [{"role": "user", "content": "synthetic input"}], **request_fields},
        RouteTarget(provider, "upstream-model"),
    )


def _event(chunk):
    event_line, data_line, boundary = chunk.split("\n", 2)
    assert boundary == "\n"
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


@pytest.mark.parametrize("provider", ["codex", "grok", "custom-responses"])
def test_silent_stream_pings_without_restarting_pending_read(monkeypatch, provider):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01, raising=False)

    async def scenario():
        transport = ControlledResponsesTransport(_CREATED)
        response = await _response(transport, provider=provider)
        assert response.status_code == 200
        assert transport.read_calls == 1
        assert transport.close_calls == 0
        try:
            first = await asyncio.wait_for(anext(response.body_iterator), timeout=1)
            event_name, payload = _event(first)
            assert event_name == "message_start"
            assert payload["message"]["model"] == "claude-requested"
            for _ in range(3):
                assert await asyncio.wait_for(anext(response.body_iterator), timeout=0.3) == _PING
                assert transport.read_calls == 2
                assert transport.active_reads == 1
                assert transport.cancelled_reads == 0
            pending_read = transport.read_tasks[-1]
            transport.events.put_nowait(_TEXT)
            transport.events.put_nowait(_COMPLETED)
            transport.events.put_nowait(None)
            chunks = [chunk async for chunk in response.body_iterator]
            assert [_event(chunk)[0] for chunk in chunks] == [
                "content_block_start", "content_block_delta", "content_block_stop",
                "message_delta", "message_stop",
            ]
            assert _event(chunks[1])[1]["delta"] == {"type": "text_delta", "text": "hello"}
            assert transport.read_tasks[1] is pending_read
            assert pending_read.done()
            assert transport.active_reads == 0
            assert transport.close_calls == 1
        finally:
            await response.body_iterator.aclose()

    asyncio.run(scenario())


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.timeouts = []
        self.should_tick = True

    def time(self):
        return self.now

    async def wait(self, tasks, *, timeout):
        self.timeouts.append(timeout)
        if not self.should_tick:
            return await asyncio.wait(tasks, timeout=1)
        await asyncio.sleep(0)
        done = {task for task in tasks if task.done()}
        if done:
            return done, set(tasks) - done
        self.now += timeout
        return set(), set(tasks)

    def install(self, monkeypatch):
        monkeypatch.setattr(relay, "asyncio", SimpleNamespace(
            **{name: getattr(asyncio, name) for name in (
                "Task", "CancelledError", "create_task", "to_thread",
            )},
            get_running_loop=lambda: self,
            wait=self.wait,
        ))


def test_heartbeats_expire_at_translated_output_inactivity_bound(monkeypatch):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        closed = asyncio.Event()
        reader = None

        async def translated_events():
            nonlocal reader
            try:
                yield relay._format_sse("message_start", {"type": "message_start"})
                reader = asyncio.current_task()
                await asyncio.Event().wait()
            finally:
                closed.set()

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert _event(await anext(body))[0] == "message_start"
            chunks = [await anext(body) for _ in range(60)]
            assert chunks[:-1] == [_PING] * 59
            assert _event(chunks[-1]) == ("error", {
                "type": "error", "error": {
                    "type": "api_error",
                    "message": "responses stream timed out after 600 seconds without translated output",
                },
            })
            assert clock.now == 600
            assert closed.is_set()
            assert reader.done()
            assert reader.cancelled()
            with pytest.raises(StopAsyncIteration):
                await anext(body)
            assert clock.now == 600
        finally:
            await body.aclose()

    asyncio.run(scenario())


def test_real_translated_progress_allows_streams_longer_than_inactivity_limit(monkeypatch):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        gates = [asyncio.Event(), asyncio.Event()]

        async def translated_events():
            yield relay._format_sse("message_start", {"type": "message_start"})
            for index, gate in enumerate(gates):
                await gate.wait()
                yield relay._format_sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": str(index)},
                })
            yield relay._format_sse("message_stop", {"type": "message_stop"})

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert _event(await anext(body))[0] == "message_start"
            for index, gate in enumerate(gates):
                assert [await anext(body) for _ in range(59)] == [_PING] * 59
                gate.set()
                name, payload = _event(await anext(body))
                assert name == "content_block_delta"
                assert payload["delta"]["text"] == str(index)
            assert clock.now == 1180
            assert _event(await anext(body))[0] == "message_stop"
            with pytest.raises(StopAsyncIteration):
                await anext(body)
        finally:
            await body.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal_type", [
    "response.completed", "response.incomplete", "response.failed", "error",
])
def test_terminal_output_stops_without_provider_eof_or_later_pings(monkeypatch, terminal_type):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        terminal = {
            "type": terminal_type,
            "response": {"output": [], "error": {"message": "synthetic upstream failure"}},
            "error": {"message": "synthetic upstream failure"},
        }
        transport = ControlledResponsesTransport(_CREATED, terminal)
        response = await _response(transport)
        try:
            assert _event(await anext(response.body_iterator))[0] == "message_start"
            expected = "error" if terminal_type in ("response.failed", "error") else "message_stop"
            names = []
            while not names or names[-1] != expected:
                name, payload = _event(await asyncio.wait_for(anext(response.body_iterator), timeout=1))
                names.append(name)
                assert name != "ping"
                if name == "error":
                    assert payload["error"]["message"] == "synthetic upstream failure"
            assert names == (["error"] if expected == "error" else ["message_delta", "message_stop"])
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(response.body_iterator), timeout=0.3)
            assert transport.read_calls == 2
            assert transport.close_calls == 1
            assert transport.active_reads == 0
        finally:
            await response.body_iterator.aclose()

    asyncio.run(scenario())


def test_liveness_defaults_are_private_ten_and_six_hundred_seconds():
    assert relay._RESPONSES_HEARTBEAT_INTERVAL_SECONDS == 10
    assert relay._RESPONSES_OUTPUT_INACTIVITY_TIMEOUT_SECONDS == 600


def test_inactivity_expiry_uses_only_the_remaining_wait_budget(monkeypatch):
    clock = ManualClock()
    clock.install(monkeypatch)
    monkeypatch.setattr(relay, "_RESPONSES_OUTPUT_INACTIVITY_TIMEOUT_SECONDS", 25)

    async def scenario():
        async def translated_events():
            await asyncio.Event().wait()
            yield "unreachable"

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert [await anext(body) for _ in range(2)] == [_PING, _PING]
            name, payload = _event(await anext(body))
            assert name == "error"
            assert payload["error"]["type"] == "api_error"
            assert clock.now == 25
            assert clock.timeouts == [10, 10, 5]
        finally:
            await body.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("should_have_output", [False, True], ids=["immediate-eof", "ready-data"])
def test_immediately_ready_translated_data_and_end_do_not_ping(monkeypatch, should_have_output):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        expected = [relay._format_sse("content_block_delta", {"type": "content_block_delta"})] if should_have_output else []

        async def translated_events():
            for chunk in expected:
                yield chunk

        assert [chunk async for chunk in relay._with_responses_heartbeats(translated_events())] == expected
        assert clock.now == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("traffic, should_delay_worker", [
    pytest.param("comments", False, id="comments"),
    pytest.param("unknown-events", False, id="unknown-events"),
    pytest.param("buffered-arguments", False, id="buffered-arguments"),
    pytest.param("buffered-arguments", True, id="buffered-arguments-delayed-worker"),
])
def test_raw_upstream_activity_does_not_reset_translated_inactivity(
    monkeypatch, traffic, should_delay_worker,
):
    async def scenario():
        packets = asyncio.Queue()

        class TrafficClock(ManualClock):
            async def wait(self, tasks, *, timeout):
                if self.should_tick:
                    await packets.join()
                return await super().wait(tasks, timeout=timeout)

        clock = TrafficClock()
        clock.should_tick = False
        clock.install(monkeypatch)
        initial_tasks = asyncio.all_tasks()
        release_worker = threading.Event()
        worker_finished = threading.Event()
        worker_release_times = []
        release_handle = None
        if should_delay_worker:
            loop = asyncio.get_running_loop()
            translate_event = relay.CodexToClaudeStreamTranslator.translate_event
            turns = 2 * (
                relay._RESPONSES_OUTPUT_INACTIVITY_TIMEOUT_SECONDS
                // relay._RESPONSES_HEARTBEAT_INTERVAL_SECONDS
            ) + 4

            def release_after_turns(remaining):
                nonlocal release_handle
                if remaining:
                    release_handle = loop.call_soon(release_after_turns, remaining - 1)
                else:
                    worker_release_times.append(clock.now)
                    release_worker.set()

            def delayed_translate_event(self, event):
                nonlocal release_handle
                if event.get("type") != "response.output_item.added":
                    return translate_event(self, event)
                # Outlast a full set of unsynchronized virtual timeout ticks.
                release_handle = loop.call_soon_threadsafe(release_after_turns, turns)
                release_worker.wait()
                result = translate_event(self, event)
                worker_finished.set()
                return result

            monkeypatch.setattr(
                relay.CodexToClaudeStreamTranslator, "translate_event", delayed_translate_event,
            )

        class TrafficStream(httpx.AsyncByteStream):
            def __init__(self):
                self.chunks_read = 0
                self.read_tasks = []
                self.close_calls = 0

            async def __aiter__(self):
                self.chunks_read += 1
                yield f"data: {json.dumps(_CREATED)}\n\n".encode()
                while True:
                    self.read_tasks.append(asyncio.current_task())
                    packet = await packets.get()
                    self.chunks_read += 1
                    try:
                        yield packet
                    finally:
                        # Resumption acknowledges the real parser/translator handoff.
                        packets.task_done()

            async def aclose(self):
                await asyncio.sleep(0)
                self.close_calls += 1

        class TrafficTransport:
            async def context_window(self, model):
                return None

            def stream_responses(self, payload, session_id):
                return stream_sse_events(
                    client, "https://synthetic.invalid/responses", payload, {}, make_error=UpstreamError,
                )

        stream = TrafficStream()
        upstream = httpx.Response(200, stream=stream)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: upstream), trust_env=False) as client:
            response = await _response(TrafficTransport(), tools=[{
                "name": "lookup", "input_schema": {"type": "object"},
            }])
            try:
                assert _event(await anext(response.body_iterator))[0] == "message_start"
                clock.should_tick = True
                if traffic == "buffered-arguments":
                    packets.put_nowait(b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"lookup"}}\n\n')
                noise = {
                    "comments": b": upstream keepalive\n\n",
                    "unknown-events": b'data: {"type":"response.unknown"}\n\n',
                    "buffered-arguments": b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":" "}\n\n',
                }[traffic]
                chunks = []
                for _ in range(60):
                    packets.put_nowait(noise)
                    chunks.append(await anext(response.body_iterator))
                assert chunks[:-1] == [_PING] * 59
                name, payload = _event(chunks[-1])
                assert name == "error"
                assert payload["error"]["type"] == "api_error"
                assert "600 seconds without translated output" in payload["error"]["message"]
                assert "tool_use" not in "".join(chunks)
                assert "message_stop" not in "".join(chunks)
                assert clock.now == 600
                assert stream.chunks_read == 61 + (traffic == "buffered-arguments")
                assert len(set(stream.read_tasks)) == 1
                assert stream.read_tasks[0].done()
                assert upstream.is_closed
                assert stream.close_calls == 1
                with pytest.raises(StopAsyncIteration):
                    await anext(response.body_iterator)
                assert asyncio.all_tasks() <= initial_tasks
                if should_delay_worker:
                    assert worker_finished.is_set()
                    assert worker_release_times == [0]
            finally:
                release_worker.set()
                if release_handle is not None:
                    release_handle.cancel()
                await response.body_iterator.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["B", "A"], ids=["valid-completion", "invalid-completion"])
def test_heartbeats_do_not_release_unvalidated_function_arguments(monkeypatch, mode):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        arguments = json.dumps({"mode": mode})
        item = {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "lookup"}
        transport = ControlledResponsesTransport(
            _CREATED,
            {"type": "response.output_item.added", "output_index": 0, "item": item},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": arguments[:5]},
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": arguments[5:]},
            {"type": "response.function_call_arguments.done", "output_index": 0, "arguments": arguments},
        )
        response = await _response(transport, tools=[{
            "name": "lookup", "input_schema": {
                "type": "object", "properties": {"mode": {"enum": ["B"]}},
                "required": ["mode"], "additionalProperties": False,
            },
        }])
        try:
            assert _event(await anext(response.body_iterator))[0] == "message_start"
            assert await asyncio.wait_for(anext(response.body_iterator), timeout=1) == _PING
            assert transport.read_calls == 6
            assert transport.active_reads == 1
            transport.events.put_nowait({
                "type": "response.output_item.done", "output_index": 0,
                "item": {**item, "arguments": arguments},
            })
            transport.events.put_nowait(_COMPLETED)
            events = [_event(chunk) async for chunk in response.body_iterator if chunk != _PING]
            if mode == "B":
                assert [name for name, _ in events] == [
                    "content_block_start", "content_block_delta", "content_block_stop",
                    "message_delta", "message_stop",
                ]
                assert events[0][1]["content_block"] == {
                    "type": "tool_use", "id": "call_1", "name": "lookup", "input": {},
                }
                assert json.loads(events[1][1]["delta"]["partial_json"]) == {"mode": "B"}
                assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
            else:
                assert len(events) == 1
                assert events[0][0] == "error"
                assert events[0][1]["error"]["type"] == "api_error"
                assert "declared schema" in events[0][1]["error"]["message"]
            assert transport.close_calls == 1
            assert transport.active_reads == 0
        finally:
            await response.body_iterator.aclose()

    asyncio.run(scenario())


def test_inactivity_error_stays_primary_when_pending_read_cleanup_fails(monkeypatch, caplog):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        async def translated_events():
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                raise RuntimeError("synthetic cleanup failure")

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert [await anext(body) for _ in range(59)] == [_PING] * 59
            name, payload = _event(await anext(body))
            assert name == "error"
            assert payload["error"]["type"] == "api_error"
            assert "600 seconds without translated output" in payload["error"]["message"]
            with pytest.raises(StopAsyncIteration):
                await anext(body)
        finally:
            await body.aclose()

    asyncio.run(scenario())
    assert "responses stream cleanup failed" in caplog.text
    assert "synthetic cleanup failure" in caplog.text


@pytest.mark.parametrize("should_stream", [False, True], ids=["aggregate", "stream"])
@pytest.mark.parametrize("failure, status, error_type, message", [
    (UpstreamError(400, '{"error":{"message":"synthetic invalid request"}}'), 400, "invalid_request_error", "synthetic invalid request"),
    (UpstreamError(429, '{"error":{"message":"synthetic quota failure"}}'), 429, "rate_limit_error", "synthetic quota failure"),
    (UpstreamError(503, '{"error":{"message":"synthetic overload"}}'), 503, "overloaded_error", "synthetic overload"),
    (UpstreamAuthError("synthetic authentication failure"), 401, "authentication_error", "synthetic authentication failure"),
    (httpx.ReadTimeout("synthetic read timeout"), 502, "api_error", "synthetic read timeout"),
    (None, 502, "api_error", "stream ended without any events"),
])
def test_prefetch_errors_keep_http_status_without_starting_heartbeat_body(
    monkeypatch, should_stream, failure, status, error_type, message,
):
    def forbidden_heartbeat_body(*args):
        raise AssertionError("prefetch did not succeed")

    monkeypatch.setattr(relay, "_with_responses_heartbeats", forbidden_heartbeat_body)

    async def scenario():
        transport = ControlledResponsesTransport(failure)
        response = await _response(transport, should_stream=should_stream)
        assert response.status_code == status
        body = json.loads(response.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == error_type
        assert message in body["error"]["message"]
        assert not hasattr(response, "body_iterator")
        assert transport.read_calls == 1
        assert transport.close_calls == 1
        assert transport.active_reads == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("should_stream", [False, True], ids=["aggregate", "stream"])
def test_prefetch_wait_has_no_premature_response_and_remains_cancellable(monkeypatch, should_stream):
    def forbidden_heartbeat_body(*args):
        raise AssertionError("prefetch is still pending")

    monkeypatch.setattr(relay, "_with_responses_heartbeats", forbidden_heartbeat_body)

    async def scenario():
        transport = ControlledResponsesTransport()
        task = asyncio.create_task(_response(transport, should_stream=should_stream))
        try:
            await asyncio.wait_for(transport.read_started.wait(), timeout=1)
            assert not task.done()
            assert transport.close_calls == 0
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert transport.cancelled_reads == 1
            assert transport.active_reads == 0
            assert transport.close_calls == 1
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


def test_nonstream_aggregation_does_not_use_heartbeats(monkeypatch):
    def forbidden_heartbeat_body(*args):
        raise AssertionError("nonstreaming requests do not use heartbeats")

    monkeypatch.setattr(relay, "_with_responses_heartbeats", forbidden_heartbeat_body)

    async def scenario():
        transport = ControlledResponsesTransport(_CREATED, _TEXT, _COMPLETED, None)
        response = await _response(transport, should_stream=False)
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["type"] == "message"
        assert body["model"] == "claude-requested"
        assert body["content"] == [{"type": "text", "text": "hello"}]
        assert body["stop_reason"] == "end_turn"
        assert body["usage"] == {"input_tokens": 2, "output_tokens": 1}
        assert transport.close_calls == 1
        assert transport.active_reads == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("failure, error_type", [
    (UpstreamError(429, '{"error":{"message":"synthetic midstream quota failure"}}'), "rate_limit_error"),
    (UpstreamAuthError("synthetic midstream auth failure"), "authentication_error"),
    (httpx.ReadTimeout("synthetic midstream timeout"), "api_error"),
    (ValueError("synthetic unhandled read failure"), None),
])
def test_upstream_failure_after_ping_stops_and_releases_owned_read(monkeypatch, failure, error_type):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        transport = ControlledResponsesTransport(_CREATED)
        response = await _response(transport)
        try:
            assert _event(await anext(response.body_iterator))[0] == "message_start"
            assert await asyncio.wait_for(anext(response.body_iterator), timeout=1) == _PING
            pending_read = transport.read_tasks[-1]
            transport.events.put_nowait(failure)
            if error_type is None:
                with pytest.raises(ValueError) as caught:
                    await anext(response.body_iterator)
                assert caught.value is failure
            else:
                name, payload = _event(await anext(response.body_iterator))
                assert name == "error"
                assert payload["error"]["type"] == error_type
                assert "synthetic" in payload["error"]["message"]
                with pytest.raises(StopAsyncIteration):
                    await anext(response.body_iterator)
            assert pending_read.done()
            assert transport.read_calls == 2
            assert transport.close_calls == 1
            assert transport.active_reads == 0
        finally:
            await response.body_iterator.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("after_ping", [False, True], ids=["unstarted-body", "between-heartbeats"])
@pytest.mark.parametrize("has_cleanup_error", [False, True], ids=["clean-close", "cleanup-failure"])
def test_explicit_close_joins_owned_read_and_keeps_cleanup_failures_visible(
    monkeypatch, after_ping, has_cleanup_error,
):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        transport = ControlledResponsesTransport(_CREATED)
        failure = RuntimeError("synthetic explicit-close failure")
        if has_cleanup_error:
            transport.cleanup_error = failure
        response = await _response(transport)
        pending_read = None
        if after_ping:
            assert _event(await anext(response.body_iterator))[0] == "message_start"
            assert await asyncio.wait_for(anext(response.body_iterator), timeout=1) == _PING
            pending_read = transport.read_tasks[-1]
            assert transport.active_reads == 1
        if has_cleanup_error:
            with pytest.raises(RuntimeError) as caught:
                await response.body_iterator.aclose()
            assert caught.value is failure
        else:
            await response.body_iterator.aclose()
        assert transport.close_calls == 1
        assert transport.active_reads == 0
        if pending_read is not None:
            assert pending_read.done()
            assert transport.cancelled_reads == 1
        await response.body_iterator.aclose()
        assert transport.close_calls == 1
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", [
    "header-send", "heartbeat-send", "cancel-send", "disconnect-send", "cancel-read", "disconnect-read",
])
@pytest.mark.parametrize("has_cleanup_error", [False, True], ids=["clean-close", "cleanup-failure"])
def test_real_asgi_exit_joins_pending_heartbeat_read_and_preserves_primary(
    monkeypatch, stage, has_cleanup_error, caplog,
):
    monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def scenario():
        transport = ControlledResponsesTransport(_CREATED)
        if has_cleanup_error:
            transport.cleanup_error = RuntimeError("synthetic ASGI cleanup failure")
        response = await _response(transport)
        failure = BrokenPipeError("synthetic ASGI send failure")
        ping_sent = asyncio.Event()
        disconnect = asyncio.Event()
        pending_read = None
        initial_tasks = asyncio.all_tasks()

        async def send(message):
            nonlocal pending_read
            if message["type"] == "http.response.start" and stage == "header-send":
                raise failure
            if message.get("body") != _PING.encode():
                return
            assert transport.active_reads == 1
            pending_read = transport.read_tasks[-1]
            ping_sent.set()
            if stage == "heartbeat-send":
                raise failure
            if stage.endswith("-send"):
                await asyncio.Event().wait()
            else:
                monkeypatch.setattr(relay, "_RESPONSES_HEARTBEAT_INTERVAL_SECONDS", 600)

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        is_send_failure = stage in ("header-send", "heartbeat-send")
        task = asyncio.create_task(response(
            {"type": "http", "asgi": {"spec_version": "2.4" if is_send_failure else "2.3"}},
            receive, send,
        ))
        try:
            if is_send_failure:
                with pytest.raises(ClientDisconnect) as caught:
                    await asyncio.wait_for(task, timeout=1)
                assert caught.value.__context__ is failure
            else:
                await asyncio.wait_for(ping_sent.wait(), timeout=1)
                if stage.startswith("cancel-"):
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    disconnect.set()
                    await asyncio.wait_for(task, timeout=1)
            assert transport.close_calls == 1
            assert transport.active_reads == 0
            if stage == "header-send":
                assert transport.read_calls == 1
                assert pending_read is None
            else:
                assert pending_read.done()
                assert transport.cancelled_reads == 1
                assert transport.read_calls == 2
            assert asyncio.all_tasks() <= initial_tasks
            await response.body_iterator.aclose()
            assert transport.close_calls == 1
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await response.body_iterator.aclose()

    asyncio.run(scenario())
    if has_cleanup_error:
        assert "synthetic ASGI cleanup failure" in caplog.text


def test_downstream_backpressure_keeps_only_one_undelivered_translated_chunk(monkeypatch):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        release = asyncio.Event()
        ready = asyncio.Event()
        reads = []

        async def translated_events():
            reads.append("first")
            yield relay._format_sse("content_block_delta", {"text": "first"})
            await release.wait()
            reads.append("second")
            ready.set()
            yield relay._format_sse("content_block_delta", {"text": "second"})
            reads.append("third")
            yield relay._format_sse("content_block_delta", {"text": "third"})

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert _event(await anext(body))[1] == {"text": "first"}
            await asyncio.sleep(0)
            assert reads == ["first"]
            assert await anext(body) == _PING
            release.set()
            await asyncio.wait_for(ready.wait(), timeout=1)
            await asyncio.sleep(0)
            assert reads == ["first", "second"]
            assert _event(await anext(body))[1] == {"text": "second"}
            await asyncio.sleep(0)
            assert reads == ["first", "second"]
            assert _event(await anext(body))[1] == {"text": "third"}
            with pytest.raises(StopAsyncIteration):
                await anext(body)
        finally:
            await body.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("should_close_directly", [False, True], ids=["consume-end", "close-after-error"])
def test_translated_api_error_stays_primary_when_iterator_cleanup_fails(
    monkeypatch, caplog, should_close_directly,
):
    clock = ManualClock()
    clock.install(monkeypatch)

    async def scenario():
        error = relay._format_sse("error", {
            "type": "error", "error": {"type": "api_error", "message": "synthetic primary API failure"},
        })

        async def translated_events():
            try:
                yield error
            finally:
                raise RuntimeError("synthetic error-stream cleanup failure")

        body = relay._with_responses_heartbeats(translated_events())
        try:
            assert await anext(body) == error
            if should_close_directly:
                await body.aclose()
            else:
                with pytest.raises(StopAsyncIteration):
                    await anext(body)
            assert clock.now == 0
        finally:
            await body.aclose()

    asyncio.run(scenario())
    assert "responses stream cleanup failed" in caplog.text
    assert "synthetic error-stream cleanup failure" in caplog.text
