"""Tests for the Kimi relay client's auth overlay and retry behavior."""

from __future__ import annotations

import asyncio
import gzip
from collections.abc import AsyncIterator, Callable
from typing import Any

import anyio
import httpx
import pytest

from claudex.providers.kimi_auth import KimiCredentials
from claudex.providers.kimi_client import (
    KIMI_COUNT_TOKENS_URL,
    KIMI_MESSAGES_URL,
    KIMI_MODELS_URL,
    KimiClient,
    KimiUpstreamError,
)


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(
        self, body: bytes, on_close: Callable[[], None] | None = None
    ) -> None:
        self.body = body
        self.on_close = on_close
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.on_close is not None:
            self.on_close()


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


def test_401_closes_first_response_before_refresh_and_retries_once() -> None:
    events: list[str] = []
    streams: list[_TrackingStream] = []

    class OrderingAuthManager(_FakeAuthManager):
        async def get_credentials(
            self, force_refresh: bool = False
        ) -> KimiCredentials:
            events.append(f"credentials:{force_refresh}")
            return await super().get_credentials(force_refresh)

    auth_manager = OrderingAuthManager()
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        attempts.append(authorization)
        attempt = len(attempts)
        events.append(f"send:{attempt}")
        stream = _TrackingStream(
            b'{"error":"unauthorized"}' if attempt == 1 else b"{}",
            lambda: events.append(f"close:{attempt}"),
        )
        streams.append(stream)
        return httpx.Response(401 if attempt == 1 else 200, stream=stream)

    async def scenario() -> int:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            response = await KimiClient(auth_manager, http_client).send_messages(b"{}", {})
            await response.aclose()
            return response.status_code

    assert _run(scenario()) == 200
    assert auth_manager.calls == [False, True]
    assert attempts == ["Bearer token-1", "Bearer token-2"]
    assert events == [
        "credentials:False",
        "send:1",
        "close:1",
        "credentials:True",
        "send:2",
        "close:2",
    ]
    assert [stream.close_calls for stream in streams] == [1, 1]


def test_persistent_401_raises_after_single_retry() -> None:
    auth_manager = _FakeAuthManager()
    streams: list[_TrackingStream] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        stream = _TrackingStream(b'{"error":"unauthorized"}')
        streams.append(stream)
        return httpx.Response(401, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiClient(auth_manager, http_client).send_messages(b"{}", {})

    with pytest.raises(KimiUpstreamError) as excinfo:
        _run(scenario())

    assert excinfo.value.status_code == 401
    assert auth_manager.calls == [False, True]
    assert [stream.close_calls for stream in streams] == [1, 1]


def test_non_200_raises_with_body() -> None:
    stream = _TrackingStream(b'{"type":"error"}')

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await KimiClient(_FakeAuthManager(), http_client).send_messages(b"{}", {})

    with pytest.raises(KimiUpstreamError) as excinfo:
        _run(scenario())

    assert excinfo.value.status_code == 429
    assert excinfo.value.body == '{"type":"error"}'
    assert stream.close_calls == 1


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


@pytest.mark.parametrize("method_name", ["send_messages", "count_tokens"])
@pytest.mark.parametrize("status_code", [401, 429])
@pytest.mark.parametrize("is_after_retry", [False, True], ids=["initial", "after-refresh"])
@pytest.mark.parametrize(
    "cleanup_error_type", [None, RuntimeError, httpx.ReadError],
    ids=["close-ok", "close-error", "close-http-error"],
)
def test_rejected_decode_failure_closes_before_propagating(
    method_name: str, status_code: int, is_after_retry: bool,
    cleanup_error_type: type[Exception] | None, caplog: pytest.LogCaptureFixture,
) -> None:
    auth_manager = _FakeAuthManager()

    def on_close() -> None:
        if cleanup_error_type is not None:
            raise cleanup_error_type("rejected response close failed")

    stream = _TrackingStream(b"\x00\x00", on_close=on_close)
    read_errors: list[httpx.DecodingError] = []
    requests: list[httpx.Request] = []

    class RecordingResponse(httpx.Response):
        async def aread(self) -> bytes:
            try:
                return await super().aread()
            except httpx.DecodingError as exc:
                read_errors.append(exc)
                raise

    response = RecordingResponse(
        status_code, headers={"content-encoding": "gzip"}, stream=stream
    )
    assert not response.is_closed
    assert not hasattr(response, "_content")
    first_stream = _TrackingStream(gzip.compress(b'{"error":"expired"}')) if is_after_retry else None
    first_response = (
        httpx.Response(401, headers={"content-encoding": "gzip"}, stream=first_stream)
        if first_stream is not None else None
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if first_response is not None and len(requests) == 1:
            return first_response
        return response

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = KimiClient(auth_manager, http_client)
            try:
                with pytest.raises(httpx.DecodingError) as excinfo:
                    await getattr(client, method_name)(b"{}", {})
                assert len(read_errors) == 1
                assert read_errors[0] is excinfo.value
                assert stream.close_calls == 1
                assert response.is_closed
                assert auth_manager.calls == ([False, True] if is_after_retry else [False])
                assert len(requests) == (2 if is_after_retry else 1)
                if first_response is not None:
                    assert first_response.is_closed
                    assert first_stream is not None and first_stream.close_calls == 1
                cleanup_warnings = [
                    record for record in caplog.records
                    if record.getMessage() == "Kimi rejected response cleanup failed"
                ]
                assert len(cleanup_warnings) == (0 if cleanup_error_type is None else 1)
            finally:
                await response.aclose()
                if first_response is not None:
                    await first_response.aclose()

    _run(scenario())


@pytest.mark.parametrize("method_name", ["send_messages", "count_tokens"])
@pytest.mark.parametrize("status_code", [401, 429], ids=["first-401", "terminal-429"])
@pytest.mark.parametrize("has_cleanup_error", [False, True], ids=["close-ok", "close-error"])
def test_rejected_read_cancellation_closes_before_propagating(
    method_name: str, status_code: int, has_cleanup_error: bool
) -> None:
    async def scenario() -> None:
        auth_manager = _FakeAuthManager()
        read_started = anyio.Event()
        never_finish_read = anyio.Event()
        scopes: list[anyio.CancelScope] = []
        primary_errors: list[asyncio.CancelledError] = []
        propagated_errors: list[asyncio.CancelledError] = []
        cleanup_seen: list[bool] = []
        outcomes: list[httpx.Response] = []
        requests: list[httpx.Request] = []

        class BlockingStream(_TrackingStream):
            def __init__(self) -> None:
                super().__init__(b"{}")
                self.is_close_complete = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                read_started.set()
                try:
                    await never_finish_read.wait()
                except asyncio.CancelledError as exc:
                    primary_errors.append(exc)
                    raise
                yield self.body

            async def aclose(self) -> None:
                await super().aclose()
                await anyio.sleep(0)
                self.is_close_complete = True
                if has_cleanup_error:
                    raise RuntimeError("rejected response close failed")

        stream = BlockingStream()
        response = httpx.Response(status_code, stream=stream)

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return response

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = KimiClient(auth_manager, http_client)

            async def read_request() -> None:
                with anyio.CancelScope() as scope:
                    scopes.append(scope)
                    try:
                        outcomes.append(await getattr(client, method_name)(b"{}", {}))
                    except asyncio.CancelledError as exc:
                        propagated_errors.append(exc)
                        cleanup_seen.append(stream.is_close_complete)
                        raise

            try:
                with anyio.fail_after(2):
                    async with anyio.create_task_group() as group:
                        group.start_soon(read_request)
                        await read_started.wait()
                        scopes[0].cancel()
                assert cleanup_seen == [True]
                assert len(propagated_errors) == len(primary_errors) == 1
                assert propagated_errors[0] is primary_errors[0]
                assert outcomes == []
                assert response.is_closed
                assert stream.close_calls == 1
                assert auth_manager.calls == [False]
                assert len(requests) == 1
            finally:
                await response.aclose()
                if not stream.is_close_complete:
                    if has_cleanup_error:
                        with pytest.raises(RuntimeError, match="rejected response close failed"):
                            await stream.aclose()
                    else:
                        await stream.aclose()

    _run(scenario())


@pytest.mark.parametrize("method_name", ["send_messages", "count_tokens"])
@pytest.mark.parametrize("final_status_code", [200, 401, 429, 503])
def test_readable_rejections_preserve_retry_and_open_response_handoff(
    method_name: str, final_status_code: int
) -> None:
    events: list[str] = []
    streams: list[_TrackingStream] = []
    responses: list[httpx.Response] = []
    requests: list[httpx.Request] = []
    success_body = b'{"input_tokens":12}' if method_name == "count_tokens" else b'{"id":"msg"}'
    rejection_body = b'{"error":"readable rejection"}'

    class OrderingAuthManager(_FakeAuthManager):
        async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
            if force_refresh:
                assert streams[0].close_calls == 1
                assert responses[0].is_closed
            events.append(f"credentials:{force_refresh}")
            return await super().get_credentials(force_refresh)

    auth_manager = OrderingAuthManager()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        attempt = len(requests)
        events.append(f"send:{attempt}")
        body = b'{"error":"expired"}' if attempt == 1 else (
            success_body if final_status_code == 200 else rejection_body
        )
        stream = _TrackingStream(gzip.compress(body), lambda: events.append(f"close:{attempt}"))
        streams.append(stream)
        response = httpx.Response(
            401 if attempt == 1 else final_status_code, stream=stream,
            headers={"content-encoding": "gzip", "x-native-response": "unchanged"},
        )
        responses.append(response)
        return response

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = KimiClient(auth_manager, http_client)
            if final_status_code == 200:
                response = await getattr(client, method_name)(b"{}", {})
                try:
                    assert response is responses[1]
                    assert not response.is_closed
                    assert not response.is_stream_consumed
                    assert streams[1].close_calls == 0
                    assert response.headers["x-native-response"] == "unchanged"
                    assert await response.aread() == success_body
                finally:
                    await response.aclose()
            else:
                with pytest.raises(KimiUpstreamError) as excinfo:
                    await getattr(client, method_name)(b"{}", {})
                assert excinfo.value.status_code == final_status_code
                assert excinfo.value.body == rejection_body.decode()
            assert auth_manager.calls == [False, True]
            assert [request.headers["Authorization"] for request in requests] == [
                "Bearer token-1", "Bearer token-2"
            ]
            expected_url = KIMI_MESSAGES_URL if method_name == "send_messages" else KIMI_COUNT_TOKENS_URL
            assert all(str(request.url) == f"{expected_url}?beta=true" for request in requests)
            assert all(request.content == b"{}" for request in requests)
            assert [stream.close_calls for stream in streams] == [1, 1]
            assert all(response.is_closed for response in responses)
            assert events == [
                "credentials:False", "send:1", "close:1",
                "credentials:True", "send:2", "close:2",
            ]

    _run(scenario())
