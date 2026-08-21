"""Tests for static Anthropic Messages-compatible transport and policies."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator
from typing import Any, get_type_hints

import httpx
import pytest
from starlette.requests import Request

from claudex.config import AnthropicCompatibleProvider
from claudex.providers.anthropic_compatible_client import (
    AnthropicCompatibleClient,
    AnthropicCompatibleUpstreamError,
)
from claudex.providers.backends import (
    AnthropicBackend,
    AnthropicMessagesTransport,
    AnthropicStreamReadFailure,
)
from claudex.relay.anthropic_backend import _relay_via_anthropic_backend
from claudex.relay.anthropic_compatible import (
    _anthropic_compatible_error_to_claude,
    _anthropic_compatible_request_headers,
)
from claudex.upstream_errors import UpstreamAuthError, UpstreamError

_PROVIDER_NAME = "messages-api"
_BASE_URL = "https://messages.example/api/v1"
_API_KEY = "static-anthropic-secret"


def _provider(
    *, base_url: str = _BASE_URL, api_key: str = _API_KEY
) -> AnthropicCompatibleProvider:
    return AnthropicCompatibleProvider(base_url=base_url, api_key=api_key)


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": headers,
        }
    )


class _TrackedByteStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        error: httpx.HTTPError | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self.read_calls = 0
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.read_calls += 1
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.close_calls += 1


class _CloseFailingByteStream(_TrackedByteStream):
    def __init__(self, chunks: list[bytes], close_error: BaseException) -> None:
        super().__init__(chunks)
        self._close_error = close_error

    async def aclose(self) -> None:
        self.close_calls += 1
        raise self._close_error


def test_send_messages_uses_only_the_versioned_messages_url_and_static_bearer() -> None:
    captured: list[httpx.Request] = []
    stream = _TrackedByteStream([b'{"type":"message"}'])

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            assert _API_KEY not in repr(client)
            response = await client.send_messages(
                b'{"model":"upstream-model"}',
                {
                    "Authorization": "Bearer caller-token",
                    "authorization": "Bearer duplicate-caller-token",
                    "X-API-Key": "caller-api-key",
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
            assert response.is_closed is False
            assert stream.read_calls == 0
            assert stream.close_calls == 0
            await response.aclose()
            assert stream.close_calls == 1

    asyncio.run(scenario())

    (request,) = captured
    assert request.method == "POST"
    assert str(request.url) == f"{_BASE_URL}/messages"
    assert request.headers["authorization"] == f"Bearer {_API_KEY}"
    assert "x-api-key" not in request.headers
    assert request.headers["content-type"] == "application/json"
    assert captured[0].content == b'{"model":"upstream-model"}'
    assert not any(
        endpoint in str(request.url)
        for endpoint in ("/models", "/messages/count_tokens")
    )


def test_direct_constructor_normalizes_trailing_base_url_slashes() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(204, stream=_TrackedByteStream([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME,
                _provider(base_url=f"{_BASE_URL}///"),
                http_client,
            )
            response = await client.send_messages(b"{}", {})
            await response.aclose()

    asyncio.run(scenario())

    assert [str(request.url) for request in captured] == [f"{_BASE_URL}/messages"]


@pytest.mark.parametrize(
    ("base_url", "suffix"),
    [
        (f"{_BASE_URL}?tenant=one", "query"),
        (f"{_BASE_URL}#deployment", "fragment"),
    ],
)
def test_direct_constructor_rejects_base_url_suffix_semantics(
    base_url: str, suffix: str
) -> None:
    http_client = httpx.AsyncClient()
    try:
        with pytest.raises(
            ValueError,
            match=(
                "base_url must be a versioned API prefix without a query or "
                f"fragment; found {suffix}"
            ),
        ):
            AnthropicCompatibleClient(
                _PROVIDER_NAME,
                _provider(base_url=base_url),
                http_client,
            )
    finally:
        asyncio.run(http_client.aclose())


def test_static_header_policy_removes_credentials_and_hop_by_hop_headers() -> None:
    headers = _anthropic_compatible_request_headers(
        _request(
            [
                (b"Authorization", b"Bearer caller-token"),
                (b"X-API-Key", b"caller-key"),
                (b"Host", b"gateway.local"),
                (b"Content-Length", b"999"),
                (b"Connection", b"keep-alive"),
                (b"Accept-Encoding", b"gzip"),
                (b"Transfer-Encoding", b"chunked"),
                (b"content-type", b"application/json"),
                (b"x-safe-diagnostic", b"request-1"),
            ]
        )
    )

    assert headers == {
        "content-type": "application/json",
        "x-safe-diagnostic": "request-1",
        "anthropic-version": "2023-06-01",
    }


def test_static_header_policy_aggregates_betas_and_all_connection_nominations() -> None:
    headers = _anthropic_compatible_request_headers(
        _request(
            [
                (b"Connection", b"X-Remove-First, Keep-Alive"),
                (b"connection", b"x-remove-second, TE"),
                (b"X-Remove-First", b"secret-one"),
                (b"x-remove-second", b"secret-two"),
                (b"Keep-Alive", b"timeout=5"),
                (b"Proxy-Authenticate", b"challenge"),
                (b"Proxy-Authorization", b"proxy-secret"),
                (b"TE", b"trailers"),
                (b"Trailer", b"x-checksum"),
                (b"Transfer-Encoding", b"chunked"),
                (b"Upgrade", b"websocket"),
                (b"Anthropic-Beta", b"first-beta, oauth-2025-04-20"),
                (b"anthropic-beta", b"second-beta"),
                (b"ANTHROPIC-BETA", b"oauth-2025-04-20, third-beta"),
                (b"X-Safe", b"preserved"),
            ]
        )
    )

    assert headers == {
        "x-safe": "preserved",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "first-beta,second-beta,third-beta",
    }


def test_static_header_policy_preserves_anthropic_version() -> None:
    headers = _anthropic_compatible_request_headers(
        _request([(b"Anthropic-Version", b"2025-01-01")])
    )

    assert headers["anthropic-version"] == "2025-01-01"


@pytest.mark.parametrize(
    ("raw_beta", "expected_beta"),
    [
        ("oauth-2025-04-20", None),
        (
            "claude-code-20250219, oauth-2025-04-20, interleaved-thinking-2025-05-14",
            "claude-code-20250219,interleaved-thinking-2025-05-14",
        ),
        ("claude-code-20250219", "claude-code-20250219"),
    ],
)
def test_static_header_policy_removes_only_the_oauth_beta(
    raw_beta: str, expected_beta: str | None
) -> None:
    headers = _anthropic_compatible_request_headers(
        _request([(b"anthropic-beta", raw_beta.encode())])
    )

    assert headers.get("anthropic-beta") == expected_beta
    assert "oauth-2025-04-20" not in headers.get("anthropic-beta", "")


@pytest.mark.parametrize("status_code", [401, 403, 429, 500])
def test_non_success_response_is_consumed_closed_once_and_not_retried(
    status_code: int,
) -> None:
    calls = 0
    stream = _TrackedByteStream(
        [f"upstream rejected {_API_KEY}".encode()]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            await client.send_messages(b"{}", {})

    with pytest.raises(AnthropicCompatibleUpstreamError) as exc_info:
        asyncio.run(scenario())

    error = exc_info.value
    assert isinstance(error, UpstreamError)
    assert error.status_code == status_code
    assert error.body == "upstream rejected [REDACTED]"
    assert calls == 1
    assert stream.read_calls == 1
    assert stream.close_calls == 1
    assert _API_KEY not in error.body
    assert _API_KEY not in str(error)
    assert _API_KEY not in repr(error)


def test_non_success_body_read_failure_closes_once_and_propagates() -> None:
    failure = httpx.ReadError(f"response body interrupted with {_API_KEY}")
    stream = _TrackedByteStream([], failure)
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            await client.send_messages(b"{}", {})

    with pytest.raises(httpx.ReadError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value is failure
    assert _API_KEY not in str(exc_info.value)
    assert _API_KEY not in repr(exc_info.value)
    assert calls == 1
    assert stream.read_calls == 1
    assert stream.close_calls == 1


def test_non_success_close_failure_keeps_primary_status_and_closes_once() -> None:
    close_failure = RuntimeError(f"close failed with {_API_KEY}")
    stream = _CloseFailingByteStream([b"upstream failed"], close_failure)
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, stream=stream)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            await client.send_messages(b"{}", {})

    with pytest.raises(AnthropicCompatibleUpstreamError) as exc_info:
        asyncio.run(scenario())

    error = exc_info.value
    assert error.status_code == 503
    assert error.body == "upstream failed"
    assert calls == 1
    assert stream.read_calls == 1
    assert stream.close_calls == 1
    assert error.__cause__ is close_failure
    representations = (str(error), repr(error), str(close_failure), repr(close_failure))
    assert all(_API_KEY not in representation for representation in representations)


def test_network_http_error_category_is_preserved_without_retry() -> None:
    failure = httpx.ConnectError(f"connection refused with {_API_KEY}")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            await client.send_messages(b"{}", {})

    with pytest.raises(httpx.ConnectError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value is failure
    assert _API_KEY not in str(exc_info.value)
    assert _API_KEY not in repr(exc_info.value)
    assert calls == 1


@pytest.mark.parametrize(
    "api_key",
    [
        pytest.param('quote"-backslash\\-control\n\t', id="quotes-controls"),
        pytest.param("unicode-눈-☃", id="unicode-escapes"),
        pytest.param("regex-.*+$?[](){}|^", id="regex-specials"),
        pytest.param("123456789", id="numeric-literal"),
    ],
)
def test_api_key_redaction_covers_raw_json_and_http_exception_forms(
    api_key: str,
) -> None:
    http_client = httpx.AsyncClient()
    client = AnthropicCompatibleClient(
        _PROVIDER_NAME, _provider(api_key=api_key), http_client
    )
    raw_encoded_key = json.dumps(api_key, ensure_ascii=True)[1:-1]
    values = [
        client._redact_api_key(f"literal:{api_key}"),
        client._redact_api_key(f"encoded:{raw_encoded_key}"),
        client._redact_api_key(
            json.dumps(
                {api_key: {"message": f"decoded:{api_key}"}},
                ensure_ascii=True,
            )
        ),
    ]
    failure = httpx.ConnectError(
        json.dumps({"message": api_key}, ensure_ascii=True)
    )
    sanitized_failure = client._sanitize_http_error(failure)
    asyncio.run(http_client.aclose())

    representations = [*values, str(sanitized_failure), repr(sanitized_failure)]
    for representation in representations:
        assert api_key not in representation
        assert raw_encoded_key not in representation
    parsed = json.loads(values[2])
    assert list(parsed) == ["[REDACTED]"]
    assert parsed["[REDACTED]"]["message"] == "decoded:[REDACTED]"
    assert isinstance(sanitized_failure, httpx.HTTPError)


def test_non_json_python_escape_forms_are_redacted_end_to_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = 'quote"-single\'-backslash\\-newline\n-tab\t-control\x01-unicode-눈-☃'
    provider_name = f"unsafe\n{api_key}\tprovider"
    python_forms = [
        api_key,
        api_key.encode("unicode_escape").decode("ascii"),
        repr(api_key),
        ascii(api_key),
        repr(api_key.encode("utf-8")),
        repr(f"Bearer {api_key}".encode("utf-8")),
    ]
    upstream_body = "diagnostic " + " | ".join(python_forms)
    response = httpx.Response(500, content=upstream_body.encode())
    http_client = httpx.AsyncClient()

    def build_request(
        method: str,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
    ) -> httpx.Request:
        assert headers["Authorization"] == f"Bearer {api_key}"
        return httpx.Request(method, url, content=content)

    async def send(request: httpx.Request, *, stream: bool) -> httpx.Response:
        assert stream is True
        return response

    http_client.build_request = build_request
    http_client.send = send
    client = AnthropicCompatibleClient(
        provider_name, _provider(api_key=api_key), http_client
    )

    async def scenario() -> AnthropicCompatibleUpstreamError:
        try:
            await client.send_messages(b"{}", {})
        except AnthropicCompatibleUpstreamError as exc:
            return exc
        raise AssertionError("the non-success response was returned as success")

    try:
        error = asyncio.run(scenario())
    finally:
        asyncio.run(http_client.aclose())
    caplog.set_level(logging.WARNING, logger="claudex.server")
    status_code, result = _anthropic_compatible_error_to_claude(error)

    assert status_code == 500
    representations = [
        error.body,
        str(error),
        repr(error),
        repr(result),
        caplog.text,
    ]
    for representation in representations:
        for secret_form in python_forms:
            assert secret_form not in representation
    assert api_key not in error.body
    assert api_key not in str(error)
    assert api_key not in repr(error)


def test_local_protocol_bytes_diagnostic_is_redacted_without_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = 'quote"-backslash\\-newline\n-tab\t-control\x02-unicode-눈-☃'
    diagnostic = f"Illegal header value {f'Bearer {api_key}'.encode('utf-8')!r}"
    failure = httpx.LocalProtocolError(diagnostic)
    calls = 0
    http_client = httpx.AsyncClient()

    def build_request(
        method: str,
        url: str,
        *,
        content: bytes,
        headers: dict[str, str],
    ) -> httpx.Request:
        assert headers["Authorization"] == f"Bearer {api_key}"
        return httpx.Request(method, url, content=content)

    async def send(request: httpx.Request, *, stream: bool) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise failure

    http_client.build_request = build_request
    http_client.send = send
    client = AnthropicCompatibleClient(
        _PROVIDER_NAME, _provider(api_key=api_key), http_client
    )

    async def scenario() -> None:
        await client.send_messages(b"{}", {})

    try:
        with pytest.raises(httpx.LocalProtocolError) as exc_info:
            asyncio.run(scenario())
    finally:
        asyncio.run(http_client.aclose())

    error = exc_info.value
    caplog.set_level(logging.WARNING, logger="claudex.server")
    status_code, result = _anthropic_compatible_error_to_claude(error)
    secret_forms = {
        api_key,
        api_key.encode("unicode_escape").decode("ascii"),
        repr(api_key),
        ascii(api_key),
        repr(api_key.encode("utf-8")),
        repr(f"Bearer {api_key}".encode("utf-8")),
    }

    assert error is failure
    assert isinstance(error, httpx.HTTPError)
    assert calls == 1
    assert status_code == 502
    for representation in (str(error), repr(error), repr(result), caplog.text):
        for secret_form in secret_forms:
            assert secret_form not in representation


def test_network_failure_becomes_502_through_the_generic_relay() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused")

    async def scenario() -> tuple[int, dict[str, Any]]:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            backend = AnthropicBackend(
                transport=client,
                header_policy=_anthropic_compatible_request_headers,
                error_policy=_anthropic_compatible_error_to_claude,
            )
            response = await _relay_via_anthropic_backend(
                _request([]),
                {"model": "requested-model", "messages": []},
                "upstream-model",
                backend,
            )
            return response.status_code, json.loads(response.body)

    status_code, body = asyncio.run(scenario())

    assert status_code == 502
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert calls == 1


def test_oversized_numeric_error_falls_back_with_status_and_redaction() -> None:
    oversized_number = "9" * 5000
    upstream_body = (
        '{"type":"error","error":{"type":"api_error",'
        f'"message":"rejected {_API_KEY}"}},"number":{oversized_number}}}'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, text=upstream_body)

    async def scenario() -> AnthropicCompatibleUpstreamError:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            try:
                await client.send_messages(b"{}", {})
            except AnthropicCompatibleUpstreamError as exc:
                return exc
        raise AssertionError("the parser-limit response was returned as success")

    error = asyncio.run(scenario())
    status_code, body = _anthropic_compatible_error_to_claude(error)

    assert error.status_code == 529
    assert status_code == 529
    assert body["type"] == "error"
    assert body["error"]["type"] == "overloaded_error"
    assert _API_KEY not in error.body
    assert _API_KEY not in str(error)
    assert _API_KEY not in repr(error)
    assert _API_KEY not in repr(body)


def test_direct_constructor_escapes_unsafe_provider_label_formatting() -> None:
    unsafe_name = "unsafe\nprovider\t\x1b[31m"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="failed")

    async def scenario() -> AnthropicCompatibleUpstreamError:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                unsafe_name, _provider(), http_client
            )
            try:
                await client.send_messages(b"{}", {})
            except AnthropicCompatibleUpstreamError as exc:
                return exc
        raise AssertionError("the non-success response was returned as success")

    error = asyncio.run(scenario())

    assert "\n" not in str(error).replace("\\n", "")
    assert "\t" not in str(error).replace("\\t", "")
    assert "\x1b" not in str(error).replace("\\x1b", "")
    assert "unsafe\\nprovider\\t\\x1b[31m upstream returned 500" in str(error)


def test_provider_label_cannot_repeat_the_configured_api_key() -> None:
    api_key = "same-provider-label-and-key"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="failed")

    async def scenario() -> AnthropicCompatibleUpstreamError:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                api_key, _provider(api_key=api_key), http_client
            )
            try:
                await client.send_messages(b"{}", {})
            except AnthropicCompatibleUpstreamError as exc:
                return exc
        raise AssertionError("the non-success response was returned as success")

    error = asyncio.run(scenario())

    assert api_key not in str(error)
    assert api_key not in repr(error)


def test_safe_anthropic_error_body_and_status_are_preserved() -> None:
    upstream_body = {
        "type": "error",
        "error": {
            "type": "overloaded_error",
            "message": "capacity unavailable",
        },
        "request_id": "request-1",
    }

    status_code, body = _anthropic_compatible_error_to_claude(
        AnthropicCompatibleUpstreamError(529, json.dumps(upstream_body), "safe-name")
    )

    assert status_code == 529
    assert body == upstream_body


@pytest.mark.parametrize(
    ("raw_body", "expected_message"),
    [
        ("plain upstream failure", "plain upstream failure"),
        (
            '{"type":"message","value":"unexpected"}',
            '{"type":"message","value":"unexpected"}',
        ),
        (
            '{"type":"error","error":{"message":"missing error type"}}',
            "missing error type",
        ),
        (
            '{"type":"error","error":{"type":"api_error",'
            '"message":"unsafe number"},"value":NaN}',
            "unsafe number",
        ),
    ],
)
def test_non_anthropic_error_bodies_are_wrapped_safely(
    raw_body: str, expected_message: str
) -> None:
    status_code, body = _anthropic_compatible_error_to_claude(
        AnthropicCompatibleUpstreamError(500, raw_body, _PROVIDER_NAME)
    )

    assert status_code == 500
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": expected_message},
    }


def test_auth_failure_names_configured_credentials_without_login_guidance() -> None:
    upstream = AnthropicCompatibleUpstreamError(
        401,
        '{"type":"error","error":{"type":"authentication_error",'
        '"message":"run kimi login"}}',
        _PROVIDER_NAME,
    )

    status_code, body = _anthropic_compatible_error_to_claude(upstream)
    serialized = json.dumps(body).lower()

    assert status_code == 401
    assert body["error"]["type"] == "authentication_error"
    assert "configured" in serialized
    assert "api key" in serialized
    assert "kimi" not in serialized
    assert "login" not in serialized


def test_network_and_stream_failures_map_to_502_without_secret_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="claudex.server")
    network_failure = httpx.ConnectError(f"failed with {_API_KEY}")
    stream_failure = AnthropicStreamReadFailure(
        httpx.ReadError(f"stream failed with {_API_KEY}")
    )

    network_status, network_body = _anthropic_compatible_error_to_claude(
        network_failure
    )
    stream_status, stream_body = _anthropic_compatible_error_to_claude(
        stream_failure
    )

    assert network_status == 502
    assert stream_status == 502
    assert network_body["error"]["type"] == "api_error"
    assert stream_body["error"]["type"] == "api_error"
    assert _API_KEY not in repr(network_body)
    assert _API_KEY not in repr(stream_body)
    assert _API_KEY not in caplog.text


def test_redacted_upstream_error_stays_redacted_in_policy_logs_and_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream_body = {
        "type": "error",
        "error": {"type": "api_error", "message": f"invalid {_API_KEY}"},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=upstream_body)

    async def scenario() -> AnthropicCompatibleUpstreamError:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            try:
                await client.send_messages(b"{}", {})
            except AnthropicCompatibleUpstreamError as exc:
                return exc
        raise AssertionError("the non-success response was returned as success")

    error = asyncio.run(scenario())
    caplog.set_level(logging.WARNING, logger="claudex.server")
    status_code, body = _anthropic_compatible_error_to_claude(error)

    assert status_code == 500
    assert body["error"]["message"] == "invalid [REDACTED]"
    representations = (error.body, str(error), repr(error), repr(body), caplog.text)
    assert all(_API_KEY not in representation for representation in representations)


def test_policies_and_client_match_existing_anthropic_backend_contracts() -> None:
    client_parameters = tuple(
        inspect.signature(AnthropicCompatibleClient.send_messages).parameters
    )
    protocol_parameters = tuple(
        inspect.signature(AnthropicMessagesTransport.send_messages).parameters
    )
    assert client_parameters == protocol_parameters
    assert get_type_hints(AnthropicCompatibleClient.send_messages) == {
        "body": bytes,
        "headers": dict[str, str],
        "return": httpx.Response,
    }
    assert tuple(
        inspect.signature(_anthropic_compatible_request_headers).parameters
    ) == ("request",)
    assert get_type_hints(_anthropic_compatible_request_headers) == {
        "request": Request,
        "return": dict[str, str],
    }
    assert tuple(
        inspect.signature(_anthropic_compatible_error_to_claude).parameters
    ) == ("exc",)
    assert get_type_hints(_anthropic_compatible_error_to_claude) == {
        "exc": (
            UpstreamAuthError
            | UpstreamError
            | httpx.HTTPError
            | AnthropicStreamReadFailure
        ),
        "return": tuple[int, dict[str, Any]],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, stream=_TrackedByteStream([]))

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AnthropicCompatibleClient(
                _PROVIDER_NAME, _provider(), http_client
            )
            backend = AnthropicBackend(
                transport=client,
                header_policy=_anthropic_compatible_request_headers,
                error_policy=_anthropic_compatible_error_to_claude,
            )
            assert backend.transport is client
            assert backend.token_counter is None
            assert backend.catalog_loader is None
            assert not hasattr(client, "count_tokens")
            assert not hasattr(client, "list_models")
            response = await backend.transport.send_messages(b"{}", {})
            assert response.status_code == 204
            assert response.is_closed is False
            await response.aclose()

    asyncio.run(scenario())
