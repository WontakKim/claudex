"""Tests for structural provider transports and route backend bindings."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import fields
from typing import Any, get_args, get_origin, get_type_hints

import httpx
import pytest
from starlette.requests import Request

from claudex.providers.backends import (
    AnthropicBackend,
    AnthropicErrorPolicy,
    AnthropicHeaderPolicy,
    AnthropicMessagesTransport,
    AnthropicTokenCounter,
    ResponsesBackend,
    ResponsesPayloadAdapter,
    ResponsesTransport,
    RouteBackend,
    WireKind,
)
from claudex.providers.codex_client import CodexClient
from claudex.providers.grok_client import GrokClient
from claudex.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex.providers.openai_compatible_client import OpenAICompatibleClient
from claudex.relay.kimi import _kimi_request_headers, _kimi_upstream_error_to_claude


class _ResponsesTransport:
    def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        async def events() -> AsyncIterator[dict[str, Any]]:
            yield {"payload": payload, "session_id": session_id}

        return events()

    async def list_models(self) -> list[str]:
        return ["responses-model"]

    async def context_window(self, model: str) -> int | None:
        return 128_000 if model == "responses-model" else None


class _OpenResponseStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"data"


class _AnthropicMessagesTransport:
    async def send_messages(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-body-size": str(len(body)), **headers},
            stream=_OpenResponseStream(),
        )

    async def list_models(self) -> Any:
        return {"data": [{"id": "messages-model"}]}


async def _adapt_payload(
    payload: dict[str, Any], model: str
) -> dict[str, Any]:
    return {**payload, "model": model}


def _header_policy(request: Request) -> dict[str, str]:
    return {"x-request-path": request.url.path}


def _error_policy(error: KimiUpstreamError) -> tuple[int, dict[str, Any]]:
    return error.status_code, {"error": {"message": error.body}}


async def _count_tokens(body: bytes, headers: dict[str, str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"input_tokens": len(body), "header_count": len(headers)},
    )


def _parameter_shape(
    candidate: Callable[..., Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (parameter.name, parameter.kind)
        for parameter in inspect.signature(candidate).parameters.values()
    )


def _callable_contract(alias: Any) -> tuple[tuple[Any, ...], Any]:
    parameter_types, return_type = get_args(alias)
    return tuple(parameter_types), return_type


def test_wire_kind_is_closed_to_the_two_supported_wire_protocols() -> None:
    assert list(WireKind) == [
        WireKind.RESPONSES,
        WireKind.ANTHROPIC_MESSAGES,
    ]
    assert WireKind.RESPONSES.value == "responses"
    assert WireKind.ANTHROPIC_MESSAGES.value == "anthropic_messages"


def test_backend_type_determines_wire_kind_without_redundant_instance_field() -> None:
    responses = ResponsesBackend(_ResponsesTransport(), _adapt_payload, None)
    anthropic = AnthropicBackend(
        _AnthropicMessagesTransport(),
        _header_policy,
        _error_policy,
        _count_tokens,
    )

    assert [field.name for field in fields(ResponsesBackend)] == [
        "transport",
        "adapt_payload",
        "signature_namespace",
    ]
    assert [field.name for field in fields(AnthropicBackend)] == [
        "transport",
        "header_policy",
        "error_policy",
        "token_counter",
    ]
    assert responses.wire_kind is WireKind.RESPONSES
    assert anthropic.wire_kind is WireKind.ANTHROPIC_MESSAGES
    assert set(get_args(RouteBackend)) == {ResponsesBackend, AnthropicBackend}


def test_backend_fields_retain_the_declared_callable_contracts() -> None:
    responses_hints = get_type_hints(ResponsesBackend)
    anthropic_hints = get_type_hints(AnthropicBackend)

    assert responses_hints["transport"] is ResponsesTransport
    assert responses_hints["adapt_payload"] == ResponsesPayloadAdapter
    assert responses_hints["signature_namespace"] == str | None
    assert anthropic_hints["transport"] is AnthropicMessagesTransport
    assert anthropic_hints["header_policy"] == AnthropicHeaderPolicy
    assert anthropic_hints["error_policy"] == AnthropicErrorPolicy
    assert anthropic_hints["token_counter"] == AnthropicTokenCounter


def test_binding_callables_follow_the_existing_relay_call_shapes() -> None:
    async def scenario() -> None:
        responses = ResponsesBackend(_ResponsesTransport(), _adapt_payload, "custom")
        payload = await responses.adapt_payload({"input": []}, "responses-model")
        events = [
            event
            async for event in responses.transport.stream_responses(payload, "session-1")
        ]
        assert events == [
            {
                "payload": {"input": [], "model": "responses-model"},
                "session_id": "session-1",
            }
        ]

        anthropic = AnthropicBackend(
            _AnthropicMessagesTransport(),
            _header_policy,
            _error_policy,
            _count_tokens,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/messages",
                "headers": [],
            }
        )
        headers = anthropic.header_policy(request)
        response = await anthropic.transport.send_messages(b"{}", headers)
        assert not response.is_closed
        assert response.headers["x-request-path"] == "/v1/messages"
        await response.aclose()

        status_code, error_body = anthropic.error_policy(
            KimiUpstreamError(429, "rate limited")
        )
        assert status_code == 429
        assert error_body == {"error": {"message": "rate limited"}}

        token_response = await anthropic.token_counter(b"{}", headers)
        assert token_response.json() == {"input_tokens": 2, "header_count": 1}

    asyncio.run(scenario())


def test_kimi_binding_candidates_match_declared_policy_contracts() -> None:
    header_parameters, header_return = _callable_contract(AnthropicHeaderPolicy)
    assert header_parameters == (Request,)
    assert header_return == dict[str, str]
    assert _parameter_shape(_kimi_request_headers) == (
        ("request", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    )
    assert not inspect.iscoroutinefunction(_kimi_request_headers)
    assert get_type_hints(_kimi_request_headers) == {
        "request": Request,
        "return": dict[str, str],
    }

    error_parameters, error_return = _callable_contract(AnthropicErrorPolicy)
    assert error_parameters == (KimiUpstreamError,)
    assert error_return == tuple[int, dict[str, Any]]
    assert _parameter_shape(_kimi_upstream_error_to_claude) == (
        ("exc", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    )
    assert not inspect.iscoroutinefunction(_kimi_upstream_error_to_claude)
    assert get_type_hints(_kimi_upstream_error_to_claude) == {
        "exc": KimiUpstreamError,
        "return": tuple[int, dict[str, Any]],
    }

    kimi_client = object.__new__(KimiClient)
    backend = AnthropicBackend(
        transport=kimi_client,
        header_policy=_kimi_request_headers,
        error_policy=_kimi_upstream_error_to_claude,
        token_counter=kimi_client.count_tokens,
    )
    assert backend.transport is kimi_client
    assert backend.header_policy is _kimi_request_headers
    assert backend.error_policy is _kimi_upstream_error_to_claude
    assert backend.token_counter.__self__ is kimi_client
    assert backend.token_counter.__func__ is KimiClient.count_tokens

    counter_parameters, counter_return = _callable_contract(AnthropicTokenCounter)
    assert counter_parameters == (bytes, dict[str, str])
    assert get_origin(counter_return) is Awaitable
    assert get_args(counter_return) == (httpx.Response,)
    bound_counter = backend.token_counter
    assert _parameter_shape(bound_counter) == (
        ("body", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("headers", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    )
    assert inspect.iscoroutinefunction(bound_counter)
    assert get_type_hints(bound_counter) == {
        "body": bytes,
        "headers": dict[str, str],
        "return": httpx.Response,
    }


@pytest.mark.parametrize(
    "client_type",
    [CodexClient, GrokClient, OpenAICompatibleClient],
)
def test_existing_responses_clients_match_the_structural_transport_shape(
    client_type: type[Any],
) -> None:
    assert _parameter_shape(client_type.stream_responses) == _parameter_shape(
        ResponsesTransport.stream_responses
    )
    assert inspect.isasyncgenfunction(client_type.stream_responses)
    stream_hints = get_type_hints(client_type.stream_responses)
    assert stream_hints == {
        "payload": dict[str, Any],
        "session_id": str,
        "return": AsyncIterator[dict[str, Any]],
    }
    assert _parameter_shape(client_type.list_models) == _parameter_shape(
        ResponsesTransport.list_models
    )
    assert inspect.iscoroutinefunction(client_type.list_models)
    assert get_type_hints(client_type.list_models)["return"] == list[str]
    assert _parameter_shape(client_type.context_window) == _parameter_shape(
        ResponsesTransport.context_window
    )
    assert inspect.iscoroutinefunction(client_type.context_window)
    assert get_type_hints(client_type.context_window) == {
        "model": str,
        "return": int | None,
    }


def test_existing_kimi_client_matches_the_open_response_transport_shape() -> None:
    assert _parameter_shape(KimiClient.send_messages) == _parameter_shape(
        AnthropicMessagesTransport.send_messages
    )
    assert inspect.iscoroutinefunction(KimiClient.send_messages)
    assert get_type_hints(KimiClient.send_messages) == {
        "body": bytes,
        "headers": dict[str, str],
        "return": httpx.Response,
    }
    assert tuple(inspect.signature(KimiClient.list_models).parameters) == ("self",)
    assert inspect.iscoroutinefunction(KimiClient.list_models)


def test_transport_protocols_remain_structural_only_and_wire_specific() -> None:
    assert not hasattr(ResponsesTransport, "send_messages")
    assert not hasattr(AnthropicMessagesTransport, "stream_responses")
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(_ResponsesTransport(), ResponsesTransport)
    with pytest.raises(TypeError, match="runtime_checkable"):
        isinstance(_AnthropicMessagesTransport(), AnthropicMessagesTransport)
