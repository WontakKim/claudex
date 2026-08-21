"""Structural transport contracts and route backend bindings."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol, TypeAlias

import httpx
from starlette.requests import Request

from claudex.providers.kimi_client import KimiUpstreamError


class WireKind(Enum):
    """Wire protocols supported by mapped routes."""

    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


class ResponsesTransport(Protocol):
    """Structural contract shared by Responses-family provider clients.

    Catalog methods describe optional routing metadata. Their presence does
    not make a successful catalog lookup a prerequisite for inference.
    """

    def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Return an event iterator whose upstream stream the caller must close."""
        ...

    async def list_models(self) -> list[str]: ...

    async def context_window(self, model: str) -> int | None: ...


class AnthropicMessagesTransport(Protocol):
    """Structural contract for native Anthropic Messages transports."""

    async def send_messages(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        """Return an open response whose ownership transfers to the caller."""
        ...

    async def list_models(self) -> Any: ...


ResponsesPayloadAdapter: TypeAlias = Callable[
    [dict[str, Any], str], Awaitable[dict[str, Any]]
]
ResponsesProbePayloadAdapter: TypeAlias = Callable[
    [dict[str, Any], str], dict[str, Any]
]
AnthropicHeaderPolicy: TypeAlias = Callable[[Request], dict[str, str]]
AnthropicErrorPolicy: TypeAlias = Callable[
    [KimiUpstreamError], tuple[int, dict[str, Any]]
]

# Native token counters transfer ownership of the returned response to the caller,
# matching the transport's send_messages contract.
AnthropicTokenCounter: TypeAlias = Callable[
    [bytes, dict[str, str]], Awaitable[httpx.Response]
]


@dataclass(frozen=True)
class ResponsesBackend:
    """Bound policies for a Responses-family route."""

    wire_kind: ClassVar[WireKind] = WireKind.RESPONSES

    transport: ResponsesTransport
    adapt_payload: ResponsesPayloadAdapter
    adapt_probe_payload: ResponsesProbePayloadAdapter
    signature_namespace: str | None


@dataclass(frozen=True)
class AnthropicBackend:
    """Bound policies for a native Anthropic Messages route."""

    wire_kind: ClassVar[WireKind] = WireKind.ANTHROPIC_MESSAGES

    transport: AnthropicMessagesTransport
    header_policy: AnthropicHeaderPolicy
    error_policy: AnthropicErrorPolicy
    token_counter: AnthropicTokenCounter


RouteBackend: TypeAlias = ResponsesBackend | AnthropicBackend
