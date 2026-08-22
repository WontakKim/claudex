"""Native Anthropic Messages backend relay serving path."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from claudex.providers.backends import AnthropicBackend, AnthropicStreamReadFailure
from claudex.relay.common import (
    _PASSTHROUGH_SKIP_RESPONSE_HEADERS,
    _format_sse,
)
from claudex.upstream_errors import UpstreamAuthError, UpstreamError

logger = logging.getLogger("claudex.server")


def _rewrite_message_start_data(line: str, requested_model: str) -> str:
    data = line[5:].strip()
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return line
    message = parsed.get("message") if isinstance(parsed, dict) else None
    if not isinstance(message, dict) or "model" not in message:
        return line
    message["model"] = requested_model
    return "data: " + json.dumps(parsed, ensure_ascii=False)


async def _rewrite_anthropic_sse(
    upstream_response: httpx.Response,
    requested_model: str,
    backend: AnthropicBackend,
) -> AsyncIterator[bytes]:
    """Relay complete Messages SSE events and restore the requested model.

    Owns upstream_response after iteration starts. `_AnthropicStreamRelay`
    covers the before-first-iteration close case and avoids double-closing it.
    """
    current_event = ""
    event_lines: list[str] = []
    try:
        async for line in upstream_response.aiter_lines():
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif current_event == "message_start" and line.startswith("data:"):
                line = _rewrite_message_start_data(line, requested_model)
            event_lines.append(line)
            if line:
                continue
            yield ("\n".join(event_lines) + "\n").encode()
            event_lines.clear()
            current_event = ""
            # Buffered upstream lines can otherwise cause repeated writes before
            # Starlette gets a turn to cancel this task on client disconnect.
            await asyncio.sleep(0)
        if event_lines:
            yield ("\n".join(event_lines) + "\n").encode()
    except httpx.HTTPError as exc:
        if event_lines:
            yield ("\n".join(event_lines) + "\n").encode()
            await asyncio.sleep(0)
        _, body = backend.error_policy(AnthropicStreamReadFailure(exc))
        yield _format_sse("error", body).encode()
    finally:
        await upstream_response.aclose()


class _AnthropicStreamRelay:
    """Own one open Messages response across every stream termination path."""

    def __init__(
        self,
        upstream_response: httpx.Response,
        requested_model: str,
        backend: AnthropicBackend,
    ) -> None:
        self._upstream_response = upstream_response
        self._chunks = _rewrite_anthropic_sse(
            upstream_response, requested_model, backend
        )
        self._closed = False

    def __aiter__(self) -> _AnthropicStreamRelay:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._chunks.__anext__()
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._chunks.aclose()
        if not self._upstream_response.is_closed:
            await self._upstream_response.aclose()


class _AnthropicStreamingResponse(StreamingResponse):
    """Close the owned Messages stream even before or between body reads."""

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            await self.body_iterator.aclose()


async def _relay_via_anthropic_backend(
    request: Request,
    claude_request: dict[str, Any],
    upstream_model: str,
    backend: AnthropicBackend,
) -> Response:
    """Relay a mapped Messages request through its bound native backend."""
    requested_model = str(claude_request.get("model", ""))
    outgoing = dict(claude_request)
    outgoing["model"] = upstream_model
    logger.info(
        "%s -> native Messages:%s (stream=%s, messages=%d)",
        requested_model or "?",
        upstream_model,
        bool(claude_request.get("stream")),
        len(claude_request.get("messages") or []),
    )

    try:
        upstream_response = await backend.transport.send_messages(
            json.dumps(outgoing, ensure_ascii=False).encode(),
            backend.header_policy(request),
        )
    except (UpstreamAuthError, UpstreamError, httpx.HTTPError) as exc:
        status_code, body = backend.error_policy(exc)
        return JSONResponse(body, status_code=status_code)

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_RESPONSE_HEADERS
    }
    if claude_request.get("stream"):
        return _AnthropicStreamingResponse(
            _AnthropicStreamRelay(upstream_response, requested_model, backend),
            headers=response_headers,
        )

    try:
        payload = await upstream_response.aread()
    finally:
        await upstream_response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return Response(payload, headers=response_headers)
    if isinstance(parsed, dict) and "model" in parsed:
        parsed["model"] = requested_model
        payload = json.dumps(parsed, ensure_ascii=False).encode()
    return Response(payload, headers=response_headers)


async def _count_tokens_via_anthropic_backend(
    request: Request,
    body: dict[str, Any],
    upstream_model: str,
    backend: AnthropicBackend,
) -> JSONResponse | None:
    """Use the bound native token counter; None requests local estimation."""
    if backend.token_counter is None:
        return None

    outgoing = dict(body)
    outgoing["model"] = upstream_model
    try:
        upstream_response = await backend.token_counter(
            json.dumps(outgoing, ensure_ascii=False).encode(),
            backend.header_policy(request),
        )
    except (UpstreamAuthError, UpstreamError, httpx.HTTPError) as exc:
        logger.warning(
            "native Messages count_tokens failed, falling back to estimate: %s",
            exc,
        )
        return None
    try:
        payload = await upstream_response.aread()
    except httpx.HTTPError as exc:
        logger.warning(
            "native Messages count_tokens failed, falling back to estimate: %s",
            exc,
        )
        return None
    finally:
        await upstream_response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning(
            "native Messages count_tokens returned a non-JSON body; "
            "falling back to estimate"
        )
        return None
    return JSONResponse(parsed)
