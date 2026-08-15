"""Sibling-independent primitives shared by relay serving paths."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse

import claudex_gateway.translate.context_overflow as context_overflow
from claudex_gateway import server_support
from claudex_gateway.upstream_errors import UpstreamError

logger = logging.getLogger("claudex_gateway.server")

_ANTHROPIC_API_BASE = "https://api.anthropic.com"

# Hop-by-hop and transport headers that must not be forwarded verbatim.
# accept-encoding is dropped so httpx negotiates (and transparently decodes)
# compression itself; the matching content-encoding/length response headers
# are dropped because the forwarded body is already decoded.
_PASSTHROUGH_SKIP_REQUEST_HEADERS = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
)
_PASSTHROUGH_SKIP_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-encoding", "transfer-encoding", "connection"}
)

# A managed relay (Kimi, or a registered Claude account) replaces the
# client's Anthropic credentials with the gateway's own Bearer token, so both
# credential header forms are dropped.
_MANAGED_RELAY_SKIP_REQUEST_HEADERS = _PASSTHROUGH_SKIP_REQUEST_HEADERS | {
    "authorization",
    "x-api-key",
}

# A managed relay's token comes from an OAuth login, so the upstream request
# must advertise the OAuth beta even when the client authenticated some other
# way; ported from CLIProxyAPI's Claude-header handling.
_OAUTH_BETA = "oauth-2025-04-20"

_STATUS_TO_CLAUDE_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def _upstream_error_code(body: str) -> str | None:
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("error")
            if isinstance(detail, dict) and isinstance(detail.get("code"), str):
                return detail["code"]
    return None


def _upstream_error_to_claude(
    exc: UpstreamError,
    *,
    claude_request: dict[str, Any] | None = None,
    context_window: int | None = None,
) -> tuple[int, dict[str, Any]]:
    error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
    message = server_support._upstream_error_message(exc.body)
    code = _upstream_error_code(exc.body)
    # Overflow classification is provider-neutral (shared error-code and
    # phrase matching in translate.context_overflow), so it applies to both
    # Codex and Grok errors alike; when the caller supplies both the request
    # and a catalog-resolved context window, the rewrite is enriched with the
    # `<actual> tokens > <limit>` pair Claude Code's client needs to compact.
    estimated_tokens = None
    if (
        claude_request is not None
        and context_window is not None
        and context_overflow.is_context_overflow_error(code, message)
    ):
        estimated_tokens = context_overflow.estimate_overflow_prompt_tokens(claude_request)
    rewritten = context_overflow.rewrite_context_overflow_message(
        code, message, estimated_tokens=estimated_tokens, context_window=context_window
    )
    if rewritten is not None:
        # Anthropic reports context overflow as 400 invalid_request_error.
        return 400, server_support._claude_error_body("invalid_request_error", rewritten)
    return exc.status_code, server_support._claude_error_body(error_type, message)


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _send_to_anthropic(
    request: Request, headers: dict[str, str], content: bytes
) -> httpx.Response:
    """POST the body to the Anthropic API path the client requested.

    Returns the open streaming response regardless of status; raises
    httpx.HTTPError on transport failure. Ownership of the response
    transfers to the caller.
    """
    http_client: httpx.AsyncClient = request.app.state.http_client
    url = f"{_ANTHROPIC_API_BASE}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    upstream_request = http_client.build_request("POST", url, headers=headers, content=content)
    return await http_client.send(upstream_request, stream=True)


def _relay_anthropic_response(
    upstream_response: httpx.Response, *, on_finished: Callable[[], None] | None = None
) -> StreamingResponse:
    """Relay an open Anthropic response verbatim, owning its lifetime.

    `on_finished`, when given, runs synchronously once the body iterator is
    fully done -- success, mid-stream failure, or cancellation alike -- right
    alongside closing `upstream_response`; the balanced runner uses it to release
    a migrated attempt's migration token only once the whole stream terminates,
    not merely once its 2xx headers commit.
    """
    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_RESPONSE_HEADERS
    }

    async def forward_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_bytes():
                yield chunk
        except httpx.HTTPError as exc:
            # The status line is already sent, so a mid-stream failure can only
            # be reported in-band: SSE responses get a terminal error event,
            # anything else is simply truncated.
            logger.warning("anthropic passthrough stream aborted: %r", exc)
            content_type = upstream_response.headers.get("content-type", "")
            if content_type.lower().startswith("text/event-stream"):
                # The abort may have cut the stream mid-line, so force an event
                # boundary first to keep the injected event parseable.
                yield b"\n\n" + _format_sse(
                    "error",
                    server_support._claude_error_body("api_error", f"anthropic stream aborted: {exc!r}"),
                ).encode()
        finally:
            await upstream_response.aclose()
            if on_finished is not None:
                on_finished()

    return StreamingResponse(
        forward_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
