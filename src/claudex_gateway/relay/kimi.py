"""Kimi-native message relay and token-count serving path."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from claudex_gateway import server_support
from claudex_gateway.providers.kimi_auth import KimiAuthError
from claudex_gateway.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.relay.common import (
    _MANAGED_RELAY_SKIP_REQUEST_HEADERS,
    _OAUTH_BETA,
    _PASSTHROUGH_SKIP_RESPONSE_HEADERS,
    _STATUS_TO_CLAUDE_ERROR_TYPE,
    _format_sse,
)
from claudex_gateway.upstream_errors import UpstreamAuthError, UpstreamError

logger = logging.getLogger("claudex_gateway.server")


def _kimi_request_headers(request: Request) -> dict[str, str]:
    """Forward the client's headers with the gateway's OAuth identity.

    The caller is real Claude Code, so its own fingerprint and beta headers
    are kept; only credentials are replaced (by KimiClient) and the OAuth
    beta is guaranteed to be present.
    """
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _MANAGED_RELAY_SKIP_REQUEST_HEADERS
    }
    headers.setdefault("anthropic-version", "2023-06-01")
    betas = [beta.strip() for beta in headers.get("anthropic-beta", "").split(",") if beta.strip()]
    if _OAUTH_BETA not in betas:
        betas.append(_OAUTH_BETA)
    headers["anthropic-beta"] = ",".join(betas)
    return headers


def _kimi_upstream_error_to_claude(exc: KimiUpstreamError) -> tuple[int, dict[str, Any]]:
    if exc.status_code == 401:
        # A post-retry 401 means the gateway's credential is bad, not the
        # client's; relaying it verbatim would trigger a Claude Code re-auth.
        return 401, server_support._claude_error_body(
            "authentication_error",
            f"Kimi rejected the gateway credentials: {server_support._upstream_error_message(exc.body)}; "
            "run `kimi login` again",
        )
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(exc.body)
        if isinstance(parsed, dict) and parsed.get("type") == "error":
            # Kimi speaks the Anthropic error shape natively; relay it.
            return exc.status_code, parsed
    error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
    return exc.status_code, server_support._claude_error_body(error_type, server_support._upstream_error_message(exc.body))


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


async def _rewrite_kimi_sse(
    upstream_response: httpx.Response, requested_model: str
) -> AsyncIterator[bytes]:
    """Relay complete Kimi SSE events, restoring the requested model in message_start.

    Owns upstream_response: Starlette never closes body iterators, so every
    exit — exhaustion, error, or client disconnect unwinding through this
    generator — must release the Kimi HTTP stream here.
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
        logger.warning("kimi stream aborted: %r", exc)
        yield _format_sse(
            "error", server_support._claude_error_body("api_error", f"kimi stream aborted: {exc!r}")
        ).encode()
    finally:
        await upstream_response.aclose()


async def _relay_to_kimi(
    request: Request, claude_request: dict[str, Any], kimi_model: str
) -> Response:
    """Relay a Messages request to the Kimi coding backend near-verbatim.

    Kimi's coding endpoint speaks the Anthropic Messages API natively, so only
    the model name is rewritten on the way out and restored on the way back;
    validation stays with Kimi, exactly like the Anthropic passthrough.
    """
    kimi_client: KimiClient = request.app.state.kimi_client
    requested_model = str(claude_request.get("model", ""))
    outgoing = dict(claude_request)
    outgoing["model"] = kimi_model
    logger.info(
        "%s -> kimi:%s (stream=%s, messages=%d)",
        requested_model or "?",
        kimi_model,
        bool(claude_request.get("stream")),
        len(claude_request.get("messages") or []),
    )

    try:
        upstream_response = await kimi_client.send_messages(
            json.dumps(outgoing, ensure_ascii=False).encode(), _kimi_request_headers(request)
        )
    except KimiUpstreamError as exc:
        status_code, body = _kimi_upstream_error_to_claude(exc)
        logger.warning("kimi upstream error %s: %s", exc.status_code, exc.body[:500])
        return JSONResponse(body, status_code=status_code)
    except KimiAuthError as exc:
        return JSONResponse(server_support._claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("kimi backend unreachable: %r", exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the Kimi backend: {exc!r}"),
            status_code=502,
        )

    response_headers = {
        key: value
        for key, value in upstream_response.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_RESPONSE_HEADERS
    }
    if claude_request.get("stream"):
        return StreamingResponse(
            _rewrite_kimi_sse(upstream_response, requested_model),
            headers=response_headers,
        )

    try:
        payload = await upstream_response.aread()
    finally:
        await upstream_response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        # Not a message body we can rewrite; relay it untouched.
        return Response(payload, headers=response_headers)
    if isinstance(parsed, dict) and "model" in parsed:
        parsed["model"] = requested_model
        payload = json.dumps(parsed, ensure_ascii=False).encode()
    return Response(payload, headers=response_headers)


async def _count_tokens_via_kimi(
    request: Request, body: dict[str, Any], kimi_model: str
) -> JSONResponse | None:
    """Forward count_tokens to Kimi's native counter; None means fall back.

    Token counting is advisory (Claude Code's context display), so any Kimi
    failure degrades to the local estimate instead of failing the request.
    """
    kimi_client: KimiClient = request.app.state.kimi_client
    outgoing = dict(body)
    outgoing["model"] = kimi_model
    try:
        upstream_response = await kimi_client.count_tokens(
            json.dumps(outgoing, ensure_ascii=False).encode(), _kimi_request_headers(request)
        )
    except (UpstreamAuthError, UpstreamError, httpx.HTTPError) as exc:
        logger.warning("kimi count_tokens failed, falling back to estimate: %s", exc)
        return None
    try:
        payload = await upstream_response.aread()
    except httpx.HTTPError as exc:
        logger.warning("kimi count_tokens failed, falling back to estimate: %s", exc)
        return None
    finally:
        await upstream_response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("kimi count_tokens returned a non-JSON body; falling back to estimate")
        return None
    return JSONResponse(parsed)
