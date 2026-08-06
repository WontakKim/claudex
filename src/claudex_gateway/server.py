"""Starlette application translating Anthropic Messages requests to the Codex backend."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.resources
import json
import logging
import os
import secrets
import time
import traceback
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import claudex_gateway
from claudex_gateway.codex_auth import CodexAuthError, CodexAuthManager
from claudex_gateway.codex_client import CodexClient, CodexUpstreamError
from claudex_gateway.compaction import (
    build_reroute_headers,
    build_reroute_payload,
    is_compaction_request,
)
from claudex_gateway.config import (
    SETTINGS_KEYS,
    VALID_LOG_LEVELS,
    ConfigError,
    GatewayConfig,
    RouteTarget,
    parse_compaction_model,
    parse_route_target,
    update_settings_file,
    validate_model_map,
)
from claudex_gateway.kimi_auth import KimiAuthError, KimiAuthManager
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.translate import (
    CodexToClaudeStreamTranslator,
    TranslationError,
    assemble_claude_message,
    translate_claude_request_to_codex,
)
from claudex_gateway.translate.codex_to_claude import (
    estimate_overflow_prompt_tokens,
    is_context_overflow_error,
    rewrite_context_overflow_message,
)
from claudex_gateway.usage import (
    consume_codex_reset_credit,
    fetch_claude_usage,
    fetch_codex_usage,
    fetch_kimi_usage,
    fetch_grok_usage,
)
from claudex_gateway.grok_auth import GrokAuthError, GrokAuthManager
from claudex_gateway.grok_client import GrokClient, GrokUpstreamError, sanitize_grok_payload

logger = logging.getLogger(__name__)

_UPSTREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=30.0, pool=15.0)

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

# The Kimi relay replaces the client's Anthropic credentials with the
# gateway's own Bearer token, so both credential header forms are dropped.
_KIMI_SKIP_REQUEST_HEADERS = _PASSTHROUGH_SKIP_REQUEST_HEADERS | {"authorization", "x-api-key"}

# The gateway's Kimi token comes from an OAuth login, so the upstream request
# must advertise the OAuth beta even when the client authenticated some other
# way; ported from CLIProxyAPI's Claude-header handling.
_KIMI_OAUTH_BETA = "oauth-2025-04-20"

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

_STATUS_TO_OPENAI_ERROR_TYPE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "server_error",
    503: "server_error",
    529: "server_error",
}


def _claude_error_body(error_type: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _openai_error_body(
    error_type: str, message: str, code: str | None = None
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }


def _upstream_error_message(body: str) -> str:
    message = body
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("error")
            if isinstance(detail, dict) and detail.get("message"):
                message = str(detail["message"])
            elif isinstance(parsed.get("detail"), str):
                message = parsed["detail"]
    return message


def _upstream_error_code(body: str) -> str | None:
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            detail = parsed.get("error")
            if isinstance(detail, dict) and isinstance(detail.get("code"), str):
                return detail["code"]
    return None


def _upstream_error_to_claude(
    exc: CodexUpstreamError | GrokUpstreamError,
    *,
    claude_request: dict[str, Any] | None = None,
    context_window: int | None = None,
) -> tuple[int, dict[str, Any]]:
    error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
    message = _upstream_error_message(exc.body)
    code = _upstream_error_code(exc.body)
    # Overflow classification is provider-neutral (shared error-code and
    # phrase matching in translate.codex_to_claude), so it applies to both
    # Codex and Grok errors alike; when the caller supplies both the request
    # and a catalog-resolved context window, the rewrite is enriched with the
    # `<actual> tokens > <limit>` pair Claude Code's client needs to compact.
    estimated_tokens = None
    if (
        claude_request is not None
        and context_window is not None
        and is_context_overflow_error(code, message)
    ):
        estimated_tokens = estimate_overflow_prompt_tokens(claude_request)
    rewritten = rewrite_context_overflow_message(
        code, message, estimated_tokens=estimated_tokens, context_window=context_window
    )
    if rewritten is not None:
        # Anthropic reports context overflow as 400 invalid_request_error.
        return 400, _claude_error_body("invalid_request_error", rewritten)
    return exc.status_code, _claude_error_body(error_type, message)


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _require_local_token(
    request: Request, *, claude_error: bool = False
) -> JSONResponse | None:
    config: GatewayConfig = request.app.state.config
    expected_token = config.local_token
    if expected_token is None:
        return None
    authorization = request.headers.get("authorization", "")
    scheme, separator, provided_token = authorization.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(provided_token, expected_token)
    ):
        body = (
            _claude_error_body("authentication_error", "Missing or invalid bearer token")
            if claude_error
            else _openai_error_body(
                "authentication_error",
                "Missing or invalid bearer token",
                "invalid_api_key",
            )
        )
        return JSONResponse(
            body,
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


async def _read_json_object(
    request: Request, error_factory: Any
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JSONResponse(
            error_factory("invalid_request_error", "request body is not valid JSON"),
            status_code=400,
        )
    if not isinstance(body, dict):
        return None, JSONResponse(
            error_factory("invalid_request_error", "request body must be a JSON object"),
            status_code=400,
        )
    return body, None


def _validate_mapped_claude_request(claude_request: dict[str, Any]) -> str | None:
    """Check the required Messages fields the Codex translation consumes.

    Only mapped requests are validated; unmapped bodies pass through so
    Anthropic stays the authority for its own traffic. The Codex backend
    rejects max_output_tokens, so max_tokens is required for contract
    conformance but cannot be enforced on the Codex side.
    """
    max_tokens = claude_request.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        return "max_tokens must be a positive integer"
    if not isinstance(claude_request.get("messages"), list):
        return "messages must be an array"
    return None


def _route_for_request(config: GatewayConfig, parsed: Any) -> RouteTarget | None:
    """Return the mapped route, or None for verbatim Anthropic passthrough.

    Only requests whose model has a model_map entry are handed to a backend;
    everything else — unmapped models and unparseable bodies alike — passes
    through so the gateway stays transparent for plain Anthropic traffic.
    """
    if not isinstance(parsed, dict):
        return None
    model = parsed.get("model")
    return config.mapped_route(model if isinstance(model, str) else None)


async def _passthrough_to_anthropic(
    request: Request, raw_body: bytes
) -> JSONResponse | StreamingResponse:
    """Forward the request to the real Anthropic API and relay the response verbatim.

    The client's own credentials and beta headers are forwarded untouched, so
    passthrough traffic behaves exactly as if Claude Code talked to Anthropic
    directly.
    """
    http_client: httpx.AsyncClient = request.app.state.http_client
    url = f"{_ANTHROPIC_API_BASE}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_REQUEST_HEADERS
    }
    upstream_request = http_client.build_request("POST", url, headers=headers, content=raw_body)
    try:
        upstream_response = await http_client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("anthropic passthrough failed: %s", exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
            status_code=502,
        )
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
                    _claude_error_body("api_error", f"anthropic stream aborted: {exc!r}"),
                ).encode()
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        forward_body(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


async def _handle_messages(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = _require_local_token(request, claude_error=True)
    if unauthorized is not None:
        return unauthorized

    config: GatewayConfig = request.app.state.config

    raw_body = await request.body()
    try:
        claude_request = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        claude_request = None

    model = claude_request.get("model") if isinstance(claude_request, dict) else None
    route = _route_for_request(config, claude_request)
    if route is None:
        logger.info("%s -> anthropic passthrough", model or "?")
        return await _passthrough_to_anthropic(request, raw_body)
    if route.provider == "kimi":
        return await _relay_to_kimi(request, claude_request, route.model)
    return await _relay_via_responses_backend(request, claude_request, route)


async def _relay_via_responses_backend(
    request: Request, claude_request: dict[str, Any], route: RouteTarget
) -> JSONResponse | StreamingResponse:
    """Relay a Messages request to a Responses-API backend (Codex or Grok).

    Both providers consume the same translated payload and produce the same
    SSE event family, so one path serves both; the Grok direction only adds
    its payload sanitizer on the way out.

    Before translation, a Codex/Grok mapped request also carries the
    compaction reroute trigger: when `config.compaction_model` is set, the
    body is a detected Claude Code compaction request (Signal A), and the
    mapped model's catalog context window is a real (non-bool) integer the
    estimated prompt overflows, the request is diverted to
    `_reroute_compaction` before `translate_claude_request_to_codex` or
    `sanitize_grok_payload` ever run. A `None` return from that helper means
    "fall back": translation then runs against the untouched original body
    exactly as an ordinary mapped request would, and the context window
    already resolved for the trigger check is reused rather than looked up
    again.
    """
    config: GatewayConfig = request.app.state.config
    provider = route.provider
    upstream_model = route.model

    validation_error = _validate_mapped_claude_request(claude_request)
    if validation_error is not None:
        return JSONResponse(
            _claude_error_body("invalid_request_error", validation_error), status_code=400
        )

    if provider == "grok":
        client: CodexClient | GrokClient = request.app.state.grok_client
    else:
        client = request.app.state.codex_client

    context_window: int | None = None
    context_window_resolved = False
    if config.compaction_model is not None and is_compaction_request(claude_request):
        context_window = await client.context_window(upstream_model)
        context_window_resolved = True
        # Booleans are `int` subclasses; a catalog entry that is literally
        # `True`/`False` (e.g. a stubbed or malformed provider response) must
        # never be treated as a real window, so the bool case is excluded
        # explicitly rather than trusting `isinstance(x, int)` alone.
        if isinstance(context_window, int) and not isinstance(context_window, bool):
            estimated_prompt_tokens = estimate_overflow_prompt_tokens(claude_request)
            if estimated_prompt_tokens > context_window:
                target_model = parse_compaction_model(config.compaction_model)
                reroute_response = await _reroute_compaction(
                    request,
                    claude_request,
                    target_model=target_model,
                    mapped_model=f"{provider}:{upstream_model}",
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    context_window=context_window,
                )
                if reroute_response is not None:
                    return reroute_response

    try:
        payload = translate_claude_request_to_codex(
            claude_request, upstream_model, config.reasoning_effort_override
        )
    except TranslationError as exc:
        return JSONResponse(
            _claude_error_body("invalid_request_error", str(exc)), status_code=400
        )
    if provider == "grok":
        payload = sanitize_grok_payload(payload, upstream_model)
    session_id = payload["prompt_cache_key"]
    logger.info(
        "%s -> %s:%s (stream=%s, effort=%s, messages=%d, tools=%d)",
        claude_request.get("model", "?"),
        provider,
        upstream_model,
        bool(claude_request.get("stream")),
        (payload.get("reasoning") or {}).get("effort", "-"),
        len(claude_request.get("messages") or []),
        len(payload.get("tools") or []),
    )

    # Resolved once per request from the provider's own catalog cache (the
    # client instance is long-lived on request.app.state). Fresh-cache
    # lookups are memory-only; a cold or expired cache may synchronously
    # refresh the catalog once before falling back to stale data or None.
    # The compaction trigger above already resolved this for a detected
    # compaction request, so that lookup is reused here instead of repeated.
    if not context_window_resolved:
        context_window = await client.context_window(upstream_model)

    event_stream = client.stream_responses(payload, session_id)
    try:
        first_event = await anext(event_stream, None)
        if first_event is None:
            return JSONResponse(
                _claude_error_body("api_error", f"{provider} stream ended without any events"),
                status_code=502,
            )
    except (CodexUpstreamError, GrokUpstreamError) as exc:
        status_code, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        logger.warning(
            "%s upstream error %s: %s", provider, exc.status_code, body["error"]["message"]
        )
        return JSONResponse(body, status_code=status_code)
    except (CodexAuthError, GrokAuthError) as exc:
        return JSONResponse(_claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("%s backend unreachable: %r", provider, exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the {provider} backend: {exc!r}"),
            status_code=502,
        )

    async def upstream_events() -> AsyncIterator[dict[str, Any]]:
        # Owns event_stream: Starlette never closes body iterators, so every
        # exit — exhaustion, error, or client disconnect unwinding through
        # this generator — must release the Codex HTTP stream here.
        try:
            if first_event is not None:
                yield first_event
            async for event in event_stream:
                yield event
        finally:
            await event_stream.aclose()

    if claude_request.get("stream"):
        return StreamingResponse(
            _translate_claude_sse(claude_request, upstream_events(), context_window),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return await _aggregate_claude_response(claude_request, upstream_events(), context_window)


def _rfc3339_now() -> str:
    """Return the current UTC time as an RFC 3339 string for diagnostics records."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _reject_nonfinite_json(constant: str) -> Any:
    """`json.loads` parse_constant hook: refuse NaN/Infinity/-Infinity."""
    raise ValueError(f"non-finite JSON constant {constant!r} in reroute response")


def _assign_compaction_reroute(
    app_state: Any,
    *,
    outcome: str,
    target_model: str | None,
    mapped_model: str,
    estimated_prompt_tokens: int | None,
    context_window: int | None,
    detail: str | None,
) -> int:
    """Write a fresh `compaction_last_reroute` record and return its sequence.

    The record matches the pinned seven-key schema exactly: `outcome`,
    `timestamp`, `target_model`, `mapped_model`, `estimated_prompt_tokens`,
    `context_window`, and `detail`. `detail` must be one of the pinned
    grammar's values (`None`, `"connect_error"`, `"read_error"`,
    `"invalid_json"`, or `"http_<status>"`) — never an exception, a
    credential, a header, or an upstream body. The returned sequence is an
    internal monotonic counter that `_replace_compaction_reroute_if_current`
    uses to guard against a stale write clobbering a newer record; it is
    never part of the serialized record itself.
    """
    app_state.compaction_reroute_sequence += 1
    sequence = app_state.compaction_reroute_sequence
    app_state.compaction_last_reroute = {
        "outcome": outcome,
        "timestamp": _rfc3339_now(),
        "target_model": target_model,
        "mapped_model": mapped_model,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "context_window": context_window,
        "detail": detail,
    }
    return sequence


def _replace_compaction_reroute_if_current(
    app_state: Any, sequence: int, *, outcome: str, detail: str
) -> None:
    """Upgrade the record captured at `sequence`, unless a newer one exists.

    Only `outcome`, `timestamp`, and `detail` change; every other pinned
    field of the existing record is preserved untouched. This lets a later
    mid-stream failure upgrade its own committed `rerouted` record to
    `midstream_error` without a stale failure ever overwriting a
    subsequent, unrelated request's own record.
    """
    if app_state.compaction_reroute_sequence != sequence:
        return
    record = dict(app_state.compaction_last_reroute)
    record["outcome"] = outcome
    record["timestamp"] = _rfc3339_now()
    record["detail"] = detail
    app_state.compaction_last_reroute = record


async def _reroute_compaction(
    request: Request,
    claude_request: dict[str, Any],
    *,
    target_model: str,
    mapped_model: str,
    estimated_prompt_tokens: int,
    context_window: int,
) -> JSONResponse | StreamingResponse | None:
    """Attempt the compaction reroute call to Anthropic; `None` means fall back.

    Builds a direct, credential-scoped `POST` to
    `https://api.anthropic.com/v1/messages` on the lifespan-owned
    `httpx.AsyncClient`, from `build_reroute_payload`/`build_reroute_headers`
    and the already-canonical `target_model`. This never calls
    `_route_for_request` or `_passthrough_to_anthropic`, so the canonical
    target can never re-enter the model map and recurse, no matter how it
    might substring-match a map key. Redirects are never followed and there
    is no application-level retry: exactly one Anthropic attempt per
    triggered request, and every 3xx response is treated as a non-2xx
    outcome rather than a followed redirect.

    `stream:false` and `stream:true` share every step through the HTTP
    response: only an HTTP 2xx status commits the reroute, and any transport
    failure or non-2xx status before that point falls back to the mapped
    path with the untouched original body, closing the response without
    reading or logging it. Once 2xx is accepted, `stream:false` reads and
    re-serves the whole JSON body here; `stream:true` hands the still-open
    response to `_relay_compaction_stream`, which owns it from that point on
    and relays `aiter_bytes()` unchanged as the returned `StreamingResponse`
    body.
    """

    def record(outcome: str, detail: str | None) -> int:
        return _assign_compaction_reroute(
            request.app.state,
            outcome=outcome,
            target_model=target_model,
            mapped_model=mapped_model,
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_window=context_window,
            detail=detail,
        )

    config: GatewayConfig = request.app.state.config
    headers = build_reroute_headers(request.headers, config.local_token)
    if headers is None:
        record("skipped_no_credentials", None)
        return None

    payload = build_reroute_payload(claude_request, target_model)
    http_client: httpx.AsyncClient = request.app.state.http_client
    upstream_request = http_client.build_request(
        "POST",
        f"{_ANTHROPIC_API_BASE}/v1/messages",
        headers=headers,
        content=json.dumps(payload, ensure_ascii=False).encode(),
    )
    try:
        upstream_response = await http_client.send(
            upstream_request, stream=True, follow_redirects=False
        )
    except httpx.HTTPError as exc:
        logger.warning("compaction reroute unreachable: %r", exc)
        record("fallback_mapped", "connect_error")
        return None

    if not 200 <= upstream_response.status_code < 300:
        logger.warning(
            "compaction reroute upstream returned %s", upstream_response.status_code
        )
        record("fallback_mapped", f"http_{upstream_response.status_code}")
        await upstream_response.aclose()
        return None

    if claude_request.get("stream"):
        # Commit point: HTTP 2xx acceptance, not the first relayed byte.
        # From here on, _relay_compaction_stream owns upstream_response; it
        # is never closed in this function again.
        sequence = record("rerouted", None)
        return StreamingResponse(
            _relay_compaction_stream(upstream_response, request.app.state, sequence),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        try:
            body = await upstream_response.aread()
        except httpx.HTTPError as exc:
            logger.warning("compaction reroute response read failed: %r", exc)
            record("fallback_mapped", "read_error")
            return None
        try:
            # json.loads accepts the non-standard NaN/Infinity constants that
            # a strict Anthropic response can never contain and that Starlette
            # would refuse to serialize; reject them at parse time so a
            # malformed body falls back instead of escaping as a 500.
            parsed = json.loads(body, parse_constant=_reject_nonfinite_json)
        except (ValueError, RecursionError):
            record("fallback_mapped", "invalid_json")
            return None
        if not isinstance(parsed, dict):
            record("fallback_mapped", "invalid_json")
            return None
        try:
            # Built before the record is written: if serialization fails
            # (lone surrogates, pathological nesting), the outcome must be
            # fallback, never a request failure logged as "rerouted".
            reroute_response = JSONResponse(parsed, status_code=200)
        except (ValueError, TypeError, RecursionError):
            record("fallback_mapped", "invalid_json")
            return None
        record("rerouted", None)
        return reroute_response
    finally:
        await upstream_response.aclose()


async def _relay_compaction_stream(
    upstream_response: httpx.Response,
    app_state: Any,
    sequence: int,
) -> AsyncIterator[bytes]:
    """Relay `upstream_response.aiter_bytes()` unchanged; owns `upstream_response`.

    Starlette never closes body iterators, so every exit here — normal
    exhaustion, a mid-stream `httpx.HTTPError`, cancellation, or an explicit
    iterator close — must release the Anthropic HTTP stream via `finally`.

    The reroute already committed (`rerouted`) before this generator starts,
    so a mid-stream `httpx.HTTPError` can only be reported in-band: an event
    boundary followed by a parseable `event: error` using the existing
    Claude error envelope, with no raw exception text. That failure also
    upgrades this request's diagnostics record to `midstream_error`/
    `read_error` via the compare-and-swap helper, but only while `sequence`
    is still current — a stale failure must never clobber a newer request's
    own record.
    """
    try:
        async for chunk in upstream_response.aiter_bytes():
            yield chunk
    except httpx.HTTPError as exc:
        logger.warning("compaction reroute stream aborted: %r", exc)
        yield b"\n\n" + _format_sse(
            "error",
            _claude_error_body("api_error", "compaction reroute stream aborted"),
        ).encode()
        _replace_compaction_reroute_if_current(
            app_state, sequence, outcome="midstream_error", detail="read_error"
        )
    finally:
        await upstream_response.aclose()


async def _translate_claude_sse(
    claude_request: dict[str, Any],
    upstream_events: AsyncGenerator[dict[str, Any], None],
    context_window: int | None = None,
) -> AsyncIterator[str]:
    translator = CodexToClaudeStreamTranslator(claude_request, context_window=context_window)
    try:
        async for event in upstream_events:
            for event_name, payload in translator.translate_event(event):
                yield _format_sse(event_name, payload)
    except (CodexUpstreamError, GrokUpstreamError) as exc:
        _, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        yield _format_sse("error", body)
    except (CodexAuthError, GrokAuthError, httpx.HTTPError) as exc:
        logger.warning("responses stream aborted: %s", exc)
        error_type = (
            "authentication_error"
            if isinstance(exc, (CodexAuthError, GrokAuthError))
            else "api_error"
        )
        yield _format_sse("error", _claude_error_body(error_type, str(exc)))
    finally:
        await upstream_events.aclose()


async def _aggregate_claude_response(
    claude_request: dict[str, Any],
    upstream_events: AsyncGenerator[dict[str, Any], None],
    context_window: int | None = None,
) -> JSONResponse:
    translator = CodexToClaudeStreamTranslator(claude_request, context_window=context_window)
    claude_events: list[tuple[str, dict[str, Any]]] = []
    try:
        async for event in upstream_events:
            for translated in translator.translate_event(event):
                event_name, payload = translated
                if event_name == "error":
                    error_type = payload["error"]["type"]
                    status_code = 400 if error_type == "invalid_request_error" else 502
                    return JSONResponse(payload, status_code=status_code)
                claude_events.append(translated)
    except (CodexUpstreamError, GrokUpstreamError) as exc:
        status_code, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        return JSONResponse(body, status_code=status_code)
    except (CodexAuthError, GrokAuthError) as exc:
        return JSONResponse(_claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("responses stream aborted: %r", exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the upstream backend: {exc!r}"),
            status_code=502,
        )
    finally:
        # Early error returns above must not leave the upstream stream open.
        await upstream_events.aclose()

    message = assemble_claude_message(claude_events)
    if message is None:
        return JSONResponse(
            _claude_error_body("api_error", "codex stream ended without a terminal response event"),
            status_code=502,
        )
    return JSONResponse(message)


def _kimi_request_headers(request: Request) -> dict[str, str]:
    """Forward the client's headers with the gateway's OAuth identity.

    The caller is real Claude Code, so its own fingerprint and beta headers
    are kept; only credentials are replaced (by KimiClient) and the OAuth
    beta is guaranteed to be present.
    """
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _KIMI_SKIP_REQUEST_HEADERS
    }
    headers.setdefault("anthropic-version", "2023-06-01")
    betas = [beta.strip() for beta in headers.get("anthropic-beta", "").split(",") if beta.strip()]
    if _KIMI_OAUTH_BETA not in betas:
        betas.append(_KIMI_OAUTH_BETA)
    headers["anthropic-beta"] = ",".join(betas)
    return headers


def _kimi_upstream_error_to_claude(exc: KimiUpstreamError) -> tuple[int, dict[str, Any]]:
    if exc.status_code == 401:
        # A post-retry 401 means the gateway's credential is bad, not the
        # client's; relaying it verbatim would trigger a Claude Code re-auth.
        return 401, _claude_error_body(
            "authentication_error",
            f"Kimi rejected the gateway credentials: {_upstream_error_message(exc.body)}; "
            "run `kimi login` again",
        )
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(exc.body)
        if isinstance(parsed, dict) and parsed.get("type") == "error":
            # Kimi speaks the Anthropic error shape natively; relay it.
            return exc.status_code, parsed
    error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
    return exc.status_code, _claude_error_body(error_type, _upstream_error_message(exc.body))


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
            "error", _claude_error_body("api_error", f"kimi stream aborted: {exc!r}")
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
        return JSONResponse(_claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("kimi backend unreachable: %r", exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the Kimi backend: {exc!r}"),
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
    except (KimiAuthError, KimiUpstreamError, httpx.HTTPError) as exc:
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


async def _handle_count_tokens(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = _require_local_token(request, claude_error=True)
    if unauthorized is not None:
        return unauthorized

    config: GatewayConfig = request.app.state.config

    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None

    route = _route_for_request(config, body)
    if route is None:
        return await _passthrough_to_anthropic(request, raw_body)
    if route.provider == "kimi":
        counted = await _count_tokens_via_kimi(request, body, route.model)
        if counted is not None:
            return counted

    # Rough characters/4 estimate: mapped prompts must not be sent to
    # Anthropic just to be counted, and no Codex tokenizer or token-count
    # endpoint is available (Kimi's native counter is preferred above but
    # may be down). Good enough for Claude Code's context-usage display;
    # not exact for billing or hard context-limit decisions.
    estimated = max(len(json.dumps(body, ensure_ascii=False)) // 4, 1)
    return JSONResponse({"input_tokens": estimated})


async def _handle_hello(request: Request) -> JSONResponse:
    # The launcher compares the version against its own to detect a stale
    # daemon left running across a package update, and matches pid/nonce
    # against its daemon record before signaling the process. The dashboard
    # reads local_auth_required to know whether to ask for the local token —
    # only the boolean is exposed, never the token itself.
    config: GatewayConfig = request.app.state.config
    return JSONResponse(
        {
            "hello": "claudex-gateway",
            "version": claudex_gateway.__version__,
            "pid": os.getpid(),
            "nonce": request.app.state.daemon_nonce,
            "local_auth_required": config.local_token is not None,
        }
    )


async def _handle_health(request: Request) -> JSONResponse:
    config: GatewayConfig = request.app.state.config
    codex_auth_manager: CodexAuthManager = request.app.state.codex_auth_manager
    kimi_auth_manager: KimiAuthManager = request.app.state.kimi_auth_manager
    grok_auth_manager: GrokAuthManager = request.app.state.grok_auth_manager
    providers: dict[str, dict[str, Any]] = {}

    try:
        credentials = await codex_auth_manager.get_credentials()
        providers["codex"] = {
            "status": "ok",
            "auth_mode": "api_key" if credentials.is_api_key else "chatgpt",
            "account": credentials.account_id,
            "email": credentials.email,
        }
    except CodexAuthError as exc:
        providers["codex"] = {"status": "error", "detail": str(exc)}

    # A missing OAuth login only degrades readiness when the map routes to
    # that provider, so setups not using it keep reporting healthy. The flag
    # is exposed so the dashboard can render an unused login failure as
    # neutral, not error.
    kimi_required = config.maps_to_provider("kimi")
    try:
        kimi_credentials = await kimi_auth_manager.get_credentials()
        providers["kimi"] = {
            "status": "ok",
            "required": kimi_required,
            "account": kimi_credentials.account,
        }
    except KimiAuthError as exc:
        providers["kimi"] = {"status": "error", "detail": str(exc), "required": kimi_required}

    grok_required = config.maps_to_provider("grok")
    try:
        grok_credentials = await grok_auth_manager.get_credentials()
        providers["grok"] = {
            "status": "ok",
            "required": grok_required,
            "auth_mode": "api_key" if grok_credentials.is_api_key else "oauth",
            "account": grok_credentials.email,
        }
    except GrokAuthError as exc:
        providers["grok"] = {"status": "error", "detail": str(exc), "required": grok_required}

    is_ready = (
        providers["codex"]["status"] == "ok"
        and (providers["kimi"]["status"] == "ok" or not kimi_required)
        and (providers["grok"]["status"] == "ok" or not grok_required)
    )
    return JSONResponse(
        {"status": "ok" if is_ready else "error", "providers": providers},
        status_code=200 if is_ready else 503,
    )


# The single runtime-editable map. Everything else in GatewayConfig is either
# fixed at startup (bind address, auth directories) or out of the admin API's
# mapping-only scope.
_ADMIN_MAP_KEYS = ("model_map",)


def _admin_guard(request: Request) -> JSONResponse | None:
    """Reject admin requests that could originate from another origin.

    Browsers can fire requests at localhost from any web page (drive-by
    requests, DNS rebinding), so beyond the optional bearer token the admin
    surface only answers when the Host header names the gateway itself.
    """
    denied = _require_local_token(request)
    if denied is not None:
        return denied
    config: GatewayConfig = request.app.state.config
    hostname = (request.url.hostname or "").lower()
    if hostname not in {"localhost", "127.0.0.1", "::1", config.host.lower()}:
        return JSONResponse(
            _openai_error_body(
                "permission_error",
                f"admin API refuses Host {hostname!r} (DNS-rebinding guard)",
            ),
            status_code=403,
        )
    return None


def _mapping_payload(config: GatewayConfig) -> dict[str, Any]:
    return {
        "model_map": config.model_map,
        # The dashboard renders the board view-only while the corresponding
        # environment variable overrides the settings file.
        "env_locked": {
            key: SETTINGS_KEYS[key] if os.environ.get(SETTINGS_KEYS[key]) is not None else None
            for key in _ADMIN_MAP_KEYS
        },
        "codex_home": str(config.codex_home),
        "grok_home": str(config.grok_home),
        "kimi_code_home": str(config.kimi_code_home),
    }


async def _handle_admin_mapping_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_mapping_payload(request.app.state.config))


def _require_json_content_type(request: Request) -> JSONResponse | None:
    """Reject admin writes that are not application/json.

    Requiring application/json forces cross-origin browser requests into a
    CORS preflight, which this server never approves.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.partition(";")[0].strip().lower() != "application/json":
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "admin API requires Content-Type: application/json",
            ),
            status_code=415,
        )
    return None


async def _handle_admin_mapping_put(request: Request) -> JSONResponse:
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None:
        return error
    unknown = sorted(set(body) - set(_ADMIN_MAP_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_ADMIN_MAP_KEYS)}",
            ),
            status_code=400,
        )
    if not body:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"provide at least one of: {', '.join(_ADMIN_MAP_KEYS)}",
            ),
            status_code=400,
        )

    updates: dict[str, dict[str, str]] = {}
    for key in _ADMIN_MAP_KEYS:
        if key not in body:
            continue
        value = body[key]
        if not isinstance(value, dict):
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error",
                    f"{key} must be a JSON object mapping model names",
                ),
                status_code=400,
            )
        try:
            updates[key] = validate_model_map(key, value)
        except ConfigError as exc:
            return JSONResponse(
                _openai_error_body("invalid_request_error", str(exc)),
                status_code=400,
            )
        # An environment variable outranks settings.json at every boot, so a
        # persisted change would silently vanish on restart — refuse instead.
        env_name = SETTINGS_KEYS[key]
        if os.environ.get(env_name) is not None:
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error",
                    f"{env_name} is set in the gateway's environment and overrides "
                    f"{key}; unset it to manage the mapping at runtime",
                ),
                status_code=409,
            )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, dict(updates))
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically and only for
        # runtime-safe fields; in-flight requests keep their config snapshot.
        new_config = replace(config, **updates)
        request.app.state.config = new_config
    return JSONResponse(_mapping_payload(new_config))


class _LogBufferHandler(logging.Handler):
    """Keeps the most recent gateway log records for the dashboard's Log tab."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info and record.exc_info != (None, None, None):
                message += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            self.records.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": message,
                }
            )
        except Exception:  # noqa: BLE001 - a logging handler must never raise
            self.handleError(record)


async def _handle_admin_logs(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    log_buffer: _LogBufferHandler = request.app.state.log_buffer
    return JSONResponse({"logs": list(log_buffer.records)})


async def _handle_admin_usage(request: Request) -> JSONResponse:
    """Probe provider subscription usage for the dashboard's usage cards.

    Each provider answers from its own usage endpoint with the local CLI
    credentials (see usage.py); a failure on one side never masks the other.
    ?provider=claude|codex|kimi|grok refreshes a single card; without it all run.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    provider = request.query_params.get("provider")
    if provider not in (None, "claude", "codex", "kimi", "grok"):
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error", "provider must be one of: claude, codex, kimi, grok"
            ),
            status_code=400,
        )
    probes: dict[str, Any] = {}
    if provider in (None, "claude"):
        probes["claude"] = fetch_claude_usage(request.app.state.http_client)
    if provider in (None, "codex"):
        probes["codex"] = fetch_codex_usage(
            request.app.state.http_client, request.app.state.codex_auth_manager
        )
    if provider in (None, "kimi"):
        probes["kimi"] = fetch_kimi_usage(
            request.app.state.http_client, request.app.state.kimi_auth_manager
        )
    if provider in (None, "grok"):
        probes["grok"] = fetch_grok_usage(
            request.app.state.http_client, request.app.state.grok_auth_manager
        )
    results = await asyncio.gather(*probes.values())
    payload = dict(zip(probes, results))
    payload["fetched_at"] = time.time()
    return JSONResponse(payload)


async def _handle_admin_codex_reset_credit(request: Request) -> JSONResponse:
    """Spend one Codex reset credit — irreversible, so never call it implicitly.

    The admin lock serializes attempts so two clicks cannot both reach the
    backend, and the redeem key is held until an attempt settles: a request
    that timed out may or may not have burned the credit, so the retry reuses
    the key and lets the backend deduplicate instead of spending a second one.
    The key lives in memory only, so a gateway restart forfeits that guard.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    async with request.app.state.admin_lock:
        redeem_request_id = request.app.state.codex_reset_key or str(uuid.uuid4())
        request.app.state.codex_reset_key = redeem_request_id
        result = await consume_codex_reset_credit(
            request.app.state.http_client,
            request.app.state.codex_auth_manager,
            redeem_request_id,
        )
        if result["status"] == "ok":
            request.app.state.codex_reset_key = None
    return JSONResponse(result)


# Root covers the gateway's own loggers; the uvicorn loggers do not
# propagate to root, so they are adjusted explicitly.
_LOG_LEVEL_LOGGER_NAMES = ("", "uvicorn", "uvicorn.access", "uvicorn.error")


def _apply_log_level(log_level: str) -> None:
    level = getattr(logging, log_level.upper())
    for name in _LOG_LEVEL_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


def _log_level_payload(config: GatewayConfig) -> dict[str, Any]:
    env_name = SETTINGS_KEYS["log_level"]
    return {
        "log_level": config.log_level,
        "choices": list(VALID_LOG_LEVELS),
        "env_locked": env_name if os.environ.get(env_name) is not None else None,
    }


async def _handle_admin_log_level_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_log_level_payload(request.app.state.config))


async def _handle_admin_log_level_put(request: Request) -> JSONResponse:
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    value = body.get("log_level")
    if not isinstance(value, str) or value.strip().lower() not in VALID_LOG_LEVELS:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"log_level must be one of: {', '.join(VALID_LOG_LEVELS)}",
            ),
            status_code=400,
        )
    value = value.strip().lower()
    env_name = SETTINGS_KEYS["log_level"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"log_level; unset it to manage the level at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, {"log_level": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body("server_error", f"could not persist settings: {exc}"),
                status_code=500,
            )
        new_config = replace(config, log_level=value)
        request.app.state.config = new_config
    _apply_log_level(value)
    logger.info("log level set to %s", value)
    return JSONResponse(_log_level_payload(new_config))


async def _handle_dashboard(request: Request) -> Response:
    """Serve the runtime dashboard, embedded in the package as dashboard.html."""
    try:
        page = (
            importlib.resources.files("claudex_gateway")
            .joinpath("dashboard.html")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError) as exc:
        logger.warning("dashboard asset unavailable: %s", exc)
        return JSONResponse(
            _openai_error_body("server_error", "dashboard.html is missing from the package"),
            status_code=500,
        )
    return Response(page, media_type="text/html; charset=utf-8")


# Inline SVG so the package ships no binary asset; the glyph is a two-way
# relay in the dashboard's accent color.
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect width="16" height="16" rx="3" fill="#1d5fbf"/>'
    '<path d="M10.2 3.6 12.6 6l-2.4 2.4M12.2 6H3.4M5.8 12.4 3.4 10l2.4-2.4M3.8 10h8.8"'
    ' fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"'
    ' stroke-linejoin="round"/>'
    "</svg>"
)


async def _handle_favicon(request: Request) -> Response:
    """Answer the browser's automatic /favicon.ico probe for the dashboard.

    Without this route every dashboard visit logs a 404 in the access log.
    """
    return Response(
        _FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def _handle_admin_codex_models(request: Request) -> JSONResponse:
    """Proxy the live Codex model catalog for the dashboard's model columns."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    codex_client: CodexClient = request.app.state.codex_client
    try:
        models = await codex_client.list_models()
    except CodexAuthError as exc:
        return JSONResponse(
            _openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except CodexUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            _openai_error_body(error_type, _upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            _openai_error_body("server_error", f"failed to reach the Codex backend: {exc}"),
            status_code=502,
        )
    return JSONResponse({"models": models})


async def _handle_admin_grok_models(request: Request) -> JSONResponse:
    """Relay Grok's live model catalog (IDs only) for model_map authoring.

    Same convenience role as the Codex/Kimi catalog endpoints: the gateway
    never validates map targets against the list, so this only feeds the
    dashboard's add-node suggestions.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    grok_client: GrokClient = request.app.state.grok_client
    try:
        models = await grok_client.list_models()
    except GrokAuthError as exc:
        return JSONResponse(
            _openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except GrokUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            _openai_error_body(error_type, _upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            _openai_error_body("server_error", f"failed to reach the Grok backend: {exc}"),
            status_code=502,
        )
    return JSONResponse({"models": models})


async def _handle_admin_kimi_models(request: Request) -> JSONResponse:
    """Relay Kimi's live model catalog verbatim for model_map authoring.

    The gateway never validates map targets against a model list — values
    after the kimi: prefix bypass untouched so newly released models work
    without a gateway update. This endpoint only exists as a convenience
    source of valid IDs (and future dashboard presets).
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    kimi_client: KimiClient = request.app.state.kimi_client
    try:
        catalog = await kimi_client.list_models()
    except KimiAuthError as exc:
        return JSONResponse(
            _openai_error_body("authentication_error", str(exc)), status_code=401
        )
    except KimiUpstreamError as exc:
        error_type = _STATUS_TO_OPENAI_ERROR_TYPE.get(exc.status_code, "server_error")
        return JSONResponse(
            _openai_error_body(error_type, _upstream_error_message(exc.body)),
            status_code=exc.status_code,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            _openai_error_body("server_error", f"failed to reach the Kimi backend: {exc}"),
            status_code=502,
        )
    return JSONResponse(catalog)


_CONNECTION_TEST_TIMEOUT = 30.0


async def _probe_codex_route(codex_client: CodexClient, target: str) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload = translate_claude_request_to_codex(claude_request, target, "low")
    events = codex_client.stream_responses(payload, payload["prompt_cache_key"])
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        raise CodexUpstreamError(502, "codex stream ended without any events")
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_grok_route(grok_client: GrokClient, target: str) -> str:
    claude_request = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    payload = translate_claude_request_to_codex(claude_request, target, "low")
    payload = sanitize_grok_payload(payload, target)
    events = grok_client.stream_responses(payload, payload["prompt_cache_key"])
    try:
        first_event = await anext(events, None)
    finally:
        await events.aclose()
    if first_event is None:
        raise GrokUpstreamError(502, "grok stream ended without any events")
    response = first_event.get("response") if isinstance(first_event, dict) else None
    model = response.get("model") if isinstance(response, dict) else None
    return model if isinstance(model, str) else target


async def _probe_kimi_route(kimi_client: KimiClient, target_model: str) -> str:
    claude_request = {
        "model": target_model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": _KIMI_OAUTH_BETA,
    }
    response = await kimi_client.send_messages(json.dumps(claude_request).encode(), headers)
    try:
        payload = await response.aread()
    finally:
        await response.aclose()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return target_model
    model = parsed.get("model") if isinstance(parsed, dict) else None
    return model if isinstance(model, str) else target_model


async def _handle_admin_connection_test(request: Request) -> JSONResponse:
    """Send one minimal request through the gateway to verify a target model id.

    The result — success or failure — is always a 200 with the outcome in the
    body; non-200 responses are reserved for invalid test requests themselves.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    target = body.get("target")
    if not isinstance(target, str) or not target.strip():
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error", "target must be a non-empty string"
            ),
            status_code=400,
        )
    target = target.strip()

    started_at = time.monotonic()

    def result(
        ok: bool, status: int | None, detail: str | None = None, response_model: str | None = None
    ) -> JSONResponse:
        return JSONResponse(
            {
                "ok": ok,
                "status": status,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "target": target,
                "response_model": response_model,
                "detail": detail,
            }
        )

    # The target carries the same provider-prefix syntax as model_map values,
    # so the dashboard's test box works for kimi: targets with no UI change.
    try:
        route = parse_route_target(target)
    except ConfigError as exc:
        return JSONResponse(
            _openai_error_body("invalid_request_error", str(exc)), status_code=400
        )

    try:
        if route.provider == "kimi":
            probe = _probe_kimi_route(request.app.state.kimi_client, route.model)
        elif route.provider == "grok":
            probe = _probe_grok_route(request.app.state.grok_client, route.model)
        else:
            probe = _probe_codex_route(request.app.state.codex_client, route.model)
        response_model = await asyncio.wait_for(probe, _CONNECTION_TEST_TIMEOUT)
    except (CodexUpstreamError, KimiUpstreamError, GrokUpstreamError) as exc:
        return result(False, exc.status_code, _upstream_error_message(exc.body))
    except (CodexAuthError, KimiAuthError, GrokAuthError) as exc:
        return result(False, 401, str(exc))
    except TimeoutError:
        return result(False, None, f"no response within {_CONNECTION_TEST_TIMEOUT:.0f}s")
    except httpx.HTTPError as exc:
        return result(False, None, f"failed to reach the upstream: {exc}")
    return result(True, 200, response_model=response_model)


def create_app(config: GatewayConfig, daemon_nonce: str | None = None) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        log_buffer = _LogBufferHandler()
        logging.getLogger().addHandler(log_buffer)
        app.state.log_buffer = log_buffer
        try:
            async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as http_client:
                codex_auth_manager = CodexAuthManager(config.codex_home / "auth.json", http_client)
                kimi_auth_manager = KimiAuthManager(config.kimi_code_home, http_client)
                grok_auth_manager = GrokAuthManager(config.grok_home / "auth.json", http_client)
                app.state.config = config
                app.state.admin_lock = asyncio.Lock()
                # Redeem key of a reset attempt whose outcome never came back.
                app.state.codex_reset_key = None
                # Diagnostics for the compaction reroute (see
                # _assign_compaction_reroute): no reroute has been attempted
                # yet, and the sequence counter starts a fresh count for this
                # process's lifetime.
                app.state.compaction_last_reroute = None
                app.state.compaction_reroute_sequence = 0
                app.state.codex_auth_manager = codex_auth_manager
                app.state.codex_client = CodexClient(codex_auth_manager, http_client)
                app.state.kimi_auth_manager = kimi_auth_manager
                app.state.kimi_client = KimiClient(kimi_auth_manager, http_client)
                app.state.grok_auth_manager = grok_auth_manager
                app.state.grok_client = GrokClient(grok_auth_manager, http_client)
                app.state.http_client = http_client

                try:
                    credentials = await codex_auth_manager.get_credentials()
                    logger.info(
                        "codex credentials ready (mode=%s, account=%s)",
                        "api_key" if credentials.is_api_key else "chatgpt",
                        credentials.account_id,
                    )
                except CodexAuthError as exc:
                    logger.warning("codex direction unavailable: %s", exc)

                if config.maps_to_provider("kimi"):
                    try:
                        await kimi_auth_manager.get_credentials()
                        logger.info("kimi credentials ready")
                    except KimiAuthError as exc:
                        logger.warning("kimi direction unavailable: %s", exc)

                if config.maps_to_provider("grok"):
                    try:
                        await grok_auth_manager.get_credentials()
                        logger.info("grok credentials ready")
                    except GrokAuthError as exc:
                        logger.warning("grok direction unavailable: %s", exc)

                yield
        finally:
            logging.getLogger().removeHandler(log_buffer)

    app = Starlette(
        routes=[
            Route("/", _handle_dashboard, methods=["GET"]),
            Route("/favicon.ico", _handle_favicon, methods=["GET"]),
            Route("/v1/messages", _handle_messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", _handle_count_tokens, methods=["POST"]),
            Route("/api/hello", _handle_hello, methods=["GET"]),
            Route("/health", _handle_health, methods=["GET"]),
            Route("/admin/mapping", _handle_admin_mapping_get, methods=["GET"]),
            Route("/admin/mapping", _handle_admin_mapping_put, methods=["PUT"]),
            Route("/admin/log-level", _handle_admin_log_level_get, methods=["GET"]),
            Route("/admin/log-level", _handle_admin_log_level_put, methods=["PUT"]),
            Route("/admin/logs", _handle_admin_logs, methods=["GET"]),
            Route("/admin/usage", _handle_admin_usage, methods=["GET"]),
            Route(
                "/admin/codex/reset-credit",
                _handle_admin_codex_reset_credit,
                methods=["POST"],
            ),
            Route("/admin/codex/models", _handle_admin_codex_models, methods=["GET"]),
            Route("/admin/kimi/models", _handle_admin_kimi_models, methods=["GET"]),
            Route("/admin/grok/models", _handle_admin_grok_models, methods=["GET"]),
            Route("/admin/test", _handle_admin_connection_test, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.daemon_nonce = daemon_nonce
    return app
