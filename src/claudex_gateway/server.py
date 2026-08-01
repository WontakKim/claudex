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
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import claudex_gateway
from claudex_gateway.codex_auth import CodexAuthError, CodexAuthManager
from claudex_gateway.codex_client import CodexClient, CodexUpstreamError
from claudex_gateway.config import (
    SETTINGS_KEYS,
    VALID_LOG_LEVELS,
    ConfigError,
    GatewayConfig,
    RouteTarget,
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
from claudex_gateway.translate.codex_to_claude import rewrite_context_overflow_message
from claudex_gateway.usage import (
    consume_codex_reset_credit,
    fetch_claude_usage,
    fetch_codex_usage,
    fetch_kimi_usage,
)

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


def _upstream_error_to_claude(exc: CodexUpstreamError) -> tuple[int, dict[str, Any]]:
    error_type = _STATUS_TO_CLAUDE_ERROR_TYPE.get(exc.status_code, "api_error")
    message = _upstream_error_message(exc.body)
    rewritten = rewrite_context_overflow_message(_upstream_error_code(exc.body), message)
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
    codex_model = route.model

    validation_error = _validate_mapped_claude_request(claude_request)
    if validation_error is not None:
        return JSONResponse(
            _claude_error_body("invalid_request_error", validation_error), status_code=400
        )

    codex_client: CodexClient = request.app.state.codex_client
    try:
        payload = translate_claude_request_to_codex(
            claude_request, codex_model, config.reasoning_effort_override
        )
    except TranslationError as exc:
        return JSONResponse(
            _claude_error_body("invalid_request_error", str(exc)), status_code=400
        )
    session_id = payload["prompt_cache_key"]
    logger.info(
        "%s -> %s (stream=%s, effort=%s, messages=%d, tools=%d)",
        claude_request.get("model", "?"),
        codex_model,
        bool(claude_request.get("stream")),
        payload["reasoning"]["effort"],
        len(claude_request.get("messages") or []),
        len(payload.get("tools") or []),
    )

    event_stream = codex_client.stream_responses(payload, session_id)
    try:
        first_event = await anext(event_stream, None)
        if first_event is None:
            return JSONResponse(
                _claude_error_body("api_error", "codex stream ended without any events"),
                status_code=502,
            )
    except CodexUpstreamError as exc:
        status_code, body = _upstream_error_to_claude(exc)
        logger.warning("codex upstream error %s: %s", exc.status_code, body["error"]["message"])
        return JSONResponse(body, status_code=status_code)
    except CodexAuthError as exc:
        return JSONResponse(_claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("codex backend unreachable: %r", exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the Codex backend: {exc!r}"),
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
            _translate_claude_sse(claude_request, upstream_events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    return await _aggregate_claude_response(claude_request, upstream_events())


async def _translate_claude_sse(
    claude_request: dict[str, Any], upstream_events: AsyncGenerator[dict[str, Any], None]
) -> AsyncIterator[str]:
    translator = CodexToClaudeStreamTranslator(claude_request)
    try:
        async for event in upstream_events:
            for event_name, payload in translator.translate_event(event):
                yield _format_sse(event_name, payload)
    except CodexUpstreamError as exc:
        _, body = _upstream_error_to_claude(exc)
        yield _format_sse("error", body)
    except (CodexAuthError, httpx.HTTPError) as exc:
        logger.warning("codex stream aborted: %s", exc)
        error_type = "authentication_error" if isinstance(exc, CodexAuthError) else "api_error"
        yield _format_sse("error", _claude_error_body(error_type, str(exc)))
    finally:
        await upstream_events.aclose()


async def _aggregate_claude_response(
    claude_request: dict[str, Any], upstream_events: AsyncGenerator[dict[str, Any], None]
) -> JSONResponse:
    translator = CodexToClaudeStreamTranslator(claude_request)
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
    except CodexUpstreamError as exc:
        status_code, body = _upstream_error_to_claude(exc)
        return JSONResponse(body, status_code=status_code)
    except CodexAuthError as exc:
        return JSONResponse(_claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("codex stream aborted: %r", exc)
        return JSONResponse(
            _claude_error_body("api_error", f"failed to reach the Codex backend: {exc!r}"),
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
            "run `claudex-gateway login kimi` again",
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
    providers: dict[str, dict[str, Any]] = {}

    try:
        credentials = await codex_auth_manager.get_credentials()
        providers["codex"] = {
            "status": "ok",
            "auth_mode": "api_key" if credentials.is_api_key else "chatgpt",
            "account": credentials.account_id,
        }
    except CodexAuthError as exc:
        providers["codex"] = {"status": "error", "detail": str(exc)}

    # A missing Kimi login only degrades readiness when the map routes to it,
    # so codex-only setups keep reporting healthy. The flag is exposed so the
    # dashboard can render an unused Kimi login failure as neutral, not error.
    kimi_required = config.maps_to_provider("kimi")
    try:
        await kimi_auth_manager.get_credentials()
        providers["kimi"] = {"status": "ok", "required": kimi_required}
    except KimiAuthError as exc:
        providers["kimi"] = {"status": "error", "detail": str(exc), "required": kimi_required}

    is_ready = providers["codex"]["status"] == "ok" and (
        providers["kimi"]["status"] == "ok" or not kimi_required
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
        "kimi_auth_file": str(config.kimi_auth_file),
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
    ?provider=claude|codex|kimi refreshes a single card; without it all run.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    provider = request.query_params.get("provider")
    if provider not in (None, "claude", "codex", "kimi"):
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error", "provider must be one of: claude, codex, kimi"
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


async def _probe_codex_route(codex_client: CodexClient, source: str, target: str) -> str:
    claude_request = {
        "model": source,
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
    """Send one minimal request through the gateway to verify a model pairing.

    The result — success or failure — is always a 200 with the outcome in the
    body; non-200 responses are reserved for invalid test requests themselves.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    source = body.get("source")
    target = body.get("target")
    if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error", "source and target must be non-empty strings"
            ),
            status_code=400,
        )
    source, target = source.strip(), target.strip()

    started_at = time.monotonic()

    def result(
        ok: bool, status: int | None, detail: str | None = None, response_model: str | None = None
    ) -> JSONResponse:
        return JSONResponse(
            {
                "ok": ok,
                "status": status,
                "latency_ms": int((time.monotonic() - started_at) * 1000),
                "source": source,
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
        else:
            probe = _probe_codex_route(request.app.state.codex_client, source, route.model)
        response_model = await asyncio.wait_for(probe, _CONNECTION_TEST_TIMEOUT)
    except (CodexUpstreamError, KimiUpstreamError) as exc:
        return result(False, exc.status_code, _upstream_error_message(exc.body))
    except (CodexAuthError, KimiAuthError) as exc:
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
                kimi_auth_manager = KimiAuthManager(config.kimi_auth_file, http_client)
                app.state.config = config
                app.state.admin_lock = asyncio.Lock()
                # Redeem key of a reset attempt whose outcome never came back.
                app.state.codex_reset_key = None
                app.state.codex_auth_manager = codex_auth_manager
                app.state.codex_client = CodexClient(codex_auth_manager, http_client)
                app.state.kimi_auth_manager = kimi_auth_manager
                app.state.kimi_client = KimiClient(kimi_auth_manager, http_client)
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

                yield
        finally:
            logging.getLogger().removeHandler(log_buffer)

    app = Starlette(
        routes=[
            Route("/", _handle_dashboard, methods=["GET"]),
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
            Route("/admin/test", _handle_admin_connection_test, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.daemon_nonce = daemon_nonce
    return app
