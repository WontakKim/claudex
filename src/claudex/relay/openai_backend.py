"""Responses-backend translation and compaction relay serving path."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

import claudex.translate.context_overflow as context_overflow
from claudex import server_support
from claudex.providers.codex_client import CODEX_FAST_TIER_WIRE_VALUE, CodexClient
from claudex.compaction import (
    build_reroute_headers,
    build_reroute_payload,
    is_compaction_request,
)
from claudex.config import GatewayConfig, RouteTarget, parse_compaction_model
from claudex.providers.grok_client import GrokClient, sanitize_grok_payload
from claudex.providers.openai_compatible_client import OpenAICompatibleClient
from claudex.relay.common import (
    _ANTHROPIC_API_BASE,
    _format_sse,
    _upstream_error_to_claude,
)
from claudex.translate import (
    CodexToClaudeStreamTranslator,
    TranslationError,
    assemble_claude_message,
    translate_claude_request_to_codex,
)
from claudex.upstream_errors import UpstreamAuthError, UpstreamError

logger = logging.getLogger("claudex.server")


async def _resolve_context_window(
    config: GatewayConfig,
    client: CodexClient | GrokClient | OpenAICompatibleClient,
    provider: str,
    upstream_model: str,
) -> int | None:
    override = config.context_window_map.get(f"{provider}:{upstream_model}")
    if override is not None:
        return override
    return await client.context_window(upstream_model)


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


async def _relay_via_responses_backend(
    request: Request, claude_request: dict[str, Any], route: RouteTarget
) -> JSONResponse | StreamingResponse:
    """Relay a Messages request to a Responses-API backend.

    Codex, Grok, and custom providers consume the same translated payload and
    produce the same SSE event family, so one path serves all of them; only the
    Grok direction adds its payload sanitizer on the way out.

    Before translation, a Responses-mapped request also carries the compaction
    reroute trigger: when `config.compaction_model` is set, the
    body is a detected Claude Code compaction request (Signal A), and the
    mapped model's configured or catalog context window is a real (non-bool)
    integer the estimated prompt overflows, the request is diverted to
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
            server_support._claude_error_body("invalid_request_error", validation_error), status_code=400
        )

    custom_client = request.app.state.custom_provider_clients.get(provider)
    if custom_client is not None:
        client: CodexClient | GrokClient | OpenAICompatibleClient = custom_client
    elif provider == "grok":
        client = request.app.state.grok_client
    else:
        client = request.app.state.codex_client

    context_window: int | None = None
    context_window_resolved = False
    if config.compaction_model is not None and is_compaction_request(claude_request):
        context_window = await _resolve_context_window(config, client, provider, upstream_model)
        context_window_resolved = True
        # Booleans are `int` subclasses; a catalog entry that is literally
        # `True`/`False` (e.g. a stubbed or malformed provider response) must
        # never be treated as a real window, so the bool case is excluded
        # explicitly rather than trusting `isinstance(x, int)` alone.
        if isinstance(context_window, int) and not isinstance(context_window, bool):
            estimated_prompt_tokens = context_overflow.estimate_overflow_prompt_tokens(claude_request)
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
        service_tier = None
        if provider == "codex" and config.codex_service_tier == "fast":
            # Fast is optional: unknown models and failed catalog refreshes fall
            # back to the standard tier rather than blocking the request.
            if await client.supports_fast_tier(upstream_model):
                service_tier = CODEX_FAST_TIER_WIRE_VALUE
            else:
                logger.debug(
                    "fast tier requested but the codex catalog does not advertise it for %s",
                    upstream_model,
                )
        payload = translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            config.reasoning_effort_override,
            service_tier=service_tier,
        )
    except TranslationError as exc:
        return JSONResponse(
            server_support._claude_error_body("invalid_request_error", str(exc)), status_code=400
        )
    if provider == "grok":
        payload = sanitize_grok_payload(payload, upstream_model)
    session_id = payload["prompt_cache_key"]
    logger.info(
        "%s -> %s:%s (stream=%s, effort=%s, tier=%s, messages=%d, tools=%d)",
        claude_request.get("model", "?"),
        provider,
        upstream_model,
        bool(claude_request.get("stream")),
        (payload.get("reasoning") or {}).get("effort", "-"),
        payload.get("service_tier") or "standard",
        len(claude_request.get("messages") or []),
        len(payload.get("tools") or []),
    )

    # Resolved once per request from the exact-slug config override or the
    # provider's own catalog cache (the client is long-lived on app state).
    # Fresh-cache lookups are memory-only; a cold or expired cache may refresh
    # once before falling back to stale data or None. The compaction trigger
    # above already resolved this for a detected compaction request, so that
    # result is reused here instead of looked up again.
    if not context_window_resolved:
        context_window = await _resolve_context_window(config, client, provider, upstream_model)

    event_stream = client.stream_responses(payload, session_id)
    try:
        first_event = await anext(event_stream, None)
        if first_event is None:
            return JSONResponse(
                server_support._claude_error_body("api_error", f"{provider} stream ended without any events"),
                status_code=502,
            )
    except UpstreamError as exc:
        status_code, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        logger.warning(
            "%s upstream error %s: %s", provider, exc.status_code, body["error"]["message"]
        )
        return JSONResponse(body, status_code=status_code)
    except UpstreamAuthError as exc:
        return JSONResponse(server_support._claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("%s backend unreachable: %r", provider, exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the {provider} backend: {exc!r}"),
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
        # From here on, the relay owns upstream_response; it is never
        # closed in this function again.
        sequence = record("rerouted", None)
        return _OwnedStreamingResponse(
            _CompactionStreamRelay(upstream_response, request.app.state, sequence),
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


class _CompactionStreamRelay:
    """Relay a committed Anthropic stream unchanged; owns `upstream_response`.

    This is a hand-written async iterator rather than an async generator
    because a generator cannot honor the closure contract: `aclose()` before
    the first iteration never enters the body (so `finally` never runs), and
    code placed after a `yield` only executes if the consumer requests
    another item. Here `aclose()` releases the upstream response no matter
    when it is called, and every terminal path — exhaustion, mid-stream
    failure, cancellation — closes it inside `__anext__` itself.

    The reroute already committed (`rerouted`) before iteration starts, so a
    mid-stream `httpx.HTTPError` can only be reported in-band: an event
    boundary followed by a parseable `event: error` using the existing
    Claude error envelope, with no raw exception text. The diagnostics
    upgrade to `midstream_error`/`read_error` happens BEFORE the terminal
    chunk is handed out, so it does not depend on the consumer ever
    requesting another item, and it goes through the compare-and-swap helper
    so a stale failure never clobbers a newer request's record.
    """

    def __init__(self, upstream_response: httpx.Response, app_state: Any, sequence: int) -> None:
        self._upstream_response = upstream_response
        self._app_state = app_state
        self._sequence = sequence
        self._chunks = upstream_response.aiter_bytes()
        self._finished = False

    def __aiter__(self) -> _CompactionStreamRelay:
        return self

    async def _close_upstream(self) -> None:
        # Best-effort, idempotent release of the Anthropic HTTP stream; a
        # secondary transport error during cleanup must not mask the
        # original outcome.
        with contextlib.suppress(Exception):
            await self._upstream_response.aclose()

    async def __anext__(self) -> bytes:
        if self._finished:
            raise StopAsyncIteration
        try:
            return await self._chunks.__anext__()
        except StopAsyncIteration:
            self._finished = True
            await self._close_upstream()
            raise
        except httpx.HTTPError as exc:
            logger.warning("compaction reroute stream aborted: %r", exc)
            self._finished = True
            _replace_compaction_reroute_if_current(
                self._app_state,
                self._sequence,
                outcome="midstream_error",
                detail="read_error",
            )
            await self._close_upstream()
            return b"\n\n" + _format_sse(
                "error",
                server_support._claude_error_body("api_error", "compaction reroute stream aborted"),
            ).encode()
        except BaseException:
            # Cancellation (or any unexpected failure) while waiting on the
            # upstream read: release the stream, then let it propagate.
            self._finished = True
            await self._close_upstream()
            raise

    async def aclose(self) -> None:
        self._finished = True
        await self._close_upstream()


class _OwnedStreamingResponse(StreamingResponse):
    """StreamingResponse that always `aclose()`s its body iterator.

    Starlette's `stream_response` never closes the body iterator, so a
    client disconnect or send failure while the iterator sits between
    chunks would otherwise leak the upstream HTTP stream. The iterator's
    `aclose()` is idempotent and safe at any point, including before the
    first chunk was ever requested.
    """

    async def stream_response(self, send: Any) -> None:
        try:
            await super().stream_response(send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


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
    except UpstreamError as exc:
        _, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        yield _format_sse("error", body)
    except (UpstreamAuthError, httpx.HTTPError) as exc:
        logger.warning("responses stream aborted: %s", exc)
        error_type = (
            "authentication_error"
            if isinstance(exc, UpstreamAuthError)
            else "api_error"
        )
        yield _format_sse("error", server_support._claude_error_body(error_type, str(exc)))
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
    except UpstreamError as exc:
        status_code, body = _upstream_error_to_claude(
            exc, claude_request=claude_request, context_window=context_window
        )
        return JSONResponse(body, status_code=status_code)
    except UpstreamAuthError as exc:
        return JSONResponse(server_support._claude_error_body("authentication_error", str(exc)), status_code=401)
    except httpx.HTTPError as exc:
        logger.warning("responses stream aborted: %r", exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the upstream backend: {exc!r}"),
            status_code=502,
        )
    finally:
        # Early error returns above must not leave the upstream stream open.
        await upstream_events.aclose()

    message = assemble_claude_message(claude_events)
    if message is None:
        return JSONResponse(
            server_support._claude_error_body("api_error", "codex stream ended without a terminal response event"),
            status_code=502,
        )
    return JSONResponse(message)
