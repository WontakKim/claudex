"""Anthropic Messages relay handlers and backend routing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import secrets
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

import claudex_gateway.translate.context_overflow as context_overflow
from claudex_gateway import server_support
from claudex_gateway.claude_account_pool import (
    _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    AccountCooldownTracker,
    build_serving_chain,
    rate_limit_cooldown_seconds,
)
from claudex_gateway.claude_accounts import (
    AccountRecord,
    AccountRegistryError,
    load_registry,
)
from claudex_gateway.claude_ambient_account import AmbientAccountProvider, is_duplicate_identity
from claudex_gateway.claude_auth import (
    ClaudeAccountAuthError,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.balanced.router import ClaudeBalancedRouter
from claudex_gateway.balanced.runtime import ClaudeBalancedRuntime
from claudex_gateway.balanced.selection import (
    AccountCandidate,
    NoEligibleAccountError,
    SessionKey,
    binding_windows,
    capability_key,
    derive_session_key,
    derive_stateless_routing_digest,
    is_eligible_candidate,
    pick_weighted_hrw,
    quota_family,
    select_weights,
    warning_factor,
)
from claudex_gateway.codex_client import CODEX_FAST_TIER_WIRE_VALUE, CodexClient
from claudex_gateway.compaction import (
    build_reroute_headers,
    build_reroute_payload,
    is_compaction_request,
)
from claudex_gateway.config import GatewayConfig, RouteTarget, parse_compaction_model
from claudex_gateway.grok_client import GrokClient, sanitize_grok_payload
from claudex_gateway.kimi_auth import KimiAuthError
from claudex_gateway.kimi_client import KimiClient, KimiUpstreamError
from claudex_gateway.openai_compatible_client import OpenAICompatibleClient
from claudex_gateway.translate import (
    CodexToClaudeStreamTranslator,
    TranslationError,
    assemble_claude_message,
    translate_claude_request_to_codex,
)
from claudex_gateway.upstream_errors import UpstreamAuthError, UpstreamError

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
    request: Request, raw_body: bytes, parsed_body: Any
) -> JSONResponse | StreamingResponse:
    """Forward the request to the real Anthropic API and relay the response verbatim.

    With no claude_account.id configured, the client's own credentials and
    beta headers are forwarded untouched, so passthrough traffic behaves
    exactly as if Claude Code talked to Anthropic directly. With one
    configured, the request is served with the registered account instead —
    and when claude_account.routing selects the "fallback" mode, with the
    whole pool, serving account first (see `_passthrough_with_claude_pool`).
    The "balanced" mode is checked first, ahead of the claude_account.id
    gate: it spreads sessions across the whole registered pool by weighted
    HRW and never falls through to single-account or fallback routing. It
    dispatches only through an active `ClaudeBalancedRuntime`, otherwise it
    returns the reserved fail-closed 503.
    """
    config: GatewayConfig = request.app.state.config
    if config.claude_account_routing_mode == "balanced":
        return await _passthrough_with_claude_balanced(request, raw_body, parsed_body)
    if config.claude_account_id is not None:
        if config.claude_account_routing_mode == "fallback":
            return await _passthrough_with_claude_pool(
                request, raw_body, parsed_body, config.claude_account_id
            )
        return await _passthrough_with_claude_account(
            request, raw_body, parsed_body, config.claude_account_id
        )
    logger.info("anthropic passthrough: verbatim client credentials for %s", request.url.path)
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_REQUEST_HEADERS
    }
    try:
        upstream_response = await _send_to_anthropic(request, headers, raw_body)
    except httpx.HTTPError as exc:
        logger.warning("anthropic passthrough failed: %s", exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
            status_code=502,
        )
    return _relay_anthropic_response(upstream_response)


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


def _claude_account_unavailable(message: str) -> JSONResponse:
    # A misconfigured or broken serving account is the gateway's fault, never
    # the client's, and silently falling back to the client's own credentials
    # (usually a dummy token by now) would serve confusing 401s — fail loudly.
    return JSONResponse(server_support._claude_error_body("api_error", message), status_code=503)


def _claude_account_request_headers(request: Request, access_token: str) -> dict[str, str]:
    """Forward the client's headers with the registered account's identity.

    The caller is real Claude Code, so its own fingerprint and beta headers
    are kept; only credentials are replaced and the OAuth beta is guaranteed
    to be present (the client may have authenticated with the gateway-local
    token instead of an OAuth login of its own).
    """
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _MANAGED_RELAY_SKIP_REQUEST_HEADERS
    }
    headers["authorization"] = f"Bearer {access_token}"
    headers.setdefault("anthropic-version", "2023-06-01")
    betas = [beta.strip() for beta in headers.get("anthropic-beta", "").split(",") if beta.strip()]
    if _OAUTH_BETA not in betas:
        betas.append(_OAUTH_BETA)
    headers["anthropic-beta"] = ",".join(betas)
    return headers


def _rewrite_metadata_account_uuid(
    raw_body: bytes, parsed_body: Any, account_uuid: str | None
) -> bytes:
    """Return the body to forward, renaming metadata.user_id to the serving account.

    Claude Code's `metadata.user_id` is a JSON string embedding the client's
    own `account_uuid`; a request served with a registered account's token
    must not name a different account, so the uuid is replaced with the
    serving account's (or stripped when the capture recorded none). Bodies
    that don't match that shape are forwarded unchanged — passthrough stays
    transparent for anything that isn't the known Claude Code wire form.
    """
    if not isinstance(parsed_body, dict):
        return raw_body
    metadata = parsed_body.get("metadata")
    if not isinstance(metadata, dict):
        return raw_body
    user_id = metadata.get("user_id")
    if not isinstance(user_id, str):
        return raw_body
    try:
        parsed_user_id = json.loads(user_id)
    except json.JSONDecodeError:
        return raw_body
    if not isinstance(parsed_user_id, dict) or "account_uuid" not in parsed_user_id:
        return raw_body
    if account_uuid:
        parsed_user_id["account_uuid"] = account_uuid
    else:
        del parsed_user_id["account_uuid"]
    rewritten_body = dict(parsed_body)
    rewritten_metadata = dict(metadata)
    # Claude Code serializes user_id compactly; match it rather than
    # introducing a gateway-shaped variant of the same string.
    rewritten_metadata["user_id"] = json.dumps(
        parsed_user_id, ensure_ascii=False, separators=(",", ":")
    )
    rewritten_body["metadata"] = rewritten_metadata
    return json.dumps(rewritten_body, ensure_ascii=False).encode()


@dataclass
class _FailedAttempt:
    """A per-account failure the pool may recover from by trying the next account.

    `response` is exactly what single-account serving would have returned,
    so a one-account pool — or the last surviving failure — reproduces the
    pre-pool behavior.
    """

    response: Response
    rate_limited: bool = False


def _replay_buffered_response(status_code: int, headers: dict[str, str], body: bytes) -> Response:
    """Rebuild a fully-buffered upstream response for the client.

    httpx already decoded the transfer (gzip etc.), so the encoding and
    hop-by-hop headers must be dropped exactly as the streaming relay does.
    """
    filtered = {
        key: value
        for key, value in headers.items()
        if key.lower() not in _PASSTHROUGH_SKIP_RESPONSE_HEADERS
    }
    return Response(content=body, status_code=status_code, headers=filtered)


async def _attempt_with_account(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    record: AccountRecord,
    *,
    rate_limit_failover: bool,
    commit_hook: Callable[[httpx.Response], Awaitable[None]] | None = None,
    on_relay_complete: Callable[[], None] | None = None,
    on_quota_429: Callable[[float], Awaitable[None]] | None = None,
    on_response: Callable[[httpx.Response], Awaitable[None]] | None = None,
) -> Response | _FailedAttempt:
    """Serve an Anthropic passthrough request with one registered account's token.

    Retries exactly once with force-refreshed credentials on HTTP 401 —
    safe because no response byte has been relayed yet; a post-retry 401
    means the registered account itself was rejected, which is reported as
    such rather than relayed verbatim (a raw 401 would send the client into
    a pointless re-login of its own).

    Returns a `_FailedAttempt` for account-specific failures (auth problems,
    a 429 that just started this account's cooldown) so the pool can try the
    next account; every other outcome — success, other upstream statuses,
    and transport errors, which are not account-specific — is terminal.

    With `rate_limit_failover` off (routing mode "disabled"), a 429 is not
    an account-specific failure: it streams back verbatim with zero cooldown
    bookkeeping, exactly like any other upstream status.

    `commit_hook` and `on_relay_complete` implement the balanced runner's
    migration commit protocol. On every path that reaches the final
    `_relay_anthropic_response` call, `commit_hook` is awaited with the open
    `upstream_response` before any byte is forwarded, and `on_relay_complete`
    becomes the relay's `on_finished` hook. The 401 and 429 failover branches
    do not invoke either hook.

    `on_quota_429`, when given, is awaited with the derived cooldown immediately
    after an account-specific 429 marks the shared in-memory tracker. Balanced
    routing uses it to select the cooldown scope and persist the result.
    `on_response`, when given, is awaited with the open upstream response at
    the same point as `commit_hook`; balanced routing uses it to record eligible
    capability evidence from an explicit 2xx. All hooks default to `None`, so
    non-balanced callers need no special handling.
    """
    account_id = record.id
    manager = server_support._claude_account_auth_manager(request.app.state, account_id)
    try:
        credentials = await manager.get_credentials()
    except ClaudeAccountReauthRequiredError as exc:
        await server_support._mark_account_needs_reauth_best_effort(request.app.state, account_id)
        return _FailedAttempt(
            _claude_account_unavailable(
                f"claude account {account_id} needs re-authentication: {exc}; "
                "log in again from the dashboard or re-add it with "
                "`claudex-gateway account add`"
            )
        )
    except ClaudeAccountAuthError as exc:
        logger.info(
            "claude relay: account %.8s (%s) unusable, trying next: %s",
            account_id,
            record.email,
            exc,
        )
        return _FailedAttempt(
            _claude_account_unavailable(f"claude account {account_id} is unusable: {exc}")
        )

    logger.info(
        "claude relay: attempting %s with account %.8s (%s)",
        request.url.path,
        account_id,
        record.email,
    )
    content = _rewrite_metadata_account_uuid(raw_body, parsed_body, credentials.account_uuid)
    try:
        upstream_response = await _send_to_anthropic(
            request, _claude_account_request_headers(request, credentials.access_token), content
        )
        if upstream_response.status_code == 401:
            await upstream_response.aclose()
            try:
                credentials = await manager.get_credentials(force_refresh=True)
            except ClaudeAccountReauthRequiredError as exc:
                await server_support._mark_account_needs_reauth_best_effort(request.app.state, account_id)
                return _FailedAttempt(
                    _claude_account_unavailable(
                        f"claude account {account_id} needs re-authentication: {exc}; "
                        "log in again from the dashboard or re-add it with "
                        "`claudex-gateway account add`"
                    )
                )
            except ClaudeAccountAuthError as exc:
                logger.info(
                    "claude relay: account %.8s (%s) unusable, trying next: %s",
                    account_id,
                    record.email,
                    exc,
                )
                return _FailedAttempt(
                    _claude_account_unavailable(
                        f"claude account {account_id} was rejected and could not be "
                        f"refreshed: {exc}"
                    )
                )
            upstream_response = await _send_to_anthropic(
                request,
                _claude_account_request_headers(request, credentials.access_token),
                content,
            )
    except httpx.HTTPError as exc:
        logger.warning("anthropic passthrough failed: %s", exc)
        return JSONResponse(
            server_support._claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
            status_code=502,
        )
    if upstream_response.status_code == 401:
        await upstream_response.aclose()
        # A freshly refreshed token that Anthropic still rejects is durably
        # dead — only a human re-login recovers it, which is what the
        # needs-reauth state means.
        await server_support._mark_account_needs_reauth_best_effort(request.app.state, account_id)
        return _FailedAttempt(
            JSONResponse(
                server_support._claude_error_body(
                    "authentication_error",
                    f"Anthropic rejected the registered claude account {account_id} "
                    "after a token refresh; log in again from the dashboard or "
                    "re-add it with `claudex-gateway account add`",
                ),
                status_code=401,
            )
        )
    if upstream_response.status_code == 429 and rate_limit_failover:
        # Buffer the (small) error response and release the connection before
        # any next attempt reuses the shared client. Quota 429s carry no
        # machine-readable reset (.docs/research/claude-429-shape.md), so the
        # deadline falls back to the cached usage envelope or a short default.
        response_headers = dict(upstream_response.headers)
        response_body = await upstream_response.aread()
        await upstream_response.aclose()
        cooldown_seconds = rate_limit_cooldown_seconds(
            response_headers,
            response_body,
            request.app.state.claude_account_usage_cache.peek(account_id),
        )
        request.app.state.claude_account_cooldowns.mark(account_id, cooldown_seconds)
        logger.warning(
            "claude account %.8s rate-limited by Anthropic; cooling down for %.0fs",
            account_id,
            cooldown_seconds,
        )
        if on_quota_429 is not None:
            await on_quota_429(cooldown_seconds)
        return _FailedAttempt(
            _replay_buffered_response(429, response_headers, response_body),
            rate_limited=True,
        )
    if commit_hook is not None:
        await commit_hook(upstream_response)
    if on_response is not None:
        await on_response(upstream_response)
    logger.info(
        "claude relay: account %.8s (%s) serving %s -> upstream %s",
        account_id,
        record.email,
        request.url.path,
        upstream_response.status_code,
    )
    return _relay_anthropic_response(upstream_response, on_finished=on_relay_complete)


# ==========================================================================
# Balanced-mode durable cooldowns and capability evidence
# ==========================================================================


async def _install_balanced_quota_cooldown(
    app_state: Any,
    router: ClaudeBalancedRouter,
    *,
    account_id: str,
    account_incarnation_id: str,
    model: str,
    cooldown_seconds: float,
) -> None:
    """Install and persist a balanced-mode cooldown after an upstream 429.

    The Fable family gate is evaluated before choosing scope. A family-scoped
    cooldown uses the Fable reset as its deadline; otherwise the same
    account-wide `cooldown_seconds` applied to the shared in-memory tracker
    drives the deadline. The high-priority durable write is awaited so a
    restart does not repeat a burst of 429s against a cooling account.
    """
    now = time.monotonic()
    gate = router.classify_cooldown_scope(
        account_id=account_id, model=model, upstream_status_code=429, now=now
    )
    fingerprint = server_support._account_profile_fingerprint(app_state, account_id)
    if gate.scope == "family":
        assert gate.family_deadline is not None
        pending_write = router.install_cooldown(
            account_id=account_id,
            account_incarnation_id=account_incarnation_id,
            account_profile_fingerprint=fingerprint,
            scope="family",
            model_family=quota_family(model),
            deadline=gate.family_deadline,
            reason=gate.reason,
        )
    else:
        pending_write = router.install_cooldown(
            account_id=account_id,
            account_incarnation_id=account_incarnation_id,
            account_profile_fingerprint=fingerprint,
            scope="account",
            deadline=now + cooldown_seconds,
            reason=gate.reason,
        )
    if pending_write is not None:
        try:
            await pending_write.wait_async()
        except Exception:
            router.persistence_degraded = True


async def _record_balanced_capability_evidence(
    app_state: Any,
    router: ClaudeBalancedRouter,
    *,
    account_id: str,
    account_incarnation_id: str,
    model: str,
    upstream_response: httpx.Response,
) -> None:
    """Record balanced-mode capability evidence from a successful 2xx.

    Only `eligible` evidence for the exact capability key is recorded.
    `classify_capability_evidence` is the authoritative gate; this path never
    records `denied` evidence or infers evidence across keys.
    """
    fingerprint = server_support._account_profile_fingerprint(app_state, account_id)
    if fingerprint is None:
        return
    router.classify_capability_evidence(
        account_id=account_id,
        capability_key=capability_key(model),
        account_incarnation_id=account_incarnation_id,
        account_profile_fingerprint=fingerprint,
        status_code=upstream_response.status_code,
        evidence_source="serve_path_2xx",
    )


async def _passthrough_with_claude_account(
    request: Request, raw_body: bytes, parsed_body: Any, serving_account_id: str
) -> Response:
    """Serve a passthrough request with the single configured account.

    This is the routing mode "disabled" path: only the serving account is
    used, and every upstream status — including 429 — relays verbatim with
    no cooldown bookkeeping. The registry is re-resolved on every request
    (read-through, no cache) so CLI- and dashboard-side account changes take
    effect without a restart.
    """
    try:
        records = load_registry()
    except AccountRegistryError as exc:
        return _claude_account_unavailable(f"cannot read the claude account registry: {exc}")
    record = next((record for record in records if record.id == serving_account_id), None)
    if record is None:
        return _claude_account_unavailable(
            f"configured claude account {serving_account_id} is not registered; "
            "pick another with `claudex-gateway account use` or disable it "
            "with `claudex-gateway account use off`"
        )
    if record.state != "ready":
        return _claude_account_unavailable(
            f"configured claude account {serving_account_id} is in state "
            f"{record.state!r}, not ready"
        )
    outcome = await _attempt_with_account(
        request, raw_body, parsed_body, record, rate_limit_failover=False
    )
    if isinstance(outcome, _FailedAttempt):
        return outcome.response
    return outcome


async def _passthrough_with_claude_pool(
    request: Request, raw_body: bytes, parsed_body: Any, serving_account_id: str
) -> Response:
    """Serve a passthrough request from the account pool, ordered fallback.

    The serving account goes first, the remaining ready accounts follow in
    registration order, and accounts inside a rate-limit cooldown are
    skipped — expiry readmits them, so traffic fails back automatically.
    The registry is re-resolved on every request (read-through, no cache) so
    CLI- and dashboard-side account changes take effect without a restart.

    Failover happens strictly before any client byte: a streaming relay is
    only constructed for terminal outcomes, and every failover branch fully
    closed its upstream response first.
    """
    try:
        records = load_registry()
    except AccountRegistryError as exc:
        return _claude_account_unavailable(f"cannot read the claude account registry: {exc}")
    tracker: AccountCooldownTracker = request.app.state.claude_account_cooldowns
    chain = build_serving_chain(serving_account_id, records, tracker)

    if not chain.attempts:
        if chain.cooling_ids:
            remaining = tracker.min_remaining_seconds() or _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
            return JSONResponse(
                server_support._claude_error_body(
                    "rate_limit_error",
                    "every registered claude account is rate-limited; "
                    "retry after the cooldown",
                ),
                status_code=429,
                headers={"retry-after": str(max(1, math.ceil(remaining)))},
            )
        if not chain.serving_registered:
            return _claude_account_unavailable(
                f"configured claude account {serving_account_id} is not registered; "
                "pick another with `claudex-gateway account use` or disable it "
                "with `claudex-gateway account use off`"
            )
        if chain.serving_state != "ready":
            return _claude_account_unavailable(
                f"configured claude account {serving_account_id} is in state "
                f"{chain.serving_state!r}, not ready"
            )
        return _claude_account_unavailable("no usable claude account is registered")

    first_failure: Response | None = None
    last_rate_limited: Response | None = None
    for position, record in enumerate(chain.attempts):
        outcome = await _attempt_with_account(
            request, raw_body, parsed_body, record, rate_limit_failover=True
        )
        if not isinstance(outcome, _FailedAttempt):
            if position:
                logger.warning(
                    "claude account failover: serving with %.8s after %d failed attempt(s)",
                    record.id,
                    position,
                )
            return outcome
        if outcome.rate_limited:
            last_rate_limited = outcome.response
        elif first_failure is None:
            first_failure = outcome.response

    # Every account failed. A rate-limit reply is the most useful terminal
    # answer (the pool recovers by itself and Claude Code renders Anthropic
    # 429s natively); otherwise surface the highest-priority account's
    # failure, which is the most actionable one.
    if last_rate_limited is not None:
        return last_rate_limited
    assert first_failure is not None  # attempts was non-empty, so one branch recorded
    return first_failure


# One acquiring or draining transition should settle the request. This limit
# prevents pathological back-to-back transitions from spinning forever.
_BALANCED_TRANSITION_WAIT_LIMIT = 4


def _balanced_routing_not_active() -> JSONResponse:
    """Return the reserved 503 for an inconsistent balanced-routing state.

    This means `claude_account.routing` is published as "balanced" with no
    usable runtime outside a controlled transition. Requests arriving during
    a transition wait and dispatch under the resulting mode instead.
    """
    return _claude_account_unavailable("balanced routing is not active")


async def _passthrough_with_claude_balanced(
    request: Request, raw_body: bytes, parsed_body: Any
) -> Response:
    """Dispatch fail-closed and transition-aware through the balanced runtime.

    Only an active `ClaudeBalancedRuntime` ever serves this mode. A request that
    arrives while a controlled enable ("acquiring") or exit ("draining") transition is
    in flight awaits it (`ClaudeBalancedRuntime.wait_for_transition`), then re-reads the
    published `claude_account.routing` mode and dispatches under THAT mode — it is
    never rejected merely because a controlled transition is running. The 503
    "balanced routing is not active" is reserved for the inconsistent state outside a
    controlled transition; balanced traffic never falls through to single-account or
    fallback routing.
    """
    app_state = request.app.state
    runtime: ClaudeBalancedRuntime = app_state.claude_balanced_runtime
    for _ in range(_BALANCED_TRANSITION_WAIT_LIMIT):
        if runtime.status in ("acquiring", "draining"):
            await runtime.wait_for_transition()
            config: GatewayConfig = app_state.config
            if config.claude_account_routing_mode != "balanced":
                return await _passthrough_to_anthropic(request, raw_body, parsed_body)
            continue
        if runtime.begin_request():
            try:
                return await _passthrough_with_balanced_pool(request, raw_body, parsed_body, runtime)
            finally:
                runtime.end_request()
        break
    return _balanced_routing_not_active()


# ==========================================================================
# Balanced serve-path retries, commit-at-headers, and exhaustion responses
# ==========================================================================


def _balanced_candidates(
    records: Iterable[AccountRecord], router: ClaudeBalancedRouter, *, family: str, now: float
) -> list[AccountCandidate]:
    """One `AccountCandidate` per registered account for a request's whole retry chain.

    `account_cooldown_until` and `family_cooldown_until` are absolute monotonic
    deadlines read from the router's durable cooldown state. Balanced routing
    never consults the fallback pool's account-wide, in-memory
    `AccountCooldownTracker`. `capability_denied` remains false because the
    current classifier records only `eligible` capability evidence.
    """
    candidates = []
    for record in records:
        ready = record.state == "ready"
        candidates.append(
            AccountCandidate(
                account_id=record.id,
                account_incarnation_id=record.account_incarnation_id,
                ready=ready,
                account_cooldown_until=router.account_cooldown_deadline(record.id, now=now) if ready else None,
                family_cooldown_until=(
                    router.family_cooldown_deadline(record.id, family, now=now) if ready else None
                ),
            )
        )
    return candidates


def _balanced_pick_account(
    router: ClaudeBalancedRouter,
    *,
    session_key_digest: bytes,
    model: str,
    candidates: Sequence[AccountCandidate],
    seed: bytes,
    already_attempted: frozenset[str] = frozenset(),
    now: float | None = None,
) -> str:
    """A weighted-HRW pick against the router's live pressure/in-flight state, WITHOUT
    touching the pin map: `place_session`'s own pick logic, reconstructed here from its
    public building blocks, so unpinnable/count_tokens-fallback routing and each
    migration hop's next-target selection never insert a pin-map entry.
    """
    now = time.monotonic() if now is None else now
    eligible = [
        candidate
        for candidate in candidates
        if is_eligible_candidate(candidate, now=now, already_attempted=already_attempted)
    ]
    if not eligible:
        raise NoEligibleAccountError("no eligible account is available for balanced routing")
    family = quota_family(model)
    account_ids = [candidate.account_id for candidate in eligible]
    floor = router.candidate_set_unknown_floor(account_ids, family, now=now)
    pressures = {
        account_id: router.account_pressure(account_id, family, now=now, floor=floor)
        for account_id in account_ids
    }
    windows = binding_windows(family)
    warning_factors = {
        account_id: warning_factor(router.observations, account_id, windows, now=now)
        for account_id in account_ids
    }
    in_flight = {account_id: router.in_flight_count(account_id) for account_id in account_ids}
    weights = select_weights(
        account_ids, pressures=pressures, warning_factors=warning_factors, in_flight=in_flight
    )
    return pick_weighted_hrw(weights=weights, seed=seed, session_key_digest=session_key_digest)


def _balanced_eligible_candidate_set(
    records_by_id: Mapping[str, AccountRecord]
) -> list[AccountRecord]:
    """Return registered, ready, capability-eligible accounts, ignoring cooldowns.

    The current classifier records only `eligible` capability evidence, so
    every registered, ready account qualifies.
    """
    return [record for record in records_by_id.values() if record.state == "ready"]


def _balanced_all_cooling_response(
    records_by_id: Mapping[str, AccountRecord],
    router: ClaudeBalancedRouter,
    *,
    family: str,
    chain_exhausted_429: Response | None,
) -> Response:
    """Respond after a retry chain or initial placement finds no eligible account.

    A chain that exhausted on a real upstream 429 relays that response
    verbatim. Otherwise this synthesizes an Anthropic-compatible 429 with
    `Retry-After` based on the earliest unblock time among registered, ready,
    capability-eligible accounts. A disabled or capability-denied account
    cannot shorten that value. An empty candidate set returns 503.
    """
    if chain_exhausted_429 is not None:
        return chain_exhausted_429
    candidate_set = _balanced_eligible_candidate_set(records_by_id)
    if not candidate_set:
        return JSONResponse(
            server_support._claude_error_body(
                "api_error", "no registered account is eligible for the requested model"
            ),
            status_code=503,
        )
    now = time.monotonic()

    # An account unblocks at the later of its account-wide and family
    # deadlines; Retry-After uses the earliest unblock time across accounts.
    def _unblock_at(record: AccountRecord) -> float:
        account_deadline = router.account_cooldown_deadline(record.id, now=now) or now
        family_deadline = router.family_cooldown_deadline(record.id, family, now=now) or now
        return max(account_deadline, family_deadline)

    min_unblock_at = min(_unblock_at(record) for record in candidate_set)
    retry_after = max(1, math.ceil(min_unblock_at - now))
    return JSONResponse(
        server_support._claude_error_body(
            "rate_limit_error",
            "every eligible claude account is rate-limited; retry after the cooldown",
        ),
        status_code=429,
        headers={"retry-after": str(retry_after)},
    )


async def _passthrough_with_balanced_pool(
    request: Request, raw_body: bytes, parsed_body: Any, runtime: ClaudeBalancedRuntime
) -> Response:
    """Serve one request through an active balanced runtime.

    The registry is read through without a cache, so CLI and dashboard account
    changes take effect immediately. The session key is derived from the parsed
    body before `_rewrite_metadata_account_uuid` mutates it. Token-count requests
    use `_serve_balanced_count_tokens` instead of this placement and migration
    flow.
    """
    assert runtime.router is not None
    try:
        records = load_registry()
    except AccountRegistryError as exc:
        return _claude_account_unavailable(f"cannot read the claude account registry: {exc}")

    records_by_id = {record.id: record for record in records}
    provider: AmbientAccountProvider | None = request.app.state.claude_ambient_accounts
    if provider is not None:
        member = provider.pool_member()
        if member is not None and not is_duplicate_identity(member, records):
            records_by_id[member.record.id] = member.record
            logger.debug(
                "balanced: ambient account %.8s (%s) joined the candidate set",
                member.record.id,
                member.record.email,
            )
        elif member is not None:
            logger.debug(
                "balanced: ambient login %s suppressed, registered account has the same identity",
                member.record.email,
            )
    model = parsed_body.get("model") if isinstance(parsed_body, dict) else None
    model = model if isinstance(model, str) else ""
    # The routing identity is frozen here, once per request: the same model
    # string (and therefore the same quota family) feeds the key derivation
    # and every downstream placement/cooldown decision.
    session_key = (
        derive_session_key(parsed_body, runtime.epoch_seed, quota_family(model))
        if isinstance(parsed_body, dict)
        else None
    )

    if request.url.path.endswith("/count_tokens"):
        return await _serve_balanced_count_tokens(
            request, raw_body, parsed_body, runtime, records_by_id, session_key, model
        )
    if session_key is not None:
        return await _serve_balanced_pinned_message(
            request, raw_body, parsed_body, runtime, records_by_id, session_key, model
        )
    return await _serve_balanced_stateless_message(
        request, raw_body, parsed_body, runtime, records_by_id, model
    )


async def _serve_balanced_stateless_message(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    model: str,
) -> Response:
    """Route an unpinnable request using one fresh stateless HRW digest.

    The digest is reused for the request's complete retry chain, never
    persisted, and never inserted into the pin map.
    """
    router = runtime.router
    assert router is not None
    family = quota_family(model)
    digest = derive_stateless_routing_digest(runtime.epoch_seed, secrets.token_bytes(32))
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=time.monotonic())

    attempted: set[str] = set()
    chain_429: Response | None = None
    while True:
        try:
            account_id = _balanced_pick_account(
                router,
                session_key_digest=digest,
                model=model,
                candidates=candidates,
                seed=runtime.epoch_seed,
                already_attempted=frozenset(attempted),
            )
        except NoEligibleAccountError:
            return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=chain_429)

        attempted.add(account_id)
        record = records_by_id[account_id]

        async def _on_quota_429(
            cooldown_seconds: float, *, _account_id: str = record.id, _incarnation: str = record.account_incarnation_id
        ) -> None:
            await _install_balanced_quota_cooldown(
                request.app.state,
                router,
                account_id=_account_id,
                account_incarnation_id=_incarnation,
                model=model,
                cooldown_seconds=cooldown_seconds,
            )

        async def _on_response(
            upstream_response: httpx.Response,
            *,
            _account_id: str = record.id,
            _incarnation: str = record.account_incarnation_id,
        ) -> None:
            await _record_balanced_capability_evidence(
                request.app.state,
                router,
                account_id=_account_id,
                account_incarnation_id=_incarnation,
                model=model,
                upstream_response=upstream_response,
            )

        router.begin_attempt(account_id)
        try:
            outcome = await _attempt_with_account(
                request,
                raw_body,
                parsed_body,
                record,
                rate_limit_failover=True,
                on_quota_429=_on_quota_429,
                on_response=_on_response,
            )
        finally:
            router.end_attempt(account_id)
        if not isinstance(outcome, _FailedAttempt):
            return outcome
        chain_429 = outcome.response if outcome.rate_limited else None


async def _serve_balanced_pinned_message(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    session_key: SessionKey,
    model: str,
) -> Response:
    """Place or follow `session_key`'s pin, then serve its migration chain.

    Every request resolving a pin generation awaits its `pending_durability`
    barrier before an upstream attempt. A quota 429, existing cooldown,
    removed or disabled account, or classified account-specific auth failure
    creates a reservation and migration-attempt token before retrying another
    eligible account. The target's upstream 2xx headers commit the migration
    and await the durable pin write before any downstream byte is forwarded.
    Once a 2xx response is relayed, no cross-account retry occurs.
    """
    router = runtime.router
    assert router is not None
    family = quota_family(model)
    digest = session_key.digest
    now = time.monotonic()
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=now)

    try:
        placement = router.place_session(
            session_key=session_key, model=model, candidates=candidates, seed=runtime.epoch_seed, now=now
        )
    except NoEligibleAccountError:
        return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=None)

    if placement.created and placement.durability_barrier is not None:
        asyncio.create_task(router.submit_new_pin_durability(digest))
    await router.await_pin_durability(digest)

    pin = router.get_pin(digest)
    router.touch_pin(
        digest,
        is_message_request=True,
        key_is_live=pin is not None,
        account_still_registered=pin is not None and pin.account_id in records_by_id,
    )

    attempted: set[str] = set()
    chain_429: Response | None = None
    owner_attempt_id: str | None = None
    current_target = ""

    while True:
        pin = router.get_pin(digest)
        if pin is None:
            # Extremely rare mid-flight eviction/removal race: re-place fresh.
            try:
                placement = router.place_session(
                    session_key=session_key,
                    model=model,
                    candidates=_balanced_candidates(
                        records_by_id.values(), router, family=family, now=time.monotonic()
                    ),
                    seed=runtime.epoch_seed,
                )
            except NoEligibleAccountError:
                return _balanced_all_cooling_response(
                    records_by_id, router, family=family, chain_exhausted_429=chain_429
                )
            if placement.created and placement.durability_barrier is not None:
                asyncio.create_task(router.submit_new_pin_durability(digest))
            await router.await_pin_durability(digest)
            owner_attempt_id = None
            continue

        if owner_attempt_id is None:
            current_target = pin.account_id

        record = records_by_id.get(current_target)
        cooldown_check_now = time.monotonic()
        eligible_now = (
            record is not None
            and record.state == "ready"
            and router.account_cooldown_deadline(current_target, now=cooldown_check_now) is None
            and router.family_cooldown_deadline(current_target, family, now=cooldown_check_now) is None
        )
        attempted.add(current_target)

        if eligible_now:
            assert record is not None
            attempt_id = owner_attempt_id
            source_account = pin.account_id
            source_generation = pin.generation
            target_record = record

            async def _commit_hook(
                upstream_response: httpx.Response,
                *,
                _attempt_id: str | None = attempt_id,
                _source_account: str = source_account,
                _source_generation: int = source_generation,
                _target_record: AccountRecord = target_record,
            ) -> None:
                if _attempt_id is None or upstream_response.status_code // 100 != 2:
                    return
                commit_outcome, _committed_pin, barrier = router.commit_at_headers(
                    digest,
                    attempt_id=_attempt_id,
                    source_account=_source_account,
                    source_generation=_source_generation,
                    target_account=_target_record.id,
                    target_account_incarnation_id=_target_record.account_incarnation_id,
                    target_still_registered=_target_record.id in records_by_id,
                )
                if commit_outcome == "committed" and barrier is not None:
                    asyncio.create_task(router.submit_new_pin_durability(digest))
                    await router.await_pin_durability(digest)

            def _on_relay_complete(*, _attempt_id: str | None = attempt_id) -> None:
                # M(target) stays incremented for the whole streamed response; this
                # only releases the migration token once the stream terminates. A
                # no-op if `commit_at_headers` above never ran (not a migration leg)
                # or already resolved (or failed to resolve, e.g. cas_lost).
                if _attempt_id is not None:
                    router.resolve_migration_owner_terminal(
                        digest, attempt_id=_attempt_id, outcome="terminal_failure"
                    )

            async def _on_quota_429(
                cooldown_seconds: float,
                *,
                _account_id: str = target_record.id,
                _incarnation: str = target_record.account_incarnation_id,
            ) -> None:
                await _install_balanced_quota_cooldown(
                    request.app.state,
                    router,
                    account_id=_account_id,
                    account_incarnation_id=_incarnation,
                    model=model,
                    cooldown_seconds=cooldown_seconds,
                )

            async def _on_response(
                upstream_response: httpx.Response,
                *,
                _account_id: str = target_record.id,
                _incarnation: str = target_record.account_incarnation_id,
            ) -> None:
                await _record_balanced_capability_evidence(
                    request.app.state,
                    router,
                    account_id=_account_id,
                    account_incarnation_id=_incarnation,
                    model=model,
                    upstream_response=upstream_response,
                )

            if owner_attempt_id is not None:
                try:
                    outcome = await _attempt_with_account(
                        request,
                        raw_body,
                        parsed_body,
                        target_record,
                        rate_limit_failover=True,
                        commit_hook=_commit_hook,
                        on_relay_complete=_on_relay_complete,
                        on_quota_429=_on_quota_429,
                        on_response=_on_response,
                    )
                except BaseException:
                    # If this owner fails before constructing a response, the
                    # streaming relay's `on_finished` hook cannot run. Release the
                    # reservation and migration token here instead.
                    router.resolve_migration_owner_terminal(
                        digest, attempt_id=owner_attempt_id, outcome="terminal_failure"
                    )
                    raise
            else:
                router.begin_attempt(current_target)
                try:
                    outcome = await _attempt_with_account(
                        request,
                        raw_body,
                        parsed_body,
                        target_record,
                        rate_limit_failover=True,
                        on_quota_429=_on_quota_429,
                        on_response=_on_response,
                    )
                finally:
                    router.end_attempt(current_target)

            if not isinstance(outcome, _FailedAttempt):
                return outcome
            chain_429 = outcome.response if outcome.rate_limited else None
        else:
            chain_429 = None

        # --- Migration reservation and next-target selection ----------------
        if owner_attempt_id is not None:
            router.resolve_migration_preheader_failure(digest, attempt_id=owner_attempt_id)
            owner_attempt_id = None

        try:
            next_target = _balanced_pick_account(
                router,
                session_key_digest=session_key.scoring_digest_or_default,
                model=model,
                candidates=candidates,
                seed=runtime.epoch_seed,
                already_attempted=frozenset(attempted),
            )
        except NoEligibleAccountError:
            return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=chain_429)

        reservation, is_owner = router.acquire_migration_reservation(
            digest,
            source_account=pin.account_id,
            source_generation=pin.generation,
            target_account=next_target,
            attempt_id=uuid.uuid4().hex,
        )
        if is_owner:
            owner_attempt_id = reservation.owner_attempt_id
            current_target = next_target
        else:
            await router.wait_for_migration_reservation(reservation)
            owner_attempt_id = None
        # loop: re-reads the pin (never trusts the stale target embedded in the
        # by-then-cleared reservation a waiter just waited on).


async def _serve_balanced_count_tokens(
    request: Request,
    raw_body: bytes,
    parsed_body: Any,
    runtime: ClaudeBalancedRuntime,
    records_by_id: Mapping[str, AccountRecord],
    session_key: SessionKey | None,
    model: str,
) -> Response:
    """Serve token counting without mutating balanced-routing state.

    Follow an existing pin without refreshing `last_seen`; otherwise use
    stateless-digest placement. This path never creates a pin, reservation,
    cooldown, or capability evidence, and never retries across accounts
    (`rate_limit_failover=False`).
    """
    router = runtime.router
    assert router is not None
    if session_key is not None:
        pin = router.get_pin(session_key.digest)
        if pin is not None:
            record = records_by_id.get(pin.account_id)
            if record is not None:
                if pin.pending_durability is not None:
                    await router.await_pin_durability(session_key.digest)
                outcome = await _attempt_with_account(
                    request, raw_body, parsed_body, record, rate_limit_failover=False
                )
                return outcome.response if isinstance(outcome, _FailedAttempt) else outcome

    scoring_digest = (
        session_key.scoring_digest_or_default
        if session_key is not None
        else derive_stateless_routing_digest(runtime.epoch_seed, secrets.token_bytes(32))
    )
    family = quota_family(model)
    candidates = _balanced_candidates(records_by_id.values(), router, family=family, now=time.monotonic())
    try:
        account_id = _balanced_pick_account(
            router,
            session_key_digest=scoring_digest,
            model=model,
            candidates=candidates,
            seed=runtime.epoch_seed,
        )
    except NoEligibleAccountError:
        return _balanced_all_cooling_response(records_by_id, router, family=family, chain_exhausted_429=None)
    record = records_by_id[account_id]
    outcome = await _attempt_with_account(
        request, raw_body, parsed_body, record, rate_limit_failover=False
    )
    return outcome.response if isinstance(outcome, _FailedAttempt) else outcome


async def _handle_messages(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = server_support._require_local_token(request, claude_error=True)
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
        return await _passthrough_to_anthropic(request, raw_body, claude_request)
    if route.provider == "kimi":
        return await _relay_to_kimi(request, claude_request, route.model)
    return await _relay_via_responses_backend(request, claude_request, route)


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
        context_window = await client.context_window(upstream_model)
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


async def _handle_count_tokens(request: Request) -> JSONResponse | StreamingResponse:
    unauthorized = server_support._require_local_token(request, claude_error=True)
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
        return await _passthrough_to_anthropic(request, raw_body, body)
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
