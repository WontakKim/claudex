"""Starlette application translating Anthropic Messages requests to the Codex backend."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.resources
import json
import logging
import math
import os
import re
import secrets
import time
import traceback
import uuid
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import claudex_gateway
from claudex_gateway import claude_unified_headers, paths
from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.claude_account_pool import (
    _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    AccountCooldownTracker,
    build_serving_chain,
    rate_limit_cooldown_seconds,
)
from claudex_gateway.claude_accounts import (
    AccountNotFoundError,
    AccountRecord,
    AccountRegistryError,
    list_accounts,
    load_registry,
    mark_account_needs_reauth,
    remove_account,
)
from claudex_gateway.claude_ambient_account import (
    AmbientAccountProvider,
    AmbientClaudeAuthManager,
    is_duplicate_identity,
)
from claudex_gateway.claude_auth import (
    ClaudeAccountAuthError,
    ClaudeAccountAuthManager,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.claude_account_profile import load_account_profile_fingerprint
from claudex_gateway.claude_balanced_router import (
    AccountCandidate,
    BalancedPrepareError,
    ClaudeBalancedRouter,
    ClaudeBalancedRuntime,
    ClaudeUsagePollCoordinator,
    NoEligibleAccountError,
    SessionKey,
    UsagePollAccount,
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
from claudex_gateway.claude_login_session import (
    ClaudeLoginSession,
    LoginSessionStateError,
    capture_lock_path,
)
from claudex_gateway.locking import try_file_lock
from claudex_gateway.codex_auth import CodexAuthError, CodexAuthManager
from claudex_gateway.codex_client import (
    CODEX_FAST_TIER_WIRE_VALUE,
    CodexClient,
    CodexUpstreamError,
)
from claudex_gateway.compaction import (
    build_reroute_headers,
    build_reroute_payload,
    is_compaction_request,
)
from claudex_gateway.config import (
    SETTINGS_KEYS,
    VALID_CLAUDE_ACCOUNT_ROUTING_MODES,
    VALID_LOG_LEVELS,
    ConfigError,
    GatewayConfig,
    RouteTarget,
    parse_claude_account_id,
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
    _provider_result,
    consume_codex_reset_credit,
    fetch_claude_account_usage,
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
    HRW and never falls through to single-account/fallback routing (T-10
    Step 6) — only through an active `ClaudeBalancedRuntime`, or the
    reserved fail-closed 503.
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
            _claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
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

    `on_finished` (T-11), when given, runs synchronously once the body iterator
    is fully done -- success, mid-stream failure, or cancellation alike -- right
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
                    _claude_error_body("api_error", f"anthropic stream aborted: {exc!r}"),
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
    return JSONResponse(_claude_error_body("api_error", message), status_code=503)


def _account_profile_fingerprint(app_state: Any, account_id: str) -> str | None:
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        member = provider.pool_member()
        return member.profile_fingerprint if member is not None else None
    return load_account_profile_fingerprint(paths.accounts_dir("claude") / account_id)


async def _mark_account_needs_reauth_best_effort(app_state: Any, account_id: str) -> None:
    """Persist needs-reauth for `account_id`, never failing the caller.

    A concurrent `account remove` (AccountNotFoundError) or a registry I/O
    problem must not mask the response the caller is about to return.
    """
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        logger.debug("not marking ambient claude account %s needs-reauth", account_id)
        return
    try:
        await asyncio.to_thread(mark_account_needs_reauth, account_id)
    except AccountRegistryError as exc:
        logger.warning("could not mark claude account %s needs-reauth: %s", account_id, exc)


def _claude_account_auth_manager(
    app_state: Any, account_id: str
) -> ClaudeAccountAuthManager | AmbientClaudeAuthManager:
    """Return the cached per-account manager, creating it on first use.

    Managers are keyed by account id so a `use`-switch mid-flight gets a
    fresh manager for the new directory while requests still draining on the
    old account keep theirs. No lock is needed: there is no await between
    the lookup and the insert.
    """
    provider: AmbientAccountProvider | None = app_state.claude_ambient_accounts
    if provider is not None and provider.is_ambient_account_id(account_id):
        return provider.auth_manager()
    managers: dict[str, ClaudeAccountAuthManager] = app_state.claude_account_auth_managers
    manager = managers.get(account_id)
    if manager is None:
        manager = ClaudeAccountAuthManager(
            paths.accounts_dir("claude") / account_id, app_state.http_client
        )
        managers[account_id] = manager
    return manager


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

    `commit_hook`/`on_relay_complete` (T-11) are the balanced runner's minimal
    extension for the migration commit protocol: when this attempt is about to
    relay (every path that reaches the final `_relay_anthropic_response` call,
    not the 401/429-failover branches above), `commit_hook` — if given — is
    awaited with the still-open `upstream_response` before any byte is
    forwarded, and `on_relay_complete` is threaded through as the relay's
    `on_finished` hook. Both default to `None`, which reproduces the prior
    behavior exactly.

    `on_quota_429`/`on_response` (T-12) are the balanced runner's own minimal
    extension: `on_quota_429`, if given, is awaited with the just-derived
    `cooldown_seconds` right after this account-specific 429 marks the shared
    in-memory tracker (unchanged, fallback-mode behavior) — the balanced
    caller uses it to classify the design v2 §6.4 family gate and install the
    resulting cooldown durably. `on_response`, if given, is awaited with the
    still-open `upstream_response` at the same point as `commit_hook` — the
    balanced caller uses it to record eligible capability evidence on an
    explicit 2xx (adjudication G). Both default to `None`, which reproduces
    the prior behavior exactly.
    """
    account_id = record.id
    manager = _claude_account_auth_manager(request.app.state, account_id)
    try:
        credentials = await manager.get_credentials()
    except ClaudeAccountReauthRequiredError as exc:
        await _mark_account_needs_reauth_best_effort(request.app.state, account_id)
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
                await _mark_account_needs_reauth_best_effort(request.app.state, account_id)
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
            _claude_error_body("api_error", f"failed to reach the Anthropic API: {exc}"),
            status_code=502,
        )
    if upstream_response.status_code == 401:
        await upstream_response.aclose()
        # A freshly refreshed token that Anthropic still rejects is durably
        # dead — only a human re-login recovers it, which is what the
        # needs-reauth state means.
        await _mark_account_needs_reauth_best_effort(request.app.state, account_id)
        return _FailedAttempt(
            JSONResponse(
                _claude_error_body(
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
# Balanced-mode durable cooldowns and capability evidence (T-12, design v2
# §6.4/§5.5, adjudication G)
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
    """Balanced-mode 429 cooldown installation: classify the Fable family gate
    (design v2 §6.4) before choosing scope. A FAMILY-scoped cooldown uses the
    gate's own Fable reset as its deadline; otherwise the SAME account-wide
    `cooldown_seconds` the shared in-memory tracker was just marked with
    drives the deadline. Always awaits the durable write (§5.5: cooldown
    installation is high priority) so a restart never repeats a burst of
    429s against an account that just cooled down.
    """
    now = time.monotonic()
    gate = router.classify_cooldown_scope(
        account_id=account_id, model=model, upstream_status_code=429, now=now
    )
    fingerprint = _account_profile_fingerprint(app_state, account_id)
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
    """Balanced-mode successful-2xx capability-evidence recording (adjudication
    G): v1 records ONLY `eligible` evidence, and only from an explicit
    successful 2xx for the exact capability key — `classify_capability_evidence`
    itself is the one true gate (never `denied`, never inferred across keys).
    """
    fingerprint = _account_profile_fingerprint(app_state, account_id)
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


async def _ingest_balanced_unified_headers(
    app_state: Any,
    router: ClaudeBalancedRouter,
    *,
    account_id: str,
    account_incarnation_id: str,
    model: str,
    upstream_response: httpx.Response,
) -> None:
    """Balanced-mode unified-header ingestion (T-15): merge the responding
    account's own upstream 2xx `anthropic-ratelimit-*` headers into the
    router's live usage state before any byte relays. A non-2xx response is
    never a source of usage data, and the currently-empty, committed
    `claude_unified_headers.RECOGNIZED_HEADERS` table (no T-14 capture was
    available at commit time) makes every call inert until it is
    regenerated — checked here first so an inert build never pays the
    fingerprint lookup below on every request.
    """
    if upstream_response.status_code // 100 != 2:
        return
    if not claude_unified_headers.RECOGNIZED_HEADERS:
        return
    fingerprint = _account_profile_fingerprint(app_state, account_id)
    router.ingest_unified_response_headers(
        upstream_response.headers,
        account_id=account_id,
        serving_account_id=account_id,
        account_incarnation_id=account_incarnation_id,
        account_profile_fingerprint=fingerprint,
        model=model,
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
                _claude_error_body(
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


# The transition-aware wait loop (Step 6) re-checks at most this many times: one
# controlled acquiring/draining transition settling is the expected case, so this only
# bounds a pathological back-to-back-transitions edge case from spinning forever.
_BALANCED_TRANSITION_WAIT_LIMIT = 4


def _balanced_routing_not_active() -> JSONResponse:
    """The reserved 503 for the *inconsistent* state (Step 6): claude_account.routing
    persisted/published as "balanced" with no usable balanced runtime, outside a
    controlled acquiring/draining transition. Never used merely because a controlled
    transition is in flight — those requests await it and dispatch under the
    post-transition mode instead (`_passthrough_with_claude_balanced`).
    """
    return _claude_account_unavailable("balanced routing is not active")


async def _passthrough_with_claude_balanced(
    request: Request, raw_body: bytes, parsed_body: Any
) -> Response:
    """Dispatch fail-closed and transition-aware through the balanced runtime (Step 6).

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
# Balanced serve-path chain runner (T-11): §4.5 commit-at-headers protocol
# and §6.5 exhaustion responses, analogous to `_passthrough_with_claude_pool`.
# ==========================================================================


def _balanced_candidates(
    records: Iterable[AccountRecord], router: ClaudeBalancedRouter, *, family: str, now: float
) -> list[AccountCandidate]:
    """One `AccountCandidate` per registered account for a request's whole retry chain.

    `account_cooldown_until`/`family_cooldown_until` are absolute monotonic
    deadlines read from the router's OWN durable cooldown state (T-12) — the
    balanced-mode fallback pool's shared in-memory `AccountCooldownTracker` is
    no longer consulted for balanced eligibility (it stays account-wide-only
    and fallback-mode-only, design v2 §6.4). `capability_denied` stays at its
    default: v1 never records `denied` capability evidence (adjudication G).
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
    """§6.5's spec-gate-pinned candidate set `C`: registered AND ready AND
    capability-not-denied, ignoring cooldowns. `capability_denied` never fires in
    v1 (only `eligible` capability evidence is ever recorded, adjudication G), so
    every registered, ready account currently qualifies.
    """
    return [record for record in records_by_id.values() if record.state == "ready"]


def _balanced_all_cooling_response(
    records_by_id: Mapping[str, AccountRecord],
    router: ClaudeBalancedRouter,
    *,
    family: str,
    chain_exhausted_429: Response | None,
) -> Response:
    """§6.5: a retry chain (or an initial placement) found no eligible candidate.

    A chain that just exhausted on a real upstream 429 relays THAT response verbatim.
    Otherwise this synthesizes the local Anthropic-compatible 429, with
    `Retry-After` clamped to the earliest `unblock_at` over the candidate set `C` — a
    disabled (not ready) or capability-denied account never enters `C`, so its own
    cooldown deadline, however soon, can never shorten `Retry-After` — or, when `C`
    is empty, the adjudicated 503.
    """
    if chain_exhausted_429 is not None:
        return chain_exhausted_429
    candidate_set = _balanced_eligible_candidate_set(records_by_id)
    if not candidate_set:
        return JSONResponse(
            _claude_error_body(
                "api_error", "no registered account is eligible for the requested model"
            ),
            status_code=503,
        )
    now = time.monotonic()

    # unblock_at(a, family) = max(account-wide deadline, family deadline) or now
    # (§6.5); the per-account MAX across scopes, then the MIN across accounts.
    def _unblock_at(record: AccountRecord) -> float:
        account_deadline = router.account_cooldown_deadline(record.id, now=now) or now
        family_deadline = router.family_cooldown_deadline(record.id, family, now=now) or now
        return max(account_deadline, family_deadline)

    min_unblock_at = min(_unblock_at(record) for record in candidate_set)
    retry_after = max(1, math.ceil(min_unblock_at - now))
    return JSONResponse(
        _claude_error_body(
            "rate_limit_error",
            "every eligible claude account is rate-limited; retry after the cooldown",
        ),
        status_code=429,
        headers={"retry-after": str(retry_after)},
    )


async def _passthrough_with_balanced_pool(
    request: Request, raw_body: bytes, parsed_body: Any, runtime: ClaudeBalancedRuntime
) -> Response:
    """Serve one request through an active balanced runtime (T-11).

    Mirrors `_passthrough_with_claude_pool`'s read-through registry pattern (no cache,
    so CLI/dashboard account changes take effect immediately). The session key is
    derived from the parsed body BEFORE `_rewrite_metadata_account_uuid`'s mutation
    (that rewrite only ever happens later, inside `_attempt_with_account`).
    `/v1/messages/count_tokens` follows the council-pinned rule instead of this
    function's own placement/migration flow: see `_serve_balanced_count_tokens`.
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
    """No session key is derivable: weighted-HRW routing over one fresh stateless
    digest (T-4 `derive_stateless_routing_digest`), reused across this request's
    whole retry chain and never persisted — no pin-map entry is ever created.
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
            await _ingest_balanced_unified_headers(
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
    """Place or follow `session_key`'s pin, then serve it with the full §4.2-§4.5
    migration chain: every request resolving a pin generation awaits its
    `pending_durability` barrier before an upstream attempt; a §4.2 migration
    trigger (quota 429, already cooling, removed/disabled, or a positively
    classified account-specific auth failure — `_attempt_with_account`'s own
    `_FailedAttempt` classification) creates a reservation + migration-attempt
    token and retries the next eligible account (retry-chain exclusion); the
    target's upstream 2xx headers run the T-8 commit section and await the
    durable pin write before ANY downstream byte is forwarded; once 2xx is
    relayed there is no cross-account retry.
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
                await _ingest_balanced_unified_headers(
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
                    # Cancellation (or any other failure) before this owner's
                    # response was ever constructed: the streaming relay's own
                    # `on_finished` hook never gets a chance to run, so release the
                    # reservation/token here instead (T-8 Step 3).
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

        # --- §4.2 migration trigger -----------------------------------------
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
    """Council-pinned count_tokens rule: follow an existing pin (no `last_seen`
    refresh) when one resolves, else fall back to the same stateless-digest
    placement — NEVER creating a pin, reservation, cooldown, or capability
    evidence, and never retrying across accounts (`rate_limit_failover=False`).
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
        return await _passthrough_to_anthropic(request, raw_body, claude_request)
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
        service_tier = None
        if provider == "codex" and config.codex_service_tier == "fast":
            # Fast is optional: unknown models and failed catalog refreshes fall
            # back to the standard tier rather than blocking the request.
            if await client.supports_fast_tier(upstream_model):
                service_tier = CODEX_FAST_TIER_WIRE_VALUE
        payload = translate_claude_request_to_codex(
            claude_request,
            upstream_model,
            config.reasoning_effort_override,
            service_tier=service_tier,
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
                _claude_error_body("api_error", "compaction reroute stream aborted"),
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


# The single runtime-editable field on the compaction admin surface.
_COMPACTION_KEYS = ("model",)


def _compaction_payload(config: GatewayConfig, app_state: Any) -> dict[str, Any]:
    """Pinned {model, env_locked, last_reroute} envelope for /admin/compaction.

    `model` is the raw "claude:<id>" value (or None), so a GET/PUT round-trip
    is loss-free. `env_locked` is a plain boolean — true whenever
    CLAUDEX_COMPACTION_MODEL is present in the environment, including an
    empty value, mirroring _resolve's own "present even if empty" env
    precedence. `last_reroute` is exactly the pinned seven-key diagnostics
    record from _assign_compaction_reroute (or None before any reroute has
    been attempted); its internal sequence counter is never part of the
    record and so never serialized here.
    """
    env_name = SETTINGS_KEYS["compaction.model"]
    return {
        "model": config.compaction_model,
        "env_locked": os.environ.get(env_name) is not None,
        "last_reroute": app_state.compaction_last_reroute,
    }


async def _handle_admin_compaction_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_compaction_payload(request.app.state.config, request.app.state))


async def _handle_admin_compaction_put(request: Request) -> JSONResponse:
    """Set or clear the compaction reroute target.

    Only a successful PUT returns the state envelope; the 409 (env-locked)
    path uses the existing admin error envelope, so a caller needing current
    state issues a GET afterward.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_COMPACTION_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_COMPACTION_KEYS)}",
            ),
            status_code=400,
        )
    if "model" not in body:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error", "provide 'model' (a string or null)"
            ),
            status_code=400,
        )

    value = body["model"]
    if value is not None:
        if not isinstance(value, str):
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error", "model must be a string or null"
                ),
                status_code=400,
            )
        try:
            parse_compaction_model(value)
        except ConfigError as exc:
            return JSONResponse(
                _openai_error_body("invalid_request_error", str(exc)),
                status_code=400,
            )

    # An environment variable outranks settings.json at every boot (even set
    # to an empty string), so a persisted change would silently vanish on
    # restart — refuse before the lock or any file/config read.
    env_name = SETTINGS_KEYS["compaction.model"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"compaction.model; unset it to manage the setting at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            if value is None:
                # A disabled setting is represented by the key's absence, so
                # a JSON null is never persisted.
                update_settings_file(
                    config.settings_file, {}, deletions=("compaction.model",)
                )
            else:
                update_settings_file(config.settings_file, {"compaction.model": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically; in-flight
        # requests keep their config snapshot.
        new_config = replace(config, compaction_model=value)
        request.app.state.config = new_config
    return JSONResponse(_compaction_payload(new_config, request.app.state))


# The single runtime-editable field on the claude-account admin surface.
_CLAUDE_ACCOUNT_KEYS = ("account_id",)


def _claude_account_payload(config: GatewayConfig) -> dict[str, Any]:
    """Pinned {account_id, env_locked} envelope for claude pool/serving.

    `account_id` is the raw canonical uuid (or None), so a GET/PUT
    round-trip is loss-free. `env_locked` mirrors the compaction envelope:
    true whenever CLAUDEX_CLAUDE_ACCOUNT_ID is present in the environment,
    including an empty value.
    """
    env_name = SETTINGS_KEYS["claude_account.id"]
    return {
        "account_id": config.claude_account_id,
        "env_locked": os.environ.get(env_name) is not None,
    }


async def _handle_admin_claude_serving_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_claude_account_payload(request.app.state.config))


def _claude_account_env_locked() -> JSONResponse | None:
    """409 when CLAUDEX_CLAUDE_ACCOUNT_ID overrides runtime writes.

    An environment variable outranks settings.json at every boot (even set
    to an empty string), so a persisted change would silently vanish on
    restart — refuse before the lock or any file/config read.
    """
    env_name = SETTINGS_KEYS["claude_account.id"]
    if os.environ.get(env_name) is None:
        return None
    return JSONResponse(
        _openai_error_body(
            "invalid_request_error",
            f"{env_name} is set in the gateway's environment and overrides "
            f"claude_account.id; unset it to manage the setting at runtime",
        ),
        status_code=409,
    )


async def _handle_admin_claude_serving_put(request: Request) -> JSONResponse:
    """Pin the registered account serving Anthropic passthrough.

    The pin is cleared with DELETE, never with a null PUT — the two writes
    stay distinct so a partial payload can't silently disable serving.
    Only a successful write returns the state envelope; the 409
    (env-locked) path uses the existing admin error envelope, so a caller
    needing current state issues a GET afterward.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied

    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_ACCOUNT_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_CLAUDE_ACCOUNT_KEYS)}",
            ),
            status_code=400,
        )
    value = body.get("account_id")
    if not isinstance(value, str):
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "provide 'account_id' as a string; to clear the serving "
                "account, DELETE this endpoint instead",
            ),
            status_code=400,
        )
    try:
        parse_claude_account_id(value)
    except ConfigError as exc:
        return JSONResponse(
            _openai_error_body("invalid_request_error", str(exc)),
            status_code=400,
        )
    # The env lock dooms every write, so it is checked before any registry
    # I/O — a registry hiccup must not turn the required 409 into a 500.
    denied = _claude_account_env_locked()
    if denied is not None:
        return denied

    async with request.app.state.admin_lock:
        # Selecting an unregistered account would turn every passthrough
        # request into a 503, so refuse it here where the mistake is cheap.
        # The membership check runs under the same lock that serializes the
        # accounts/{id} DELETE, so pinning and removal are linearizable
        # within this daemon (a concurrent CLI `account remove` can still
        # race from another process; the serve path re-resolves per request
        # and degrades to a loud 503 by design).
        try:
            records = load_registry()
        except AccountRegistryError as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"cannot read the claude account registry: {exc}"
                ),
                status_code=500,
            )
        if not any(record.id == value for record in records):
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error",
                    f"no account registered with id {value}; "
                    "see `claudex-gateway account list`",
                ),
                status_code=400,
            )
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(config.settings_file, {"claude_account.id": value})
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        # Swap only after the file write succeeded, atomically; in-flight
        # requests keep their config snapshot.
        new_config = replace(config, claude_account_id=value)
        request.app.state.config = new_config
    return JSONResponse(_claude_account_payload(new_config))


async def _handle_admin_claude_serving_delete(request: Request) -> JSONResponse:
    """Clear the serving pin: passthrough forwards client credentials again.

    A disabled setting is represented by the key's absence, so a JSON null
    is never persisted. Clearing an already-clear pin is a no-op 200.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    denied = _claude_account_env_locked()
    if denied is not None:
        return denied

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        try:
            update_settings_file(
                config.settings_file, {}, deletions=("claude_account.id",)
            )
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        new_config = replace(config, claude_account_id=None)
        request.app.state.config = new_config
    return JSONResponse(_claude_account_payload(new_config))


_CLAUDE_ROUTING_KEYS = ("mode",)


def _claude_routing_payload(config: GatewayConfig) -> dict[str, Any]:
    """Pinned {mode, env_locked} envelope for claude pool/routing."""
    env_name = SETTINGS_KEYS["claude_account.routing"]
    return {
        "mode": config.claude_account_routing_mode,
        "env_locked": os.environ.get(env_name) is not None,
    }


async def _handle_admin_claude_routing_get(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse(_claude_routing_payload(request.app.state.config))


def _persist_claude_routing_mode(config: GatewayConfig, mode: str) -> None:
    """Write `claude_account.routing` to `mode`'s on-disk representation.

    "disabled" is represented by the settings key's absence; every other
    mode ("fallback", "balanced") persists the policy document
    ({"mode": mode}), whose object form leaves room for a mode to carry its
    own config block without renaming the key.
    """
    if mode == "disabled":
        update_settings_file(config.settings_file, {}, deletions=("claude_account.routing",))
    else:
        update_settings_file(config.settings_file, {"claude_account.routing": {"mode": mode}})


async def _handle_admin_claude_routing_put(request: Request) -> JSONResponse:
    """Select the pool routing policy: "disabled", "fallback", or "balanced".

    Enabling ("disabled"/"fallback" -> "balanced") and intentionally exiting
    ("balanced" -> "disabled"/"fallback") balanced routing are both
    transactional under `admin_lock` (T-10 Step 5): enabling prepares the
    complete `ClaudeBalancedRuntime` — opening the runtime store, restoring
    its state, constructing the router, and verifying every ready account's
    T-3 profile_fingerprint — while the OLD mode keeps serving, and persists
    settings only once every check passes, immediately before publishing the
    prepared runtime; a failure at any point tears preparation down and
    leaves the old mode untouched. Exiting drains in-flight balanced
    dispatch, durably PERSISTS the target mode first (T-20, fix for gap
    G-3 — a failure here aborts the exit, leaving the runtime "active" with
    its epoch and pins untouched and this handler returning 500 with the
    mode unchanged), THEN rotates (invalidates) the epoch, THEN publishes
    the target mode in memory before waking any request that arrived
    mid-transition. Switching between "disabled" and "fallback" is
    unaffected — the pre-existing settings-file swap.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    mode = body.get("mode")
    unknown = sorted(set(body) - set(_CLAUDE_ROUTING_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; "
                f"supported: {', '.join(_CLAUDE_ROUTING_KEYS)}",
            ),
            status_code=400,
        )
    if mode not in VALID_CLAUDE_ACCOUNT_ROUTING_MODES:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "provide 'mode' as one of "
                f"{', '.join(VALID_CLAUDE_ACCOUNT_ROUTING_MODES)}",
            ),
            status_code=400,
        )
    env_name = SETTINGS_KEYS["claude_account.routing"]
    if os.environ.get(env_name) is not None:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"{env_name} is set in the gateway's environment and overrides "
                f"claude_account.routing; unset it to manage the setting at runtime",
            ),
            status_code=409,
        )

    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        current_mode = config.claude_account_routing_mode
        runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime

        if mode == "balanced" and current_mode != "balanced":
            lease = getattr(request.app.state, "claude_pool_lease", None)
            if lease is None:
                return JSONResponse(
                    _openai_error_body(
                        "server_error",
                        "the claude account pool lease is not held; balanced "
                        "routing cannot be enabled",
                    ),
                    status_code=500,
                )
            try:
                accounts = list_accounts()
            except AccountRegistryError as exc:
                return JSONResponse(
                    _openai_error_body(
                        "server_error", f"cannot read the claude account registry: {exc}"
                    ),
                    status_code=500,
                )

            def _persist_balanced() -> None:
                _persist_claude_routing_mode(config, "balanced")
                request.app.state.config = replace(config, claude_account_routing_mode="balanced")

            try:
                await runtime.prepare_and_publish(
                    accounts=accounts,
                    accounts_root=paths.accounts_dir("claude"),
                    runtime_db_path=paths.claude_account_pool_runtime_db(),
                    persist=_persist_balanced,
                    entry="admin_enable",
                    usage_cache=request.app.state.claude_account_usage_cache,
                )
            except BalancedPrepareError as exc:
                return JSONResponse(
                    _openai_error_body("invalid_request_error", str(exc)), status_code=400
                )
            except Exception as exc:
                return JSONResponse(
                    _openai_error_body(
                        "server_error", f"could not enable balanced routing: {exc}"
                    ),
                    status_code=500,
                )
            return JSONResponse(_claude_routing_payload(request.app.state.config))

        if mode != "balanced" and current_mode == "balanced":

            def _persist_target() -> None:
                _persist_claude_routing_mode(config, mode)

            def _publish_target() -> None:
                request.app.state.config = replace(config, claude_account_routing_mode=mode)

            try:
                await runtime.exit_mode(mode, persist=_persist_target, publish=_publish_target)
            except (ConfigError, OSError) as exc:
                return JSONResponse(
                    _openai_error_body(
                        "server_error", f"could not persist settings: {exc}"
                    ),
                    status_code=500,
                )
            return JSONResponse(_claude_routing_payload(request.app.state.config))

        try:
            _persist_claude_routing_mode(config, mode)
        except (ConfigError, OSError) as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not persist settings: {exc}"
                ),
                status_code=500,
            )
        new_config = replace(config, claude_account_routing_mode=mode)
        request.app.state.config = new_config
    return JSONResponse(_claude_routing_payload(new_config))


# --------------------------------------------------------------------------
# Balanced-mode usage isolation (T-13): while balanced routing is the
# currently PUBLISHED and ACTIVE mode, usage reads are cache-only (never
# `ClaudeAccountUsageCache.get`/upstream) and manual refresh only ever
# enqueues on the coordinator -- fallback/disabled mode is entirely
# untouched by any of this and keeps the pre-existing fetch path/envelope.
# --------------------------------------------------------------------------

_USAGE_WINDOW_FRESH_MAX_AGE_SECONDS = 5 * 60
_USAGE_WINDOW_AGING_MAX_AGE_SECONDS = 30 * 60


def _active_balanced_runtime(request: Request) -> ClaudeBalancedRuntime | None:
    """The live runtime iff "balanced" is the currently PUBLISHED routing mode
    AND the runtime itself is active -- the exact isolation boundary Steps
    4/5/6 draw between balanced-only usage behavior and every other mode. A
    non-balanced request must never see this as non-`None` (Step 5's "never
    queued for a coordinator that is not running").
    """
    config: GatewayConfig = request.app.state.config
    if config.claude_account_routing_mode != "balanced":
        return None
    runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime
    return runtime if runtime.status == "active" else None


def _usage_window_state(age_seconds: float) -> str:
    """Per-window freshness label for the balanced-mode usage read (Step 4)."""
    if age_seconds <= _USAGE_WINDOW_FRESH_MAX_AGE_SECONDS:
        return "fresh"
    if age_seconds <= _USAGE_WINDOW_AGING_MAX_AGE_SECONDS:
        return "aging"
    return "stale"



# The account-level binding-window pair required for `fresh` (§2.1):
# envelope names for `five_hour`/`seven_day`. `fable_weekly` is a scoped,
# Fable-only extra window and must never be required for -- or substitute
# for a missing member of -- this pair.
_USAGE_FRESHNESS_BINDING_WINDOWS = ("session", "weekly")


def _compute_usage_freshness(
    ready_ids: list[str], cache: ClaudeAccountUsageCache, *, persistence_degraded: bool
) -> tuple[str, dict[str, Any]]:
    """Step 6's aggregate `usage_freshness` plus per-account diagnostics.

    `"fresh"`: every ready account has BOTH binding windows
    (`_USAGE_FRESHNESS_BINDING_WINDOWS`) present, each at most 5 minutes
    old -- a ready account missing either window does not count as fresh,
    even if every window it does have is recent. `"degraded"`: persistence
    is degraded, or no window across the whole ready set is at most 30
    minutes old. Otherwise `"partial"`.
    """
    per_account: dict[str, Any] = {}
    all_fresh = True
    any_within_degraded_window = False
    for account_id in ready_ids:
        peeked = cache.peek_with_metadata(account_id)
        if peeked is None or not peeked[1]:
            all_fresh = False
            per_account[account_id] = {"oldest_age_seconds": None, "window_count": 0}
            continue
        _, metadata = peeked
        ages = [window["age_seconds"] for window in metadata.values()]
        per_account[account_id] = {
            "oldest_age_seconds": max(ages),
            "window_count": len(ages),
        }
        binding_ages = [
            metadata[window_name]["age_seconds"]
            for window_name in _USAGE_FRESHNESS_BINDING_WINDOWS
            if window_name in metadata
        ]
        if (
            len(binding_ages) < len(_USAGE_FRESHNESS_BINDING_WINDOWS)
            or max(binding_ages) > _USAGE_WINDOW_FRESH_MAX_AGE_SECONDS
        ):
            all_fresh = False
        if min(ages) <= _USAGE_WINDOW_AGING_MAX_AGE_SECONDS:
            any_within_degraded_window = True

    if persistence_degraded or (ready_ids and not any_within_degraded_window):
        return "degraded", per_account
    if all_fresh:
        return "fresh", per_account
    return "partial", per_account


async def _handle_admin_claude_pool_status(request: Request) -> JSONResponse:
    """Per-account routing state: what the serving chain would see right now.

    This is telemetry over the registry plus the daemon-memory cooldown
    tracker — never the configured pin, which lives at pool/serving. While
    balanced routing is active, this also carries the balanced `usage_freshness`
    diagnostic (Step 6); in every other mode both fields are `None`.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            _openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    tracker: AccountCooldownTracker = request.app.state.claude_account_cooldowns
    members: list[dict[str, Any]] = []
    for record in records:
        if record.state != "ready":
            members.append(
                {
                    "account_id": record.id,
                    "routing_state": "unavailable",
                    "reason": record.state,
                }
            )
            continue
        cooldown_until = _cooling_down_until_millis(tracker, record.id)
        if cooldown_until is not None:
            members.append(
                {
                    "account_id": record.id,
                    "routing_state": "cooldown",
                    "cooldown_until": cooldown_until,
                }
            )
        else:
            members.append({"account_id": record.id, "routing_state": "ready"})

    usage_freshness: str | None = None
    usage_diagnostics: dict[str, Any] | None = None
    runtime = _active_balanced_runtime(request)
    if runtime is not None:
        ready_ids = [record.id for record in records if record.state == "ready"]
        cache: ClaudeAccountUsageCache = request.app.state.claude_account_usage_cache
        persistence_degraded = runtime.router.persistence_degraded if runtime.router is not None else True
        usage_freshness, per_account = _compute_usage_freshness(
            ready_ids, cache, persistence_degraded=persistence_degraded
        )
        coordinator = runtime.usage_poll_coordinator
        usage_diagnostics = {
            "persistence_degraded": persistence_degraded,
            "accounts": per_account,
            "coordinator": vars(coordinator.diagnostics()) if coordinator is not None else None,
        }
    return JSONResponse(
        {
            "members": members,
            "usage_freshness": usage_freshness,
            "usage_diagnostics": usage_diagnostics,
        }
    )


# --------------------------------------------------------------------------
# Admin claude-accounts surface (dashboard account management)
# --------------------------------------------------------------------------

_CLAUDE_LOGIN_CODE_KEYS = ("code",)
_CLAUDE_LOGIN_REPLACE_KEYS = ("existing_account_id",)
_LOGIN_ATTEMPT_HEADER = "x-login-attempt"
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


def _require_login_attempt(
    request: Request, session: ClaudeLoginSession | None
) -> JSONResponse | None:
    """409 unless X-Login-Attempt names the active session's attempt.

    Runs immediately after the admin guard — before content-type, body,
    or state validation — so a stale caller always learns it is stale
    instead of receiving an incidental 400/415/state error first. With no
    active session there is no attempt any header could name, so every
    attached call is stale by definition.
    """
    if (
        session is not None
        and request.headers.get(_LOGIN_ATTEMPT_HEADER) == session.attempt_id
    ):
        return None
    return JSONResponse(
        _openai_error_body(
            "invalid_request_error",
            "X-Login-Attempt does not name the active login attempt; "
            "re-attach via GET /admin/providers/claude/login",
            "stale_login",
        ),
        status_code=409,
    )


def _account_usage_fetch(app_state: Any) -> Any:
    """The usage cache's fetch closure: account id -> (result, retry_after).

    Resolves the per-account auth manager lazily and converts a conclusive
    invalid_grant into a registry needs-reauth mark plus an "unavailable"
    envelope — the cache itself must never see that exception.
    """

    async def fetch(account_id: str) -> tuple[dict[str, Any], float | None]:
        manager = _claude_account_auth_manager(app_state, account_id)
        try:
            return await fetch_claude_account_usage(app_state.http_client, manager)
        except ClaudeAccountReauthRequiredError:
            await _mark_account_needs_reauth_best_effort(app_state, account_id)
            return (
                _provider_result(
                    "claude",
                    status="unavailable",
                    error="account needs re-authentication; log in again from the dashboard",
                ),
                None,
            )

    return fetch


def _local_claude_login_fields() -> dict[str, Any] | None:
    """Identity snapshot of this machine's ambient Claude Code login.

    Read from the CLI's own config file (`~/.claude.json`, or
    `$CLAUDE_CONFIG_DIR/.claude.json` when the override is set): the same
    `oauthAccount` block a capture snapshots — identity and plan metadata,
    never secrets. The dashboard's accounts screen shows it as the "로컬
    CLI 로그인" hero, which is informational only and unrelated to serving.
    Missing or malformed files degrade to None (no local login).
    """
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    path = (
        Path(override).expanduser() / ".claude.json"
        if override
        else Path.home() / ".claude.json"
    )
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    account = parsed.get("oauthAccount")
    if not isinstance(account, dict):
        return None

    def _text_field(key: str) -> str | None:
        value = account.get(key)
        return value if isinstance(value, str) and value else None

    email = _text_field("emailAddress")
    if email is None:
        return None
    return {
        "accountUuid": _text_field("accountUuid"),
        "email": email,
        "organizationName": _text_field("organizationName"),
        "planType": _text_field("organizationType"),
        "rateLimitTier": _text_field("organizationRateLimitTier"),
    }


def _account_plan_fields(account_id: str) -> dict[str, Any]:
    """Plan metadata from the account's captured oauth-account.json.

    `organizationType` (e.g. claude_max) and `organizationRateLimitTier`
    (e.g. default_claude_max_20x) are login-time snapshots — refreshed only
    by a re-login — and the file holds no secrets. Missing or malformed
    files degrade to nulls; the account list never fails over plan info.
    """
    path = paths.accounts_dir("claude") / account_id / "oauth-account.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {"planType": None, "rateLimitTier": None}
    if not isinstance(parsed, dict):
        return {"planType": None, "rateLimitTier": None}
    organization_type = parsed.get("organizationType")
    rate_limit_tier = parsed.get("organizationRateLimitTier")
    return {
        "planType": organization_type if isinstance(organization_type, str) else None,
        "rateLimitTier": rate_limit_tier if isinstance(rate_limit_tier, str) else None,
    }


async def _handle_admin_claude_accounts_get(request: Request) -> JSONResponse:
    """List every registered account (registry metadata only — never secrets).

    Deliberately just the collection: the local-login hero lives at
    `claude/local`, the serving pin at `claude/pool/serving`, and cooldown
    telemetry at `claude/pool/status` — each readable (and cacheable) on
    its own.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            _openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    return JSONResponse(
        {
            "accounts": [
                {**record.to_row(), **_account_plan_fields(record.id)}
                for record in records
            ]
        }
    )


async def _handle_admin_claude_local_get(request: Request) -> JSONResponse:
    """This machine's ambient Claude Code login (informational hero card)."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    return JSONResponse({"local": _local_claude_login_fields()})


async def _handle_admin_claude_account_delete(request: Request) -> Response:
    """Remove a registered account (dashboard analog of `account remove`).

    Refuses while the account is the serving pin — silently unpinning here
    would flip passthrough back to client credentials as a side effect.
    The registry mutation itself is crash-safe under registry.lock
    (tombstone protocol); afterwards the daemon-memory remnants (cached
    auth manager, cooldown) are dropped, and — while balanced routing is
    active (T-12, design v2 §5.7) — the router's own removal matrix runs and
    every durable row (pins, cooldowns, usage observations, capability
    evidence) for the removed incarnation is deleted, awaited before this
    responds.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    account_id = request.path_params["account_id"]
    # The pin check and the removal share the admin lock with the
    # pool/serving writes, so a concurrent serving PUT cannot pin this
    # account between the check and the removal (or vice versa).
    async with request.app.state.admin_lock:
        config: GatewayConfig = request.app.state.config
        if config.claude_account_id == account_id:
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error",
                    f"account {account_id} is the serving account; clear the pin "
                    "at pool/serving first",
                ),
                status_code=409,
            )
        try:
            records = list_accounts()
        except AccountRegistryError as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"cannot read the claude account registry: {exc}"
                ),
                status_code=500,
            )
        removed_record = next((record for record in records if record.id == account_id), None)
        try:
            await asyncio.to_thread(remove_account, account_id)
        except AccountNotFoundError as exc:
            return JSONResponse(
                _openai_error_body("invalid_request_error", str(exc)), status_code=404
            )
        except AccountRegistryError as exc:
            return JSONResponse(
                _openai_error_body(
                    "server_error", f"could not remove the account: {exc}"
                ),
                status_code=500,
            )
        request.app.state.claude_account_auth_managers.pop(account_id, None)
        request.app.state.claude_account_cooldowns.clear(account_id)
        runtime: ClaudeBalancedRuntime = request.app.state.claude_balanced_runtime
        if runtime.status == "active" and runtime.router is not None and removed_record is not None:
            runtime.router.remove_account(account_id, removed_record.account_incarnation_id)
            await runtime.router.await_account_removal_durability(removed_record.account_incarnation_id)
    return Response(status_code=204)


def _cooling_down_until_millis(tracker: AccountCooldownTracker, account_id: str) -> int | None:
    """Epoch-ms cooldown deadline for the row overlay (registry-timestamp unit)."""
    remaining = tracker.remaining_seconds(account_id)
    if remaining <= 0.0:
        return None
    return int((time.time() + remaining) * 1000)


async def _handle_admin_claude_accounts_usage(request: Request) -> JSONResponse:
    """Per-account usage.

    Fallback/disabled mode (and balanced published but not yet/no-longer
    active) is served exactly as before, through the TTL cache's fetch path
    — unchanged envelope, TTL/backoff/global-cooldown semantics, no
    force-refresh; needs-reauth rows get a synthesized "unavailable" without
    touching the network.

    Active balanced mode is cache-only (T-13 Step 4): it never calls
    `cache.get`/upstream, reading `peek_with_metadata` instead and reporting
    each window's age/source/reset/state. A `?refresh` request in this mode
    (Step 5) enqueues a coalesced, globally rate-limited manual poll on the
    balanced coordinator and reports it as `queued` in the response — it
    never fetches inline, and cached data is returned immediately either
    way. `?refresh` outside active balanced mode is inert: a non-balanced
    request must never be queued for a coordinator that is not running.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            _openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    account_param = request.query_params.get("account")
    if account_param is not None:
        records = [record for record in records if record.id == account_param]
        if not records:
            return JSONResponse(
                _openai_error_body(
                    "invalid_request_error",
                    f"no account registered with id {account_param}",
                ),
                status_code=400,
            )
    ready_ids = [record.id for record in records if record.state == "ready"]
    cache: ClaudeAccountUsageCache = request.app.state.claude_account_usage_cache
    runtime = _active_balanced_runtime(request)

    if runtime is not None:
        refresh_requested = request.query_params.get("refresh") is not None
        coordinator = runtime.usage_poll_coordinator
        results: dict[str, Any] = {}
        for account_id in ready_ids:
            if refresh_requested and coordinator is not None:
                coordinator.request_manual_refresh(account_id)
            peeked = cache.peek_with_metadata(account_id)
            if peeked is None:
                envelope = _provider_result(
                    "claude",
                    status="unavailable",
                    error="no usage observation yet; the balanced poll coordinator "
                    "has not polled this account",
                )
                windows: dict[str, Any] = {}
            else:
                envelope = dict(peeked[0])
                windows = {
                    window_name: {
                        "age_seconds": metadata["age_seconds"],
                        "source": metadata["source"],
                        "reset_at": metadata["reset_at"],
                        "state": _usage_window_state(metadata["age_seconds"]),
                    }
                    for window_name, metadata in peeked[1].items()
                }
            envelope["windows"] = windows
            envelope["queued"] = (
                coordinator.is_manual_refresh_pending(account_id) if coordinator is not None else False
            )
            results[account_id] = envelope
        for record in records:
            if record.state != "ready":
                results[record.id] = {
                    **_provider_result(
                        "claude",
                        status="unavailable",
                        error="account needs re-authentication; log in again from the dashboard",
                    ),
                    "windows": {},
                    "queued": False,
                }
        return JSONResponse(
            {
                "accounts": results,
                "fetched_at": time.time(),
                "queued": any(account["queued"] for account in results.values()),
            }
        )

    results = await cache.get(ready_ids)
    for record in records:
        if record.state != "ready":
            results[record.id] = _provider_result(
                "claude",
                status="unavailable",
                error="account needs re-authentication; log in again from the dashboard",
            )
    return JSONResponse({"accounts": results, "fetched_at": time.time()})


async def _handle_admin_claude_login_get(request: Request) -> JSONResponse:
    """Poll the login session. A bare GET is discovery/attach — it returns
    the full status (including attempt_id) so a fresh tab can pin itself;
    a GET that carries X-Login-Attempt is an attached poll and is guarded
    (including against a cleared slot: a dead attempt is a stale one)."""
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    if _LOGIN_ATTEMPT_HEADER in request.headers:
        denied = _require_login_attempt(request, session)
        if denied is not None:
            return denied
    if session is None:
        return JSONResponse({"status": "idle"})
    return JSONResponse(session.status())


async def _handle_admin_claude_login_post(request: Request) -> JSONResponse:
    """Start a dashboard login session (single concurrent session).

    The slot check and assignment have no await between them, so two
    concurrent POSTs cannot both create a session. The cross-process capture
    lock additionally excludes a CLI `account add` running on this machine.
    """
    denied = _admin_guard(request) or _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    if session is not None and not session.is_terminal:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "a login session is already active; poll GET /admin/providers/claude/login",
                "login-active",
            ),
            status_code=409,
        )
    lock_handle = try_file_lock(capture_lock_path())
    if lock_handle is None:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "another Claude login is in progress on this machine "
                "(a CLI `account add`?); retry once it finishes",
                "login-locked",
            ),
            status_code=409,
        )
    session = ClaudeLoginSession(lock_handle)
    request.app.state.claude_login_session = session
    session.start()
    # The full envelope (not a minimal status) so the creating tab can pin
    # its attempt_id without a follow-up GET racing another tab's POST.
    return JSONResponse(session.status(), status_code=201)


async def _handle_admin_claude_login_code_post(request: Request) -> JSONResponse:
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_CODE_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; supported: code",
            ),
            status_code=400,
        )
    code = body.get("code")
    code = code.strip() if isinstance(code, str) else None
    # A pasted code must be exactly one stdin line for the login child;
    # control characters would smuggle extra lines or terminal noise.
    if not code or _CONTROL_CHARACTER_PATTERN.search(code):
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "provide 'code' as a non-empty single-line string",
            ),
            status_code=400,
        )
    try:
        await session.submit_code(code)
    except LoginSessionStateError as exc:
        return JSONResponse(
            _openai_error_body("invalid_request_error", str(exc)),
            status_code=409,
        )
    return JSONResponse({"status": session.status()["status"]})


async def _handle_admin_claude_login_replace_post(request: Request) -> JSONResponse:
    """Confirm replacing the duplicate registration the session collided with.

    The body names the account being replaced (the `existing_account_id`
    from status()) — a confirmation, not a generation token; the session
    rejects a mismatch. Declining is DELETE (cancel), not a body variant.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    denied = _require_json_content_type(request)
    if denied is not None:
        return denied
    body, error = await _read_json_object(request, _openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_REPLACE_KEYS))
    if unknown:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; supported: existing_account_id",
            ),
            status_code=400,
        )
    existing_account_id = body.get("existing_account_id")
    if not isinstance(existing_account_id, str) or not existing_account_id:
        return JSONResponse(
            _openai_error_body(
                "invalid_request_error",
                "provide 'existing_account_id' as a non-empty string",
            ),
            status_code=400,
        )
    try:
        session.confirm_replace(existing_account_id)
    except LoginSessionStateError as exc:
        return JSONResponse(
            _openai_error_body("invalid_request_error", str(exc)),
            status_code=409,
        )
    return JSONResponse({"status": session.status()["status"]})


async def _handle_admin_claude_login_delete(request: Request) -> JSONResponse:
    """Cancel an active session, or clear a terminal one from the slot.

    Attempt-addressed like every mutating login command: the caller must
    name the attempt it is cancelling, so a stale tab (or a call after
    the slot was cleared) gets 409 stale_login instead of touching a
    session it never attached to.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    denied = _require_login_attempt(request, session)
    if denied is not None:
        return denied
    if session.is_terminal:
        request.app.state.claude_login_session = None
        return JSONResponse({"status": "idle"})
    session.request_cancel()
    return JSONResponse({"status": "cancelling"})


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
        "anthropic-beta": _OAUTH_BETA,
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
        # Every daemon capable of serving the Claude account pool — disabled,
        # fallback, or balanced routing mode alike — takes this lease before
        # any endpoint is exposed and holds it for the process lifetime;
        # routing-mode transitions never acquire or release it (Adjudication
        # C). This is the same nonblocking exclusive lock used to serialize
        # `account login` (see capture_lock_path() above).
        pool_dir = paths.claude_account_pool_dir()
        pool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        claude_pool_lease = try_file_lock(paths.claude_account_pool_lock())
        if claude_pool_lease is None:
            raise RuntimeError(
                "claude account pool is already served by another process (balanced-router.lock held)"
            )
        app.state.claude_pool_lease = claude_pool_lease
        try:
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

                    if config.claude_account_routing_mode == "balanced":
                        # A persisted "balanced" setting is prepared right here, after
                        # the pool lease is held and `app.state.config` already reads
                        # "balanced" -- a request racing this window sees `status ==
                        # "acquiring"` and awaits it (T-10 Step 6), never a stale mode.
                        # `persist` is a no-op: settings are already on disk, which is
                        # exactly why this runs.
                        try:
                            await app.state.claude_balanced_runtime.prepare_and_publish(
                                accounts=list_accounts(),
                                accounts_root=paths.accounts_dir("claude"),
                                runtime_db_path=paths.claude_account_pool_runtime_db(),
                                persist=lambda: None,
                                entry="startup_restore",
                                usage_cache=app.state.claude_account_usage_cache,
                            )
                            logger.info(
                                "balanced routing runtime restored (epoch=%s)",
                                app.state.claude_balanced_runtime.epoch_id,
                            )
                        except Exception as exc:
                            # Degrade, don't crash the daemon: balanced dispatch fails
                            # closed (503 "balanced routing is not active") until an
                            # admin fixes the underlying issue and re-enables it.
                            logger.error(
                                "could not activate persisted balanced routing at "
                                "startup: %s",
                                exc,
                                exc_info=True,
                            )

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
        finally:
            # Process shutdown while settings remain "balanced" preserves the
            # persisted mode, epoch id/seed, pins, observations, cooldowns, and
            # capability evidence (T-10 Step 4) -- distinct from an intentional
            # `exit_mode`, which invalidates them. This must run before the T-9
            # lease releases, so no other process can open the runtime store
            # while this one is still draining/closing it.
            await app.state.claude_balanced_runtime.shutdown_preserving_epoch()
            app.state.claude_pool_lease.release()

    app = Starlette(
        routes=[
            Route("/", _handle_dashboard, methods=["GET"]),
            Route("/favicon.ico", _handle_favicon, methods=["GET"]),
            Route("/v1/messages", _handle_messages, methods=["POST"]),
            Route("/v1/messages/count_tokens", _handle_count_tokens, methods=["POST"]),
            Route("/api/hello", _handle_hello, methods=["GET"]),
            Route("/health", _handle_health, methods=["GET"]),
            # Admin tree (confirmed design, .docs/research/admin-api-reorg-gptpro.md):
            # settings/* for gateway-wide settings, providers/{p}/* for each
            # backend's own surface, top-level logs/usage/test as cross-cutting
            # observability. No aliases: old paths 404.
            Route("/admin/settings/mapping", _handle_admin_mapping_get, methods=["GET"]),
            Route("/admin/settings/mapping", _handle_admin_mapping_put, methods=["PUT"]),
            Route(
                "/admin/settings/log-level", _handle_admin_log_level_get, methods=["GET"]
            ),
            Route(
                "/admin/settings/log-level", _handle_admin_log_level_put, methods=["PUT"]
            ),
            Route(
                "/admin/settings/compaction", _handle_admin_compaction_get, methods=["GET"]
            ),
            Route(
                "/admin/settings/compaction", _handle_admin_compaction_put, methods=["PUT"]
            ),
            Route(
                "/admin/providers/codex/models",
                _handle_admin_codex_models,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/codex/reset-credit",
                _handle_admin_codex_reset_credit,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/kimi/models", _handle_admin_kimi_models, methods=["GET"]
            ),
            Route(
                "/admin/providers/grok/models", _handle_admin_grok_models, methods=["GET"]
            ),
            Route(
                "/admin/providers/claude/local",
                _handle_admin_claude_local_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/accounts",
                _handle_admin_claude_accounts_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/accounts/{account_id}",
                _handle_admin_claude_account_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/login",
                _handle_admin_claude_login_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/login/code",
                _handle_admin_claude_login_code_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/login/replace",
                _handle_admin_claude_login_replace_post,
                methods=["POST"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_put,
                methods=["PUT"],
            ),
            Route(
                "/admin/providers/claude/pool/serving",
                _handle_admin_claude_serving_delete,
                methods=["DELETE"],
            ),
            Route(
                "/admin/providers/claude/pool/routing",
                _handle_admin_claude_routing_get,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/routing",
                _handle_admin_claude_routing_put,
                methods=["PUT"],
            ),
            Route(
                "/admin/providers/claude/pool/status",
                _handle_admin_claude_pool_status,
                methods=["GET"],
            ),
            Route(
                "/admin/providers/claude/pool/usage",
                _handle_admin_claude_accounts_usage,
                methods=["GET"],
            ),
            Route("/admin/logs", _handle_admin_logs, methods=["GET"]),
            Route("/admin/usage", _handle_admin_usage, methods=["GET"]),
            Route("/admin/test", _handle_admin_connection_test, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.state.daemon_nonce = daemon_nonce
    # Lazily-created ClaudeAccountAuthManager per registered account id.
    # Initialized here (not in the lifespan) so the dict exists even when a
    # test drives the app without entering the lifespan context.
    app.state.claude_account_auth_managers = {}
    app.state.claude_ambient_accounts = (
        AmbientAccountProvider() if config.claude_account_include_local_login else None
    )
    # Dashboard login session slot (single concurrent session) and the
    # per-account usage cache — the fetch closure resolves http_client from
    # app.state at call time, so wiring here works with or without the
    # lifespan having run.
    app.state.claude_login_session = None
    app.state.claude_account_usage_cache = ClaudeAccountUsageCache(
        fetch=_account_usage_fetch(app.state)
    )
    # Rate-limit cooldowns are ephemeral runtime state and live only in this
    # process — the registry records exclusively durable facts (design §8).
    app.state.claude_account_cooldowns = AccountCooldownTracker()
    # Starts "disabled" — a persisted "balanced" mode is prepared+published
    # during lifespan startup (after the T-9 pool lease is held); set here
    # (not in the lifespan) so it exists, and balanced dispatch fails closed
    # rather than crashing, even for a test that drives the app without
    # entering the lifespan context.
    app.state.claude_balanced_runtime = ClaudeBalancedRuntime()

    def _ambient_usage_poll_supplier(
        records: Sequence[AccountRecord],
    ) -> UsagePollAccount | None:
        provider: AmbientAccountProvider | None = app.state.claude_ambient_accounts
        if provider is None:
            return None
        member = provider.pool_member()
        if member is None or is_duplicate_identity(member, records):
            return None
        return UsagePollAccount(
            account_id=member.record.id,
            account_incarnation_id=member.record.account_incarnation_id,
            account_profile_fingerprint=member.profile_fingerprint,
        )

    app.state.claude_balanced_runtime.ambient_usage_poll_supplier = (
        _ambient_usage_poll_supplier
    )
    return app
