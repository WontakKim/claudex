"""Registered-account and fallback-pool relay serving paths."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from claudex import paths, server_support
from claudex.claude.account_attempts import (
    AccountLegContext,
    AccountLegTracker,
    emit_account_leg_log,
    try_begin_account_leg,
)
from claudex.claude.account_pool import (
    _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    AccountCooldownTracker,
    build_serving_chain,
    rate_limit_cooldown_decision,
)
from claudex.claude.accounts import (
    AccountRecord,
    AccountRegistryError,
    load_registry,
)
from claudex.claude.auth import (
    ClaudeAccountAuthError,
    ClaudeAccountReauthRequiredError,
)
from claudex.claude.quota_429 import (
    Quota429Mark,
    build_quota_429_record,
    enrich_record_fallback,
    finalize_quota_429_record,
)
from claudex.claude.session_fingerprint import (
    extract_session_uuid,
    load_or_create_fingerprint_seed,
    observability_session_fingerprint,
)
from claudex.relay.common import (
    _MANAGED_RELAY_SKIP_REQUEST_HEADERS,
    _OAUTH_BETA,
    _PASSTHROUGH_SKIP_RESPONSE_HEADERS,
    _relay_anthropic_response,
    _send_to_anthropic,
)

logger = logging.getLogger("claudex.server")


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
    attempt_context: AccountLegContext | None,
    rate_limit_failover: bool,
    commit_hook: Callable[[httpx.Response], Awaitable[None]] | None = None,
    on_relay_complete: Callable[[], None] | None = None,
    on_quota_429: Callable[[Quota429Mark], Awaitable[str]] | None = None,
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

    `on_quota_429`, when given, is awaited with the attributed mark immediately
    after an account-specific 429 marks the shared in-memory tracker. Balanced
    routing uses it to select the cooldown scope and persist canonical evidence.
    `on_response`, when given, is awaited with the open upstream response at
    the same point as `commit_hook`; balanced routing uses it to record eligible
    capability evidence from an explicit 2xx. All hooks default to `None`, so
    non-balanced callers need no special handling.
    """
    account_id = record.id
    result = "failed"
    model: str | None = None
    try:
        candidate_model = parsed_body.get("model") if isinstance(parsed_body, dict) else None
        model = candidate_model if isinstance(candidate_model, str) else None
        manager = server_support._claude_account_auth_manager(request.app.state, account_id)
        try:
            credentials = await manager.get_credentials()
        except ClaudeAccountReauthRequiredError as exc:
            await server_support._mark_account_needs_reauth_best_effort(
                request.app.state, account_id
            )
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
                    f"claude account {account_id} is unusable: {exc}"
                )
            )

        logger.info(
            "claude relay: attempting %s with account %.8s (%s)",
            request.url.path,
            account_id,
            record.email,
        )
        content = _rewrite_metadata_account_uuid(
            raw_body, parsed_body, credentials.account_uuid
        )
        try:
            upstream_response = await _send_to_anthropic(
                request,
                _claude_account_request_headers(request, credentials.access_token),
                content,
            )
            if upstream_response.status_code == 401:
                await upstream_response.aclose()
                try:
                    credentials = await manager.get_credentials(force_refresh=True)
                except ClaudeAccountReauthRequiredError as exc:
                    await server_support._mark_account_needs_reauth_best_effort(
                        request.app.state, account_id
                    )
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
                    _claude_account_request_headers(
                        request, credentials.access_token
                    ),
                    content,
                )
        except httpx.HTTPError as exc:
            logger.warning("anthropic passthrough failed: %s", exc)
            return JSONResponse(
                server_support._claude_error_body(
                    "api_error", f"failed to reach the Anthropic API: {exc}"
                ),
                status_code=502,
            )

        if upstream_response.status_code == 429:
            result = "rate_limited"
        elif 200 <= upstream_response.status_code < 300:
            result = "success"

        if upstream_response.status_code == 401:
            await upstream_response.aclose()
            # A freshly refreshed token that Anthropic still rejects is durably
            # dead — only a human re-login recovers it, which is what the
            # needs-reauth state means.
            await server_support._mark_account_needs_reauth_best_effort(
                request.app.state, account_id
            )
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
            # Preserve the occurrence-aware collection for evidence capture. The
            # replay path needs its own plain mapping after httpx has decoded the
            # buffered response.
            response_headers = upstream_response.headers
            replay_headers = dict(response_headers)
            response_body = await upstream_response.aread()
            await upstream_response.aclose()
            cooldown_decision = rate_limit_cooldown_decision(
                response_headers,
                response_body,
                request.app.state.claude_account_usage_cache.peek(account_id),
            )
            request.app.state.claude_account_cooldowns.mark(
                account_id, cooldown_decision.seconds
            )
            logger.warning(
                "claude account %.8s rate-limited by Anthropic; cooling down for %.0fs",
                account_id,
                cooldown_decision.seconds,
            )

            session_literals = extract_session_uuid(
                parsed_body if isinstance(parsed_body, dict) else {}
            )
            context = attempt_context
            if context is None:
                context_mode = (
                    "balanced_stateless" if on_quota_429 is not None else "fallback"
                )
                context = AccountLegContext(
                    mode=context_mode,
                    ordinal=1,
                    pin_created=None,
                    first_started_monotonic=0.0,
                    started_monotonic=0.0,
                    previous_started_monotonic=None,
                    session_literals=session_literals or (),
                )
            mode = {
                "balanced_stateless": "balanced",
                "balanced_pinned": "balanced",
                "fallback": "fallback",
            }.get(context.mode, "balanced" if on_quota_429 is not None else "fallback")
            quota_record = build_quota_429_record(
                occurred_at_utc=datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                mode=mode,
                account_id=account_id,
                model=model,
                cooldown_seconds=cooldown_decision.seconds,
                cooldown_source=cooldown_decision.source,
                context=context,
                parsed_body=parsed_body,
                raw_body=raw_body,
                response_headers=response_headers,
                response_body=response_body,
                session_literals=session_literals or (),
            )
            mark = Quota429Mark(
                cooldown_seconds=cooldown_decision.seconds,
                cooldown_source=cooldown_decision.source,
                record=quota_record,
                session_literals=session_literals,
            )
            if on_quota_429 is not None:
                canonical_record = await on_quota_429(mark)
            else:
                try:
                    fingerprint_seed = load_or_create_fingerprint_seed(
                        paths.claude_account_pool_dir()
                    )
                    session_fingerprint = (
                        observability_session_fingerprint(
                            fingerprint_seed, session_literals[1]
                        )
                        if fingerprint_seed is not None
                        and session_literals is not None
                        else None
                    )
                    enrich_record_fallback(
                        quota_record, session_fingerprint=session_fingerprint
                    )
                    canonical_record = finalize_quota_429_record(quota_record)
                except Exception:
                    logger.warning("failed to enrich Claude 429 incident evidence")
                    enrich_record_fallback(
                        quota_record, session_fingerprint=None
                    )
                    quota_record["record_degraded"] = True
                    quota_record["degradation_reason"] = (
                        "evidence_enrichment_failed"
                    )
                    canonical_record = finalize_quota_429_record(quota_record)

            incident_writer = getattr(
                request.app.state, "claude_quota_429_incident_writer", None
            )
            if incident_writer is None:
                logger.warning(
                    "Claude 429 incident writer is unavailable; skipping incident persistence"
                )
            else:
                try:
                    await incident_writer.append_record(canonical_record)
                except Exception:
                    logger.warning("failed to append Claude 429 incident record")
            return _FailedAttempt(
                _replay_buffered_response(429, replay_headers, response_body),
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
        return _relay_anthropic_response(
            upstream_response, on_finished=on_relay_complete
        )
    except BaseException:
        result = "exception"
        raise
    finally:
        if attempt_context is not None:
            try:
                emit_account_leg_log(
                    logger,
                    attempt_context,
                    account_id=account_id,
                    model=model,
                    result=result,
                    parsed_body=parsed_body,
                    raw_body=raw_body,
                    occurred_at_utc=datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                )
            except BaseException:
                # Observability must never replace the account-leg outcome.
                pass


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
        session_uuid = extract_session_uuid(
            parsed_body if isinstance(parsed_body, dict) else {}
        )
        leg_tracker: AccountLegTracker | None = AccountLegTracker(
            "disabled", session_literals=session_uuid or ()
        )
    except Exception:
        leg_tracker = None
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
    attempt_context = try_begin_account_leg(leg_tracker, None)
    outcome = await _attempt_with_account(
        request,
        raw_body,
        parsed_body,
        record,
        attempt_context=attempt_context,
        rate_limit_failover=False,
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
        session_uuid = extract_session_uuid(
            parsed_body if isinstance(parsed_body, dict) else {}
        )
        leg_tracker = AccountLegTracker(
            "fallback", session_literals=session_uuid or ()
        )
    except Exception:
        leg_tracker = None
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
        attempt_context = try_begin_account_leg(leg_tracker, None)
        outcome = await _attempt_with_account(
            request,
            raw_body,
            parsed_body,
            record,
            attempt_context=attempt_context,
            rate_limit_failover=True,
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
