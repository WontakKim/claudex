"""Claude account pool, usage, and interactive login admin handlers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from claudex_gateway import paths, server_support
from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.admin.common import (
    _admin_guard,
    _read_json_object,
    _require_json_content_type,
)
from claudex_gateway.balanced.runtime import ClaudeBalancedRuntime
from claudex_gateway.claude_account_pool import AccountCooldownTracker
from claudex_gateway.claude_accounts import (
    AccountNotFoundError,
    AccountRegistryError,
    list_accounts,
    remove_account,
)
from claudex_gateway.claude_login_session import (
    ClaudeLoginSession,
    LoginSessionStateError,
    capture_lock_path,
)
from claudex_gateway.config import GatewayConfig
from claudex_gateway.locking import try_file_lock
from claudex_gateway.usage.envelope import provider_result


_USAGE_WINDOW_FRESH_MAX_AGE_SECONDS = 5 * 60


_USAGE_WINDOW_AGING_MAX_AGE_SECONDS = 30 * 60


def _active_balanced_runtime(request: Request) -> ClaudeBalancedRuntime | None:
    """Return the live runtime only when balanced mode is published and active.

    This boundary keeps balanced-only usage behavior out of every other mode;
    a non-balanced request must never enqueue work for a coordinator that is
    not running.
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
            server_support._openai_error_body(
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
        server_support._openai_error_body(
            "invalid_request_error",
            "X-Login-Attempt does not name the active login attempt; "
            "re-attach via GET /admin/providers/claude/login",
            "stale_login",
        ),
        status_code=409,
    )


def _local_claude_login_fields() -> dict[str, Any] | None:
    """Identity snapshot of this machine's ambient Claude Code login.

    Read from the CLI's own config file (`~/.claude.json`, or
    `$CLAUDE_CONFIG_DIR/.claude.json` when the override is set): the same
    `oauthAccount` block a capture snapshots — identity and plan metadata,
    never secrets. The dashboard presents this informational snapshot as the
    local CLI login; it does not affect request serving.
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
            server_support._openai_error_body(
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
                server_support._openai_error_body(
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
                server_support._openai_error_body(
                    "server_error", f"cannot read the claude account registry: {exc}"
                ),
                status_code=500,
            )
        removed_record = next((record for record in records if record.id == account_id), None)
        try:
            await asyncio.to_thread(remove_account, account_id)
        except AccountNotFoundError as exc:
            return JSONResponse(
                server_support._openai_error_body("invalid_request_error", str(exc)), status_code=404
            )
        except AccountRegistryError as exc:
            return JSONResponse(
                server_support._openai_error_body(
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

    Active balanced mode is cache-only: it never calls `cache.get` or the
    upstream, instead reading `peek_with_metadata` and reporting each window's
    age, source, reset time, and state. A `?refresh` request in this mode
    enqueues a coalesced, globally rate-limited manual poll on the balanced
    coordinator and reports it as `queued` in the response. It never fetches
    inline, and cached data is returned immediately either way. `?refresh`
    outside active balanced mode is inert: a non-balanced request must never be
    queued for a coordinator that is not running.
    """
    denied = _admin_guard(request)
    if denied is not None:
        return denied
    try:
        records = list_accounts()
    except AccountRegistryError as exc:
        return JSONResponse(
            server_support._openai_error_body(
                "server_error", f"cannot read the claude account registry: {exc}"
            ),
            status_code=500,
        )
    account_param = request.query_params.get("account")
    if account_param is not None:
        records = [record for record in records if record.id == account_param]
        if not records:
            return JSONResponse(
                server_support._openai_error_body(
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
                envelope = provider_result(
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
                    **provider_result(
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
            results[record.id] = provider_result(
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
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    if body:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unexpected keys: {', '.join(sorted(body))}; POST an empty JSON object",
            ),
            status_code=400,
        )

    session: ClaudeLoginSession | None = request.app.state.claude_login_session
    if session is not None and not session.is_terminal:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "a login session is already active; poll GET /admin/providers/claude/login",
                "login-active",
            ),
            status_code=409,
        )
    lock_handle = try_file_lock(capture_lock_path())
    if lock_handle is None:
        return JSONResponse(
            server_support._openai_error_body(
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
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_CODE_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
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
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'code' as a non-empty single-line string",
            ),
            status_code=400,
        )
    try:
        await session.submit_code(code)
    except LoginSessionStateError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)),
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
    body, error = await _read_json_object(request, server_support._openai_error_body)
    if error is not None or body is None:
        return error
    unknown = sorted(set(body) - set(_CLAUDE_LOGIN_REPLACE_KEYS))
    if unknown:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                f"unknown keys: {', '.join(unknown)}; supported: existing_account_id",
            ),
            status_code=400,
        )
    existing_account_id = body.get("existing_account_id")
    if not isinstance(existing_account_id, str) or not existing_account_id:
        return JSONResponse(
            server_support._openai_error_body(
                "invalid_request_error",
                "provide 'existing_account_id' as a non-empty string",
            ),
            status_code=400,
        )
    try:
        session.confirm_replace(existing_account_id)
    except LoginSessionStateError as exc:
        return JSONResponse(
            server_support._openai_error_body("invalid_request_error", str(exc)),
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
