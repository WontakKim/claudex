"""Subscription usage probes for the dashboard's Usage tab.

Adopts Orca's approach: read the local CLI credentials and call each
provider's own usage endpoint directly — gateway traffic plays no part.
Claude usage comes from the Anthropic OAuth usage API with the Claude Code
credentials (macOS Keychain or ~/.claude/.credentials.json); Codex usage
comes from the ChatGPT backend with the gateway's own Codex credentials.

The probes are advisory: they never raise, returning a status dict whose
"status" field is "ok", "unavailable" (no usable credentials), or "error"
(the provider rejected or could not be reached).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from claudex_gateway.claude_auth import (
    ClaudeAccountAuthError,
    ClaudeAccountAuthManager,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.codex_auth import CodexAuthManager, CodexCredentials
from claudex_gateway.grok_auth import GrokAuthManager, GrokCredentials

logger = logging.getLogger(__name__)

_CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_CLAUDE_OAUTH_BETA = "oauth-2025-04-20"
# Match the Claude Code CLI's user-agent to stay aligned with the OAuth
# usage API contract (mirrors Orca's claude-fetcher).
_CLAUDE_CODE_USER_AGENT = "claude-code/2.1.0"
_CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
_KEYCHAIN_TIMEOUT_SECONDS = 3.0

_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
# Spending an earned reset credit clears the rate-limit windows early. Unlike
# every other call in this module this one writes, and burns something the
# user cannot get back, so it only ever runs on an explicit request.
_CODEX_RESET_CONSUME_URL = "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume"
# Outcome codes the backend answers a consume request with (mirrors Orca).
_CODEX_RESET_OUTCOMES = ("reset", "nothing_to_reset", "no_credit", "already_redeemed")

_USAGE_TIMEOUT = httpx.Timeout(10.0)

_SESSION_WINDOW_MINUTES = 300
_WEEKLY_WINDOW_MINUTES = 10080
# Tolerate the one-minute drift seen in older Codex bucket lengths.
_WINDOW_TOLERANCE_MINUTES = 1


def _provider_result(
    provider: str,
    *,
    status: str,
    error: str | None,
    session: dict[str, Any] | None = None,
    weekly: dict[str, Any] | None = None,
    plan_type: str | None = None,
    reset_credits_available: int | None = None,
    fable_weekly: dict[str, Any] | None = None,
    monthly: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "status": status,
        "error": error,
        "session": session,
        "weekly": weekly,
        "plan_type": plan_type,
        "reset_credits_available": reset_credits_available,
        "fable_weekly": fable_weekly,
        "monthly": monthly,
        "updated_at": time.time(),
    }


def _reset_epoch_seconds(value: Any) -> float | None:
    """Normalize a resets_at value (ISO 8601 string or epoch s/ms) to seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # 1e10 sits between any plausible seconds epoch (<2286) and any
        # milliseconds epoch (>2001), distinguishing the two units.
        return float(value) / 1000 if value > 10_000_000_000 else float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return _reset_epoch_seconds(float(text))
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _map_claude_window(raw: Any, window_minutes: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    used = raw.get("utilization")
    if not isinstance(used, (int, float)) or isinstance(used, bool):
        used = raw.get("used_percentage")
    if not isinstance(used, (int, float)) or isinstance(used, bool):
        return None
    return {
        "used_percent": min(100.0, max(0.0, float(used))),
        "window_minutes": window_minutes,
        "resets_at": _reset_epoch_seconds(raw.get("resets_at")),
    }


def _map_fable_weekly_window(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract Fable's model-scoped weekly quota, separate from the shared 7-day window.

    Model quotas moved to the structured limits[] entries; the legacy flat
    fable_* fields are kept as fallbacks for older responses (mirrors Orca).
    """
    limits = data.get("limits")
    scoped = None
    if isinstance(limits, list):
        for limit in limits:
            if not isinstance(limit, dict) or limit.get("kind") != "weekly_scoped":
                continue
            scope = limit.get("scope")
            model = scope.get("model") if isinstance(scope, dict) else None
            display_name = model.get("display_name") if isinstance(model, dict) else None
            # is_active marks the currently-binding limit, not data validity.
            if isinstance(display_name, str) and display_name.strip().lower() == "fable":
                scoped = limit
                break
    if scoped is not None:
        mapped = _map_claude_window(
            {"used_percentage": scoped.get("percent"), "resets_at": scoped.get("resets_at")},
            _WEEKLY_WINDOW_MINUTES,
        )
        if mapped is not None:
            return mapped
    for legacy_field in ("fable_weekly", "fable_seven_day", "seven_day_fable"):
        mapped = _map_claude_window(data.get(legacy_field), _WEEKLY_WINDOW_MINUTES)
        if mapped is not None:
            return mapped
    return None


# ---------------------------------------------------------------------------
# Claude credentials — macOS Keychain first, ~/.claude/.credentials.json last
# ---------------------------------------------------------------------------


def _claude_config_dir() -> Path:
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override).expanduser() if override else Path.home() / ".claude"


def _claude_keychain_services() -> list[str]:
    # Claude Code 2.1+ scopes the Keychain item by CLAUDE_CONFIG_DIR using the
    # first 8 hex chars of sha256(dir); older builds use the unsuffixed item.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if not config_dir:
        return [_CLAUDE_KEYCHAIN_SERVICE]
    suffix = hashlib.sha256(config_dir.encode()).hexdigest()[:8]
    return [f"{_CLAUDE_KEYCHAIN_SERVICE}-{suffix}", _CLAUDE_KEYCHAIN_SERVICE]


async def _keychain_password(service: str, account: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "security",
            "find-generic-password",
            "-s",
            service,
            "-a",
            account,
            "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), _KEYCHAIN_TIMEOUT_SECONDS)
    except (OSError, TimeoutError):
        return None
    if proc.returncode != 0:
        return None
    return stdout.decode().strip() or None


def _parse_claude_credentials(raw: str) -> str | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    oauth = parsed.get("claudeAiOauth") if isinstance(parsed, dict) else None
    token = oauth.get("accessToken") if isinstance(oauth, dict) else None
    return token if isinstance(token, str) and token else None


async def _resolve_claude_oauth_token() -> str | None:
    account = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    for service in _claude_keychain_services():
        password = await _keychain_password(service, account)
        if password:
            token = _parse_claude_credentials(password)
            if token:
                return token
    try:
        raw = (_claude_config_dir() / ".credentials.json").read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_claude_credentials(raw)


# ---------------------------------------------------------------------------
# Provider probes
# ---------------------------------------------------------------------------


async def fetch_claude_usage(http_client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch Claude subscription windows via the Anthropic OAuth usage API."""
    token = await _resolve_claude_oauth_token()
    if token is None:
        return _provider_result(
            "claude",
            status="unavailable",
            error="no Claude Code OAuth credentials found; sign in with `claude` first",
        )
    result, _retry_after, _status_code = await _fetch_claude_usage_with_token(http_client, token)
    return result


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a Retry-After header (delta seconds or HTTP-date) into seconds."""
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(raw).timestamp() - time.time()
        except (TypeError, ValueError):
            return None
    return max(0.0, seconds)


async def _fetch_claude_usage_with_token(
    http_client: httpx.AsyncClient, token: str
) -> tuple[dict[str, Any], float | None, int | None]:
    """One usage-API GET with `token`.

    Returns `(result, retry_after_seconds, status_code)`: retry_after is only
    populated from a 429's Retry-After header, and status_code is None when
    the request never reached the API. Callers that don't care (the ambient
    probe) discard the extra elements.
    """
    try:
        response = await http_client.get(
            _CLAUDE_USAGE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": _CLAUDE_OAUTH_BETA,
                "User-Agent": _CLAUDE_CODE_USER_AGENT,
            },
            timeout=_USAGE_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.warning("claude usage fetch failed: %s", exc)
        return (
            _provider_result(
                "claude", status="error", error=f"failed to reach the Anthropic usage API: {exc}"
            ),
            None,
            None,
        )
    status_code = response.status_code
    if status_code == 401:
        return (
            _provider_result(
                "claude",
                status="error",
                error="Claude OAuth token rejected (401); sign in again with `claude`",
            ),
            None,
            status_code,
        )
    if status_code == 429:
        return (
            _provider_result(
                "claude",
                status="error",
                error="usage API rate-limited (429); try again shortly",
            ),
            _retry_after_seconds(response),
            status_code,
        )
    if status_code != 200:
        return (
            _provider_result(
                "claude",
                status="error",
                error=f"usage API returned {status_code}: {response.text[:200]}",
            ),
            None,
            status_code,
        )
    try:
        data = response.json()
    except json.JSONDecodeError:
        return (
            _provider_result("claude", status="error", error="usage API returned a non-JSON body"),
            None,
            status_code,
        )
    if not isinstance(data, dict):
        return (
            _provider_result(
                "claude", status="error", error="usage API returned an unexpected payload"
            ),
            None,
            status_code,
        )
    return (
        _provider_result(
            "claude",
            status="ok",
            error=None,
            session=_map_claude_window(data.get("five_hour"), _SESSION_WINDOW_MINUTES),
            weekly=_map_claude_window(data.get("seven_day"), _WEEKLY_WINDOW_MINUTES),
            fable_weekly=_map_fable_weekly_window(data),
        ),
        None,
        status_code,
    )


async def fetch_claude_account_usage(
    http_client: httpx.AsyncClient, auth_manager: ClaudeAccountAuthManager
) -> tuple[dict[str, Any], float | None]:
    """Per-registered-account usage probe using the account's own bearer.

    Documented deviation from this module's never-raise contract:
    `ClaudeAccountReauthRequiredError` propagates so the caller can mark the
    registry row needs-reauth. Every other failure returns a normal
    `_provider_result` envelope. A 401 result gets one force-refresh retry,
    mirroring the serving path.
    """
    try:
        credentials = await auth_manager.get_credentials()
    except ClaudeAccountReauthRequiredError:
        raise
    except ClaudeAccountAuthError as exc:
        return (_provider_result("claude", status="unavailable", error=str(exc)), None)

    result, retry_after, status_code = await _fetch_claude_usage_with_token(
        http_client, credentials.access_token
    )
    if status_code == 401:
        try:
            credentials = await auth_manager.get_credentials(force_refresh=True)
        except ClaudeAccountReauthRequiredError:
            raise
        except ClaudeAccountAuthError as exc:
            return (_provider_result("claude", status="unavailable", error=str(exc)), None)
        result, retry_after, status_code = await _fetch_claude_usage_with_token(
            http_client, credentials.access_token
        )
        if status_code == 401:
            result = _provider_result(
                "claude",
                status="error",
                error="account token rejected (401) even after refresh",
            )
    return (result, retry_after)


def _map_codex_window(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    used = raw.get("used_percent")
    if not isinstance(used, (int, float)) or isinstance(used, bool):
        return None
    window_seconds = raw.get("limit_window_seconds")
    window_minutes = (
        math.ceil(window_seconds / 60)
        if isinstance(window_seconds, (int, float))
        and not isinstance(window_seconds, bool)
        and window_seconds > 0
        else None
    )
    return {
        "used_percent": min(100.0, max(0.0, float(used))),
        "window_minutes": window_minutes,
        "resets_at": _reset_epoch_seconds(raw.get("reset_at")),
    }


def _codex_window_kind(window_minutes: Any) -> str | None:
    if not isinstance(window_minutes, (int, float)) or isinstance(window_minutes, bool):
        return None
    if abs(window_minutes - _SESSION_WINDOW_MINUTES) <= _WINDOW_TOLERANCE_MINUTES:
        return "session"
    if abs(window_minutes - _WEEKLY_WINDOW_MINUTES) <= _WINDOW_TOLERANCE_MINUTES:
        return "weekly"
    return None


def _classify_codex_windows(
    primary: dict[str, Any] | None, secondary: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    session = weekly = None
    for window in (primary, secondary):
        kind = _codex_window_kind(window.get("window_minutes")) if window else None
        if kind == "session" and session is None:
            session = window
        elif kind == "weekly" and weekly is None:
            weekly = window
    # Unknown durations keep the legacy primary/session, secondary/weekly mapping.
    if session is None and primary and _codex_window_kind(primary["window_minutes"]) is None:
        session = primary
    if weekly is None and secondary and _codex_window_kind(secondary["window_minutes"]) is None:
        weekly = secondary
    return session, weekly


def _codex_backend_headers(credentials: CodexCredentials) -> dict[str, str]:
    """Identify as the Codex client to the ChatGPT backend (mirrors Orca)."""
    headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "User-Agent": "codex-cli",
        "OpenAI-Beta": "codex-1",
        "originator": "Codex Desktop",
    }
    if credentials.account_id:
        headers["ChatGPT-Account-Id"] = credentials.account_id
    return headers


async def fetch_codex_usage(
    http_client: httpx.AsyncClient, auth_manager: CodexAuthManager
) -> dict[str, Any]:
    """Fetch Codex plan windows via the ChatGPT backend's own usage endpoint."""
    try:
        credentials = await auth_manager.get_credentials()
    except Exception as exc:  # CodexAuthError and anything the file layer raises
        return _provider_result("codex", status="unavailable", error=str(exc))
    if credentials.is_api_key:
        return _provider_result(
            "codex",
            status="unavailable",
            error="API key billing has no plan usage windows",
        )
    headers = _codex_backend_headers(credentials)
    try:
        response = await http_client.get(_CODEX_USAGE_URL, headers=headers, timeout=_USAGE_TIMEOUT)
    except httpx.HTTPError as exc:
        logger.warning("codex usage fetch failed: %s", exc)
        return _provider_result(
            "codex", status="error", error=f"failed to reach the ChatGPT usage API: {exc}"
        )
    if response.status_code != 200:
        return _provider_result(
            "codex",
            status="error",
            error=f"usage API returned {response.status_code}: {response.text[:200]}",
        )
    try:
        data = response.json()
    except json.JSONDecodeError:
        return _provider_result(
            "codex", status="error", error="usage API returned a non-JSON body"
        )
    if not isinstance(data, dict):
        return _provider_result(
            "codex", status="error", error="usage API returned an unexpected payload"
        )
    rate_limit = data.get("rate_limit")
    if not isinstance(rate_limit, dict):
        rate_limit = {}
    session, weekly = _classify_codex_windows(
        _map_codex_window(rate_limit.get("primary_window")),
        _map_codex_window(rate_limit.get("secondary_window")),
    )
    plan_type = data.get("plan_type")
    reset_credits = data.get("rate_limit_reset_credits")
    available = reset_credits.get("available_count") if isinstance(reset_credits, dict) else None
    return _provider_result(
        "codex",
        status="ok",
        error=None,
        session=session,
        weekly=weekly,
        plan_type=plan_type if isinstance(plan_type, str) and plan_type else None,
        reset_credits_available=(
            available if isinstance(available, int) and not isinstance(available, bool) else None
        ),
    )


async def consume_codex_reset_credit(
    http_client: httpx.AsyncClient,
    auth_manager: CodexAuthManager,
    redeem_request_id: str,
) -> dict[str, Any]:
    """Spend one earned Codex reset credit to clear the rate-limit windows.

    Irreversible: the credit is gone once the backend accepts the request.
    redeem_request_id is the idempotency key the backend deduplicates on, so a
    caller retrying after a timeout MUST send the same one — a fresh key would
    spend a second credit for the same intent.

    Returns "ok" only when the backend gave a definitive outcome; the caller
    uses that to decide whether the attempt is settled. "error" means the
    outcome is unknown, not that nothing happened.
    """
    try:
        credentials = await auth_manager.get_credentials()
    except Exception as exc:  # CodexAuthError and anything the file layer raises
        return {"status": "unavailable", "outcome": None, "error": str(exc)}
    if credentials.is_api_key:
        return {
            "status": "unavailable",
            "outcome": None,
            "error": "API key billing has no reset credits",
        }
    try:
        response = await http_client.post(
            _CODEX_RESET_CONSUME_URL,
            headers={**_codex_backend_headers(credentials), "Content-Type": "application/json"},
            json={"redeem_request_id": redeem_request_id},
            timeout=_USAGE_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.warning("codex reset credit request failed: %s", exc)
        return {
            "status": "error",
            "outcome": None,
            "error": f"failed to reach the ChatGPT reset API: {exc}",
        }
    if response.status_code != 200:
        return {
            "status": "error",
            "outcome": None,
            "error": f"reset API returned {response.status_code}: {response.text[:200]}",
        }
    try:
        data = response.json()
    except json.JSONDecodeError:
        return {
            "status": "error",
            "outcome": None,
            "error": "reset API returned a non-JSON body",
        }
    code = data.get("code") if isinstance(data, dict) else None
    if code not in _CODEX_RESET_OUTCOMES:
        # The credit may well have been spent, so this is not a settled "ok".
        return {
            "status": "error",
            "outcome": None,
            "error": f"reset API returned an unknown outcome: {code!r}",
        }
    logger.info("codex reset credit consumed: %s", code)
    return {"status": "ok", "outcome": code, "error": None}


# --- Grok --------------------------------------------------------------------
# Mirrors Orca's grok-fetcher: the chat-proxy billing endpoint is read with
# the Grok CLI's OAuth session; the credits view carries the weekly window,
# and the default view is the monthly-budget fallback for unified-billing
# accounts whose credits view omits creditUsagePercent.

_GROK_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing"
_GROK_BILLING_CREDITS_URL = _GROK_BILLING_URL + "?format=credits"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
_MONTHLY_WINDOW_MINUTES = 43200


def _to_float(value: Any) -> float | None:
    """Accept quota numbers as strings ("100") as well as ints/floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _grok_billing_headers(credentials: GrokCredentials) -> dict[str, str]:
    # The header set must match the Grok CLI or Grok rejects the request.
    headers = {
        "Authorization": f"Bearer {credentials.access_token}",
        "X-XAI-Token-Auth": _XAI_TOKEN_AUTH_VALUE,
        "Accept": "application/json",
    }
    if credentials.user_id:
        headers["x-userid"] = credentials.user_id
    return headers


def _grok_plan_tier(config: dict[str, Any]) -> str | None:
    tier = config.get("subscriptionTier")
    return tier.strip().lower() if isinstance(tier, str) and tier.strip() else None


def _grok_period_end(config: dict[str, Any]) -> Any:
    period = config.get("currentPeriod")
    end = period.get("end") if isinstance(period, dict) else None
    return end or config.get("billingPeriodEnd")


def _grok_has_confirmed_weekly_period(config: dict[str, Any]) -> bool:
    """True when the period bounds prove a weekly window exists.

    Monthly unified-billing responses can also carry a weekly currentPeriod;
    matching billing bounds identify Grok's omitted protobuf zero
    unambiguously (mirrors Orca).
    """
    period = config.get("currentPeriod")
    if not isinstance(period, dict) or period.get("type") != "USAGE_PERIOD_TYPE_WEEKLY":
        return False
    start, end = period.get("start"), period.get("end")
    return bool(
        start
        and end
        and start == config.get("billingPeriodStart")
        and end == config.get("billingPeriodEnd")
    )


def _map_grok_weekly_credits(config: dict[str, Any]) -> dict[str, Any] | None:
    used_percent = _to_float(config.get("creditUsagePercent"))
    if used_percent is None and _grok_has_confirmed_weekly_period(config):
        used_percent = 0.0
    if used_percent is None:
        return None
    return {
        "used_percent": min(100.0, max(0.0, used_percent)),
        "window_minutes": _WEEKLY_WINDOW_MINUTES,
        "resets_at": _reset_epoch_seconds(_grok_period_end(config)),
    }


def _parse_money_val(value: Any) -> float | None:
    # Grok money fields travel as {"val": "12.34"} wrappers.
    raw = value.get("val") if isinstance(value, dict) else None
    return _to_float(raw)


def _map_grok_monthly(config: dict[str, Any]) -> dict[str, Any] | None:
    limit = _parse_money_val(config.get("monthlyLimit"))
    used = _parse_money_val(config.get("used"))
    if limit is None or used is None or limit <= 0:
        return None
    return {
        "used_percent": min(100.0, max(0.0, used / limit * 100)),
        "window_minutes": _MONTHLY_WINDOW_MINUTES,
        "resets_at": _reset_epoch_seconds(_grok_period_end(config)),
    }


async def _fetch_grok_billing(
    http_client: httpx.AsyncClient, credentials: GrokCredentials, url: str
) -> dict[str, Any]:
    """GET a billing view; the result dict is a config payload or an error marker."""
    try:
        response = await http_client.get(
            url, headers=_grok_billing_headers(credentials), timeout=_USAGE_TIMEOUT
        )
    except httpx.HTTPError as exc:
        return {"_error": f"failed to reach the Grok billing API: {exc}"}
    if response.status_code in (401, 403):
        return {
            "_error": f"Grok access token rejected ({response.status_code}); run `grok login` again"
        }
    if response.status_code != 200:
        return {
            "_error": f"billing API returned {response.status_code}: {response.text[:200]}"
        }
    try:
        data = response.json()
    except json.JSONDecodeError:
        return {"_error": "billing API returned a non-JSON body"}
    if not isinstance(data, dict):
        return {"_error": "billing API returned an unexpected payload"}
    return data


async def fetch_grok_usage(
    http_client: httpx.AsyncClient, auth_manager: GrokAuthManager
) -> dict[str, Any]:
    """Fetch Grok quota via the Grok chat-proxy billing endpoint (mirrors Orca)."""
    try:
        credentials = await auth_manager.get_credentials()
    except Exception as exc:  # GrokAuthError and anything the file layer raises
        return _provider_result("grok", status="unavailable", error=str(exc))

    data = await _fetch_grok_billing(http_client, credentials, _GROK_BILLING_CREDITS_URL)
    error = data.pop("_error", None)
    if error is not None:
        return _provider_result("grok", status="error", error=error)
    config = data.get("config") if isinstance(data.get("config"), dict) else data
    weekly = _map_grok_weekly_credits(config)
    if weekly is not None:
        return _provider_result(
            "grok",
            status="ok",
            error=None,
            weekly=weekly,
            plan_type=_grok_plan_tier(config),
        )

    fallback = await _fetch_grok_billing(http_client, credentials, _GROK_BILLING_URL)
    error = fallback.pop("_error", None)
    if error is not None:
        return _provider_result("grok", status="error", error=error)
    fallback_config = fallback.get("config") if isinstance(fallback.get("config"), dict) else fallback
    monthly = _map_grok_monthly(fallback_config)
    if monthly is not None:
        return _provider_result(
            "grok",
            status="ok",
            error=None,
            monthly=monthly,
            # The tier rides the credits view's config, like Orca's fetcher.
            plan_type=_grok_plan_tier(config) or _grok_plan_tier(fallback_config),
        )
    # A signed-in account whose billing views expose no quota has no bar to
    # show — 'unavailable' hides the card body instead of painting an alert.
    return _provider_result(
        "grok", status="unavailable", error="billing response did not include credit usage"
    )
