"""Codex, Kimi, and Grok subscription usage probes for the dashboard's Usage tab.

The probes read local provider credentials and call each provider's own usage
endpoint directly; gateway traffic plays no part. They are advisory: they
never raise, returning a status dict whose "status" field is "ok",
"unavailable" (no usable credentials), or "error" (the provider rejected or
could not be reached).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import httpx

from claudex_gateway.providers.codex_auth import CodexAuthManager, CodexCredentials
from claudex_gateway.providers.kimi_auth import KimiAuthManager
from claudex_gateway.providers.grok_auth import GrokAuthManager, GrokCredentials
from claudex_gateway.usage.envelope import (
    SESSION_WINDOW_MINUTES,
    USAGE_TIMEOUT,
    WEEKLY_WINDOW_MINUTES,
    fetch_usage_payload,
    provider_result,
    reset_epoch_seconds,
)

logger = logging.getLogger(__name__)

_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
# Spending an earned reset credit clears the rate-limit windows early. Unlike
# every other call in this module this one writes, and burns something the
# user cannot get back, so it only ever runs on an explicit request.
_CODEX_RESET_CONSUME_URL = "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits/consume"
# Outcome codes the backend answers a consume request with (mirrors Orca).
_CODEX_RESET_OUTCOMES = ("reset", "nothing_to_reset", "no_credit", "already_redeemed")

_KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"

# Tolerate the one-minute drift seen in older Codex bucket lengths.
_WINDOW_TOLERANCE_MINUTES = 1

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
        "resets_at": reset_epoch_seconds(raw.get("reset_at")),
    }


def _codex_window_kind(window_minutes: Any) -> str | None:
    if not isinstance(window_minutes, (int, float)) or isinstance(window_minutes, bool):
        return None
    if abs(window_minutes - SESSION_WINDOW_MINUTES) <= _WINDOW_TOLERANCE_MINUTES:
        return "session"
    if abs(window_minutes - WEEKLY_WINDOW_MINUTES) <= _WINDOW_TOLERANCE_MINUTES:
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
        return provider_result("codex", status="unavailable", error=str(exc))
    if credentials.is_api_key:
        return provider_result(
            "codex",
            status="unavailable",
            error="API key billing has no plan usage windows",
        )
    headers = _codex_backend_headers(credentials)
    data, error_result, _response = await fetch_usage_payload(
        http_client,
        _CODEX_USAGE_URL,
        headers,
        provider="codex",
        api_label="ChatGPT",
        special_statuses={},
        logger=logger,
    )
    if error_result is not None:
        return error_result
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
    return provider_result(
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
            timeout=USAGE_TIMEOUT,
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


# ---------------------------------------------------------------------------
# Kimi usage — payload mapping mirrors Orca's kimi-fetcher
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    """Kimi reports quota numbers as strings ("100"); accept ints/floats too."""
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


def _kimi_window_minutes(window: Any) -> float | None:
    """Convert a limits[] window ({duration, timeUnit}) to minutes."""
    if not isinstance(window, dict):
        return None
    duration = _to_float(window.get("duration"))
    if duration is None:
        return None
    unit = str(window.get("timeUnit") or "").upper()
    if "SECOND" in unit:
        return duration / 60
    if "HOUR" in unit:
        return duration * 60
    if "DAY" in unit:
        return duration * 60 * 24
    # MINUTE and unrecognized units both read as minutes (mirrors Orca).
    return duration


def _map_kimi_quota(raw: Any, window_minutes: float | None) -> dict[str, Any] | None:
    """Map a Kimi quota detail ({limit, used|remaining, resetTime}) to a window."""
    if not isinstance(raw, dict):
        return None
    limit = _to_float(raw.get("limit"))
    used = _to_float(raw.get("used"))
    if used is None:
        # Older responses only carry remaining; used = limit - remaining.
        remaining = _to_float(raw.get("remaining"))
        if remaining is not None and limit is not None:
            used = limit - remaining
    if limit is None or limit <= 0 or used is None:
        return None
    # The CLI has shipped both resetTime and resetAt spellings.
    resets_at = reset_epoch_seconds(raw.get("resetTime") or raw.get("resetAt"))
    return {
        "used_percent": min(100.0, max(0.0, used / limit * 100)),
        "window_minutes": window_minutes,
        "resets_at": resets_at,
    }


def _map_kimi_session_window(limits: Any) -> dict[str, Any] | None:
    """Pick the limits[] entry closest to a 5-hour session (mirrors Orca)."""
    if not isinstance(limits, list):
        return None
    best: dict[str, Any] | None = None
    best_distance = math.inf
    for entry in limits:
        if not isinstance(entry, dict):
            continue
        minutes = _kimi_window_minutes(entry.get("window")) or SESSION_WINDOW_MINUTES
        mapped = _map_kimi_quota(entry.get("detail"), minutes)
        if mapped is None:
            continue
        distance = abs(minutes - SESSION_WINDOW_MINUTES)
        if best is None or distance < best_distance:
            best, best_distance = mapped, distance
    return best


async def fetch_kimi_usage(
    http_client: httpx.AsyncClient, auth_manager: KimiAuthManager
) -> dict[str, Any]:
    """Fetch Kimi For Coding quota via the coding backend's usages endpoint.

    Orca reads the Kimi CLI's credentials read-only; the gateway owns its
    OAuth tokens, so the auth manager's refresh path applies as usual.
    """
    try:
        credentials = await auth_manager.get_credentials()
    except Exception as exc:  # KimiAuthError and anything the file layer raises
        return provider_result("kimi", status="unavailable", error=str(exc))
    # Bearer token + Accept only; the endpoint authenticates by token.
    data, error_result, _response = await fetch_usage_payload(
        http_client,
        _KIMI_USAGE_URL,
        {
            "Authorization": f"Bearer {credentials.access_token}",
            "Accept": "application/json",
        },
        provider="kimi",
        api_label="Kimi",
        special_statuses={
            401: "Kimi access token rejected (401); run `kimi login` again"
        },
        logger=logger,
    )
    if error_result is not None:
        return error_result
    # The top-level usage block is the weekly quota; limits[] carries the
    # shorter rolling windows, of which the 5-hour one is the session view.
    session = _map_kimi_session_window(data.get("limits"))
    weekly = _map_kimi_quota(data.get("usage"), WEEKLY_WINDOW_MINUTES)
    if session is None and weekly is None:
        return provider_result(
            "kimi", status="error", error="usage response did not include quota windows"
        )
    user = data.get("user")
    membership = user.get("membership") if isinstance(user, dict) else None
    level = membership.get("level") if isinstance(membership, dict) else None
    plan_type = (
        level.removeprefix("LEVEL_").lower()
        if isinstance(level, str) and level.startswith("LEVEL_")
        else None
    )
    return provider_result(
        "kimi",
        status="ok",
        error=None,
        session=session,
        weekly=weekly,
        plan_type=plan_type,
    )


# --- Grok --------------------------------------------------------------------
# Mirrors Orca's grok-fetcher: the chat-proxy billing endpoint is read with
# the Grok CLI's OAuth session; the credits view carries the weekly window,
# and the default view is the monthly-budget fallback for unified-billing
# accounts whose credits view omits creditUsagePercent.

_GROK_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing"
_GROK_BILLING_CREDITS_URL = _GROK_BILLING_URL + "?format=credits"
_XAI_TOKEN_AUTH_VALUE = "xai-grok-cli"
_MONTHLY_WINDOW_MINUTES = 43200


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
        "window_minutes": WEEKLY_WINDOW_MINUTES,
        "resets_at": reset_epoch_seconds(_grok_period_end(config)),
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
        "resets_at": reset_epoch_seconds(_grok_period_end(config)),
    }


async def _fetch_grok_billing(
    http_client: httpx.AsyncClient, credentials: GrokCredentials, url: str
) -> dict[str, Any]:
    """GET a billing view; the result dict is a config payload or an error marker."""
    try:
        response = await http_client.get(
            url, headers=_grok_billing_headers(credentials), timeout=USAGE_TIMEOUT
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
        return provider_result("grok", status="unavailable", error=str(exc))

    data = await _fetch_grok_billing(http_client, credentials, _GROK_BILLING_CREDITS_URL)
    error = data.pop("_error", None)
    if error is not None:
        return provider_result("grok", status="error", error=error)
    config = data.get("config") if isinstance(data.get("config"), dict) else data
    weekly = _map_grok_weekly_credits(config)
    if weekly is not None:
        return provider_result(
            "grok",
            status="ok",
            error=None,
            weekly=weekly,
            plan_type=_grok_plan_tier(config),
        )

    fallback = await _fetch_grok_billing(http_client, credentials, _GROK_BILLING_URL)
    error = fallback.pop("_error", None)
    if error is not None:
        return provider_result("grok", status="error", error=error)
    fallback_config = fallback.get("config") if isinstance(fallback.get("config"), dict) else fallback
    monthly = _map_grok_monthly(fallback_config)
    if monthly is not None:
        return provider_result(
            "grok",
            status="ok",
            error=None,
            monthly=monthly,
            # The tier rides the credits view's config, like Orca's fetcher.
            plan_type=_grok_plan_tier(config) or _grok_plan_tier(fallback_config),
        )
    # A signed-in account whose billing views expose no quota has no bar to
    # show — 'unavailable' hides the card body instead of painting an alert.
    return provider_result(
        "grok", status="unavailable", error="billing response did not include credit usage"
    )
