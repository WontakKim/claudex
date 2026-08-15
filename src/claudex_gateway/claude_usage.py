"""Claude subscription usage probes for the dashboard's Usage tab.

Claude usage comes from the Anthropic OAuth usage API with the Claude Code
credentials (macOS Keychain or ~/.claude/.credentials.json).

The probes are advisory: they never raise, returning a status dict whose
"status" field is "ok", "unavailable" (no usable credentials), or "error"
(the provider rejected or could not be reached).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from claudex_gateway.claude.auth import (
    ClaudeAccountAuthError,
    ClaudeAccountAuthManager,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.usage.envelope import (
    SESSION_WINDOW_MINUTES,
    WEEKLY_WINDOW_MINUTES,
    fetch_usage_payload,
    provider_result,
    reset_epoch_seconds,
)

logger = logging.getLogger(__name__)

_CLAUDE_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
_CLAUDE_OAUTH_BETA = "oauth-2025-04-20"
# Match the Claude Code CLI's user-agent to stay aligned with the OAuth
# usage API contract (mirrors Orca's claude-fetcher).
_CLAUDE_CODE_USER_AGENT = "claude-code/2.1.0"
_CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
_KEYCHAIN_TIMEOUT_SECONDS = 3.0

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
        "resets_at": reset_epoch_seconds(raw.get("resets_at")),
    }


def _map_fable_weekly_window(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract Fable's model-scoped weekly quota from structured limits[] entries."""
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
            WEEKLY_WINDOW_MINUTES,
        )
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
        return provider_result(
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
    data, error_result, response = await fetch_usage_payload(
        http_client,
        _CLAUDE_USAGE_URL,
        {
            "Authorization": f"Bearer {token}",
            "anthropic-beta": _CLAUDE_OAUTH_BETA,
            "User-Agent": _CLAUDE_CODE_USER_AGENT,
        },
        provider="claude",
        api_label="Anthropic",
        special_statuses={
            401: "Claude OAuth token rejected (401); sign in again with `claude`",
            429: "usage API rate-limited (429); try again shortly",
        },
        logger=logger,
    )
    status_code = response.status_code if response is not None else None
    retry_after = _retry_after_seconds(response) if status_code == 429 else None
    if error_result is not None:
        return error_result, retry_after, status_code
    return (
        provider_result(
            "claude",
            status="ok",
            error=None,
            session=_map_claude_window(data.get("five_hour"), SESSION_WINDOW_MINUTES),
            weekly=_map_claude_window(data.get("seven_day"), WEEKLY_WINDOW_MINUTES),
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
    `provider_result` envelope. A 401 result gets one force-refresh retry,
    mirroring the serving path.
    """
    try:
        credentials = await auth_manager.get_credentials()
    except ClaudeAccountReauthRequiredError:
        raise
    except ClaudeAccountAuthError as exc:
        return (provider_result("claude", status="unavailable", error=str(exc)), None)

    result, retry_after, status_code = await _fetch_claude_usage_with_token(
        http_client, credentials.access_token
    )
    if status_code == 401:
        try:
            credentials = await auth_manager.get_credentials(force_refresh=True)
        except ClaudeAccountReauthRequiredError:
            raise
        except ClaudeAccountAuthError as exc:
            return (provider_result("claude", status="unavailable", error=str(exc)), None)
        result, retry_after, status_code = await _fetch_claude_usage_with_token(
            http_client, credentials.access_token
        )
        if status_code == 401:
            result = provider_result(
                "claude",
                status="error",
                error="account token rejected (401) even after refresh",
            )
    return (result, retry_after)
