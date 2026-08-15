"""Canonical usage result envelopes and reset-time normalization."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

import httpx

USAGE_TIMEOUT = httpx.Timeout(10.0)

SESSION_WINDOW_MINUTES = 300
WEEKLY_WINDOW_MINUTES = 10080


def provider_result(
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


async def fetch_usage_payload(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    *,
    provider: str,
    api_label: str,
    special_statuses: dict[int, str],
    logger: logging.Logger,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, httpx.Response | None]:
    """Returns (data, error_result, response); exactly one of data/error_result
    is non-None; response is None only when the request never reached the API."""
    try:
        response = await http_client.get(url, headers=headers, timeout=USAGE_TIMEOUT)
    except httpx.HTTPError as exc:
        logger.warning(f"{provider} usage fetch failed: %s", exc)
        return (
            None,
            provider_result(
                provider,
                status="error",
                error=f"failed to reach the {api_label} usage API: {exc}",
            ),
            None,
        )
    special = special_statuses.get(response.status_code)
    if special is not None:
        return None, provider_result(provider, status="error", error=special), response
    if response.status_code != 200:
        return (
            None,
            provider_result(
                provider,
                status="error",
                error=f"usage API returned {response.status_code}: {response.text[:200]}",
            ),
            response,
        )
    try:
        data = response.json()
    except json.JSONDecodeError:
        return (
            None,
            provider_result(
                provider, status="error", error="usage API returned a non-JSON body"
            ),
            response,
        )
    if not isinstance(data, dict):
        return (
            None,
            provider_result(
                provider, status="error", error="usage API returned an unexpected payload"
            ),
            response,
        )
    return data, None, response


def reset_epoch_seconds(value: Any) -> float | None:
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
            return reset_epoch_seconds(float(text))
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None
