"""Pure retry classification for ChatGPT backend responses."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

MIN_RETRY_AFTER_MS = 1_000
MAX_RETRY_AFTER_MS = 60_000
MAX_RATE_LIMIT_JITTER_MS = 2_000

_CHALLENGE_MARKERS = ("cf-chl", "challenge", "captcha", "__cf_bm")
_DECIMAL_SECONDS_PATTERN = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class RetryAction:
    """A typed instruction derived from one backend response."""

    action: Literal[
        "ok",
        "rate_limit",
        "session_expired",
        "challenge",
        "blocked",
        "entitlement",
        "origin_retry",
        "fatal",
    ]
    reason: str
    retry_after_ms: int | None = None


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key.lower(): value
        for key, value in headers.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _has_challenge_marker(body: str) -> bool:
    body_lower = body.lower()
    return any(marker in body_lower for marker in _CHALLENGE_MARKERS)


def _parse_retry_after_ms(value: str) -> int:
    stripped = value.strip()
    if _DECIMAL_SECONDS_PATTERN.fullmatch(stripped) is None:
        return MIN_RETRY_AFTER_MS
    try:
        retry_after_ms = int(stripped) * 1_000
    except ValueError:
        return MIN_RETRY_AFTER_MS
    return max(retry_after_ms, MIN_RETRY_AFTER_MS)


def classify_backend_response(
    status: int, headers: Mapping[str, str], body: str = ""
) -> RetryAction:
    """Classify a ChatGPT backend response without performing a retry."""
    normalized_headers = _normalize_headers(headers)
    cf_mitigated = normalized_headers.get("cf-mitigated", "").lower().strip()

    if cf_mitigated == "block":
        return RetryAction(
            action="blocked",
            reason="Cloudflare mitigation: block (cf-mitigated: block)",
        )
    if cf_mitigated == "challenge":
        return RetryAction(
            action="challenge",
            reason="Cloudflare mitigation: challenge (cf-mitigated: challenge)",
        )

    has_cf_ray = bool(normalized_headers.get("cf-ray"))

    if status == 200:
        return RetryAction(action="ok", reason="HTTP 200")

    if status == 401:
        return RetryAction(
            action="session_expired", reason="Session expired (HTTP 401)"
        )

    if status == 403:
        if has_cf_ray and _has_challenge_marker(body):
            return RetryAction(
                action="challenge",
                reason="Cloudflare bot challenge on HTTP 403 response",
            )
        return RetryAction(
            action="entitlement",
            reason="HTTP 403: account not entitled or model forbidden",
        )

    if status == 429:
        retry_after_ms = _parse_retry_after_ms(
            normalized_headers.get("retry-after", "")
        )
        return RetryAction(
            action="rate_limit",
            retry_after_ms=retry_after_ms,
            reason=f"HTTP 429: rate limited; retry after {retry_after_ms}ms",
        )

    if 500 <= status < 600:
        if has_cf_ray and _has_challenge_marker(body):
            return RetryAction(
                action="challenge",
                reason="Cloudflare edge 5xx response with challenge markers",
            )
        return RetryAction(
            action="origin_retry",
            reason=f"HTTP {status} from origin: bounded retry",
        )

    return RetryAction(action="fatal", reason=f"Unexpected HTTP {status}")


def rate_limit_delay_ms(retry_after_ms: int | None) -> int:
    """Add bounded jitter to a rate-limit delay and cap the final wait."""
    base_delay_ms = retry_after_ms if retry_after_ms is not None else 0
    jitter_ms = random.randint(0, MAX_RATE_LIMIT_JITTER_MS)
    return min(
        max(base_delay_ms, MIN_RETRY_AFTER_MS) + jitter_ms,
        MAX_RETRY_AFTER_MS,
    )
