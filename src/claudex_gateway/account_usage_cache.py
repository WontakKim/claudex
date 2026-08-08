"""In-memory TTL cache for per-account Claude usage probes.

The Anthropic OAuth usage endpoint is tightly rate-limited, and the dashboard
asks for every registered account at once. ``ClaudeAccountUsageCache`` wraps
the per-account probe with a success TTL, a per-account failure backoff, a
single lock that serializes every upstream fetch (so N accounts never hit the
endpoint concurrently), and a global cooldown honoring a 429's Retry-After —
during which stale results are served and nothing touches the network.
Modeled on ``ContextWindowCache``; per-key entries and the rate-limit gate
are the additions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from claudex_gateway.usage import _provider_result

logger = logging.getLogger(__name__)

_TTL_SECONDS = 120.0
_FAILURE_BACKOFF_SECONDS = 60.0
_RETRY_AFTER_MIN_SECONDS = 5.0
_RETRY_AFTER_MAX_SECONDS = 3600.0


@dataclass
class _Entry:
    result: dict[str, Any]
    fetched_at: float
    ok: bool


class ClaudeAccountUsageCache:
    """Caches account-id -> usage-envelope results fetched on demand.

    ``fetch`` maps an account id to ``(result, retry_after_seconds)`` — the
    shape ``usage.fetch_claude_account_usage`` returns — and must not raise
    for ordinary failures; an unexpected exception is logged, converted to an
    error envelope, and backed off, never propagated (a broken probe must not
    500 the endpoint that calls this).
    """

    def __init__(
        self,
        fetch: Callable[[str], Awaitable[tuple[dict[str, Any], float | None]]],
        *,
        ttl_seconds: float = _TTL_SECONDS,
        failure_backoff_seconds: float = _FAILURE_BACKOFF_SECONDS,
        retry_after_min: float = _RETRY_AFTER_MIN_SECONDS,
        retry_after_max: float = _RETRY_AFTER_MAX_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._ttl_seconds = ttl_seconds
        self._failure_backoff_seconds = failure_backoff_seconds
        self._retry_after_min = retry_after_min
        self._retry_after_max = retry_after_max
        self._clock = clock
        self._lock = asyncio.Lock()
        self._entries: dict[str, _Entry] = {}
        # Monotonic deadline before which NO account may fetch upstream. The
        # usage API's rate limit is per client, not per account, so one 429
        # cools every account down together.
        self._not_before: float = 0.0

    async def get(self, account_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return a usage envelope per account id, fetching only where needed.

        Fresh entries (success TTL, or failure backoff for non-ok results)
        are served without any locking. Misses are refetched one at a time
        under the single lock; while the global Retry-After cooldown is
        active, misses serve their stale entry — or a synthesized error
        envelope when the account has never been fetched.
        """
        results: dict[str, dict[str, Any]] = {}
        misses: list[str] = []
        now = self._clock()
        for account_id in account_ids:
            entry = self._entries.get(account_id)
            if entry is not None and self._is_fresh(entry, now):
                results[account_id] = entry.result
            else:
                misses.append(account_id)

        if misses:
            async with self._lock:
                for account_id in misses:
                    results[account_id] = await self._refresh(account_id)
        return results

    def peek(self, account_id: str) -> dict[str, Any] | None:
        """Return the cached envelope for `account_id` without ever fetching.

        Entry age is deliberately ignored: the consumer (the serve path's
        429 cooldown derivation) only reads future `resets_at` values, which
        a stale envelope still reports accurately.
        """
        entry = self._entries.get(account_id)
        return entry.result if entry is not None else None

    async def _refresh(self, account_id: str) -> dict[str, Any]:
        # Re-check under the lock: a concurrent get() may have refreshed this
        # account, or started a cooldown, while we waited.
        now = self._clock()
        entry = self._entries.get(account_id)
        if entry is not None and self._is_fresh(entry, now):
            return entry.result
        if now < self._not_before:
            if entry is not None:
                return entry.result
            return _provider_result(
                "claude",
                status="error",
                error="usage API rate-limited; retrying after the cooldown",
            )

        try:
            result, retry_after = await self._fetch(account_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "per-account usage fetch failed unexpectedly for %s", account_id, exc_info=True
            )
            result = _provider_result(
                "claude", status="error", error="usage probe failed unexpectedly"
            )
            retry_after = None

        ok = result.get("status") == "ok"
        self._entries[account_id] = _Entry(result=result, fetched_at=self._clock(), ok=ok)

        # A 429 opens the global cooldown: Retry-After when the server sent
        # one (clamped to sane bounds), the failure backoff otherwise. The
        # error strings are this codebase's own (usage.py), so the substring
        # check is a stable signal for the headerless case.
        rate_limited = retry_after is not None or (not ok and "429" in (result.get("error") or ""))
        if rate_limited:
            cooldown = (
                min(self._retry_after_max, max(self._retry_after_min, retry_after))
                if retry_after is not None
                else self._failure_backoff_seconds
            )
            self._not_before = self._clock() + cooldown
        return result

    def _is_fresh(self, entry: _Entry, now: float) -> bool:
        window = self._ttl_seconds if entry.ok else self._failure_backoff_seconds
        return now - entry.fetched_at < window
