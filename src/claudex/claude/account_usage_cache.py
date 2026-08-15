"""In-memory TTL cache for per-account Claude usage probes.

The Anthropic OAuth usage endpoint is tightly rate-limited, and the dashboard
asks for every registered account at once. ``ClaudeAccountUsageCache`` wraps
the per-account probe with a success TTL, a per-account failure backoff, a
single lock that serializes every upstream fetch (so N accounts never hit the
endpoint concurrently), and a global cooldown honoring a 429's Retry-After —
during which stale results are served and nothing touches the network.
Modeled on ``ModelCatalogCache``; per-key entries and the rate-limit gate
are the additions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from claudex.usage.envelope import provider_result

logger = logging.getLogger(__name__)

_TTL_SECONDS = 120.0
_FAILURE_BACKOFF_SECONDS = 60.0
_RETRY_AFTER_MIN_SECONDS = 5.0
_RETRY_AFTER_MAX_SECONDS = 3600.0

# The only per-window source this cache ever observes: a live usage-API fetch.
_SOURCE_USAGE_API = "usage_api"

# Envelope keys (see usage_envelope.provider_result / _map_claude_window) that carry
# a per-window quota reading and are eligible for observation metadata.
_WINDOW_NAMES = ("session", "weekly", "fable_weekly")


@dataclass
class _WindowObservation:
    """When and how a single window's reading was captured for an entry."""

    observed_at: float
    source: str
    resets_at: float | None


@dataclass
class _Entry:
    result: dict[str, Any]
    fetched_at: float
    ok: bool
    windows: dict[str, _WindowObservation] = field(default_factory=dict)


@dataclass(frozen=True)
class PollResult:
    """Outcome of one coordinator-driven poll attempt.

    `source` identifies which cache decision served `result` for one account:
    an actual upstream call (`"fetched"`), an entry still within its
    TTL/failure-backoff window (`"cache_hit"`), or the shared Retry-After
    cooldown suppressing the attempt (`"cooldown"`).
    """

    source: Literal["fetched", "cache_hit", "cooldown"]
    result: dict[str, Any]


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

    def peek_with_metadata(
        self, account_id: str
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
        """Return the cached envelope plus per-window observation metadata.

        Like `peek`, this never fetches. Unlike `peek`, it reports how each
        window's reading was obtained: `age_seconds` (computed against the
        cache clock, not ignored), `source`, and the normalized `reset_at`
        epoch. Windows absent from the last successful fetch — or all
        windows, when the last fetch failed — contribute no metadata entry.
        Returns None only when nothing has ever been cached for the account.
        """
        entry = self._entries.get(account_id)
        if entry is None:
            return None
        now = self._clock()
        metadata = {
            window_name: {
                "age_seconds": now - observation.observed_at,
                "source": observation.source,
                "reset_at": observation.resets_at,
            }
            for window_name, observation in entry.windows.items()
        }
        return entry.result, metadata

    async def poll(self, account_id: str, *, force: bool = False) -> PollResult:
        """Attempt one account using the cache's freshness and cooldown rules.

        Reuses this cache's own freshness/backoff/cooldown decisions exactly
        as `get` would for a single id, but reports WHICH decision served the
        result instead of only the envelope -- so a caller with its own
        scheduling cadence (the balanced poll coordinator) can tell whether
        this call actually consumed a real upstream request. `force=True`
        (the coordinator's manual-refresh path) skips this cache's own
        TTL/failure-backoff freshness gate -- an on-demand refresh must be
        able to fetch again even though the last one is still "fresh" -- but
        never bypasses the shared Retry-After cooldown, which is a hard
        upstream constraint, not a scheduling preference. WHEN to call this
        for a given account is entirely the caller's business; this only
        ever reports what happened.
        """
        now = self._clock()
        entry = self._entries.get(account_id)
        if not force and entry is not None and self._is_fresh(entry, now):
            return PollResult(source="cache_hit", result=entry.result)
        async with self._lock:
            now = self._clock()
            entry = self._entries.get(account_id)
            if not force and entry is not None and self._is_fresh(entry, now):
                return PollResult(source="cache_hit", result=entry.result)
            if now < self._not_before:
                result = (
                    entry.result
                    if entry is not None
                    else provider_result(
                        "claude",
                        status="error",
                        error="usage API rate-limited; retrying after the cooldown",
                    )
                )
                return PollResult(source="cooldown", result=result)
            result = await self._refresh(account_id, force=force)
            return PollResult(source="fetched", result=result)

    async def _refresh(self, account_id: str, *, force: bool = False) -> dict[str, Any]:
        # Re-check under the lock: a concurrent get() may have refreshed this
        # account, or started a cooldown, while we waited. `force` (only ever
        # passed by `poll`) skips this re-check too -- a manual refresh must
        # still fetch even if a concurrent caller just did.
        now = self._clock()
        entry = self._entries.get(account_id)
        if not force and entry is not None and self._is_fresh(entry, now):
            return entry.result
        if now < self._not_before:
            if entry is not None:
                return entry.result
            return provider_result(
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
            result = provider_result(
                "claude", status="error", error="usage probe failed unexpectedly"
            )
            retry_after = None

        ok = result.get("status") == "ok"
        fetched_at = self._clock()
        windows: dict[str, _WindowObservation] = {}
        if ok:
            for window_name in _WINDOW_NAMES:
                window = result.get(window_name)
                if isinstance(window, dict):
                    windows[window_name] = _WindowObservation(
                        observed_at=fetched_at,
                        source=_SOURCE_USAGE_API,
                        resets_at=window.get("resets_at"),
                    )
        self._entries[account_id] = _Entry(
            result=result, fetched_at=fetched_at, ok=ok, windows=windows
        )

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
