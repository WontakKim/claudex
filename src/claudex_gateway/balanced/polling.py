"""Usage polling coordination for balanced routing."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.balanced.router import ClaudeBalancedRouter
from claudex_gateway.balanced.selection import _PEEK_WINDOW_TO_BINDING
from claudex_gateway.balanced.state_store import ClaudePoolRuntimeStateStore

# ==========================================================================
# Balanced usage poll coordinator
# ==========================================================================

# This coordinator is the only caller that performs per-account usage fetches
# while balanced routing is active.

# Budget: at most one actual upstream call per this many seconds, burst one
# (no credit banking -- a call arriving before the interval elapses is
# refused outright, not deferred or accumulated).
_USAGE_POLL_INTERVAL_SECONDS = 30.0

# Manual refresh: at most one NEW enqueue globally per this many seconds. A
# repeat request for an ALREADY-pending account always coalesces (never
# counted against this limiter, never refused).
_MANUAL_REFRESH_RATE_LIMIT_SECONDS = 5 * 60.0

PollTickOutcome = Literal["fetched", "budget_wait", "cooldown", "idle"]


@dataclass(frozen=True)
class PollTickResult:
    """`ClaudeUsagePollCoordinator.run_due_poll`'s outcome for one tick.

    `"fetched"` is the only outcome that performed a real upstream call
    (`account_id`/`manual`/`ok` describe it); every other outcome touched no
    network at all.
    """

    outcome: PollTickOutcome
    account_id: str | None = None
    manual: bool = False
    ok: bool | None = None


@dataclass(frozen=True)
class UsagePollAccount:
    """Identity fields needed to persist a ready account's usage observation.

    These are exactly the fields
    `ClaudePoolRuntimeStateStore.upsert_usage_observation` needs
    to persist a durable observation row for it.
    `account_profile_fingerprint` is `None` when the account has not
    captured one yet (mirrors `_install_balanced_quota_cooldown`'s own lazy
    lookup) -- the durable write is then skipped for that account this tick,
    though the in-memory router ingestion still happens.
    """

    account_id: str
    account_incarnation_id: str
    account_profile_fingerprint: str | None


@dataclass
class _AccountPollDiagnostics:
    """Per-account diagnostics updated only by an actual fetch."""

    last_outcome: str | None = None
    last_polled_monotonic: float | None = None
    last_ok: bool | None = None
    consecutive_failures: int = 0


@dataclass(frozen=True)
class UsagePollDiagnostics:
    """Aggregate usage poll coordinator diagnostics."""

    fetched_count: int
    cache_hit_count: int
    cooldown_count: int
    manual_enqueued_count: int
    manual_rate_limited_count: int
    manual_served_count: int
    last_tick_outcome: str | None
    last_fetched_account_id: str | None
    last_fetched_monotonic: float | None


class ClaudeUsagePollCoordinator:
    """Coordinate balanced-mode usage polling under a global fetch budget.

    This is the only caller of `ClaudeAccountUsageCache` that performs real
    upstream fetches while balanced routing is active.

    `run_due_poll` performs AT MOST ONE actual upstream call per invocation
    (sequential scheduling) and never more than one per
    `poll_interval_seconds` (the budget, burst one -- `_last_call_monotonic`
    is the sole gate, checked before touching the cache at all). Within a
    tick, ready accounts are tried in `_due_order`: every account with no
    successful observation at all yet ("missing", including one whose last
    attempt failed) is tried before any account that already has data,
    which is itself tried oldest-first, ties broken by account id for a
    deterministic, stable order across ticks. A candidate's own due-ness
    beyond that ordering is delegated entirely to
    `ClaudeAccountUsageCache.poll` -- an account still within its own
    TTL/failure-backoff window reports `"cache_hit"` (no call, try the next
    candidate in the SAME tick) rather than blocking the budget, which is
    exactly how a persistently failing account yields its slot to the rest
    of the pool instead of starving it (its 60s failure backoff alone caps
    how often it can even be attempted).

    Manual refresh (`request_manual_refresh`) never fetches inline -- it
    only enqueues, coalesced per account and globally rate-limited
    (`manual_rate_limit_seconds`) -- and a pending manual account is only
    ever tried by `run_due_poll` once at least one automatic (non-manual)
    account has been serviced by THIS coordinator at least once, and only
    once every currently-due automatic candidate has yielded no fetch this
    tick (automatic work always wins a tick outright). A manual attempt
    forces the cache to actually re-fetch (`force=True`) even though the
    account may still be within its own TTL -- the entire point of an
    on-demand refresh -- but, like every other attempt, never bypasses the
    shared Retry-After cooldown.
    """

    def __init__(
        self,
        *,
        cache: ClaudeAccountUsageCache,
        router: ClaudeBalancedRouter,
        store: ClaudePoolRuntimeStateStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        poll_interval_seconds: float = _USAGE_POLL_INTERVAL_SECONDS,
        manual_rate_limit_seconds: float = _MANUAL_REFRESH_RATE_LIMIT_SECONDS,
    ) -> None:
        self._cache = cache
        self._router = router
        self._store = store
        self._clock = clock
        self._wall_clock = wall_clock
        self._poll_interval_seconds = poll_interval_seconds
        self._manual_rate_limit_seconds = manual_rate_limit_seconds

        self._last_call_monotonic: float | None = None
        self._any_automatic_serviced = False
        self._pending_manual: dict[str, float] = {}
        self._last_manual_enqueue_at: float | None = None
        self._diagnostics: dict[str, _AccountPollDiagnostics] = {}
        self._last_tick_outcome: str | None = None
        self._last_fetched_account_id: str | None = None
        self._last_fetched_monotonic: float | None = None

        self.fetched_count = 0
        self.cache_hit_count = 0
        self.cooldown_count = 0
        self.manual_enqueued_count = 0
        self.manual_rate_limited_count = 0
        self.manual_served_count = 0

    @property
    def poll_interval_seconds(self) -> float:
        """Return the fetch interval used by the runtime's background driver.

        The coordinator permits at most one actual upstream call per interval.
        """
        return self._poll_interval_seconds

    # -- manual refresh: enqueue-only, coalesced, globally rate-limited -----

    def request_manual_refresh(self, account_id: str, *, now: float | None = None) -> bool:
        """Enqueue a manual refresh for `account_id`; returns whether it is
        queued afterward (freshly enqueued or already coalesced) -- `False`
        only when it is neither already pending NOR within the global rate
        limit's cooldown.
        """
        now = self._clock() if now is None else now
        if account_id in self._pending_manual:
            return True
        if (
            self._last_manual_enqueue_at is not None
            and now - self._last_manual_enqueue_at < self._manual_rate_limit_seconds
        ):
            self.manual_rate_limited_count += 1
            return False
        self._pending_manual[account_id] = now
        self._last_manual_enqueue_at = now
        self.manual_enqueued_count += 1
        return True

    def is_manual_refresh_pending(self, account_id: str) -> bool:
        return account_id in self._pending_manual

    # -- diagnostics --------------------------------------------------------

    def _diagnostic(self, account_id: str) -> _AccountPollDiagnostics:
        return self._diagnostics.setdefault(account_id, _AccountPollDiagnostics())

    def account_diagnostics(self, account_id: str) -> dict[str, Any] | None:
        entry = self._diagnostics.get(account_id)
        if entry is None:
            return None
        return {
            "last_outcome": entry.last_outcome,
            "last_polled_monotonic": entry.last_polled_monotonic,
            "last_ok": entry.last_ok,
            "consecutive_failures": entry.consecutive_failures,
            "manual_pending": account_id in self._pending_manual,
        }

    def diagnostics(self) -> UsagePollDiagnostics:
        return UsagePollDiagnostics(
            fetched_count=self.fetched_count,
            cache_hit_count=self.cache_hit_count,
            cooldown_count=self.cooldown_count,
            manual_enqueued_count=self.manual_enqueued_count,
            manual_rate_limited_count=self.manual_rate_limited_count,
            manual_served_count=self.manual_served_count,
            last_tick_outcome=self._last_tick_outcome,
            last_fetched_account_id=self._last_fetched_account_id,
            last_fetched_monotonic=self._last_fetched_monotonic,
        )

    # -- due ordering: missing-window-first, then oldest-due, stable --------

    def _due_order(self, ready_account_ids: Sequence[str]) -> list[str]:
        missing: list[str] = []
        aged: list[tuple[float, str]] = []
        for account_id in sorted(set(ready_account_ids)):
            peeked = self._cache.peek_with_metadata(account_id)
            if peeked is None or not peeked[1]:
                missing.append(account_id)
                continue
            _, metadata = peeked
            min_age = min(window["age_seconds"] for window in metadata.values())
            aged.append((min_age, account_id))
        aged.sort(key=lambda item: (-item[0], item[1]))
        return missing + [account_id for _age, account_id in aged]

    # -- one tick: at most one actual upstream call --------------------------

    async def run_due_poll(
        self,
        ready_account_ids: Sequence[str],
        *,
        accounts: Mapping[str, UsagePollAccount] | None = None,
        now: float | None = None,
    ) -> PollTickResult:
        """Perform at most one actual upstream call for this tick.

        `ready_account_ids` is re-supplied every call (the caller's own
        registry snapshot) -- the coordinator keeps no membership state of
        its own. When supplied, `accounts` also persists a successful
        observation; otherwise only in-memory router ingestion occurs.
        """
        now = self._clock() if now is None else now
        if (
            self._last_call_monotonic is not None
            and now - self._last_call_monotonic < self._poll_interval_seconds
        ):
            self._last_tick_outcome = "budget_wait"
            return PollTickResult(outcome="budget_wait")

        ready_ids = list(ready_account_ids)
        for account_id in self._due_order(ready_ids):
            outcome = await self._attempt(account_id, manual=False, accounts=accounts)
            if outcome is not None:
                return outcome

        ready_id_set = set(ready_ids)
        pending_ready = [aid for aid in self._pending_manual if aid in ready_id_set]
        if pending_ready and self._any_automatic_serviced:
            account_id = min(pending_ready, key=lambda aid: self._pending_manual[aid])
            outcome = await self._attempt(account_id, manual=True, accounts=accounts)
            if outcome is not None:
                return outcome

        self._last_tick_outcome = "idle"
        return PollTickResult(outcome="idle")

    async def _attempt(
        self,
        account_id: str,
        *,
        manual: bool,
        accounts: Mapping[str, UsagePollAccount] | None,
    ) -> PollTickResult | None:
        """One candidate's attempt. Returns `None` (try the next candidate,
        same tick) for a `"cache_hit"` -- no call was made, so the budget is
        untouched -- and the tick's final result for `"cooldown"`/`"fetched"`.
        """
        poll_result = await self._cache.poll(account_id, force=manual)
        if poll_result.source == "cache_hit":
            self.cache_hit_count += 1
            return None
        if poll_result.source == "cooldown":
            self.cooldown_count += 1
            self._last_tick_outcome = "cooldown"
            return PollTickResult(outcome="cooldown", account_id=account_id, manual=manual)

        # "fetched": this tick's one actual upstream call.
        ok = poll_result.result.get("status") == "ok"
        self._last_call_monotonic = self._clock()
        self.fetched_count += 1
        self._last_fetched_account_id = account_id
        self._last_fetched_monotonic = self._last_call_monotonic
        self._last_tick_outcome = "fetched"

        diagnostic = self._diagnostic(account_id)
        diagnostic.last_outcome = "fetched"
        diagnostic.last_polled_monotonic = self._last_call_monotonic
        diagnostic.last_ok = ok
        diagnostic.consecutive_failures = 0 if ok else diagnostic.consecutive_failures + 1

        self._ingest_observation(account_id, accounts=accounts)
        self._pending_manual.pop(account_id, None)
        if manual:
            self.manual_served_count += 1
        else:
            self._any_automatic_serviced = True

        return PollTickResult(outcome="fetched", account_id=account_id, manual=manual, ok=ok)

    def _ingest_observation(
        self, account_id: str, *, accounts: Mapping[str, UsagePollAccount] | None
    ) -> None:
        """Feed the fresh observation into the router's in-memory
        `ObservationView` (always) and, when a store and this account's
        identity are both available, submit one durable row per window as a
        fire-and-forget write, like `classify_capability_evidence`'s
        own low-priority store write: the router's in-memory state (which
        the picker actually reads) is already updated synchronously above,
        so nothing waits on this.
        """
        peeked = self._cache.peek_with_metadata(account_id)
        self._router.ingest_usage_peek(account_id, peeked)
        if peeked is None or self._store is None or accounts is None:
            return
        snapshot = accounts.get(account_id)
        if snapshot is None or snapshot.account_profile_fingerprint is None:
            return
        envelope, metadata = peeked
        wall_now = self._wall_clock()
        for peek_window, binding_window in _PEEK_WINDOW_TO_BINDING.items():
            window_envelope = envelope.get(peek_window)
            window_meta = metadata.get(peek_window)
            if not isinstance(window_envelope, dict) or not isinstance(window_meta, dict):
                continue
            used_percent = window_envelope.get("used_percent")
            if not isinstance(used_percent, (int, float)):
                continue
            raw_age = window_meta.get("age_seconds")
            age_seconds = float(raw_age) if isinstance(raw_age, (int, float)) else 0.0
            raw_reset_at = window_meta.get("reset_at")
            reset_at_utc = float(raw_reset_at) if isinstance(raw_reset_at, (int, float)) else None
            reset_identity = f"{reset_at_utc:.3f}" if reset_at_utc is not None else "none"
            self._store.upsert_usage_observation(
                account_id=account_id,
                window=binding_window,
                account_incarnation_id=snapshot.account_incarnation_id,
                account_profile_fingerprint=snapshot.account_profile_fingerprint,
                used_percent=min(100.0, max(0.0, float(used_percent))),
                reset_identity=reset_identity,
                reset_at_utc=reset_at_utc,
                observed_at_utc=wall_now - age_seconds,
                source=str(window_meta.get("source") or "usage_api"),
            )
