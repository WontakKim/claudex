"""Pin, cooldown, capability, and migration state for balanced routing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from claudex_gateway.balanced.selection import (
    DEFAULT_PIN_MAP_MAX_ENTRIES,
    DEFAULT_PIN_TTL_CONTENT_HASH_SECONDS,
    DEFAULT_PIN_TTL_UUID_SECONDS,
    _PEEK_WINDOW_TO_BINDING,
    AccountCandidate,
    FamilyGateOutcome,
    NoEligibleAccountError,
    ObservationView,
    SessionKey,
    _wall_to_monotonic,
    binding_windows,
    classify_balanced_cooldown_scope,
    is_eligible_candidate,
    pick_weighted_hrw,
    quota_family,
    select_weights,
    unknown_floor,
    warning_factor,
)
from claudex_gateway.balanced.state_model import RestoreResult
from claudex_gateway.balanced.state_store import ClaudePoolRuntimeStateStore

# -- durable cooldowns and capability evidence -----------------------------

# New cooldowns use `claude_account_pool.rate_limit_cooldown_seconds`'s
# [5s, 7d] clamp. Restored cooldowns use a looser [1s, 7d] clamp.
_COOLDOWN_RESTORE_MIN_SECONDS = 1.0
_COOLDOWN_RESTORE_MAX_SECONDS = 7 * 24 * 3600.0

# Capability evidence is always "eligible" under a fixed classifier version
# and a one-hour TTL; this classifier never writes "denied" evidence.
CAPABILITY_CLASSIFIER_VERSION = "v1"
CAPABILITY_EVIDENCE_TTL_SECONDS = 3600.0


@dataclass
class _CooldownEntry:
    # One deadline per clock; the entry expires when EITHER passes. The
    # monotonic deadline bounds the cooldown against wall-clock jumps; the
    # wall deadline bounds it against the monotonic clock pausing while the
    # machine sleeps (macOS `time.monotonic` does not advance during sleep).
    deadline_monotonic: float
    deadline_wall: float
    account_incarnation_id: str
    reason: str


def _entry_cooldown_deadline(entry: _CooldownEntry | None, now: float, wall_now: float) -> float | None:
    """The entry's effective monotonic deadline — the earlier of its two
    deadlines, re-expressed on the monotonic clock — or `None` when the entry
    is absent or either deadline has passed."""
    if entry is None:
        return None
    remaining = min(entry.deadline_monotonic - now, entry.deadline_wall - wall_now)
    if remaining <= 0.0:
        return None
    return now + remaining


@dataclass(frozen=True)
class _CapabilityEvidenceEntry:
    account_incarnation_id: str
    account_profile_fingerprint: str
    classifier_version: str
    expires_at_monotonic: float


# -- pin map ---------------------------------------------------------------


class PendingDurabilityBarrier:
    """A cancellation-safe, resolve-exactly-once gate for one pin generation's durable write.

    Every request resolving this pin generation, not just its creator, awaits
    `wait()`, which shields the underlying event so a waiter's own cancellation
    can never cancel or double-fire the barrier's resolution before starting
    an upstream attempt.
    `resolve()` runs exactly once, whether the durable write succeeded or
    failed; a second call is a no-op.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._resolved = False

    @property
    def is_resolved(self) -> bool:
        return self._resolved

    def resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._event.set()

    async def wait(self) -> None:
        await asyncio.shield(self._event.wait())


@dataclass
class PinEntry:
    """One in-memory pin-map row: `digest -> {account_id, key_kind, last_seen_monotonic, generation, ...}`."""

    session_key_digest: bytes
    key_kind: str
    account_id: str
    account_incarnation_id: str
    generation: int
    last_seen_monotonic: float
    expires_at_monotonic: float
    migration_reserved: bool = False
    pending_durability: PendingDurabilityBarrier | None = None
    # Diagnostic metadata only — the family is already baked into the salted
    # `session_key_digest` and never participates in any key or decision.
    model_family: str = ""


@dataclass(frozen=True)
class PlacementResult:
    """`place_session`'s outcome: the picked/followed account and its pin's identity."""

    account_id: str
    session_key_digest: bytes
    key_kind: str
    generation: int
    created: bool
    durability_barrier: PendingDurabilityBarrier | None


# -- migration machinery: reservations, waiters, tokens, generation CAS -----

MigrationOutcome = Literal[
    "pending",
    "committed",
    "cas_lost",
    "retryable_preheader_failure",
    "terminal_failure",
    "target_removed",
    "mode_stopped",
]

CommitOutcome = Literal["committed", "cas_lost", "target_removed"]


@dataclass
class MigrationReservation:
    """One attempt-owned, cancellation-safe reservation per session generation.

    The reservation is created in the same no-await critical section as its
    migration-attempt token. `resolved_event` and the stored `outcome` form the
    resolution primitive every waiter awaits through `asyncio.shield`, so
    cancelling one waiter cannot cancel or double-fire the shared event.
    `outcome` starts `"pending"` and becomes immutable after the owner-terminal
    path verifies `owner_attempt_id` and resolves it. Reservations are never
    persisted because they refer to live tasks and cancellation scopes.
    """

    source_account: str
    source_generation: int
    target_account: str
    owner_attempt_id: str
    outcome: MigrationOutcome = "pending"
    resolved_event: asyncio.Event = field(default_factory=asyncio.Event)


class ClaudeBalancedRouter:
    """Balanced picker state for pressures, weighted HRW, and the pin map.

    Owns the in-memory `ObservationView`, the `digest -> PinEntry` pin map
    (TTLs, LRU eviction, exactly-once-decrementing counters), each
    account's in-flight attempt count `M(a)`, and the atomic
    `place_session` pick+pin-insert critical section — synchronous
    end-to-end (no `await`), so nothing can interleave between picking an
    account and inserting its pin. Durable persistence (the
    `pending_durability` barrier and the coalesced `last_seen` refresh)
    goes through an optional `ClaudePoolRuntimeStateStore`. It also owns the
    migration machinery: per-session-generation reservations, their
    `asyncio.shield`-based waiter protocol,
    migration-attempt tokens keyed by attempt id (the `M(a)` in-flight term),
    the generation/owner CAS performed at upstream 2xx headers
    (`commit_at_headers`), and the account-removal transition matrix.
    Reservations and tokens are NEVER persisted and always start empty.
    """

    def __init__(
        self,
        *,
        balanced_epoch_id: str,
        store: ClaudePoolRuntimeStateStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        pin_ttl_uuid_seconds: float = DEFAULT_PIN_TTL_UUID_SECONDS,
        pin_ttl_content_hash_seconds: float = DEFAULT_PIN_TTL_CONTENT_HASH_SECONDS,
        pin_map_max_entries: int = DEFAULT_PIN_MAP_MAX_ENTRIES,
    ) -> None:
        self.balanced_epoch_id = balanced_epoch_id
        self.observations = ObservationView()
        self.persistence_degraded = False

        self._store = store
        self._clock = clock
        self._wall_clock = wall_clock
        self._pin_ttl_uuid_seconds = pin_ttl_uuid_seconds
        self._pin_ttl_content_hash_seconds = pin_ttl_content_hash_seconds
        self._pin_map_max_entries = pin_map_max_entries

        self._pins: dict[bytes, PinEntry] = {}
        self._in_flight: dict[str, int] = {}
        self._restored_any_valid_pin = False

        # Every removal path (expiry, LRU eviction, explicit removal,
        # restore) goes through `_remove_pin`, so each pin is decremented
        # exactly once no matter which path removed it.
        self.removed_pin_counts: dict[str, int] = {}
        self.total_removed_pins = 0
        # Incremented whenever the bound is at capacity and every remaining
        # candidate is migration-reserved (so eviction cannot reclaim
        # space) — the "soft bound" being exceeded.
        self.soft_bound_overflow_count = 0

        # Reservations and migration tokens are process-local and never
        # persisted, so both start empty and have no restore path.
        self._reservations: dict[bytes, MigrationReservation] = {}
        self._migration_tokens: dict[str, str] = {}
        self._removed_accounts: set[str] = set()
        self.migration_outcome_counts: dict[str, int] = {}
        self.migration_cas_lost = 0
        self.migration_commit_rejected_target_removed = 0

        # Durable cooldowns are account-wide by default and Fable-family
        # scoped only when `classify_cooldown_scope` says so.
        self._account_cooldowns: dict[str, _CooldownEntry] = {}
        self._family_cooldowns: dict[tuple[str, str], _CooldownEntry] = {}
        # Capability evidence is keyed by the exact
        # `(account_id, capability_key)` pair and never inferred across keys.
        self._capability_evidence: dict[tuple[str, str], _CapabilityEvidenceEntry] = {}
        # `remove_account`'s durable incarnation-scoped deletion, keyed by
        # incarnation so a caller can await its completion afterward.
        self._pending_incarnation_removals: dict[str, Any] = {}

    # -- in-flight attempt counting: M(a) -----------------------------------

    def begin_attempt(self, account_id: str) -> None:
        self._in_flight[account_id] = self._in_flight.get(account_id, 0) + 1

    def end_attempt(self, account_id: str) -> None:
        current = self._in_flight.get(account_id, 0)
        self._in_flight[account_id] = max(0, current - 1)

    def in_flight_count(self, account_id: str) -> int:
        return self._in_flight.get(account_id, 0)

    # -- observation ingestion -----------------------------------------------

    def ingest_usage_peek(
        self, account_id: str, peeked: tuple[dict[str, Any], dict[str, dict[str, Any]]] | None
    ) -> None:
        """Adapt cached usage metadata into the router's `ObservationView`."""
        if peeked is None:
            return
        envelope, metadata = peeked
        now = self._clock()
        wall_now = self._wall_clock()
        for peek_window, binding_window in _PEEK_WINDOW_TO_BINDING.items():
            window_meta = metadata.get(peek_window)
            window_envelope = envelope.get(peek_window)
            if not isinstance(window_meta, dict) or not isinstance(window_envelope, dict):
                continue
            used_percent = window_envelope.get("used_percent")
            if not isinstance(used_percent, (int, float)):
                continue
            raw_age = window_meta.get("age_seconds")
            age_seconds = float(raw_age) if isinstance(raw_age, (int, float)) else 0.0
            raw_reset_at = window_meta.get("reset_at")
            reset_at = (
                _wall_to_monotonic(float(raw_reset_at), wall_now=wall_now, monotonic_now=now)
                if isinstance(raw_reset_at, (int, float))
                else None
            )
            self.observations.ingest_window(
                account_id,
                binding_window,
                used_percent=float(used_percent),
                source=str(window_meta.get("source") or "usage_api"),
                observed_at=now - age_seconds,
                reset_at=reset_at,
            )

    def ingest_observation(
        self,
        account_id: str,
        window: str,
        *,
        used_percent: float,
        source: str,
        age_seconds: float = 0.0,
        reset_in_seconds: float | None = None,
        reset_identity: str | None = None,
    ) -> None:
        """Directly ingest one source-tagged window observation."""
        now = self._clock()
        reset_at = now + reset_in_seconds if reset_in_seconds is not None else None
        self.observations.ingest_window(
            account_id,
            window,
            used_percent=used_percent,
            source=source,
            observed_at=now - age_seconds,
            reset_at=reset_at,
            reset_identity=reset_identity,
        )

    def ingest_allowed_warning(self, account_id: str, window: str, *, reset_identity: str | None) -> None:
        self.observations.ingest_allowed_warning(
            account_id, window, observed_at=self._clock(), reset_identity=reset_identity
        )

    # -- pressure / weight computation over the current candidate set ------

    def account_pressure(self, account_id: str, family: str, *, now: float, floor: float) -> float:
        """`P(a)`: the max, across `a`'s binding windows, of each window's pressure or `floor` if UNKNOWN."""
        values = []
        for window in binding_windows(family):
            pressure_value = self.observations.window_pressure(account_id, window, now=now)
            values.append(pressure_value if pressure_value is not None else floor)
        return max(values)

    def candidate_set_unknown_floor(
        self, account_ids: Sequence[str], family: str, *, now: float
    ) -> float:
        """`unknown_floor` computed over every binding window of the CURRENT candidate set."""
        complete: list[float] = []
        for account_id in account_ids:
            for window in binding_windows(family):
                pressure_value = self.observations.window_pressure(account_id, window, now=now)
                if pressure_value is not None:
                    complete.append(pressure_value)
        return unknown_floor(complete)

    def _candidate_weights(
        self, eligible: Sequence[AccountCandidate], family: str, *, now: float
    ) -> dict[str, float]:
        account_ids = [candidate.account_id for candidate in eligible]
        floor = self.candidate_set_unknown_floor(account_ids, family, now=now)
        pressures = {
            account_id: self.account_pressure(account_id, family, now=now, floor=floor)
            for account_id in account_ids
        }
        windows = binding_windows(family)
        warning_factors = {
            account_id: warning_factor(self.observations, account_id, windows, now=now)
            for account_id in account_ids
        }
        in_flight = {account_id: self.in_flight_count(account_id) for account_id in account_ids}
        return select_weights(account_ids, pressures=pressures, warning_factors=warning_factors, in_flight=in_flight)

    # -- cold start ---------------------------------------------------------

    def _is_cold_start(
        self,
        eligible_by_id: dict[str, AccountCandidate],
        all_candidates_by_id: dict[str, AccountCandidate],
        family: str,
        serving_account_id: str | None,
        *,
        now: float,
    ) -> bool:
        """All windows UNKNOWN, no valid restored pins for the current epoch, zero live pins
        (already purged of expiry by the caller), zero active migrations, and the serving pin
        itself eligible -> the first session goes to the serving pin.
        """
        if serving_account_id is None:
            return False
        if self._pins:
            return False
        if self._restored_any_valid_pin:
            return False
        if self.active_migration_count() != 0:
            return False
        for account_id in all_candidates_by_id:
            for window in binding_windows(family):
                if self.observations.window_pressure(account_id, window, now=now) is not None:
                    return False
        return serving_account_id in eligible_by_id

    # -- pin map: reads, TTL/LRU/counters -------------------------------------

    def get_pin(self, digest: bytes) -> PinEntry | None:
        return self._pins.get(digest)

    def pin_count(self) -> int:
        return len(self._pins)

    def active_migration_count(self) -> int:
        return sum(1 for entry in self._pins.values() if entry.migration_reserved)

    def set_migration_reserved(self, digest: bytes, reserved: bool) -> None:
        entry = self._pins.get(digest)
        if entry is not None:
            entry.migration_reserved = reserved

    def _pin_ttl_seconds(self, key_kind: str) -> float:
        return self._pin_ttl_uuid_seconds if key_kind == "uuid" else self._pin_ttl_content_hash_seconds

    def purge_expired_pins(self, *, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        expired = [digest for digest, entry in self._pins.items() if entry.expires_at_monotonic <= now]
        for digest in expired:
            self._remove_pin(digest, reason="expired")
        return len(expired)

    def remove_pin(self, digest: bytes) -> bool:
        """Remove an externally invalidated pin, such as for account removal."""
        return self._remove_pin(digest, reason="removed") is not None

    def _remove_pin(self, digest: bytes, *, reason: str) -> PinEntry | None:
        entry = self._pins.pop(digest, None)
        if entry is None:
            return None
        self.removed_pin_counts[reason] = self.removed_pin_counts.get(reason, 0) + 1
        self.total_removed_pins += 1
        return entry

    def _lru_victim(self, *, key_kind: str) -> bytes | None:
        candidates = [
            (entry.last_seen_monotonic, digest)
            for digest, entry in self._pins.items()
            if entry.key_kind == key_kind and not entry.migration_reserved
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    def _ensure_capacity(self, *, now: float) -> None:
        """Bound 10,000, eviction order expired -> LRU content-hash -> LRU uuid.

        Entries holding an active migration reservation are never evicted
        (a soft bound): if every remaining entry is reserved, the bound is
        exceeded and `soft_bound_overflow_count` records it instead.
        """
        if len(self._pins) < self._pin_map_max_entries:
            return
        self.purge_expired_pins(now=now)
        if len(self._pins) < self._pin_map_max_entries:
            return
        victim = self._lru_victim(key_kind="content_hash") or self._lru_victim(key_kind="uuid")
        if victim is None:
            self.soft_bound_overflow_count += 1
            return
        self._remove_pin(victim, reason="evicted_lru")

    def touch_pin(
        self,
        digest: bytes,
        *,
        is_message_request: bool,
        key_is_live: bool,
        account_still_registered: bool,
        now: float | None = None,
    ) -> bool:
        """The `last_seen` refresh predicate: a real /v1/messages request resolving a live key
        with current registry membership, applied before the upstream outcome is known.
        """
        entry = self._pins.get(digest)
        if entry is None:
            return False
        if not (is_message_request and key_is_live and account_still_registered):
            return False
        now = self._clock() if now is None else now
        entry.last_seen_monotonic = now
        entry.expires_at_monotonic = now + self._pin_ttl_seconds(entry.key_kind)
        return True

    # -- atomic pick and pin insertion --------------------------------------

    def place_session(
        self,
        *,
        session_key: SessionKey,
        model: str,
        candidates: Sequence[AccountCandidate],
        seed: bytes,
        serving_account_id: str | None = None,
        already_attempted: frozenset[str] = frozenset(),
        now: float | None = None,
    ) -> PlacementResult:
        """The atomic pick+pin-insert critical section: entirely synchronous (no `await`),
        so no other coroutine can interleave between the pick and the pin-map insert.
        """
        family = quota_family(model)
        if session_key.family and session_key.family != family:
            raise ValueError(
                f"session key family {session_key.family!r} does not match the model's quota family {family!r}"
            )

        now = self._clock() if now is None else now
        self.purge_expired_pins(now=now)

        existing = self._pins.get(session_key.digest)
        if existing is not None:
            return PlacementResult(
                account_id=existing.account_id,
                session_key_digest=session_key.digest,
                key_kind=existing.key_kind,
                generation=existing.generation,
                created=False,
                durability_barrier=existing.pending_durability,
            )

        all_by_id = {candidate.account_id: candidate for candidate in candidates}
        eligible = [
            candidate
            for candidate in candidates
            if is_eligible_candidate(candidate, now=now, already_attempted=already_attempted)
        ]
        eligible_by_id = {candidate.account_id: candidate for candidate in eligible}
        if not eligible:
            raise NoEligibleAccountError("no eligible account is available to place this session")

        if self._is_cold_start(eligible_by_id, all_by_id, family, serving_account_id, now=now):
            chosen_id = serving_account_id
            assert chosen_id is not None
        else:
            weights = self._candidate_weights(eligible, family, now=now)
            # Score with the LOGICAL digest (correlated HRW): equal family
            # weights co-locate one session's family pins on one account.
            chosen_id = pick_weighted_hrw(
                weights=weights,
                seed=seed,
                session_key_digest=session_key.scoring_digest_or_default,
                serving_account_id=serving_account_id,
            )

        chosen = eligible_by_id[chosen_id]
        self._ensure_capacity(now=now)
        barrier = PendingDurabilityBarrier()
        entry = PinEntry(
            session_key_digest=session_key.digest,
            key_kind=session_key.kind,
            account_id=chosen.account_id,
            account_incarnation_id=chosen.account_incarnation_id,
            generation=0,
            last_seen_monotonic=now,
            expires_at_monotonic=now + self._pin_ttl_seconds(session_key.kind),
            pending_durability=barrier,
            model_family=session_key.family,
        )
        self._pins[session_key.digest] = entry
        return PlacementResult(
            account_id=chosen.account_id,
            session_key_digest=session_key.digest,
            key_kind=session_key.kind,
            generation=0,
            created=True,
            durability_barrier=barrier,
        )

    # -- durable persistence ------------------------------------------------

    async def submit_new_pin_durability(self, digest: bytes) -> None:
        """Submit the initial pin's HIGH-PRIORITY durable write and resolve its barrier.

        Meant to run as a task independent of any one request's own
        cancellation (spawned by whoever created the pin right after
        `place_session` returns, never awaited inline inside its no-await
        critical section). On store failure, `persistence_degraded` is set
        and the in-memory pin is retained; the barrier still resolves
        exactly once either way, releasing every blocked request.
        """
        entry = self._pins.get(digest)
        if entry is None or entry.pending_durability is None:
            return
        barrier = entry.pending_durability
        if self._store is None:
            barrier.resolve()
            return
        try:
            pending_write = self._store.upsert_pin(
                session_key_digest=digest,
                key_kind=entry.key_kind,
                account_id=entry.account_id,
                account_incarnation_id=entry.account_incarnation_id,
                last_seen_utc=self._wall_clock(),
                expires_at_utc=self._wall_clock() + (entry.expires_at_monotonic - self._clock()),
                generation=entry.generation,
                balanced_epoch_id=self.balanced_epoch_id,
                model_family=entry.model_family,
                high_priority=True,
            )
            await pending_write.wait_async()
        except Exception:
            self.persistence_degraded = True
        finally:
            barrier.resolve()

    async def await_pin_durability(self, digest: bytes) -> None:
        """Every request resolving this pin generation awaits its barrier before an upstream attempt."""
        entry = self._pins.get(digest)
        if entry is None or entry.pending_durability is None:
            return
        await entry.pending_durability.wait()

    async def refresh_pin_durable_last_seen(self, digest: bytes) -> None:
        """Refresh durable `last_seen`, coalesced to at most once per minute per pin."""
        entry = self._pins.get(digest)
        if entry is None or self._store is None:
            return
        pending_write = self._store.touch_pin_last_seen(digest, self._wall_clock())
        if pending_write is None:
            return
        try:
            await pending_write.wait_async()
        except Exception:
            self.persistence_degraded = True

    # -- restore from the runtime store -------------------------------------

    def restore_from_store(
        self, restore_result: RestoreResult, *, now: float | None = None, wall_now: float | None = None
    ) -> int:
        """Convert restored rows (wall-clock) into monotonic in-memory state,
        recomputing pin counters.

        `restore_result.pins` already excludes expired and epoch-mismatched
        rows (`ClaudePoolRuntimeStateStore.restore`); every remaining row is
        durable, so it carries no `pending_durability` barrier. Cooldowns and
        capability evidence get the same wall-to-monotonic conversion, each
        with its own restore clamp and validation. A cooldown's remaining
        duration clamps to `[1s, 7d]`, and a restored capability row is kept
        only while it is still `eligible`, still carries a TTL, is not yet
        expired, and matches this build's `CAPABILITY_CLASSIFIER_VERSION` (the
        incarnation/fingerprint match is enforced lazily, at query time,
        against the live candidate the caller supplies — the same backstop
        pattern pins rely on).
        """
        monotonic_now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now

        self._pins.clear()
        self.removed_pin_counts.clear()
        self.total_removed_pins = 0
        self.soft_bound_overflow_count = 0
        self._account_cooldowns.clear()
        self._family_cooldowns.clear()
        self._capability_evidence.clear()

        for digest, row in restore_result.pins.items():
            self._pins[digest] = PinEntry(
                session_key_digest=digest,
                key_kind=row.key_kind,
                account_id=row.account_id,
                account_incarnation_id=row.account_incarnation_id,
                generation=row.generation,
                last_seen_monotonic=_wall_to_monotonic(
                    row.last_seen_utc, wall_now=wall_now, monotonic_now=monotonic_now
                ),
                expires_at_monotonic=_wall_to_monotonic(
                    row.expires_at_utc, wall_now=wall_now, monotonic_now=monotonic_now
                ),
                pending_durability=None,
                model_family=row.model_family,
            )
        self._restored_any_valid_pin = bool(restore_result.pins)

        for (account_id, scope, model_family), cooldown_row in restore_result.cooldowns.items():
            remaining = cooldown_row.deadline_utc - wall_now
            clamped_remaining = max(
                _COOLDOWN_RESTORE_MIN_SECONDS, min(_COOLDOWN_RESTORE_MAX_SECONDS, remaining)
            )
            entry = _CooldownEntry(
                deadline_monotonic=monotonic_now + clamped_remaining,
                deadline_wall=wall_now + clamped_remaining,
                account_incarnation_id=cooldown_row.account_incarnation_id,
                reason=cooldown_row.reason,
            )
            if scope == "account":
                self._account_cooldowns[account_id] = entry
            else:
                self._family_cooldowns[(account_id, model_family)] = entry

        for (account_id, capability_key_value), capability_row in restore_result.capability_evidence.items():
            if capability_row.state != "eligible" or capability_row.expires_at_utc is None:
                continue  # Restored denials are ignored; eligible evidence requires a TTL.
            if capability_row.classifier_version != CAPABILITY_CLASSIFIER_VERSION:
                continue
            expires_at_monotonic = _wall_to_monotonic(
                capability_row.expires_at_utc, wall_now=wall_now, monotonic_now=monotonic_now
            )
            if expires_at_monotonic <= monotonic_now:
                continue
            self._capability_evidence[(account_id, capability_key_value)] = _CapabilityEvidenceEntry(
                account_incarnation_id=capability_row.account_incarnation_id,
                account_profile_fingerprint=capability_row.account_profile_fingerprint,
                classifier_version=capability_row.classifier_version,
                expires_at_monotonic=expires_at_monotonic,
            )

        return len(self._pins)

    # -- account-wide and Fable family-scoped cooldowns ---------------------

    def classify_cooldown_scope(
        self, *, account_id: str, model: str, upstream_status_code: int, now: float | None = None
    ) -> FamilyGateOutcome:
        """`classify_balanced_cooldown_scope` against this router's own live
        `observations` and clock."""
        now = self._clock() if now is None else now
        return classify_balanced_cooldown_scope(
            self.observations,
            account_id=account_id,
            model=model,
            upstream_status_code=upstream_status_code,
            now=now,
        )

    def install_cooldown(
        self,
        *,
        account_id: str,
        account_incarnation_id: str,
        account_profile_fingerprint: str | None,
        scope: Literal["account", "family"],
        deadline: float,
        reason: str,
        model_family: str = "",
        evidence: str = "",
        now: float | None = None,
        wall_now: float | None = None,
    ) -> Any:
        """Install a cooldown (account-wide, or Fable family-scoped when
        `scope == "family"`), in-memory and — when both a store and a profile
        fingerprint are available — durably and at high priority. `deadline`
        is monotonic; the entry's wall deadline is derived from it (see
        `_CooldownEntry`).
        Returns the store's `PendingWrite`, or `None` when nothing was
        submitted (no store, or no fingerprint yet).
        """
        now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now
        entry = _CooldownEntry(
            deadline_monotonic=deadline,
            deadline_wall=wall_now + (deadline - now),
            account_incarnation_id=account_incarnation_id,
            reason=reason,
        )
        if scope == "account":
            self._account_cooldowns[account_id] = entry
        else:
            self._family_cooldowns[(account_id, model_family)] = entry

        if self._store is None or account_profile_fingerprint is None:
            return None
        return self._store.upsert_cooldown(
            account_id=account_id,
            scope=scope,
            model_family=model_family,
            account_incarnation_id=account_incarnation_id,
            account_profile_fingerprint=account_profile_fingerprint,
            deadline_utc=entry.deadline_wall,
            reason=reason,
            evidence=evidence,
            updated_at_utc=wall_now,
            high_priority=True,
        )

    def account_cooldown_deadline(
        self, account_id: str, *, now: float | None = None, wall_now: float | None = None
    ) -> float | None:
        """The account-wide cooldown's monotonic deadline, or `None` when absent
        or expired on either clock. The returned value reflects the earlier of
        the entry's two deadlines, expressed on the monotonic clock."""
        now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now
        return _entry_cooldown_deadline(self._account_cooldowns.get(account_id), now, wall_now)

    def family_cooldown_deadline(
        self, account_id: str, model_family: str, *, now: float | None = None, wall_now: float | None = None
    ) -> float | None:
        """The `(account_id, model_family)` cooldown's monotonic deadline, or `None`
        when absent or expired on either clock. The returned value reflects the
        earlier of the entry's two deadlines, expressed on the monotonic clock."""
        now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now
        return _entry_cooldown_deadline(self._family_cooldowns.get((account_id, model_family)), now, wall_now)

    # -- eligible-only capability evidence with TTLs -----------------------

    def classify_capability_evidence(
        self,
        *,
        account_id: str,
        capability_key: str,
        account_incarnation_id: str,
        account_profile_fingerprint: str,
        status_code: int,
        evidence_source: str,
        now: float | None = None,
        wall_now: float | None = None,
    ) -> None:
        """Record `eligible` evidence only for an explicit successful 2xx.

        Evidence applies to the exact `capability_key`. Every other status
        records nothing, and this classifier never writes `denied` evidence,
        so the model-ineligible migration trigger remains dormant.
        """
        if status_code // 100 != 2:
            return
        now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now
        expires_at_monotonic = now + CAPABILITY_EVIDENCE_TTL_SECONDS
        self._capability_evidence[(account_id, capability_key)] = _CapabilityEvidenceEntry(
            account_incarnation_id=account_incarnation_id,
            account_profile_fingerprint=account_profile_fingerprint,
            classifier_version=CAPABILITY_CLASSIFIER_VERSION,
            expires_at_monotonic=expires_at_monotonic,
        )
        if self._store is not None:
            self._store.upsert_capability_evidence(
                account_id=account_id,
                capability_key=capability_key,
                account_incarnation_id=account_incarnation_id,
                account_profile_fingerprint=account_profile_fingerprint,
                state="eligible",
                evidence_source=evidence_source,
                classifier_version=CAPABILITY_CLASSIFIER_VERSION,
                observed_at_utc=wall_now,
                expires_at_utc=wall_now + CAPABILITY_EVIDENCE_TTL_SECONDS,
                high_priority=False,
            )

    def is_capability_eligible(
        self,
        account_id: str,
        capability_key: str,
        *,
        account_incarnation_id: str,
        account_profile_fingerprint: str,
        now: float | None = None,
    ) -> bool:
        """True only for a fresh, EXACT-key match whose incarnation, profile
        fingerprint, and classifier version all still match what was recorded
        — never inferred from a different capability key, never trusted past
        its TTL or across a reauth/plan change.
        """
        entry = self._capability_evidence.get((account_id, capability_key))
        if entry is None:
            return False
        now = self._clock() if now is None else now
        if entry.expires_at_monotonic <= now:
            return False
        if entry.account_incarnation_id != account_incarnation_id:
            return False
        if entry.account_profile_fingerprint != account_profile_fingerprint:
            return False
        if entry.classifier_version != CAPABILITY_CLASSIFIER_VERSION:
            return False
        return True

    # -- migration reservations and waiters --------------------------------

    def get_migration_reservation(self, digest: bytes) -> MigrationReservation | None:
        """Read the current reservation without awaiting.

        A waiter's loop starts with this critical-section read. It returns
        `None` when no migration is in flight for the session.
        """
        return self._reservations.get(digest)

    def acquire_migration_reservation(
        self,
        digest: bytes,
        *,
        source_account: str,
        source_generation: int,
        target_account: str,
        attempt_id: str,
    ) -> tuple[MigrationReservation, bool]:
        """Atomically acquire a migration reservation and attempt token.

        This critical section is entirely synchronous, so no coroutine can
        interleave between the check and insert. If a pending reservation
        exists, return it with `False`; this caller waits and re-reads routing
        state rather than selecting an independent target. Otherwise create a
        reservation and attempt token in the same critical section, protect
        the pin from eviction, and return the reservation with `True`.
        """
        existing = self._reservations.get(digest)
        if existing is not None and existing.outcome == "pending":
            return existing, False
        reservation = MigrationReservation(
            source_account=source_account,
            source_generation=source_generation,
            target_account=target_account,
            owner_attempt_id=attempt_id,
        )
        self._reservations[digest] = reservation
        self._migration_tokens[attempt_id] = target_account
        self.begin_attempt(target_account)
        self.set_migration_reserved(digest, True)
        return reservation, True

    async def wait_for_migration_reservation(self, reservation: MigrationReservation) -> None:
        """Wait for a migration reservation without exposing its shared event.

        `asyncio.shield` prevents a waiter's cancellation from cancelling or
        double-firing the shared resolution. Afterward the caller re-enters the
        critical section and re-reads routing state instead of trusting the
        reservation's potentially stale target.
        """
        await asyncio.shield(reservation.resolved_event.wait())

    def resolve_migration_reservation(
        self, digest: bytes, *, attempt_id: str, outcome: MigrationOutcome
    ) -> bool:
        """Resolve an owner-terminal path idempotently without awaiting.

        Verify `owner_attempt_id`, record the immutable outcome, clear the
        reservation, un-protect the pin from eviction, then set its event exactly once.
        Returns `True` only when THIS call performed the transition; a reservation that
        is missing, already resolved, or owned by a different attempt is a no-op that
        returns `False` — safe to call more than once, from more than one terminal path.
        """
        reservation = self._reservations.get(digest)
        if (
            reservation is None
            or reservation.outcome != "pending"
            or reservation.owner_attempt_id != attempt_id
        ):
            return False
        reservation.outcome = outcome
        self.migration_outcome_counts[outcome] = self.migration_outcome_counts.get(outcome, 0) + 1
        del self._reservations[digest]
        self.set_migration_reserved(digest, False)
        reservation.resolved_event.set()
        return True

    # -- migration-attempt tokens and in-flight counts ---------------------

    def migration_token_target(self, attempt_id: str) -> str | None:
        """The live token's target account for `attempt_id`, or `None` if released/absent."""
        return self._migration_tokens.get(attempt_id)

    def release_migration_token(self, attempt_id: str) -> str | None:
        """Release a migration token idempotently.

        The first call decrements the target's `M(a)`; every later
        call for the same `attempt_id` is a no-op that returns `None`.
        """
        target_account = self._migration_tokens.pop(attempt_id, None)
        if target_account is not None:
            self.end_attempt(target_account)
        return target_account

    def resolve_migration_owner_terminal(
        self, digest: bytes, *, attempt_id: str, outcome: MigrationOutcome
    ) -> bool:
        """Resolve every migration-owner terminal path idempotently.

        Every owner exit — cancellation, timeout, mode drain, an exception
        before response headers, target rejection, CAS loss, or success —
        funnels through this synchronous method. It is safe to call from a
        `finally` block because no cancellation can interrupt the transition
        between ownership verification and event signaling. It resolves the
        owned reservation to `outcome` (a no-op if it is not pending, not owned by
        `attempt_id`, or was already resolved by an earlier explicit terminal path such
        as `commit_at_headers`), then releases the attempt's migration token exactly
        once (a no-op if already released). Returns whether THIS call resolved the
        reservation.
        """
        resolved_now = self.resolve_migration_reservation(digest, attempt_id=attempt_id, outcome=outcome)
        self.release_migration_token(attempt_id)
        return resolved_now

    def resolve_migration_preheader_failure(
        self,
        digest: bytes,
        *,
        attempt_id: str,
        outcome: MigrationOutcome = "retryable_preheader_failure",
    ) -> bool:
        """Resolve a pre-header quota or eligibility failure in safe order.

        Before calling this method, the caller must classify the response,
        update in-memory cooldown or capability evidence, and fence the target.
        This ensures woken waiters re-read updated state. The method then
        resolves the reservation, wakes waiters, and releases the migration
        token synchronously. Persistence and diagnostics are scheduled only
        after it returns.

        If `remove_account` separately marked the source account removed, this
        also deletes the orphaned source pin so later requests place afresh.
        """
        reservation = self._reservations.get(digest)
        source_account = reservation.source_account if reservation is not None else None
        resolved_now = self.resolve_migration_owner_terminal(digest, attempt_id=attempt_id, outcome=outcome)
        if resolved_now and source_account is not None and source_account in self._removed_accounts:
            self.remove_pin(digest)
        return resolved_now

    # -- commit point at upstream 2xx headers -------------------------------

    def commit_at_headers(
        self,
        digest: bytes,
        *,
        attempt_id: str,
        source_account: str,
        source_generation: int,
        target_account: str,
        target_account_incarnation_id: str,
        target_still_registered: bool = True,
        ingest_headers: Callable[[], None] | None = None,
    ) -> tuple[CommitOutcome, PinEntry | None, PendingDurabilityBarrier | None]:
        """Commit a migration at upstream 2xx headers without awaiting.

        Reservation resolution happens here and is never deferred to stream
        completion:

        1. Run `ingest_headers()`, when supplied, before anything else reads
           pressure or eligibility state.
        2. Validate target membership. If the target was removed before its 2xx
           headers arrived, resolve the reservation as `target_removed`, wake
           waiters, and leave the pin untouched.
        3. the generation/owner CAS: `pin.account_id == source_account`,
           `pin.generation == source_generation`,
           `reservation.owner_attempt_id == attempt_id` — on failure: outcome
           `cas_lost` (the newer pin is never overwritten, `migration_cas_lost`
           increments), the reservation resolves and wakes waiters;
        4. on success: the pin flips to `target_account`, `generation += 1`, a fresh
           `pending_durability` barrier is attached to the new generation, the
           reservation is cleared, and its event is set exactly once — outcome
           `committed`.

        The migration-attempt token remains live until the caller releases it
        through `resolve_migration_owner_terminal` or `release_migration_token`
        from the attempt's common `finally` after the upstream body or stream
        terminates. `M(target_account)` stays incremented for the whole streamed
        response, not just until these headers commit.

        Returns `(outcome, pin, barrier)`. On `committed`, the caller awaits the
        barrier's high-priority durability completion through
        `submit_new_pin_durability` and `await_pin_durability` outside this
        critical section before forwarding any downstream byte. Same-session
        waiters re-read the pin and await
        that same barrier before sending upstream.
        """
        if ingest_headers is not None:
            ingest_headers()

        if not target_still_registered:
            self.migration_commit_rejected_target_removed += 1
            self.resolve_migration_reservation(digest, attempt_id=attempt_id, outcome="target_removed")
            return "target_removed", self._pins.get(digest), None

        reservation = self._reservations.get(digest)
        pin = self._pins.get(digest)
        cas_ok = (
            pin is not None
            and pin.account_id == source_account
            and pin.generation == source_generation
            and reservation is not None
            and reservation.owner_attempt_id == attempt_id
        )
        if not cas_ok:
            # `migration_cas_lost` counts THIS attempt's own CAS rejection, independent
            # of whether a still-pending reservation is left to formally resolve to
            # `cas_lost` (a sufficiently stale attempt may find its reservation already
            # cleared under a different terminal outcome).
            self.migration_cas_lost += 1
            self.resolve_migration_reservation(digest, attempt_id=attempt_id, outcome="cas_lost")
            return "cas_lost", pin, None

        assert pin is not None
        barrier = PendingDurabilityBarrier()
        pin.account_id = target_account
        pin.account_incarnation_id = target_account_incarnation_id
        pin.generation += 1
        pin.pending_durability = barrier
        self.resolve_migration_reservation(digest, attempt_id=attempt_id, outcome="committed")
        return "committed", pin, barrier

    # -- account removal transition matrix ---------------------------------

    def remove_account(self, account_id: str, incarnation: str) -> dict[str, int]:
        """Apply the account-removal transition matrix.

        Target removal after upstream headers is handled at commit time by
        `commit_at_headers`'s `target_still_registered` check because it
        depends on whether upstream 2xx already arrived. This operation never
        bulk-resets migration tokens or their in-flight counters. Each owned
        token is released exactly once by its attempt's terminal cleanup, and
        lazy membership checks remain the correctness backstop. The returned
        counts describe each removal outcome.

        The operation also drops in-memory cooldown and capability-evidence
        entries for `account_id`, then submits a high-priority durable deletion
        of every pin, cooldown, usage observation, and capability-evidence row
        belonging to `incarnation`. Incarnation scoping ensures a later account
        that reuses `account_id` is never touched. Await
        `await_account_removal_durability(incarnation)` for that deletion to
        complete.
        """
        self._removed_accounts.add(account_id)
        case_counts = {
            "ordinary_pin_removed": 0,
            "source_removed_migration_continues": 0,
            "target_removed_before_headers": 0,
            "both_removed_aborted": 0,
        }

        touching_digests = {
            digest for digest, entry in self._pins.items() if entry.account_id == account_id
        } | {
            digest
            for digest, reservation in self._reservations.items()
            if account_id in (reservation.source_account, reservation.target_account)
        }

        for digest in touching_digests:
            reservation = self._reservations.get(digest)
            if reservation is None:
                # Case 1: an ordinary pinned account, no active migration.
                if self.remove_pin(digest):
                    case_counts["ordinary_pin_removed"] += 1
                continue

            source_removed = reservation.source_account in self._removed_accounts
            target_removed = reservation.target_account in self._removed_accounts
            if source_removed and target_removed:
                # Case 5: both source and target removed through a race - abort outright.
                self.resolve_migration_reservation(
                    digest, attempt_id=reservation.owner_attempt_id, outcome="target_removed"
                )
                self.remove_pin(digest)
                case_counts["both_removed_aborted"] += 1
            elif target_removed:
                # Case 3: target removed before upstream 2xx - resolve, wake, prohibit
                # commit; the (source) pin is left intact for later requests/backstop.
                self.resolve_migration_reservation(
                    digest, attempt_id=reservation.owner_attempt_id, outcome="target_removed"
                )
                case_counts["target_removed_before_headers"] += 1
            elif source_removed:
                # Case 2: source of an active migration, target still eligible - leave
                # the pin and reservation untouched; the owner completes the migration
                # (the CAS never requires the source to still be registered).
                case_counts["source_removed_migration_continues"] += 1

        self._account_cooldowns.pop(account_id, None)
        for key in [key for key in self._family_cooldowns if key[0] == account_id]:
            del self._family_cooldowns[key]
        for key in [key for key in self._capability_evidence if key[0] == account_id]:
            del self._capability_evidence[key]

        if self._store is not None:
            self._pending_incarnation_removals[incarnation] = self._store.delete_all_for_incarnation(
                incarnation
            )

        return case_counts

    async def await_account_removal_durability(self, incarnation: str) -> None:
        """Await the durable incarnation-scoped deletion `remove_account` submitted
        for `incarnation`, if any — so a caller (e.g. the admin removal handler)
        can be sure the durable rows are gone before it responds. A no-op when
        no such deletion is pending (no store, or already awaited).
        """
        pending_write = self._pending_incarnation_removals.pop(incarnation, None)
        if pending_write is None:
            return
        try:
            await pending_write.wait_async()
        except Exception:
            self.persistence_degraded = True
