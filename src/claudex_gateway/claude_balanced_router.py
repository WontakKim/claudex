"""Domain-separated session-key derivation for balanced (session-affinity) routing.

Design v2 §2.4/§5.1 (adjudications Q3, H): session keys are HMAC-SHA256
digests under the durable per-deployment epoch seed. Raw Claude Code UUIDs
and message content are never stored — only the digest. Two branches decide
what gets hashed, tried in order:

* uuid — Claude Code's `metadata.user_id` is a JSON string embedding a
  `session_id` (the same field `server._rewrite_metadata_account_uuid`
  parses for `account_uuid` rewriting). A candidate is accepted only when it
  round-trips through `uuid.UUID` with no leading/trailing whitespace and
  carries the RFC 4122 variant; the canonical `str(uuid.UUID(...))` form is
  hashed, so equivalent-but-differently-cased input yields the same digest.
* content_hash — used whenever the uuid branch is unavailable (missing
  metadata, non-string/malformed/non-RFC-4122 `session_id`, ...): the
  complete first `messages` element whose `role` is exactly `"user"` (the
  client's own object, before any gateway mutation) is canonicalized to JSON
  and hashed instead.

A request with neither a usable `session_id` nor a first user message has no
session key at all — the caller must treat it as unpinnable and route it
statelessly via `derive_stateless_routing_digest`.

`hrw_unit_interval` (§2.4) is the separate rendezvous-hashing (Highest
Random Weight) sample: a per-(session, account) digest reduced to a
uniformly distributed point in the open interval (0, 1), so the account with
the largest sample can be picked without keeping any pin-map state.

The rest of this module (design v2 §2, §5.1-§5.3) is the balanced picker
core built on top of the above: `ObservationView` turns per-window usage
readings into freshness-ladder-adjusted pressures, `quota_family` and the
positive-set / amended-emergency weight formulas turn pressures into a
weighted-HRW draw, and `ClaudeBalancedRouter.place_session` performs the
atomic pick-and-pin critical section against an in-memory, TTL/LRU-bounded
pin map with a cancellation-safe `pending_durability` barrier for the
initial durable write.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal

from claudex_gateway.claude_pool_runtime_state import ClaudePoolRuntimeStateStore, RestoreResult

_SESSION_KEY_DOMAIN = b"claudex-session-key-v1"
_HRW_DOMAIN = b"claudex-balanced-hrw-v1"
_STATELESS_REQUEST_DOMAIN = b"claudex-stateless-request-v1"


@dataclass(frozen=True)
class SessionKey:
    """A domain-separated session-affinity digest and the branch that produced it."""

    digest: bytes
    kind: Literal["uuid", "content_hash"]


def _hmac_sha256(seed: bytes, message: bytes) -> bytes:
    return hmac.new(seed, message, sha256).digest()


def _length_prefixed(*fields: bytes) -> bytes:
    """Concatenate `fields`, each preceded by its 8-byte big-endian length.

    Length-prefixing keeps concatenated variable-length fields unambiguous:
    without it, hashing `("ab", "c")` and `("a", "bc")` would collide.
    """
    framed = bytearray()
    for field in fields:
        framed += len(field).to_bytes(8, "big")
        framed += field
    return bytes(framed)


def _uuid_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Try the uuid branch; None on anything that isn't a clean RFC 4122 session_id."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return None
    user_id = metadata.get("user_id")
    if not isinstance(user_id, str):
        return None
    try:
        parsed_user_id = json.loads(user_id)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_user_id, dict):
        return None
    candidate = parsed_user_id.get("session_id")
    if not isinstance(candidate, str) or candidate != candidate.strip():
        return None
    try:
        parsed = uuid.UUID(candidate)
    except ValueError:
        return None
    if parsed.variant != uuid.RFC_4122:
        return None
    canonical_utf8 = str(parsed).encode("utf-8")
    digest = _hmac_sha256(
        seed, _SESSION_KEY_DOMAIN + b"\x00uuid\x00" + _length_prefixed(canonical_utf8)
    )
    return SessionKey(digest=digest, kind="uuid")


def _first_user_message(body: dict[str, Any]) -> dict[str, Any] | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return message
    return None


def _content_hash_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Try the content-hash branch; None when the body has no user message."""
    first_user_message = _first_user_message(body)
    if first_user_message is None:
        return None
    canonical_utf8 = json.dumps(
        first_user_message,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = _hmac_sha256(
        seed,
        _SESSION_KEY_DOMAIN + b"\x00content_hash\x00" + _length_prefixed(canonical_utf8),
    )
    return SessionKey(digest=digest, kind="content_hash")


def derive_session_key(body: dict[str, Any], seed: bytes) -> SessionKey | None:
    """Derive a session-affinity key for `body`, or None when it is unpinnable.

    Tries the uuid branch first (Claude Code's own `session_id`), then falls
    back to hashing the first user message. Returns None only when neither
    branch has anything to hash.
    """
    return _uuid_session_key(body, seed) or _content_hash_session_key(body, seed)


def hrw_unit_interval(seed: bytes, session_key_digest: bytes, account_id: str) -> float:
    """Map (seed, session_key_digest, account_id) onto the open interval (0, 1).

    Highest-Random-Weight sampling (design v2 §2.4) routes a session to the
    account with the largest sample; the mapping is deterministic and needs
    no stored state to keep a session pinned to the same account run after
    run. `mac`'s high 53 bits give the full double-precision mantissa worth
    of entropy, offset by 0.5 so the interval stays strictly open.
    """
    mac = _hmac_sha256(
        seed,
        _HRW_DOMAIN
        + b"\x00"
        + _length_prefixed(session_key_digest, account_id.encode("utf-8")),
    )
    high_53_bits = int.from_bytes(mac, "big") >> (8 * len(mac) - 53)
    return (high_53_bits + 0.5) / 2**53


def derive_stateless_routing_digest(seed: bytes, nonce: bytes) -> bytes:
    """HMAC digest identifying one stateless (unpinnable) request's retry chain.

    The caller supplies one fresh 32-byte random nonce per request; the
    resulting digest is reused across that request's retries, never
    persisted, and never creates a pin-map entry.
    """
    return _hmac_sha256(seed, _STATELESS_REQUEST_DOMAIN + b"\x00" + _length_prefixed(nonce))


# ==========================================================================
# Balanced picker core (design v2 §2, §5.1-§5.3)
# ==========================================================================

# -- freshness ladder / unknown_floor (design v2 §2) -----------------------

_FRESH_EXACT_SECONDS = 5 * 60
_FRESH_PLUS5_SECONDS = 15 * 60
_FRESH_PLUS10_SECONDS = 30 * 60
_FRESH_PLUS5_PP = 5.0
_FRESH_PLUS10_PP = 10.0

_UNKNOWN_FLOOR_MARGIN = 10.0
_UNKNOWN_FLOOR_CAP = 90.0

# The 0.5 haircut window: an `allowed_warning` signal only applies while it
# is at most this many seconds old.
_WARNING_FRESH_SECONDS = 5 * 60
_WARNING_HAIRCUT_FACTOR = 0.5

# The "2" in `H(a) = max(0, 100 - P(a) - 2*M(a))` and the emergency branch's
# `W = C0^2 / (C0 + 2M)`.
_IN_FLIGHT_PRESSURE_WEIGHT = 2.0

_NON_FABLE_WINDOWS: tuple[str, ...] = ("five_hour", "seven_day")
_FABLE_WINDOWS: tuple[str, ...] = ("five_hour", "seven_day", "fable_weekly")

# `ClaudeAccountUsageCache.peek_with_metadata` (T-6) window names -> this
# module's binding-window names (also `ClaudePoolRuntimeStateStore`'s
# `usage_observations.window` values).
_PEEK_WINDOW_TO_BINDING: dict[str, str] = {
    "session": "five_hour",
    "weekly": "seven_day",
    "fable_weekly": "fable_weekly",
}

# Pin map (design v2 §5.1)
DEFAULT_PIN_TTL_UUID_SECONDS = 5 * 3600
DEFAULT_PIN_TTL_CONTENT_HASH_SECONDS = 30 * 60
DEFAULT_PIN_MAP_MAX_ENTRIES = 10_000


def quota_family(model: str) -> str:
    """The binding-window family a `model` id belongs to (adjudication G).

    The ASCII case-insensitive token "fable", bounded on each side by the
    string's start/end or a non-alphanumeric (ASCII) character, selects the
    "fable" family — which adds the `fable_weekly` binding window on top of
    the default `[five_hour, seven_day]` pair. Every other model (including
    a token merely containing "fable" as a substring of a larger word, e.g.
    "unfable" or "fabled") uses the "default" family.
    """
    lowered = model.lower()
    token = "fable"
    start = 0
    while True:
        index = lowered.find(token, start)
        if index == -1:
            return "default"
        before = lowered[index - 1] if index > 0 else None
        after_index = index + len(token)
        after = lowered[after_index] if after_index < len(lowered) else None
        if not _is_ascii_alnum(before) and not _is_ascii_alnum(after):
            return "fable"
        start = index + 1


def _is_ascii_alnum(char: str | None) -> bool:
    return char is not None and char.isascii() and char.isalnum()


def binding_windows(family: str) -> tuple[str, ...]:
    """Design v2 §2: non-Fable accounts bind on `[five_hour, seven_day]`; Fable adds `fable_weekly`."""
    return _FABLE_WINDOWS if family == "fable" else _NON_FABLE_WINDOWS


def freshness_adjusted_pressure(
    used_percent: float, age_seconds: float, *, reset_passed: bool
) -> float | None:
    """Design v2 §2's freshness ladder, or `None` for UNKNOWN.

    <=5 min: the exact reading. 5-15 min: +5 percentage points. 15-30 min:
    +10 percentage points. Anything older than 30 minutes — or a window
    whose reset has already passed — is UNKNOWN (`None`); a missing reading
    is UNKNOWN by construction (the caller never invokes this at all).
    """
    if reset_passed:
        return None
    if age_seconds <= _FRESH_EXACT_SECONDS:
        return min(100.0, used_percent)
    if age_seconds <= _FRESH_PLUS5_SECONDS:
        return min(100.0, used_percent + _FRESH_PLUS5_PP)
    if age_seconds <= _FRESH_PLUS10_SECONDS:
        return min(100.0, used_percent + _FRESH_PLUS10_PP)
    return None


def unknown_floor(complete_pressures: Sequence[float]) -> float:
    """`unknown_floor = min(90, min(complete_pressures) + 10)`, or 0 with none complete.

    `complete_pressures` is every freshness-complete (non-UNKNOWN) window
    pressure across the CURRENT candidate set — every window of every
    account still being considered for this placement, not just the one
    account whose pressure is being filled in.
    """
    if not complete_pressures:
        return 0.0
    return min(_UNKNOWN_FLOOR_CAP, min(complete_pressures) + _UNKNOWN_FLOOR_MARGIN)


def _wall_to_monotonic(wall_value: float, *, wall_now: float, monotonic_now: float) -> float:
    """Design v2 §5.2: convert a stored wall-clock timestamp into the router's monotonic domain."""
    return monotonic_now + (wall_value - wall_now)


@dataclass
class _WindowObservation:
    used_percent: float
    observed_at: float
    reset_at: float | None
    reset_identity: str | None
    source: str


@dataclass
class _WarningSignal:
    observed_at: float
    reset_identity: str | None


class ObservationView:
    """Per-account, per-window usage observations feeding pressure computation.

    Fed from two sources (design v2 §2): periodic `peek_with_metadata`-shaped
    polls and directly ingested, source-tagged observations — both go
    through `ingest_window`, which keeps only the latest reading per
    `(account_id, window)`. `ingest_allowed_warning` separately retains the
    upstream "allowed: warning" signal a fresh matching-reset-identity
    window observation can haircut against. Every timestamp lives in one
    caller-supplied clock domain (the owning router's monotonic clock);
    callers convert wall-clock inputs (e.g. `resets_at` epochs) before
    ingesting.
    """

    def __init__(self) -> None:
        self._windows: dict[tuple[str, str], _WindowObservation] = {}
        self._warnings: dict[tuple[str, str], _WarningSignal] = {}

    def ingest_window(
        self,
        account_id: str,
        window: str,
        *,
        used_percent: float,
        source: str,
        observed_at: float,
        reset_at: float | None = None,
        reset_identity: str | None = None,
    ) -> None:
        self._windows[(account_id, window)] = _WindowObservation(
            used_percent=used_percent,
            observed_at=observed_at,
            reset_at=reset_at,
            reset_identity=reset_identity,
            source=source,
        )

    def ingest_allowed_warning(
        self, account_id: str, window: str, *, observed_at: float, reset_identity: str | None
    ) -> None:
        self._warnings[(account_id, window)] = _WarningSignal(
            observed_at=observed_at, reset_identity=reset_identity
        )

    def window_pressure(self, account_id: str, window: str, *, now: float) -> float | None:
        """The freshness-ladder-adjusted pressure for one window, or `None` (UNKNOWN)."""
        observation = self._windows.get((account_id, window))
        if observation is None:
            return None
        reset_passed = observation.reset_at is not None and observation.reset_at <= now
        return freshness_adjusted_pressure(
            observation.used_percent, now - observation.observed_at, reset_passed=reset_passed
        )

    def window_reset_identity(self, account_id: str, window: str) -> str | None:
        observation = self._windows.get((account_id, window))
        return observation.reset_identity if observation is not None else None

    def has_active_warning(self, account_id: str, window: str, *, now: float) -> bool:
        """True while a fresh (<=5 min) `allowed_warning` with a matching reset identity is retained."""
        warning = self._warnings.get((account_id, window))
        if warning is None:
            return False
        if now - warning.observed_at > _WARNING_FRESH_SECONDS:
            return False
        return warning.reset_identity == self.window_reset_identity(account_id, window)


def warning_factor(view: ObservationView, account_id: str, windows: Sequence[str], *, now: float) -> float:
    """`W(a) = H(a) x warning_factor(a)`: the 0.5 haircut while any binding window retains a matching warning."""
    if any(view.has_active_warning(account_id, window, now=now) for window in windows):
        return _WARNING_HAIRCUT_FACTOR
    return 1.0


# -- weight computation: positive-set rule + amended emergency branch ------


def account_headroom(pressure: float, in_flight: int) -> float:
    """`H(a) = max(0, 100 - P(a) - 2*M(a))`."""
    return max(0.0, 100.0 - pressure - _IN_FLIGHT_PRESSURE_WEIGHT * in_flight)


def emergency_capacity(pressure: float, factor: float) -> float:
    """`C0 = max(1, (100 - P) x factor)`: the amended emergency branch's capacity floor.

    Only reached when every eligible candidate's ordinary weight is zero
    (the positive-set is empty): it guarantees a strictly positive input to
    `emergency_weight` even once raw headroom has bottomed out, so
    `place_session` can still commit to somebody instead of dividing by
    zero.
    """
    C0 = max(1.0, (100.0 - pressure) * factor)
    return C0


def emergency_weight(pressure: float, factor: float, in_flight: int) -> float:
    """`W = C0^2 / (C0 + 2M)`: the amended emergency branch's weight."""
    C0 = emergency_capacity(pressure, factor)
    return (C0 * C0) / (C0 + _IN_FLIGHT_PRESSURE_WEIGHT * in_flight)


def select_weights(
    account_ids: Sequence[str],
    *,
    pressures: Mapping[str, float],
    warning_factors: Mapping[str, float],
    in_flight: Mapping[str, int],
) -> dict[str, float]:
    """The positive-set rule: score only positive-weight accounts, unless every account is zero.

    Returns `{account_id: weight}` for exactly the accounts that may be
    drawn from. When at least one candidate's ordinary `W(a) = H(a) x
    warning_factor(a)` is positive, every zero-weight candidate is dropped
    (a zero-weight account is never scored while a positive-weight one
    exists) and the positive ones are returned as-is. Only when the
    positive-set is empty does the amended emergency branch (`C0`,
    `emergency_weight`) run, for every candidate.
    """
    ordinary_weights = {
        account_id: account_headroom(pressures[account_id], in_flight.get(account_id, 0))
        * warning_factors.get(account_id, 1.0)
        for account_id in account_ids
    }
    positive_set = {account_id: weight for account_id, weight in ordinary_weights.items() if weight > 0}
    if positive_set:
        return positive_set
    return {
        account_id: emergency_weight(
            pressures[account_id], warning_factors.get(account_id, 1.0), in_flight.get(account_id, 0)
        )
        for account_id in account_ids
    }


def resolve_tie_break(tied_account_ids: Sequence[str], *, serving_account_id: str | None) -> str:
    """§2's tie rule: the serving pin wins if it is one of the tied accounts, else the lexically smallest id."""
    if serving_account_id is not None and serving_account_id in tied_account_ids:
        return serving_account_id
    return min(tied_account_ids)


def pick_weighted_hrw(
    *,
    weights: Mapping[str, float],
    seed: bytes,
    session_key_digest: bytes,
    serving_account_id: str | None = None,
) -> str:
    """Weighted HRW draw: `score = -ln(u)/W(a)`; the lowest score wins.

    `u` is `hrw_unit_interval`'s per-(session, account) sample. A tie in the
    lowest score resolves via `resolve_tie_break`.
    """
    if not weights:
        raise NoEligibleAccountError("no candidate carries a positive weight to draw from")
    scored = [
        (-math.log(hrw_unit_interval(seed, session_key_digest, account_id)) / weight, account_id)
        for account_id, weight in weights.items()
    ]
    lowest_score = min(score for score, _account_id in scored)
    tied = [account_id for score, account_id in scored if score == lowest_score]
    if len(tied) == 1:
        return tied[0]
    return resolve_tie_break(tied, serving_account_id=serving_account_id)


# -- eligibility filtering ---------------------------------------------------


class NoEligibleAccountError(RuntimeError):
    """Raised when no candidate survives eligibility filtering (or carries a positive weight)."""


@dataclass(frozen=True)
class AccountCandidate:
    """One account's picker-relevant eligibility state for a single `place_session` call."""

    account_id: str
    account_incarnation_id: str
    ready: bool = True
    capability_denied: bool = False
    account_cooldown_until: float | None = None
    family_cooldown_until: float | None = None


def is_eligible_candidate(
    candidate: AccountCandidate, *, now: float, already_attempted: frozenset[str]
) -> bool:
    """ready, capability != denied, not account-cooling, not family-cooling, not already attempted."""
    if not candidate.ready:
        return False
    if candidate.capability_denied:
        return False
    if candidate.account_id in already_attempted:
        return False
    if candidate.account_cooldown_until is not None and candidate.account_cooldown_until > now:
        return False
    if candidate.family_cooldown_until is not None and candidate.family_cooldown_until > now:
        return False
    return True


# -- pin map (design v2 §5.1-§5.3) ------------------------------------------


class PendingDurabilityBarrier:
    """A cancellation-safe, resolve-exactly-once gate for one pin generation's durable write.

    Spec-gate ruling (§5.3): every request resolving this pin generation —
    not just its creator — awaits `wait()`, which shields the underlying
    event so a waiter's own cancellation can never cancel (or double-fire)
    the barrier's resolution, before starting an upstream attempt.
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
# (design v2 §4.3-§4.5, §5.7)

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
    """One migration reservation per session generation (design v2 §4.3), attempt-owned
    and cancellation-safe.

    Created in the same no-await critical section as its migration-attempt token
    (`ClaudeBalancedRouter.acquire_migration_reservation`). `resolved_event` (an
    `asyncio.Event`) plus the stored `outcome` is the resolution primitive every waiter
    awaits through `asyncio.shield` — cancelling one waiter never cancels, and never
    double-fires, the shared event. `outcome` starts `"pending"` and is set exactly once
    by the owner-terminal path after it verifies `owner_attempt_id`; once terminal it is
    immutable — a second attempted resolution is always a no-op (enforced by the router).
    Never persisted (§4.3): reservations refer to live tasks and cancellation scopes that
    die with the process.
    """

    source_account: str
    source_generation: int
    target_account: str
    owner_attempt_id: str
    outcome: MigrationOutcome = "pending"
    resolved_event: asyncio.Event = field(default_factory=asyncio.Event)


class ClaudeBalancedRouter:
    """Design v2 §2/§5's balanced picker core: pressures, weighted HRW, pin map.

    Owns the in-memory `ObservationView`, the `digest -> PinEntry` pin map
    (TTLs, LRU eviction, exactly-once-decrementing counters), each
    account's in-flight attempt count `M(a)`, and the atomic
    `place_session` pick+pin-insert critical section — synchronous
    end-to-end (no `await`), so nothing can interleave between picking an
    account and inserting its pin. Durable persistence (the
    `pending_durability` barrier and the coalesced `last_seen` refresh,
    §5.3) goes through an optional `ClaudePoolRuntimeStateStore`. It also
    owns the design v2 §4.3-§4.5/§5.7 migration machinery: per-session-
    generation reservations, their `asyncio.shield`-based waiter protocol,
    migration-attempt tokens keyed by attempt id (the §2.4 `M(a)` term),
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

        # Migration machinery (design v2 §4.3-§4.5, §5.7): reservations and
        # tokens are NEVER persisted, so both always start empty here —
        # there is no restore path for either.
        self._reservations: dict[bytes, MigrationReservation] = {}
        self._migration_tokens: dict[str, str] = {}
        self._removed_accounts: set[str] = set()
        self.migration_outcome_counts: dict[str, int] = {}
        self.migration_cas_lost = 0
        self.migration_commit_rejected_target_removed = 0

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
        """Adapt `ClaudeAccountUsageCache.peek_with_metadata`'s shape (T-6) into `ObservationView`."""
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
        """Directly ingest one source-tagged window observation (design v2 §2)."""
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

    # -- cold start (design v2 §2.4) -----------------------------------------

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
        """Explicit external removal (e.g. account removal, §5.7)."""
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

    # -- atomic pick + pin-insert (design v2 §5.1) ---------------------------

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

        family = quota_family(model)
        if self._is_cold_start(eligible_by_id, all_by_id, family, serving_account_id, now=now):
            chosen_id = serving_account_id
            assert chosen_id is not None
        else:
            weights = self._candidate_weights(eligible, family, now=now)
            chosen_id = pick_weighted_hrw(
                weights=weights,
                seed=seed,
                session_key_digest=session_key.digest,
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

    # -- durable persistence (design v2 §5.3) --------------------------------

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
        """§5.3's durable `last_seen` refresh, coalesced <=1/60s per pin by the store itself."""
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

    # -- restore-from-store initialization (design v2 §5.2, §5.5) -----------

    def restore_from_store(
        self, restore_result: RestoreResult, *, now: float | None = None, wall_now: float | None = None
    ) -> int:
        """Convert restored pin rows (wall-clock) into the monotonic pin map, recomputing counters.

        `restore_result.pins` already excludes expired and epoch-mismatched
        rows (`ClaudePoolRuntimeStateStore.restore`); every remaining row is
        durable, so it carries no `pending_durability` barrier.
        """
        monotonic_now = self._clock() if now is None else now
        wall_now = self._wall_clock() if wall_now is None else wall_now

        self._pins.clear()
        self.removed_pin_counts.clear()
        self.total_removed_pins = 0
        self.soft_bound_overflow_count = 0

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
            )
        self._restored_any_valid_pin = bool(restore_result.pins)
        return len(self._pins)

    # -- migration reservations and waiters (design v2 §4.3) -----------------

    def get_migration_reservation(self, digest: bytes) -> MigrationReservation | None:
        """The no-await critical-section read a waiter's loop starts with (§4.3 step 1):
        the current reservation reference for `digest`, or `None` if no migration is
        in flight for this session.
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
        """The atomic reservation+token acquisition critical section (§4.3-§4.4):
        entirely synchronous (no `await`), the same single-event-loop discipline as
        `place_session`, so no other coroutine can interleave between the check and
        the insert.

        If a reservation is already pending for `digest`, returns THAT reservation and
        `False` — this caller becomes a WAITER on the existing migration; concurrent
        same-session requests wait and re-read, they never pick an independent target
        (§4.3). Otherwise this caller becomes the OWNER: a fresh pending reservation and
        its migration-attempt token (keyed by `attempt_id`, created in the SAME critical
        section, §4.4) are inserted, the pin is marked `migration_reserved` so eviction
        cannot reclaim it mid-migration, and `(reservation, True)` is returned.
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
        """§4.3's waiter protocol step 2: `await asyncio.shield(...)` on the reservation's
        event, so a waiter's own cancellation can never cancel — or double-fire — the
        shared resolution. The caller re-enters the critical section and re-reads all
        routing state afterward (steps 3-4); it never trusts the stale target embedded
        in the (by-then-cleared) reservation it just waited on.
        """
        await asyncio.shield(reservation.resolved_event.wait())

    def resolve_migration_reservation(
        self, digest: bytes, *, attempt_id: str, outcome: MigrationOutcome
    ) -> bool:
        """The no-await, idempotent core of every owner-terminal path (§4.3): verify
        `owner_attempt_id`, record the (from-here-immutable) outcome, clear the
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

    # -- migration-attempt tokens: M(a) (design v2 §4.4) ---------------------

    def migration_token_target(self, attempt_id: str) -> str | None:
        """The live token's target account for `attempt_id`, or `None` if released/absent."""
        return self._migration_tokens.get(attempt_id)

    def release_migration_token(self, attempt_id: str) -> str | None:
        """`migration_tokens.pop(attempt_id, None)` — inherently idempotent (§4.4): the
        first call releases the token and decrements the target's `M(a)`; every later
        call for the same `attempt_id` is a no-op that returns `None`.
        """
        target_account = self._migration_tokens.pop(attempt_id, None)
        if target_account is not None:
            self.end_attempt(target_account)
        return target_account

    def resolve_migration_owner_terminal(
        self, digest: bytes, *, attempt_id: str, outcome: MigrationOutcome
    ) -> bool:
        """The ONE idempotent owner-terminal API (Steps 2-3): every owner exit path —
        request cancellation, timeout, mode drain, an exception raised before response
        headers, target rejection, CAS loss, or success — funnels through this. Entirely
        synchronous (no `await`), so it is safe to call from a `finally` block while
        reservation state is mutated: no cancellation can interrupt the transition
        between ownership verification and event signaling. Resolves the owned
        reservation to `outcome` (a no-op if it is not pending, not owned by
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
        """Design v2 §4.4's terminal ordering for a pre-header quota/eligibility failure:
        ① classify the response, ② install the cooldown/capability evidence in memory,
        and ③ fence the target are the caller's own responsibility, run BEFORE calling
        this (so a woken waiter re-reads state that already reflects them); this method
        performs ④ resolve the reservation and wake waiters and ⑤ release the migration
        token, synchronously and without an intervening await, via
        `resolve_migration_owner_terminal`. ⑥ persistence/diagnostics are scheduled by
        the caller AFTER this returns.

        Also applies §5.7 case 2's follow-up clause: if this reservation's source
        account was separately marked removed (`remove_account`), the now-orphaned
        source pin is deleted too, so a later request places fresh instead of reusing a
        pin to a removed account.
        """
        reservation = self._reservations.get(digest)
        source_account = reservation.source_account if reservation is not None else None
        resolved_now = self.resolve_migration_owner_terminal(digest, attempt_id=attempt_id, outcome=outcome)
        if resolved_now and source_account is not None and source_account in self._removed_accounts:
            self.remove_pin(digest)
        return resolved_now

    # -- commit point: upstream 2xx headers (design v2 §4.5) -----------------

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
        """ONE no-await critical section performing design v2 §4.5's commit at the
        target's upstream 2xx headers. Reservation resolution happens HERE — never
        deferred to stream completion:

        1. `ingest_headers()` (if supplied) — the caller's own ratelimit-header
           ingestion hook (§3.2), run first so the fresh observation is visible before
           anything else in this section reads pressure/eligibility state;
        2. validate target membership (§5.7 case 4: the target was removed by the time
           its 2xx headers arrived) — on failure: outcome `target_removed`, the
           reservation resolves and wakes waiters, no pin is touched;
        3. the generation/owner CAS: `pin.account_id == source_account`,
           `pin.generation == source_generation`,
           `reservation.owner_attempt_id == attempt_id` — on failure: outcome
           `cas_lost` (the newer pin is never overwritten, `migration_cas_lost`
           increments), the reservation resolves and wakes waiters;
        4. on success: the pin flips to `target_account`, `generation += 1`, a fresh
           `pending_durability` barrier is attached to the new generation, the
           reservation is cleared, and its event is set exactly once — outcome
           `committed`.

        The migration-attempt TOKEN is untouched here by design (§4.4/Step 5): it
        survives until the caller releases it (`resolve_migration_owner_terminal` /
        `release_migration_token`) from the attempt's common `finally`, once the
        upstream body/stream terminates — `M(target_account)` stays incremented for the
        whole streamed response, not just until these headers commit.

        Returns `(outcome, pin, barrier)`. On `committed`, the caller awaits `barrier`'s
        high-priority durability completion (`submit_new_pin_durability` /
        `await_pin_durability`, T-7 semantics) OUTSIDE this critical section, before
        forwarding any downstream byte; same-session waiters re-read the pin and await
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

    # -- account removal transition matrix (design v2 §5.7) ------------------

    def remove_account(self, account_id: str) -> dict[str, int]:
        """§5.7's account-removal transition matrix, cases 1-5 (case 4 is handled at
        commit time by `commit_at_headers`'s `target_still_registered` check, since it
        depends on whether upstream 2xx already arrived). Never bulk-resets migration
        tokens or the in-flight counters underneath them — an owned token is released
        exactly once, only by its own attempt's own terminal cleanup (case 3: "removed
        exactly once by terminal cleanup"), not by this batch operation; the mandatory
        lazy per-lookup/pre-commit membership re-check remains the correctness
        backstop. Returns a per-case removal count for diagnostics.
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
        return case_counts
