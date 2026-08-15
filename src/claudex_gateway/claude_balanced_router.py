"""Domain-separated session-key derivation for balanced (session-affinity) routing.

Session keys are HMAC-SHA256 digests under the durable per-deployment epoch
seed. Raw Claude Code UUIDs and message content are never stored — only the
digest. Two branches decide
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

`hrw_unit_interval` is the separate rendezvous-hashing (Highest Random
Weight) sample: a per-(session, account) digest reduced to a
uniformly distributed point in the open interval (0, 1), so the account with
the largest sample can be picked without keeping any pin-map state.

The rest of this module implements the balanced picker core.
`ObservationView` turns per-window usage readings into
freshness-ladder-adjusted pressures, `quota_family` and the
positive-set and emergency weight formulas turn pressures into a
weighted-HRW draw, and `ClaudeBalancedRouter.place_session` performs the
atomic pick-and-pin critical section against an in-memory, TTL/LRU-bounded
pin map with a cancellation-safe `pending_durability` barrier for the
initial durable write.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from claudex_gateway.account_usage_cache import ClaudeAccountUsageCache
from claudex_gateway.claude_account_profile import load_account_profile_fingerprint
from claudex_gateway.claude_accounts import AccountRecord, list_accounts
from claudex_gateway.balanced.state_model import RestoreResult, RestoreValidationContext
from claudex_gateway.balanced.state_store import ClaudePoolRuntimeStateStore

logger = logging.getLogger(__name__)

_SESSION_KEY_DOMAIN = b"claudex-session-key-v1"
# The pin identity is the LOGICAL session digest salted with the request's
# quota family, so one Claude Code session carries one independent pin per
# family (correlated-HRW design). The domain version is part of the pin-key
# ABI: bumping it requires bumping `balanced.state_model.SCHEMA_VERSION`
# so pre-existing rows are quarantined rather than left as unreachable
# entries that would distort the LRU and cold-start counts.
_PIN_KEY_DOMAIN = b"claudex-pin-key-v2"
_HRW_DOMAIN = b"claudex-balanced-hrw-v1"
_STATELESS_REQUEST_DOMAIN = b"claudex-stateless-request-v1"

# `quota_family` is a closed enum and part of the pin-key ABI: adding a
# family (or moving a model between families) creates new pin identities.
_QUOTA_FAMILIES = ("default", "fable")


@dataclass(frozen=True)
class SessionKey:
    """A domain-separated session-affinity identity and the branch that produced it.

    `digest` is the family-salted PIN identity: it keys the pin map,
    reservations, migration CAS, and the durable `pins` rows. Requests of
    different quota families in one logical session deliberately stop
    contending on it. `scoring_digest` is the family-agnostic LOGICAL
    session digest used for weighted-HRW scoring, so that when family
    weights are equal both families of one session co-locate on the same
    account, diverging only under family-specific pressure or cooldowns.
    An empty `scoring_digest` means "same as `digest`" (hand-built keys in
    tests); read it through `scoring_digest_or_default`.
    """

    digest: bytes
    kind: Literal["uuid", "content_hash"]
    scoring_digest: bytes = b""
    family: str = ""

    @property
    def scoring_digest_or_default(self) -> bytes:
        return self.scoring_digest or self.digest


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


def derive_session_key(body: dict[str, Any], seed: bytes, family: str) -> SessionKey | None:
    """Derive the (session, quota family) affinity key for `body`, or None when unpinnable.

    Tries the uuid branch first (Claude Code's own `session_id`), then falls
    back to hashing the first user message; either branch yields the LOGICAL
    session digest. The returned key's pin `digest` is that logical digest
    salted with `family` under `_PIN_KEY_DOMAIN`, while `scoring_digest`
    stays the unsalted logical digest (see `SessionKey`). `family` must be
    one of `quota_family`'s closed enum values — it is baked into durable
    pin identities, so a stray value would silently mint a new key space.
    Returns None only when neither branch has anything to hash.
    """
    if family not in _QUOTA_FAMILIES:
        raise ValueError(f"invalid quota family: {family!r}")
    logical = _uuid_session_key(body, seed) or _content_hash_session_key(body, seed)
    if logical is None:
        return None
    pin_digest = _hmac_sha256(
        seed, _PIN_KEY_DOMAIN + b"\x00" + _length_prefixed(logical.digest, family.encode("utf-8"))
    )
    return SessionKey(
        digest=pin_digest, kind=logical.kind, scoring_digest=logical.digest, family=family
    )


def hrw_unit_interval(seed: bytes, session_key_digest: bytes, account_id: str) -> float:
    """Map (seed, session_key_digest, account_id) onto the open interval (0, 1).

    Highest-Random-Weight sampling routes a session to the account with the
    largest sample. The deterministic mapping needs no stored state to keep a
    session pinned to the same account across runs. `mac`'s high 53 bits give
    the full double-precision mantissa worth
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
# Balanced picker core
# ==========================================================================

# -- freshness ladder and unknown floor ------------------------------------

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

# Map `ClaudeAccountUsageCache.peek_with_metadata` window names to the
# binding-window names also stored in `usage_observations.window`.
_PEEK_WINDOW_TO_BINDING: dict[str, str] = {
    "session": "five_hour",
    "weekly": "seven_day",
    "fable_weekly": "fable_weekly",
}

# Pin-map bounds
DEFAULT_PIN_TTL_UUID_SECONDS = 5 * 3600
DEFAULT_PIN_TTL_CONTENT_HASH_SECONDS = 30 * 60
DEFAULT_PIN_MAP_MAX_ENTRIES = 10_000


def _is_ascii_alnum(char: str | None) -> bool:
    return char is not None and char.isascii() and char.isalnum()


def _bounded_token_present(lowered: str, token: str) -> bool:
    """True when `token` appears in `lowered`, bounded on each side by the
    string's start/end or a non-alphanumeric (ASCII) character — so a token
    merely contained in a larger word (e.g. "fable" inside "unfable" or
    "fabled") never matches.
    """
    start = 0
    while True:
        index = lowered.find(token, start)
        if index == -1:
            return False
        before = lowered[index - 1] if index > 0 else None
        after_index = index + len(token)
        after = lowered[after_index] if after_index < len(lowered) else None
        if not _is_ascii_alnum(before) and not _is_ascii_alnum(after):
            return True
        start = index + 1


def quota_family(model: str) -> str:
    """Return the binding-window family for a model id.

    The ASCII case-insensitive token "fable", bounded on each side by the
    string's start/end or a non-alphanumeric (ASCII) character, selects the
    "fable" family — which adds the `fable_weekly` binding window on top of
    the default `[five_hour, seven_day]` pair. Every other model (including
    a token merely containing "fable" as a substring of a larger word, e.g.
    "unfable" or "fabled") uses the "default" family.
    """
    return "fable" if _bounded_token_present(model.lower(), "fable") else "default"


# Capability evidence uses the bounded, case-insensitive family token a model
# id carries, otherwise the exact lowercased model id. Tokens are checked in
# fixed order; real model ids never carry more than one of these tokens.
_CAPABILITY_KEY_TOKENS: tuple[str, ...] = ("fable", "opus", "sonnet", "haiku")


def capability_key(model: str) -> str:
    """Return the capability-evidence key for a model id.

    A bounded (never a larger word's substring), case-insensitive match
    against `_CAPABILITY_KEY_TOKENS` wins; every other model id falls back to
    `"model:<lowercase id>"`, so capability evidence never accidentally
    conflates two unrelated model ids that merely share no family token.
    """
    lowered = model.lower()
    for token in _CAPABILITY_KEY_TOKENS:
        if _bounded_token_present(lowered, token):
            return token
    return f"model:{lowered}"


def binding_windows(family: str) -> tuple[str, ...]:
    """Return the usage windows that bind the requested quota family."""
    return _FABLE_WINDOWS if family == "fable" else _NON_FABLE_WINDOWS


def freshness_adjusted_pressure(
    used_percent: float, age_seconds: float, *, reset_passed: bool
) -> float | None:
    """Apply the freshness ladder, returning `None` for an unknown reading.

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
    """Convert a stored wall-clock timestamp into the router's monotonic domain."""
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


@dataclass(frozen=True)
class RealWindowReading:
    """One window's REAL (never `unknown_floor`, never inferred) freshness-
    adjusted reading, plus its raw age. The family cooldown gate consumes
    this value directly.
    """

    adjusted_pressure: float
    age_seconds: float


class ObservationView:
    """Per-account, per-window usage observations feeding pressure computation.

    Periodic `peek_with_metadata`-shaped polls and directly ingested,
    source-tagged observations both go through `ingest_window`, which keeps
    only the latest reading per `(account_id, window)`.
    `ingest_allowed_warning` separately retains the
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

    def window_reset_at(self, account_id: str, window: str) -> float | None:
        observation = self._windows.get((account_id, window))
        return observation.reset_at if observation is not None else None

    def real_window_reading(
        self, account_id: str, window: str, *, now: float, max_age_seconds: float
    ) -> RealWindowReading | None:
        """The window's own freshness-adjusted pressure, but ONLY when a REAL
        observation exists and is at most `max_age_seconds` old — `None` for
        anything else (missing, aged out, or its reset already passed). Reads the
        raw stored observation directly, never `unknown_floor`-substituted or
        emergency-weighted, because the family gate requires a real, fresh
        observation.
        """
        observation = self._windows.get((account_id, window))
        if observation is None:
            return None
        age_seconds = now - observation.observed_at
        if age_seconds > max_age_seconds:
            return None
        reset_passed = observation.reset_at is not None and observation.reset_at <= now
        adjusted = freshness_adjusted_pressure(
            observation.used_percent, age_seconds, reset_passed=reset_passed
        )
        if adjusted is None:
            return None
        return RealWindowReading(adjusted_pressure=adjusted, age_seconds=age_seconds)

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


# -- weight computation: positive-set rule + emergency branch ------


def account_headroom(pressure: float, in_flight: int) -> float:
    """`H(a) = max(0, 100 - P(a) - 2*M(a))`."""
    return max(0.0, 100.0 - pressure - _IN_FLIGHT_PRESSURE_WEIGHT * in_flight)


def emergency_capacity(pressure: float, factor: float) -> float:
    """`C0 = max(1, (100 - P) x factor)`: the emergency branch's capacity floor.

    Only reached when every eligible candidate's ordinary weight is zero
    (the positive-set is empty): it guarantees a strictly positive input to
    `emergency_weight` even once raw headroom has bottomed out, so
    `place_session` can still commit to somebody instead of dividing by
    zero.
    """
    C0 = max(1.0, (100.0 - pressure) * factor)
    return C0


def emergency_weight(pressure: float, factor: float, in_flight: int) -> float:
    """`W = C0^2 / (C0 + 2M)`: the emergency branch's weight."""
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
    positive-set is empty does the emergency branch (`C0`,
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
    """Prefer a tied serving pin, otherwise return the lexically smallest id."""
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


# -- Fable family-scoped cooldown gate --------------------------------------

# Thresholds are percentage points; observation age is measured in seconds.
_FAMILY_GATE_MAX_OBSERVATION_AGE_SECONDS = 15 * 60
_FAMILY_GATE_FABLE_WEEKLY_MIN_PERCENT = 99.0
_FAMILY_GATE_FIVE_HOUR_MAX_PERCENT = 70.0
_FAMILY_GATE_SEVEN_DAY_MAX_PERCENT = 70.0


@dataclass(frozen=True)
class FamilyGateOutcome:
    """`classify_balanced_cooldown_scope`'s verdict: which `install_cooldown`
    scope to install, and why."""

    scope: Literal["account", "family"]
    reason: str
    family_deadline: float | None = None


def classify_balanced_cooldown_scope(
    observations: ObservationView,
    *,
    account_id: str,
    model: str,
    upstream_status_code: int,
    now: float,
) -> FamilyGateOutcome:
    """Classify whether a cooldown can be scoped to the Fable family.

    A family-scoped cooldown keyed by account and family installs only when
    all six conditions hold. Each is checked independently, and a single
    failing condition falls all the way back to
    account-wide, never a partial family scope:

    1. the request's model family is "fable";
    2. the account-specific failure is an ACTUAL upstream quota 429;
    3. `fable_weekly` has a real, fresh reading that is never an
       `unknown_floor`, emergency weight, or inferred value; here it must be
       present, at most 15 minutes old, and adjusted to at least 99%;
    4. `five_hour`'s reading is REAL/fresh with an adjusted pressure <=70%;
    5. `seven_day`'s reading is REAL/fresh with an adjusted pressure <=70%;
    6. the Fable reset is present and still in the future — it becomes the
       scoped deadline (`family_deadline`).

    Otherwise account-wide. A 429 never implies model ineligibility on its
    own — this only ever decides cooldown SCOPE, never capability evidence.
    """
    if quota_family(model) != "fable":
        return FamilyGateOutcome("account", "request_family_not_fable")
    if upstream_status_code != 429:
        return FamilyGateOutcome("account", "not_upstream_quota_429")

    fable_weekly = observations.real_window_reading(
        account_id, "fable_weekly", now=now, max_age_seconds=_FAMILY_GATE_MAX_OBSERVATION_AGE_SECONDS
    )
    if fable_weekly is None or fable_weekly.adjusted_pressure < _FAMILY_GATE_FABLE_WEEKLY_MIN_PERCENT:
        return FamilyGateOutcome("account", "fable_weekly_not_saturated")

    five_hour = observations.real_window_reading(
        account_id, "five_hour", now=now, max_age_seconds=_FAMILY_GATE_MAX_OBSERVATION_AGE_SECONDS
    )
    if five_hour is None or five_hour.adjusted_pressure > _FAMILY_GATE_FIVE_HOUR_MAX_PERCENT:
        return FamilyGateOutcome("account", "five_hour_not_clear")

    seven_day = observations.real_window_reading(
        account_id, "seven_day", now=now, max_age_seconds=_FAMILY_GATE_MAX_OBSERVATION_AGE_SECONDS
    )
    if seven_day is None or seven_day.adjusted_pressure > _FAMILY_GATE_SEVEN_DAY_MAX_PERCENT:
        return FamilyGateOutcome("account", "seven_day_not_clear")

    reset_at = observations.window_reset_at(account_id, "fable_weekly")
    if reset_at is None or reset_at <= now:
        return FamilyGateOutcome("account", "fable_reset_not_valid")

    return FamilyGateOutcome("family", "fable_family_gate_satisfied", family_deadline=reset_at)


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


# ==========================================================================
# Balanced routing runtime lifecycle
# ==========================================================================

BalancedRuntimeStatus = Literal["disabled", "acquiring", "active", "draining"]


class BalancedPrepareError(RuntimeError):
    """Raised by `ClaudeBalancedRuntime.prepare_and_publish` when the runtime cannot be
    safely readied because a ready account has no valid profile fingerprint. The
    caller (server.py's PUT routing handler) reports this to the admin client;
    preparation is always torn down first, so the previous routing mode is left
    untouched.
    """


class ClaudeBalancedRuntime:
    """Owns balanced routing's whole process-lifetime state machine.

    `status` gates every dispatch (`server._passthrough_with_claude_balanced`):
    "disabled" (no runtime — balanced traffic fails closed), "acquiring" (an enable is
    being prepared — the OLD mode still serves, since `claude_account.routing` itself
    hasn't flipped to "balanced" yet), "active" (`store`/`router` are live and
    `begin_request` admits new dispatch slots), "draining" (an intentional exit or
    process shutdown is underway — no new slot is admitted, in-flight ones are
    awaited). A request arriving mid-transition awaits `wait_for_transition`
    before re-reading the published routing mode and dispatching under it. The
    transition event is cleared for the whole
    "acquiring"/"draining" window and only set once the new state is fully published,
    so a woken waiter never observes a stale mode.

    The server layer supplies `persist` and `publish` hooks so this class can
    enforce the ordering its two distinct lifecycle operations require without
    owning `GatewayConfig` or the settings file itself: enabling persists settings
    before the prepared runtime is published (`prepare_and_publish`); exiting
    persists and publishes the target mode before waking transition waiters
    (`exit_mode`).
    Process shutdown (`shutdown_preserving_epoch`) takes no hook at all — it never
    touches persisted settings or epoch metadata, so a restart can restore them.

    `begin_request` and `end_request` maintain this class's request-slot
    counter, entirely independent of `ClaudeBalancedRouter`'s
    per-account `M(a)` attempt counting: it exists purely so the drain step of
    `exit_mode`/`shutdown_preserving_epoch` has something to wait on.
    """

    def __init__(self) -> None:
        self.status: BalancedRuntimeStatus = "disabled"
        self.epoch_id: str | None = None
        self.epoch_seed: bytes = b""
        self.router: ClaudeBalancedRouter | None = None
        # The usage poll coordinator exists only while this runtime is active;
        # it is `None` in "disabled", "acquiring", or "draining",
        # and also `None` in "active" when `prepare_and_publish` was called
        # without a `usage_cache` (every direct, non-HTTP caller keeps this
        # optional; server.py's real routing handler always supplies one).
        self.usage_poll_coordinator: ClaudeUsagePollCoordinator | None = None
        self.ambient_usage_poll_supplier: (
            Callable[[Sequence[AccountRecord]], UsagePollAccount | None] | None
        ) = None

        self._store: ClaudePoolRuntimeStateStore | None = None
        # The background driver calls `usage_poll_coordinator.run_due_poll`
        # while this runtime is "active" -- `None` whenever no coordinator is
        # driving (every status
        # other than "active", or "active" without a `usage_cache`).
        self._usage_poll_driver_task: asyncio.Task[None] | None = None
        self._accounts_root: Path | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._transition_event = asyncio.Event()
        self._transition_event.set()
        self._active_requests = 0
        self._drain_complete = asyncio.Event()
        self._drain_complete.set()

    @property
    def epoch_active(self) -> bool:
        """Return whether this runtime's current epoch is live in memory.

        This is true only while status is "active". The runtime store's
        `epoch_active` metadata has no public setter and is never written by
        this class; this property is computed from `status` alone.
        """
        return self.status == "active"

    async def wait_for_transition(self) -> None:
        """Block while a controlled enable ("acquiring") or exit ("draining")
        transition is in flight; returns immediately once none is (including when none
        ever was).
        """
        await self._transition_event.wait()

    def begin_request(self) -> bool:
        """Admit one balanced dispatch slot iff `status == "active"`. Returns whether
        the slot was admitted; a caller that admits one MUST call `end_request` exactly
        once, however the dispatch ends.
        """
        if self.status != "active":
            return False
        self._active_requests += 1
        self._drain_complete.clear()
        return True

    def end_request(self) -> None:
        """Release one slot `begin_request` admitted. Always safe to call — a defensive
        no-op floor at zero, never below."""
        if self._active_requests > 0:
            self._active_requests -= 1
        if self._active_requests == 0:
            self._drain_complete.set()

    # -- enabling: prepare the complete runtime, publish only once settings commit ----

    async def prepare_and_publish(
        self,
        *,
        accounts: Sequence[AccountRecord],
        accounts_root: Path,
        runtime_db_path: Path,
        persist: Callable[[], None],
        entry: Literal["startup_restore", "admin_enable"],
        usage_cache: ClaudeAccountUsageCache | None = None,
    ) -> None:
        """Prepare and atomically publish balanced routing.

        Opens and validates the runtime store, restores its state, constructs the
        router, and verifies every ready account has a valid profile
        fingerprint while the old mode remains published, so traffic is
        unaffected until every check passes. `persist()` — the coordinator hook that
        persists+swaps `claude_account.routing` to "balanced" — is invoked exactly
        once, after every check passes and strictly before this runtime is published
        (`status` flips to "active"): it is the commit point. A failure anywhere up to
        and including `persist()` itself tears the (partial) preparation down (closing
        any opened store) and leaves `status` "disabled" — the old mode keeps serving,
        exactly as if this call had never happened.

        `entry` distinguishes the two supported reasons for preparing a runtime:
        `"startup_restore"` is the daemon lifespan restoring an already-persisted
        `"balanced"` mode across a process restart, which reuses the runtime DB's
        existing epoch/pins exactly as `shutdown_preserving_epoch` left them.
        `"admin_enable"` is every OTHER re-entry (an admin PUT transitioning into
        balanced) and durably rotates the epoch — wiping every pin — right after
        the store opens, before anything below restores from it: `exit_mode` is
        the only path that ever invalidates an epoch, and its own rotation can
        itself degrade (`persistence_degraded`) and leave the runtime DB holding
        a stale, intentionally-exited epoch and its pins; an administrative
        re-entry must never resurrect that state, so it always mints a fresh one
        regardless of what the store still contains.

        When supplied, `usage_cache` creates a fresh
        `ClaudeUsagePollCoordinator` for this runtime
        against the same router/store this call just prepared -- omitted, that
        attribute stays `None` (every non-server caller of this method).
        """
        async with self._lifecycle_lock:
            if self.status != "disabled":
                raise RuntimeError(
                    f"cannot enable balanced routing from status {self.status!r}"
                )
            self.status = "acquiring"
            self._transition_event.clear()
            store: ClaudePoolRuntimeStateStore | None = None
            try:
                store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
                if entry == "admin_enable":
                    # Administrative enablement must mint a fresh epoch and
                    # wipe its pins durably before anything below reads or
                    # restores from the store, so a degraded exit's
                    # stale epoch/pins can never be resurrected by a later
                    # administrative re-entry.
                    await store.rotate_epoch().wait_async()
                for record in accounts:
                    if record.state != "ready":
                        continue
                    fingerprint = load_account_profile_fingerprint(accounts_root / record.id)
                    if fingerprint is None:
                        raise BalancedPrepareError(
                            f"claude account {record.id} has no valid "
                            "profile_fingerprint (missing or non-UUID accountUuid); it "
                            "cannot participate in balanced routing until it is "
                            "re-authenticated"
                        )

                restore_result = store.restore(RestoreValidationContext(now_utc=time.time()))
                router = ClaudeBalancedRouter(
                    balanced_epoch_id=store.balanced_epoch_id, store=store
                )
                router.restore_from_store(restore_result)

                persist()

                self._store = store
                self.epoch_id = store.balanced_epoch_id
                self.epoch_seed = store.epoch_seed
                self.router = router
                self._accounts_root = accounts_root
                self.usage_poll_coordinator = (
                    ClaudeUsagePollCoordinator(cache=usage_cache, router=router, store=store)
                    if usage_cache is not None
                    else None
                )
                self.status = "active"
                self._start_usage_poll_driver()
            except BaseException:
                if store is not None:
                    store.close()
                self.status = "disabled"
                raise
            finally:
                self._transition_event.set()

    # -- usage poll driver: runs while active -------------------------------

    def _start_usage_poll_driver(self) -> None:
        """Start the background poll driver after the runtime becomes active.

        This is a no-op when `prepare_and_publish` was called without a
        `usage_cache`, so `usage_poll_coordinator` is `None` and nothing needs
        driving. Must only be called from inside the `_lifecycle_lock` critical
        section that just set `status = "active"`.
        """
        if self.usage_poll_coordinator is None:
            return
        self._usage_poll_driver_task = asyncio.create_task(self._run_usage_poll_driver())

    async def _stop_usage_poll_driver(self) -> None:
        """Cancel and await the poll driver if it is running.

        This must finish before the store closes in both `exit_mode` and
        `shutdown_preserving_epoch`, so no poll tick can ever touch a closed store.
        """
        task = self._usage_poll_driver_task
        self._usage_poll_driver_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_usage_poll_driver(self) -> None:
        """Run the poll driver sequentially until it is cancelled.

        The first tick runs immediately to warm the coordinator, then the loop
        sleeps for the coordinator's own scheduling budget before trying again.
        It runs until `_stop_usage_poll_driver` cancels it and re-reads
        `usage_poll_coordinator`
        every iteration rather than capturing it once, so it stops cleanly the
        moment the attribute is cleared instead of racing a stale reference.
        """
        while True:
            coordinator = self.usage_poll_coordinator
            if coordinator is None:
                return
            await self._usage_poll_driver_tick(coordinator)
            await asyncio.sleep(coordinator.poll_interval_seconds)

    async def _usage_poll_driver_tick(self, coordinator: ClaudeUsagePollCoordinator) -> None:
        """One driver iteration: re-reads the registry (read-through, exactly like
        `server._passthrough_with_balanced_pool`, so an account added/removed
        while balanced routing is active takes effect immediately) and calls
        `run_due_poll` with the current ready set. Never lets an exception escape
        -- an unexpected failure here (a bad registry read, a broken fingerprint
        file, ...) must never crash this loop or surface into a request path.
        """
        try:
            records = list_accounts()
            ready_ids = [record.id for record in records if record.state == "ready"]
            accounts: dict[str, UsagePollAccount] = {}
            accounts_root = self._accounts_root
            if accounts_root is not None:
                for record in records:
                    if record.state != "ready":
                        continue
                    accounts[record.id] = UsagePollAccount(
                        account_id=record.id,
                        account_incarnation_id=record.account_incarnation_id,
                        account_profile_fingerprint=load_account_profile_fingerprint(
                            accounts_root / record.id
                        ),
                    )
            if self.ambient_usage_poll_supplier is not None:
                ambient_account = self.ambient_usage_poll_supplier(records)
                if ambient_account is not None:
                    ready_ids.append(ambient_account.account_id)
                    accounts[ambient_account.account_id] = ambient_account
            await coordinator.run_due_poll(ready_ids, accounts=accounts)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("balanced usage poll driver tick failed", exc_info=True)

    # -- intentional exit: drain, persist, invalidate the epoch, publish -------------

    async def exit_mode(
        self,
        target_mode: str,
        *,
        persist: Callable[[], None] | None = None,
        publish: Callable[[], None],
    ) -> None:
        """Exit balanced mode intentionally for fallback or disabled routing.

        Unlike process shutdown (`shutdown_preserving_epoch`), this is the
        ONLY path that rotates the epoch (invalidating every current-epoch pin) and
        marks it inactive, so a later re-entry (`prepare_and_publish`) always starts
        a fresh epoch.

        Sequence: mark draining (blocks new balanced entrants — `begin_request` only
        admits while "active"), drain in-flight attempts, `persist()` — the
        coordinator hook that durably writes `claude_account.routing` to
        `target_mode`, omitted (`None`, the default) by callers with no settings
        layer of their own — THEN rotate the epoch durably (waits for the
        rotation to commit), THEN `publish()` — the coordinator hook that swaps
        the in-memory published mode, so a woken transition waiter re-reads the
        ALREADY-published target mode — then wake every transition waiter and
        close the store, discarding balanced-only state.

        Crash/cancellation contract: `persist()` is the commit point, and
        everything up to and including it is pre-commit. ANY `BaseException` —
        including `asyncio.CancelledError`, which is not an `Exception` — raised
        while draining or persisting aborts the exit entirely: this runtime
        returns to "active" with the transition event set and its epoch, store,
        and durable pins untouched, and the exception propagates to the caller
        (the PUT handler returns 500 with the mode unchanged; a cancelled caller
        sees its cancellation, with the runtime left cleanly "active" rather than
        wedged "draining"). Once `persist()` has succeeded (or was omitted), the
        target mode is authoritative and this runtime is committed to exiting no
        matter what happens next: finalization (rotate the epoch, degrading on
        failure to `persistence_degraded` rather than rolling back to balanced,
        then `publish()`, then stop+await the poll driver, close the store, and
        land on "disabled") runs in a separate task shielded from the caller's
        own cancellation, so it always completes even if the caller is cancelled
        partway through; the caller's cancellation (if any) is re-raised only
        once that finalization has finished.
        """
        if target_mode == "balanced":
            raise ValueError('exit_mode target_mode must not be "balanced"')
        async with self._lifecycle_lock:
            if self.status != "active":
                raise RuntimeError(
                    f"cannot exit balanced routing from status {self.status!r}"
                )
            self.status = "draining"
            self._transition_event.clear()
            try:
                await self._drain_complete.wait()
                if persist is not None:
                    persist()
            except BaseException:
                # Pre-commit: nothing balanced-only has been touched yet (and
                # persist(), if reached, never committed). Resume serving under
                # "active" with the epoch, store, and pins untouched -- this
                # also covers a cancellation delivered while still draining, so
                # the runtime is never left wedged in "draining" with the
                # transition event cleared.
                self.status = "active"
                self._transition_event.set()
                raise

            # Post-commit: persist() succeeded (or was omitted). Finalization
            # must run to completion no matter what happens to the caller from
            # here on, so it runs in its own task, shielded from the caller's
            # cancellation; a cancellation delivered to the caller only
            # surfaces again once finalization has actually finished.
            finalize_task = asyncio.create_task(self._finalize_exit(publish))
            try:
                await asyncio.shield(finalize_task)
            except asyncio.CancelledError:
                await finalize_task
                raise

    async def _finalize_exit(self, publish: Callable[[], None]) -> None:
        """Complete `exit_mode` after its settings commit.

        Rotate the epoch without rolling back on failure, publish the target mode,
        stop+await the poll driver, close the store, and reset to "disabled". Always
        run inside `asyncio.shield` by its only caller, `exit_mode`, so it completes
        exactly once `persist()` has committed, independent of the caller's own
        cancellation.
        """
        try:
            store = self._store
            assert store is not None
            try:
                await store.rotate_epoch().wait_async()
            except Exception:
                # The target mode is already durably persisted -- the commit
                # point already passed -- so a cleanup failure here must never
                # roll back to balanced; it only degrades the epoch cleanup
                # itself.
                logger.warning(
                    "balanced exit: epoch rotation failed after the target "
                    "mode was already persisted; continuing the exit with "
                    "epoch cleanup persistence_degraded",
                    exc_info=True,
                )
            publish()
        finally:
            # Cancel and await the driver before closing the store so no
            # in-flight or newly scheduled poll tick can touch it once closed.
            await self._stop_usage_poll_driver()
            if self._store is not None:
                self._store.close()
            self._store = None
            self.router = None
            self.usage_poll_coordinator = None
            self._accounts_root = None
            self.epoch_id = None
            self.epoch_seed = b""
            self.status = "disabled"
            self._transition_event.set()

    # -- process shutdown: drain and close, preserving every persisted setting --------

    async def shutdown_preserving_epoch(self) -> None:
        """Drain and close the runtime while preserving the current epoch.

        Unlike `exit_mode`, this never rotates the epoch or touches persisted
        settings, and takes no coordinator hook — so a restart's `prepare_and_publish`
        finds the SAME epoch id/seed/pins/observations/cooldowns/capability evidence
        right where this left them. A no-op when balanced routing was never prepared
        this run (`status == "disabled"`).
        """
        async with self._lifecycle_lock:
            if self.status == "disabled":
                return
            self.status = "draining"
            self._transition_event.clear()
            try:
                await self._drain_complete.wait()
            finally:
                # Use the same cancel-before-close ordering as `exit_mode`;
                # process shutdown must not race a poll tick against the store
                # it is about to close either.
                await self._stop_usage_poll_driver()
                if self._store is not None:
                    self._store.close()
                self._store = None
                self.router = None
                self.usage_poll_coordinator = None
                self._accounts_root = None
                self.epoch_id = None
                self.epoch_seed = b""
                self.status = "disabled"
                self._transition_event.set()
