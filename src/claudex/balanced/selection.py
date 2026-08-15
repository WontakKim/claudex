"""Session identity and account-selection primitives for balanced routing."""

from __future__ import annotations

import hmac
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

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
