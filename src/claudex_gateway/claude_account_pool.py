"""Cooldown tracking and ordered fallback for registered Claude accounts.

In fallback mode, every request first uses the admin-selected serving
account. When that account is rate-limited, the gateway cools it down in
memory and tries the next ready account in registration order, failing back
automatically once the cooldown expires. Balanced session-affinity routing
is owned by ``balanced.router`` and does not use this tracker for eligibility.

Cooldowns in this module are process-local; the registry stores durable facts
only. A cooldown deadline is derived from the best signal a 429 provides.
OAuth quota rejections commonly omit Retry-After, rate-limit headers, and a
body reset timestamp, so the cached usage envelope is the most accurate
fallback. A short default keeps the pool self-healing when no reset signal is
available.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from claudex_gateway.claude_accounts import AccountRecord
from claudex_gateway.usage import _reset_epoch_seconds

logger = logging.getLogger(__name__)

_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 60.0
_COOLDOWN_MIN_SECONDS = 5.0
_COOLDOWN_MAX_SECONDS = 7 * 24 * 3600.0
# Observed only in API-key documentation, never on OAuth quota 429s — parsed
# tolerantly in case it appears on RPM-style rejections.
_RESET_HEADER = "anthropic-ratelimit-unified-reset"
_BODY_RESET_KEYS = ("resets_at", "reset_at", "reset")
_BODY_WALK_MAX_DEPTH = 6
_USAGE_WINDOW_KEYS = ("session", "weekly", "fable_weekly")
_USAGE_EXHAUSTED_PERCENT = 99.0


class AccountCooldownTracker:
    """In-memory per-account cooldown deadlines on monotonic AND wall clocks.

    Each mark records one deadline per clock, and the cooldown expires when
    EITHER passes. The monotonic deadline bounds the cooldown against wall-
    clock jumps; the wall deadline bounds it against the monotonic clock
    pausing while the machine sleeps (macOS `time.monotonic` does not advance
    during sleep, so a monotonic-only deadline overshoots the real reset by
    however long the machine slept).

    Expired deadlines are pruned lazily, which is also what makes fail-back
    automatic: an expired entry simply stops excluding its account from the
    serving chain. No lock: callers run on one event loop and no method
    awaits between reading and writing the deadline map.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._clock = clock
        self._wall_clock = wall_clock
        self._deadlines: dict[str, tuple[float, float]] = {}

    def mark(self, account_id: str, seconds: float) -> None:
        """Start (or replace) a cooldown of `seconds` for `account_id`."""
        duration = max(0.0, seconds)
        self._deadlines[account_id] = (self._clock() + duration, self._wall_clock() + duration)

    def is_cooling(self, account_id: str) -> bool:
        return self.remaining_seconds(account_id) > 0.0

    def remaining_seconds(self, account_id: str) -> float:
        deadlines = self._deadlines.get(account_id)
        if deadlines is None:
            return 0.0
        deadline_monotonic, deadline_wall = deadlines
        remaining = min(deadline_monotonic - self._clock(), deadline_wall - self._wall_clock())
        if remaining <= 0.0:
            del self._deadlines[account_id]
            return 0.0
        return remaining

    def clear(self, account_id: str) -> None:
        """Drop any cooldown for `account_id` (e.g. the account was removed)."""
        self._deadlines.pop(account_id, None)

    def min_remaining_seconds(self) -> float | None:
        """The shortest active cooldown, or None when nothing is cooling."""
        remaining = [self.remaining_seconds(account_id) for account_id in list(self._deadlines)]
        active = [seconds for seconds in remaining if seconds > 0.0]
        return min(active) if active else None


@dataclass(frozen=True)
class ServingChain:
    """The attempt order for one request, plus why it may be empty.

    `serving_registered`/`serving_state` describe the admin-selected account
    so the caller can reproduce the exact single-account error responses when
    no attempt is possible; `cooling_ids` distinguishes "everything is rate-
    limited right now" (a 429 with Retry-After) from "nothing is usable at
    all" (a 503).
    """

    attempts: tuple[AccountRecord, ...]
    cooling_ids: tuple[str, ...]
    serving_registered: bool
    serving_state: str | None


def build_serving_chain(
    serving_account_id: str,
    records: Sequence[AccountRecord],
    tracker: AccountCooldownTracker,
) -> ServingChain:
    """Order the usable accounts: serving account first, rest by (createdAt, id).

    needs-reauth rows are excluded entirely (only a human re-login recovers
    them); ready-but-cooling rows are excluded from `attempts` but reported
    in `cooling_ids`.
    """
    serving_record = next((record for record in records if record.id == serving_account_id), None)
    attempts: list[AccountRecord] = []
    cooling_ids: list[str] = []

    def _place(record: AccountRecord) -> None:
        if record.state != "ready":
            return
        if tracker.is_cooling(record.id):
            cooling_ids.append(record.id)
        else:
            attempts.append(record)

    if serving_record is not None:
        _place(serving_record)
    for record in sorted(records, key=lambda record: (record.created_at, record.id)):
        if record.id != serving_account_id:
            _place(record)

    return ServingChain(
        attempts=tuple(attempts),
        cooling_ids=tuple(cooling_ids),
        serving_registered=serving_record is not None,
        serving_state=serving_record.state if serving_record is not None else None,
    )


def rate_limit_cooldown_seconds(
    headers: Mapping[str, str],
    body: bytes,
    usage_envelope: dict[str, Any] | None,
    *,
    wall_clock: Callable[[], float] = time.time,
) -> float:
    """Derive how long a 429'd account should sit out; never raises.

    First successfully parsed signal wins: ① Retry-After header → ② ratelimit
    reset header → ③ a reset timestamp anywhere in the error body → ④ the
    cached usage envelope (earliest reset among exhausted windows) → ⑤ the
    default. Under-cooling is safe — a premature retry just re-429s and
    re-marks — so the result is clamped rather than second-guessed.
    """
    now = wall_clock()
    lowered = {key.lower(): value for key, value in headers.items()}

    seconds = _parse_retry_after(lowered.get("retry-after"), now)
    if seconds is None:
        seconds = _epoch_delta(_reset_epoch_seconds(lowered.get(_RESET_HEADER)), now)
    if seconds is None:
        seconds = _epoch_delta(_body_reset_epoch(body), now)
    if seconds is None:
        seconds = _usage_reset_delta(usage_envelope, now)
    if seconds is None:
        seconds = _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    return min(_COOLDOWN_MAX_SECONDS, max(_COOLDOWN_MIN_SECONDS, seconds))


def _parse_retry_after(raw: str | None, now: float) -> float | None:
    """Delta-seconds or HTTP-date, mirroring `usage._retry_after_seconds`."""
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).timestamp() - now
    except (TypeError, ValueError):
        return None


def _epoch_delta(epoch_seconds: float | None, now: float) -> float | None:
    if epoch_seconds is None:
        return None
    delta = epoch_seconds - now
    return delta if delta > 0.0 else None


def _body_reset_epoch(body: bytes) -> float | None:
    """Find the first normalizable reset timestamp anywhere in a JSON body."""
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return _walk_for_reset(parsed, _BODY_WALK_MAX_DEPTH)


def _walk_for_reset(node: Any, depth: int) -> float | None:
    if depth <= 0:
        return None
    if isinstance(node, dict):
        for key in _BODY_RESET_KEYS:
            epoch = _reset_epoch_seconds(node.get(key))
            if epoch is not None:
                return epoch
        for value in node.values():
            epoch = _walk_for_reset(value, depth - 1)
            if epoch is not None:
                return epoch
    elif isinstance(node, list):
        for value in node:
            epoch = _walk_for_reset(value, depth - 1)
            if epoch is not None:
                return epoch
    return None


def _usage_reset_delta(usage_envelope: dict[str, Any] | None, now: float) -> float | None:
    """Earliest future reset among the envelope's exhausted windows.

    The min is deliberate: under-cooling self-heals (see caller), while the
    max could park an account far past the window that actually 429'd.
    """
    if not isinstance(usage_envelope, dict):
        return None
    deltas: list[float] = []
    for key in _USAGE_WINDOW_KEYS:
        window = usage_envelope.get(key)
        if not isinstance(window, dict):
            continue
        used_percent = window.get("used_percent")
        if not isinstance(used_percent, (int, float)) or isinstance(used_percent, bool):
            continue
        if used_percent < _USAGE_EXHAUSTED_PERCENT:
            continue
        delta = _epoch_delta(_reset_epoch_seconds(window.get("resets_at")), now)
        if delta is not None:
            deltas.append(delta)
    return min(deltas) if deltas else None
