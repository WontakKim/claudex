"""Schema constants and persisted value types for balanced routing state."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Covers the pin-key ABI as well as the table shapes: pins are keyed by a
# quota-family-salted digest (`claude_balanced_router._PIN_KEY_DOMAIN`), so
# any change to that derivation must bump this version — `open_` then
# quarantines the older database wholesale instead of restoring rows keyed
# under a different ABI into the live pin map and cold-start counts.
SCHEMA_VERSION = 2

# Step 5's mandated timings: coalesced observation/capability (and every
# other ordinary row) write flushes within one second; a pin's last_seen
# heartbeat is accepted at most once every 60 seconds.
DEFAULT_DEBOUNCE_SECONDS = 1.0
DEFAULT_PIN_LAST_SEEN_MIN_INTERVAL_SECONDS = 60.0
DEFAULT_RETRY_BACKOFF_INITIAL_SECONDS = 0.5
DEFAULT_RETRY_BACKOFF_MAX_SECONDS = 30.0

_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE pins (
  session_key_digest BLOB PRIMARY KEY, key_kind TEXT NOT NULL CHECK (key_kind IN ('uuid','content_hash')),
  account_id TEXT NOT NULL, account_incarnation_id TEXT NOT NULL,
  last_seen_utc REAL NOT NULL, expires_at_utc REAL NOT NULL,
  generation INTEGER NOT NULL CHECK (generation >= 0), balanced_epoch_id TEXT NOT NULL,
  model_family TEXT NOT NULL DEFAULT ''
) WITHOUT ROWID;
CREATE INDEX pins_by_account ON pins(account_id);
CREATE TABLE cooldowns (
  account_id TEXT NOT NULL, scope TEXT NOT NULL CHECK (scope IN ('account','family')),
  model_family TEXT NOT NULL DEFAULT '', account_incarnation_id TEXT NOT NULL,
  account_profile_fingerprint TEXT NOT NULL, deadline_utc REAL NOT NULL,
  reason TEXT NOT NULL, evidence TEXT NOT NULL, updated_at_utc REAL NOT NULL,
  PRIMARY KEY (account_id, scope, model_family)
) WITHOUT ROWID;
CREATE TABLE usage_observations (
  account_id TEXT NOT NULL, window TEXT NOT NULL CHECK (window IN ('five_hour','seven_day','fable_weekly')),
  account_incarnation_id TEXT NOT NULL, account_profile_fingerprint TEXT NOT NULL,
  used_percent REAL NOT NULL CHECK (used_percent >= 0 AND used_percent <= 100),
  reset_identity TEXT NOT NULL, reset_at_utc REAL, observed_at_utc REAL NOT NULL,
  source TEXT NOT NULL, unified_status TEXT, unified_claim TEXT,
  PRIMARY KEY (account_id, window)
) WITHOUT ROWID;
CREATE TABLE capability_evidence (
  account_id TEXT NOT NULL, capability_key TEXT NOT NULL,
  account_incarnation_id TEXT NOT NULL, account_profile_fingerprint TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('eligible','denied')),
  evidence_source TEXT NOT NULL, classifier_version TEXT NOT NULL,
  observed_at_utc REAL NOT NULL, expires_at_utc REAL,
  PRIMARY KEY (account_id, capability_key)
) WITHOUT ROWID;
"""


class UnsupportedSchemaVersionError(RuntimeError):
    """Raised by `open_` when the on-disk schema is newer than this build supports."""


# --------------------------------------------------------------------------
# Row shapes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PinRow:
    session_key_digest: bytes
    key_kind: str
    account_id: str
    account_incarnation_id: str
    last_seen_utc: float
    expires_at_utc: float
    generation: int
    balanced_epoch_id: str
    # Diagnostic metadata only — the family is already baked into the salted
    # `session_key_digest` and is never part of any key or validation.
    model_family: str = ""


@dataclass(frozen=True)
class CooldownRow:
    account_id: str
    scope: str
    model_family: str
    account_incarnation_id: str
    account_profile_fingerprint: str
    deadline_utc: float
    reason: str
    evidence: str
    updated_at_utc: float


@dataclass(frozen=True)
class UsageObservationRow:
    account_id: str
    window: str
    account_incarnation_id: str
    account_profile_fingerprint: str
    used_percent: float
    reset_identity: str
    reset_at_utc: float | None
    observed_at_utc: float
    source: str
    unified_status: str | None
    unified_claim: str | None


@dataclass(frozen=True)
class CapabilityEvidenceRow:
    account_id: str
    capability_key: str
    account_incarnation_id: str
    account_profile_fingerprint: str
    state: str
    evidence_source: str
    classifier_version: str
    observed_at_utc: float
    expires_at_utc: float | None


@dataclass(frozen=True)
class RestoreValidationContext:
    """Everything `restore` needs to judge whether a persisted row is still valid."""

    now_utc: float


@dataclass(frozen=True)
class RestoreResult:
    """The rows `restore` accepted, plus a per-invalidation-class skip count."""

    pins: dict[bytes, PinRow]
    cooldowns: dict[tuple[str, str, str], CooldownRow]
    usage_observations: dict[tuple[str, str], UsageObservationRow]
    capability_evidence: dict[tuple[str, str], CapabilityEvidenceRow]
    skip_counts: dict[str, int]


# --------------------------------------------------------------------------
# Writer-thread job bookkeeping
# --------------------------------------------------------------------------


class PendingWrite:
    """A handle to one submitted mutation's eventual commit (or drop)."""

    def __init__(self, sequence: int) -> None:
        self.sequence = sequence
        self._event = threading.Event()
        self._error: BaseException | None = None

    def _mark_done(self, error: BaseException | None = None) -> None:
        self._error = error
        self._event.set()

    def wait(self, timeout: float | None = None) -> None:
        """Block the calling thread until this write has committed or dropped.

        Raises `TimeoutError` if `timeout` elapses first, or re-raises a
        non-retryable error (e.g. a constraint violation) the write hit.
        """
        if not self._event.wait(timeout):
            raise TimeoutError(f"persistence sequence {self.sequence} did not complete in time")
        if self._error is not None:
            raise self._error

    async def wait_async(self, timeout: float | None = None) -> None:
        """Await completion without blocking the event loop.

        Delegates the blocking wait to the default executor thread pool, so
        a caller can `await` a high-priority write's completion from async
        code while the writer thread does its work off the event loop.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.wait, timeout)


@dataclass
class _Job:
    sequence: int
    """This row's earliest queue position; coalescing never changes it."""
    payload_sequence: int
    """The most recently submitted payload's own sequence; used for fencing."""
    scope_key: tuple[Any, ...] | None
    priority: bool
    flush_after: float
    apply: Callable[[sqlite3.Connection], None]
    pending_writes: list[PendingWrite]
    barrier: tuple[Any, ...] | None = None
    fence: tuple[tuple[Any, ...], ...] = ()
    attempts: int = 0
    next_attempt_at: float | None = None
