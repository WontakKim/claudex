"""SQLite-backed runtime-state store for balanced Claude account routing.

One database, at the path the caller supplies (production uses
``paths.claude_account_pool_runtime_db()``), holds every piece of state the
balanced router needs to survive a restart —
session pins, account cooldowns, usage observations, and capability
evidence — behind a single serialized writer thread. The database runs in
WAL journal mode with ``synchronous=FULL`` (a commit is not acknowledged
until it is durable on disk), lives in a 0700 directory, and is itself mode
0600.

``ClaudePoolRuntimeStateStore.open_`` gates the on-disk schema version
before touching anything mutable: a newer schema than this build knows is
refused with the file left byte-identical, while a corrupt or unsupported
older store (together with its ``-wal``/``-shm`` siblings) is quarantined
and replaced with a fresh store and a fresh epoch seed.

Every mutation is submitted to one background writer thread and receives a
strictly increasing persistence sequence number. Repeated submissions for
the same logical row coalesce — only the latest payload is written — but
the row keeps the *earliest* queue position it was ever assigned, so a
constantly-refreshed row cannot starve behind newer arrivals forever.
Ordinary submissions debounce for ``DEFAULT_DEBOUNCE_SECONDS`` to allow
coalescing; a high-priority submission (epoch invalidation, incarnation
deletion, and pin upserts submitted with ``high_priority=True`` — the pin
migration case) cancels every pending job's debounce and forces the queue
to flush, in sequence order, before it commits itself. It never commits
ahead of an earlier, still-pending job.

A write that fails with a transient SQLite error (locking, I/O) is retried
with exponential backoff on a side track that does not block the queue —
``persistence_degraded`` is true while any retry is outstanding — so one
stuck write cannot indefinitely delay a later high-priority operation. To
keep a resurrected retry from ever undoing that later operation's effect,
every row-scoped write is *fenced*: at the moment it is finally about to
execute (fresh attempt or retry), it is dropped as a no-op if a
higher-sequence epoch invalidation or incarnation deletion for its scope
has already committed. Final state therefore depends on persistence
sequence order, never on wall-clock retry timing.
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from claudex.balanced.state_model import (
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_PIN_LAST_SEEN_MIN_INTERVAL_SECONDS,
    DEFAULT_RETRY_BACKOFF_INITIAL_SECONDS,
    DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
    SCHEMA_VERSION,
    _SCHEMA_SQL,
    CapabilityEvidenceRow,
    CooldownRow,
    PendingWrite,
    PinRow,
    RestoreResult,
    RestoreValidationContext,
    UnsupportedSchemaVersionError,
    UsageObservationRow,
    _Job,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Filesystem-level open helpers (run before any writer-thread state exists)
# --------------------------------------------------------------------------


def _read_schema_version_readonly(path: Path) -> int | None:
    """Read `meta.schema_version` through a read-only connection, or None.

    None covers both "no schema_version row" and "not a readable SQLite
    database at all" (corrupt file, foreign file, wrong table shape) — every
    case Step 2 treats as "quarantine and replace". Opened read-only so a
    refusal path (newer schema) never has the chance to write a byte.
    """
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _quarantine_existing(path: Path) -> None:
    """Rename `path` and its `-wal`/`-shm` siblings aside, if any exist."""
    siblings = [path.with_name(path.name + suffix) for suffix in ("", "-wal", "-shm")]
    if not any(sibling.exists() for sibling in siblings):
        return
    stamp = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    for sibling in siblings:
        if sibling.exists():
            sibling.rename(sibling.with_name(sibling.name + f".quarantined-{stamp}"))


def _seed_meta(conn: sqlite3.Connection, *, epoch_id: str, epoch_seed_hex: str) -> None:
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [
            ("schema_version", str(SCHEMA_VERSION)),
            ("balanced_epoch_id", epoch_id),
            ("epoch_seed_hex", epoch_seed_hex),
            ("epoch_active", "0"),
        ],
    )


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


class ClaudePoolRuntimeStateStore:
    """Owns one SQLite database and its sole serialized writer thread."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        pin_last_seen_min_interval_seconds: float = DEFAULT_PIN_LAST_SEEN_MIN_INTERVAL_SECONDS,
        retry_backoff_initial_seconds: float = DEFAULT_RETRY_BACKOFF_INITIAL_SECONDS,
        retry_backoff_max_seconds: float = DEFAULT_RETRY_BACKOFF_MAX_SECONDS,
        fault_injector: Callable[[tuple[Any, ...] | None, int], None] | None = None,
    ) -> None:
        self._conn = conn
        self._path = path
        self._clock = clock
        self._debounce_seconds = debounce_seconds
        self._pin_last_seen_min_interval_seconds = pin_last_seen_min_interval_seconds
        self._retry_backoff_initial_seconds = retry_backoff_initial_seconds
        self._retry_backoff_max_seconds = retry_backoff_max_seconds
        # Test-only seam: called before every write attempt (fresh or
        # retried). Production callers must never set this. A raised
        # exception is treated exactly like a transient SQLite failure.
        self._fault_injector = fault_injector

        self._cv = threading.Condition()
        self._pending: list[_Job] = []
        self._retrying: list[_Job] = []
        self._committed_barriers: dict[tuple[Any, ...], int] = {}
        self._next_sequence = 1
        self._closed = False
        self._stopping = False
        self._persistence_degraded = False

        self._meta_lock = threading.Lock()
        self._balanced_epoch_id = ""
        self._epoch_seed = b""
        self._epoch_active = False
        self._schema_version = SCHEMA_VERSION

        self._pin_touch_lock = threading.Lock()
        self._last_pin_touch_accepted: dict[bytes, float] = {}

        self._writer_thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open_(cls, path: Path, **kwargs: Any) -> ClaudePoolRuntimeStateStore:
        """Open (or create) the runtime-state store at `path`.

        Refuses (raising `UnsupportedSchemaVersionError`, file untouched) a
        database whose `schema_version` is newer than `SCHEMA_VERSION`. A
        corrupt or older-schema database is quarantined together with its
        `-wal`/`-shm` siblings and replaced by a fresh store with a fresh
        epoch seed. Every reopen enforces 0700/0600 permissions and WAL +
        `synchronous=FULL`.
        """
        path = Path(path)
        needs_fresh = True
        if path.exists():
            version = _read_schema_version_readonly(path)
            if version is not None and version > SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"runtime-state database at {path} has schema_version={version}, "
                    f"newer than the {SCHEMA_VERSION} this build supports; refusing to touch it"
                )
            needs_fresh = version != SCHEMA_VERSION

        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)  # defense in depth against a permissive umask
        if needs_fresh:
            _quarantine_existing(path)
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        os.chmod(path, 0o600)  # defense in depth against a permissive umask

        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        if needs_fresh:
            conn.executescript(_SCHEMA_SQL)
            epoch_seed = secrets.token_bytes(32)
            _seed_meta(conn, epoch_id=str(uuid.uuid4()), epoch_seed_hex=epoch_seed.hex())

        store = cls(conn, path, **kwargs)
        store._load_epoch_from_meta()
        store._start_writer_thread()
        return store

    def _load_epoch_from_meta(self) -> None:
        rows = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        with self._meta_lock:
            self._balanced_epoch_id = rows["balanced_epoch_id"]
            self._epoch_seed = bytes.fromhex(rows["epoch_seed_hex"])
            self._epoch_active = rows.get("epoch_active") == "1"
            self._schema_version = int(rows["schema_version"])

    def _start_writer_thread(self) -> None:
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="claude-pool-runtime-state-writer", daemon=True
        )
        self._writer_thread.start()

    @property
    def writer_thread_ident(self) -> int | None:
        return self._writer_thread.ident if self._writer_thread is not None else None

    def close(self) -> None:
        """Reject new submissions, drain everything accepted, checkpoint, stop.

        Idempotent — a second call is a no-op.
        """
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._stopping = True
            now = self._clock()
            for job in self._pending:
                job.flush_after = min(job.flush_after, now)
            self._cv.notify_all()
        if self._writer_thread is not None:
            self._writer_thread.join()

    # -- epoch state ---------------------------------------------------------

    @property
    def balanced_epoch_id(self) -> str:
        with self._meta_lock:
            return self._balanced_epoch_id

    @property
    def epoch_seed(self) -> bytes:
        with self._meta_lock:
            return self._epoch_seed

    @property
    def epoch_active(self) -> bool:
        with self._meta_lock:
            return self._epoch_active

    @property
    def schema_version(self) -> int:
        return self._schema_version

    @property
    def persistence_degraded(self) -> bool:
        return self._persistence_degraded

    def rotate_epoch(self) -> PendingWrite:
        """Mint a fresh epoch id and seed, and wipe every existing pin.

        Always high priority: a rotated epoch invalidates every pin's
        session affinity immediately, so this must flush ahead of nothing
        and be visible to any pin write already in flight (Step 9's causal
        rule for epoch invalidation).
        """
        new_epoch_id = str(uuid.uuid4())
        new_seed = secrets.token_bytes(32)

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute("UPDATE meta SET value = ? WHERE key = 'balanced_epoch_id'", (new_epoch_id,))
            conn.execute("UPDATE meta SET value = ? WHERE key = 'epoch_seed_hex'", (new_seed.hex(),))
            conn.execute("DELETE FROM pins")
            with self._meta_lock:
                self._balanced_epoch_id = new_epoch_id
                self._epoch_seed = new_seed

        return self._submit(scope_key=None, apply=apply, high_priority=True, barrier=("epoch",))

    def invalidate_epoch_pins(self) -> PendingWrite:
        """Delete every pin whose `balanced_epoch_id` no longer matches current.

        A standalone cleanup primitive (also the basis `rotate_epoch` builds
        on): safe to call any time pins may reference a superseded epoch.
        Always high priority per Step 9's causal rule.
        """

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM pins WHERE balanced_epoch_id != ?", (self.balanced_epoch_id,))

        return self._submit(scope_key=None, apply=apply, high_priority=True, barrier=("epoch",))

    # -- pins ------------------------------------------------------------

    def upsert_pin(
        self,
        *,
        session_key_digest: bytes,
        key_kind: str,
        account_id: str,
        account_incarnation_id: str,
        last_seen_utc: float,
        expires_at_utc: float,
        generation: int,
        balanced_epoch_id: str,
        model_family: str = "",
        high_priority: bool = False,
    ) -> PendingWrite:
        if key_kind not in ("uuid", "content_hash"):
            raise ValueError(f"invalid pin key_kind: {key_kind!r}")
        if generation < 0:
            raise ValueError(f"invalid pin generation: {generation!r}")

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO pins (session_key_digest, key_kind, account_id, account_incarnation_id,
                                   last_seen_utc, expires_at_utc, generation, balanced_epoch_id,
                                   model_family)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_key_digest) DO UPDATE SET
                    key_kind = excluded.key_kind,
                    account_id = excluded.account_id,
                    account_incarnation_id = excluded.account_incarnation_id,
                    last_seen_utc = excluded.last_seen_utc,
                    expires_at_utc = excluded.expires_at_utc,
                    generation = excluded.generation,
                    balanced_epoch_id = excluded.balanced_epoch_id,
                    model_family = excluded.model_family
                """,
                (
                    session_key_digest,
                    key_kind,
                    account_id,
                    account_incarnation_id,
                    last_seen_utc,
                    expires_at_utc,
                    generation,
                    balanced_epoch_id,
                    model_family,
                ),
            )

        return self._submit(
            scope_key=("pins", session_key_digest),
            apply=apply,
            high_priority=high_priority,
            fence=(("epoch",), ("incarnation", account_incarnation_id)),
        )

    def touch_pin_last_seen(
        self, session_key_digest: bytes, last_seen_utc: float, expires_at_utc: float
    ) -> PendingWrite | None:
        """Refresh a pin's activity timestamps, throttled to once per pin per interval.

        Returns None (no submission at all) when called again for the same
        pin within `pin_last_seen_min_interval_seconds` of the last accepted
        call — Step 5's 60-second-per-pin throttle.
        """
        with self._pin_touch_lock:
            now = self._clock()
            last_accepted = self._last_pin_touch_accepted.get(session_key_digest)
            if last_accepted is not None and now - last_accepted < self._pin_last_seen_min_interval_seconds:
                return None
            self._last_pin_touch_accepted[session_key_digest] = now

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE pins SET last_seen_utc = ?, expires_at_utc = ? WHERE session_key_digest = ?",
                (last_seen_utc, expires_at_utc, session_key_digest),
            )

        # A distinct scope from ("pins", digest): the partial row carries only
        # last_seen_utc and expires_at_utc, so it must remain disjoint from the
        # full upsert scope to prevent either payload from clobbering the other.
        return self._submit(
            scope_key=("pins_last_seen", session_key_digest),
            apply=apply,
            high_priority=False,
            fence=(("epoch",),),
        )

    def delete_pin(self, session_key_digest: bytes, *, high_priority: bool = False) -> PendingWrite:
        def apply(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM pins WHERE session_key_digest = ?", (session_key_digest,))

        return self._submit(scope_key=("pins", session_key_digest), apply=apply, high_priority=high_priority)

    def get_pin(self, session_key_digest: bytes) -> PinRow | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT session_key_digest, key_kind, account_id, account_incarnation_id, "
                "last_seen_utc, expires_at_utc, generation, balanced_epoch_id, model_family "
                "FROM pins WHERE session_key_digest = ?",
                (session_key_digest,),
            ).fetchone()
        return PinRow(*row) if row is not None else None

    def pin_count(self) -> int:
        with self._read_connection() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM pins").fetchone()
        return count

    # -- cooldowns ---------------------------------------------------------

    def upsert_cooldown(
        self,
        *,
        account_id: str,
        scope: str,
        model_family: str,
        account_incarnation_id: str,
        account_profile_fingerprint: str,
        deadline_utc: float,
        reason: str,
        evidence: str,
        updated_at_utc: float,
        high_priority: bool = False,
    ) -> PendingWrite:
        if scope not in ("account", "family"):
            raise ValueError(f"invalid cooldown scope: {scope!r}")

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO cooldowns (account_id, scope, model_family, account_incarnation_id,
                                        account_profile_fingerprint, deadline_utc, reason, evidence, updated_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (account_id, scope, model_family) DO UPDATE SET
                    account_incarnation_id = excluded.account_incarnation_id,
                    account_profile_fingerprint = excluded.account_profile_fingerprint,
                    deadline_utc = excluded.deadline_utc,
                    reason = excluded.reason,
                    evidence = excluded.evidence,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    account_id,
                    scope,
                    model_family,
                    account_incarnation_id,
                    account_profile_fingerprint,
                    deadline_utc,
                    reason,
                    evidence,
                    updated_at_utc,
                ),
            )

        return self._submit(
            scope_key=("cooldowns", account_id, scope, model_family),
            apply=apply,
            high_priority=high_priority,
            fence=(("incarnation", account_incarnation_id),),
        )

    def delete_cooldown(
        self, account_id: str, scope: str, model_family: str, *, high_priority: bool = False
    ) -> PendingWrite:
        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM cooldowns WHERE account_id = ? AND scope = ? AND model_family = ?",
                (account_id, scope, model_family),
            )

        return self._submit(
            scope_key=("cooldowns", account_id, scope, model_family), apply=apply, high_priority=high_priority
        )

    def get_cooldown(self, account_id: str, scope: str, model_family: str) -> CooldownRow | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT account_id, scope, model_family, account_incarnation_id, "
                "account_profile_fingerprint, deadline_utc, reason, evidence, updated_at_utc "
                "FROM cooldowns WHERE account_id = ? AND scope = ? AND model_family = ?",
                (account_id, scope, model_family),
            ).fetchone()
        return CooldownRow(*row) if row is not None else None

    # -- usage observations --------------------------------------------------

    def upsert_usage_observation(
        self,
        *,
        account_id: str,
        window: str,
        account_incarnation_id: str,
        account_profile_fingerprint: str,
        used_percent: float,
        reset_identity: str,
        reset_at_utc: float | None,
        observed_at_utc: float,
        source: str,
        unified_status: str | None = None,
        unified_claim: str | None = None,
        high_priority: bool = False,
    ) -> PendingWrite:
        if window not in ("five_hour", "seven_day", "fable_weekly"):
            raise ValueError(f"invalid usage window: {window!r}")
        if not 0 <= used_percent <= 100:
            raise ValueError(f"used_percent out of range: {used_percent!r}")

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO usage_observations (account_id, window, account_incarnation_id,
                    account_profile_fingerprint, used_percent, reset_identity, reset_at_utc,
                    observed_at_utc, source, unified_status, unified_claim)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (account_id, window) DO UPDATE SET
                    account_incarnation_id = excluded.account_incarnation_id,
                    account_profile_fingerprint = excluded.account_profile_fingerprint,
                    used_percent = excluded.used_percent,
                    reset_identity = excluded.reset_identity,
                    reset_at_utc = excluded.reset_at_utc,
                    observed_at_utc = excluded.observed_at_utc,
                    source = excluded.source,
                    unified_status = excluded.unified_status,
                    unified_claim = excluded.unified_claim
                """,
                (
                    account_id,
                    window,
                    account_incarnation_id,
                    account_profile_fingerprint,
                    used_percent,
                    reset_identity,
                    reset_at_utc,
                    observed_at_utc,
                    source,
                    unified_status,
                    unified_claim,
                ),
            )

        return self._submit(
            scope_key=("usage_observations", account_id, window),
            apply=apply,
            high_priority=high_priority,
            fence=(("incarnation", account_incarnation_id),),
        )

    def delete_usage_observation(self, account_id: str, window: str, *, high_priority: bool = False) -> PendingWrite:
        def apply(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM usage_observations WHERE account_id = ? AND window = ?", (account_id, window))

        return self._submit(
            scope_key=("usage_observations", account_id, window), apply=apply, high_priority=high_priority
        )

    def get_usage_observation(self, account_id: str, window: str) -> UsageObservationRow | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT account_id, window, account_incarnation_id, account_profile_fingerprint, "
                "used_percent, reset_identity, reset_at_utc, observed_at_utc, source, "
                "unified_status, unified_claim FROM usage_observations WHERE account_id = ? AND window = ?",
                (account_id, window),
            ).fetchone()
        return UsageObservationRow(*row) if row is not None else None

    # -- capability evidence --------------------------------------------------

    def upsert_capability_evidence(
        self,
        *,
        account_id: str,
        capability_key: str,
        account_incarnation_id: str,
        account_profile_fingerprint: str,
        state: str,
        evidence_source: str,
        classifier_version: str,
        observed_at_utc: float,
        expires_at_utc: float | None = None,
        high_priority: bool = False,
    ) -> PendingWrite:
        if state not in ("eligible", "denied"):
            raise ValueError(f"invalid capability state: {state!r}")

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO capability_evidence (account_id, capability_key, account_incarnation_id,
                    account_profile_fingerprint, state, evidence_source, classifier_version,
                    observed_at_utc, expires_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (account_id, capability_key) DO UPDATE SET
                    account_incarnation_id = excluded.account_incarnation_id,
                    account_profile_fingerprint = excluded.account_profile_fingerprint,
                    state = excluded.state,
                    evidence_source = excluded.evidence_source,
                    classifier_version = excluded.classifier_version,
                    observed_at_utc = excluded.observed_at_utc,
                    expires_at_utc = excluded.expires_at_utc
                """,
                (
                    account_id,
                    capability_key,
                    account_incarnation_id,
                    account_profile_fingerprint,
                    state,
                    evidence_source,
                    classifier_version,
                    observed_at_utc,
                    expires_at_utc,
                ),
            )

        return self._submit(
            scope_key=("capability_evidence", account_id, capability_key),
            apply=apply,
            high_priority=high_priority,
            fence=(("incarnation", account_incarnation_id),),
        )

    def delete_capability_evidence(
        self, account_id: str, capability_key: str, *, high_priority: bool = False
    ) -> PendingWrite:
        def apply(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM capability_evidence WHERE account_id = ? AND capability_key = ?",
                (account_id, capability_key),
            )

        return self._submit(
            scope_key=("capability_evidence", account_id, capability_key), apply=apply, high_priority=high_priority
        )

    def get_capability_evidence(self, account_id: str, capability_key: str) -> CapabilityEvidenceRow | None:
        with self._read_connection() as conn:
            row = conn.execute(
                "SELECT account_id, capability_key, account_incarnation_id, account_profile_fingerprint, "
                "state, evidence_source, classifier_version, observed_at_utc, expires_at_utc "
                "FROM capability_evidence WHERE account_id = ? AND capability_key = ?",
                (account_id, capability_key),
            ).fetchone()
        return CapabilityEvidenceRow(*row) if row is not None else None

    # -- incarnation-scoped deletion -----------------------------------------

    def delete_all_for_incarnation(self, account_incarnation_id: str) -> PendingWrite:
        """Delete every row tied to `account_incarnation_id`, across all four tables.

        Always high priority; registers an incarnation barrier so a
        still-pending or retrying write for this incarnation can never
        resurrect a row afterward (Step 9's causal rule for incarnation
        deletion).
        """

        def apply(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM pins WHERE account_incarnation_id = ?", (account_incarnation_id,))
            conn.execute("DELETE FROM cooldowns WHERE account_incarnation_id = ?", (account_incarnation_id,))
            conn.execute("DELETE FROM usage_observations WHERE account_incarnation_id = ?", (account_incarnation_id,))
            conn.execute(
                "DELETE FROM capability_evidence WHERE account_incarnation_id = ?", (account_incarnation_id,)
            )

        return self._submit(
            scope_key=None, apply=apply, high_priority=True, barrier=("incarnation", account_incarnation_id)
        )

    # -- restore -------------------------------------------------------------

    def restore(self, validation_context: RestoreValidationContext) -> RestoreResult:
        """Load every row, skip-and-delete anything validation rejects.

        Runs as a single high-priority job on the writer thread (so it
        still goes through the sole serialized writer) and blocks the
        caller until it finishes — meant to be called once at startup,
        before ordinary traffic starts submitting writes.
        """
        box: dict[str, RestoreResult] = {}

        def apply(conn: sqlite3.Connection) -> None:
            box["result"] = self._do_restore(conn, validation_context)

        pending_write = self._submit(scope_key=None, apply=apply, high_priority=True)
        pending_write.wait()
        return box["result"]

    def _do_restore(self, conn: sqlite3.Connection, ctx: RestoreValidationContext) -> RestoreResult:
        skip_counts: Counter[str] = Counter()
        current_epoch = self.balanced_epoch_id

        pins: dict[bytes, PinRow] = {}
        stale_pin_digests: list[bytes] = []
        for row in conn.execute(
            "SELECT session_key_digest, key_kind, account_id, account_incarnation_id, "
            "last_seen_utc, expires_at_utc, generation, balanced_epoch_id, model_family FROM pins"
        ):
            pin = PinRow(*row)
            if pin.expires_at_utc <= ctx.now_utc:
                skip_counts["pins.expired"] += 1
                stale_pin_digests.append(pin.session_key_digest)
            elif pin.balanced_epoch_id != current_epoch:
                skip_counts["pins.epoch_mismatch"] += 1
                stale_pin_digests.append(pin.session_key_digest)
            else:
                pins[pin.session_key_digest] = pin
        for digest in stale_pin_digests:
            conn.execute("DELETE FROM pins WHERE session_key_digest = ?", (digest,))

        cooldowns: dict[tuple[str, str, str], CooldownRow] = {}
        stale_cooldown_keys: list[tuple[str, str, str]] = []
        for row in conn.execute(
            "SELECT account_id, scope, model_family, account_incarnation_id, "
            "account_profile_fingerprint, deadline_utc, reason, evidence, updated_at_utc FROM cooldowns"
        ):
            cooldown = CooldownRow(*row)
            key = (cooldown.account_id, cooldown.scope, cooldown.model_family)
            if cooldown.deadline_utc <= ctx.now_utc:
                skip_counts["cooldowns.expired"] += 1
                stale_cooldown_keys.append(key)
            else:
                cooldowns[key] = cooldown
        for account_id, scope, model_family in stale_cooldown_keys:
            conn.execute(
                "DELETE FROM cooldowns WHERE account_id = ? AND scope = ? AND model_family = ?",
                (account_id, scope, model_family),
            )

        usage_observations: dict[tuple[str, str], UsageObservationRow] = {}
        stale_observation_keys: list[tuple[str, str]] = []
        for row in conn.execute(
            "SELECT account_id, window, account_incarnation_id, account_profile_fingerprint, "
            "used_percent, reset_identity, reset_at_utc, observed_at_utc, source, "
            "unified_status, unified_claim FROM usage_observations"
        ):
            observation = UsageObservationRow(*row)
            key = (observation.account_id, observation.window)
            if observation.reset_at_utc is not None and observation.reset_at_utc <= ctx.now_utc:
                skip_counts["usage_observations.stale_reset"] += 1
                stale_observation_keys.append(key)
            else:
                usage_observations[key] = observation
        for account_id, window in stale_observation_keys:
            conn.execute("DELETE FROM usage_observations WHERE account_id = ? AND window = ?", (account_id, window))

        capability_evidence: dict[tuple[str, str], CapabilityEvidenceRow] = {}
        stale_capability_keys: list[tuple[str, str]] = []
        for row in conn.execute(
            "SELECT account_id, capability_key, account_incarnation_id, account_profile_fingerprint, "
            "state, evidence_source, classifier_version, observed_at_utc, expires_at_utc FROM capability_evidence"
        ):
            capability = CapabilityEvidenceRow(*row)
            key = (capability.account_id, capability.capability_key)
            if capability.expires_at_utc is not None and capability.expires_at_utc <= ctx.now_utc:
                skip_counts["capability_evidence.expired"] += 1
                stale_capability_keys.append(key)
            else:
                capability_evidence[key] = capability
        for account_id, capability_key in stale_capability_keys:
            conn.execute(
                "DELETE FROM capability_evidence WHERE account_id = ? AND capability_key = ?",
                (account_id, capability_key),
            )

        return RestoreResult(
            pins=pins,
            cooldowns=cooldowns,
            usage_observations=usage_observations,
            capability_evidence=capability_evidence,
            skip_counts=dict(skip_counts),
        )

    # -- reads ----------------------------------------------------------------

    @contextmanager
    def _read_connection(self):
        if self._closed:
            raise RuntimeError("ClaudePoolRuntimeStateStore is closed")
        conn = sqlite3.connect(str(self._path))
        try:
            yield conn
        finally:
            conn.close()

    # -- submission / writer thread --------------------------------------------

    def _submit(
        self,
        *,
        scope_key: tuple[Any, ...] | None,
        apply: Callable[[sqlite3.Connection], None],
        high_priority: bool,
        barrier: tuple[Any, ...] | None = None,
        fence: tuple[tuple[Any, ...], ...] = (),
    ) -> PendingWrite:
        with self._cv:
            if self._closed:
                raise RuntimeError("ClaudePoolRuntimeStateStore is closed; no new writes are accepted")
            sequence = self._next_sequence
            self._next_sequence += 1
            now = self._clock()
            pending_write = PendingWrite(sequence)

            existing = None
            if scope_key is not None:
                existing = next((job for job in self._pending if job.scope_key == scope_key), None)

            if existing is not None:
                # Coalesce: the latest payload wins, but the row keeps the
                # earliest queue position it was ever assigned.
                existing.apply = apply
                existing.payload_sequence = sequence
                existing.priority = existing.priority or high_priority
                existing.pending_writes.append(pending_write)
                if barrier is not None:
                    existing.barrier = barrier
                if fence:
                    existing.fence = fence
            else:
                flush_after = now if high_priority else now + self._debounce_seconds
                self._pending.append(
                    _Job(
                        sequence=sequence,
                        payload_sequence=sequence,
                        scope_key=scope_key,
                        priority=high_priority,
                        flush_after=flush_after,
                        apply=apply,
                        pending_writes=[pending_write],
                        barrier=barrier,
                        fence=fence,
                    )
                )

            if high_priority:
                # Cancel debounce for every pending job, not just this one:
                # a high-priority submission flushes everything queued
                # ahead of it, in sequence order, before it commits itself.
                for job in self._pending:
                    job.flush_after = min(job.flush_after, now)

            self._cv.notify_all()
            return pending_write

    def _writer_loop(self) -> None:
        while True:
            job, is_retry = self._wait_for_next_job()
            if job is None:
                break
            self._attempt(job, is_retry)
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            self._conn.close()

    def _wait_for_next_job(self) -> tuple[_Job | None, bool]:
        with self._cv:
            while True:
                now = self._clock()
                if self._pending and self._pending[0].flush_after <= now:
                    return self._pending.pop(0), False
                retry_index = next(
                    (index for index, job in enumerate(self._retrying) if job.next_attempt_at is not None and job.next_attempt_at <= now),
                    None,
                )
                if retry_index is not None:
                    return self._retrying.pop(retry_index), True
                if self._stopping and not self._pending and not self._retrying:
                    return None, False
                deadlines: list[float] = []
                if self._pending:
                    deadlines.append(self._pending[0].flush_after)
                deadlines.extend(job.next_attempt_at for job in self._retrying if job.next_attempt_at is not None)
                timeout = max(0.0, min(deadlines) - now) if deadlines else None
                self._cv.wait(timeout=timeout)

    def _is_fenced(self, job: _Job) -> bool:
        return any(
            self._committed_barriers.get(scope, -1) > job.payload_sequence for scope in job.fence
        )

    def _attempt(self, job: _Job, is_retry: bool) -> None:
        try:
            if self._fault_injector is not None:
                self._fault_injector(job.scope_key, job.payload_sequence)
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                if not self._is_fenced(job):
                    job.apply(conn)
                    if job.barrier is not None:
                        current = self._committed_barriers.get(job.barrier, -1)
                        self._committed_barriers[job.barrier] = max(current, job.payload_sequence)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        except sqlite3.IntegrityError as exc:
            # A constraint violation is never transient; retrying it would
            # loop forever. Surface it to whoever is waiting and drop it.
            logger.error("claude pool runtime-state write rejected by a schema constraint", exc_info=True)
            for pending_write in job.pending_writes:
                pending_write._mark_done(error=exc)
            return
        except Exception:
            job.attempts += 1
            backoff = min(
                self._retry_backoff_max_seconds,
                self._retry_backoff_initial_seconds * (2 ** (job.attempts - 1)),
            )
            job.next_attempt_at = self._clock() + backoff
            logger.warning(
                "claude pool runtime-state write failed (attempt %d); retrying in %.2fs",
                job.attempts,
                backoff,
                exc_info=True,
            )
            with self._cv:
                self._retrying.append(job)
                self._persistence_degraded = True
                self._cv.notify_all()
            return

        with self._cv:
            self._persistence_degraded = bool(self._retrying)
        for pending_write in job.pending_writes:
            pending_write._mark_done()
