"""Tests for the Claude account-pool SQLite runtime-state store."""

from __future__ import annotations

import asyncio
import sqlite3
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import claudex.balanced.state_model as state_model
import claudex.balanced.state_store as state_store
from claudex.balanced.state_model import (
    SCHEMA_VERSION,
    RestoreValidationContext,
    UnsupportedSchemaVersionError,
    _SCHEMA_SQL,
)
from claudex.balanced.state_store import ClaudePoolRuntimeStateStore

# --------------------------------------------------------------------------
# Fixtures and small row-builder helpers
# --------------------------------------------------------------------------


@pytest.fixture
def store_factory(tmp_path: Path):
    """Yields a factory that opens stores under `tmp_path`, closing them all at teardown."""
    opened: list[ClaudePoolRuntimeStateStore] = []
    counter = {"n": 0}

    def make(path: Path | None = None, **kwargs: Any) -> ClaudePoolRuntimeStateStore:
        if path is None:
            counter["n"] += 1
            path = tmp_path / f"runtime-{counter['n']}.sqlite3"
        kwargs.setdefault("debounce_seconds", 0.05)
        store = ClaudePoolRuntimeStateStore.open_(path, **kwargs)
        opened.append(store)
        return store

    yield make
    for store in opened:
        store.close()


def _cooldown_kwargs(**overrides: Any) -> dict[str, Any]:
    now = time.time()
    base = dict(
        account_id="acct-1",
        scope="account",
        model_family="",
        account_incarnation_id="inc-1",
        account_profile_fingerprint="fp-1",
        deadline_utc=now + 3600,
        reason="rate_limited",
        evidence="{}",
        updated_at_utc=now,
    )
    base.update(overrides)
    return base


def _observation_kwargs(**overrides: Any) -> dict[str, Any]:
    now = time.time()
    base = dict(
        account_id="acct-1",
        window="five_hour",
        account_incarnation_id="inc-1",
        account_profile_fingerprint="fp-1",
        used_percent=42.5,
        reset_identity="reset-1",
        reset_at_utc=now + 3600,
        observed_at_utc=now,
        source="usage_api",
    )
    base.update(overrides)
    return base


def _capability_kwargs(**overrides: Any) -> dict[str, Any]:
    now = time.time()
    base = dict(
        account_id="acct-1",
        capability_key="opus_4_5",
        account_incarnation_id="inc-1",
        account_profile_fingerprint="fp-1",
        state="eligible",
        evidence_source="probe",
        classifier_version="v1",
        observed_at_utc=now,
        expires_at_utc=now + 3600,
    )
    base.update(overrides)
    return base


def _pin_kwargs(**overrides: Any) -> dict[str, Any]:
    now = time.time()
    base = dict(
        session_key_digest=b"digest-default",
        key_kind="uuid",
        account_id="acct-1",
        account_incarnation_id="inc-1",
        last_seen_utc=now,
        expires_at_utc=now + 3600,
        generation=0,
        balanced_epoch_id="unset",
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Schema, pragmas, permissions, meta
# --------------------------------------------------------------------------


def test_persistence_symbols_have_canonical_owners() -> None:
    assert SCHEMA_VERSION is state_model.SCHEMA_VERSION
    assert RestoreValidationContext is state_model.RestoreValidationContext
    assert UnsupportedSchemaVersionError is state_model.UnsupportedSchemaVersionError
    assert ClaudePoolRuntimeStateStore is state_store.ClaudePoolRuntimeStateStore
    assert RestoreValidationContext.__module__ == "claudex.balanced.state_model"
    assert UnsupportedSchemaVersionError.__module__ == "claudex.balanced.state_model"
    assert ClaudePoolRuntimeStateStore.__module__ == "claudex.balanced.state_store"


def test_schema_contains_only_the_five_declared_tables(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path)
    store.close()

    conn = sqlite3.connect(str(path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert tables == {"meta", "pins", "cooldowns", "usage_observations", "capability_evidence"}


def test_wal_journal_mode_and_full_synchronous_are_set(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path)
    store.close()

    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # 2 == FULL
    finally:
        conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file mode bits are not meaningful on Windows")
def test_directory_and_file_permissions_are_owner_only(tmp_path: Path) -> None:
    # A nested, not-yet-existing directory proves it gets created too.
    nested_path = tmp_path / "nested" / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(nested_path)
    try:
        assert stat.S_IMODE(nested_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(nested_path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_meta_contains_required_keys_on_fresh_creation(store_factory) -> None:
    store = store_factory()
    assert store.schema_version == SCHEMA_VERSION
    assert store.balanced_epoch_id
    assert len(store.epoch_seed) == 32
    assert store.epoch_active is False


def test_reopen_reuses_existing_schema_and_epoch(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store1 = ClaudePoolRuntimeStateStore.open_(path)
    epoch_id = store1.balanced_epoch_id
    epoch_seed = store1.epoch_seed
    store1.upsert_cooldown(**_cooldown_kwargs()).wait(timeout=5)
    store1.close()

    store2 = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert store2.balanced_epoch_id == epoch_id
        assert store2.epoch_seed == epoch_seed
        assert store2.get_cooldown("acct-1", "account", "") is not None
    finally:
        store2.close()


def test_restore_reads_database_written_in_pre_split_format(tmp_path: Path) -> None:
    path = tmp_path / "pre-split-runtime.sqlite3"
    now = time.time()
    epoch_id = "pre-split-epoch"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("balanced_epoch_id", epoch_id),
                ("epoch_seed_hex", "01" * 32),
                ("epoch_active", "0"),
            ],
        )
        conn.execute(
            """
            INSERT INTO pins (session_key_digest, key_kind, account_id, account_incarnation_id,
                              last_seen_utc, expires_at_utc, generation, balanced_epoch_id, model_family)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (b"pre-split-pin", "uuid", "acct-1", "inc-1", now, now + 3600, 3, epoch_id, "fable"),
        )
        conn.execute(
            """
            INSERT INTO cooldowns (account_id, scope, model_family, account_incarnation_id,
                                   account_profile_fingerprint, deadline_utc, reason, evidence, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("acct-1", "family", "fable", "inc-1", "fp-1", now + 3600, "rate_limited", "{}", now),
        )
        conn.execute(
            """
            INSERT INTO usage_observations (account_id, window, account_incarnation_id,
                                            account_profile_fingerprint, used_percent, reset_identity,
                                            reset_at_utc, observed_at_utc, source, unified_status, unified_claim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "acct-1",
                "fable_weekly",
                "inc-1",
                "fp-1",
                42.5,
                "reset-1",
                now + 3600,
                now,
                "usage_api",
                "ok",
                "claim-1",
            ),
        )
        conn.execute(
            """
            INSERT INTO capability_evidence (account_id, capability_key, account_incarnation_id,
                                             account_profile_fingerprint, state, evidence_source,
                                             classifier_version, observed_at_utc, expires_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("acct-1", "opus_4_5", "inc-1", "fp-1", "eligible", "probe", "v1", now, now + 3600),
        )
        conn.commit()
    finally:
        conn.close()

    store = ClaudePoolRuntimeStateStore.open_(path, debounce_seconds=0.0)
    try:
        restored = store.restore(RestoreValidationContext(now_utc=now))
        assert restored.pins[b"pre-split-pin"].generation == 3
        assert restored.pins[b"pre-split-pin"].model_family == "fable"
        assert restored.cooldowns[("acct-1", "family", "fable")].reason == "rate_limited"
        assert restored.usage_observations[("acct-1", "fable_weekly")].unified_claim == "claim-1"
        assert restored.capability_evidence[("acct-1", "opus_4_5")].state == "eligible"
    finally:
        store.close()


# --------------------------------------------------------------------------
# Row round trips
# --------------------------------------------------------------------------


def test_pin_upsert_and_get_round_trip(store_factory) -> None:
    store = store_factory()
    kwargs = _pin_kwargs(session_key_digest=b"digest-1", balanced_epoch_id=store.balanced_epoch_id)
    store.upsert_pin(**kwargs).wait(timeout=5)

    pin = store.get_pin(b"digest-1")
    assert pin is not None
    assert pin.account_id == kwargs["account_id"]
    assert pin.key_kind == "uuid"
    assert pin.generation == 0
    assert pin.balanced_epoch_id == store.balanced_epoch_id


def test_pin_upsert_overwrites_existing_row(store_factory) -> None:
    store = store_factory()
    digest = b"digest-2"
    store.upsert_pin(**_pin_kwargs(session_key_digest=digest, account_id="acct-a", balanced_epoch_id=store.balanced_epoch_id)).wait(
        timeout=5
    )
    store.upsert_pin(
        **_pin_kwargs(session_key_digest=digest, account_id="acct-b", generation=1, balanced_epoch_id=store.balanced_epoch_id)
    ).wait(timeout=5)

    pin = store.get_pin(digest)
    assert pin is not None
    assert pin.account_id == "acct-b"
    assert pin.generation == 1


def test_pin_round_trip_preserves_model_family(store_factory) -> None:
    store = store_factory()
    kwargs = _pin_kwargs(
        session_key_digest=b"digest-family",
        balanced_epoch_id=store.balanced_epoch_id,
        model_family="fable",
    )
    store.upsert_pin(**kwargs).wait(timeout=5)

    pin = store.get_pin(b"digest-family")
    assert pin is not None
    assert pin.model_family == "fable"

    restore_result = store.restore(RestoreValidationContext(now_utc=time.time()))
    assert restore_result.pins[b"digest-family"].model_family == "fable"


def test_pin_delete_round_trip(store_factory) -> None:
    store = store_factory()
    digest = b"digest-3"
    store.upsert_pin(**_pin_kwargs(session_key_digest=digest, balanced_epoch_id=store.balanced_epoch_id)).wait(timeout=5)
    assert store.get_pin(digest) is not None

    store.delete_pin(digest).wait(timeout=5)
    assert store.get_pin(digest) is None


def test_cooldown_upsert_and_get_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_cooldown(**_cooldown_kwargs(reason="first")).wait(timeout=5)
    store.upsert_cooldown(**_cooldown_kwargs(reason="second")).wait(timeout=5)

    row = store.get_cooldown("acct-1", "account", "")
    assert row is not None
    assert row.reason == "second"


def test_cooldown_delete_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_cooldown(**_cooldown_kwargs()).wait(timeout=5)
    assert store.get_cooldown("acct-1", "account", "") is not None

    store.delete_cooldown("acct-1", "account", "").wait(timeout=5)
    assert store.get_cooldown("acct-1", "account", "") is None


def test_usage_observation_upsert_and_get_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_usage_observation(**_observation_kwargs(used_percent=10.0)).wait(timeout=5)
    store.upsert_usage_observation(**_observation_kwargs(used_percent=55.0, unified_status="ok")).wait(timeout=5)

    row = store.get_usage_observation("acct-1", "five_hour")
    assert row is not None
    assert row.used_percent == 55.0
    assert row.unified_status == "ok"


def test_usage_observation_delete_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_usage_observation(**_observation_kwargs()).wait(timeout=5)
    assert store.get_usage_observation("acct-1", "five_hour") is not None

    store.delete_usage_observation("acct-1", "five_hour").wait(timeout=5)
    assert store.get_usage_observation("acct-1", "five_hour") is None


def test_capability_evidence_upsert_and_get_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_capability_evidence(**_capability_kwargs(state="eligible")).wait(timeout=5)
    store.upsert_capability_evidence(**_capability_kwargs(state="denied")).wait(timeout=5)

    row = store.get_capability_evidence("acct-1", "opus_4_5")
    assert row is not None
    assert row.state == "denied"


def test_capability_evidence_delete_round_trip(store_factory) -> None:
    store = store_factory()
    store.upsert_capability_evidence(**_capability_kwargs()).wait(timeout=5)
    assert store.get_capability_evidence("acct-1", "opus_4_5") is not None

    store.delete_capability_evidence("acct-1", "opus_4_5").wait(timeout=5)
    assert store.get_capability_evidence("acct-1", "opus_4_5") is None


def test_invalid_enum_values_are_rejected_synchronously(store_factory) -> None:
    store = store_factory()
    with pytest.raises(ValueError):
        store.upsert_pin(**_pin_kwargs(key_kind="bogus", balanced_epoch_id=store.balanced_epoch_id))
    with pytest.raises(ValueError):
        store.upsert_cooldown(**_cooldown_kwargs(scope="bogus"))
    with pytest.raises(ValueError):
        store.upsert_usage_observation(**_observation_kwargs(window="bogus"))
    with pytest.raises(ValueError):
        store.upsert_capability_evidence(**_capability_kwargs(state="bogus"))


def test_delete_all_for_incarnation_removes_rows_across_every_table(store_factory) -> None:
    store = store_factory()
    digest = b"digest-incarnation"
    store.upsert_pin(**_pin_kwargs(session_key_digest=digest, account_incarnation_id="inc-x", balanced_epoch_id=store.balanced_epoch_id)).wait(
        timeout=5
    )
    store.upsert_cooldown(**_cooldown_kwargs(account_incarnation_id="inc-x")).wait(timeout=5)
    store.upsert_usage_observation(**_observation_kwargs(account_incarnation_id="inc-x")).wait(timeout=5)
    store.upsert_capability_evidence(**_capability_kwargs(account_incarnation_id="inc-x")).wait(timeout=5)

    store.delete_all_for_incarnation("inc-x").wait(timeout=5)

    assert store.get_pin(digest) is None
    assert store.get_cooldown("acct-1", "account", "") is None
    assert store.get_usage_observation("acct-1", "five_hour") is None
    assert store.get_capability_evidence("acct-1", "opus_4_5") is None


# --------------------------------------------------------------------------
# Epoch rotation / invalidation
# --------------------------------------------------------------------------


def test_rotate_epoch_changes_id_and_seed_and_wipes_pins(store_factory) -> None:
    store = store_factory()
    old_epoch_id = store.balanced_epoch_id
    old_seed = store.epoch_seed
    store.upsert_pin(**_pin_kwargs(session_key_digest=b"digest-rot", balanced_epoch_id=old_epoch_id)).wait(timeout=5)
    assert store.pin_count() == 1

    store.rotate_epoch().wait(timeout=5)

    assert store.balanced_epoch_id != old_epoch_id
    assert store.epoch_seed != old_seed
    assert store.pin_count() == 0


def test_invalidate_epoch_pins_removes_only_pins_from_a_stale_epoch(store_factory) -> None:
    store = store_factory()
    current_epoch = store.balanced_epoch_id
    store.upsert_pin(**_pin_kwargs(session_key_digest=b"digest-current", balanced_epoch_id=current_epoch)).wait(timeout=5)
    store.upsert_pin(**_pin_kwargs(session_key_digest=b"digest-stale", balanced_epoch_id="some-other-epoch")).wait(timeout=5)

    store.invalidate_epoch_pins().wait(timeout=5)

    assert store.get_pin(b"digest-current") is not None
    assert store.get_pin(b"digest-stale") is None


# --------------------------------------------------------------------------
# Restore: every skip-and-delete invalidation class, plus valid rows kept
# --------------------------------------------------------------------------


def test_restore_skips_and_deletes_expired_pins(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_pin(
        **_pin_kwargs(session_key_digest=b"expired-pin", expires_at_utc=now - 10, balanced_epoch_id=store.balanced_epoch_id)
    ).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert b"expired-pin" not in result.pins
    assert result.skip_counts.get("pins.expired") == 1
    assert store.get_pin(b"expired-pin") is None


def test_restore_skips_and_deletes_pins_from_a_stale_epoch(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_pin(
        **_pin_kwargs(session_key_digest=b"stale-epoch-pin", expires_at_utc=now + 3600, balanced_epoch_id="old-epoch")
    ).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert b"stale-epoch-pin" not in result.pins
    assert result.skip_counts.get("pins.epoch_mismatch") == 1
    assert store.get_pin(b"stale-epoch-pin") is None


def test_restore_skips_and_deletes_expired_cooldowns(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_cooldown(**_cooldown_kwargs(deadline_utc=now - 5)).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert ("acct-1", "account", "") not in result.cooldowns
    assert result.skip_counts.get("cooldowns.expired") == 1
    assert store.get_cooldown("acct-1", "account", "") is None


def test_restore_skips_and_deletes_stale_reset_usage_observations(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_usage_observation(**_observation_kwargs(reset_at_utc=now - 5)).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert ("acct-1", "five_hour") not in result.usage_observations
    assert result.skip_counts.get("usage_observations.stale_reset") == 1
    assert store.get_usage_observation("acct-1", "five_hour") is None


def test_restore_skips_and_deletes_expired_capability_evidence(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_capability_evidence(**_capability_kwargs(expires_at_utc=now - 5)).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert ("acct-1", "opus_4_5") not in result.capability_evidence
    assert result.skip_counts.get("capability_evidence.expired") == 1
    assert store.get_capability_evidence("acct-1", "opus_4_5") is None


def test_restore_keeps_every_still_valid_row(store_factory) -> None:
    store = store_factory()
    now = time.time()
    digest = b"digest-valid"
    store.upsert_pin(**_pin_kwargs(session_key_digest=digest, expires_at_utc=now + 3600, balanced_epoch_id=store.balanced_epoch_id)).wait(
        timeout=5
    )
    store.upsert_cooldown(**_cooldown_kwargs(deadline_utc=now + 3600)).wait(timeout=5)
    store.upsert_usage_observation(**_observation_kwargs(reset_at_utc=now + 3600)).wait(timeout=5)
    store.upsert_capability_evidence(**_capability_kwargs(expires_at_utc=now + 3600)).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert digest in result.pins
    assert ("acct-1", "account", "") in result.cooldowns
    assert ("acct-1", "five_hour") in result.usage_observations
    assert ("acct-1", "opus_4_5") in result.capability_evidence
    assert sum(result.skip_counts.values()) == 0


def test_restore_keeps_a_capability_row_with_no_expiry(store_factory) -> None:
    store = store_factory()
    now = time.time()
    store.upsert_capability_evidence(**_capability_kwargs(expires_at_utc=None)).wait(timeout=5)

    result = store.restore(RestoreValidationContext(now_utc=now))

    assert ("acct-1", "opus_4_5") in result.capability_evidence


# --------------------------------------------------------------------------
# Newer-schema refusal (byte-identical) and quarantine of corrupt/older stores
# --------------------------------------------------------------------------


def test_newer_schema_version_is_refused_and_file_left_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path)
    store.close()

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()

    before = path.read_bytes()
    with pytest.raises(UnsupportedSchemaVersionError):
        ClaudePoolRuntimeStateStore.open_(path)
    after = path.read_bytes()

    assert before == after
    assert not any(tmp_path.glob("*.quarantined-*"))


def test_corrupt_database_is_quarantined_and_replaced_with_a_fresh_store(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a sqlite database" * 4)

    store = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert store.schema_version == SCHEMA_VERSION
        assert store.pin_count() == 0
    finally:
        store.close()

    quarantined = list(tmp_path.glob("runtime.sqlite3.quarantined-*"))
    assert len(quarantined) == 1


def test_older_unsupported_schema_is_quarantined_and_replaced_with_a_fresh_store(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path)
    old_epoch_id = store.balanced_epoch_id
    store.upsert_cooldown(**_cooldown_kwargs()).wait(timeout=5)
    store.close()

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    store2 = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert store2.balanced_epoch_id != old_epoch_id
        # The quarantined store's data must not leak into the fresh one.
        assert store2.get_cooldown("acct-1", "account", "") is None
    finally:
        store2.close()

    quarantined = list(tmp_path.glob("runtime.sqlite3.quarantined-*"))
    assert len(quarantined) == 1


def test_schema_v1_pin_rows_are_quarantined_on_open(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path)
    pin_kwargs = _pin_kwargs(balanced_epoch_id=store.balanced_epoch_id, model_family="fable")
    store.upsert_pin(**pin_kwargs).wait(timeout=5)
    store.close()

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    reopened = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert reopened.get_pin(pin_kwargs["session_key_digest"]) is None
        assert reopened.pin_count() == 0
        restored = reopened.restore(RestoreValidationContext(now_utc=time.time()))
        assert restored.pins == {}
    finally:
        reopened.close()

    quarantined = list(tmp_path.glob("runtime.sqlite3.quarantined-*"))
    assert len(quarantined) == 1


def test_quarantine_moves_wal_and_shm_siblings_together(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path, debounce_seconds=5.0)
    store.upsert_cooldown(**_cooldown_kwargs())  # left pending; do not close (keeps -wal alive)
    wal_path = path.with_name(path.name + "-wal")
    # Force a checkpoint-free WAL presence by not closing; give the writer a
    # moment to have at least created the WAL file on first write attempt.
    store.close()

    # Recreate a WAL sibling manually to simulate a live, uncheckpointed store.
    wal_path.write_bytes(b"fake-wal-bytes")
    corrupt_conn_path = path
    corrupt_conn_path.write_bytes(b"not a sqlite database at all")

    store2 = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert store2.schema_version == SCHEMA_VERSION
    finally:
        store2.close()

    quarantined_main = list(tmp_path.glob("runtime.sqlite3.quarantined-*"))
    quarantined_wal = list(tmp_path.glob("runtime.sqlite3-wal.quarantined-*"))
    assert len(quarantined_main) == 1
    assert len(quarantined_wal) == 1


# --------------------------------------------------------------------------
# Coalescing
# --------------------------------------------------------------------------


def test_same_row_coalescing_keeps_only_the_latest_payload(store_factory) -> None:
    store = store_factory(debounce_seconds=0.2)
    store.upsert_cooldown(**_cooldown_kwargs(reason="stale"))
    final = store.upsert_cooldown(**_cooldown_kwargs(reason="fresh"))
    final.wait(timeout=5)

    row = store.get_cooldown("acct-1", "account", "")
    assert row is not None
    assert row.reason == "fresh"


def test_same_row_coalescing_preserves_the_earliest_queue_position(store_factory) -> None:
    """Step 4: coalescing may replace the payload, but the row keeps the
    earliest queue position — its debounce deadline is anchored to the
    first submission, not reset by later ones for the same row."""
    store = store_factory(debounce_seconds=0.3)
    started = time.monotonic()
    store.upsert_cooldown(**_cooldown_kwargs(reason="first"))
    time.sleep(0.15)
    second = store.upsert_cooldown(**_cooldown_kwargs(reason="second"))
    second.wait(timeout=5)
    elapsed = time.monotonic() - started

    # If coalescing had reset the debounce timer, this would take roughly
    # 0.15 + 0.3 = 0.45s. Preserving the earliest position keeps it near 0.3s.
    assert elapsed < 0.4, f"coalescing appears to have reset the debounce deadline: {elapsed:.3f}s"
    row = store.get_cooldown("acct-1", "account", "")
    assert row is not None and row.reason == "second"


def test_pin_last_seen_touch_does_not_clobber_a_pending_full_upsert(store_factory) -> None:
    store = store_factory(debounce_seconds=0.2)
    digest = b"digest-touch-safety"
    pending_upsert = store.upsert_pin(
        **_pin_kwargs(session_key_digest=digest, account_id="acct-full", balanced_epoch_id=store.balanced_epoch_id)
    )
    touch = store.touch_pin_last_seen(digest, time.time() + 1)
    assert touch is not None
    pending_upsert.wait(timeout=5)
    touch.wait(timeout=5)

    pin = store.get_pin(digest)
    assert pin is not None
    assert pin.account_id == "acct-full"  # the full upsert's other columns survived


# --------------------------------------------------------------------------
# Timing: default debounce window and the pin last_seen throttle
# --------------------------------------------------------------------------


def test_default_debounce_flushes_within_roughly_one_second(tmp_path: Path) -> None:
    store = ClaudePoolRuntimeStateStore.open_(tmp_path / "runtime.sqlite3")
    try:
        started = time.monotonic()
        pending = store.upsert_usage_observation(**_observation_kwargs())
        pending.wait(timeout=5)
        elapsed = time.monotonic() - started
        assert 0.4 <= elapsed <= 1.8, f"debounced flush took {elapsed:.3f}s, expected roughly one second"
    finally:
        store.close()


def test_pin_last_seen_writes_are_throttled_to_once_per_60_seconds(tmp_path: Path) -> None:
    fake_time = {"t": 1_000_000.0}

    def clock() -> float:
        return fake_time["t"]

    store = ClaudePoolRuntimeStateStore.open_(tmp_path / "runtime.sqlite3", clock=clock, debounce_seconds=0.0)
    try:
        digest = b"digest-throttle"
        store.upsert_pin(**_pin_kwargs(session_key_digest=digest, last_seen_utc=fake_time["t"], balanced_epoch_id=store.balanced_epoch_id)).wait(
            timeout=5
        )

        first = store.touch_pin_last_seen(digest, fake_time["t"])
        assert first is not None
        first.wait(timeout=5)

        fake_time["t"] += 30.0  # inside the 60s window
        assert store.touch_pin_last_seen(digest, fake_time["t"]) is None

        fake_time["t"] += 31.0  # now past 60s since the first accepted touch
        second = store.touch_pin_last_seen(digest, fake_time["t"])
        assert second is not None
        second.wait(timeout=5)

        pin = store.get_pin(digest)
        assert pin is not None
        assert pin.last_seen_utc == fake_time["t"]
    finally:
        store.close()


# --------------------------------------------------------------------------
# Causal ordering: high priority bypasses debounce without overtaking
# --------------------------------------------------------------------------


def test_high_priority_delete_flushes_pending_upsert_first_without_overtaking(store_factory) -> None:
    """Step 9 / Note A check #4.

    A cooldown upsert is left pending (debounced); a high-priority delete
    for the *same* row is submitted right after. If priority ever jumped
    the earlier pending write, the delete would run first and the later
    insert would leave the row present. The row ending up absent is only
    possible if the earlier (lower) sequence committed first.
    """
    store = store_factory(debounce_seconds=5.0)
    pending = store.upsert_cooldown(**_cooldown_kwargs())
    high_priority_delete = store.delete_cooldown("acct-1", "account", "", high_priority=True)

    high_priority_delete.wait(timeout=2)
    pending.wait(timeout=1)

    assert store.get_cooldown("acct-1", "account", "") is None


def test_high_priority_migration_flushes_unrelated_pending_writes_first(store_factory) -> None:
    """A high-priority pin migration (reassigning a pin to a new account)
    must flush every other still-pending write ahead of it, in order, then
    commit — not just its own row."""
    store = store_factory(debounce_seconds=5.0)
    epoch = store.balanced_epoch_id
    unrelated = store.upsert_pin(**_pin_kwargs(session_key_digest=b"digest-unrelated", account_id="acct-y", balanced_epoch_id=epoch))
    store.upsert_pin(**_pin_kwargs(session_key_digest=b"digest-migrate", account_id="acct-a", balanced_epoch_id=epoch))
    migration = store.upsert_pin(
        **_pin_kwargs(session_key_digest=b"digest-migrate", account_id="acct-b", generation=1, balanced_epoch_id=epoch, high_priority=True)
    )

    migration.wait(timeout=2)
    unrelated.wait(timeout=1)  # already flushed as a side effect of the forced flush; must not need the 5s debounce

    assert store.get_pin(b"digest-unrelated") is not None
    migrated_pin = store.get_pin(b"digest-migrate")
    assert migrated_pin is not None
    assert migrated_pin.account_id == "acct-b"


# --------------------------------------------------------------------------
# Causal ordering: retry of a failed/coalesced write cannot resurrect a row
# --------------------------------------------------------------------------


def test_retried_pin_write_cannot_resurrect_a_row_after_epoch_invalidation(tmp_path: Path) -> None:
    """Step 9 / Note A check #5.

    A pin upsert fails its first attempts (simulated transient failure) and
    goes into backoff. While it is still degraded and has not yet
    succeeded, a high-priority epoch rotation commits and wipes pins. Only
    once that has already committed does the original write's retry
    finally succeed. Sequence ordering — not wall-clock retry timing — must
    determine the final state: the row must not reappear.
    """
    fail_scope = ("pins", b"digest-retry-epoch")
    state = {"fails_remaining": 3}

    def fault_injector(scope_key: Any, sequence: int) -> None:
        if scope_key == fail_scope and state["fails_remaining"] > 0:
            state["fails_remaining"] -= 1
            raise sqlite3.OperationalError("simulated transient failure")

    store = ClaudePoolRuntimeStateStore.open_(
        tmp_path / "runtime.sqlite3",
        debounce_seconds=0.02,
        retry_backoff_initial_seconds=0.03,
        retry_backoff_max_seconds=0.03,
        fault_injector=fault_injector,
    )
    try:
        epoch_before = store.balanced_epoch_id
        pending_upsert = store.upsert_pin(
            **_pin_kwargs(session_key_digest=b"digest-retry-epoch", balanced_epoch_id=epoch_before)
        )

        deadline = time.monotonic() + 5
        while not store.persistence_degraded and time.monotonic() < deadline:
            time.sleep(0.005)
        assert store.persistence_degraded, "expected the upsert to have failed at least once by now"

        invalidation = store.rotate_epoch()
        invalidation.wait(timeout=5)
        assert store.balanced_epoch_id != epoch_before

        pending_upsert.wait(timeout=5)  # let the stale retry finish, whatever it does

        assert store.get_pin(b"digest-retry-epoch") is None
    finally:
        store.close()


def test_retried_cooldown_write_cannot_resurrect_a_row_after_incarnation_deletion(tmp_path: Path) -> None:
    """Same causal guarantee as above, exercised via `delete_all_for_incarnation`."""
    fail_scope = ("cooldowns", "acct-retry", "account", "")
    state = {"fails_remaining": 3}

    def fault_injector(scope_key: Any, sequence: int) -> None:
        if scope_key == fail_scope and state["fails_remaining"] > 0:
            state["fails_remaining"] -= 1
            raise sqlite3.OperationalError("simulated transient failure")

    store = ClaudePoolRuntimeStateStore.open_(
        tmp_path / "runtime.sqlite3",
        debounce_seconds=0.02,
        retry_backoff_initial_seconds=0.03,
        retry_backoff_max_seconds=0.03,
        fault_injector=fault_injector,
    )
    try:
        pending_upsert = store.upsert_cooldown(**_cooldown_kwargs(account_id="acct-retry", account_incarnation_id="inc-retry"))

        deadline = time.monotonic() + 5
        while not store.persistence_degraded and time.monotonic() < deadline:
            time.sleep(0.005)
        assert store.persistence_degraded, "expected the upsert to have failed at least once by now"

        deletion = store.delete_all_for_incarnation("inc-retry")
        deletion.wait(timeout=5)

        pending_upsert.wait(timeout=5)  # let the stale retry finish, whatever it does

        assert store.get_cooldown("acct-retry", "account", "") is None
    finally:
        store.close()


# --------------------------------------------------------------------------
# Persistence degradation and recovery
# --------------------------------------------------------------------------


def test_persistence_degrades_then_recovers_after_transient_failures(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fault_injector(scope_key: Any, sequence: int) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise sqlite3.OperationalError("simulated transient failure")

    store = ClaudePoolRuntimeStateStore.open_(
        tmp_path / "runtime.sqlite3",
        debounce_seconds=0.0,
        retry_backoff_initial_seconds=0.02,
        retry_backoff_max_seconds=0.02,
        fault_injector=fault_injector,
    )
    try:
        assert store.persistence_degraded is False
        pending = store.upsert_cooldown(**_cooldown_kwargs())

        deadline = time.monotonic() + 5
        while not store.persistence_degraded and time.monotonic() < deadline:
            time.sleep(0.005)
        assert store.persistence_degraded is True

        pending.wait(timeout=5)
        assert store.persistence_degraded is False
        assert store.get_cooldown("acct-1", "account", "") is not None
    finally:
        store.close()


# --------------------------------------------------------------------------
# Writer-thread identity
# --------------------------------------------------------------------------


def test_writer_thread_is_single_and_applies_every_write(tmp_path: Path) -> None:
    idents: list[int] = []

    def recorder(scope_key: Any, sequence: int) -> None:
        idents.append(threading.get_ident())

    threads_before = threading.active_count()
    store = ClaudePoolRuntimeStateStore.open_(
        tmp_path / "runtime.sqlite3", debounce_seconds=0.0, fault_injector=recorder
    )
    try:
        assert threading.active_count() == threads_before + 1
        for index in range(5):
            store.upsert_cooldown(**_cooldown_kwargs(account_id=f"acct-{index}")).wait(timeout=5)
        assert idents
        assert all(ident == store.writer_thread_ident for ident in idents)
    finally:
        store.close()
    assert threading.active_count() == threads_before


# --------------------------------------------------------------------------
# close(): reject-new, idempotent, drain-before-return, checkpoint
# --------------------------------------------------------------------------


def test_close_rejects_new_submissions(tmp_path: Path) -> None:
    store = ClaudePoolRuntimeStateStore.open_(tmp_path / "runtime.sqlite3")
    store.close()
    with pytest.raises(RuntimeError):
        store.upsert_cooldown(**_cooldown_kwargs())


def test_close_is_idempotent(tmp_path: Path) -> None:
    store = ClaudePoolRuntimeStateStore.open_(tmp_path / "runtime.sqlite3")
    store.close()
    store.close()  # must not raise or hang


def test_close_drains_pending_writes_before_returning(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path, debounce_seconds=5.0)
    store.upsert_cooldown(**_cooldown_kwargs())

    started = time.monotonic()
    store.close()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, "close() waited out the full debounce instead of forcing an immediate flush"

    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT reason FROM cooldowns WHERE account_id = 'acct-1'").fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == "rate_limited"


def test_close_checkpoints_the_wal(tmp_path: Path) -> None:
    path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(path, debounce_seconds=0.0)
    store.upsert_cooldown(**_cooldown_kwargs()).wait(timeout=5)
    store.close()

    wal_path = path.with_name(path.name + "-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    store2 = ClaudePoolRuntimeStateStore.open_(path)
    try:
        assert store2.get_cooldown("acct-1", "account", "") is not None
    finally:
        store2.close()


# --------------------------------------------------------------------------
# High-priority completion can be awaited without blocking the event loop
# --------------------------------------------------------------------------


def test_high_priority_write_can_be_awaited_without_blocking_the_event_loop(tmp_path: Path) -> None:
    def slow_fault(scope_key: Any, sequence: int) -> None:
        time.sleep(0.3)

    async def scenario() -> int:
        # A high-priority write that takes noticeable wall time (via a slow
        # apply through the writer thread) must not stall the event loop's
        # own concurrent progress while it is awaited.
        slow_store = ClaudePoolRuntimeStateStore.open_(
            tmp_path / "runtime-slow.sqlite3", debounce_seconds=0.0, fault_injector=slow_fault
        )
        try:
            ticks: list[float] = []

            async def ticker() -> None:
                for _ in range(15):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            pending = slow_store.rotate_epoch()

            ticker_task = asyncio.create_task(ticker())
            await asyncio.gather(pending.wait_async(timeout=5), ticker_task)
            return len(ticks)
        finally:
            slow_store.close()

    tick_count = asyncio.run(scenario())
    assert tick_count >= 5, "the event loop appears to have been blocked while awaiting the high-priority write"
