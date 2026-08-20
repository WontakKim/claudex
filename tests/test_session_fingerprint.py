"""Tests for shared session UUID fingerprints and fallback seed persistence."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import stat
from pathlib import Path
from typing import Any

import pytest

import claudex.claude.session_fingerprint as session_fingerprint
from claudex.balanced.selection import (
    _HRW_DOMAIN,
    _PIN_KEY_DOMAIN,
    _SESSION_KEY_DOMAIN,
    _STATELESS_REQUEST_DOMAIN,
    _uuid_session_key,
)
from claudex.claude.session_fingerprint import (
    SESSION_FINGERPRINT_DOMAIN,
    extract_session_uuid,
    load_or_create_fingerprint_seed,
    observability_session_fingerprint,
)


_CANONICAL_UUID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def _body_with_session_id(session_id: Any) -> dict[str, Any]:
    return {
        "metadata": {
            "user_id": json.dumps(
                {
                    "device_id": "d" * 64,
                    "account_uuid": "client-account-uuid",
                    "session_id": session_id,
                },
                separators=(",", ":"),
            )
        }
    }


def test_extract_session_uuid_accepts_valid_uuid_and_returns_raw_and_canonical() -> None:
    raw_spelling = _CANONICAL_UUID.upper()

    extracted = extract_session_uuid(_body_with_session_id(raw_spelling))

    assert extracted == (raw_spelling, _CANONICAL_UUID)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"metadata": None},
        {"metadata": []},
        {"metadata": {"user_id": None}},
        {"metadata": {"user_id": "not-json"}},
        {"metadata": {"user_id": "[]"}},
        {"metadata": {"user_id": "{}"}},
        _body_with_session_id(None),
        _body_with_session_id(""),
        _body_with_session_id(f" {_CANONICAL_UUID}"),
        _body_with_session_id(f"{_CANONICAL_UUID} "),
        _body_with_session_id("not-a-uuid"),
        _body_with_session_id("11111111-2222-4333-0444-555555555555"),
    ],
)
def test_extract_session_uuid_rejects_noncanonical_shapes(body: dict[str, Any]) -> None:
    assert extract_session_uuid(body) is None


def test_extract_session_uuid_matches_selection_acceptance() -> None:
    seed = b"selection-acceptance-seed"
    bodies = [
        _body_with_session_id(_CANONICAL_UUID),
        _body_with_session_id(_CANONICAL_UUID.upper()),
        _body_with_session_id(_CANONICAL_UUID.replace("-", "")),
        _body_with_session_id(f"urn:uuid:{_CANONICAL_UUID}"),
        _body_with_session_id(f" {_CANONICAL_UUID}"),
        _body_with_session_id("11111111-2222-4333-0444-555555555555"),
        {"metadata": {"user_id": "not-json"}},
        {},
    ]

    for body in bodies:
        assert (extract_session_uuid(body) is not None) == (
            _uuid_session_key(body, seed) is not None
        )


def test_observability_session_fingerprint_uses_distinct_domain() -> None:
    seed = b"observability-domain-seed"
    canonical_utf8 = _CANONICAL_UUID.encode("utf-8")
    expected = hmac.new(
        seed,
        SESSION_FINGERPRINT_DOMAIN + b"\x00" + canonical_utf8,
        hashlib.sha256,
    ).hexdigest()

    fingerprint = observability_session_fingerprint(seed, _CANONICAL_UUID)

    assert fingerprint == expected
    assert len(fingerprint) == 64
    assert SESSION_FINGERPRINT_DOMAIN not in {
        _SESSION_KEY_DOMAIN,
        _PIN_KEY_DOMAIN,
        _HRW_DOMAIN,
        _STATELESS_REQUEST_DOMAIN,
    }
    for routing_domain in (
        _SESSION_KEY_DOMAIN,
        _PIN_KEY_DOMAIN,
        _HRW_DOMAIN,
        _STATELESS_REQUEST_DOMAIN,
    ):
        routing_fingerprint = hmac.new(
            seed, routing_domain + b"\x00" + canonical_utf8, hashlib.sha256
        ).hexdigest()
        assert fingerprint != routing_fingerprint


def test_load_or_create_fingerprint_seed_creates_0600_64hex(tmp_path: Path) -> None:
    pool_dir = tmp_path / "claude-account-pool"

    seed = load_or_create_fingerprint_seed(pool_dir)

    assert seed is not None
    assert len(seed) == 32
    seed_path = pool_dir / "session-fingerprint-seed"
    persisted = seed_path.read_text(encoding="ascii")
    assert persisted == seed.hex()
    assert len(persisted) == 64
    assert all(character in "0123456789abcdef" for character in persisted)
    assert stat.S_IMODE(seed_path.stat().st_mode) == 0o600
    assert list(pool_dir.glob(".session-fingerprint-seed.tmp-*")) == []


def test_load_or_create_fingerprint_seed_reuses_existing_and_never_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool_dir = tmp_path / "existing"
    first_seed = load_or_create_fingerprint_seed(pool_dir)
    assert first_seed is not None
    seed_path = pool_dir / "session-fingerprint-seed"
    original_inode = seed_path.stat().st_ino

    def fail_if_random_is_requested(_length: int) -> bytes:
        raise AssertionError("an existing seed must not be regenerated")

    with monkeypatch.context() as reuse_patch:
        reuse_patch.setattr(session_fingerprint.os, "urandom", fail_if_random_is_requested)
        assert load_or_create_fingerprint_seed(pool_dir) == first_seed

    assert seed_path.stat().st_ino == original_inode
    assert seed_path.read_text(encoding="ascii") == first_seed.hex()

    race_pool_dir = tmp_path / "race"
    race_pool_dir.mkdir()
    winner = b"w" * 32
    loser = b"l" * 32
    race_seed_path = race_pool_dir / "session-fingerprint-seed"

    def publish_winner_and_lose(_source: Path, destination: Path) -> None:
        destination.write_text(winner.hex(), encoding="ascii")
        destination.chmod(0o600)
        raise FileExistsError

    monkeypatch.setattr(session_fingerprint.os, "urandom", lambda _length: loser)
    monkeypatch.setattr(session_fingerprint.os, "link", publish_winner_and_lose)

    assert load_or_create_fingerprint_seed(race_pool_dir) == winner
    assert race_seed_path.read_text(encoding="ascii") == winner.hex()
    assert list(race_pool_dir.glob(".session-fingerprint-seed.tmp-*")) == []


def test_load_or_create_fingerprint_seed_corruption_returns_none_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool_dir = tmp_path / "corrupt"
    pool_dir.mkdir()
    seed_path = pool_dir / "session-fingerprint-seed"
    corrupt_seed = "A" * 64
    seed_path.write_text(corrupt_seed, encoding="ascii")
    seed_path.chmod(0o600)

    def fail_if_random_is_requested(_length: int) -> bytes:
        raise AssertionError("a corrupt seed must not be regenerated")

    monkeypatch.setattr(session_fingerprint.os, "urandom", fail_if_random_is_requested)
    with caplog.at_level(
        logging.WARNING, logger="claudex.claude.session_fingerprint"
    ):
        loaded = load_or_create_fingerprint_seed(pool_dir)

    assert loaded is None
    assert seed_path.read_text(encoding="ascii") == corrupt_seed
    assert any("could not be read or validated" in message for message in caplog.messages)
    assert all(corrupt_seed not in message for message in caplog.messages)


def test_load_or_create_fingerprint_seed_directory_sync_failure_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pool_dir = tmp_path / "sync-failure"

    def fail_directory_sync(_directory: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(session_fingerprint, "_fsync_directory", fail_directory_sync)
    with caplog.at_level(
        logging.WARNING, logger="claudex.claude.session_fingerprint"
    ):
        seed = load_or_create_fingerprint_seed(pool_dir)

    assert seed is None
    assert any("could not be made durable" in message for message in caplog.messages)
    seed_path = pool_dir / "session-fingerprint-seed"
    persisted = seed_path.read_text(encoding="ascii")
    assert len(persisted) == 64
    assert all(character in "0123456789abcdef" for character in persisted)
    assert list(pool_dir.glob(".session-fingerprint-seed.tmp-*")) == []

    monkeypatch.undo()
    recovered = load_or_create_fingerprint_seed(pool_dir)
    assert recovered is not None
    assert recovered.hex() == persisted


def test_selection_uuid_session_key_digest_unchanged_after_refactor() -> None:
    seed = b"routing-digest-compatibility-seed"
    raw_spelling = _CANONICAL_UUID.upper()
    canonical_utf8 = _CANONICAL_UUID.encode("utf-8")
    expected_digest = hmac.new(
        seed,
        _SESSION_KEY_DOMAIN
        + b"\x00uuid\x00"
        + len(canonical_utf8).to_bytes(8, "big")
        + canonical_utf8,
        hashlib.sha256,
    ).digest()

    session_key = _uuid_session_key(_body_with_session_id(raw_spelling), seed)

    assert session_key is not None
    assert session_key.kind == "uuid"
    assert session_key.digest == expected_digest
