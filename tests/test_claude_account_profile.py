"""Tests for the versioned account profile fingerprint."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from claudex.claude.account_profile import (
    compute_account_profile_fingerprint,
    load_account_profile_fingerprint,
)

_FIXTURE: dict[str, Any] = {
    "accountUuid": "11111111-2222-3333-4444-555555555555",
    "organizationUuid": "66666666-7777-8888-9999-aaaaaaaaaaaa",
    "organizationType": "claude_max",
    "organizationRateLimitTier": "default_claude_max_20x",
    "userRateLimitTier": "default",
    "seatTier": "enterprise",
    "billingType": "stripe",
    "hasExtraUsageEnabled": True,
    "organizationRole": "admin",
    "workspaceRole": "member",
    "subscriptionCreatedAt": "2024-01-01T00:00:00Z",
    "ccOnboardingFlags": ["flag_a", "flag_b"],
    "claudeCodeTrialDurationDays": 14,
    "claudeCodeTrialEndsAt": "2024-02-01T00:00:00Z",
    "organizationName": "Example Org",
    "emailAddress": "user@example.com",
    "displayName": "User Name",
    "accountCreatedAt": "2023-01-01T00:00:00Z",
    "profileFetchedAt": "2024-06-01T00:00:00Z",
}

_EXPECTED_FIXTURE_FINGERPRINT = (
    "v1:be47c41c5f5f2f1742a34bb5279b730bfc62a22fadaf1c17b1a80be16bfdd0b8"
)


def test_stable_digest_for_fixed_fixture() -> None:
    assert compute_account_profile_fingerprint(_FIXTURE) == _EXPECTED_FIXTURE_FINGERPRINT
    # Deterministic across repeated calls, not just a one-off match.
    assert compute_account_profile_fingerprint(copy.deepcopy(_FIXTURE)) == _EXPECTED_FIXTURE_FINGERPRINT


def test_absent_key_and_json_null_are_equivalent() -> None:
    with_null = copy.deepcopy(_FIXTURE)
    with_null["seatTier"] = None
    without_key = copy.deepcopy(_FIXTURE)
    del without_key["seatTier"]

    assert compute_account_profile_fingerprint(with_null) == compute_account_profile_fingerprint(
        without_key
    )


def test_unrecognized_fields_are_excluded() -> None:
    baseline = compute_account_profile_fingerprint(_FIXTURE)

    with_extra_field = copy.deepcopy(_FIXTURE)
    with_extra_field["someBrandNewField"] = "unexpected"

    assert compute_account_profile_fingerprint(with_extra_field) == baseline


def test_uppercase_uuid_canonicalizes_to_same_digest() -> None:
    uppercased = copy.deepcopy(_FIXTURE)
    uppercased["accountUuid"] = _FIXTURE["accountUuid"].upper()
    uppercased["organizationUuid"] = _FIXTURE["organizationUuid"].upper()

    assert compute_account_profile_fingerprint(uppercased) == compute_account_profile_fingerprint(
        _FIXTURE
    )


def test_missing_account_uuid_yields_none() -> None:
    missing = copy.deepcopy(_FIXTURE)
    del missing["accountUuid"]
    assert compute_account_profile_fingerprint(missing) is None


def test_invalid_account_uuid_yields_none() -> None:
    invalid = copy.deepcopy(_FIXTURE)
    invalid["accountUuid"] = "not-a-uuid"
    assert compute_account_profile_fingerprint(invalid) is None

    empty = copy.deepcopy(_FIXTURE)
    empty["accountUuid"] = ""
    assert compute_account_profile_fingerprint(empty) is None

    wrong_type = copy.deepcopy(_FIXTURE)
    wrong_type["accountUuid"] = 12345
    assert compute_account_profile_fingerprint(wrong_type) is None


def test_load_account_profile_fingerprint_malformed_file_returns_none(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"
    account_dir.mkdir()
    (account_dir / "oauth-account.json").write_text("{not valid json", encoding="utf-8")

    assert load_account_profile_fingerprint(account_dir) is None


def test_load_account_profile_fingerprint_missing_file_returns_none(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"
    account_dir.mkdir()

    assert load_account_profile_fingerprint(account_dir) is None


def test_load_account_profile_fingerprint_reads_recognized_fields(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"
    account_dir.mkdir()
    (account_dir / "oauth-account.json").write_text(json.dumps(_FIXTURE), encoding="utf-8")

    assert load_account_profile_fingerprint(account_dir) == _EXPECTED_FIXTURE_FINGERPRINT


def test_changing_recognized_field_changes_digest() -> None:
    baseline = compute_account_profile_fingerprint(_FIXTURE)

    changed = copy.deepcopy(_FIXTURE)
    changed["organizationRateLimitTier"] = "default_claude_pro"

    assert compute_account_profile_fingerprint(changed) != baseline


def test_changing_excluded_field_does_not_change_digest() -> None:
    baseline = compute_account_profile_fingerprint(_FIXTURE)

    changed = copy.deepcopy(_FIXTURE)
    changed["organizationName"] = "A Totally Different Org"
    changed["emailAddress"] = "someone-else@example.com"
    changed["displayName"] = "Someone Else"
    changed["accountCreatedAt"] = "2020-01-01T00:00:00Z"
    changed["profileFetchedAt"] = "2025-01-01T00:00:00Z"

    assert compute_account_profile_fingerprint(changed) == baseline
