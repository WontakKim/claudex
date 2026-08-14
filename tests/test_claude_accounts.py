"""Tests for the Claude account registry storage layer."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import pytest

from claudex_gateway import claude_accounts, paths


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _credentials_payload() -> dict[str, Any]:
    return {"accessToken": "at-1", "refreshToken": "rt-1"}


def _oauth_payload() -> dict[str, Any]:
    return {"accountUuid": "org-account-1"}


def _accounts_root() -> Path:
    return paths.accounts_dir("claude")


def _registry_path() -> Path:
    return _accounts_root() / "registry.json"


def _write_raw_registry(rows: list[Any]) -> None:
    root = _accounts_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "registry.json").write_text(json.dumps(rows), encoding="utf-8")


def _eight_key_row(**overrides: Any) -> dict[str, Any]:
    """A registry row missing both required incarnation identity fields."""
    row: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "email": "user@example.com",
        "organizationUuid": "org-1",
        "organizationName": "Example Org",
        "createdAt": 1_700_000_000_000,
        "updatedAt": 1_700_000_000_000,
        "lastAuthenticatedAt": 1_700_000_000_000,
        "state": "ready",
    }
    row.update(overrides)
    return row


def _base_row(**overrides: Any) -> dict[str, Any]:
    """The exact current 10-key row shape."""
    row: dict[str, Any] = {
        **_eight_key_row(),
        "accountIncarnationId": str(uuid.uuid4()),
        "upstreamAccountUuid": None,
    }
    row.update(overrides)
    return row


def _add(email: str = "user@example.com", **kwargs: Any) -> claude_accounts.AccountRecord:
    kwargs.setdefault("organization_uuid", None)
    kwargs.setdefault("organization_name", None)
    kwargs.setdefault("credentials_json", _credentials_payload())
    kwargs.setdefault("oauth_account_json", None)
    return claude_accounts.add_account(email, **kwargs)


def _assert_pristine() -> None:
    """After a rolled-back `add_account`, only `registry.lock` (created by
    `file_lock` itself on entry) and possibly an untouched `registry.json`
    may remain — no staging directory, no canonical account directory."""
    root = _accounts_root()
    if root.exists():
        names = {entry.name for entry in root.iterdir()}
        assert names <= {"registry.json", "registry.lock"}
    assert claude_accounts.load_registry() == []


class _FailOnceOnMatch:
    """Monkeypatch-friendly wrapper: raises `OSError` the first time its
    positional args satisfy `predicate`, delegating to `real` every other
    call — used to inject a failure at one precise call site."""

    def __init__(self, real: Callable[..., Any], predicate: Callable[..., bool]) -> None:
        self._real = real
        self._predicate = predicate
        self._triggered = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._triggered and self._predicate(*args, **kwargs):
            self._triggered = True
            raise OSError("injected failure")
        return self._real(*args, **kwargs)


def _child_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    return env


def _fsync_directory_failing_on_nth_root_call(
    fail_on_call: int,
    exception_factory: Callable[[], BaseException] = lambda: OSError("injected fsync failure"),
) -> tuple[Callable[[Path], None], dict[str, int]]:
    """Wrap `_fsync_directory` so it raises on the `fail_on_call`-th call
    whose argument is `accounts/claude/` itself, and behaves normally
    otherwise. `_write_json_atomic` also fsyncs the *staging* directory when
    writing `credentials.json`/`oauth-account.json`, so counting every call
    indiscriminately would target the wrong step; only `accounts/claude/`
    calls are the pre-commit barrier / post-replace / tombstone-purge
    fsyncs this module documents as its commit boundaries.
    `exception_factory` lets a test inject a non-`OSError` interruption
    (e.g. `KeyboardInterrupt`) at the same boundaries.
    """
    real = claude_accounts._fsync_directory
    accounts_root = _accounts_root()
    call_count = {"n": 0}

    def _wrapped(directory: Path) -> None:
        is_root_call = directory == accounts_root
        if is_root_call:
            call_count["n"] += 1
            if call_count["n"] == fail_on_call:
                raise exception_factory()
        real(directory)

    return _wrapped, call_count


# --------------------------------------------------------------------------
# Round trip, row shape, normalization
# --------------------------------------------------------------------------


def test_add_and_list_round_trip() -> None:
    record = _add(
        "User@Example.com",
        organization_uuid="org-1",
        organization_name="Example Org",
        oauth_account_json=_oauth_payload(),
    )
    assert record.email == "user@example.com"
    assert record.state == "ready"
    assert record.created_at == record.updated_at == record.last_authenticated_at

    [listed] = claude_accounts.list_accounts()
    assert listed == record


def test_registry_row_has_exact_camel_case_keys() -> None:
    _add(organization_uuid="org-1", organization_name="Example Org")
    [row] = json.loads(_registry_path().read_text())
    assert set(row) == {
        "id",
        "email",
        "organizationUuid",
        "organizationName",
        "createdAt",
        "updatedAt",
        "lastAuthenticatedAt",
        "state",
        "accountIncarnationId",
        "upstreamAccountUuid",
    }


def test_credentials_and_oauth_account_payloads_are_json_objects_never_strings_or_null() -> None:
    record = _add(oauth_account_json=_oauth_payload())
    account_dir = _accounts_root() / record.id
    credentials = json.loads((account_dir / "credentials.json").read_text())
    oauth_account = json.loads((account_dir / "oauth-account.json").read_text())
    assert isinstance(credentials, dict)
    assert isinstance(oauth_account, dict)
    assert credentials == _credentials_payload()
    assert oauth_account == _oauth_payload()


def test_add_account_with_no_oauth_account_persists_empty_object() -> None:
    record = _add(oauth_account_json=None)
    oauth_path = _accounts_root() / record.id / "oauth-account.json"
    parsed = json.loads(oauth_path.read_text())
    assert parsed == {}
    assert isinstance(parsed, dict)


def test_add_account_normalizes_email_and_organization_fields() -> None:
    record = _add(
        "  User@Example.COM  ",
        organization_uuid="  org-1  ",
        organization_name="   ",
    )
    assert record.email == "user@example.com"
    assert record.organization_uuid == "org-1"
    assert record.organization_name is None  # blank after trim -> null


def test_list_accounts_sorted_by_created_at_then_id(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([300, 100, 100])
    monkeypatch.setattr(claude_accounts, "_now_millis", lambda: next(timestamps))
    newest = _add("a@example.com")
    tied_1 = _add("b@example.com")
    tied_2 = _add("c@example.com")

    listed = claude_accounts.list_accounts()
    tied_ids_sorted = sorted([tied_1.id, tied_2.id])
    assert [record.id for record in listed] == [*tied_ids_sorted, newest.id]


# --------------------------------------------------------------------------
# update_account_credentials: in-place credential replacement (re-auth)
# --------------------------------------------------------------------------


def test_update_account_credentials_replaces_files_and_bumps_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1_000, 1_000, 2_000])
    monkeypatch.setattr(claude_accounts, "_now_millis", lambda: next(timestamps))
    original = _add(
        organization_uuid="org-1",
        organization_name="Old Org Name",
        oauth_account_json={"accountUuid": "acct-old"},
    )
    untouched = _add("other@example.com")

    updated = claude_accounts.update_account_credentials(
        "user@example.com",
        "org-1",
        "New Org Name",
        {"accessToken": "at-2", "refreshToken": "rt-2"},
        {"accountUuid": "acct-new"},
    )

    assert updated.id == original.id
    assert updated.created_at == original.created_at
    assert updated.updated_at == updated.last_authenticated_at == 2_000
    assert updated.state == "ready"
    assert updated.organization_name == "New Org Name"

    account_dir = _accounts_root() / original.id
    assert json.loads((account_dir / "credentials.json").read_text()) == {
        "accessToken": "at-2",
        "refreshToken": "rt-2",
    }
    assert json.loads((account_dir / "oauth-account.json").read_text()) == {
        "accountUuid": "acct-new"
    }

    listed = {record.id: record for record in claude_accounts.list_accounts()}
    assert listed[original.id] == updated
    assert listed[untouched.id] == untouched


def test_update_account_credentials_normalizes_email_for_the_identity_lookup() -> None:
    original = _add("user@example.com", organization_uuid="org-1")
    updated = claude_accounts.update_account_credentials(
        "  USER@Example.COM  ", "org-1", None, {"accessToken": "at-2"}, None
    )
    assert updated.id == original.id


def test_update_account_credentials_unknown_identity_raises_without_writes() -> None:
    original = _add(organization_uuid="org-1")
    with pytest.raises(claude_accounts.AccountNotFoundError):
        claude_accounts.update_account_credentials(
            # Same email, different organization: the identity key is the
            # (email, organizationUuid) pair, not the email alone.
            "user@example.com",
            "org-2",
            None,
            {"accessToken": "at-2"},
            None,
        )
    account_dir = _accounts_root() / original.id
    assert json.loads((account_dir / "credentials.json").read_text()) == _credentials_payload()
    [listed] = claude_accounts.list_accounts()
    assert listed == original


def test_update_account_credentials_validates_inputs_before_any_write() -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        claude_accounts.update_account_credentials(
            "user@example.com",
            None,
            None,
            "not-a-dict",  # type: ignore[arg-type]
            None,
        )
    assert not paths.runtime_dir().exists()


# --------------------------------------------------------------------------
# mark_account_needs_reauth
# --------------------------------------------------------------------------


def test_mark_needs_reauth_flips_state_and_bumps_updated_at_only() -> None:
    original = _add()
    marked = claude_accounts.mark_account_needs_reauth(original.id)

    assert marked.state == "needs-reauth"
    assert marked.updated_at >= original.updated_at
    assert marked.last_authenticated_at == original.last_authenticated_at
    assert marked.created_at == original.created_at
    assert marked.id == original.id
    [listed] = claude_accounts.list_accounts()
    assert listed == marked


def test_mark_needs_reauth_is_idempotent_without_a_registry_write() -> None:
    original = _add()
    first = claude_accounts.mark_account_needs_reauth(original.id)
    mtime_after_first = _registry_path().stat().st_mtime_ns

    second = claude_accounts.mark_account_needs_reauth(original.id)

    assert second == first
    assert _registry_path().stat().st_mtime_ns == mtime_after_first


def test_mark_needs_reauth_unknown_id_raises_not_found() -> None:
    _add()
    with pytest.raises(claude_accounts.AccountNotFoundError):
        claude_accounts.mark_account_needs_reauth(str(uuid.uuid4()))


def test_mark_needs_reauth_rejects_non_canonical_id() -> None:
    with pytest.raises(claude_accounts.AccountNotFoundError):
        claude_accounts.mark_account_needs_reauth("../../etc/passwd")


def test_load_registry_accepts_needs_reauth_state() -> None:
    _write_raw_registry([_base_row(state="needs-reauth")])
    [record] = claude_accounts.load_registry()
    assert record.state == "needs-reauth"


def test_update_account_credentials_resets_needs_reauth_to_ready() -> None:
    original = _add(organization_uuid="org-1")
    claude_accounts.mark_account_needs_reauth(original.id)

    updated = claude_accounts.update_account_credentials(
        "user@example.com", "org-1", None, {"accessToken": "at-2"}, None
    )

    assert updated.id == original.id
    assert updated.state == "ready"


# --------------------------------------------------------------------------
# Validation before any filesystem write
# --------------------------------------------------------------------------


def test_add_account_rejects_missing_email_before_any_write() -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        _add("   ")
    assert not paths.runtime_dir().exists()


def test_add_account_rejects_non_dict_credentials_before_any_write() -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        _add(credentials_json="not-a-dict")  # type: ignore[arg-type]
    assert not paths.runtime_dir().exists()


def test_add_account_rejects_non_dict_oauth_account_before_any_write() -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        _add(oauth_account_json="not-a-dict")  # type: ignore[arg-type]
    assert not paths.runtime_dir().exists()


@pytest.mark.parametrize(
    "bad_email", ["foo\tbar@example.com", "foo\nbar@example.com", "foo\x00bar@example.com"]
)
def test_add_account_rejects_control_characters_in_email(bad_email: str) -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        _add(bad_email)
    assert not paths.runtime_dir().exists()


@pytest.mark.parametrize("field", ["organization_uuid", "organization_name"])
def test_add_account_rejects_control_characters_in_organization_fields(field: str) -> None:
    with pytest.raises(claude_accounts.AccountRegistryError):
        _add(**{field: "org\twith-control"})
    assert not paths.runtime_dir().exists()


# --------------------------------------------------------------------------
# load_registry: malformed (file-level) vs. strict row-level validation
# --------------------------------------------------------------------------


def test_load_registry_returns_empty_list_when_missing() -> None:
    assert claude_accounts.load_registry() == []


def test_load_registry_rejects_malformed_json_file_level_error() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    (root / "registry.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(claude_accounts.AccountRegistryError, match="not valid JSON"):
        claude_accounts.load_registry()


def test_load_registry_rejects_non_array_root_malformed_file_level_error() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    (root / "registry.json").write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    with pytest.raises(claude_accounts.AccountRegistryError, match="JSON array"):
        claude_accounts.load_registry()


def test_load_registry_rejects_malformed_row_that_is_not_an_object() -> None:
    _write_raw_registry(["not-an-object"])
    with pytest.raises(claude_accounts.AccountRegistryError, match="row 0"):
        claude_accounts.load_registry()


@pytest.mark.parametrize(
    "overrides",
    [
        {"extraKey": "boom"},
        {"createdAt": True},
        {"updatedAt": True},
        {"lastAuthenticatedAt": True},
        {"createdAt": -1},
        {"state": "cooling-down"},
        {"email": ""},
    ],
    ids=[
        "unknown-key",
        "bool-created-at",
        "bool-updated-at",
        "bool-last-authenticated-at",
        "negative-created-at",
        "wrong-state",
        "empty-email",
    ],
)
def test_load_registry_strict_rejects_invalid_rows(overrides: dict[str, Any]) -> None:
    _write_raw_registry([_base_row(**overrides)])
    with pytest.raises(claude_accounts.AccountRegistryError):
        claude_accounts.load_registry()


def test_load_registry_strict_rejects_missing_key() -> None:
    row = _base_row()
    del row["state"]
    _write_raw_registry([row])
    with pytest.raises(claude_accounts.AccountRegistryError, match="missing keys"):
        claude_accounts.load_registry()


def test_load_registry_strict_rejects_duplicate_ids() -> None:
    shared_id = str(uuid.uuid4())
    rows = [
        _base_row(id=shared_id, email="a@example.com"),
        _base_row(id=shared_id, email="b@example.com"),
    ]
    _write_raw_registry(rows)
    with pytest.raises(claude_accounts.AccountRegistryError, match="duplicate id"):
        claude_accounts.load_registry()


def test_load_registry_strict_rejects_duplicate_identity_keys() -> None:
    rows = [
        _base_row(email="same@example.com", organizationUuid=None),
        _base_row(email="same@example.com", organizationUuid=None),
    ]
    _write_raw_registry(rows)
    with pytest.raises(claude_accounts.AccountRegistryError, match="duplicate account identity"):
        claude_accounts.load_registry()


def test_load_registry_strict_rejects_non_canonical_id() -> None:
    _write_raw_registry([_base_row(id="../../etc/passwd")])
    with pytest.raises(claude_accounts.AccountRegistryError, match="row 0"):
        claude_accounts.load_registry()


def test_load_registry_accepts_exact_current_schema_without_rewriting() -> None:
    _write_raw_registry([_base_row()])
    mtime_before = _registry_path().stat().st_mtime_ns

    [record] = claude_accounts.load_registry()

    assert record.account_incarnation_id
    assert _registry_path().stat().st_mtime_ns == mtime_before


def test_empty_registry_array_is_valid_current_state_and_not_rewritten() -> None:
    _write_raw_registry([])
    mtime_before = _registry_path().stat().st_mtime_ns

    assert claude_accounts.load_registry() == []
    assert claude_accounts.load_registry() == []

    assert _registry_path().stat().st_mtime_ns == mtime_before


def test_load_registry_rejects_exact_eight_key_row_without_mutating_file() -> None:
    _write_raw_registry([_eight_key_row()])
    registry_path = _registry_path()
    bytes_before = registry_path.read_bytes()
    mtime_before = registry_path.stat().st_mtime_ns

    with pytest.raises(claude_accounts.AccountRegistryError) as exc_info:
        claude_accounts.load_registry()

    detail = str(exc_info.value)
    assert "accountIncarnationId" in detail
    assert "upstreamAccountUuid" in detail
    assert registry_path.read_bytes() == bytes_before
    assert registry_path.stat().st_mtime_ns == mtime_before


# --------------------------------------------------------------------------
# Incarnation identity: assignment and the reauthentication transition table
# --------------------------------------------------------------------------


def test_add_account_assigns_incarnation_and_captures_upstream_account_uuid() -> None:
    upstream_uuid = str(uuid.uuid4())
    record = _add(oauth_account_json={"accountUuid": upstream_uuid})

    assert record.upstream_account_uuid == upstream_uuid
    uuid.UUID(record.account_incarnation_id)  # a fresh canonical incarnation id was assigned


def test_add_account_without_capturable_upstream_uuid_still_assigns_incarnation() -> None:
    record = _add(oauth_account_json={"accountUuid": "not-a-uuid"})

    assert record.upstream_account_uuid is None
    uuid.UUID(record.account_incarnation_id)


def test_reauth_same_uuid_keeps_incarnation() -> None:
    upstream_uuid = str(uuid.uuid4())
    original = _add(organization_uuid="org-1", oauth_account_json={"accountUuid": upstream_uuid})

    updated = claude_accounts.update_account_credentials(
        "user@example.com", "org-1", None, {"accessToken": "at-2"}, {"accountUuid": upstream_uuid}
    )

    assert updated.account_incarnation_id == original.account_incarnation_id
    assert updated.upstream_account_uuid == upstream_uuid


def test_reauth_null_to_known_upstream_uuid_keeps_incarnation_and_stores_it() -> None:
    original = _add(organization_uuid="org-1", oauth_account_json=None)
    assert original.upstream_account_uuid is None
    upstream_uuid = str(uuid.uuid4())

    updated = claude_accounts.update_account_credentials(
        "user@example.com", "org-1", None, {"accessToken": "at-2"}, {"accountUuid": upstream_uuid}
    )

    assert updated.account_incarnation_id == original.account_incarnation_id
    assert updated.upstream_account_uuid == upstream_uuid


def test_reauth_known_to_missing_upstream_uuid_preserves_uuid_and_incarnation() -> None:
    upstream_uuid = str(uuid.uuid4())
    original = _add(organization_uuid="org-1", oauth_account_json={"accountUuid": upstream_uuid})

    updated = claude_accounts.update_account_credentials(
        "user@example.com", "org-1", None, {"accessToken": "at-2"}, None
    )

    assert updated.account_incarnation_id == original.account_incarnation_id
    assert updated.upstream_account_uuid == upstream_uuid  # never erased by a failed capture


def test_reauth_both_null_upstream_uuid_remain_null_and_keep_incarnation() -> None:
    original = _add(organization_uuid="org-1", oauth_account_json=None)

    updated = claude_accounts.update_account_credentials(
        "user@example.com", "org-1", None, {"accessToken": "at-2"}, None
    )

    assert updated.account_incarnation_id == original.account_incarnation_id
    assert updated.upstream_account_uuid is None


def test_reauth_valid_uuid_change_rotates_incarnation() -> None:
    original = _add(
        organization_uuid="org-1", oauth_account_json={"accountUuid": str(uuid.uuid4())}
    )
    new_upstream_uuid = str(uuid.uuid4())

    updated = claude_accounts.update_account_credentials(
        "user@example.com",
        "org-1",
        None,
        {"accessToken": "at-2"},
        {"accountUuid": new_upstream_uuid},
    )

    assert updated.account_incarnation_id != original.account_incarnation_id
    assert updated.upstream_account_uuid == new_upstream_uuid


# --------------------------------------------------------------------------
# remove_account, including the tombstone protocol
# --------------------------------------------------------------------------


def test_remove_unknown_id_raises_not_found() -> None:
    with pytest.raises(claude_accounts.AccountNotFoundError):
        claude_accounts.remove_account(str(uuid.uuid4()))


def test_remove_account_rejects_non_canonical_id_path_traversal() -> None:
    record = _add()
    with pytest.raises(claude_accounts.AccountNotFoundError):
        claude_accounts.remove_account(f"../{record.id}")
    # Zero state change: the legitimate account is still fully present.
    assert (_accounts_root() / record.id).is_dir()
    assert [r.id for r in claude_accounts.list_accounts()] == [record.id]


def test_remove_account_full_round_trip_purges_the_tombstone() -> None:
    record = _add()
    account_dir = _accounts_root() / record.id
    claude_accounts.remove_account(record.id)
    assert claude_accounts.list_accounts() == []
    assert not account_dir.exists()
    assert not (_accounts_root() / f"{record.id}.tombstone").exists()


def test_remove_account_missing_directory_is_not_an_error() -> None:
    record = _add()
    shutil.rmtree(_accounts_root() / record.id)
    claude_accounts.remove_account(record.id)
    assert claude_accounts.list_accounts() == []


# --------------------------------------------------------------------------
# File / directory modes (POSIX)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode bits are not meaningful on Windows"
)
def test_file_and_directory_modes_are_owner_only_on_posix() -> None:
    record = _add(oauth_account_json=_oauth_payload())
    root = _accounts_root()
    account_dir = root / record.id

    assert stat.S_IMODE(paths.runtime_dir().stat().st_mode) == 0o700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(account_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "registry.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "registry.lock").stat().st_mode) == 0o600
    assert stat.S_IMODE((account_dir / "credentials.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((account_dir / "oauth-account.json").stat().st_mode) == 0o600


# --------------------------------------------------------------------------
# Rollback: injected failures before the registry commit point
# --------------------------------------------------------------------------


def test_rollback_on_credentials_write_failure_leaves_no_partial_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = claude_accounts._write_json_atomic
    failer = _FailOnceOnMatch(real_write, lambda path, data: path.name == "credentials.json")
    monkeypatch.setattr(claude_accounts, "_write_json_atomic", failer)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    _assert_pristine()


def test_rollback_on_staging_rename_failure_leaves_no_partial_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failer = _FailOnceOnMatch(os.rename, lambda src, dst: Path(src).name.startswith(".staging-"))
    monkeypatch.setattr(os, "rename", failer)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    _assert_pristine()


def test_rollback_on_registry_write_failure_leaves_no_partial_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = claude_accounts._write_json_atomic
    failer = _FailOnceOnMatch(real_write, lambda path, data: path.name == "registry.json")
    monkeypatch.setattr(claude_accounts, "_write_json_atomic", failer)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    _assert_pristine()


def test_rollback_on_precommit_fsync_barrier_failure_removes_the_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 1st `accounts/claude/` fsync is the pre-commit durability barrier,
    # strictly before the registry commit point.
    wrapped, call_count = _fsync_directory_failing_on_nth_root_call(fail_on_call=1)
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="failed to add the account"):
        _add()

    # The rollback itself used the (now working) real fsync, so it must have
    # fully cleaned up: no staging dir, no canonical dir, registry untouched.
    assert call_count["n"] == 2
    _assert_pristine()


# --------------------------------------------------------------------------
# Post-commit fsync failures: durability uncertain, never rolled back
# --------------------------------------------------------------------------


def test_add_post_commit_fsync_failure_reports_uncertain_but_keeps_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 2nd `accounts/claude/` fsync is the registry writer's own
    # post-`os.replace` fsync — strictly after the registry commit point.
    wrapped, _call_count = _fsync_directory_failing_on_nth_root_call(fail_on_call=2)
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="durability"):
        _add()

    # The registry os.replace() already succeeded: the account is committed
    # even though the caller sees an error about durability.
    [listed] = claude_accounts.list_accounts()
    assert listed.email == "user@example.com"


def test_remove_post_commit_fsync_failure_leaves_the_tombstone_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _add()
    real_fsync_directory = claude_accounts._fsync_directory
    # Call order for remove: 1st `accounts/claude/` fsync is the
    # tombstone-rename fsync; the 2nd is the registry writer's own
    # post-`os.replace` fsync — the boundary this test targets.
    wrapped, _call_count = _fsync_directory_failing_on_nth_root_call(fail_on_call=2)
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="durability"):
        claude_accounts.remove_account(record.id)

    tombstone = _accounts_root() / f"{record.id}.tombstone"
    assert tombstone.is_dir()
    assert not (_accounts_root() / record.id).exists()
    assert claude_accounts.load_registry() == []

    # Restore the real fsync (without undoing the HOME isolation the shared
    # `monkeypatch` fixture also holds) before letting the next mutation's
    # crash recovery resolve the leftover tombstone: the registry already
    # lacks the row, so it must be purged, not restored.
    monkeypatch.setattr(claude_accounts, "_fsync_directory", real_fsync_directory)
    other = _add("other@example.com")
    assert not tombstone.exists()
    assert {r.id for r in claude_accounts.list_accounts()} == {other.id}


def test_add_interrupt_after_registry_replace_never_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-OSError interruption (KeyboardInterrupt) landing after the
    # registry os.replace() must be classified as post-commit: the account
    # directory stays, the committed registry row stays.
    wrapped, _call_count = _fsync_directory_failing_on_nth_root_call(
        fail_on_call=2, exception_factory=KeyboardInterrupt
    )
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="durability"):
        _add()

    [listed] = claude_accounts.list_accounts()
    assert listed.email == "user@example.com"
    assert (_accounts_root() / listed.id).is_dir()


def test_remove_interrupt_after_registry_replace_keeps_the_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _add()
    wrapped, _call_count = _fsync_directory_failing_on_nth_root_call(
        fail_on_call=2, exception_factory=KeyboardInterrupt
    )
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="durability"):
        claude_accounts.remove_account(record.id)

    # Post-commit: the tombstone is neither restored nor purged, and the
    # registry no longer contains the row.
    assert (_accounts_root() / f"{record.id}.tombstone").is_dir()
    assert not (_accounts_root() / record.id).exists()
    assert claude_accounts.load_registry() == []


def test_remove_precommit_tombstone_fsync_failure_restores_the_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _add()
    # The 1st `accounts/claude/` fsync during remove is the pre-commit
    # tombstone durability barrier: the registry still references the
    # account, so a failure here must rename the tombstone back.
    wrapped, _call_count = _fsync_directory_failing_on_nth_root_call(fail_on_call=1)
    monkeypatch.setattr(claude_accounts, "_fsync_directory", wrapped)

    with pytest.raises(claude_accounts.AccountRegistryError, match="failed to tombstone"):
        claude_accounts.remove_account(record.id)

    assert (_accounts_root() / record.id).is_dir()
    assert not (_accounts_root() / f"{record.id}.tombstone").exists()
    [listed] = claude_accounts.list_accounts()
    assert listed.id == record.id


def _interrupt_registry_replace_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch os.replace so the REGISTRY replacement really happens and then
    raises KeyboardInterrupt — modeling an async interrupt landing at the
    replacement boundary where the commit outcome is unknowable."""
    real_replace = os.replace

    def _replace_then_interrupt(src: object, dst: object, **kwargs: object) -> None:
        real_replace(src, dst, **kwargs)  # type: ignore[arg-type]
        if Path(str(dst)).name == "registry.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", _replace_then_interrupt)


def test_add_interrupt_at_the_replace_boundary_never_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _interrupt_registry_replace_after_success(monkeypatch)

    with pytest.raises(claude_accounts.AccountRegistryError, match="uncertain|unknown"):
        _add()

    # The replace completed before the interrupt: the account directory and
    # the committed registry row must both survive.
    [listed] = claude_accounts.list_accounts()
    assert listed.email == "user@example.com"
    assert (_accounts_root() / listed.id).is_dir()


def test_remove_interrupt_at_the_replace_boundary_keeps_the_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _add()
    _interrupt_registry_replace_after_success(monkeypatch)

    with pytest.raises(claude_accounts.AccountRegistryError, match="uncertain|unknown"):
        claude_accounts.remove_account(record.id)

    # Outcome-unknown at the boundary: the tombstone is neither restored nor
    # purged, and the replaced registry no longer contains the row.
    assert (_accounts_root() / f"{record.id}.tombstone").is_dir()
    assert not (_accounts_root() / record.id).exists()
    assert claude_accounts.load_registry() == []


def test_tombstone_purge_failure_surfaces_an_error_while_the_row_stays_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _add()
    real_remove_tree = claude_accounts._remove_directory_tree
    failer = _FailOnceOnMatch(real_remove_tree, lambda path: path.name.endswith(".tombstone"))
    monkeypatch.setattr(claude_accounts, "_remove_directory_tree", failer)

    with pytest.raises(claude_accounts.AccountRegistryError, match="tombstone"):
        claude_accounts.remove_account(record.id)

    assert claude_accounts.load_registry() == []
    assert (_accounts_root() / f"{record.id}.tombstone").exists()


# --------------------------------------------------------------------------
# Crash recovery: explicit tombstone-state reconciliation
# --------------------------------------------------------------------------


def test_crash_recovery_restores_tombstone_when_registry_still_has_the_row() -> None:
    record = _add()
    final_dir = _accounts_root() / record.id
    tombstone = _accounts_root() / f"{record.id}.tombstone"
    final_dir.rename(tombstone)  # simulate a crash between rename and registry replace

    other = _add("other@example.com")

    assert not tombstone.exists()
    assert final_dir.is_dir()
    assert {r.id for r in claude_accounts.list_accounts()} == {record.id, other.id}


def test_crash_recovery_purges_tombstone_when_registry_lacks_the_row() -> None:
    record = _add()
    final_dir = _accounts_root() / record.id
    tombstone = _accounts_root() / f"{record.id}.tombstone"
    final_dir.rename(tombstone)
    # Simulate the registry having already been committed without the row.
    claude_accounts._write_json_atomic(_registry_path(), [])

    other = _add("other@example.com")

    assert not tombstone.exists()
    assert {r.id for r in claude_accounts.list_accounts()} == {other.id}


def test_crash_recovery_both_final_dir_and_tombstone_present_is_inconsistency_error() -> None:
    record = _add()
    final_dir = _accounts_root() / record.id
    tombstone = _accounts_root() / f"{record.id}.tombstone"
    shutil.copytree(final_dir, tombstone)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add("other@example.com")


# --------------------------------------------------------------------------
# Missing registry + leftover artifacts
# --------------------------------------------------------------------------


def test_missing_registry_with_tombstone_only_raises_and_deletes_nothing() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    tombstone = root / f"{uuid.uuid4()}.tombstone"
    tombstone.mkdir(mode=0o700)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    assert tombstone.exists()


def test_missing_registry_with_orphan_only_raises_and_deletes_nothing() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    orphan = root / f".orphan-{uuid.uuid4()}"
    orphan.mkdir(mode=0o700)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    assert orphan.exists()


def test_missing_registry_with_final_uuid_dir_only_raises_and_deletes_nothing() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    leftover = root / str(uuid.uuid4())
    leftover.mkdir(mode=0o700)

    with pytest.raises(claude_accounts.AccountRegistryError):
        _add()
    assert leftover.exists()


def test_missing_registry_with_only_stale_staging_dirs_is_swept_and_succeeds() -> None:
    root = _accounts_root()
    root.mkdir(parents=True)
    stale_staging = root / ".staging-leftover"
    stale_staging.mkdir(mode=0o700)

    record = _add()
    assert not stale_staging.exists()
    assert [r.id for r in claude_accounts.list_accounts()] == [record.id]


# --------------------------------------------------------------------------
# Sweep and orphan reporting
# --------------------------------------------------------------------------


def test_sweep_removes_only_staging_directories() -> None:
    record = _add()
    root = _accounts_root()
    stray_staging = root / ".staging-stray"
    stray_staging.mkdir(mode=0o700)
    stray_other = root / ".not-a-staging-dir"
    stray_other.mkdir(mode=0o700)

    other = _add("other@example.com")

    assert not stray_staging.exists()
    assert stray_other.exists()  # sweep only ever touches ".staging-*"
    assert {r.id for r in claude_accounts.list_accounts()} == {record.id, other.id}


def test_orphaned_uuid_directory_is_preserved_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = _add()
    orphan_id = str(uuid.uuid4())
    orphan_dir = _accounts_root() / orphan_id
    orphan_dir.mkdir(mode=0o700)
    (orphan_dir / "credentials.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="claudex_gateway.claude_accounts"):
        other = _add("other@example.com")

    assert orphan_dir.is_dir()  # never auto-deleted
    assert any(orphan_id in message for message in caplog.messages)
    assert {r.id for r in claude_accounts.list_accounts()} == {record.id, other.id}


# --------------------------------------------------------------------------
# Concurrency: real subprocess writers
# --------------------------------------------------------------------------

_ADD_SCRIPT = """
import sys
from claudex_gateway import claude_accounts as ca

try:
    record = ca.add_account(sys.argv[1], None, None, {"accessToken": "at"}, None)
    print("ok", record.id)
except ca.DuplicateAccountError:
    print("duplicate")
    sys.exit(3)
"""


def test_concurrent_distinct_adds_both_succeed(tmp_path: Path) -> None:
    env = _child_env(tmp_path)
    p1 = subprocess.Popen(
        [sys.executable, "-c", _ADD_SCRIPT, "one@example.com"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", _ADD_SCRIPT, "two@example.com"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)
    assert p1.returncode == 0, (out1, err1)
    assert p2.returncode == 0, (out2, err2)

    emails = {record.email for record in claude_accounts.list_accounts()}
    assert emails == {"one@example.com", "two@example.com"}


def test_concurrent_duplicate_adds_only_one_succeeds(tmp_path: Path) -> None:
    env = _child_env(tmp_path)
    p1 = subprocess.Popen(
        [sys.executable, "-c", _ADD_SCRIPT, "dup@example.com"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", _ADD_SCRIPT, "dup@example.com"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)

    assert {p1.returncode, p2.returncode} == {0, 3}, (out1, err1, out2, err2)

    records = claude_accounts.list_accounts()
    assert len(records) == 1
    assert records[0].email == "dup@example.com"


# --------------------------------------------------------------------------
# Crash recovery: subprocess SIGKILL at each commit boundary
# --------------------------------------------------------------------------

_CRASH_AFTER_TOMBSTONE_RENAME_SCRIPT = """
import os
import signal
import sys

real_rename = os.rename

def crashing_rename(src, dst):
    real_rename(src, dst)
    if str(dst).endswith(".tombstone"):
        os.kill(os.getpid(), signal.SIGKILL)

os.rename = crashing_rename

from claudex_gateway import claude_accounts as ca
ca.remove_account(sys.argv[1])
print("should not reach here")
"""

_CRASH_AFTER_REGISTRY_REPLACE_SCRIPT = """
import os
import signal
import sys

real_replace = os.replace

def crashing_replace(src, dst):
    real_replace(src, dst)
    if str(dst).endswith("registry.json"):
        os.kill(os.getpid(), signal.SIGKILL)

os.replace = crashing_replace

from claudex_gateway import claude_accounts as ca

if sys.argv[1] == "add":
    ca.add_account(sys.argv[2], None, None, {"accessToken": "at"}, None)
else:
    ca.remove_account(sys.argv[2])
print("should not reach here")
"""

_CRASH_AFTER_STAGING_RENAME_SCRIPT = """
import os
import signal
import sys

real_rename = os.rename

def crashing_rename(src, dst):
    real_rename(src, dst)
    if os.path.basename(str(src)).startswith(".staging-"):
        os.kill(os.getpid(), signal.SIGKILL)

os.rename = crashing_rename

from claudex_gateway import claude_accounts as ca
ca.add_account(sys.argv[1], None, None, {"accessToken": "at"}, None)
print("should not reach here")
"""


def test_sigkill_after_tombstone_rename_crash_recovery_restores_it(tmp_path: Path) -> None:
    record = _add()
    env = _child_env(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_TOMBSTONE_RENAME_SCRIPT, record.id],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr

    tombstone = _accounts_root() / f"{record.id}.tombstone"
    assert tombstone.is_dir()
    assert claude_accounts.load_registry()[0].id == record.id  # row still present

    other = _add("other@example.com")

    assert not tombstone.exists()
    assert (_accounts_root() / record.id).is_dir()
    assert {r.id for r in claude_accounts.list_accounts()} == {record.id, other.id}


def test_sigkill_after_registry_replace_in_remove_crash_recovery_purges_tombstone(
    tmp_path: Path,
) -> None:
    record = _add()
    env = _child_env(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_REGISTRY_REPLACE_SCRIPT, "remove", record.id],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr

    tombstone = _accounts_root() / f"{record.id}.tombstone"
    assert tombstone.is_dir()
    assert claude_accounts.load_registry() == []  # the commit already happened

    other = _add("other@example.com")

    assert not tombstone.exists()
    assert {r.id for r in claude_accounts.list_accounts()} == {other.id}


def test_sigkill_after_registry_replace_in_add_crash_recovery_is_a_no_op(tmp_path: Path) -> None:
    env = _child_env(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_REGISTRY_REPLACE_SCRIPT, "add", "user@example.com"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr

    [record] = claude_accounts.load_registry()
    assert (_accounts_root() / record.id).is_dir()

    other = _add("other@example.com")
    assert {r.id for r in claude_accounts.list_accounts()} == {record.id, other.id}


def test_sigkill_after_staging_rename_in_add_leaves_a_reported_orphan(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A registry must already exist and be valid for a leftover canonical
    # directory to be treated as a soft "orphan" to report: with no registry
    # at all yet, the same leftover is instead a blocking inconsistency (see
    # test_missing_registry_with_final_uuid_dir_only_raises_and_deletes_nothing).
    first = _add("first@example.com")

    env = _child_env(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CRASH_AFTER_STAGING_RENAME_SCRIPT, "other@example.com"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stdout + proc.stderr

    assert [r.id for r in claude_accounts.load_registry()] == [first.id]  # 2nd commit never happened
    root = _accounts_root()
    leftover_dirs = [
        entry
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != first.id
    ]
    assert len(leftover_dirs) == 1  # the orphaned canonical-UUID directory
    orphan_id = leftover_dirs[0].name

    with caplog.at_level(logging.WARNING, logger="claudex_gateway.claude_accounts"):
        record = _add("third@example.com")

    assert leftover_dirs[0].exists()  # never auto-deleted
    assert any(orphan_id in message for message in caplog.messages)
    assert {r.id for r in claude_accounts.list_accounts()} == {first.id, record.id}
