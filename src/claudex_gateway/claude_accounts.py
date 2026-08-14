"""Persistent registry of registered Claude accounts (Orca-style model: account
metadata apart from secrets — design: `.docs/design/multi-account-pool.md` §2/§7).

Storage layout, all rooted at `paths.accounts_dir("claude")`::

    accounts/claude/
    ├── registry.json      # bare JSON array of rows, no secrets, mode 0600
    ├── registry.lock      # cross-process mutation lock (claudex_gateway.locking)
    └── <uuid>/            # one directory per account, mode 0700
        ├── credentials.json      # mode 0600
        └── oauth-account.json    # mode 0600

Row JSON (exact key set, camelCase): `id`, `email`, `organizationUuid`,
`organizationName`, `createdAt`, `updatedAt`, `lastAuthenticatedAt`, `state`,
`accountIncarnationId`, `upstreamAccountUuid`. This exact 10-key schema is the
only supported persisted format; every other row shape is rejected outright.

`accountIncarnationId` is a random id assigned once per "distinct login" and
never reused: it survives ordinary reauthentication (same or newly-learned
upstream account) but rotates when reauthentication captures a *different*
valid `upstreamAccountUuid` than the one already on file — see
`_resolve_reauth_incarnation`. `upstreamAccountUuid` is the canonical
(lowercase, hyphenated) Anthropic account uuid captured from
`oauth-account.json`'s `accountUuid`, or `None` when it cannot be
established; once known, it is never erased by a later capture that fails.

Registry mutations (`add_account`, `remove_account`) happen only from CLI
processes and are serialized across processes by `locking.file_lock` on
`registry.lock`, held for the complete read/check/write of each mutation.
`add_account` stages the new account directory under a `.staging-<uuid4>`
name and installs it atomically before the registry is ever allowed to
reference it; `remove_account` renames the directory to a `<id>.tombstone`
name before removing the registry row, so a crash between the two steps is
always recoverable. In both cases the registry's own `os.replace` is the
sole transaction commit point (see `add_account`/`remove_account`
docstrings). Crash recovery reconciles any leftover `.staging-*` or
`*.tombstone` directories at the start of every mutation, under the lock.

This module is the storage layer only: no CLI, no OAuth capture flow, no
prompting, and no credential values ever appear in an exception message or
a log line.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from claudex_gateway import paths
from claudex_gateway.locking import file_lock

logger = logging.getLogger(__name__)

_PROVIDER = "claude"
_REGISTRY_FILENAME = "registry.json"
_LOCK_FILENAME = "registry.lock"
_CREDENTIALS_FILENAME = "credentials.json"
_OAUTH_ACCOUNT_FILENAME = "oauth-account.json"
_TOMBSTONE_SUFFIX = ".tombstone"
_STAGING_PREFIX = ".staging-"

_STATE_READY = "ready"
_STATE_NEEDS_REAUTH = "needs-reauth"
_VALID_STATES = (_STATE_READY, _STATE_NEEDS_REAUTH)

_ROW_KEYS = (
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
)
_ROW_KEYS_SET = frozenset(_ROW_KEYS)

_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


class AccountRegistryError(Exception):
    """Raised for registry-level failures: malformed files, I/O, or a broken
    on-disk invariant. Messages never contain credential values or raw file
    content — only paths, ids, and a description of what went wrong."""


class DuplicateAccountError(AccountRegistryError):
    """Raised by `add_account` when `(email, organizationUuid)` is already
    registered."""


class AccountNotFoundError(AccountRegistryError):
    """Raised by `remove_account` when the id is not a canonical UUID, or is
    canonical but not currently registered."""


@dataclass(frozen=True)
class AccountRecord:
    id: str
    email: str
    organization_uuid: str | None
    organization_name: str | None
    created_at: int
    updated_at: int
    last_authenticated_at: int
    state: str
    # A random id assigned once per distinct login and never reused; see the
    # module docstring and `_resolve_reauth_incarnation` for the exact
    # survive-vs-rotate rule across reauthentication.
    account_incarnation_id: str
    # The canonical (lowercase, hyphenated) Anthropic account uuid captured
    # from `oauth-account.json`'s `accountUuid`, or `None` when it has never
    # been established. Once known, never erased by a later capture that
    # fails to establish it again.
    upstream_account_uuid: str | None

    def to_row(self) -> dict[str, Any]:
        """The exact camelCase JSON shape persisted in `registry.json`."""
        return {
            "id": self.id,
            "email": self.email,
            "organizationUuid": self.organization_uuid,
            "organizationName": self.organization_name,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "lastAuthenticatedAt": self.last_authenticated_at,
            "state": self.state,
            "accountIncarnationId": self.account_incarnation_id,
            "upstreamAccountUuid": self.upstream_account_uuid,
        }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_registry() -> list[AccountRecord]:
    """Load and strictly validate `registry.json`, in on-disk row order.

    Returns an empty list when the file does not exist, or when it exists
    and holds an empty JSON array — both are valid current state and are
    never rewritten just to be read. This is a plain read: no lock is taken
    and no crash recovery runs (both are the concern of the mutating
    operations below), matching an atomically-replaced file a reader can
    always see in either its old or new complete state.
    """
    accounts_root = paths.accounts_dir(_PROVIDER)
    registry_path = _registry_path(accounts_root)
    parsed = _read_registry_array(registry_path)
    return _parse_current_rows(parsed, path=registry_path)


def list_accounts() -> list[AccountRecord]:
    """Return every registered account sorted by `(createdAt, id)`."""
    return sorted(load_registry(), key=lambda record: (record.created_at, record.id))


def add_account(
    email: str,
    organization_uuid: str | None,
    organization_name: str | None,
    credentials_json: dict[str, Any],
    oauth_account_json: dict[str, Any] | None,
) -> AccountRecord:
    """Register a new Claude account (staged-directory commit protocol).

    Every input is validated and normalized before any filesystem write.
    `credentials_json` must already be a parsed JSON object; `oauth_account_json`
    must be a parsed JSON object or `None` (persisted on disk as `{}` when
    `None`) — any other type is rejected before any filesystem mutation.

    The new account directory is written under a `.staging-<uuid4>` name,
    renamed to its canonical `<id>` name, and `accounts/claude/` is fsynced
    (durability barrier) *before* the registry is touched, so a committed
    registry row can never outlive its account directory. The registry's own
    `os.replace` — performed by the shared atomic JSON writer — is the sole
    commit point: any failure before it returns is rolled back by removing
    the owned staging/canonical directory; any failure strictly after it
    returns (the writer's own post-replace directory fsync) never rolls the
    directory back, since the account may already be visible to readers.
    """
    (
        normalized_email,
        normalized_organization_uuid,
        normalized_organization_name,
        oauth_payload,
    ) = _validate_account_inputs(
        email, organization_uuid, organization_name, credentials_json, oauth_account_json
    )

    accounts_root = paths.accounts_dir(_PROVIDER)
    _makedirs_0700(paths.runtime_dir())
    _makedirs_0700(accounts_root)

    with file_lock(_lock_path(accounts_root)):
        records = _recover(accounts_root)
        for existing in records:
            if (
                existing.email == normalized_email
                and existing.organization_uuid == normalized_organization_uuid
            ):
                raise DuplicateAccountError(
                    f"an account is already registered for {normalized_email!r}"
                )

        now = _now_millis()
        account_id = str(uuid.uuid4())
        record = AccountRecord(
            id=account_id,
            email=normalized_email,
            organization_uuid=normalized_organization_uuid,
            organization_name=normalized_organization_name,
            created_at=now,
            updated_at=now,
            last_authenticated_at=now,
            state=_STATE_READY,
            account_incarnation_id=str(uuid.uuid4()),
            upstream_account_uuid=_derive_upstream_account_uuid(oauth_payload),
        )

        staging_dir = accounts_root / f"{_STAGING_PREFIX}{uuid.uuid4()}"
        try:
            _create_new_directory(staging_dir, mode=0o700)
            _write_json_atomic(staging_dir / _CREDENTIALS_FILENAME, credentials_json)
            _write_json_atomic(staging_dir / _OAUTH_ACCOUNT_FILENAME, oauth_payload)
        except BaseException as exc:
            _rollback_new_account_directory(staging_dir, accounts_root, exc)

        final_dir = accounts_root / account_id
        try:
            os.rename(staging_dir, final_dir)
        except OSError as exc:
            _rollback_new_account_directory(staging_dir, accounts_root, exc)

        try:
            _fsync_directory(accounts_root)
        except OSError as exc:
            # Durability barrier, still strictly before the registry commit
            # point: a full rollback of the canonical directory is safe.
            _rollback_new_account_directory(final_dir, accounts_root, exc)

        registry_path = _registry_path(accounts_root)
        try:
            _write_json_atomic(registry_path, [r.to_row() for r in [*records, record]])
        except _PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {account_id} may already be registered in {registry_path}; "
                "registry durability after commit is uncertain"
            ) from exc
        except BaseException as exc:
            _rollback_new_account_directory(final_dir, accounts_root, exc)

        return record


def _validate_account_inputs(
    email: str,
    organization_uuid: str | None,
    organization_name: str | None,
    credentials_json: dict[str, Any],
    oauth_account_json: dict[str, Any] | None,
) -> tuple[str, str | None, str | None, dict[str, Any]]:
    """Normalize identity fields and validate payload shapes, strictly before
    any filesystem write. Returns `(email, organizationUuid,
    organizationName, oauth_payload)` with `oauth_payload` defaulted to `{}`
    when `oauth_account_json` is `None`."""
    try:
        normalized_email = _normalize_email(email)
        normalized_organization_uuid = _normalize_optional_text(
            organization_uuid, field="organizationUuid"
        )
        normalized_organization_name = _normalize_optional_text(
            organization_name, field="organizationName"
        )
    except ValueError as exc:
        raise AccountRegistryError(str(exc)) from exc
    if not isinstance(credentials_json, dict):
        raise AccountRegistryError("credentials_json must be a JSON object")
    if oauth_account_json is not None and not isinstance(oauth_account_json, dict):
        raise AccountRegistryError("oauth_account_json must be a JSON object or null")
    oauth_payload: dict[str, Any] = oauth_account_json if oauth_account_json is not None else {}
    return (
        normalized_email,
        normalized_organization_uuid,
        normalized_organization_name,
        oauth_payload,
    )


def update_account_credentials(
    email: str,
    organization_uuid: str | None,
    organization_name: str | None,
    credentials_json: dict[str, Any],
    oauth_account_json: dict[str, Any] | None,
) -> AccountRecord:
    """Replace a registered account's stored credentials in place (re-auth).

    The account is addressed by the same `(email, organizationUuid)`
    identity key that `add_account` refuses duplicates on; raises
    `AccountNotFoundError` when no account matches. The row keeps its `id`
    and `createdAt` — so a `claude_account.id` selection keeps working —
    while `updatedAt` and `lastAuthenticatedAt` are bumped, `state` resets
    to ready, and `organizationName` is refreshed from the new capture.
    `accountIncarnationId`/`upstreamAccountUuid` follow the reauthentication
    transition table in `_resolve_reauth_incarnation`: ordinary
    reauthentication (same or not-yet-known upstream uuid, or a capture that
    fails to establish one) keeps the incarnation; only a newly captured
    valid upstream uuid that differs from a previously known one rotates it.

    Both credential files are replaced (each atomically) strictly BEFORE the
    registry row is rewritten: a crash between the two leaves fresh working
    credentials under a stale row, never a bumped row over stale
    credentials. A half-applied file pair is not rolled back — the freshly
    captured credentials are strictly newer than what they replaced, and
    re-running the same update reconverges.
    """
    (
        normalized_email,
        normalized_organization_uuid,
        normalized_organization_name,
        oauth_payload,
    ) = _validate_account_inputs(
        email, organization_uuid, organization_name, credentials_json, oauth_account_json
    )

    accounts_root = paths.accounts_dir(_PROVIDER)
    _makedirs_0700(paths.runtime_dir())
    _makedirs_0700(accounts_root)

    with file_lock(_lock_path(accounts_root)):
        records = _recover(accounts_root)
        existing = next(
            (
                record
                for record in records
                if record.email == normalized_email
                and record.organization_uuid == normalized_organization_uuid
            ),
            None,
        )
        if existing is None:
            raise AccountNotFoundError(f"no account registered for {normalized_email!r}")

        now = _now_millis()
        newly_captured_upstream_account_uuid = _derive_upstream_account_uuid(oauth_payload)
        account_incarnation_id, upstream_account_uuid = _resolve_reauth_incarnation(
            existing, newly_captured_upstream_account_uuid
        )
        updated = AccountRecord(
            id=existing.id,
            email=normalized_email,
            organization_uuid=normalized_organization_uuid,
            organization_name=normalized_organization_name,
            created_at=existing.created_at,
            updated_at=now,
            last_authenticated_at=now,
            state=_STATE_READY,
            account_incarnation_id=account_incarnation_id,
            upstream_account_uuid=upstream_account_uuid,
        )

        account_dir = accounts_root / existing.id
        try:
            _write_json_atomic(account_dir / _CREDENTIALS_FILENAME, credentials_json)
            _write_json_atomic(account_dir / _OAUTH_ACCOUNT_FILENAME, oauth_payload)
        except _PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"credentials for account {existing.id} were replaced with uncertain "
                "durability; re-run `account add` to reconverge"
            ) from exc
        except OSError as exc:
            raise AccountRegistryError(
                f"failed to replace credentials for account {existing.id}"
            ) from exc

        registry_path = _registry_path(accounts_root)
        rows = [
            (updated if record.id == existing.id else record).to_row() for record in records
        ]
        try:
            _write_json_atomic(registry_path, rows)
        except _PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {existing.id} was updated in {registry_path} but durability "
                "after commit is uncertain"
            ) from exc
        except BaseException as exc:
            raise AccountRegistryError(
                f"credentials for account {existing.id} were replaced but the registry "
                f"row update failed; row metadata in {registry_path} is stale"
            ) from exc

        return updated


def mark_account_needs_reauth(account_id: str) -> AccountRecord:
    """Set a registered account's state to needs-reauth (durable fact: only a
    fresh interactive login can recover the account).

    This is the one state transition the daemon performs; recovery to ready
    happens exclusively through `update_account_credentials`, which is why
    this is not a generic state setter. Idempotent: an account already in
    needs-reauth is returned unchanged without a registry write. Bumps
    `updatedAt` only — `lastAuthenticatedAt` records logins, not failures.
    Raises `AccountNotFoundError` for an unknown or non-canonical id.
    """
    canonical_id = _canonicalize_account_id(account_id)

    accounts_root = paths.accounts_dir(_PROVIDER)
    _makedirs_0700(paths.runtime_dir())
    _makedirs_0700(accounts_root)

    with file_lock(_lock_path(accounts_root)):
        records = _recover(accounts_root)
        existing = next((record for record in records if record.id == canonical_id), None)
        if existing is None:
            raise AccountNotFoundError(f"no account registered with id {canonical_id}")
        if existing.state == _STATE_NEEDS_REAUTH:
            return existing

        updated = replace(existing, updated_at=_now_millis(), state=_STATE_NEEDS_REAUTH)
        registry_path = _registry_path(accounts_root)
        rows = [
            (updated if record.id == canonical_id else record).to_row() for record in records
        ]
        try:
            _write_json_atomic(registry_path, rows)
        except _PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {canonical_id} was marked needs-reauth in {registry_path} "
                "but durability after commit is uncertain"
            ) from exc
        except OSError as exc:
            raise AccountRegistryError(
                f"failed to mark account {canonical_id} needs-reauth in {registry_path}"
            ) from exc

        return updated


def remove_account(account_id: str) -> None:
    """Remove a registered account by id (tombstone commit protocol).

    `account_id` is untrusted (CLI input, or a hand-edited registry row) and
    is canonicalized to UUID text before it ever reaches the filesystem: a
    non-canonical id (e.g. containing `../`) is rejected with
    `AccountNotFoundError` and causes no state change whatsoever.

    When the account directory exists it is renamed to `<id>.tombstone`
    before the registry row is dropped. The registry's own `os.replace` is
    the sole commit point, mirroring `add_account`: a failure before it
    returns restores the tombstone; a failure strictly after it returns (the
    writer's own post-replace directory fsync) never restores *or* purges
    the tombstone, since the new registry may already be visible to
    readers — it is left for the next mutation's crash recovery to resolve.
    Only once the registry replace is known-durable is the tombstone purged.
    A missing account directory is not an error: the registry is the source
    of truth.
    """
    canonical_id = _canonicalize_account_id(account_id)
    accounts_root = paths.accounts_dir(_PROVIDER)
    _makedirs_0700(paths.runtime_dir())
    _makedirs_0700(accounts_root)

    with file_lock(_lock_path(accounts_root)):
        records = _recover(accounts_root)
        remaining = [record for record in records if record.id != canonical_id]
        if len(remaining) == len(records):
            raise AccountNotFoundError(f"no account registered with id {canonical_id}")

        final_dir = accounts_root / canonical_id
        tombstone_dir = accounts_root / f"{canonical_id}{_TOMBSTONE_SUFFIX}"
        directory_present = _verify_removable_directory(final_dir, accounts_root)

        if directory_present:
            tombstoned = False
            try:
                os.rename(final_dir, tombstone_dir)
                tombstoned = True
                _fsync_directory(accounts_root)
            except OSError as exc:
                if tombstoned:
                    # The pre-commit durability barrier failed while the
                    # registry still references the account: restore the
                    # canonical directory so registry and disk stay in step.
                    try:
                        os.rename(tombstone_dir, final_dir)
                        _fsync_directory(accounts_root)
                    except OSError as restore_exc:
                        raise AccountRegistryError(
                            f"failed to restore {final_dir} after a tombstone "
                            "durability failure"
                        ) from restore_exc
                raise AccountRegistryError(
                    f"failed to tombstone account directory {final_dir}"
                ) from exc

        registry_path = _registry_path(accounts_root)
        try:
            _write_json_atomic(registry_path, [record.to_row() for record in remaining])
        except _PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {canonical_id} was removed from {registry_path} but durability "
                f"after commit is uncertain; a tombstone may remain at {tombstone_dir}"
            ) from exc
        except BaseException as exc:
            if directory_present:
                try:
                    os.rename(tombstone_dir, final_dir)
                    _fsync_directory(accounts_root)
                except OSError as restore_exc:
                    raise AccountRegistryError(
                        f"failed to restore {final_dir} after a registry write failure"
                    ) from restore_exc
            raise AccountRegistryError(f"failed to remove account {canonical_id}") from exc

        if directory_present:
            try:
                _remove_directory_tree(tombstone_dir)
                _fsync_directory(accounts_root)
            except OSError as exc:
                raise AccountRegistryError(
                    f"account {canonical_id} was removed but the tombstone at "
                    f"{tombstone_dir} could not be purged"
                ) from exc


# --------------------------------------------------------------------------
# Row validation and normalization
# --------------------------------------------------------------------------


def _read_registry_array(path: Path) -> list[object]:
    """Read `registry.json` into its raw, un-row-validated JSON array.

    A missing file degrades to `[]` (absent registry is valid current
    state). Malformed JSON or a non-array root is a file-level error; strict
    row validation is handled by `_parse_current_rows`.
    """
    if not path.exists():
        return []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AccountRegistryError(f"cannot read registry {path}") from exc
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AccountRegistryError(f"registry {path} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise AccountRegistryError(f"registry {path} must contain a JSON array")
    return parsed


def _register_unique_or_raise(
    record_id: str,
    identity: tuple[str, str | None],
    seen_ids: set[str],
    seen_identities: set[tuple[str, str | None]],
    *,
    path: Path,
    index: int,
) -> None:
    if record_id in seen_ids:
        raise AccountRegistryError(f"registry {path}: duplicate id at row {index}")
    seen_ids.add(record_id)
    if identity in seen_identities:
        raise AccountRegistryError(
            f"registry {path}: duplicate account identity at row {index}"
        )
    seen_identities.add(identity)


def _parse_current_rows(parsed: list[object], *, path: Path) -> list[AccountRecord]:
    """Strictly parse rows using the exact current schema."""
    records: list[AccountRecord] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str | None]] = set()
    for index, row in enumerate(parsed):
        record = _parse_row(row, path=path, index=index)
        _register_unique_or_raise(
            record.id,
            (record.email, record.organization_uuid),
            seen_ids,
            seen_identities,
            path=path,
            index=index,
        )
        records.append(record)
    return records


def _check_row_keys(
    row: object, expected_keys: frozenset[str], *, path: Path, index: int
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise AccountRegistryError(f"registry {path}: row {index} must be a JSON object")
    row_keys = set(row)
    unknown_keys = row_keys - expected_keys
    if unknown_keys:
        raise AccountRegistryError(
            f"registry {path}: row {index} has unknown keys: {', '.join(sorted(unknown_keys))}"
        )
    missing_keys = expected_keys - row_keys
    if missing_keys:
        raise AccountRegistryError(
            f"registry {path}: row {index} is missing keys: {', '.join(sorted(missing_keys))}"
        )
    return row


def _validate_shared_fields(
    row: dict[str, Any], *, path: Path, index: int
) -> tuple[str, str, str | None, str | None, int, int, int, str]:
    """Validate the registry fields other than incarnation identity.

    Returns `(id, email, organizationUuid, organizationName, createdAt,
    updatedAt, lastAuthenticatedAt, state)`.
    """
    try:
        record_id = _validate_id(row["id"])
        email = _normalize_email(row["email"])
        organization_uuid = _normalize_optional_text(
            row["organizationUuid"], field="organizationUuid"
        )
        organization_name = _normalize_optional_text(
            row["organizationName"], field="organizationName"
        )
        created_at = _validate_timestamp(row["createdAt"], field="createdAt")
        updated_at = _validate_timestamp(row["updatedAt"], field="updatedAt")
        last_authenticated_at = _validate_timestamp(
            row["lastAuthenticatedAt"], field="lastAuthenticatedAt"
        )
        state = _validate_state(row["state"])
    except ValueError as exc:
        raise AccountRegistryError(f"registry {path}: row {index} is invalid: {exc}") from exc
    return (
        record_id,
        email,
        organization_uuid,
        organization_name,
        created_at,
        updated_at,
        last_authenticated_at,
        state,
    )


def _parse_row(row: object, *, path: Path, index: int) -> AccountRecord:
    """Strictly parse one exact current-schema (10-key) row."""
    checked = _check_row_keys(row, _ROW_KEYS_SET, path=path, index=index)
    (
        record_id,
        email,
        organization_uuid,
        organization_name,
        created_at,
        updated_at,
        last_authenticated_at,
        state,
    ) = _validate_shared_fields(checked, path=path, index=index)
    try:
        account_incarnation_id = _validate_canonical_uuid_field(
            checked["accountIncarnationId"], field="accountIncarnationId"
        )
        upstream_account_uuid = _validate_optional_canonical_uuid_field(
            checked["upstreamAccountUuid"], field="upstreamAccountUuid"
        )
    except ValueError as exc:
        raise AccountRegistryError(f"registry {path}: row {index} is invalid: {exc}") from exc

    return AccountRecord(
        id=record_id,
        email=email,
        organization_uuid=organization_uuid,
        organization_name=organization_name,
        created_at=created_at,
        updated_at=updated_at,
        last_authenticated_at=last_authenticated_at,
        state=state,
        account_incarnation_id=account_incarnation_id,
        upstream_account_uuid=upstream_account_uuid,
    )


def _reject_control_characters(value: str, *, field: str) -> None:
    if _CONTROL_CHARACTER_PATTERN.search(value):
        raise ValueError(f"{field} contains control characters")


def _normalize_email(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("email must be a string")
    _reject_control_characters(raw, field="email")
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("email is required")
    return trimmed.lower()


def _normalize_optional_text(raw: object, *, field: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string or null")
    _reject_control_characters(raw, field=field)
    trimmed = raw.strip()
    return trimmed or None


def _validate_timestamp(raw: object, *, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer epoch-millisecond timestamp")
    if raw < 0:
        raise ValueError(f"{field} must be nonnegative")
    return raw


def _validate_state(raw: object) -> str:
    if raw not in _VALID_STATES:
        raise ValueError(f"state must be one of {_VALID_STATES!r}")
    return raw


def _canonical_uuid_or_none(value: str) -> str | None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    canonical = str(parsed)
    return canonical if canonical == value else None


def _validate_canonical_uuid_field(raw: object, *, field: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a string")
    canonical = _canonical_uuid_or_none(raw)
    if canonical is None:
        raise ValueError(f"{field} must be a canonical UUID")
    return canonical


def _validate_optional_canonical_uuid_field(raw: object, *, field: str) -> str | None:
    if raw is None:
        return None
    return _validate_canonical_uuid_field(raw, field=field)


def _validate_id(raw: object) -> str:
    return _validate_canonical_uuid_field(raw, field="id")


def _is_canonical_uuid(value: str) -> bool:
    return _canonical_uuid_or_none(value) is not None


def _canonicalize_account_id(account_id: object) -> str:
    if not isinstance(account_id, str):
        raise AccountNotFoundError("account id must be a string")
    canonical = _canonical_uuid_or_none(account_id)
    if canonical is None:
        raise AccountNotFoundError(f"{account_id!r} is not a valid account id")
    return canonical


def _canonicalize_upstream_uuid(raw: object) -> str | None:
    """Derive the canonical (lowercase, hyphenated) form of a captured
    `accountUuid` value, or `None` when it is missing, not a string, or not
    a valid UUID. Never raises: a malformed or absent upstream identity must
    never block account registration or reauthentication, it only means the
    upstream uuid could not be established this time."""
    if not isinstance(raw, str):
        return None
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError):
        return None
    return str(parsed)


def _derive_upstream_account_uuid(oauth_payload: dict[str, Any]) -> str | None:
    """The canonical upstream account uuid captured in-memory at
    `add_account`/`update_account_credentials` time, from the same
    `oauth_payload` dict that gets persisted to `oauth-account.json`."""
    return _canonicalize_upstream_uuid(oauth_payload.get("accountUuid"))


def _resolve_reauth_incarnation(
    existing: AccountRecord, newly_captured_upstream_account_uuid: str | None
) -> tuple[str, str | None]:
    """Apply the reauthentication transition table for
    `account_incarnation_id`/`upstream_account_uuid`. Returns the
    `(account_incarnation_id, upstream_account_uuid)` pair the updated row
    should carry.

    Failure to capture a valid upstream uuid this time (`None`) never
    erases a previously known one nor rotates the incarnation — the
    existing values simply carry forward (covers both "known upstream uuid
    plus missing/malformed metadata" and "both null"). A previously unknown
    upstream uuid that is now known keeps the incarnation and records the
    newly captured uuid. The same valid uuid captured again keeps the
    incarnation. Only a newly captured valid uuid that *differs* from a
    previously known valid uuid rotates the incarnation.
    """
    if newly_captured_upstream_account_uuid is None:
        return existing.account_incarnation_id, existing.upstream_account_uuid
    if existing.upstream_account_uuid is None:
        return existing.account_incarnation_id, newly_captured_upstream_account_uuid
    if existing.upstream_account_uuid == newly_captured_upstream_account_uuid:
        return existing.account_incarnation_id, newly_captured_upstream_account_uuid
    return str(uuid.uuid4()), newly_captured_upstream_account_uuid


def _now_millis() -> int:
    return time.time_ns() // 1_000_000


# --------------------------------------------------------------------------
# Secure atomic JSON writer
# --------------------------------------------------------------------------


class _PostCommitFsyncError(Exception):
    """Internal signal from `_write_json_atomic`: the commit either succeeded
    with a later failure (post-replace directory fsync), or its outcome is
    UNKNOWN because an interruption such as `KeyboardInterrupt` landed at the
    replacement boundary itself. Every *other* exception raised by
    `_write_json_atomic` means the replace conclusively never happened —
    callers may run destructive rollback only for those, and must treat this
    type as "committed or outcome unknown": never remove a canonical account
    directory, never restore a tombstone; leave crash recovery to reconcile."""

    def __init__(self, directory: Path, cause: BaseException) -> None:
        super().__init__(
            f"replace into {directory} committed with uncertain durability, "
            "or its outcome is unknown"
        )
        self.directory = directory


def _write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write `data` as JSON to `path` with mode 0600.

    Writes into a fresh, unique, same-directory staging file created with
    `O_CREAT | O_EXCL | O_WRONLY` (so no concurrent writer can ever be
    handed a partially-written file under `path`'s name), `fchmod`s it to
    0600 before a single byte is written as defense in depth against a
    permissive umask, writes + flushes + fsyncs its content, then
    `os.replace`s it over `path` — the sole commit point — and fsyncs the
    containing directory so the rename is durable. Any leftover staging file
    is removed on failure.
    """
    staging_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"

    fd = os.open(staging_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)  # defense in depth: correct for any umask window
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Strictly before the replacement boundary: the commit conclusively
        # never happened, so the staging file may be removed and callers may
        # roll back.
        _remove_path_if_exists(staging_path)
        raise

    # Replacement boundary. From here on, destructive rollback is forbidden:
    # only a synchronous OSError from os.replace proves the commit did not
    # happen. Any other interruption (KeyboardInterrupt, SystemExit) leaves
    # the outcome UNKNOWN — the syscall may or may not have completed — so it
    # is classified with the post-commit marker and crash recovery reconciles.
    try:
        os.replace(staging_path, path)
    except OSError:
        _remove_path_if_exists(staging_path)
        raise
    except BaseException as exc:
        _remove_path_if_exists(staging_path)
        raise _PostCommitFsyncError(path.parent, exc) from exc

    try:
        _fsync_directory(path.parent)
    except BaseException as exc:
        raise _PostCommitFsyncError(path.parent, exc) from exc


def _remove_path_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        # No directory-fsync primitive on Windows; os.replace()'s own
        # metadata journaling is the platform's durability guarantee.
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# Directory helpers
# --------------------------------------------------------------------------


def _registry_path(accounts_root: Path) -> Path:
    return accounts_root / _REGISTRY_FILENAME


def _lock_path(accounts_root: Path) -> Path:
    return accounts_root / _LOCK_FILENAME


def _makedirs_0700(path: Path) -> None:
    """Create `path` and any missing parents with mode 0700.

    `Path.mkdir(parents=True)` only applies the requested mode to the leaf
    directory — any missing parents it creates along the way get the
    umask-influenced default instead. This walks up to the first existing
    ancestor and creates each missing level explicitly with `mkdir(mode=...)`
    followed by an explicit `chmod`, so no directory in the chain (`~/.claudex`,
    the provider directory, ...) is ever left wider than 0700 by a
    permissive umask. Directories that already exist are left untouched.
    """
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)  # defense in depth against a permissive umask


def _create_new_directory(path: Path, *, mode: int) -> None:
    os.mkdir(path, mode)
    os.chmod(path, mode)  # defense in depth against a permissive umask


def _remove_directory_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _is_real_directory(path: Path) -> bool:
    """True iff `path` exists and is a real directory, verified via `lstat`
    so a symlink is never mistaken for (or followed as) the real thing."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(info.st_mode)


def _verify_removable_directory(path: Path, expected_parent: Path) -> bool:
    """Return whether `path` exists as a real, on-disk directory safe to
    rename/remove as the account directory.

    Returns `False` when nothing exists at `path` — a missing account
    directory is not an error, since the registry is the source of truth.
    Raises `AccountRegistryError` (never silently skips) when something
    exists at `path` that is unsafe to treat as the account directory: a
    symlink, a non-directory, or a directory whose parent is not exactly
    `expected_parent`.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise AccountRegistryError(f"refusing to follow symlink at {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise AccountRegistryError(f"refusing to operate on non-directory at {path}")
    if path.parent != expected_parent:
        raise AccountRegistryError(f"refusing to operate on {path} outside {expected_parent}")
    return True


def _rollback_new_account_directory(
    directory: Path, accounts_root: Path, cause: BaseException
) -> NoReturn:
    """Remove an owned staging/canonical directory after an `add_account`
    failure strictly before the registry commit point, and raise. A failure
    during the rollback itself names only the owned path — never credential
    data — in the resulting `AccountRegistryError`."""
    try:
        _remove_directory_tree(directory)
        _fsync_directory(accounts_root)
    except OSError as cleanup_exc:
        raise AccountRegistryError(
            f"failed to roll back {directory} after account creation failed"
        ) from cleanup_exc
    raise AccountRegistryError("failed to add the account") from cause


# --------------------------------------------------------------------------
# Crash recovery and incomplete-add sweep
# --------------------------------------------------------------------------


def _recover(accounts_root: Path) -> list[AccountRecord]:
    """Run at the start of every mutating operation, under the registry lock.

    Ordering rule: stale `.staging-*` directories (provably gateway-owned
    incomplete `add_account` attempts) are always safe to sweep first. Then,
    if `registry.json` itself is missing while any *other* persistent
    account artifact exists (a canonical UUID directory, a `*.tombstone`, or
    a `.orphan-*`), that is an unrecoverable inconsistency: raise without
    touching anything else. Only once an existing registry has been
    successfully parsed and validated does tombstone reconciliation run.

    Every caller already holds `registry.lock`, preserving the complete
    read/check/write lock scope required by mutations.
    """
    _sweep_staging_directories(accounts_root)

    registry_path = _registry_path(accounts_root)
    if not registry_path.exists():
        leftover = _list_non_staging_entries(accounts_root)
        if leftover:
            raise AccountRegistryError(
                f"registry {registry_path} is missing but account artifacts exist under "
                f"{accounts_root}; refusing to guess and deleting nothing"
            )
        return []

    parsed = _read_registry_array(registry_path)
    records = _parse_current_rows(parsed, path=registry_path)
    _reconcile_tombstones(accounts_root, records)
    _report_orphaned_directories(accounts_root, records)
    return records


def _sweep_staging_directories(accounts_root: Path) -> None:
    """Remove `.staging-*` directories left behind by a crashed `add_account`.

    Provably gateway-owned and always safe to delete: nothing else ever
    creates that name shape, and no registry row can reference one
    (canonical ids are UUIDs, never `.staging-...`). Each candidate is
    `lstat`-verified to be a real directory, never a symlink, before removal.
    """
    if not accounts_root.exists():
        return
    swept = False
    # Snapshot the listing before removing entries from it: mutating a
    # directory while a live `iterdir()` scan of it is in progress is not
    # something POSIX guarantees the outcome of.
    for entry in list(accounts_root.iterdir()):
        if not entry.name.startswith(_STAGING_PREFIX):
            continue
        if not _is_real_directory(entry):
            continue
        _remove_directory_tree(entry)
        swept = True
    if swept:
        _fsync_directory(accounts_root)


def _list_non_staging_entries(accounts_root: Path) -> list[Path]:
    if not accounts_root.exists():
        return []
    ignored_names = {_REGISTRY_FILENAME, _LOCK_FILENAME}
    return [
        entry
        for entry in accounts_root.iterdir()
        if entry.name not in ignored_names and not entry.name.startswith(_STAGING_PREFIX)
    ]


def _reconcile_tombstones(accounts_root: Path, records: list[AccountRecord]) -> None:
    registered_ids = {record.id for record in records}
    seen_ids: set[str] = set()
    # Snapshot the listing before renaming/removing entries from it, for the
    # same reason as `_sweep_staging_directories` above.
    for entry in list(accounts_root.iterdir()):
        if not entry.name.endswith(_TOMBSTONE_SUFFIX) or not _is_real_directory(entry):
            continue
        account_id = entry.name[: -len(_TOMBSTONE_SUFFIX)]
        if account_id in seen_ids:
            raise AccountRegistryError(f"multiple tombstones found for account {account_id}")
        seen_ids.add(account_id)

        final_dir = accounts_root / account_id
        if _is_real_directory(final_dir):
            raise AccountRegistryError(
                f"both the account directory and a tombstone exist for {account_id}"
            )

        if account_id in registered_ids:
            # The registry still names this account: the crash happened
            # before the registry commit, so undo the tombstone rename.
            os.rename(entry, final_dir)
        else:
            # The registry no longer names this account: the crash happened
            # after the registry commit, so retry the tombstone purge.
            _remove_directory_tree(entry)
        _fsync_directory(accounts_root)


def _report_orphaned_directories(accounts_root: Path, records: list[AccountRecord]) -> None:
    """Log (never delete) canonical-UUID directories the registry no longer
    references. Such a directory may be the only surviving copy of a real
    account — e.g. a crash between installing it and committing the
    registry row — so it is always left in place for a human to resolve."""
    registered_ids = {record.id for record in records}
    for entry in accounts_root.iterdir():
        if entry.name in (_REGISTRY_FILENAME, _LOCK_FILENAME):
            continue
        if entry.name.startswith(_STAGING_PREFIX) or entry.name.endswith(_TOMBSTONE_SUFFIX):
            continue
        if not _is_real_directory(entry) or not _is_canonical_uuid(entry.name):
            continue
        if entry.name not in registered_ids:
            logger.warning(
                "account directory %s under %s is not referenced by the registry; "
                "leaving it in place",
                entry.name,
                accounts_root,
            )
