"""Persistent registry facade for Claude accounts.

Storage layout, all rooted at `paths.accounts_dir("claude")`::

    accounts/claude/
    ├── registry.json      # bare JSON array of rows, no secrets, mode 0600
    ├── registry.lock      # cross-process mutation lock (claudex.locking)
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

Registry mutations (`add_account`, `remove_account`) are serialized across
processes by `locking.file_lock` on
`registry.lock`, held for the complete read/check/write of each mutation.
`add_account` stages the new account directory under a `.staging-<uuid4>`
name and installs it atomically before the registry is ever allowed to
reference it; `remove_account` renames the directory to a `<id>.tombstone`
name before removing the registry row, so a crash between the two steps is
always recoverable. In both cases the registry's own `os.replace` is the
sole transaction commit point (see `add_account`/`remove_account`
docstrings). Crash recovery reconciles any leftover `.staging-*` or
`*.tombstone` directories at the start of every mutation, under the lock.

This module retains registry orchestration and the public API while delegating
canonical model validation and filesystem persistence to the lower layers.
No CLI, OAuth capture flow, or prompting lives here, and no credential values
ever appear in an exception message or a log line.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any

from claudex import paths
from claudex.claude import account_model as claude_account_model
from claudex.claude import account_store as claude_account_store
from claudex.locking import file_lock

AccountRecord = claude_account_model.AccountRecord
AccountRegistryError = claude_account_model.AccountRegistryError
DuplicateAccountError = claude_account_model.DuplicateAccountError
AccountNotFoundError = claude_account_model.AccountNotFoundError

# Retained for the login-session normalization boundary; canonical ownership is
# in the dependency-free model module.
_normalize_email = claude_account_model._normalize_email
_normalize_optional_text = claude_account_model._normalize_optional_text

_PROVIDER = "claude"


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
    registry_path = claude_account_store._registry_path(accounts_root)
    parsed = claude_account_store._read_registry_array(registry_path)
    return claude_account_model._parse_current_rows(parsed, path=registry_path)


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
    ) = claude_account_model._validate_account_inputs(
        email, organization_uuid, organization_name, credentials_json, oauth_account_json
    )

    accounts_root = paths.accounts_dir(_PROVIDER)
    claude_account_store._makedirs_0700(paths.runtime_dir())
    claude_account_store._makedirs_0700(accounts_root)

    with file_lock(claude_account_store._lock_path(accounts_root)):
        records = claude_account_store._recover(accounts_root)
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
            state=claude_account_model._STATE_READY,
            account_incarnation_id=str(uuid.uuid4()),
            upstream_account_uuid=claude_account_model._derive_upstream_account_uuid(
                oauth_payload
            ),
        )

        staging_dir = (
            accounts_root / f"{claude_account_store._STAGING_PREFIX}{uuid.uuid4()}"
        )
        try:
            claude_account_store._create_new_directory(staging_dir, mode=0o700)
            claude_account_store._write_json_atomic(
                staging_dir / claude_account_store._CREDENTIALS_FILENAME,
                credentials_json,
            )
            claude_account_store._write_json_atomic(
                staging_dir / claude_account_store._OAUTH_ACCOUNT_FILENAME,
                oauth_payload,
            )
        except BaseException as exc:
            claude_account_store._rollback_new_account_directory(
                staging_dir, accounts_root, exc
            )

        final_dir = accounts_root / account_id
        try:
            claude_account_store._rename_directory(staging_dir, final_dir)
        except OSError as exc:
            claude_account_store._rollback_new_account_directory(
                staging_dir, accounts_root, exc
            )

        try:
            claude_account_store._fsync_directory(accounts_root)
        except OSError as exc:
            # Durability barrier, still strictly before the registry commit
            # point: a full rollback of the canonical directory is safe.
            claude_account_store._rollback_new_account_directory(
                final_dir, accounts_root, exc
            )

        registry_path = claude_account_store._registry_path(accounts_root)
        try:
            claude_account_store._write_json_atomic(
                registry_path, [record.to_row() for record in [*records, record]]
            )
        except claude_account_store._PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {account_id} may already be registered in {registry_path}; "
                "registry durability after commit is uncertain"
            ) from exc
        except BaseException as exc:
            claude_account_store._rollback_new_account_directory(
                final_dir, accounts_root, exc
            )

        return record


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
    ) = claude_account_model._validate_account_inputs(
        email, organization_uuid, organization_name, credentials_json, oauth_account_json
    )

    accounts_root = paths.accounts_dir(_PROVIDER)
    claude_account_store._makedirs_0700(paths.runtime_dir())
    claude_account_store._makedirs_0700(accounts_root)

    with file_lock(claude_account_store._lock_path(accounts_root)):
        records = claude_account_store._recover(accounts_root)
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
        newly_captured_upstream_account_uuid = (
            claude_account_model._derive_upstream_account_uuid(oauth_payload)
        )
        account_incarnation_id, upstream_account_uuid = (
            claude_account_model._resolve_reauth_incarnation(
                existing, newly_captured_upstream_account_uuid
            )
        )
        updated = AccountRecord(
            id=existing.id,
            email=normalized_email,
            organization_uuid=normalized_organization_uuid,
            organization_name=normalized_organization_name,
            created_at=existing.created_at,
            updated_at=now,
            last_authenticated_at=now,
            state=claude_account_model._STATE_READY,
            account_incarnation_id=account_incarnation_id,
            upstream_account_uuid=upstream_account_uuid,
        )

        account_dir = accounts_root / existing.id
        try:
            claude_account_store._write_json_atomic(
                account_dir / claude_account_store._CREDENTIALS_FILENAME,
                credentials_json,
            )
            claude_account_store._write_json_atomic(
                account_dir / claude_account_store._OAUTH_ACCOUNT_FILENAME,
                oauth_payload,
            )
        except claude_account_store._PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"credentials for account {existing.id} were replaced with uncertain "
                "durability; re-run `account add` to reconverge"
            ) from exc
        except OSError as exc:
            raise AccountRegistryError(
                f"failed to replace credentials for account {existing.id}"
            ) from exc

        registry_path = claude_account_store._registry_path(accounts_root)
        rows = [
            (updated if record.id == existing.id else record).to_row() for record in records
        ]
        try:
            claude_account_store._write_json_atomic(registry_path, rows)
        except claude_account_store._PostCommitFsyncError as exc:
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
    canonical_id = claude_account_model._canonicalize_account_id(account_id)

    accounts_root = paths.accounts_dir(_PROVIDER)
    claude_account_store._makedirs_0700(paths.runtime_dir())
    claude_account_store._makedirs_0700(accounts_root)

    with file_lock(claude_account_store._lock_path(accounts_root)):
        records = claude_account_store._recover(accounts_root)
        existing = next((record for record in records if record.id == canonical_id), None)
        if existing is None:
            raise AccountNotFoundError(f"no account registered with id {canonical_id}")
        if existing.state == claude_account_model._STATE_NEEDS_REAUTH:
            return existing

        updated = replace(
            existing,
            updated_at=_now_millis(),
            state=claude_account_model._STATE_NEEDS_REAUTH,
        )
        registry_path = claude_account_store._registry_path(accounts_root)
        rows = [
            (updated if record.id == canonical_id else record).to_row()
            for record in records
        ]
        try:
            claude_account_store._write_json_atomic(registry_path, rows)
        except claude_account_store._PostCommitFsyncError as exc:
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
    canonical_id = claude_account_model._canonicalize_account_id(account_id)
    accounts_root = paths.accounts_dir(_PROVIDER)
    claude_account_store._makedirs_0700(paths.runtime_dir())
    claude_account_store._makedirs_0700(accounts_root)

    with file_lock(claude_account_store._lock_path(accounts_root)):
        records = claude_account_store._recover(accounts_root)
        remaining = [record for record in records if record.id != canonical_id]
        if len(remaining) == len(records):
            raise AccountNotFoundError(f"no account registered with id {canonical_id}")

        final_dir = accounts_root / canonical_id
        tombstone_dir = (
            accounts_root / f"{canonical_id}{claude_account_store._TOMBSTONE_SUFFIX}"
        )
        directory_present = claude_account_store._verify_removable_directory(
            final_dir, accounts_root
        )

        if directory_present:
            tombstoned = False
            try:
                claude_account_store._rename_directory(final_dir, tombstone_dir)
                tombstoned = True
                claude_account_store._fsync_directory(accounts_root)
            except OSError as exc:
                if tombstoned:
                    # The pre-commit durability barrier failed while the
                    # registry still references the account: restore the
                    # canonical directory so registry and disk stay in step.
                    try:
                        claude_account_store._rename_directory(tombstone_dir, final_dir)
                        claude_account_store._fsync_directory(accounts_root)
                    except OSError as restore_exc:
                        raise AccountRegistryError(
                            f"failed to restore {final_dir} after a tombstone "
                            "durability failure"
                        ) from restore_exc
                raise AccountRegistryError(
                    f"failed to tombstone account directory {final_dir}"
                ) from exc

        registry_path = claude_account_store._registry_path(accounts_root)
        try:
            claude_account_store._write_json_atomic(
                registry_path, [record.to_row() for record in remaining]
            )
        except claude_account_store._PostCommitFsyncError as exc:
            raise AccountRegistryError(
                f"account {canonical_id} was removed from {registry_path} but durability "
                f"after commit is uncertain; a tombstone may remain at {tombstone_dir}"
            ) from exc
        except BaseException as exc:
            if directory_present:
                try:
                    claude_account_store._rename_directory(tombstone_dir, final_dir)
                    claude_account_store._fsync_directory(accounts_root)
                except OSError as restore_exc:
                    raise AccountRegistryError(
                        f"failed to restore {final_dir} after a registry write failure"
                    ) from restore_exc
            raise AccountRegistryError(f"failed to remove account {canonical_id}") from exc

        if directory_present:
            try:
                claude_account_store._remove_directory_tree(tombstone_dir)
                claude_account_store._fsync_directory(accounts_root)
            except OSError as exc:
                raise AccountRegistryError(
                    f"account {canonical_id} was removed but the tombstone at "
                    f"{tombstone_dir} could not be purged"
                ) from exc


def _now_millis() -> int:
    return time.time_ns() // 1_000_000
