"""Filesystem persistence for the Claude account registry."""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, NoReturn

from .account_model import (
    AccountRecord,
    AccountRegistryError,
    _is_canonical_uuid,
    _parse_current_rows,
)

logger = logging.getLogger(__name__)

_REGISTRY_FILENAME = "registry.json"
_LOCK_FILENAME = "registry.lock"
_CREDENTIALS_FILENAME = "credentials.json"
_OAUTH_ACCOUNT_FILENAME = "oauth-account.json"
_TOMBSTONE_SUFFIX = ".tombstone"
_STAGING_PREFIX = ".staging-"


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


def _rename_directory(source: Path, destination: Path) -> None:
    os.rename(source, destination)


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
            _rename_directory(entry, final_dir)
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
