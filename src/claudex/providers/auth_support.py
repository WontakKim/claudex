"""Shared credential-file and token-refresh support for providers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

_State = TypeVar("_State")


def write_private_json_atomic(path: Path, data: Any) -> None:
    """Durably replace a private JSON file through a same-directory staging file."""
    ensure_private_directory(path.parent)
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    staging_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")

    file_descriptor = os.open(
        staging_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        try:
            os.fchmod(file_descriptor, 0o600)
            handle = os.fdopen(file_descriptor, "wb")
        except BaseException:
            os.close(file_descriptor)
            raise
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _remove_path_if_exists(staging_path)
        raise

    try:
        os.replace(staging_path, path)
    except BaseException:
        _remove_path_if_exists(staging_path)
        raise
    _fsync_directory(path.parent)


def ensure_private_directory(directory: Path) -> None:
    """Create `directory` and any missing parents with mode 0700.

    `Path.mkdir(parents=True)` applies the requested mode only to the leaf;
    parents it creates get the umask-influenced default instead. This walks
    the missing chain explicitly so no intermediate directory is ever wider
    than 0700.
    """
    missing_directories: list[Path] = []
    current = directory
    while not current.exists():
        missing_directories.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for missing_directory in reversed(missing_directories):
        missing_directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(missing_directory, 0o700)
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    os.chmod(directory, 0o700)


def _remove_path_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def load_json_credentials(
    path: Path,
    *,
    error_cls: type[Exception],
    missing_message: str,
    encoding: str | None,
) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding=encoding)
    except FileNotFoundError as exc:
        raise error_cls(missing_message) from exc
    try:
        credentials = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise error_cls(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(credentials, dict):
        raise error_cls(f"{path} has an unexpected format")
    return credentials


def replace_credentials_file(
    path: Path, temp_file: Path, text: str, encoding: str | None
) -> None:
    temp_file.write_text(text, encoding=encoding)
    os.chmod(temp_file, 0o600)
    os.replace(temp_file, path)


async def refresh_when_still_stale(
    lock: asyncio.Lock,
    *,
    force_refresh: bool,
    stale_token: str,
    reload: Callable[[], _State],
    token_of: Callable[[_State], str],
    is_stale: Callable[[_State], bool],
    refresh: Callable[[_State], Awaitable[_State]],
) -> _State:
    """Remember which token looked stale: concurrent 401 retries all force-refresh,
    and only the first one holding the lock should actually rotate — with rotating
    refresh tokens a second POST for the same generation can invalidate the fresh
    credentials. The re-read also picks up a rotation the CLI itself just wrote.
    """
    async with lock:
        state = reload()
        if (force_refresh and token_of(state) == stale_token) or is_stale(state):
            state = await refresh(state)
        return state
