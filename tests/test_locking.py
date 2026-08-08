"""Tests for the shared cross-process file lock primitive."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from claudex_gateway.locking import file_lock, try_file_lock

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# Acquires `file_lock` on the path given as argv[1], announces success on
# stdout so the parent process knows the lock is held, sleeps for
# argv[2] seconds while still holding it, then releases and exits.
_LOCK_HOLDER_SCRIPT = """
import sys
import time
from pathlib import Path

from claudex_gateway.locking import file_lock

path = Path(sys.argv[1])
hold_seconds = float(sys.argv[2])
with file_lock(path):
    print("locked", flush=True)
    time.sleep(hold_seconds)
print("released", flush=True)
"""


def _spawn_lock_holder(lock_path: Path, hold_seconds: float) -> subprocess.Popen[str]:
    """Start a subprocess that holds `lock_path`'s lock for `hold_seconds`.

    Blocks until the child confirms it has acquired the lock, so callers
    never race the child's own `file_lock` acquisition.
    """
    process = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER_SCRIPT, str(lock_path), str(hold_seconds)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    announcement = process.stdout.readline().strip()
    assert announcement == "locked", f"child did not report holding the lock: {announcement!r}"
    return process


def _can_lock_without_blocking(path: Path) -> bool:
    """Probe whether `path`'s OS-level lock is free, without blocking."""
    fd = os.open(path, os.O_RDWR)
    try:
        if sys.platform == "win32":
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_second_process_is_excluded_until_the_first_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "registry.lock"
    holder = _spawn_lock_holder(lock_path, hold_seconds=1.0)
    try:
        assert not _can_lock_without_blocking(lock_path)
    finally:
        assert holder.wait(timeout=10) == 0

    assert _can_lock_without_blocking(lock_path)


def test_blocking_acquire_waits_for_the_other_process_to_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "registry.lock"
    hold_seconds = 1.0
    holder = _spawn_lock_holder(lock_path, hold_seconds=hold_seconds)
    try:
        started_at = time.monotonic()
        with file_lock(lock_path):
            elapsed = time.monotonic() - started_at
        assert elapsed >= hold_seconds * 0.75
    finally:
        assert holder.wait(timeout=10) == 0


def test_lock_is_released_when_the_caller_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "registry.lock"

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with file_lock(lock_path):
            raise _Boom("boom")

    # A fresh acquire must succeed immediately; it would hang if the lock
    # from the failed `with` block above were still held.
    with file_lock(lock_path):
        pass


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode bits are not meaningful on Windows"
)
def test_lock_file_mode_is_owner_only_on_posix(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "registry.lock"
    with file_lock(lock_path):
        pass
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_parent_directory_is_created(tmp_path: Path) -> None:
    lock_dir = tmp_path / "nested" / "state"
    lock_path = lock_dir / "registry.lock"
    assert not lock_dir.exists()

    with file_lock(lock_path):
        pass

    assert lock_dir.is_dir()
    assert lock_path.is_file()


def test_try_file_lock_returns_none_while_another_process_holds(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "capture.lock"
    holder = _spawn_lock_holder(lock_path, hold_seconds=1.0)
    try:
        assert try_file_lock(lock_path) is None
    finally:
        assert holder.wait(timeout=10) == 0

    handle = try_file_lock(lock_path)
    assert handle is not None
    handle.release()


def test_try_file_lock_handle_excludes_others_until_released(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "capture.lock"
    handle = try_file_lock(lock_path)
    assert handle is not None
    try:
        assert not _can_lock_without_blocking(lock_path)
    finally:
        handle.release()
    assert _can_lock_without_blocking(lock_path)


def test_try_file_lock_release_is_idempotent(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "capture.lock"
    handle = try_file_lock(lock_path)
    assert handle is not None
    handle.release()
    handle.release()
    assert _can_lock_without_blocking(lock_path)


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX file mode bits are not meaningful on Windows"
)
def test_try_file_lock_creates_owner_only_file_and_parent(tmp_path: Path) -> None:
    lock_dir = tmp_path / "nested" / "state"
    lock_path = lock_dir / "capture.lock"
    assert not lock_dir.exists()

    handle = try_file_lock(lock_path)
    assert handle is not None
    handle.release()

    assert lock_dir.is_dir()
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
