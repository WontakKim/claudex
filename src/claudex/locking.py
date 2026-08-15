"""Cross-process, blocking, stdlib-only file lock shared by every module
that persists state under ``~/.claudex``.

`file_lock` wraps `fcntl.flock` on POSIX and `msvcrt.locking` on Windows
behind one context manager, so callers on both platforms get identical
blocking-acquire semantics: the lock is held for the duration of the
caller's `with` block and is always released, even when the caller raises.
"""

from __future__ import annotations

import errno
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive lock on `path`, blocking until it is available.

    Creates `path`'s parent directory (mode 0700, owner-only) and `path`
    itself if either is missing, and chmods `path` to 0600 so the lock file
    is never group- or world-readable. The lock is released, and the file
    descriptor closed, on every exit from the `with` block, including when
    the caller raises.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(path, 0o600)
        _acquire(fd)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


class FileLockHandle:
    """An exclusively held file lock that outlives a `with` block.

    `file_lock` scopes the hold to a context manager; a handle instead keeps
    the lock across awaits and method calls (the daemon holds the capture
    lock for a whole login session) until `release()` is called. The lock
    lives on the open file description, so no thread affinity applies, and
    the OS drops it if the process dies with the fd open.
    """

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        """Release the lock and close the fd. Safe to call more than once."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _release(fd)
        finally:
            os.close(fd)


def try_file_lock(path: Path) -> FileLockHandle | None:
    """Attempt a non-blocking exclusive lock on `path`.

    Returns a `FileLockHandle` on success, or `None` when another process
    (or another fd in this process) already holds the lock. Directory and
    file creation follow `file_lock` exactly (0700 parent, 0600 lock file).
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.chmod(path, 0o600)
        if not _try_acquire(fd):
            os.close(fd)
            return None
    except BaseException:
        os.close(fd)
        raise
    return FileLockHandle(fd)


def _try_acquire(fd: int) -> bool:
    """One non-blocking exclusive-lock attempt; False when contended."""
    if sys.platform == "win32":
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
    else:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise


def _acquire(fd: int) -> None:
    """Block until `fd` is exclusively locked, retrying transient failures."""
    if sys.platform == "win32":
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                return
            except OSError as exc:
                # A single LK_LOCK call gives up after ~10s of contention and
                # raises instead of blocking indefinitely like POSIX flock;
                # retry that case (EACCES/EDEADLK) and EINTR until acquired,
                # so the caller still sees a blocking acquire.
                if exc.errno not in (errno.EINTR, errno.EACCES, errno.EDEADLK):
                    raise
    else:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                return
            except OSError as exc:
                if exc.errno != errno.EINTR:
                    raise


def _release(fd: int) -> None:
    """Release the lock held on `fd`."""
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
