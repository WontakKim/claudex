"""Single source of truth for every path rooted at `~/.claudex`.

Every helper here returns a `Path` without touching the filesystem, so any
module can depend on it at import time without risking import-time side
effects or import cycles. This module must import only the standard library.
"""

from __future__ import annotations

from pathlib import Path


def runtime_dir() -> Path:
    return Path.home() / ".claudex"


def settings_file() -> Path:
    return runtime_dir() / "settings.json"


def daemon_record_file() -> Path:
    # The historical gateway.pid path now holds a JSON identity record
    # (pid/host/port/nonce); the name is kept so pre-0.4 installs are
    # detected as legacy records instead of being silently ignored.
    return runtime_dir() / "gateway.pid"


def log_file() -> Path:
    return runtime_dir() / "gateway.log"


def accounts_dir(provider: str) -> Path:
    """Return the accounts directory for `provider`.

    `provider` must be exactly one path component: nonempty, containing no
    `/` or `\\` separator, and not `.` or `..`. This keeps a provider string
    from ever escaping the accounts root.
    """
    if not provider or "/" in provider or "\\" in provider or provider in (".", ".."):
        raise ValueError(f"invalid provider name: {provider!r}")
    return runtime_dir() / "accounts" / provider
