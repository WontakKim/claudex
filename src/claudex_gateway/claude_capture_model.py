"""Shared data types for Claude credential capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CapturedAccount:
    credentials_json: dict[str, Any]
    oauth_account_json: dict[str, Any] | None
    email: str
    organization_uuid: str | None
    organization_name: str | None


class CaptureError(Exception):
    """Raised when Claude credential capture fails."""


class CaptureCancelled(CaptureError):
    """Raised when the user cancels an interactive login (SIGINT/SIGTERM/SIGHUP).

    Distinct from a plain `CaptureError` so a caller (the CLI) can map
    cancellation to exit 130 instead of a generic failure exit code.
    """


class KeychainBackend(Protocol):
    """Reads and deletes macOS Keychain generic-password items.

    `read` is tri-state: a string password means the item was found; `None`
    means the Keychain conclusively reported that no such item exists (the
    documented item-not-found exit); any other outcome is an operational
    failure and must raise `CaptureError`. `delete` treats
    conclusively-missing as success and raises `CaptureError` for every
    other failure. Both operations must apply a bounded timeout and always
    reap the subprocess they spawn.
    """

    def read(self, service: str, account: str) -> str | None: ...

    def delete(self, service: str, account: str) -> None: ...
