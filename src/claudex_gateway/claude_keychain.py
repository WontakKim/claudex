"""macOS Keychain support for Claude credential capture."""

from __future__ import annotations

import hashlib
import os
import subprocess
import unicodedata

from claudex_gateway.claude_capture_model import CaptureError, KeychainBackend

_SECURITY_BIN = "/usr/bin/security"
# `security`'s documented exit status for "the specified item could not be
# found in the keychain" (SecKeychainSearchCopyNext errSecItemNotFound).
_KEYCHAIN_ITEM_NOT_FOUND_STATUS = 44
_KEYCHAIN_TIMEOUT_SECONDS = 5.0

# Capture only ever addresses the scoped service form below as a credential
# source. The legacy unscoped service name exists solely for read-only change
# detection (`read_legacy_login_fingerprint`): it is never captured from,
# never written, and never deleted.
_KEYCHAIN_SERVICE_FORMAT = "Claude Code-credentials-{suffix}"
LEGACY_KEYCHAIN_SERVICE = "Claude Code-credentials"


def _run_security(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `/usr/bin/security` with a bounded timeout, always reaping it."""
    try:
        return subprocess.run(
            [_SECURITY_BIN, *args],
            capture_output=True,
            text=True,
            timeout=_KEYCHAIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(
            f"`{_SECURITY_BIN} {args[0]}` timed out after {_KEYCHAIN_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise CaptureError(f"failed to invoke {_SECURITY_BIN}: {exc}") from exc


class SecurityKeychainBackend:
    """Production `KeychainBackend`, always invoking `/usr/bin/security` by
    absolute path rather than resolving `security` through PATH."""

    def read(self, service: str, account: str) -> str | None:
        result = _run_security(["find-generic-password", "-s", service, "-a", account, "-w"])
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode == _KEYCHAIN_ITEM_NOT_FOUND_STATUS:
            return None
        raise CaptureError(
            f"Keychain read for service {service!r} failed (security exited "
            f"{result.returncode})"
        )

    def delete(self, service: str, account: str) -> None:
        result = _run_security(["delete-generic-password", "-s", service, "-a", account])
        if result.returncode in (0, _KEYCHAIN_ITEM_NOT_FOUND_STATUS):
            return
        raise CaptureError(
            f"Keychain delete for service {service!r} failed (security exited "
            f"{result.returncode})"
        )


def default_keychain_backend() -> KeychainBackend:
    return SecurityKeychainBackend()


def scoped_keychain_service(raw_config_dir: str) -> str:
    """Derive the scoped Keychain service name for `raw_config_dir`.

    Hashes the RAW caller-supplied string: no `~` expansion, absolutizing,
    separator cleanup, or `Path` conversion happens before hashing, so a
    trailing slash or `.` component changes the result on purpose. Unicode
    variants of the same path (NFC vs NFD) must hash identically, so the
    string is NFC-normalized before hashing.
    """
    normalized = unicodedata.normalize("NFC", raw_config_dir)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return _KEYCHAIN_SERVICE_FORMAT.format(suffix=digest[:8])


def keychain_account() -> str:
    """The Keychain account attribute: USER, else USERNAME, else fail closed.

    Never falls back to a guessed literal — a wrong account would silently
    miss Claude's item and could strand the temp credential.
    """
    account = os.environ.get("USER") or os.environ.get("USERNAME")
    if not account:
        raise CaptureError(
            "cannot determine the Keychain account attribute: neither USER nor "
            "USERNAME is set in the environment"
        )
    return account
