"""Capture a Claude Code login into a `CapturedAccount`, scoped-Keychain-only.

Two entry points feed the multi-account pool:

- `capture_from_config_dir` reads an *existing* login from a caller-supplied
  `CLAUDE_CONFIG_DIR`-style directory, strictly read-only. This backs the
  headless `--from <dir>` path.
- `capture_interactive` spawns `claude auth login --claudeai` in a
  gateway-created temporary config directory and captures the result. This
  backs the interactive login path and requires a TTY.

Neither entry point writes the account registry; both return a
`CapturedAccount` for the caller to persist.

Only Claude Code builds that store credentials in a *scoped* macOS Keychain
item (`Claude Code-credentials-<hex[:8]>`, see `_scoped_keychain_service`)
are supported. There is no legacy unscoped Keychain item, no read/write
fallback, and no other recovery path: when the expected scoped item is
absent, capture fails closed with a `CaptureError` naming the expected
service. A version gate pins the exact `claude` builds this module has been
verified against and runs before any login spawn, so an unverified binary
never reaches a login that could touch the user's Keychain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from claudex_gateway import locking, paths

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SECURITY_BIN = "/usr/bin/security"
# `security`'s documented exit status for "the specified item could not be
# found in the keychain" (SecKeychainSearchCopyNext errSecItemNotFound).
_KEYCHAIN_ITEM_NOT_FOUND_STATUS = 44
_KEYCHAIN_TIMEOUT_SECONDS = 5.0

# The base Keychain service name is deliberately never spelled as a standalone
# quoted literal: this module only ever addresses the scoped form below, and
# there is no unscoped/legacy service to fall back to.
_KEYCHAIN_SERVICE_FORMAT = "Claude Code-credentials-{suffix}"

# The tested-version allowlist is exact, not a `>=` range: an unlisted build
# may have moved the Keychain item's scoping or format in a way this module
# does not yet account for, so it is refused unless the caller opts in.
_SUPPORTED_CLAUDE_VERSIONS = frozenset({"2.1.222"})
_ALLOW_UNTESTED_CLAUDE_ENV = "CLAUDEX_ALLOW_UNTESTED_CLAUDE"
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_VERSION_CHECK_TIMEOUT_SECONDS = 10.0

_LOGIN_LOCK_FILENAME = "claude-capture.lock"
_TEMP_DIR_PREFIX = "claudex-claude-login-"
_MAX_TEMP_DIR_ATTEMPTS = 3
_PROCESS_GROUP_GRACE_SECONDS = 5.0

# Read source files with a byte cap so a hostile or corrupt config directory
# cannot make capture read an unbounded amount of data into memory.
_SOURCE_FILE_BYTE_CAP = 1024 * 1024

# Never leaves the child able to authenticate any other way than the fresh
# interactive login, and never lets a provider selector redirect the CLI away
# from api.anthropic.com's OAuth login flow.
_ENV_VARS_TO_REMOVE = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "CLAUDE_SECURESTORAGE_CONFIG_DIR",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
)


# ---------------------------------------------------------------------------
# Production Keychain backend
# ---------------------------------------------------------------------------


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


class _SecurityKeychainBackend:
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


def _default_keychain_backend() -> KeychainBackend:
    return _SecurityKeychainBackend()


# ---------------------------------------------------------------------------
# Scoped selector algorithm
# ---------------------------------------------------------------------------


def _scoped_keychain_service(raw_config_dir: str) -> str:
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


def _keychain_account() -> str:
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


# ---------------------------------------------------------------------------
# Source file reading (shared by both entry points)
# ---------------------------------------------------------------------------


def _read_capped_text(path: Path) -> str | None:
    """Read `path` as UTF-8 text, capped at `_SOURCE_FILE_BYTE_CAP` bytes.

    Returns `None` when the file does not exist; raises `CaptureError` for
    every other read failure, an oversized file, or invalid UTF-8.
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(_SOURCE_FILE_BYTE_CAP + 1)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CaptureError(f"could not read {path}: {exc}") from exc
    if len(data) > _SOURCE_FILE_BYTE_CAP:
        raise CaptureError(f"{path} exceeds the {_SOURCE_FILE_BYTE_CAP}-byte read cap")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureError(f"{path} is not valid UTF-8: {exc}") from exc


def _parse_json_object(raw: str, *, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CaptureError(f"{source} has an unexpected format (expected a JSON object)")
    return parsed


def _read_credentials_blob(config_dir: str, keychain: KeychainBackend) -> dict[str, Any]:
    """Read the credentials blob for `config_dir`: scoped Keychain item on
    macOS, `<config_dir>/.credentials.json` everywhere else."""
    if sys.platform == "darwin":
        service = _scoped_keychain_service(config_dir)
        account = _keychain_account()
        password = keychain.read(service, account)
        if password is None:
            raise CaptureError(
                f"no Claude Code credentials found under the expected scoped "
                f"Keychain service {service!r}; sign in with "
                "`claude auth login --claudeai` first"
            )
        if len(password.encode("utf-8")) > _SOURCE_FILE_BYTE_CAP:
            raise CaptureError(
                f"Keychain service {service!r} exceeds the "
                f"{_SOURCE_FILE_BYTE_CAP}-byte read cap"
            )
        return _parse_json_object(password, source=f"Keychain service {service!r}")
    credentials_path = Path(config_dir) / ".credentials.json"
    raw = _read_capped_text(credentials_path)
    if raw is None:
        raise CaptureError(f"{credentials_path} not found; sign in with `claude` first")
    return _parse_json_object(raw, source=str(credentials_path))


def _read_oauth_account_block(config_dir: str) -> dict[str, Any] | None:
    """Read `<config_dir>/.claude.json`'s `oauthAccount` block, if present.

    A missing `.claude.json` is fine (this source is optional); malformed
    JSON is not and raises `CaptureError`.
    """
    claude_json_path = Path(config_dir) / ".claude.json"
    raw = _read_capped_text(claude_json_path)
    if raw is None:
        return None
    parsed = _parse_json_object(raw, source=str(claude_json_path))
    oauth_account = parsed.get("oauthAccount")
    if oauth_account is None:
        return None
    if not isinstance(oauth_account, dict):
        raise CaptureError(f"{claude_json_path}'s oauthAccount is not a JSON object")
    return oauth_account


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def _normalize_identity_value(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().casefold()


def _clean_identity_value(value: Any) -> str | None:
    """NFC-normalize and trim a candidate identity value; anything that is
    not a string, or that is empty after trimming, counts as absent."""
    if not isinstance(value, str):
        return None
    cleaned = unicodedata.normalize("NFC", value).strip()
    return cleaned or None


def _resolve_identity_field(
    field: str,
    oauth_account: dict[str, Any] | None,
    claude_ai_oauth: dict[str, Any] | None,
    oauth_account_key: str,
    claude_ai_oauth_key: str,
    *,
    reject_conflict: bool = True,
) -> str | None:
    """Resolve one identity field: `oauthAccount` wins, `claudeAiOauth` is the
    fallback. Presence is decided on the normalized, trimmed value, so a
    whitespace-only entry counts as absent rather than shadowing (or falsely
    conflicting with) the other source. With `reject_conflict`, differing
    values from both sources fail closed; without it (organizationName, per
    spec) plain precedence applies."""
    from_oauth_account = _clean_identity_value(
        oauth_account.get(oauth_account_key) if oauth_account else None
    )
    from_claude_ai_oauth = _clean_identity_value(
        claude_ai_oauth.get(claude_ai_oauth_key) if claude_ai_oauth else None
    )
    if (
        reject_conflict
        and from_oauth_account is not None
        and from_claude_ai_oauth is not None
        and _normalize_identity_value(from_oauth_account)
        != _normalize_identity_value(from_claude_ai_oauth)
    ):
        raise CaptureError(
            f"conflicting {field} between .claude.json's oauthAccount "
            f"({from_oauth_account!r}) and the credentials blob's claudeAiOauth "
            f"({from_claude_ai_oauth!r}); refusing to register one account's "
            "metadata with another's token"
        )
    return from_oauth_account or from_claude_ai_oauth


def _resolve_identity(
    credentials_json: dict[str, Any], oauth_account_json: dict[str, Any] | None
) -> tuple[str, str | None, str | None]:
    claude_ai_oauth = credentials_json.get("claudeAiOauth")
    claude_ai_oauth = claude_ai_oauth if isinstance(claude_ai_oauth, dict) else None

    email = _resolve_identity_field(
        "email", oauth_account_json, claude_ai_oauth, "emailAddress", "email"
    )
    if not email:
        raise CaptureError(
            "could not resolve a nonempty account email from either "
            ".claude.json's oauthAccount block or the credentials blob's "
            "claudeAiOauth"
        )
    organization_uuid = _resolve_identity_field(
        "organizationUuid",
        oauth_account_json,
        claude_ai_oauth,
        "organizationUuid",
        "organizationUuid",
    )
    organization_name = _resolve_identity_field(
        "organizationName",
        oauth_account_json,
        claude_ai_oauth,
        "organizationName",
        "organizationName",
        reject_conflict=False,
    )
    return email, organization_uuid, organization_name


def _capture_from_config_dir_impl(config_dir: str, keychain: KeychainBackend) -> CapturedAccount:
    credentials_json = _read_credentials_blob(config_dir, keychain)
    oauth_account_json = _read_oauth_account_block(config_dir)
    email, organization_uuid, organization_name = _resolve_identity(
        credentials_json, oauth_account_json
    )
    return CapturedAccount(
        credentials_json=credentials_json,
        oauth_account_json=oauth_account_json,
        email=email,
        organization_uuid=organization_uuid,
        organization_name=organization_name,
    )


def capture_from_config_dir(config_dir: str) -> CapturedAccount:
    """Capture a previously-completed Claude Code login from `config_dir`.

    Strictly read-only: never spawns `claude`, never writes or deletes
    anything. On macOS this reads only the scoped Keychain item for the
    exact `config_dir` string given (see `_scoped_keychain_service`); on
    every other platform it reads `<config_dir>/.credentials.json`.
    """
    return _capture_from_config_dir_impl(config_dir, _default_keychain_backend())


# ---------------------------------------------------------------------------
# Pre-spawn version gate
# ---------------------------------------------------------------------------


def _resolve_claude_executable() -> str:
    resolved = shutil.which("claude")
    if resolved is None:
        raise CaptureError("the `claude` CLI was not found on PATH; install Claude Code first")
    # A relative PATH entry yields a relative result; absolutize so the
    # version check and the login spawn are guaranteed to run the same file
    # regardless of any working-directory change between them.
    return os.path.abspath(resolved)


def _stat_identity(path: str) -> tuple[int, int, int, int]:
    info = os.stat(path)
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)


def _parse_claude_version(stdout: str, stderr: str) -> str | None:
    match = _VERSION_RE.search(stdout) or _VERSION_RE.search(stderr)
    return match.group(1) if match else None


def _check_claude_version(claude_path: str) -> None:
    """Run `claude --version` and enforce the tested-version allowlist.

    Runs before any login spawn: an unparseable version, or one outside the
    allowlist without the opt-in override, aborts before the same resolved
    `claude` path is ever used to spawn a login that could mutate the user's
    Keychain.
    """
    version_env = dict(os.environ)
    version_env["DISABLE_UPDATES"] = "1"
    try:
        result = subprocess.run(
            [claude_path, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_CHECK_TIMEOUT_SECONDS,
            env=version_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(
            f"`claude --version` timed out after {_VERSION_CHECK_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise CaptureError(f"failed to run `claude --version`: {exc}") from exc

    if result.returncode != 0:
        raise CaptureError(f"`claude --version` exited with status {result.returncode}")
    version = _parse_claude_version(result.stdout, result.stderr)
    if version is None:
        raise CaptureError("could not parse a version from `claude --version` output")
    opted_in = os.environ.get(_ALLOW_UNTESTED_CLAUDE_ENV) == "1"
    if version not in _SUPPORTED_CLAUDE_VERSIONS and not opted_in:
        allowed = ", ".join(sorted(_SUPPORTED_CLAUDE_VERSIONS))
        raise CaptureError(
            f"claude {version} is not in the tested version allowlist ({allowed}); "
            f"set {_ALLOW_UNTESTED_CLAUDE_ENV}=1 to proceed at your own risk"
        )


# ---------------------------------------------------------------------------
# Interactive login
# ---------------------------------------------------------------------------


class _LoginSignalReceived(Exception):
    """Internal marker raised from a SIGTERM/SIGHUP handler during login wait."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def _child_process_env(config_dir: str) -> dict[str, str]:
    """Build the login child's environment: copy, point at the temp config
    dir, disable auto-update, and remove every auth-selector variable that
    could steer `claude auth login` away from a fresh interactive login."""
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = config_dir
    env["DISABLE_UPDATES"] = "1"
    for var in _ENV_VARS_TO_REMOVE:
        env.pop(var, None)
    return env


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """SIGTERM the login process group, wait a bounded grace period, then
    SIGKILL, always reaping the leader afterward."""
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        process.wait()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=_PROCESS_GROUP_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_login(claude_path: str, config_dir: str, timeout_secs: int) -> None:
    """Spawn `claude auth login --claudeai` and wait for it to finish.

    stdin/stdout/stderr are inherited untouched — no output is piped or
    scanned. A nonzero exit is treated as a login denial/failure. The caller
    (`capture_interactive`) owns the SIGTERM/SIGHUP handlers, which raise
    `_LoginSignalReceived`; this function's job is to guarantee the login's
    process group is torn down whenever cancellation or timeout lands after
    the spawn. Timeout raises a plain `CaptureError`; cancellation raises
    `CaptureCancelled`.
    """
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            [claude_path, "auth", "login", "--claudeai"],
            env=_child_process_env(config_dir),
            start_new_session=True,
        )
        returncode = process.wait(timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise CaptureError(
            f"`claude auth login --claudeai` timed out after {timeout_secs}s"
        ) from None
    except (_LoginSignalReceived, KeyboardInterrupt) as exc:
        if process is not None:
            _terminate_process_group(process)
        raise CaptureCancelled("Claude login was cancelled") from exc

    if returncode != 0:
        raise CaptureError(f"`claude auth login --claudeai` exited with status {returncode}")


def _mint_temp_config_dir(keychain: KeychainBackend) -> str:
    """Create a fresh temp Claude config dir, retrying on macOS if a scoped
    Keychain item already exists for that exact path.

    Never deletes a colliding item — it belongs to a different login — the
    directory is discarded and a new one is minted instead, up to
    `_MAX_TEMP_DIR_ATTEMPTS` times.
    """
    for _attempt in range(_MAX_TEMP_DIR_ATTEMPTS):
        candidate = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
        if sys.platform != "darwin":
            return candidate
        try:
            service = _scoped_keychain_service(candidate)
            existing = keychain.read(service, _keychain_account())
        except BaseException:
            # The collision precheck failed: do not leak the freshly minted
            # empty directory alongside the propagating error.
            os.rmdir(candidate)
            raise
        if existing is None:
            return candidate
        os.rmdir(candidate)
    raise CaptureError(
        "could not mint a Claude config dir free of an existing scoped Keychain "
        f"item after {_MAX_TEMP_DIR_ATTEMPTS} attempts"
    )


def _cleanup_temp_config_dir(config_dir: str, keychain: KeychainBackend) -> None:
    """Delete the scoped Keychain item and remove the temp dir.

    Both actions are attempted even when one fails; any failure raises a
    sanitized `CaptureError` afterward instead of reporting success.
    """
    errors: list[str] = []
    if sys.platform == "darwin":
        try:
            keychain.delete(_scoped_keychain_service(config_dir), _keychain_account())
        except CaptureError as exc:
            errors.append(f"failed to delete the scoped Keychain item: {exc}")
    try:
        shutil.rmtree(config_dir)
    except OSError as exc:
        errors.append(f"failed to remove the temp Claude config dir: {exc}")
    if errors:
        raise CaptureError("; ".join(errors))


def capture_interactive(timeout_secs: int = 180) -> CapturedAccount:
    """Capture a fresh Claude Code login via an interactive `claude auth
    login --claudeai` prompt.

    POSIX-only; requires a TTY on stdin. Holds a cross-process lock over the
    whole operation, runs the pre-spawn version gate, logs in inside a
    gateway-created temp `CLAUDE_CONFIG_DIR`, captures the result, then
    always deletes the temp dir's scoped Keychain item and removes the temp
    dir.
    """
    if sys.platform == "win32":
        raise CaptureError(
            "interactive Claude login is not supported on Windows in this "
            "version; capture from an existing login with `--from <dir>` instead"
        )
    if not sys.stdin.isatty():
        raise CaptureError(
            "interactive Claude login requires an interactive terminal; capture "
            "from an existing login with `--from <dir>` instead"
        )

    with locking.file_lock(paths.runtime_dir() / _LOGIN_LOCK_FILENAME):
        claude_path = _resolve_claude_executable()
        identity = _stat_identity(claude_path)
        _check_claude_version(claude_path)

        keychain = _default_keychain_backend()
        config_dir = _mint_temp_config_dir(keychain)

        # Cancellation scope: SIGTERM/SIGHUP handlers are installed BEFORE
        # the login child is spawned and stay installed until credential
        # capture *and* cleanup have finished, so no signal in that window
        # can terminate the gateway while the login child, the temp
        # credentials, or the scoped Keychain item still exist. During
        # cleanup the handlers defer instead of raising, so Keychain
        # deletion and directory removal are never interrupted; a deferred
        # cancellation surfaces as `CaptureCancelled` after cleanup.
        signal_state = {"deferring": False, "deferred": False}

        def _on_cancel_signal(signal_number: int, _frame: Any) -> None:
            if signal_state["deferring"]:
                signal_state["deferred"] = True
                return
            raise _LoginSignalReceived(signal_number)

        previous_sigterm = signal.signal(signal.SIGTERM, _on_cancel_signal)
        previous_sighup = signal.signal(signal.SIGHUP, _on_cancel_signal)
        try:
            try:
                if _stat_identity(claude_path) != identity:
                    raise CaptureError(
                        f"the resolved `claude` executable at {claude_path} changed "
                        "between the version check and the login spawn; aborting"
                    )
                _run_login(claude_path, config_dir, timeout_secs)
                captured = _capture_from_config_dir_impl(config_dir, keychain)
            except (_LoginSignalReceived, KeyboardInterrupt) as exc:
                # Cancellation outside `_run_login`'s own window (e.g. while
                # reading the captured credentials) still cleans up fully —
                # via the finally below — and reports `CaptureCancelled`.
                raise CaptureCancelled("Claude login was cancelled") from exc
        finally:
            signal_state["deferring"] = True
            try:
                _cleanup_temp_config_dir(config_dir, keychain)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
                signal.signal(signal.SIGHUP, previous_sighup)
        if signal_state["deferred"]:
            raise CaptureCancelled("Claude login was cancelled")
        return captured
