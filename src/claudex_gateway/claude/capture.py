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
item (`Claude Code-credentials-<hex[:8]>`, see `scoped_keychain_service`)
are supported. The legacy unscoped Keychain item is never a credential
source, never written, and never deleted: when the expected scoped item is
absent, capture fails closed with a `CaptureError` naming the expected
service. Interactive capture does fingerprint the legacy item (read-only,
sha256) around the login spawn, so a failed capture can warn when an
unexpected `claude` build wrote the machine-level sign-in instead of the
scoped item.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

from claudex_gateway import locking, paths
from .capture_model import CapturedAccount, CaptureCancelled, CaptureError, KeychainBackend
from .keychain import (
    LEGACY_KEYCHAIN_SERVICE,
    default_keychain_backend,
    keychain_account,
    scoped_keychain_service,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGIN_LOCK_FILENAME = "claude-capture.lock"
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
        service = scoped_keychain_service(config_dir)
        account = keychain_account()
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


def capture_from_config_dir(
    config_dir: str, keychain: KeychainBackend | None = None
) -> CapturedAccount:
    """Capture a previously-completed Claude Code login from `config_dir`.

    Strictly read-only: never spawns `claude`, never writes or deletes
    anything. On macOS this reads only the scoped Keychain item for the
    exact `config_dir` string given (see `scoped_keychain_service`); on
    every other platform it reads `<config_dir>/.credentials.json`.
    """
    if keychain is None:
        keychain = default_keychain_backend()
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


# ---------------------------------------------------------------------------
# Claude executable resolution & legacy sign-in change detection
# ---------------------------------------------------------------------------


def resolve_claude_executable() -> str:
    resolved = shutil.which("claude")
    if resolved is None:
        raise CaptureError("the `claude` CLI was not found on PATH; install Claude Code first")
    # A relative PATH entry yields a relative result; absolutize so the login
    # spawn runs the same file regardless of any later working-directory change.
    return os.path.abspath(resolved)


# Distinguishes "the legacy item conclusively does not exist" (None) from "the
# legacy item could not be read at all" — comparing the latter against a later
# read would fabricate a change warning.
LEGACY_STATE_UNAVAILABLE = object()


def read_legacy_login_fingerprint(keychain: KeychainBackend) -> object:
    """Fingerprint the machine-level legacy Keychain sign-in, read-only.

    Returns the sha256 hex digest of the legacy item's value, `None` when the
    item conclusively does not exist, or `LEGACY_STATE_UNAVAILABLE` when the
    read failed or the platform has no Keychain. Never raises and never keeps
    the raw value: detection is best-effort and must not block or outlive a
    capture.
    """
    if sys.platform != "darwin":
        return LEGACY_STATE_UNAVAILABLE
    try:
        value = keychain.read(LEGACY_KEYCHAIN_SERVICE, keychain_account())
    except CaptureError:
        return LEGACY_STATE_UNAVAILABLE
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _warn_if_legacy_login_changed(keychain: KeychainBackend, baseline: object) -> None:
    """After a failed capture, warn when the legacy sign-in changed under it.

    The scoped-only contract means a `claude` build that writes the
    machine-level legacy Keychain item instead of the scoped one fails
    capture — but has silently replaced the user's own `claude` CLI sign-in.
    This turns that silent replacement into an explicit stderr warning with
    recovery guidance. Read-only and best-effort: any comparison gap
    (unavailable baseline or recheck) skips the warning rather than raising
    over the original capture error.
    """
    if baseline is LEGACY_STATE_UNAVAILABLE:
        return
    current = read_legacy_login_fingerprint(keychain)
    if current is LEGACY_STATE_UNAVAILABLE or current == baseline:
        return
    print(
        "WARNING: this machine's Claude Code sign-in (Keychain item "
        f"{LEGACY_KEYCHAIN_SERVICE!r}) changed during the Claude login "
        "capture. The login that just failed to capture may have replaced "
        "your existing `claude` CLI sign-in; run `claude` in a terminal and "
        "sign in again if so.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Interactive login
# ---------------------------------------------------------------------------


class _LoginSignalReceived(Exception):
    """Internal marker raised from a SIGINT/SIGTERM/SIGHUP handler during the
    interactive-capture cancellation scope."""

    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


# SIGHUP does not exist on Windows; guard the constant so importing this
# module (needed for the supported Windows `--from` path) never fails.
_CANCEL_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
    *((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()),
)

_LOGIN_WAIT_POLL_SECONDS = 0.1


class _CancellationScope:
    """Owns SIGINT, SIGTERM, and SIGHUP for one interactive capture with
    RECORD-ONLY handlers.

    A signal handler that can raise leaves some instruction boundary where
    the raise preempts the next critical statement — no amount of masking
    closes the interval before the mask is acquired. So inside this scope a
    cancellation signal never raises: it only sets `deferred`, and the
    capture flow polls that flag at its safe points (the login wait loop,
    and the final check after cleanup). Handlers are installed on entry —
    before the temp config dir is minted or the login child spawned, when no
    resource exists yet — and restored only on exit, after credential
    capture and cleanup have finished. The child inherits an unblocked
    signal mask, keeping its own graceful-SIGTERM handling intact."""

    def __init__(self) -> None:
        self.deferred = False
        self._previous: dict[int, Any] = {}

    def _handle(self, _signal_number: int, _frame: Any) -> None:
        self.deferred = True

    def __enter__(self) -> "_CancellationScope":
        for signal_number in _CANCEL_SIGNALS:
            self._previous[signal_number] = signal.signal(signal_number, self._handle)
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        for signal_number, previous in self._previous.items():
            signal.signal(signal_number, previous)
        self._previous.clear()


def child_process_env(config_dir: str) -> dict[str, str]:
    """Build the login child's environment: copy, point at the temp config
    dir, disable auto-update, and remove every auth-selector variable that
    could steer `claude auth login` away from a fresh interactive login."""
    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = config_dir
    env["DISABLE_UPDATES"] = "1"
    for var in _ENV_VARS_TO_REMOVE:
        env.pop(var, None)
    return env


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS returns EPERM when the group's only remnants are zombies we
        # can no longer signal. Every live descendant of the login child runs
        # under our own uid (kill-0 succeeds for those), so EPERM here means
        # nothing actionable is left in the group.
        return False
    return True


def _terminate_process_group(pgid: int, process: subprocess.Popen[Any]) -> None:
    """SIGTERM the login process group, wait a bounded grace period keyed on
    GROUP extinction (not leader exit — a descendant that ignores SIGTERM
    must not outlive cleanup), then SIGKILL the whole group, always reaping
    the leader.

    `pgid` is the group id captured at spawn time (`start_new_session=True`
    makes the child the leader of a group numbered by its own pid); it is
    passed in rather than looked up here so that a leader that already
    exited cannot race `getpgid` into skipping the group teardown while a
    descendant survives."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        # ESRCH: the whole group is gone. EPERM (macOS): only unsignalable
        # zombies remain. Nothing left to tear down either way.
        process.wait()
        return
    deadline = time.monotonic() + _PROCESS_GROUP_GRACE_SECONDS
    while time.monotonic() < deadline:
        # Reap the leader as soon as it exits: an unreaped zombie keeps the
        # process group alive and would otherwise pin the grace loop.
        process.poll()
        if not process_group_alive(pgid):
            process.wait()
            return
        time.sleep(0.05)
    # Grace expired with the group still alive: SIGKILL the group even when
    # the leader has already exited; a group that vanished (ESRCH) or holds
    # only unsignalable zombies (EPERM, macOS) is fine.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait()


def _run_login(
    claude_path: str,
    config_dir: str,
    timeout_secs: int,
    scope: _CancellationScope | None = None,
) -> None:
    """Spawn `claude auth login --claudeai` and wait for it to finish.

    stdin/stdout/stderr are inherited untouched — no output is piped or
    scanned. A nonzero exit is treated as a login denial/failure. The caller
    (`capture_interactive`) owns the record-only SIGINT/SIGTERM/SIGHUP
    handlers through `scope`; this function polls `scope.deferred` in a
    bounded wait loop, so cancellation and timeout both reach the same
    guaranteed process-group teardown without any raise ever landing inside
    it. Timeout raises a plain `CaptureError`; cancellation raises
    `CaptureCancelled`. (`KeyboardInterrupt`/`_LoginSignalReceived` are still
    translated for callers that drive this function without a scope.)
    """
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            [claude_path, "auth", "login", "--claudeai"],
            env=child_process_env(config_dir),
            start_new_session=True,
        )
        # start_new_session makes the child the leader of a fresh group
        # numbered by its own pid; capture it now so teardown never has to
        # ask a possibly-already-exited leader for it. With record-only
        # handlers no signal can interrupt this publication.
        login_pgid = process.pid
        deadline = time.monotonic() + timeout_secs
        while True:
            if scope is not None and scope.deferred:
                _terminate_process_group(login_pgid, process)
                raise CaptureCancelled("Claude login was cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(login_pgid, process)
                raise CaptureError(
                    f"`claude auth login --claudeai` timed out after {timeout_secs}s"
                )
            try:
                returncode = process.wait(timeout=min(_LOGIN_WAIT_POLL_SECONDS, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except (_LoginSignalReceived, KeyboardInterrupt) as exc:
        # Scope-less callers install raising handlers; a raise can land
        # anywhere, so teardown is still guaranteed on this path.
        if process is not None:
            _terminate_process_group(process.pid, process)
        raise CaptureCancelled("Claude login was cancelled") from exc

    if returncode != 0:
        raise CaptureError(f"`claude auth login --claudeai` exited with status {returncode}")


def mint_temp_config_dir(keychain: KeychainBackend) -> str:
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
            service = scoped_keychain_service(candidate)
            existing = keychain.read(service, keychain_account())
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


def cleanup_temp_config_dir(config_dir: str, keychain: KeychainBackend) -> None:
    """Delete the scoped Keychain item and remove the temp dir.

    Both actions are attempted even when one fails; any failure raises a
    sanitized `CaptureError` afterward instead of reporting success.
    """
    errors: list[str] = []
    if sys.platform == "darwin":
        try:
            keychain.delete(scoped_keychain_service(config_dir), keychain_account())
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
    whole operation, logs in inside a gateway-created temp
    `CLAUDE_CONFIG_DIR`, captures the result, then always deletes the temp
    dir's scoped Keychain item and removes the temp dir. On a failed capture
    it additionally warns when the machine-level legacy Keychain sign-in
    changed during the login (see `_warn_if_legacy_login_changed`).
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

    with locking.file_lock(paths.runtime_dir() / LOGIN_LOCK_FILENAME):
        claude_path = resolve_claude_executable()
        keychain = default_keychain_backend()
        legacy_baseline = read_legacy_login_fingerprint(keychain)

        # Cancellation scope: SIGINT/SIGTERM/SIGHUP are owned by RECORD-ONLY
        # handlers from BEFORE the temp config dir is minted until credential
        # capture *and* cleanup have finished. No signal in that window can
        # raise into the flow — so neither process-group teardown nor
        # cleanup can ever be interrupted — and no signal can terminate the
        # gateway while the login child, the temp credentials, or the scoped
        # Keychain item still exist. The flow polls `scope.deferred` at its
        # safe points; a recorded cancellation surfaces as
        # `CaptureCancelled` after full cleanup. A signal BEFORE the scope
        # begins is safe: no resource exists yet.
        config_dir: str | None = None
        try:
            with _CancellationScope() as scope:
                try:
                    try:
                        config_dir = mint_temp_config_dir(keychain)
                        if scope.deferred:
                            # Cancelled during minting: skip the interactive
                            # login entirely; the finally still cleans up.
                            raise CaptureCancelled("Claude login was cancelled")
                        _run_login(claude_path, config_dir, timeout_secs, scope)
                        captured = capture_from_config_dir(config_dir, keychain)
                    except (_LoginSignalReceived, KeyboardInterrupt) as exc:
                        # Programmatic interrupts (and raising handlers from
                        # scope-less callers) still clean up fully — via the
                        # finally below — and report `CaptureCancelled`.
                        raise CaptureCancelled("Claude login was cancelled") from exc
                finally:
                    if config_dir is not None:
                        cleanup_temp_config_dir(config_dir, keychain)
            if scope.deferred:
                raise CaptureCancelled("Claude login was cancelled")
        except CaptureError:
            # Cleanup has already run (the finally above); a legacy-item
            # change warning must never mask the capture error itself.
            _warn_if_legacy_login_changed(keychain, legacy_baseline)
            raise
        return captured
