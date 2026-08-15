"""Async, daemon-driven Claude login session for the dashboard add-account flow.

The CLI's `capture_interactive` needs a TTY and inherited stdio; a dashboard
login runs the same `claude auth login --claudeai` capture with piped stdio
instead, driven by HTTP-shaped commands (status poll, code paste, replace
confirmation, cancel). This module owns that state machine; the temp-dir
mint/cleanup, scoped-Keychain capture, identity resolution, and process-group
teardown semantics are claude_capture's, imported directly so the two flows
can never drift (same-package reuse, mirrored by the test suite).

Verified against claude 2.1.224 with piped stdio
(.docs/research/claude-login-piped-stdio.md): the authorize URL is printed to
stdout (wrapped in an OSC-8 hyperlink, so ESC terminates it), the paste-code
prompt appears immediately, and completion is signalled solely by the child
exiting 0 — whether the browser's localhost callback finished the flow or a
pasted code did is invisible here, by design.

States:
    starting → awaiting-browser → completing
        → succeeded | failed | cancelled
        → awaiting-replace → succeeded | failed | cancelled

Every session carries an `attempt_id` minted at construction. Handlers pin
mutating commands (and attached polls) to it via the X-Login-Attempt header,
so a stale dashboard tab from an earlier attempt can never drive a newer
session that happens to be in the same state.

Single-writer discipline: only the driver task and the output pump (both
event-loop coroutines) mutate the state; command methods only signal events
or write to the child's stdin. Secrets never appear in status payloads,
errors, or logs — the captured credential lives in driver-local variables and
is dropped at resolution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
import uuid
from typing import Any

from pathlib import Path

from claudex_gateway import claude_accounts, paths
from claudex_gateway.claude_capture import (
    LEGACY_STATE_UNAVAILABLE,
    LOGIN_LOCK_FILENAME,
    capture_from_config_dir,
    child_process_env,
    cleanup_temp_config_dir,
    mint_temp_config_dir,
    process_group_alive,
    read_legacy_login_fingerprint,
    resolve_claude_executable,
)
from claudex_gateway.claude_capture_model import CaptureError
from claudex_gateway.claude_keychain import default_keychain_backend
from claudex_gateway.locking import FileLockHandle


def capture_lock_path() -> Path:
    """The cross-process login lock shared with the CLI's capture_interactive."""
    return paths.runtime_dir() / LOGIN_LOCK_FILENAME

logger = logging.getLogger(__name__)

_LOGIN_TIMEOUT_SECONDS = 180.0
# Bound on holding the captured credential in memory while the owner decides
# whether to replace a duplicate registration.
_CONFIRM_TIMEOUT_SECONDS = 300.0
_OUTPUT_CAP_CHARS = 4096
_PROCESS_GROUP_GRACE_SECONDS = 5.0

# Spike-verified patterns (claude 2.1.224, piped stdio). The URL is emitted
# inside an OSC-8 hyperlink, so ESC/BEL terminate it alongside whitespace.
_URL_PATTERN = re.compile(r"https://claude\.com/cai/oauth/authorize\?[^\s\x1b\x07]+")
_CODE_PROMPT_PATTERN = re.compile(r"Paste code here if prompted")
_DENIAL_PATTERN = re.compile(r"access_denied")

_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


class LoginSessionStateError(Exception):
    """A command arrived in a state that cannot honor it (handler: 409)."""


class ClaudeLoginSession:
    """One dashboard-driven login, from spawn to registry commit."""

    def __init__(
        self,
        lock_handle: FileLockHandle,
        *,
        login_timeout: float = _LOGIN_TIMEOUT_SECONDS,
        confirm_timeout: float = _CONFIRM_TIMEOUT_SECONDS,
    ) -> None:
        self._lock_handle = lock_handle
        self._login_timeout = login_timeout
        self._confirm_timeout = confirm_timeout

        self._attempt_id = uuid.uuid4().hex
        self._status = "starting"
        self._url: str | None = None
        self._code_prompt_detected = False
        self._expires_at: float | None = None
        self._email: str | None = None
        self._existing_account_id: str | None = None
        self._account_row: dict[str, Any] | None = None
        self._error: str | None = None
        self._detail: str | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._output = ""
        self._cancel_event = asyncio.Event()
        self._denial_event = asyncio.Event()
        self._confirm_event = asyncio.Event()
        self._driver_task: asyncio.Task[None] | None = None
        self._pump_task: asyncio.Task[None] | None = None

    # -- public surface ----------------------------------------------------

    def start(self) -> None:
        """Spawn the driver task. Call exactly once, from the event loop."""
        if self._driver_task is not None:
            raise LoginSessionStateError("login session already started")
        self._driver_task = asyncio.create_task(self._run())

    def status(self) -> dict[str, Any]:
        """A JSON-ready snapshot with a stable key set."""
        return {
            "attempt_id": self._attempt_id,
            "status": self._status,
            "url": self._url,
            "code_prompt_detected": self._code_prompt_detected,
            "expires_at": self._expires_at,
            "email": self._email,
            "existing_account_id": self._existing_account_id,
            "account": self._account_row,
            "error": self._error,
            "detail": self._detail,
        }

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def is_terminal(self) -> bool:
        return self._status in _TERMINAL_STATES

    async def submit_code(self, code: str) -> None:
        """Forward a pasted OAuth code to the login child's stdin."""
        if self._status != "awaiting-browser":
            raise LoginSessionStateError(
                f"a code can only be submitted while awaiting the browser "
                f"(current state: {self._status})"
            )
        process = self._process
        if process is None or process.stdin is None:
            raise LoginSessionStateError("the login process is not accepting input")
        try:
            process.stdin.write((code + "\n").encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            raise LoginSessionStateError(
                "the login process is no longer accepting input"
            ) from exc

    def confirm_replace(self, existing_account_id: str) -> None:
        """Confirm replacing the already-registered duplicate account.

        The caller must name the account being replaced — the id exposed in
        status() — so a confirmation can never apply to a different record
        than the one the owner saw. Declining is not a command: the owner
        cancels the session instead (DELETE), which the duplicate wait
        already honors.
        """
        if self._status != "awaiting-replace":
            raise LoginSessionStateError(
                f"no replacement confirmation is pending (current state: {self._status})"
            )
        if existing_account_id != self._existing_account_id:
            raise LoginSessionStateError(
                "existing_account_id does not match the pending replacement target"
            )
        self._confirm_event.set()

    def request_cancel(self) -> None:
        """Idempotently ask the driver to cancel; a no-op once terminal."""
        if self.is_terminal:
            return
        self._cancel_event.set()

    # -- driver ------------------------------------------------------------

    async def _run(self) -> None:
        keychain = default_keychain_backend()
        config_dir: str | None = None
        cleaned = False
        legacy_baseline: object = LEGACY_STATE_UNAVAILABLE
        try:
            claude_path = resolve_claude_executable()
            legacy_baseline = await asyncio.to_thread(read_legacy_login_fingerprint, keychain)
            config_dir = await asyncio.to_thread(mint_temp_config_dir, keychain)

            process = await asyncio.create_subprocess_exec(
                claude_path,
                "auth",
                "login",
                "--claudeai",
                env=child_process_env(config_dir),
                stdin=asyncio.subprocess.PIPE,
                # stdin stays open for the child's lifetime: the browser
                # callback server's lifetime is bound to it, and the pasted
                # code travels through it.
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._process = process
            pgid = process.pid
            assert process.stdout is not None
            self._pump_task = asyncio.create_task(self._pump_output(process.stdout))
            self._expires_at = time.time() + self._login_timeout

            outcome = await self._await_first(
                {
                    "cancel": self._cancel_event.wait(),
                    "denial": self._denial_event.wait(),
                    "process": process.wait(),
                },
                timeout=self._login_timeout,
            )
            if outcome == "cancel":
                await _terminate_process_group_async(pgid, process)
                self._finish_cancelled()
                return
            if outcome == "denial":
                await _terminate_process_group_async(pgid, process)
                self._fail("the browser login was denied (access_denied)")
                return
            if outcome == "timeout":
                await _terminate_process_group_async(pgid, process)
                self._fail(f"login timed out after {int(self._login_timeout)}s")
                return
            if process.returncode != 0:
                self._fail(
                    f"`claude auth login --claudeai` exited with status {process.returncode}"
                )
                return

            self._status = "completing"
            self._expires_at = None

            captured = await asyncio.to_thread(
                capture_from_config_dir, config_dir, keychain
            )
            # Early cleanup shrinks the window in which a daemon crash leaks
            # the temp dir and its scoped Keychain item.
            await asyncio.to_thread(cleanup_temp_config_dir, config_dir, keychain)
            cleaned = True
            try:
                try:
                    record = await asyncio.to_thread(
                        claude_accounts.add_account,
                        captured.email,
                        captured.organization_uuid,
                        captured.organization_name,
                        captured.credentials_json,
                        captured.oauth_account_json,
                    )
                except claude_accounts.DuplicateAccountError:
                    record = await self._resolve_duplicate(captured)
                    if record is None:
                        return
                self._account_row = record.to_row()
                self._status = "succeeded"
            finally:
                captured = None  # drop the in-memory credential at resolution
        except (CaptureError, claude_accounts.AccountRegistryError) as exc:
            self._fail(str(exc))
        except OSError as exc:
            self._fail(f"login process error: {exc}")
        except asyncio.CancelledError:
            self._fail("the login driver was cancelled")
            raise
        except Exception:
            logger.exception("unexpected failure in the Claude login driver")
            self._fail("unexpected login failure; see the gateway log")
        finally:
            # The inner finally guarantees the terminal state and the lock
            # release even when one of the awaits below is itself cancelled
            # (a CancelledError raised mid-finally would skip later lines).
            try:
                if self._pump_task is not None:
                    self._pump_task.cancel()
                    await asyncio.gather(self._pump_task, return_exceptions=True)
                process = self._process
                if process is not None and process.returncode is None:
                    await _terminate_process_group_async(process.pid, process)
                if config_dir is not None and not cleaned:
                    try:
                        await asyncio.to_thread(cleanup_temp_config_dir, config_dir, keychain)
                    except CaptureError as exc:
                        logger.warning("login temp-dir cleanup failed: %s", exc)
                if self._status == "failed":
                    await asyncio.to_thread(
                        self._warn_if_legacy_login_changed, keychain, legacy_baseline
                    )
            finally:
                if not self.is_terminal:
                    # The driver must never end in a non-terminal state.
                    self._fail("the login driver exited unexpectedly")
                self._lock_handle.release()

    async def _resolve_duplicate(
        self, captured: Any
    ) -> claude_accounts.AccountRecord | None:
        """Hold the captured credential until the owner confirms replacement.

        Returns the updated record on confirm, or None after setting a
        terminal state (cancelled or timed out — declining IS cancelling).
        The registry row being replaced is resolved before entering
        awaiting-replace so status() can name it and confirm_replace can
        pin the confirmation to it.
        """
        self._email = captured.email
        self._existing_account_id = await asyncio.to_thread(
            _find_registered_account_id, captured.email, captured.organization_uuid
        )
        if self._existing_account_id is None:
            # The duplicate row vanished between add_account's rejection and
            # this lookup (concurrent removal); the capture cannot be
            # attributed to a record the owner can see, so fail loudly.
            self._fail(
                "the duplicate account row disappeared while resolving the "
                "replacement target; retry the login"
            )
            return None
        self._expires_at = time.time() + self._confirm_timeout
        self._status = "awaiting-replace"

        outcome = await self._await_first(
            {
                "cancel": self._cancel_event.wait(),
                "confirm": self._confirm_event.wait(),
            },
            timeout=self._confirm_timeout,
        )
        if outcome == "cancel":
            self._finish_cancelled()
            return None
        if outcome == "timeout":
            self._fail("replacement confirmation timed out; the captured login was discarded")
            return None
        self._status = "completing"
        self._expires_at = None
        return await asyncio.to_thread(
            claude_accounts.update_account_credentials,
            captured.email,
            captured.organization_uuid,
            captured.organization_name,
            captured.credentials_json,
            captured.oauth_account_json,
        )

    async def _pump_output(self, stream: asyncio.StreamReader) -> None:
        """Scan the child's merged output for the URL, code prompt, denial."""
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                return
            self._output = (self._output + chunk.decode("utf-8", errors="replace"))[
                -_OUTPUT_CAP_CHARS:
            ]
            if self._url is None:
                match = _URL_PATTERN.search(self._output)
                if match:
                    self._url = match.group(0)
                    if self._status == "starting":
                        self._status = "awaiting-browser"
            if not self._code_prompt_detected and _CODE_PROMPT_PATTERN.search(self._output):
                self._code_prompt_detected = True
            if _DENIAL_PATTERN.search(self._output):
                self._denial_event.set()

    async def _await_first(
        self, waiters: dict[str, Any], timeout: float
    ) -> str:
        """Await the first of named waiters; 'timeout' when none completes.

        On simultaneous completion the priority is cancel > denial > confirm >
        process. Pending waiter tasks are always cancelled and reaped."""
        tasks = {name: asyncio.ensure_future(waiter) for name, waiter in waiters.items()}
        try:
            done, _pending = await asyncio.wait(
                tasks.values(), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                return "timeout"
            for name in ("cancel", "denial", "confirm", "process"):
                task = tasks.get(name)
                if task is not None and task in done:
                    return name
            raise AssertionError("unreachable: unnamed waiter completed")
        finally:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    def _fail(self, message: str) -> None:
        self._status = "failed"
        self._error = message
        self._expires_at = None

    def _finish_cancelled(self) -> None:
        self._status = "cancelled"
        self._expires_at = None

    def _warn_if_legacy_login_changed(self, keychain: Any, baseline: object) -> None:
        """Daemon analog of claude_capture's stderr warning: log it instead,
        so it lands in the dashboard's Logs tab. Read-only and best-effort."""
        if baseline is LEGACY_STATE_UNAVAILABLE:
            return
        current = read_legacy_login_fingerprint(keychain)
        if current is LEGACY_STATE_UNAVAILABLE or current == baseline:
            return
        logger.warning(
            "this machine's Claude Code sign-in changed during the dashboard "
            "login capture; the failed login may have replaced the machine's "
            "own `claude` CLI sign-in — run `claude` in a terminal and sign "
            "in again if so"
        )


def _find_registered_account_id(email: str, organization_uuid: str | None) -> str | None:
    """Resolve the registry row `add_account` collided with, by its own key.

    Uses the same normalization as the duplicate check so the lookup can
    only miss when the row was concurrently removed.
    """
    normalized_email = claude_accounts._normalize_email(email)
    normalized_organization_uuid = claude_accounts._normalize_optional_text(
        organization_uuid, field="organizationUuid"
    )
    for record in claude_accounts.list_accounts():
        if (
            record.email == normalized_email
            and record.organization_uuid == normalized_organization_uuid
        ):
            return record.id
    return None


async def _terminate_process_group_async(
    pgid: int, process: asyncio.subprocess.Process
) -> None:
    """SIGTERM the group, wait out the grace keyed on group extinction, then
    SIGKILL — the async mirror of claude_capture._terminate_process_group.
    The leader is always reaped."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + _PROCESS_GROUP_GRACE_SECONDS
    while time.monotonic() < deadline and process_group_alive(pgid):
        await asyncio.sleep(0.1)
    if process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()
