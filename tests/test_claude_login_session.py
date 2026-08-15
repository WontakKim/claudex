"""Tests for the daemon-driven Claude login session state machine.

Every scenario runs against the PATH-prepended fake `claude` (piped modes)
with `sys.platform` forced to "linux" so capture reads plain files instead of
touching the real macOS Keychain — mirrors test_account_cli.py. The capture
lock is acquired the same way the server handler will (try_file_lock), and
every terminal state asserts the lock was released and the temp config dir
cleaned up.
"""

from __future__ import annotations

import asyncio
import glob
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pytest

from claudex import paths
from claudex.claude import accounts as claude_accounts
from claudex.claude.login_session import (
    ClaudeLoginSession,
    LoginSessionStateError,
)
from claudex.locking import FileLockHandle, try_file_lock
from fake_claude import prepend_fake_claude


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def _linux_capture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    prepend_fake_claude(monkeypatch, tmp_path)


def _acquire_capture_lock() -> FileLockHandle:
    handle = try_file_lock(paths.runtime_dir() / "claude-capture.lock")
    assert handle is not None, "capture lock unexpectedly contended in a fresh HOME"
    return handle


def _temp_login_dirs() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "claudex-claude-login-*")))


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached within the timeout")


def _assert_released_and_clean(before_dirs: set[str]) -> None:
    # The capture lock must be reacquirable...
    handle = try_file_lock(paths.runtime_dir() / "claude-capture.lock")
    assert handle is not None, "capture lock was not released at terminal state"
    handle.release()
    # ...and no login temp dir may be left behind.
    assert _temp_login_dirs() <= before_dirs


def _run_to_terminal(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    drive=None,
    login_timeout: float = 10.0,
    confirm_timeout: float = 10.0,
) -> ClaudeLoginSession:
    """Start a session in `mode`, optionally drive it, and await terminal."""
    monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", mode)
    before_dirs = _temp_login_dirs()

    async def scenario() -> ClaudeLoginSession:
        session = ClaudeLoginSession(
            _acquire_capture_lock(),
            login_timeout=login_timeout,
            confirm_timeout=confirm_timeout,
        )
        session.start()
        if drive is not None:
            await drive(session)
        await _wait_until(lambda: session.is_terminal)
        # Wait for the driver's cleanup (temp dir, lock release) to finish so
        # the post-run assertions are deterministic.
        assert session._driver_task is not None
        await session._driver_task
        return session

    session = asyncio.run(scenario())
    _assert_released_and_clean(before_dirs)
    return session


def test_autocomplete_login_succeeds_and_registers_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _run_to_terminal("piped-autocomplete", monkeypatch)

    status = session.status()
    assert status["status"] == "succeeded"
    assert status["account"]["email"] == "fixture@example.com"
    assert status["url"].startswith("https://claude.com/cai/oauth/authorize")
    # The attempt id is minted at construction and never changes: a client
    # pinned to it at POST time stays attached for the session's lifetime.
    assert status["attempt_id"] == session.attempt_id
    assert re.fullmatch(r"[0-9a-f]{32}", status["attempt_id"])
    [record] = claude_accounts.list_accounts()
    assert record.email == "fixture@example.com"
    assert record.state == "ready"


def test_code_submission_completes_the_url_code_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: session.status()["status"] == "awaiting-browser")
        assert session.status()["code_prompt_detected"] is True
        assert session.status()["expires_at"] is not None
        await session.submit_code("good-code")

    session = _run_to_terminal("piped-url-code", monkeypatch, drive=drive)
    assert session.status()["status"] == "succeeded"
    [record] = claude_accounts.list_accounts()
    assert record.email == "fixture@example.com"


def test_bad_code_fails_with_the_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: session.status()["status"] == "awaiting-browser")
        await session.submit_code("wrong-code")

    session = _run_to_terminal("piped-url-code", monkeypatch, drive=drive)
    status = session.status()
    assert status["status"] == "failed"
    assert "exited with status 1" in status["error"]
    assert claude_accounts.list_accounts() == []


def test_denial_fails_fast_without_waiting_out_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = time.monotonic()
    session = _run_to_terminal("piped-denied", monkeypatch, login_timeout=60.0)
    elapsed = time.monotonic() - started

    status = session.status()
    assert status["status"] == "failed"
    assert "access_denied" in status["error"]
    # The fake sleeps 120s after printing the denial; fail-fast plus the 5s
    # process-group grace must come in far under the 60s login timeout.
    assert elapsed < 20.0


def test_login_timeout_kills_the_child_group(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leader_pid_file = tmp_path / "leader.pid"
    monkeypatch.setenv("CLAUDEX_TEST_LEADER_PID_FILE", str(leader_pid_file))

    session = _run_to_terminal("hang", monkeypatch, login_timeout=1.0)

    status = session.status()
    assert status["status"] == "failed"
    assert "timed out" in status["error"]
    leader_pid = int(leader_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(leader_pid, 0)


def test_cancel_terminates_as_cancelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    leader_pid_file = tmp_path / "leader.pid"
    monkeypatch.setenv("CLAUDEX_TEST_LEADER_PID_FILE", str(leader_pid_file))

    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: leader_pid_file.exists())
        session.request_cancel()
        session.request_cancel()  # idempotent

    session = _run_to_terminal("hang", monkeypatch, drive=drive, login_timeout=60.0)
    status = session.status()
    assert status["status"] == "cancelled"
    assert status["error"] is None
    leader_pid = int(leader_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(leader_pid, 0)


def _register_fixture_account() -> claude_accounts.AccountRecord:
    return claude_accounts.add_account(
        "fixture@example.com",
        None,
        None,
        {"claudeAiOauth": {"accessToken": "old-token", "email": "fixture@example.com"}},
        None,
    )


def test_duplicate_confirm_replace_updates_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _register_fixture_account()

    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: session.status()["status"] == "awaiting-replace")
        assert session.status()["email"] == "fixture@example.com"
        assert session.status()["existing_account_id"] == original.id
        assert session.status()["expires_at"] is not None
        session.confirm_replace(original.id)

    session = _run_to_terminal("piped-autocomplete", monkeypatch, drive=drive)

    status = session.status()
    assert status["status"] == "succeeded"
    assert status["account"]["id"] == original.id
    [record] = claude_accounts.list_accounts()
    assert record.id == original.id
    assert record.state == "ready"
    assert record.last_authenticated_at >= original.last_authenticated_at


def test_duplicate_confirm_replace_rejects_a_mismatched_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _register_fixture_account()

    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: session.status()["status"] == "awaiting-replace")
        with pytest.raises(LoginSessionStateError):
            session.confirm_replace("0a1b2c3d-4e5f-4678-9abc-def012345678")
        # The mismatch left the wait intact; the right id still confirms.
        session.confirm_replace(original.id)

    session = _run_to_terminal("piped-autocomplete", monkeypatch, drive=drive)

    assert session.status()["status"] == "succeeded"


def test_duplicate_cancel_declines_and_leaves_the_registry_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _register_fixture_account()

    async def drive(session: ClaudeLoginSession) -> None:
        await _wait_until(lambda: session.status()["status"] == "awaiting-replace")
        session.request_cancel()

    session = _run_to_terminal("piped-autocomplete", monkeypatch, drive=drive)

    status = session.status()
    assert status["status"] == "cancelled"
    [record] = claude_accounts.list_accounts()
    assert record == original


def test_duplicate_confirm_timeout_fails_and_discards_the_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _register_fixture_account()

    session = _run_to_terminal("piped-autocomplete", monkeypatch, confirm_timeout=0.3)

    status = session.status()
    assert status["status"] == "failed"
    assert "confirmation timed out" in status["error"]
    [record] = claude_accounts.list_accounts()
    assert record == original


def test_out_of_state_commands_raise_state_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def drive(session: ClaudeLoginSession) -> None:
        with pytest.raises(LoginSessionStateError):
            session.confirm_replace("any-id")  # nothing pending yet

    session = _run_to_terminal("piped-autocomplete", monkeypatch, drive=drive)
    assert session.status()["status"] == "succeeded"

    async def late_commands() -> None:
        with pytest.raises(LoginSessionStateError):
            await session.submit_code("late-code")
        with pytest.raises(LoginSessionStateError):
            session.confirm_replace("any-id")

    asyncio.run(late_commands())
    # Cancel after terminal is a silent no-op.
    session.request_cancel()
    assert session.status()["status"] == "succeeded"


def test_start_twice_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _run_to_terminal("piped-autocomplete", monkeypatch)

    async def restart() -> None:
        with pytest.raises(LoginSessionStateError):
            session.start()

    asyncio.run(restart())
