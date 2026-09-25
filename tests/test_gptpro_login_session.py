"""Tests for the daemon-driven ChatGPT Pro login subprocess lifecycle."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import browser, login, login_session
from claudex.gptpro.login_session import GptProLoginSession

_STATUS_KEYS = {"status", "started_at", "detail", "output", "error"}


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 2.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within the timeout")


async def _await_session(session: GptProLoginSession) -> dict[str, Any]:
    await _wait_until(lambda: session.is_terminal)
    assert session._driver_task is not None
    await session._driver_task
    return session.status()


def _command(script: str) -> Sequence[str]:
    return (sys.executable, "-c", script)


def test_login_session_succeeds_with_last_output_line_as_detail() -> None:
    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(
            command=_command("print('opening browser'); print('session saved')")
        )
        session.start()
        return await _await_session(session)

    status = asyncio.run(scenario())

    assert set(status) == _STATUS_KEYS
    assert status["status"] == "succeeded"
    assert status["detail"] == "session saved"
    assert status["output"] == "opening browser\nsession saved\n"
    assert status["error"] is None
    assert status["started_at"] is not None


def test_login_session_failure_preserves_classified_failure_line() -> None:
    script = """
import sys
print("opening browser")
print("gptpro login failed [session_rejected]: sign in again", file=sys.stderr)
raise SystemExit(1)
"""

    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(command=_command(script))
        session.start()
        return await _await_session(session)

    status = asyncio.run(scenario())

    assert set(status) == _STATUS_KEYS
    assert status["status"] == "failed"
    assert status["error"] == (
        "gptpro login failed [session_rejected]: sign in again"
    )
    assert "opening browser" in status["output"]


def test_login_session_cancel_terminates_as_cancelled() -> None:
    script = """
import time
print("waiting for sign-in", flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    pass
"""

    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(command=_command(script), timeout=10.0)
        session.start()
        await _wait_until(lambda: "waiting for sign-in" in session.status()["output"])
        session.request_cancel()
        session.request_cancel()
        status = await _await_session(session)
        session.request_cancel()
        return status

    status = asyncio.run(scenario())

    assert set(status) == _STATUS_KEYS
    assert status["status"] == "cancelled"
    assert status["detail"] == "waiting for sign-in"
    assert status["error"] is None


def test_login_session_timeout_fails_and_terminates_child() -> None:
    script = """
import time
print("waiting for sign-in", flush=True)
time.sleep(30)
"""

    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(command=_command(script), timeout=0.1)
        session.start()
        return await _await_session(session)

    status = asyncio.run(scenario())

    assert set(status) == _STATUS_KEYS
    assert status["status"] == "failed"
    assert status["error"] == "login timed out after 0.1s"


def test_login_session_rejects_a_second_start() -> None:
    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(
            command=_command("import time; time.sleep(30)"), timeout=10.0
        )
        session.start()
        with pytest.raises(RuntimeError, match="already started"):
            session.start()
        session.request_cancel()
        return await _await_session(session)

    status = asyncio.run(scenario())

    assert status["status"] == "cancelled"


def test_login_session_status_keys_are_stable_across_states() -> None:
    script = """
import time
print("running", flush=True)
time.sleep(0.1)
print("complete")
"""

    async def scenario() -> list[dict[str, Any]]:
        session = GptProLoginSession(command=_command(script))
        snapshots = [session.status()]
        session.start()
        await _wait_until(lambda: session.status()["status"] == "running")
        snapshots.append(session.status())
        snapshots.append(await _await_session(session))
        return snapshots

    snapshots = asyncio.run(scenario())

    assert [snapshot["status"] for snapshot in snapshots] == [
        "starting",
        "running",
        "succeeded",
    ]
    assert all(set(snapshot) == _STATUS_KEYS for snapshot in snapshots)
    assert snapshots[0]["started_at"] is None
    assert snapshots[1]["started_at"] is not None


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_process_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_alive(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} is still alive after {timeout}s")


def _kill_process_if_alive(pid: int) -> None:
    if not _process_is_alive(pid):
        return
    os.kill(pid, signal.SIGKILL)
    _assert_process_gone(pid)


@pytest.mark.parametrize("outcome", ["cancel", "timeout"])
def test_login_session_terminates_production_installer_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    login_group_path = tmp_path / "login-group.txt"
    installer_pid_path = tmp_path / "installer.pid"
    installer_group_path = tmp_path / "installer-group.txt"
    descendant_pid_path = tmp_path / "descendant.pid"
    real_python = sys.executable
    descendant_script = f"""
import os
import signal
import time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(descendant_pid_path)!r}).write_text(str(os.getpid()), encoding="utf-8")
time.sleep(60)
"""
    installer_executable = tmp_path / "fake-python"
    installer_executable.write_text(
        f"#!{real_python}\n"
        + f"""
import os
import subprocess
import time
from pathlib import Path
child = subprocess.Popen([{real_python!r}, "-c", {descendant_script!r}])
Path({str(installer_pid_path)!r}).write_text(str(os.getpid()), encoding="utf-8")
Path({str(installer_group_path)!r}).write_text(str(os.getpgrp()), encoding="utf-8")
print("installing Chromium", flush=True)
time.sleep(60)
""",
        encoding="utf-8",
    )
    installer_executable.chmod(0o755)
    login_script = f"""
import os
import sys
from pathlib import Path
from claudex.cli import gptpro
from claudex.gptpro import browser
from claudex.gptpro.login import LoginResult
Path({str(login_group_path)!r}).write_text(str(os.getpgrp()), encoding="utf-8")
sys.executable = {str(installer_executable)!r}
browser._PROCESS_TERMINATION_GRACE_SECONDS = 0.1
browser._OUTPUT_DRAIN_TIMEOUT_SECONDS = 0.1
async def run_login(*, on_status):
    on_status("no compatible browser found; installing Playwright Chromium")
    await browser.install_chromium(timeout_seconds=30.0)
    return LoginResult(True, None, "session saved")
gptpro.gptpro_login.run_login = run_login
raise SystemExit(gptpro._gptpro_main(["login"]))
"""
    monkeypatch.setattr(login_session, "_PROCESS_GROUP_GRACE_SECONDS", 1.0)
    process_ids: list[int] = []

    async def scenario() -> dict[str, Any]:
        session = GptProLoginSession(
            command=_command(login_script),
            timeout=1.0 if outcome == "timeout" else 10.0,
        )
        session.start()
        await _wait_until(
            lambda: all(
                path.exists()
                for path in (
                    login_group_path,
                    installer_pid_path,
                    installer_group_path,
                    descendant_pid_path,
                )
            )
        )
        if outcome == "cancel":
            session.request_cancel()
        return await _await_session(session)

    try:
        status = asyncio.run(scenario())
        installer_pid = int(installer_pid_path.read_text(encoding="utf-8"))
        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        process_ids.extend((installer_pid, descendant_pid))
        login_group = int(login_group_path.read_text(encoding="utf-8"))
        installer_group = int(installer_group_path.read_text(encoding="utf-8"))

        assert status["status"] == (
            "cancelled" if outcome == "cancel" else "failed"
        )
        assert installer_group == installer_pid
        assert installer_group != login_group
        _assert_process_gone(installer_pid)
        _assert_process_gone(descendant_pid)
    finally:
        if installer_pid_path.exists() and not process_ids:
            process_ids.append(
                int(installer_pid_path.read_text(encoding="utf-8"))
            )
        if descendant_pid_path.exists() and len(process_ids) < 2:
            process_ids.append(
                int(descendant_pid_path.read_text(encoding="utf-8"))
            )
        for process_id in process_ids:
            _kill_process_if_alive(process_id)


def test_dashboard_timeout_preserves_install_and_interactive_login_budgets() -> None:
    browser_work_seconds = (
        (2 * browser.NAVIGATION_TIMEOUT_MS + browser.COMPOSER_TIMEOUT_MS) / 1000
    )
    assert login_session._LOGIN_SESSION_TIMEOUT_SECONDS >= (
        browser.BROWSER_INSTALL_TIMEOUT_SECONDS
        + login.LOGIN_TIMEOUT_SECONDS
        + browser_work_seconds
        + 60.0
    )
    assert (
        browser._OUTPUT_DRAIN_TIMEOUT_SECONDS
        + browser._PROCESS_TERMINATION_GRACE_SECONDS
        + browser._PROCESS_EXTINCTION_TIMEOUT_SECONDS
        < login_session._PROCESS_GROUP_GRACE_SECONDS
    )
