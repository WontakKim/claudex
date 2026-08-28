"""Tests for the daemon-driven ChatGPT Pro login subprocess lifecycle."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Sequence
from typing import Any

import pytest

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
time.sleep(30)
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
