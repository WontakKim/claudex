"""Async, daemon-driven ChatGPT Pro login session for the dashboard.

The daemon runs the existing `claudex-gateway gptpro login` CLI in its own
process group, captures its merged output, and exposes a small polling state
machine without importing the optional Playwright dependency into the daemon
at module import time.

States:
    starting → running → succeeded | failed | cancelled

The CLI remains the single owner of browser login and session persistence.
This wrapper only manages its process lifetime and reports bounded progress
output to callers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

_LOGIN_SESSION_TIMEOUT_SECONDS = 360.0
_OUTPUT_CAP_CHARS = 4096
_PROCESS_GROUP_GRACE_SECONDS = 5.0
_LOGIN_COMMAND: tuple[str, ...] = (
    sys.executable,
    "-m",
    "claudex",
    "gptpro",
    "login",
)
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_FAILURE_LINE_PREFIX = "gptpro login failed ["


class GptProLoginSession:
    """One daemon-driven ChatGPT Pro login, from spawn to CLI exit."""

    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        timeout: float = _LOGIN_SESSION_TIMEOUT_SECONDS,
    ) -> None:
        self._command = tuple(command) if command is not None else _LOGIN_COMMAND
        self._timeout = timeout

        self._status = "starting"
        self._started_at: float | None = None
        self._detail: str | None = None
        self._output = ""
        self._error: str | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._cancel_event = asyncio.Event()
        self._driver_task: asyncio.Task[None] | None = None
        self._pump_task: asyncio.Task[None] | None = None

    # -- public surface ----------------------------------------------------

    def start(self) -> None:
        """Spawn the driver task. Call exactly once, from the event loop."""
        if self._driver_task is not None:
            raise RuntimeError("login session already started")
        self._started_at = time.time()
        self._driver_task = asyncio.create_task(self._run())

    def status(self) -> dict[str, Any]:
        """A JSON-ready snapshot with a stable key set."""
        return {
            "status": self._status,
            "started_at": self._started_at,
            "detail": self._detail,
            "output": self._output,
            "error": self._error,
        }

    @property
    def is_terminal(self) -> bool:
        return self._status in _TERMINAL_STATES

    def request_cancel(self) -> None:
        """Idempotently ask the driver to cancel; a no-op once terminal."""
        if self.is_terminal:
            return
        self._cancel_event.set()

    # -- driver ------------------------------------------------------------

    async def _run(self) -> None:
        wait_tasks: tuple[asyncio.Task[Any], ...] = ()
        try:
            process = await asyncio.create_subprocess_exec(
                *self._command,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                # `run_login(on_status=print)` progress must reach the daemon
                # immediately instead of waiting in the CLI's stdout buffer.
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._process = process
            self._status = "running"
            assert process.stdout is not None
            self._pump_task = asyncio.create_task(self._pump_output(process.stdout))

            process_wait_task = asyncio.create_task(process.wait())
            cancel_task = asyncio.create_task(self._cancel_event.wait())
            wait_tasks = (process_wait_task, cancel_task)
            done, _pending = await asyncio.wait(
                {process_wait_task, cancel_task},
                timeout=self._timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if cancel_task in done:
                await _terminate_process_group(process.pid, process)
                self._status = "cancelled"
                return
            if process_wait_task not in done:
                await _terminate_process_group(process.pid, process)
                self._fail(f"login timed out after {self._timeout:g}s")
                return

            await process_wait_task
            if self._pump_task is not None:
                await self._pump_task
            if process.returncode == 0:
                self._status = "succeeded"
                return
            self._fail(self._failure_line(process.returncode))
        except OSError as exc:
            self._fail(f"login process error: {exc}")
        except asyncio.CancelledError:
            self._fail("the login driver was cancelled")
            raise
        except Exception:
            logger.exception("unexpected failure in the ChatGPT Pro login driver")
            self._fail("unexpected login failure; see the gateway log")
        finally:
            for task in wait_tasks:
                if not task.done():
                    task.cancel()
            if wait_tasks:
                await asyncio.gather(*wait_tasks, return_exceptions=True)
            if self._pump_task is not None:
                self._pump_task.cancel()
                await asyncio.gather(self._pump_task, return_exceptions=True)
            process = self._process
            if process is not None and process.returncode is None:
                await _terminate_process_group(process.pid, process)
            if not self.is_terminal:
                self._fail("the login driver exited unexpectedly")

    async def _pump_output(self, stream: asyncio.StreamReader) -> None:
        """Retain bounded merged output and the last non-empty output line."""
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                return
            self._output = (self._output + chunk.decode("utf-8", errors="replace"))[
                -_OUTPUT_CAP_CHARS:
            ]
            for line in reversed(self._output.splitlines()):
                stripped = line.strip()
                if stripped:
                    self._detail = stripped
                    break

    def _failure_line(self, returncode: int | None) -> str:
        for line in reversed(self._output.splitlines()):
            stripped = line.strip()
            if stripped.startswith(_FAILURE_LINE_PREFIX):
                return stripped
        if self._detail is not None:
            return self._detail
        return f"`claudex-gateway gptpro login` exited with status {returncode}"

    def _fail(self, message: str) -> None:
        self._status = "failed"
        self._error = message


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _terminate_process_group(
    pgid: int, process: asyncio.subprocess.Process
) -> None:
    """SIGTERM the group, wait for extinction, then SIGKILL and reap."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        await process.wait()
        return

    deadline = time.monotonic() + _PROCESS_GROUP_GRACE_SECONDS
    while time.monotonic() < deadline and _process_group_alive(pgid):
        await asyncio.sleep(0.05)
    if _process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    await process.wait()
