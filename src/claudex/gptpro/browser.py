"""Playwright browser launch policy for gptpro commands."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext

NAVIGATION_TIMEOUT_MS = 20_000
COMPOSER_TIMEOUT_MS = 45_000
BROWSER_INSTALL_TIMEOUT_SECONDS = 5 * 60

_INSTALL_OUTPUT_CAP_CHARS = 2048
_OUTPUT_DRAIN_TIMEOUT_SECONDS = 1.0
_PROCESS_TERMINATION_GRACE_SECONDS = 2.0
_PROCESS_EXTINCTION_TIMEOUT_SECONDS = 0.5

_CHROME_MISSING_PATTERN = re.compile(
    r"Chromium distribution 'chrome' is not found", re.IGNORECASE
)
_PLAYWRIGHT_BROWSER_MISSING_PATTERN = re.compile(
    r"Executable doesn't exist at|playwright install(?!-deps\b)", re.IGNORECASE
)
_PLAYWRIGHT_OWNERS: dict[int, Any] = {}
PLAYWRIGHT_INSTALL_MESSAGE = (
    "playwright is not installed; reinstall the latest release tarball or run "
    "`uv sync` in a source checkout"
)
PROFILE_IN_USE_MESSAGE = "another gptpro ask is using the browser profile"


class GptProDependencyError(Exception):
    """Raised when the Playwright dependency is unavailable."""


class BrowserInstallError(Exception):
    """Raised when Playwright Chromium cannot be installed safely."""


def is_chrome_missing_error(exc: BaseException) -> bool:
    """Return whether Playwright reports that its Chrome channel is absent."""
    return _CHROME_MISSING_PATTERN.search(str(exc)) is not None


def is_browser_missing_error(exc: BaseException) -> bool:
    """Return whether no usable system or Playwright browser is installed."""
    return is_chrome_missing_error(exc) or (
        _PLAYWRIGHT_BROWSER_MISSING_PATTERN.search(str(exc)) is not None
    )


async def _start_playwright() -> Any:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise GptProDependencyError(PLAYWRIGHT_INSTALL_MESSAGE) from exc

    return await async_playwright().start()


def _process_group_alive(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _terminate_process_group(
    process_group_id: int, process: asyncio.subprocess.Process
) -> None:
    failures: list[str] = []
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        failures.append(
            f"could not terminate Chromium installer process group "
            f"{process_group_id}: {exc}"
        )

    deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline and _process_group_alive(process_group_id):
        await asyncio.sleep(0.05)

    if _process_group_alive(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            failures.append(
                f"could not kill Chromium installer process group "
                f"{process_group_id}: {exc}"
            )

    await process.wait()
    extinction_deadline = time.monotonic() + _PROCESS_EXTINCTION_TIMEOUT_SECONDS
    while (
        _process_group_alive(process_group_id)
        and time.monotonic() < extinction_deadline
    ):
        await asyncio.sleep(0.05)
    if _process_group_alive(process_group_id):
        failures.append(
            f"Chromium installer process group {process_group_id} did not exit"
        )
    if failures:
        raise BrowserInstallError("; ".join(failures))


async def _collect_bounded_output(
    stream: asyncio.StreamReader, output: list[str]
) -> None:
    while True:
        chunk = await stream.read(1024)
        if not chunk:
            return
        output[0] = (output[0] + chunk.decode("utf-8", errors="replace"))[
            -_INSTALL_OUTPUT_CAP_CHARS:
        ]


async def _finish_output_collection(task: asyncio.Task[None]) -> None:
    if task.done():
        await asyncio.gather(task, return_exceptions=True)
        return
    try:
        await asyncio.wait_for(
            asyncio.shield(task), _OUTPUT_DRAIN_TIMEOUT_SECONDS
        )
    except TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _cleanup_installer(
    process: asyncio.subprocess.Process,
    output_task: asyncio.Task[None],
) -> None:
    cleanup_failure: BrowserInstallError | None = None
    try:
        await _terminate_process_group(process.pid, process)
    except BrowserInstallError as exc:
        cleanup_failure = exc
    await _finish_output_collection(output_task)
    if cleanup_failure is not None:
        raise cleanup_failure


async def _wait_for_process_exit(
    process: asyncio.subprocess.Process, timeout_seconds: float
) -> None:
    async with asyncio.timeout(timeout_seconds):
        while process.returncode is None:
            await asyncio.sleep(0.05)


async def _wait_for_installer_cleanup(task: asyncio.Task[None]) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        try:
            await task
        except BaseException as cleanup_exc:
            cancellation.add_note(f"installer cleanup failed: {cleanup_exc}")
        raise


def _install_failure_message(summary: str, output: str) -> str:
    detail = output.strip()
    if not detail:
        return f"{summary}; installer produced no output"
    return (
        f"{summary}; installer output (last {_INSTALL_OUTPUT_CAP_CHARS} "
        f"characters): {detail}"
    )


async def install_chromium(
    *, timeout_seconds: float = BROWSER_INSTALL_TIMEOUT_SECONDS
) -> None:
    """Install matching Playwright Chromium with this Python interpreter."""
    if timeout_seconds <= 0:
        raise ValueError("Chromium installation timeout must be positive")

    command = (sys.executable, "-m", "playwright", "install", "chromium")
    command_text = shlex.join(command)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise BrowserInstallError(
            f"could not start Chromium installer `{command_text}`: {exc}"
        ) from exc

    assert process.stdout is not None
    output = [""]
    output_task = asyncio.create_task(
        _collect_bounded_output(process.stdout, output)
    )
    timeout_cause: TimeoutError | None = None
    primary_failure: BaseException | None = None
    cleanup_failure: BrowserInstallError | None = None
    try:
        await _wait_for_process_exit(process, timeout_seconds)
    except TimeoutError as exc:
        timeout_cause = exc
    except BaseException as exc:
        primary_failure = exc
    finally:
        cleanup_task = asyncio.create_task(
            _cleanup_installer(process, output_task)
        )
        try:
            await _wait_for_installer_cleanup(cleanup_task)
        except BrowserInstallError as exc:
            cleanup_failure = exc

    if timeout_cause is not None:
        error = BrowserInstallError(
            _install_failure_message(
                f"Chromium installer `{command_text}` timed out after "
                f"{timeout_seconds:g}s",
                output[0],
            )
        )
        if cleanup_failure is not None:
            error.add_note(f"installer cleanup failed: {cleanup_failure}")
        raise error from timeout_cause

    if primary_failure is not None:
        if cleanup_failure is not None:
            primary_failure.add_note(
                f"installer cleanup failed: {cleanup_failure}"
            )
        raise primary_failure

    if cleanup_failure is not None:
        raise BrowserInstallError(
            _install_failure_message(
                f"Chromium installer `{command_text}` cleanup failed: "
                f"{cleanup_failure}",
                output[0],
            )
        ) from cleanup_failure

    if process.returncode != 0:
        raise BrowserInstallError(
            _install_failure_message(
                f"Chromium installer `{command_text}` exited with status "
                f"{process.returncode}",
                output[0],
            )
        )


async def _launch_persistent_context(
    chromium: Any, profile_dir: Path, options: dict[str, object]
) -> BrowserContext:
    try:
        return await chromium.launch_persistent_context(
            str(profile_dir), channel="chrome", **options
        )
    except Exception as exc:
        if not is_chrome_missing_error(exc):
            raise
    return await chromium.launch_persistent_context(str(profile_dir), **options)


async def launch_persistent_profile(
    profile_dir: Path,
    *,
    headless: bool = False,
) -> BrowserContext:
    """Launch a persistent profile, preferring system Chrome."""
    options: dict[str, object] = {
        "headless": headless,
        "args": ["--disable-blink-features=AutomationControlled"],
        "ignore_default_args": ["--enable-automation"],
    }
    playwright = await _start_playwright()
    try:
        context = await _launch_persistent_context(
            playwright.chromium, profile_dir, options
        )
    except BaseException:
        await playwright.stop()
        raise
    _PLAYWRIGHT_OWNERS[id(context)] = playwright
    return context


async def _launch_headless_browser(chromium: Any) -> Browser:
    try:
        return await chromium.launch(channel="chrome", headless=True)
    except Exception as exc:
        if not is_chrome_missing_error(exc):
            raise
    return await chromium.launch(headless=True)


async def launch_headless_probe_chromium() -> Browser:
    """Launch a headless probe browser, preferring system Chrome."""
    playwright = await _start_playwright()
    try:
        browser = await _launch_headless_browser(playwright.chromium)
    except BaseException:
        await playwright.stop()
        raise
    _PLAYWRIGHT_OWNERS[id(browser)] = playwright
    return browser


def remove_headless_user_agent_token(user_agent: str) -> str:
    """Remove Playwright's visible headless marker from a user agent."""
    return user_agent.replace("Headless", "")


async def create_headless_probe_context(
    browser: Browser, *, storage_state: Path
) -> BrowserContext:
    """Create a probe context with saved state and a normalized user agent."""
    user_agent_context = await browser.new_context()
    try:
        user_agent_page = await user_agent_context.new_page()
        user_agent = await user_agent_page.evaluate("navigator.userAgent")
    finally:
        await user_agent_context.close()

    context_options: dict[str, object] = {"storage_state": str(storage_state)}
    if isinstance(user_agent, str) and "Headless" in user_agent:
        context_options["user_agent"] = remove_headless_user_agent_token(user_agent)
    return await browser.new_context(**context_options)


async def close_playwright_resource(resource: Browser | BrowserContext) -> None:
    """Close a launched resource and its lazily started Playwright driver."""
    playwright = _PLAYWRIGHT_OWNERS.pop(id(resource), None)
    try:
        await resource.close()
    finally:
        if playwright is not None:
            await playwright.stop()
