"""Tests for the gptpro Playwright browser launch policy."""

from __future__ import annotations

import ast
import asyncio
import builtins
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import browser


class _FakeResource:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, chrome_error: BaseException | None = None) -> None:
        self.chrome_error = chrome_error
        self.context = _FakeResource()
        self.browser = _FakeResource()
        self.persistent_calls: list[tuple[str, dict[str, object]]] = []
        self.launch_calls: list[dict[str, object]] = []

    async def launch_persistent_context(
        self, profile_dir: str, **options: object
    ) -> _FakeResource:
        self.persistent_calls.append((profile_dir, options))
        if options.get("channel") == "chrome" and self.chrome_error is not None:
            raise self.chrome_error
        return self.context

    async def launch(self, **options: object) -> _FakeResource:
        self.launch_calls.append(options)
        if options.get("channel") == "chrome" and self.chrome_error is not None:
            raise self.chrome_error
        return self.browser


class _FakePlaywright:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def test_lazy_playwright_import_has_actionable_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_playwright(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "playwright.async_api":
            raise ModuleNotFoundError(
                "No module named 'playwright'", name="playwright"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_playwright)

    with pytest.raises(browser.GptProDependencyError) as raised:
        asyncio.run(browser._start_playwright())

    assert str(raised.value) == (
        "playwright is not installed; reinstall the latest release tarball or run "
        "`uv sync` in a source checkout"
    )
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)


def test_persistent_profile_falls_back_only_for_missing_chrome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chromium = _FakeChromium(
        RuntimeError("Chromium distribution 'chrome' is not found at /Applications")
    )
    playwright = _FakePlaywright(chromium)

    async def start_playwright() -> _FakePlaywright:
        return playwright

    monkeypatch.setattr(browser, "_start_playwright", start_playwright)

    async def scenario() -> None:
        context = await browser.launch_persistent_profile(tmp_path / "profile")
        assert context is chromium.context
        await browser.close_playwright_resource(context)

    asyncio.run(scenario())

    assert len(chromium.persistent_calls) == 2
    first_profile, first_options = chromium.persistent_calls[0]
    second_profile, second_options = chromium.persistent_calls[1]
    assert first_profile == second_profile == str(tmp_path / "profile")
    assert first_options["channel"] == "chrome"
    assert "channel" not in second_options
    for options in (first_options, second_options):
        assert options["headless"] is False
        assert options["args"] == [
            "--disable-blink-features=AutomationControlled"
        ]
        assert options["ignore_default_args"] == ["--enable-automation"]
    assert chromium.context.closed
    assert playwright.stopped


def test_persistent_profile_reraises_nonmatching_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch_error = RuntimeError("browser profile is already in use")
    chromium = _FakeChromium(launch_error)
    playwright = _FakePlaywright(chromium)

    async def start_playwright() -> _FakePlaywright:
        return playwright

    monkeypatch.setattr(browser, "_start_playwright", start_playwright)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(browser.launch_persistent_profile(tmp_path / "profile"))

    assert raised.value is launch_error
    assert len(chromium.persistent_calls) == 1
    assert playwright.stopped


@pytest.mark.parametrize("resource_kind", ["persistent", "headless"])
def test_launch_cancellation_stops_the_playwright_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resource_kind: str
) -> None:
    chromium = _FakeChromium(asyncio.CancelledError())
    playwright = _FakePlaywright(chromium)

    async def start_playwright() -> _FakePlaywright:
        return playwright

    monkeypatch.setattr(browser, "_start_playwright", start_playwright)

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            if resource_kind == "persistent":
                await browser.launch_persistent_profile(tmp_path / "profile")
            else:
                await browser.launch_headless_probe_chromium()

    asyncio.run(scenario())

    assert playwright.stopped
    assert id(chromium.context) not in browser._PLAYWRIGHT_OWNERS
    assert id(chromium.browser) not in browser._PLAYWRIGHT_OWNERS


class _CancellingCloseResource(_FakeResource):
    async def close(self) -> None:
        self.closed = True
        raise asyncio.CancelledError


def test_close_cancellation_still_stops_and_unregisters_the_driver() -> None:
    resource = _CancellingCloseResource()
    playwright = _FakePlaywright(_FakeChromium())
    browser._PLAYWRIGHT_OWNERS[id(resource)] = playwright

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await browser.close_playwright_resource(resource)

    asyncio.run(scenario())

    assert resource.closed
    assert playwright.stopped
    assert id(resource) not in browser._PLAYWRIGHT_OWNERS


def test_headless_probe_uses_the_same_chrome_fallback_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chromium = _FakeChromium(
        RuntimeError("chromium DISTRIBUTION 'CHROME' IS NOT FOUND")
    )
    playwright = _FakePlaywright(chromium)

    async def start_playwright() -> _FakePlaywright:
        return playwright

    monkeypatch.setattr(browser, "_start_playwright", start_playwright)

    async def scenario() -> None:
        launched = await browser.launch_headless_probe_chromium()
        assert launched is chromium.browser
        await browser.close_playwright_resource(launched)

    asyncio.run(scenario())

    assert chromium.launch_calls == [
        {"channel": "chrome", "headless": True},
        {"headless": True},
    ]
    assert playwright.stopped


class _FakeUserAgentPage:
    async def evaluate(self, expression: str) -> str:
        assert expression == "navigator.userAgent"
        return "Mozilla/5.0 HeadlessChrome/140.0"


class _FakeUserAgentContext(_FakeResource):
    async def new_page(self) -> _FakeUserAgentPage:
        return _FakeUserAgentPage()


class _FakeProbeBrowser:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.probe_context = _FakeResource()

    async def new_context(self, **options: Any) -> _FakeResource:
        self.calls.append(options)
        if not options:
            return _FakeUserAgentContext()
        return self.probe_context


def test_probe_context_removes_headless_user_agent_token(tmp_path: Path) -> None:
    fake_browser = _FakeProbeBrowser()
    storage_state = tmp_path / "session.json"

    context = asyncio.run(
        browser.create_headless_probe_context(
            fake_browser, storage_state=storage_state
        )
    )

    assert context is fake_browser.probe_context
    assert fake_browser.calls == [
        {},
        {
            "storage_state": str(storage_state),
            "user_agent": "Mozilla/5.0 Chrome/140.0",
        },
    ]


def test_persistent_profile_supports_headless_runtime_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chromium = _FakeChromium()
    playwright = _FakePlaywright(chromium)

    async def start_playwright() -> _FakePlaywright:
        return playwright

    monkeypatch.setattr(browser, "_start_playwright", start_playwright)

    async def scenario() -> None:
        context = await browser.launch_persistent_profile(
            tmp_path / "profile", headless=True
        )
        await browser.close_playwright_resource(context)

    asyncio.run(scenario())

    assert chromium.persistent_calls[0][1]["headless"] is True
    assert playwright.stopped


def _write_fake_installer(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake-python"
    executable.write_text(
        f"#!{sys.executable}\n" + body,
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


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


def test_install_chromium_uses_running_python_and_owned_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_path = tmp_path / "install-record.json"
    executable = _write_fake_installer(
        tmp_path,
        """
import json
import os
import sys
from pathlib import Path
Path(os.environ["INSTALL_RECORD"]).write_text(json.dumps({
    "arguments": sys.argv[1:],
    "browser_path": os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
    "pid": os.getpid(),
    "process_group": os.getpgrp(),
}), encoding="utf-8")
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("INSTALL_RECORD", str(record_path))
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "browsers"))
    asyncio.run(browser.install_chromium())

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["arguments"] == ["-m", "playwright", "install", "chromium"]
    assert record["browser_path"] == str(tmp_path / "browsers")
    assert record["process_group"] == record["pid"]
    assert record["process_group"] != os.getpgrp()


def test_install_chromium_failure_has_bounded_actionable_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = _write_fake_installer(
        tmp_path,
        """
import sys
print("discard-me-" * 1000)
print("network unavailable while downloading Chromium")
raise SystemExit(7)
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    with pytest.raises(browser.BrowserInstallError) as raised:
        asyncio.run(browser.install_chromium())

    message = str(raised.value)
    assert "playwright install chromium" in message
    assert "status 7" in message
    assert "network unavailable while downloading Chromium" in message
    assert len(message) < 5_000


def test_install_chromium_reports_exit_before_descendant_closes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_python = sys.executable
    executable = _write_fake_installer(
        tmp_path,
        """
import os
import subprocess
from pathlib import Path
child = subprocess.Popen([
    os.environ["CHILD_PYTHON"],
    "-c",
    "import time; time.sleep(60)",
])
Path(os.environ["CHILD_PID_PATH"]).write_text(str(child.pid), encoding="utf-8")
print("network unavailable while downloading Chromium", flush=True)
raise SystemExit(7)
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("CHILD_PYTHON", child_python)
    monkeypatch.setenv("CHILD_PID_PATH", str(child_pid_path))
    monkeypatch.setattr(browser, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(
        browser, "_OUTPUT_DRAIN_TIMEOUT_SECONDS", 0.1, raising=False
    )
    child_pid: int | None = None
    started_at = time.monotonic()

    try:
        with pytest.raises(browser.BrowserInstallError) as raised:
            asyncio.run(browser.install_chromium(timeout_seconds=3.0))
        elapsed_seconds = time.monotonic() - started_at
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        _assert_process_gone(child_pid)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            _kill_process_if_alive(child_pid)

    message = str(raised.value)
    assert "status 7" in message
    assert "timed out" not in message
    assert "network unavailable while downloading Chromium" in message
    assert elapsed_seconds < 1.5


def test_install_chromium_failure_terminates_reparented_descendant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_python = sys.executable
    executable = _write_fake_installer(
        tmp_path,
        """
import os
import subprocess
from pathlib import Path
child = subprocess.Popen([
    os.environ["CHILD_PYTHON"],
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
])
Path(os.environ["CHILD_PID_PATH"]).write_text(str(child.pid), encoding="utf-8")
print("network unavailable while downloading Chromium", flush=True)
raise SystemExit(7)
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("CHILD_PYTHON", child_python)
    monkeypatch.setenv("CHILD_PID_PATH", str(child_pid_path))
    monkeypatch.setattr(browser, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(
        browser, "_OUTPUT_DRAIN_TIMEOUT_SECONDS", 0.1, raising=False
    )
    child_pid: int | None = None

    try:
        with pytest.raises(browser.BrowserInstallError) as raised:
            asyncio.run(browser.install_chromium())
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        _assert_process_gone(child_pid)
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            _kill_process_if_alive(child_pid)

    message = str(raised.value)
    assert "status 7" in message
    assert "network unavailable while downloading Chromium" in message


@pytest.mark.parametrize("cleanup_origin", ["terminal", "timeout"])
def test_install_chromium_cancellation_waits_for_running_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cleanup_origin: str,
) -> None:
    record_path = tmp_path / "processes.json"
    child_ready_path = tmp_path / "child-ready.txt"
    child_python = sys.executable
    parent_outcome = (
        "raise SystemExit(7)"
        if cleanup_origin == "terminal"
        else "time.sleep(60)"
    )
    executable = _write_fake_installer(
        tmp_path,
        f"""
import json
import os
import subprocess
import time
from pathlib import Path
child = subprocess.Popen([
    os.environ["CHILD_PYTHON"],
    "-c",
    "import os, signal, time; from pathlib import Path; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "Path(os.environ['CHILD_READY']).write_text('ready', encoding='utf-8'); "
    "time.sleep(60)",
])
while not Path(os.environ["CHILD_READY"]).exists():
    time.sleep(0.01)
Path(os.environ["INSTALL_RECORD"]).write_text(json.dumps({{
    "installer": os.getpid(),
    "descendant": child.pid,
}}), encoding="utf-8")
print("installer reached terminal state", flush=True)
{parent_outcome}
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("INSTALL_RECORD", str(record_path))
    monkeypatch.setenv("CHILD_READY", str(child_ready_path))
    monkeypatch.setenv("CHILD_PYTHON", child_python)
    monkeypatch.setattr(browser, "_PROCESS_TERMINATION_GRACE_SECONDS", 1.0)
    cleanup_started = asyncio.Event()
    terminate_process_group = browser._terminate_process_group
    processes: dict[str, int] | None = None

    async def observe_cleanup(
        process_group_id: int, process: asyncio.subprocess.Process
    ) -> None:
        cleanup_started.set()
        await terminate_process_group(process_group_id, process)

    monkeypatch.setattr(browser, "_terminate_process_group", observe_cleanup)

    async def scenario() -> dict[str, int]:
        timeout_seconds = 1.0 if cleanup_origin == "timeout" else 30.0
        task = asyncio.create_task(
            browser.install_chromium(timeout_seconds=timeout_seconds)
        )
        await asyncio.wait_for(cleanup_started.wait(), 2.0)
        recorded = json.loads(record_path.read_text(encoding="utf-8"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return recorded

    try:
        processes = asyncio.run(scenario())
        _assert_process_gone(processes["installer"])
        _assert_process_gone(processes["descendant"])
    finally:
        if processes is None and record_path.exists():
            processes = json.loads(record_path.read_text(encoding="utf-8"))
        if processes is not None:
            _kill_process_if_alive(processes["installer"])
            _kill_process_if_alive(processes["descendant"])


@pytest.mark.parametrize("outcome", ["cancel", "timeout"])
def test_install_chromium_terminates_installer_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    record_path = tmp_path / "processes.json"
    child_python = sys.executable
    executable = _write_fake_installer(
        tmp_path,
        """
import json
import os
import subprocess
import time
from pathlib import Path
child = subprocess.Popen([
    os.environ["CHILD_PYTHON"],
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
])
Path(os.environ["INSTALL_RECORD"]).write_text(json.dumps({
    "installer": os.getpid(),
    "installer_group": os.getpgrp(),
    "descendant": child.pid,
    "descendant_group": os.getpgid(child.pid),
}), encoding="utf-8")
print("installing Chromium", flush=True)
time.sleep(60)
""",
    )
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setenv("INSTALL_RECORD", str(record_path))
    monkeypatch.setenv("CHILD_PYTHON", child_python)
    monkeypatch.setattr(browser, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.1)
    processes: dict[str, int] | None = None

    async def scenario() -> dict[str, int]:
        timeout_seconds = 1.0 if outcome == "timeout" else 30.0
        task = asyncio.create_task(
            browser.install_chromium(timeout_seconds=timeout_seconds)
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while not record_path.exists():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("fake installer did not start")
            await asyncio.sleep(0.01)
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(browser.BrowserInstallError) as raised:
                await task
            assert "timed out after 1s" in str(raised.value)
        return json.loads(record_path.read_text(encoding="utf-8"))

    try:
        processes = asyncio.run(scenario())
        assert processes["installer_group"] == processes["installer"]
        assert processes["descendant_group"] == processes["installer_group"]
        _assert_process_gone(processes["installer"])
        _assert_process_gone(processes["descendant"])
    finally:
        if processes is None and record_path.exists():
            processes = json.loads(record_path.read_text(encoding="utf-8"))
        if processes is not None:
            _kill_process_if_alive(processes["installer"])
            _kill_process_if_alive(processes["descendant"])


def test_browser_installation_call_is_scoped_to_login() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "claudex"
    callers: list[str] = []
    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "install_chromium"
            )
            for node in ast.walk(tree)
        ):
            callers.append(str(source_path.relative_to(source_root)))

    assert callers == ["gptpro/login.py"]
