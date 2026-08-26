"""Tests for the gptpro CLI command family and top-level dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claudex import __main__ as gateway_main
from claudex.cli import gptpro as gptpro_cli
from claudex.gptpro import ask as gptpro_ask
from claudex.gptpro import login as gptpro_login
from claudex.gptpro import runtime as gptpro_runtime
from claudex.gptpro import session as gptpro_session


def _login_result(
    tmp_path: Path,
    *,
    success: bool,
    failure: gptpro_login.FailureClassification | None = None,
    message: str,
) -> gptpro_login.LoginResult:
    return gptpro_login.LoginResult(
        success=success,
        session_path=tmp_path / "session.json",
        profile_prepared=True,
        cookie_detected=success,
        session_saved=success,
        static_validation_passed=success,
        probe_navigation_passed=success,
        composer_visible=success,
        failure=failure,
        message=message,
    )


def test_gptpro_login_reports_progress_and_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run_login(*, on_status: object) -> gptpro_login.LoginResult:
        assert callable(on_status)
        on_status("waiting for sign-in")
        return _login_result(
            tmp_path,
            success=True,
            message="saved and verified the gptpro session",
        )

    monkeypatch.setattr(gptpro_login, "run_login", run_login)

    assert gptpro_cli._gptpro_main(["login"]) == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "waiting for sign-in\nsaved and verified the gptpro session\n"
    )
    assert captured.err == ""


def test_gptpro_login_failure_uses_stderr_and_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def run_login(*, on_status: object) -> gptpro_login.LoginResult:
        del on_status
        return _login_result(
            tmp_path,
            success=False,
            failure="session_rejected",
            message="sign in again because the session was rejected",
        )

    monkeypatch.setattr(gptpro_login, "run_login", run_login)

    assert gptpro_cli._gptpro_main(["login"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "session_rejected" in captured.err
    assert "run claudex-gateway gptpro login" in captured.err


def test_gptpro_syntax_error_returns_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert gptpro_cli._gptpro_main([]) == 2
    assert "usage: claudex-gateway gptpro" in capsys.readouterr().err


def test_gptpro_keyboard_interrupt_returns_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(gptpro_cli, "_gptpro_login", interrupted)

    assert gptpro_cli._gptpro_main(["login"]) == 130


def test_gptpro_status_reports_missing_session_with_next_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert gptpro_cli._gptpro_main(["status"]) == 0
    captured = capsys.readouterr()
    assert "run claudex-gateway gptpro login" in captured.out
    assert captured.err == ""


def test_gptpro_status_reports_invalid_utf8_session_without_failing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".claudex" / "gptpro" / "session.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe")

    assert gptpro_cli._gptpro_main(["status"]) == 0
    captured = capsys.readouterr()
    assert "invalid" in captured.out
    assert "run claudex-gateway gptpro login" in captured.out
    assert captured.err == ""


def test_gptpro_status_never_prints_cookie_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "cookie-value-must-not-appear"
    path = tmp_path / ".claudex" / "gptpro" / "session.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "cookies": [
                    {
                        "name": gptpro_session.AUTH_COOKIE_PREFIX,
                        "value": secret,
                        "expires": 4_000_000_000,
                    }
                ],
                "origins": [],
            }
        ),
        encoding="utf-8",
    )

    assert gptpro_cli._gptpro_main(["status"]) == 0
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err


def test_unexpected_login_error_does_not_print_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "token-value-must-not-appear"

    async def run_login(*, on_status: object) -> gptpro_login.LoginResult:
        del on_status
        raise RuntimeError(secret)

    monkeypatch.setattr(gptpro_login, "run_login", run_login)

    assert gptpro_cli._gptpro_main(["login"]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "RuntimeError" in captured.err


def test_main_dispatches_gptpro_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def gptpro_main(arguments: list[str]) -> int:
        calls.append(arguments)
        return 0

    def fail_load_config() -> object:
        raise AssertionError("configuration must not be loaded for gptpro")

    monkeypatch.setattr(sys, "argv", ["claudex-gateway", "gptpro", "status"])
    monkeypatch.setattr(gptpro_cli, "_gptpro_main", gptpro_main)
    monkeypatch.setattr(gateway_main.daemon, "_load_config", fail_load_config)

    gateway_main.main()

    assert calls == [["status"]]


def test_gptpro_status_works_with_broken_settings_file(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".claudex"
    runtime_dir.mkdir()
    (runtime_dir / "settings.json").write_text("broken-json", encoding="utf-8")
    environment = dict(os.environ)
    environment["HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "claudex", "gptpro", "status"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "run claudex-gateway gptpro login" in result.stdout
    assert result.stderr == ""


def test_gptpro_status_imports_and_runs_without_playwright(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["HOME"] = str(tmp_path)
    script = """
import builtins
real_import = builtins.__import__
def import_without_playwright(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "playwright" or name.startswith("playwright."):
        raise ModuleNotFoundError("No module named 'playwright'", name="playwright")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = import_without_playwright
from claudex.cli.gptpro import _gptpro_main
raise SystemExit(_gptpro_main(["status"]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "run claudex-gateway gptpro login" in result.stdout
    assert result.stderr == ""


class _FakeAskPage:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeAskContext:
    def __init__(self) -> None:
        self.page = _FakeAskPage()
        self.closed = False

    async def new_page(self) -> _FakeAskPage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class _FakeAskLock:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _install_cli_ask_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeAskContext, _FakeAskLock]:
    context = _FakeAskContext()
    profile_lock = _FakeAskLock()

    async def sleep(_interval: float) -> None:
        return None

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakeAskContext:
        assert profile_dir.name == "chrome-profile"
        assert headless is True
        return context

    async def close_playwright_resource(
        resource: _FakeAskContext,
    ) -> None:
        assert resource is context
        await resource.close()

    monkeypatch.setattr(
        gptpro_runtime.session,
        "session_status",
        lambda: {"valid": True, "message": "valid"},
    )
    monkeypatch.setattr(gptpro_runtime, "_sleep", sleep)
    monkeypatch.setattr(
        gptpro_runtime.locking,
        "try_file_lock",
        lambda _path: profile_lock,
    )
    monkeypatch.setattr(
        gptpro_runtime.browser,
        "launch_persistent_profile",
        launch_persistent_profile,
    )
    monkeypatch.setattr(
        gptpro_runtime.browser,
        "close_playwright_resource",
        close_playwright_resource,
    )
    return context, profile_lock


def test_gptpro_ask_prints_status_to_stderr_and_answer_once_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, profile_lock = _install_cli_ask_runtime(monkeypatch)

    async def execute_ask(
        page: _FakeAskPage,
        question: str,
        *,
        on_status: object,
        deadline: float | None = None,
    ) -> str:
        del deadline
        assert page is context.page
        assert question == "explain the result"
        assert callable(on_status)
        on_status("waiting for ChatGPT")
        return "# Final answer"

    monkeypatch.setattr(gptpro_ask, "execute_ask", execute_ask)

    assert gptpro_cli._gptpro_main(["ask", "explain the result"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "# Final answer\n"
    assert captured.err == "waiting for ChatGPT\n"
    assert context.page.closed
    assert context.closed
    assert profile_lock.released


def test_gptpro_ask_domain_failure_uses_stderr_and_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_cli_ask_runtime(monkeypatch)

    async def execute_ask(
        page: _FakeAskPage,
        question: str,
        *,
        on_status: object,
        deadline: float | None = None,
    ) -> str:
        del page, question, on_status, deadline
        raise gptpro_ask.GptProChallengeError(
            "ChatGPT presented a challenge"
        )

    monkeypatch.setattr(gptpro_ask, "execute_ask", execute_ask)

    assert gptpro_cli._gptpro_main(["ask", "question"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "gptpro ask failed [challenge]" in captured.err
    assert "complete the ChatGPT browser challenge" in captured.err


def test_gptpro_ask_session_expired_prints_login_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        gptpro_runtime.session,
        "session_status",
        lambda: {
            "valid": False,
            "message": "the saved gptpro session is expired",
        },
    )

    assert gptpro_cli._gptpro_main(["ask", "question"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "session_expired" in captured.err
    assert "run claudex-gateway gptpro login" in captured.err


def test_gptpro_ask_keyboard_interrupt_closes_runtime_and_returns_130(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context, profile_lock = _install_cli_ask_runtime(monkeypatch)

    async def execute_ask(
        page: _FakeAskPage,
        question: str,
        *,
        on_status: object,
        deadline: float | None = None,
    ) -> str:
        del page, question, on_status, deadline
        raise KeyboardInterrupt

    monkeypatch.setattr(gptpro_ask, "execute_ask", execute_ask)

    assert gptpro_cli._gptpro_main(["ask", "question"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert context.page.closed
    assert context.closed
    assert profile_lock.released
