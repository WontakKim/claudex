"""Tests for gptpro login orchestration with fake Playwright objects."""

from __future__ import annotations

import asyncio
import os
import re
import stat
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import browser, login, selectors, session


class _FakePage:
    def __init__(
        self,
        *,
        url: str = "https://chatgpt.com/",
        goto_error: Exception | None = None,
        selector_error: Exception | None = None,
    ) -> None:
        self.url = url
        self.goto_error = goto_error
        self.selector_error = selector_error
        self.goto_calls: list[tuple[str, str, int]] = []
        self.selector_calls: list[tuple[str, str, int]] = []

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_calls.append((url, wait_until, timeout))
        if self.goto_error is not None:
            raise self.goto_error

    async def wait_for_selector(
        self, selector: str, *, state: str, timeout: int
    ) -> object:
        self.selector_calls.append((selector, state, timeout))
        if self.selector_error is not None:
            raise self.selector_error
        return object()


class _FakePersistentContext:
    def __init__(
        self,
        page: _FakePage,
        cookie_batches: list[list[dict[str, object]]],
    ) -> None:
        self.pages = [page]
        self.cookie_batches = list(cookie_batches)
        self.clear_cookie_names: list[re.Pattern[str]] = []
        self.closed = False

    async def new_page(self) -> _FakePage:
        return self.pages[0]

    async def clear_cookies(self, *, name: re.Pattern[str]) -> None:
        self.clear_cookie_names.append(name)

    async def cookies(self) -> list[dict[str, object]]:
        if len(self.cookie_batches) > 1:
            return self.cookie_batches.pop(0)
        return self.cookie_batches[0]

    async def storage_state(self) -> dict[str, object]:
        return {
            "cookies": [
                {
                    "name": session.AUTH_COOKIE_PREFIX,
                    "value": "stored-secret",
                    "domain": ".chatgpt.com",
                    "expires": 4_000_000_000,
                }
            ],
            "origins": [],
        }

    async def close(self) -> None:
        self.closed = True


class _FakeProbeContext:
    def __init__(
        self, page: _FakePage, *, close_error: BaseException | None = None
    ) -> None:
        self.pages = [page]
        self.close_error = close_error
        self.closed = False

    async def new_page(self) -> _FakePage:
        return self.pages[0]

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FakeProbeBrowser:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _auth_cookie() -> dict[str, object]:
    return {
        "name": f"{session.AUTH_COOKIE_PREFIX}.0",
        "value": "login-secret",
        "domain": ".chatgpt.com",
    }


def _ignore_status(_message: str) -> None:
    pass


def _install_browser_fakes(
    monkeypatch: pytest.MonkeyPatch,
    persistent_context: _FakePersistentContext,
    probe_context: _FakeProbeContext,
    probe_browser: _FakeProbeBrowser | None = None,
) -> _FakeProbeBrowser:
    probe_browser = probe_browser or _FakeProbeBrowser()

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        assert profile_dir.name == "chrome-profile"
        assert headless is False
        return persistent_context

    async def launch_headless_probe_chromium() -> _FakeProbeBrowser:
        return probe_browser

    async def create_headless_probe_context(
        launched_browser: _FakeProbeBrowser, *, storage_state: Path
    ) -> _FakeProbeContext:
        assert launched_browser is probe_browser
        assert storage_state.name == "session.json"
        return probe_context

    monkeypatch.setattr(
        browser, "launch_persistent_profile", launch_persistent_profile
    )
    monkeypatch.setattr(
        browser, "launch_headless_probe_chromium", launch_headless_probe_chromium
    )
    monkeypatch.setattr(
        browser, "create_headless_probe_context", create_headless_probe_context
    )
    return probe_browser


def test_run_login_polls_clears_stale_cookie_saves_and_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_page = _FakePage()
    persistent_context = _FakePersistentContext(
        persistent_page, [[], [_auth_cookie()]]
    )
    probe_page = _FakePage()
    probe_context = _FakeProbeContext(probe_page)
    probe_browser = _install_browser_fakes(
        monkeypatch, persistent_context, probe_context
    )
    sleep_intervals: list[float] = []

    async def fake_sleep(interval: float) -> None:
        sleep_intervals.append(interval)

    monkeypatch.setattr(login.asyncio, "sleep", fake_sleep)
    statuses: list[str] = []

    result = asyncio.run(login.run_login(on_status=statuses.append))

    assert result.success
    assert result.failure is None
    assert sleep_intervals == [0.5]
    assert len(persistent_context.clear_cookie_names) == 1
    assert (
        persistent_context.clear_cookie_names[0].pattern
        == r"^__Secure-next-auth\.session-token"
    )
    assert persistent_page.goto_calls == [
        (login.CHATGPT_URL, "domcontentloaded", browser.NAVIGATION_TIMEOUT_MS)
    ]
    assert probe_page.selector_calls == [
        (selectors.COMPOSER_SELECTOR, "visible", browser.COMPOSER_TIMEOUT_MS)
    ]
    assert statuses == [
        "sign in to ChatGPT in the opened browser; waiting up to five minutes",
        "saving the authenticated ChatGPT session",
        "verifying the saved ChatGPT session",
    ]
    assert persistent_context.closed
    assert probe_context.closed
    assert probe_browser.closed
    session_path = tmp_path / ".claudex" / "gptpro" / "session.json"
    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(
        (tmp_path / ".claudex" / "gptpro" / "chrome-profile").stat().st_mode
    ) == 0o700


def test_run_login_times_out_and_closes_the_persistent_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(_FakePage(), [[]])

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        del profile_dir
        return persistent_context

    async def fail_if_probed() -> Any:
        raise AssertionError("the probe must not run after a login timeout")

    monkeypatch.setattr(
        browser, "launch_persistent_profile", launch_persistent_profile
    )
    monkeypatch.setattr(browser, "launch_headless_probe_chromium", fail_if_probed)
    monkeypatch.setattr(login, "LOGIN_TIMEOUT_SECONDS", 0)

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert not result.success
    assert result.failure == "login_timeout"
    assert persistent_context.closed


def test_run_login_classifies_initial_navigation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(
        _FakePage(goto_error=RuntimeError("network token-secret")),
        [[_auth_cookie()]],
    )

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        del profile_dir
        return persistent_context

    monkeypatch.setattr(
        browser, "launch_persistent_profile", launch_persistent_profile
    )

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == "navigation_failed"
    assert "token-secret" not in result.message
    assert persistent_context.closed


def test_run_login_classifies_probe_login_redirect_as_session_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(
        _FakePage(), [[_auth_cookie()]]
    )
    probe_context = _FakeProbeContext(
        _FakePage(url="https://chatgpt.com/auth/login?next=%2F")
    )
    probe_browser = _install_browser_fakes(
        monkeypatch, persistent_context, probe_context
    )

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == "session_rejected"
    assert persistent_context.closed
    assert probe_context.closed
    assert probe_browser.closed


def test_run_login_classifies_missing_composer_as_probe_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(
        _FakePage(), [[_auth_cookie()]]
    )
    probe_context = _FakeProbeContext(
        _FakePage(selector_error=TimeoutError("challenge token-secret"))
    )
    _install_browser_fakes(monkeypatch, persistent_context, probe_context)

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == "probe_retry"
    assert "token-secret" not in result.message


def test_run_login_classifies_missing_playwright_browser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        del profile_dir
        raise RuntimeError("Executable doesn't exist at /browser/chrome")

    monkeypatch.setattr(
        browser, "launch_persistent_profile", launch_persistent_profile
    )

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == "chrome_missing"


@pytest.mark.parametrize(
    ("probe_page", "expected_failure", "expected_message"),
    [
        (
            _FakePage(url="https://chatgpt.com/auth/login"),
            "session_rejected",
            "sign in again because ChatGPT rejected the saved session",
        ),
        (
            _FakePage(selector_error=TimeoutError("challenge")),
            "probe_retry",
            "retry session verification after checking the network or browser challenge",
        ),
    ],
)
def test_probe_cleanup_errors_do_not_mask_classified_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    probe_page: _FakePage,
    expected_failure: login.FailureClassification,
    expected_message: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(
        _FakePage(), [[_auth_cookie()]]
    )
    probe_context = _FakeProbeContext(
        probe_page, close_error=RuntimeError("context close failed")
    )
    probe_browser = _FakeProbeBrowser(
        close_error=RuntimeError("browser close failed")
    )
    _install_browser_fakes(
        monkeypatch,
        persistent_context,
        probe_context,
        probe_browser,
    )

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == expected_failure
    assert result.message == expected_message
    assert probe_context.closed
    assert probe_browser.closed


def test_run_login_classifies_missing_playwright_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    message = (
        "playwright is not installed; run `uv sync --extra gptpro` to enable "
        "gptpro login"
    )

    async def launch_persistent_profile(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        del profile_dir
        raise browser.GptProDependencyError(message)

    monkeypatch.setattr(
        browser, "launch_persistent_profile", launch_persistent_profile
    )

    result = asyncio.run(login.run_login(on_status=_ignore_status))

    assert result.failure == "dependency_missing"
    assert result.message == message


def test_run_login_hardens_intermediate_profile_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    persistent_context = _FakePersistentContext(_FakePage(), [[_auth_cookie()]])
    probe_context = _FakeProbeContext(_FakePage())
    _install_browser_fakes(monkeypatch, persistent_context, probe_context)

    # `parents=True` mkdir would leave ~/.claudex at the umask default (0755
    # here); every level of the chain must end up 0700 instead.
    previous_umask = os.umask(0o022)
    try:
        result = asyncio.run(login.run_login(on_status=_ignore_status))
    finally:
        os.umask(previous_umask)

    assert result.success
    runtime_dir = tmp_path / ".claudex"
    for directory in (
        runtime_dir,
        runtime_dir / "gptpro",
        runtime_dir / "gptpro" / "chrome-profile",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_run_login_fails_when_ask_runtime_holds_profile_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    holder = login.locking.try_file_lock(login.paths.gptpro_profile_lock())
    assert holder is not None
    monkeypatch.setattr(login, "PROFILE_LOCK_WAIT_SECONDS", 0)

    async def fail_launch(
        profile_dir: Path, *, headless: bool = False
    ) -> _FakePersistentContext:
        del profile_dir, headless
        raise AssertionError("login must not launch a contended profile")

    monkeypatch.setattr(browser, "launch_persistent_profile", fail_launch)
    try:
        result = asyncio.run(login.run_login(on_status=_ignore_status))
    finally:
        holder.release()

    assert not result.success
    assert result.failure == "error"
    assert result.message == "another gptpro ask is using the browser profile"
