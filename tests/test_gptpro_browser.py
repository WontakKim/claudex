"""Tests for the gptpro Playwright browser launch policy."""

from __future__ import annotations

import asyncio
import builtins
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
        "playwright is not installed; run `uv sync --extra gptpro` to enable "
        "gptpro login"
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
