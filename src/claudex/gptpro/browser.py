"""Playwright browser launch policy for gptpro commands."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext

NAVIGATION_TIMEOUT_MS = 20_000
COMPOSER_TIMEOUT_MS = 45_000

_CHROME_MISSING_PATTERN = re.compile(
    r"Chromium distribution 'chrome' is not found", re.IGNORECASE
)
_PLAYWRIGHT_BROWSER_MISSING_PATTERN = re.compile(
    r"Executable doesn't exist at|playwright install", re.IGNORECASE
)
_DISABLE_AUTOMATION_ARGUMENT = "--disable-blink-features=AutomationControlled"
_ENABLE_AUTOMATION_DEFAULT_ARGUMENT = "--enable-automation"
_PLAYWRIGHT_OWNERS: dict[int, Any] = {}
_PLAYWRIGHT_INSTALL_MESSAGE = (
    "playwright is not installed; run `uv sync --extra gptpro` to enable "
    "gptpro login"
)
PROFILE_IN_USE_MESSAGE = "another gptpro ask is using the browser profile"


class GptProDependencyError(Exception):
    """Raised when the optional Playwright dependency is unavailable."""


def is_chrome_missing_error(exc: BaseException) -> bool:
    """Return whether Playwright reports that its Chrome channel is absent."""
    return _CHROME_MISSING_PATTERN.search(str(exc)) is not None


def is_browser_missing_error(exc: BaseException) -> bool:
    """Return whether no usable system or Playwright browser is installed."""
    return is_chrome_missing_error(exc) or (
        _PLAYWRIGHT_BROWSER_MISSING_PATTERN.search(str(exc)) is not None
    )


def _merge_anti_automation_options(
    args: Sequence[str], ignore_default_args: Sequence[str]
) -> dict[str, list[str]]:
    merged_args = list(args)
    if _DISABLE_AUTOMATION_ARGUMENT not in merged_args:
        merged_args.append(_DISABLE_AUTOMATION_ARGUMENT)

    merged_ignored_args = list(ignore_default_args)
    if _ENABLE_AUTOMATION_DEFAULT_ARGUMENT not in merged_ignored_args:
        merged_ignored_args.append(_ENABLE_AUTOMATION_DEFAULT_ARGUMENT)

    return {
        "args": merged_args,
        "ignore_default_args": merged_ignored_args,
    }


async def _start_playwright() -> Any:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise GptProDependencyError(_PLAYWRIGHT_INSTALL_MESSAGE) from exc

    return await async_playwright().start()


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
    args: Sequence[str] = (),
    ignore_default_args: Sequence[str] = (),
    headless: bool = False,
) -> BrowserContext:
    """Launch a persistent profile, preferring system Chrome."""
    options: dict[str, object] = {
        "headless": headless,
        **_merge_anti_automation_options(args, ignore_default_args),
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
