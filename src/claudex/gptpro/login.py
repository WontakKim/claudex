"""Interactive login orchestration and post-save validation for gptpro."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claudex import locking, paths
from claudex.gptpro import browser, session
from claudex.gptpro.selectors import COMPOSER_SELECTOR
from claudex.providers.auth_support import ensure_private_directory

CHATGPT_URL = "https://chatgpt.com/"
LOGIN_TIMEOUT_SECONDS = 5 * 60
COOKIE_POLL_INTERVAL_SECONDS = 0.5
PROFILE_LOCK_WAIT_SECONDS = 10.0
PROFILE_LOCK_POLL_INTERVAL_SECONDS = 0.1
PROFILE_IN_USE_MESSAGE = browser.PROFILE_IN_USE_MESSAGE

FailureClassification = Literal[
    "dependency_missing",
    "chrome_missing",
    "navigation_failed",
    "login_timeout",
    "session_rejected",
    "probe_retry",
    "error",
]


@dataclass(frozen=True)
class LoginResult:
    """Outcome and completed validation stages for one login attempt."""

    success: bool
    session_path: Path
    profile_prepared: bool
    cookie_detected: bool
    session_saved: bool
    static_validation_passed: bool
    probe_navigation_passed: bool
    composer_visible: bool
    failure: FailureClassification | None
    message: str


class GptProLoginError(Exception):
    """Base error for classified gptpro login failures."""

    def __init__(
        self,
        failure: FailureClassification,
        message: str,
        *,
        probe_navigation_passed: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.probe_navigation_passed = probe_navigation_passed


def _ignore_status(message: str) -> None:
    del message


async def _acquire_profile_lock() -> locking.FileLockHandle | None:
    deadline = time.monotonic() + PROFILE_LOCK_WAIT_SECONDS
    while True:
        profile_lock = locking.try_file_lock(paths.gptpro_profile_lock())
        if profile_lock is not None:
            return profile_lock

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(PROFILE_LOCK_POLL_INTERVAL_SECONDS, remaining))


def _browser_failure(exc: BaseException) -> GptProLoginError:
    if isinstance(exc, browser.GptProDependencyError):
        return GptProLoginError("dependency_missing", str(exc))
    if browser.is_browser_missing_error(exc):
        return GptProLoginError(
            "chrome_missing",
            "install Google Chrome or the Playwright Chromium browser and retry",
        )
    return GptProLoginError("error", "retry after checking the browser installation")


async def _probe_saved_session(path: Path) -> tuple[bool, bool]:
    try:
        probe_browser = await browser.launch_headless_probe_chromium()
    except Exception as exc:
        raise _browser_failure(exc) from exc

    probe_context = None
    probe_failure: BaseException | None = None
    try:
        try:
            probe_context = await browser.create_headless_probe_context(
                probe_browser, storage_state=path
            )
        except Exception as exc:
            raise GptProLoginError(
                "probe_retry",
                "retry session verification after checking the network or browser challenge",
            ) from exc

        pages = probe_context.pages
        page = pages[0] if pages else await probe_context.new_page()
        try:
            await page.goto(
                CHATGPT_URL,
                wait_until="domcontentloaded",
                timeout=browser.NAVIGATION_TIMEOUT_MS,
            )
        except Exception as exc:
            raise GptProLoginError(
                "probe_retry",
                "retry session verification after checking the network or browser challenge",
            ) from exc

        if "/auth/login" in page.url:
            raise GptProLoginError(
                "session_rejected",
                "sign in again because ChatGPT rejected the saved session",
                probe_navigation_passed=True,
            )

        try:
            await page.wait_for_selector(
                COMPOSER_SELECTOR,
                state="visible",
                timeout=browser.COMPOSER_TIMEOUT_MS,
            )
        except Exception as exc:
            raise GptProLoginError(
                "probe_retry",
                "retry session verification after checking the network or browser challenge",
                probe_navigation_passed=True,
            ) from exc
        return True, True
    except BaseException as exc:
        probe_failure = exc
        raise
    finally:
        cleanup_failure: BaseException | None = None
        if probe_context is not None:
            try:
                await probe_context.close()
            except BaseException as exc:
                if probe_failure is not None:
                    probe_failure.add_note(
                        f"probe context cleanup failed with {type(exc).__name__}"
                    )
                else:
                    cleanup_failure = exc
        try:
            await browser.close_playwright_resource(probe_browser)
        except BaseException as exc:
            primary_failure = probe_failure or cleanup_failure
            if primary_failure is not None:
                primary_failure.add_note(
                    f"probe browser cleanup failed with {type(exc).__name__}"
                )
            else:
                cleanup_failure = exc
        if probe_failure is None and cleanup_failure is not None:
            raise cleanup_failure


async def run_login(
    *, on_status: Callable[[str], None] = _ignore_status
) -> LoginResult:
    """Open a login browser, save its auth state, and verify the saved session."""
    profile_dir = paths.gptpro_chrome_profile_dir()
    session_path = paths.gptpro_session_file()
    profile_prepared = False
    cookie_detected = False
    session_saved = False
    static_validation_passed = False
    probe_navigation_passed = False
    composer_visible = False

    def result(
        failure: FailureClassification | None, message: str
    ) -> LoginResult:
        return LoginResult(
            success=failure is None,
            session_path=session_path,
            profile_prepared=profile_prepared,
            cookie_detected=cookie_detected,
            session_saved=session_saved,
            static_validation_passed=static_validation_passed,
            probe_navigation_passed=probe_navigation_passed,
            composer_visible=composer_visible,
            failure=failure,
            message=message,
        )

    try:
        ensure_private_directory(profile_dir)
        profile_prepared = True
    except OSError:
        return result(
            "error", "prepare the gptpro browser profile directory and retry"
        )

    profile_lock = await _acquire_profile_lock()
    if profile_lock is None:
        return result("error", PROFILE_IN_USE_MESSAGE)

    try:
        context = await browser.launch_persistent_profile(
            profile_dir, headless=False
        )
    except BaseException as exc:
        profile_lock.release()
        if not isinstance(exc, Exception):
            raise
        failure = _browser_failure(exc)
        return result(failure.failure, str(failure))

    login_failure: GptProLoginError | None = None
    try:
        await context.clear_cookies(name=session.AUTH_COOKIE_NAME_PATTERN)
        pages = context.pages
        page = pages[0] if pages else await context.new_page()
        try:
            await page.goto(
                CHATGPT_URL,
                wait_until="domcontentloaded",
                timeout=browser.NAVIGATION_TIMEOUT_MS,
            )
        except Exception as exc:
            raise GptProLoginError(
                "navigation_failed",
                "retry after checking access to chatgpt.com",
            ) from exc

        on_status(
            "sign in to ChatGPT in the opened browser; waiting up to five minutes"
        )
        deadline = time.monotonic() + LOGIN_TIMEOUT_SECONDS
        while True:
            cookies = await context.cookies()
            if session.find_auth_cookie(cookies) is not None:
                cookie_detected = True
                break
            if time.monotonic() >= deadline:
                raise GptProLoginError(
                    "login_timeout",
                    "run the login command again and complete sign-in within five minutes",
                )
            await asyncio.sleep(COOKIE_POLL_INTERVAL_SECONDS)

        on_status("saving the authenticated ChatGPT session")
        await session.save_storage_state(context, session_path)
        session_saved = True
    except GptProLoginError as exc:
        login_failure = exc
    except Exception:
        login_failure = GptProLoginError(
            "error", "retry after checking the browser and session directory"
        )
    finally:
        try:
            await browser.close_playwright_resource(context)
        except Exception:
            if login_failure is None:
                login_failure = GptProLoginError(
                    "error", "retry after the login browser failed to close cleanly"
                )
        finally:
            profile_lock.release()

    if login_failure is not None:
        return result(login_failure.failure, str(login_failure))

    try:
        expires = session.load_auth_cookie_expiry(session_path)
        if session.is_expired(expires, time.time()):
            raise GptProLoginError(
                "session_rejected",
                "sign in again because the saved ChatGPT session is already expired",
            )
        static_validation_passed = True
        on_status("verifying the saved ChatGPT session")
        probe_navigation_passed, composer_visible = await _probe_saved_session(
            session_path
        )
    except session.GptProSessionError:
        login_failure = GptProLoginError(
            "session_rejected",
            "sign in again because the saved ChatGPT session is invalid",
        )
    except GptProLoginError as exc:
        login_failure = exc
        probe_navigation_passed = exc.probe_navigation_passed
    except Exception:
        login_failure = GptProLoginError(
            "error", "retry after checking the saved session and browser"
        )

    if login_failure is not None:
        return result(login_failure.failure, str(login_failure))

    return result(
        None, f"saved and verified the gptpro session at {session_path}"
    )
