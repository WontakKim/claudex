"""Warm persistent-browser lifecycle for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

from claudex import locking, paths
from claudex.gptpro import ask, browser, session

DEFAULT_MAX_CONCURRENT_ASKS = 2
MIN_SUBMISSION_JITTER_SECONDS = 1.0
MAX_SUBMISSION_JITTER_SECONDS = 2.0

AskOutcome = ask.AskOutcome
GptProAskError = ask.GptProAskError
GptProSessionExpiredError = ask.GptProSessionExpiredError

_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


async def _harden_page_user_agent(page: Any) -> None:
    """Strip the headless marker from the page's outgoing User-Agent header.

    Cloudflare can challenge `HeadlessChrome/...` user agents. Only the
    outgoing header is normalized; navigator.userAgent remains unchanged.
    """
    try:
        user_agent = await page.evaluate("navigator.userAgent")
    except Exception:
        return
    if not isinstance(user_agent, str) or "Headless" not in user_agent:
        return
    hardened = browser.remove_headless_user_agent_token(user_agent)
    try:
        await page.set_extra_http_headers({"User-Agent": hardened})
    except Exception:
        return


def _max_concurrent_asks() -> int:
    raw_value = os.environ.get("GPTPRO_MAX_CONCURRENT_ASKS")
    if raw_value is None:
        return DEFAULT_MAX_CONCURRENT_ASKS
    try:
        configured_value = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_ASKS
    return max(1, configured_value)


def _is_context_closed_error(exc: BaseException) -> bool:
    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        message = str(current).lower()
        if type(current).__name__ == "TargetClosedError" or any(
            marker in message
            for marker in (
                "target page, context or browser has been closed",
                "target page, context or browser was closed",
                "browser has been closed",
                "context has been closed",
                "page has been closed",
                "target closed",
            )
        ):
            return True

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


class AskRuntime:
    """Own one lazy persistent context and its bounded pool of ask tabs."""

    def __init__(self) -> None:
        self._context: Any | None = None
        self._profile_lock: locking.FileLockHandle | None = None
        self._initialization_lock = asyncio.Lock()
        self._ask_semaphore = asyncio.Semaphore(_max_concurrent_asks())

    async def ask(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AskOutcome:
        """Execute one ask in a fresh tab of the shared persistent context."""
        status = session.session_status()
        if not status.get("valid"):
            message = status.get("message")
            raise GptProSessionExpiredError(
                message
                if isinstance(message, str)
                else "the saved ChatGPT session is missing, expired, or invalid"
            )

        await self._get_context()
        async with self._ask_semaphore:
            await _sleep(
                random.uniform(
                    MIN_SUBMISSION_JITTER_SECONDS,
                    MAX_SUBMISSION_JITTER_SECONDS,
                )
            )
            # A concurrent page may have observed a browser crash while this
            # ask waited for admission, so re-read the shared context here.
            context = await self._get_context()
            return await self._execute_in_page(
                context,
                question,
                on_status,
                on_conversation_id,
                conversation_id,
                timeout_seconds,
            )

    async def aclose(self) -> None:
        """Close the persistent context, stop Playwright, and release its lock."""
        async with self._initialization_lock:
            context = self._context
            profile_lock = self._profile_lock
            self._context = None
            self._profile_lock = None
            try:
                if context is not None:
                    await browser.close_playwright_resource(context)
            finally:
                if profile_lock is not None:
                    profile_lock.release()

    async def _get_context(self) -> Any:
        async with self._initialization_lock:
            if self._context is not None:
                return self._context

            profile_lock = locking.try_file_lock(paths.gptpro_profile_lock())
            if profile_lock is None:
                raise GptProAskError("error", browser.PROFILE_IN_USE_MESSAGE)

            try:
                context = await browser.launch_persistent_profile(
                    paths.gptpro_chrome_profile_dir(), headless=True
                )
            except BaseException:
                profile_lock.release()
                raise

            self._context = context
            self._profile_lock = profile_lock
            return context

    async def _execute_in_page(
        self,
        context: Any,
        question: str,
        on_status: Callable[[str], None] | None,
        on_conversation_id: Callable[[str], None] | None,
        conversation_id: str | None,
        timeout_seconds: float | None,
    ) -> AskOutcome:
        page: Any | None = None
        primary_failure: BaseException | None = None
        try:
            page = await context.new_page()
            await _harden_page_user_agent(page)
            execution_options: dict[str, Any] = {
                "on_status": on_status,
                "on_conversation_id": on_conversation_id,
            }
            if conversation_id is not None:
                execution_options["conversation_id"] = conversation_id
            if timeout_seconds is not None:
                execution_options["timeout_seconds"] = timeout_seconds
            return await ask.execute_ask_outcome(
                page,
                question,
                **execution_options,
            )
        except BaseException as exc:
            primary_failure = exc
            if _is_context_closed_error(exc):
                await self._discard_context(context, exc)
            raise
        finally:
            if page is not None:
                try:
                    await page.close()
                except BaseException as exc:
                    if _is_context_closed_error(exc):
                        await self._discard_context(
                            context, primary_failure or exc
                        )
                    if primary_failure is not None:
                        primary_failure.add_note(
                            f"ask page cleanup failed with {type(exc).__name__}"
                        )
                    else:
                        raise

    async def _discard_context(
        self, context: Any, primary_failure: BaseException
    ) -> None:
        async with self._initialization_lock:
            if self._context is not context:
                return
            profile_lock = self._profile_lock
            self._context = None
            self._profile_lock = None
            try:
                await browser.close_playwright_resource(context)
            except BaseException as cleanup_failure:
                primary_failure.add_note(
                    "closed context cleanup failed with "
                    f"{type(cleanup_failure).__name__}"
                )
            finally:
                if profile_lock is not None:
                    profile_lock.release()
