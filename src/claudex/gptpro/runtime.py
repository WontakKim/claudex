"""Warm persistent-browser lifecycle for ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from claudex import locking, paths
from claudex.gptpro import ask, browser, session
from claudex.gptpro.conversation import (
    CHATGPT_URL,
    TRUSTED_ORIGIN,
    extract_assistant_turn,
)
from claudex.gptpro.selectors import PAGE_FETCH_PROBE_JS

DEFAULT_MAX_CONCURRENT_ASKS = 2
MIN_SUBMISSION_JITTER_SECONDS = 1.0
MAX_SUBMISSION_JITTER_SECONDS = 2.0
DETACH_POLL_INTERVAL_SECONDS = 45.0
DETACH_POLL_MAX_INTERVAL_SECONDS = 300.0

AskOutcome = ask.AskOutcome
GptProAskError = ask.GptProAskError
GptProSessionExpiredError = ask.GptProSessionExpiredError

_monotonic: Callable[[], float] = time.monotonic
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


def _is_transient_detach_fetch_error(exc: BaseException) -> bool:
    """Return whether a fetch can succeed after page navigation settles."""
    message = str(exc).lower()
    return (
        "execution context was destroyed" in message
        or "cannot find context with specified id" in message
    )


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


@dataclass
class _DetachedRegistration:
    conversation_id: str
    marker: str
    deadline: float
    future: asyncio.Future[AskOutcome]


class DetachPoller:
    """Poll detached ChatGPT turns through one resident browser tab."""

    def __init__(self, get_context: Callable[[], Awaitable[Any]]) -> None:
        self._get_context = get_context
        self._registrations: dict[int, _DetachedRegistration] = {}
        self._task: asyncio.Task[None] | None = None
        self._page: Any | None = None
        self._poll_interval_seconds = DETACH_POLL_INTERVAL_SECONDS
        self._is_closing = False

    def register(
        self,
        conversation_id: str,
        marker: str,
        deadline: float,
    ) -> asyncio.Future[AskOutcome]:
        """Register one detached turn and start polling lazily."""
        if self._is_closing:
            raise GptProAskError("error", "the detached answer poller is closed")
        future = asyncio.get_running_loop().create_future()
        self._registrations[id(future)] = _DetachedRegistration(
            conversation_id=conversation_id,
            marker=marker,
            deadline=deadline,
            future=future,
        )
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        return future

    def current_backoff_seconds(self) -> float:
        """Return the admission delay imposed by active polling backoff."""
        if self._poll_interval_seconds <= DETACH_POLL_INTERVAL_SECONDS:
            return 0.0
        return self._poll_interval_seconds

    async def aclose(self) -> None:
        """Stop polling, close the resident tab, and fail pending entries."""
        self._is_closing = True
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._close_page()
        self._fail_all(
            "error", "the detached answer poller closed before completion"
        )

    async def _run(self) -> None:
        try:
            while self._registrations:
                self._sweep_stale()
                if not self._registrations:
                    break
                await self._ensure_page()
                saw_rate_limit = False
                saw_success = False
                for registration_id, registration in tuple(
                    self._registrations.items()
                ):
                    if registration.future.cancelled():
                        self._registrations.pop(registration_id, None)
                        continue
                    try:
                        status = await self._poll_registration(
                            registration_id, registration
                        )
                    except Exception as exc:
                        if _is_transient_detach_fetch_error(exc):
                            break
                        raise
                    saw_rate_limit = saw_rate_limit or status == 429
                    saw_success = saw_success or 200 <= status < 300
                self._update_poll_interval(
                    saw_rate_limit=saw_rate_limit,
                    saw_success=saw_success,
                )
                self._sweep_stale()
                await self._pause_until_next_cycle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = GptProAskError(
                "error", "the detached answer poller failed unexpectedly"
            )
            failure.__cause__ = exc
            self._complete_all_with_exception(failure)
        finally:
            await self._close_page()
            if not self._registrations:
                self._poll_interval_seconds = DETACH_POLL_INTERVAL_SECONDS
            if self._task is asyncio.current_task():
                self._task = None
            if (
                self._registrations
                and self._task is None
                and not self._is_closing
            ):
                self._task = asyncio.create_task(self._run())

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        context = await self._get_context()
        page = await context.new_page()
        try:
            await _harden_page_user_agent(page)
            await page.goto(
                CHATGPT_URL,
                wait_until="domcontentloaded",
                timeout=browser.NAVIGATION_TIMEOUT_MS,
            )
        except BaseException:
            try:
                await page.close()
            finally:
                raise
        self._page = page
        return page

    async def _poll_registration(
        self,
        registration_id: int,
        registration: _DetachedRegistration,
    ) -> int:
        session_status, session_body = await self._fetch_json(
            f"{TRUSTED_ORIGIN}/api/auth/session"
        )
        if self._complete_if_stale(registration_id, registration):
            return session_status
        if session_status in (401, 403):
            self._complete_session_expired(registration_id, registration)
            return session_status
        if not 200 <= session_status < 300:
            return session_status
        access_token = (
            session_body.get("accessToken")
            if isinstance(session_body, Mapping)
            else None
        )
        if not isinstance(access_token, str) or not access_token:
            self._registrations.pop(registration_id, None)
            if not registration.future.done():
                registration.future.set_exception(
                    GptProSessionExpiredError(
                        "the ChatGPT session response did not contain an access token"
                    )
                )
            return session_status

        status, conversation = await self._fetch_json(
            f"{TRUSTED_ORIGIN}/backend-api/conversation/"
            f"{registration.conversation_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if self._complete_if_stale(registration_id, registration):
            return status
        if status in (401, 403):
            self._complete_session_expired(registration_id, registration)
            return status
        if not 200 <= status < 300 or conversation is None:
            return status
        turn = extract_assistant_turn(conversation, registration.marker)
        if turn is None or not turn.finished or not turn.text:
            return status
        self._registrations.pop(registration_id, None)
        if not registration.future.done():
            registration.future.set_result(
                AskOutcome(
                    text=turn.text,
                    marker=registration.marker,
                    conversation_id=registration.conversation_id,
                )
            )
        return status

    async def _fetch_json(
        self,
        target_url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, Mapping[str, Any] | None]:
        page = await self._ensure_page()
        result = await page.evaluate(
            PAGE_FETCH_PROBE_JS,
            {
                "url": target_url,
                "origin": TRUSTED_ORIGIN,
                "headers": dict(headers or {}),
                "timeoutMs": ask.API_FETCH_TIMEOUT_MS,
            },
        )
        if not isinstance(result, Mapping):
            return 0, None
        status = result.get("status")
        if not isinstance(status, int):
            return 0, None
        body = result.get("json")
        if isinstance(body, Mapping):
            return status, body
        body_text = result.get("text")
        if not isinstance(body_text, str):
            return status, None
        try:
            parsed = json.loads(body_text)
        except (json.JSONDecodeError, RecursionError):
            return status, None
        return status, parsed if isinstance(parsed, Mapping) else None

    def _complete_session_expired(
        self,
        registration_id: int,
        registration: _DetachedRegistration,
    ) -> None:
        self._registrations.pop(registration_id, None)
        if not registration.future.done():
            registration.future.set_exception(GptProSessionExpiredError())

    def _complete_if_stale(
        self,
        registration_id: int,
        registration: _DetachedRegistration,
    ) -> bool:
        if registration.deadline > _monotonic():
            return False
        self._registrations.pop(registration_id, None)
        if not registration.future.done():
            registration.future.set_exception(
                GptProAskError(
                    "timeout",
                    "the detached ask budget expired while polling for the answer",
                )
            )
        return True

    def _sweep_stale(self) -> None:
        for registration_id, registration in tuple(
            self._registrations.items()
        ):
            if registration.future.cancelled():
                self._registrations.pop(registration_id, None)
                continue
            self._complete_if_stale(registration_id, registration)

    async def _pause_until_next_cycle(self) -> None:
        if not self._registrations:
            return
        nearest_deadline = min(
            registration.deadline
            for registration in self._registrations.values()
        )
        delay = min(
            self._poll_interval_seconds,
            max(0.0, nearest_deadline - _monotonic()),
        )
        await _sleep(delay)

    def _update_poll_interval(
        self, *, saw_rate_limit: bool, saw_success: bool
    ) -> None:
        if saw_rate_limit:
            self._poll_interval_seconds = min(
                DETACH_POLL_MAX_INTERVAL_SECONDS,
                self._poll_interval_seconds * 2,
            )
        elif saw_success:
            self._poll_interval_seconds = DETACH_POLL_INTERVAL_SECONDS

    def _fail_all(self, failure: str, message: str) -> None:
        for registration in tuple(self._registrations.values()):
            if not registration.future.done():
                registration.future.set_exception(GptProAskError(failure, message))
        self._registrations.clear()

    def _complete_all_with_exception(self, failure: BaseException) -> None:
        for registration in tuple(self._registrations.values()):
            if not registration.future.done():
                registration.future.set_exception(failure)
        self._registrations.clear()

    async def _close_page(self) -> None:
        page = self._page
        self._page = None
        if page is None:
            return
        try:
            await page.close()
        except Exception:
            return


class AskRuntime:
    """Own one lazy persistent context and its bounded pool of ask tabs."""

    def __init__(self) -> None:
        self._context: Any | None = None
        self._profile_lock: locking.FileLockHandle | None = None
        self._initialization_lock = asyncio.Lock()
        self._ask_semaphore = asyncio.Semaphore(_max_concurrent_asks())
        self._waiting_submitters = 0
        self._poller = DetachPoller(self._get_context)

    async def ask(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
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
        self._waiting_submitters += 1
        try:
            await self._ask_semaphore.acquire()
        finally:
            self._waiting_submitters -= 1

        has_admission = True

        def release_admission() -> None:
            nonlocal has_admission
            if not has_admission:
                return
            has_admission = False
            self._ask_semaphore.release()

        try:
            submission_delay = random.uniform(
                MIN_SUBMISSION_JITTER_SECONDS,
                MAX_SUBMISSION_JITTER_SECONDS,
            ) + self._poller.current_backoff_seconds()
            await _sleep(submission_delay)
            # A concurrent page may have observed a browser crash while this
            # ask waited for admission, so re-read the shared context here.
            context = await self._get_context()
            timeout_budget = (
                ask.overall_timeout_seconds()
                if timeout_seconds is None
                else timeout_seconds
            )
            detached_deadline = _monotonic() + timeout_budget
            return await self._execute_in_page(
                context,
                question,
                on_status,
                on_conversation_id,
                on_marker,
                conversation_id,
                timeout_seconds,
                attachment_paths,
                detached_deadline,
                release_admission,
            )
        finally:
            release_admission()

    async def aclose(self) -> None:
        """Close the persistent context, stop Playwright, and release its lock."""
        async with self._initialization_lock:
            context = self._context
            profile_lock = self._profile_lock
            poller = self._poller
            self._context = None
            self._profile_lock = None
            self._poller = DetachPoller(self._get_context)
            try:
                try:
                    await poller.aclose()
                finally:
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
        on_marker: Callable[[str], None] | None,
        conversation_id: str | None,
        timeout_seconds: float | None,
        attachment_paths: Sequence[str] | None,
        detached_deadline: float,
        release_admission: Callable[[], None],
    ) -> AskOutcome:
        page: Any | None = None
        primary_failure: BaseException | None = None
        try:
            page = await context.new_page()
            await _harden_page_user_agent(page)
            execution_options: dict[str, Any] = {
                "on_status": on_status,
                "on_conversation_id": on_conversation_id,
                "on_marker": on_marker,
            }
            if conversation_id is not None:
                execution_options["conversation_id"] = conversation_id
            if timeout_seconds is not None:
                execution_options["timeout_seconds"] = timeout_seconds
            if attachment_paths is not None:
                execution_options["attachment_paths"] = attachment_paths

            async def on_detach(
                submission: ask.AskSubmission,
            ) -> AskOutcome:
                nonlocal page
                conversation_id = submission.conversation_id
                if conversation_id is None:
                    raise GptProAskError(
                        "error",
                        "the detached ask did not provide a conversation ID",
                    )
                future = self._poller.register(
                    conversation_id,
                    submission.marker,
                    detached_deadline,
                )
                detached_page = page
                try:
                    if detached_page is not None:
                        await detached_page.close()
                        page = None
                except BaseException:
                    future.cancel()
                    raise
                finally:
                    release_admission()
                return await future

            execution_options["should_detach"] = (
                lambda: self._waiting_submitters > 0
            )
            execution_options["on_detach"] = on_detach
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
            poller = self._poller
            self._context = None
            self._profile_lock = None
            self._poller = DetachPoller(self._get_context)
            try:
                try:
                    await poller.aclose()
                finally:
                    await browser.close_playwright_resource(context)
            except BaseException as cleanup_failure:
                primary_failure.add_note(
                    "closed context cleanup failed with "
                    f"{type(cleanup_failure).__name__}"
                )
            finally:
                if profile_lock is not None:
                    profile_lock.release()
