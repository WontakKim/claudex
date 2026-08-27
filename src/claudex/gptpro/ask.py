"""Network-first ChatGPT ask execution over an injected Playwright page."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from claudex.gptpro import attachments
from claudex.gptpro.browser import COMPOSER_TIMEOUT_MS, NAVIGATION_TIMEOUT_MS
from claudex.gptpro.conversation import (
    CHATGPT_URL,
    TRUSTED_ORIGIN,
    build_conversation_url,
    build_nonce_marker,
    extract_assistant_turn,
    extract_conversation_id_from_body,
    extract_conversation_id_from_url,
    is_completion_report_url,
    is_conversation_id,
    is_conversation_stream_url,
    is_trusted_origin_url,
)
from claudex.gptpro.retry import (
    RetryAction,
    classify_backend_response,
    rate_limit_delay_ms,
)
from claudex.gptpro.selectors import (
    ASSISTANT_MESSAGE_SELECTOR,
    CHALLENGE_DOM_PROBE_JS,
    COMPOSER_READBACK_PROBE_JS,
    COMPOSER_SELECTOR,
    DISMISS_MODAL_PROBE_JS,
    MESSAGE_ID_ATTRIBUTE,
    MODAL_BUTTON_TEXTS,
    MODAL_SELECTOR,
    PAGE_FETCH_PROBE_JS,
    RELOCK_USER_ECHO_PROBE_JS,
    SEND_BUTTON_READY_PROBE_JS,
    SEND_BUTTON_SELECTOR,
    STOP_BUTTON_SELECTOR,
    TOP_LEVEL_USER_IDS_PROBE_JS,
    TURN_STATE_PROBE_JS,
    USER_ECHO_PROBE_JS,
    USER_MESSAGE_SELECTOR,
    VISIBLE_MODAL_PROBE_JS,
)

AUTH_PATH_FRAGMENT = "/auth/login"
BACKEND_API_PATH_FRAGMENT = "/backend-api/"

# Browser navigation and composer budgets come from gptpro.browser.
COMPOSER_SLICE_SECONDS = 2.0
CHALLENGE_PROBE_TIMEOUT_SECONDS = 5.0
SEND_READY_TIMEOUT_SECONDS = 30.0
ECHO_PROBE_TIMEOUT_SECONDS = 15.0
ECHO_RENDER_TIMEOUT_SECONDS = 300.0
POLL_INTERVAL_SECONDS = 0.120
STABLE_POLLS_REQUIRED = 3
API_FETCH_TIMEOUT_MS = 15_000
API_TURN_ATTEMPTS = 3
API_TURN_RETRY_SECONDS = 3.0
PAGE_FETCH_TRANSIENT_BACKOFF_SECONDS = (1.0, 2.0)
IDLE_GRACE_SECONDS = 300.0
# Watchdog bound from ask submission.
# Long-running workloads raise it via GPTPRO_OVERALL_TIMEOUT_SECONDS.
OVERALL_TIMEOUT_SECONDS = 900.0
HEARTBEAT_INTERVAL_SECONDS = 30.0
ANCHOR_LOST_TIMEOUT_SECONDS = 60.0
RECOVERY_SETTLE_SECONDS = 15.0
RECOVERY_OBSERVE_SECONDS = 8.0
PRE_SUBMIT_STABILITY_TIMEOUT_SECONDS = 10.0
PRE_SUBMIT_POLL_INTERVAL_SECONDS = 0.2
PRE_SUBMIT_STABLE_TICKS = 3
MODAL_SETTLE_SECONDS = 1.0
RELOCK_POLL_INTERVAL_SECONDS = 0.5
PAGE_FETCH_NAVIGATION_RETRIES = 2

FailureClassification = Literal[
    "session_expired",
    "challenge",
    "rate_limited_timeout",
    "navigation_failed",
    "submit_failed",
    "echo_timeout",
    "no_raw_turn",
    "timeout",
    "error",
]


@dataclass(frozen=True)
class AskOutcome:
    text: str
    marker: str
    conversation_id: str | None


class GptProAskError(Exception):
    """Base error for classified gptpro ask failures."""

    def __init__(self, failure: FailureClassification, message: str) -> None:
        super().__init__(message)
        self.failure = failure


class GptProSessionExpiredError(GptProAskError):
    """The stored ChatGPT session no longer authorizes backend requests."""

    def __init__(
        self, message: str = "the ChatGPT session has expired; sign in again"
    ) -> None:
        super().__init__("session_expired", message)


class GptProChallengeError(GptProAskError):
    """Cloudflare challenged or blocked the browser page."""

    def __init__(self, message: str) -> None:
        super().__init__("challenge", message)


class _DeadlineExpired(Exception):
    pass


@dataclass
class _NetworkState:
    conversation_id: str | None = None
    weak_signal_serial: int = 0
    strong_signal_serial: int = 0
    actions: deque[RetryAction] = field(default_factory=deque)
    saw_rate_limit: bool = False


@dataclass(frozen=True)
class _TurnState:
    anchor_present: bool
    assistant_exists: bool
    assistant_text_length: int
    assistant_mutation_key: str
    has_stop: bool


_monotonic: Callable[[], float] = time.monotonic
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def overall_timeout_seconds() -> float:
    raw_value = os.environ.get("GPTPRO_OVERALL_TIMEOUT_SECONDS")
    if raw_value is None:
        return OVERALL_TIMEOUT_SECONDS
    try:
        configured_value = float(raw_value)
    except ValueError:
        return OVERALL_TIMEOUT_SECONDS
    if configured_value <= 0:
        return OVERALL_TIMEOUT_SECONDS
    return configured_value


def _read_member(value: object, name: str, default: Any = None) -> Any:
    member = getattr(value, name, default)
    if callable(member):
        try:
            return member()
        except TypeError:
            return default
    return member


def _page_url(page: object) -> str:
    value = _read_member(page, "url", "")
    return value if isinstance(value, str) else ""


def _is_backend_api_url(url: str) -> bool:
    if not is_trusted_origin_url(url):
        return False
    try:
        return BACKEND_API_PATH_FRAGMENT in urlsplit(url).path
    except ValueError:
        return False


def _is_navigation_destroyed_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "execution context was destroyed" in message
        or "cannot find context with specified id" in message
    )


def _string_headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(header_value)
        for key, header_value in value.items()
        if isinstance(key, str) and isinstance(header_value, str)
    }


class _AskExecution:
    def __init__(
        self,
        page: Any,
        question: str,
        on_status: Callable[[str], None] | None,
        on_conversation_id: Callable[[str], None] | None,
        on_marker: Callable[[str], None] | None,
        *,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> None:
        if conversation_id is not None and not is_conversation_id(conversation_id):
            raise GptProAskError(
                "error",
                "conversation_id must be a canonical ChatGPT conversation UUID",
            )
        self.page = page
        self.on_status = on_status
        self.on_conversation_id = on_conversation_id
        self.on_marker = on_marker
        timeout_budget = (
            overall_timeout_seconds()
            if timeout_seconds is None
            else timeout_seconds
        )
        self.deadline = _monotonic() + timeout_budget
        self.marker = build_nonce_marker(str(uuid4()))
        self._notify_marker(self.marker)
        self.prompt = f"{self.marker}\n\n{question}\n\n{self.marker}"
        self.attachment_paths = tuple(attachment_paths or ())
        self.network = _NetworkState(conversation_id=conversation_id)
        self.listener_tasks: set[asyncio.Task[None]] = set()
        self.request_listener: Callable[[Any], None] | None = None
        self.response_listener: Callable[[Any], None] | None = None
        self.completion_candidate_seen = False
        self.response_wait_started_at: float | None = None
        self.last_heartbeat_at: float | None = None
        self.has_seen_assistant_text = False
        self.handled_weak_signal = 0
        self.handled_strong_signal = 0
        self.active_page_fetch_url: str | None = None
        self.has_submitted = False
        self.has_locked_user_echo = False

    def _status(self, message: str) -> None:
        if self.on_status is None:
            return
        try:
            self.on_status(message)
        except Exception:
            return

    def _notify_marker(self, marker: str) -> None:
        if self.on_marker is None:
            return
        try:
            self.on_marker(marker)
        except Exception:
            return

    def _notify_conversation_id(self, conversation_id: str) -> None:
        if self.on_conversation_id is None:
            return
        try:
            self.on_conversation_id(conversation_id)
        except Exception:
            return

    def _remaining(self) -> float:
        return self.deadline - _monotonic()

    def _ensure_deadline(self) -> None:
        if self._remaining() <= 0:
            raise _DeadlineExpired

    def _timeout_ms(self, maximum_seconds: float) -> int:
        self._ensure_deadline()
        return max(1, math.ceil(min(maximum_seconds, self._remaining()) * 1_000))

    async def _await_page_operation(
        self,
        operation: Awaitable[Any],
        *,
        maximum_seconds: float | None = None,
    ) -> Any:
        remaining = self._remaining()
        if remaining <= 0:
            if inspect.iscoroutine(operation):
                operation.close()
            raise _DeadlineExpired
        timeout = (
            remaining
            if maximum_seconds is None
            else min(remaining, maximum_seconds)
        )
        reaches_deadline = maximum_seconds is None or remaining <= maximum_seconds
        try:
            return await asyncio.wait_for(operation, timeout=timeout)
        except TimeoutError as exc:
            if reaches_deadline:
                raise _DeadlineExpired from exc
            raise

    async def _pause(self, seconds: float) -> None:
        self._ensure_deadline()
        duration = min(max(seconds, 0.0), self._remaining())
        await _sleep(duration)
        if duration < seconds or self._remaining() <= 0:
            raise _DeadlineExpired

    def _maybe_heartbeat(self) -> None:
        if self.has_seen_assistant_text or self.response_wait_started_at is None:
            return
        now = _monotonic()
        last = self.last_heartbeat_at or self.response_wait_started_at
        if now - last < HEARTBEAT_INTERVAL_SECONDS:
            return
        elapsed = round(now - self.response_wait_started_at)
        self._status(f"still waiting for the ChatGPT response ({elapsed}s elapsed)")
        self.last_heartbeat_at = now

    async def _pause_while_waiting(self, seconds: float) -> None:
        end = _monotonic() + seconds
        while _monotonic() < end:
            self._maybe_heartbeat()
            heartbeat_remaining = HEARTBEAT_INTERVAL_SECONDS
            if (
                self.response_wait_started_at is not None
                and not self.has_seen_assistant_text
            ):
                last = self.last_heartbeat_at or self.response_wait_started_at
                heartbeat_remaining = max(
                    0.001, HEARTBEAT_INTERVAL_SECONDS - (_monotonic() - last)
                )
            await self._pause(min(end - _monotonic(), heartbeat_remaining))
        self._maybe_heartbeat()

    def _append_action(self, action: RetryAction) -> None:
        if action.action == "ok":
            return
        if action.action == "rate_limit":
            self.network.saw_rate_limit = True
        self.network.actions.append(action)

    def _capture_request(self, request: Any) -> None:
        try:
            url = _read_member(request, "url", "")
            if not isinstance(url, str) or not is_trusted_origin_url(url):
                return
            conversation_id = extract_conversation_id_from_url(url)
            if conversation_id is None and _is_backend_api_url(url):
                post_data = _read_member(request, "post_data", None)
                if isinstance(post_data, str):
                    conversation_id = extract_conversation_id_from_body(post_data)
            if (
                self.has_submitted
                and self.network.conversation_id is None
                and conversation_id is not None
            ):
                self.network.conversation_id = conversation_id
                self._notify_conversation_id(conversation_id)
            if not self.has_submitted:
                return
            if is_completion_report_url(url):
                self.network.strong_signal_serial += 1
            method = _read_member(request, "method", "")
            if method == "POST" and is_conversation_stream_url(url):
                self.network.weak_signal_serial += 1
        except Exception:
            return

    async def _classify_response_body(
        self,
        response: Any,
        status: int,
        headers: dict[str, str],
    ) -> None:
        body = ""
        try:
            text_result = _read_member(response, "text", "")
            if inspect.isawaitable(text_result):
                text_result = await text_result
            if isinstance(text_result, str):
                body = text_result
        except Exception:
            body = ""
        self._append_action(classify_backend_response(status, headers, body))

    def _capture_response(self, response: Any) -> None:
        try:
            url = _read_member(response, "url", "")
            status = _read_member(response, "status", 0)
            headers = _string_headers(_read_member(response, "headers", {}))
            if not isinstance(url, str) or not isinstance(status, int):
                return
            if url == self.active_page_fetch_url or not is_trusted_origin_url(url):
                return
            if 200 <= status < 400:
                return

            request = _read_member(response, "request", None)
            method = _read_member(request, "method", "")
            is_stream_post = method == "POST" and is_conversation_stream_url(url)
            if not is_stream_post and not is_completion_report_url(url):
                return

            without_body = classify_backend_response(status, headers, "")
            if status != 403 and status < 500:
                self._append_action(without_body)
                return
            if without_body.action in ("challenge", "blocked"):
                self._append_action(without_body)
                return

            task = asyncio.create_task(
                self._classify_response_body(response, status, headers)
            )
            self.listener_tasks.add(task)
            task.add_done_callback(self.listener_tasks.discard)
        except Exception:
            return

    def _install_listeners(self) -> None:
        self.request_listener = self._capture_request
        self.response_listener = self._capture_response
        try:
            self.page.on("requestfinished", self.request_listener)
            self.page.on("response", self.response_listener)
        except Exception as exc:
            raise GptProAskError(
                "error", "the browser page does not support network listeners"
            ) from exc

    async def _remove_listeners(self) -> None:
        for event, listener in (
            ("requestfinished", self.request_listener),
            ("response", self.response_listener),
        ):
            if listener is None:
                continue
            try:
                remove_listener = getattr(self.page, "remove_listener", None)
                if not callable(remove_listener):
                    remove_listener = getattr(self.page, "off", None)
                if not callable(remove_listener):
                    continue
                result = remove_listener(event, listener)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue
        tasks = tuple(self.listener_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _yield_listener_tasks(self) -> None:
        await asyncio.sleep(0)

    async def _process_network_actions(self) -> None:
        await self._yield_listener_tasks()
        while self.network.actions:
            action = self.network.actions.popleft()
            if action.action == "session_expired":
                raise GptProSessionExpiredError(action.reason)
            if action.action in ("challenge", "blocked"):
                disposition = (
                    "blocked" if action.action == "blocked" else "challenged"
                )
                raise GptProChallengeError(
                    f"Cloudflare {disposition} the ChatGPT request: {action.reason}"
                )
            if action.action == "rate_limit":
                delay_seconds = rate_limit_delay_ms(action.retry_after_ms) / 1_000
                self._status(
                    f"ChatGPT rate limited the request; waiting {delay_seconds:.1f}s"
                )
                try:
                    await self._pause_while_waiting(delay_seconds)
                except _DeadlineExpired as exc:
                    raise GptProAskError(
                        "rate_limited_timeout",
                        "the ask deadline expired while waiting for a ChatGPT "
                        "rate limit",
                    ) from exc
                continue
            if action.action == "origin_retry":
                continue
            if action.action in ("entitlement", "fatal"):
                raise GptProAskError("error", action.reason)

    def _check_authenticated(self) -> None:
        if AUTH_PATH_FRAGMENT in _page_url(self.page):
            raise GptProSessionExpiredError()

    async def _navigate(self) -> None:
        target_url = (
            CHATGPT_URL
            if self.network.conversation_id is None
            else build_conversation_url(self.network.conversation_id)
        )
        try:
            await self._await_page_operation(
                self.page.goto(
                    target_url,
                    wait_until="domcontentloaded",
                    timeout=self._timeout_ms(NAVIGATION_TIMEOUT_MS / 1_000),
                )
            )
        except _DeadlineExpired:
            raise
        except Exception as exc:
            if self._remaining() <= 0:
                raise _DeadlineExpired from exc
            raise GptProAskError(
                "navigation_failed", "could not navigate to chatgpt.com"
            ) from exc
        self._check_authenticated()
        await self._process_network_actions()

    async def _probe_challenge(self) -> list[str]:
        try:
            result = await self._await_page_operation(
                self.page.evaluate(CHALLENGE_DOM_PROBE_JS),
                maximum_seconds=CHALLENGE_PROBE_TIMEOUT_SECONDS,
            )
        except _DeadlineExpired:
            raise
        except Exception:
            return []
        if not isinstance(result, list):
            return []
        return [marker for marker in result if isinstance(marker, str) and marker]

    async def _wait_for_composer(self) -> None:
        end = min(self.deadline, _monotonic() + COMPOSER_TIMEOUT_MS / 1_000)
        while _monotonic() < end:
            await self._process_network_actions()
            self._check_authenticated()
            slice_seconds = min(COMPOSER_SLICE_SECONDS, end - _monotonic())
            try:
                await self._await_page_operation(
                    self.page.wait_for_selector(
                        COMPOSER_SELECTOR,
                        state="visible",
                        timeout=self._timeout_ms(slice_seconds),
                    )
                )
                await self._process_network_actions()
                return
            except _DeadlineExpired:
                raise
            except Exception:
                pass
            await self._process_network_actions()
            self._check_authenticated()
            markers = await self._probe_challenge()
            if markers:
                raise GptProChallengeError(
                    "Cloudflare challenge markup was detected: " + ", ".join(markers)
                )
            if _monotonic() < end:
                await self._pause(min(POLL_INTERVAL_SECONDS, end - _monotonic()))
        self._ensure_deadline()
        raise GptProAskError(
            "navigation_failed", "the ChatGPT composer did not become visible"
        )

    async def _attach_files(self) -> None:
        self._ensure_deadline()
        timeout_seconds = min(
            attachments.ATTACH_SETTLE_TIMEOUT_SECONDS, self._remaining()
        )
        try:
            await attachments.attach_files(
                self.page,
                self.attachment_paths,
                timeout_seconds=timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise GptProAskError(
                "error", f"ChatGPT attachment upload failed: {exc}"
            ) from exc

    async def _stable_pre_submit_user_ids(self) -> list[str]:
        end = min(self.deadline, _monotonic() + PRE_SUBMIT_STABILITY_TIMEOUT_SECONDS)
        previous: list[str] | None = None
        stable_ticks = 0
        latest: list[str] = []
        while _monotonic() < end:
            self._ensure_deadline()
            result = await self._await_page_operation(
                self.page.evaluate(
                    TOP_LEVEL_USER_IDS_PROBE_JS,
                    {
                        "userSelector": USER_MESSAGE_SELECTOR,
                        "idAttribute": MESSAGE_ID_ATTRIBUTE,
                    },
                )
            )
            latest = (
                [value for value in result if isinstance(value, str)]
                if isinstance(result, list)
                else []
            )
            if latest == previous:
                stable_ticks += 1
                if stable_ticks >= PRE_SUBMIT_STABLE_TICKS:
                    return latest
            else:
                previous = latest
                stable_ticks = 0
            await self._pause(
                min(PRE_SUBMIT_POLL_INTERVAL_SECONDS, end - _monotonic())
            )
        self._ensure_deadline()
        return latest

    async def _fill_and_verify(self) -> None:
        try:
            await self._await_page_operation(
                self.page.fill(COMPOSER_SELECTOR, self.prompt)
            )
            readback = await self._await_page_operation(
                self.page.evaluate(
                    COMPOSER_READBACK_PROBE_JS,
                    {"selector": COMPOSER_SELECTOR},
                )
            )
        except _DeadlineExpired:
            raise
        except Exception as exc:
            raise GptProAskError(
                "submit_failed", "could not fill the ChatGPT composer"
            ) from exc
        # The composer is a contenteditable editor: newlines round-trip as
        # <br>/paragraph markup, so an exact-string comparison fails on
        # well-formed fills. The nonce marker is the fill's contract — if
        # it made it in, the submit path can anchor on it.
        if not isinstance(readback, str) or self.marker not in readback:
            raise GptProAskError(
                "submit_failed", "the ChatGPT composer did not retain the prompt"
            )

    async def _dismiss_modal(self) -> None:
        try:
            action = await self._await_page_operation(
                self.page.evaluate(
                    DISMISS_MODAL_PROBE_JS,
                    {
                        "modalSelector": MODAL_SELECTOR,
                        "buttonTexts": list(MODAL_BUTTON_TEXTS),
                    },
                )
            )
            if action == "none":
                return
            if action == "escape":
                await self._await_page_operation(
                    self.page.keyboard.press("Escape")
                )
            if action not in ("clicked", "escape"):
                return
            await self._pause(MODAL_SETTLE_SECONDS)
            is_visible = await self._await_page_operation(
                self.page.evaluate(
                    VISIBLE_MODAL_PROBE_JS,
                    {"modalSelector": MODAL_SELECTOR},
                )
            )
        except _DeadlineExpired:
            raise
        except Exception as exc:
            raise GptProAskError(
                "submit_failed", "could not dismiss the send modal"
            ) from exc
        if is_visible:
            raise GptProAskError(
                "submit_failed", "a visible modal still blocks the send button"
            )

    async def _wait_for_send_ready(self) -> None:
        end = min(self.deadline, _monotonic() + SEND_READY_TIMEOUT_SECONDS)
        while _monotonic() < end:
            await self._process_network_actions()
            try:
                ready = await self._await_page_operation(
                    self.page.evaluate(
                        SEND_BUTTON_READY_PROBE_JS,
                        {"selector": SEND_BUTTON_SELECTOR},
                    )
                )
            except _DeadlineExpired:
                raise
            except Exception as exc:
                raise GptProAskError(
                    "submit_failed", "could not inspect the ChatGPT send button"
                ) from exc
            if ready is True:
                return
            await self._pause(min(POLL_INTERVAL_SECONDS, end - _monotonic()))
        self._ensure_deadline()
        raise GptProAskError(
            "submit_failed", "the ChatGPT send button did not become ready"
        )

    async def _click_send(self) -> None:
        await self._dismiss_modal()
        await self._wait_for_send_ready()
        await self._process_network_actions()
        try:
            self.has_submitted = True
            await self._await_page_operation(
                self.page.click(
                    SEND_BUTTON_SELECTOR,
                    timeout=self._timeout_ms(SEND_READY_TIMEOUT_SECONDS),
                )
            )
        except _DeadlineExpired:
            raise
        except Exception as exc:
            raise GptProAskError(
                "submit_failed", "could not click the ChatGPT send button"
            ) from exc
        if self.response_wait_started_at is None:
            self.response_wait_started_at = _monotonic()
            self.last_heartbeat_at = self.response_wait_started_at

    async def _probe_echo_once(self, pre_ids: list[str]) -> str | None:
        result = await self._await_page_operation(
            self.page.evaluate(
                USER_ECHO_PROBE_JS,
                {
                    "userSelector": USER_MESSAGE_SELECTOR,
                    "idAttribute": MESSAGE_ID_ATTRIBUTE,
                    "nonceMarker": self.marker,
                    "preIds": pre_ids,
                },
            )
        )
        return result if isinstance(result, str) and result else None

    async def _wait_for_echo(
        self, pre_ids: list[str], timeout_seconds: float
    ) -> str | None:
        end = min(self.deadline, _monotonic() + timeout_seconds)
        while _monotonic() < end:
            await self._process_network_actions()
            self._maybe_heartbeat()
            try:
                echo_id = await self._probe_echo_once(pre_ids)
            except _DeadlineExpired:
                raise
            except Exception as exc:
                if _is_navigation_destroyed_error(exc):
                    await self._wait_for_load_state()
                    continue
                raise GptProAskError(
                    "echo_timeout", "could not inspect the submitted user turn"
                ) from exc
            if echo_id is not None:
                return echo_id
            await self._pause(min(POLL_INTERVAL_SECONDS, end - _monotonic()))
        self._ensure_deadline()
        return None

    async def _lock_user_echo(self, pre_ids: list[str]) -> str:
        echo_id = await self._wait_for_echo(pre_ids, ECHO_PROBE_TIMEOUT_SECONDS)
        if echo_id is not None:
            return echo_id

        try:
            composer_value = await self._await_page_operation(
                self.page.evaluate(
                    COMPOSER_READBACK_PROBE_JS,
                    {"selector": COMPOSER_SELECTOR},
                )
            )
        except _DeadlineExpired:
            raise
        except Exception:
            composer_value = None
        if composer_value == self.prompt and not self.network.saw_rate_limit:
            self._status(
                "the first send click was not accepted; retrying the click once"
            )
            await self._click_send()

        echo_id = await self._wait_for_echo(pre_ids, ECHO_RENDER_TIMEOUT_SECONDS)
        if echo_id is None:
            if self.network.saw_rate_limit:
                raise GptProAskError(
                    "rate_limited_timeout",
                    "the submitted user turn did not appear after ChatGPT rate "
                    "limited the request",
                )
            raise GptProAskError(
                "echo_timeout", "the submitted user turn did not appear in ChatGPT"
            )
        return echo_id

    async def _wait_for_load_state(self) -> None:
        try:
            await self._await_page_operation(
                self.page.wait_for_load_state(
                    "domcontentloaded",
                    timeout=self._timeout_ms(NAVIGATION_TIMEOUT_MS / 1_000),
                )
            )
        except _DeadlineExpired:
            raise
        except Exception:
            pass
        self._check_authenticated()
        await self._process_network_actions()

    async def _relock_user_echo(self, timeout_seconds: float) -> str | None:
        end = min(self.deadline, _monotonic() + timeout_seconds)
        while _monotonic() < end:
            await self._process_network_actions()
            self._maybe_heartbeat()
            try:
                result = await self._await_page_operation(
                    self.page.evaluate(
                        RELOCK_USER_ECHO_PROBE_JS,
                        {
                            "userSelector": USER_MESSAGE_SELECTOR,
                            "idAttribute": MESSAGE_ID_ATTRIBUTE,
                            "nonceMarker": self.marker,
                        },
                    )
                )
            except Exception as exc:
                if not _is_navigation_destroyed_error(exc):
                    raise
                await self._wait_for_load_state()
                continue
            if isinstance(result, str) and result:
                return result
            await self._pause(min(RELOCK_POLL_INTERVAL_SECONDS, end - _monotonic()))
        self._ensure_deadline()
        return None

    async def _recover_navigation(self, locked_user_id: str) -> str:
        await self._wait_for_load_state()
        relocked_id = await self._relock_user_echo(RECOVERY_SETTLE_SECONDS)
        return relocked_id or locked_user_id

    async def _turn_state(self, locked_user_id: str) -> _TurnState:
        result = await self._await_page_operation(
            self.page.evaluate(
                TURN_STATE_PROBE_JS,
                {
                    "assistantSelector": ASSISTANT_MESSAGE_SELECTOR,
                    "userSelector": USER_MESSAGE_SELECTOR,
                    "stopSelector": STOP_BUTTON_SELECTOR,
                    "idAttribute": MESSAGE_ID_ATTRIBUTE,
                    "lockedUserId": locked_user_id,
                },
            )
        )
        if not isinstance(result, Mapping):
            raise GptProAskError("error", "ChatGPT returned an invalid DOM turn state")
        return _TurnState(
            anchor_present=result.get("anchorPresent") is not False,
            assistant_exists=result.get("assistantExists") is True,
            assistant_text_length=(
                result.get("assistantTextLength")
                if isinstance(result.get("assistantTextLength"), int)
                else 0
            ),
            assistant_mutation_key=(
                result.get("assistantMutationKey")
                if isinstance(result.get("assistantMutationKey"), str)
                else ""
            ),
            has_stop=result.get("hasStop") is True,
        )

    async def _page_fetch_json(
        self,
        target_url: str,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, Any] | None:
        navigation_retries = 0
        transient_retries = 0
        while True:
            current_url = _page_url(self.page)
            if not is_trusted_origin_url(current_url):
                self._check_authenticated()
                raise GptProAskError(
                    "error",
                    "refusing a backend fetch from an untrusted page origin",
                )
            if not is_trusted_origin_url(target_url):
                raise GptProAskError(
                    "error", "refusing a backend fetch to an untrusted origin"
                )
            try:
                self.active_page_fetch_url = target_url
                result = await self._await_page_operation(
                    self.page.evaluate(
                        PAGE_FETCH_PROBE_JS,
                        {
                            "url": target_url,
                            "origin": TRUSTED_ORIGIN,
                            "headers": dict(headers or {}),
                            "timeoutMs": min(
                                API_FETCH_TIMEOUT_MS,
                                self._timeout_ms(API_FETCH_TIMEOUT_MS / 1_000),
                            ),
                        },
                    )
                )
            except _DeadlineExpired:
                raise
            except Exception as exc:
                if (
                    not _is_navigation_destroyed_error(exc)
                    or navigation_retries >= PAGE_FETCH_NAVIGATION_RETRIES
                ):
                    return None
                navigation_retries += 1
                await self._wait_for_load_state()
                continue
            finally:
                self.active_page_fetch_url = None

            await self._process_network_actions()
            if not isinstance(result, Mapping):
                return None
            status = result.get("status")
            fetch_error = result.get("fetchError")
            is_transient_failure = (
                status == 0
                and isinstance(fetch_error, str)
                and bool(fetch_error)
                and result.get("timedOut") is not True
                and not fetch_error.startswith("Untrusted page origin")
            )
            if (
                is_transient_failure
                and transient_retries
                < len(PAGE_FETCH_TRANSIENT_BACKOFF_SECONDS)
            ):
                delay_seconds = PAGE_FETCH_TRANSIENT_BACKOFF_SECONDS[
                    transient_retries
                ]
                transient_retries += 1
                self._status(
                    "transient ChatGPT backend fetch failure; retrying in "
                    f"{delay_seconds:.0f}s"
                )
                await self._pause_while_waiting(delay_seconds)
                continue
            if not isinstance(status, int) or not 200 <= status < 300:
                if isinstance(status, int) and status > 0:
                    action = classify_backend_response(
                        status,
                        _string_headers(result.get("headers")),
                        (
                            result.get("text")
                            if isinstance(result.get("text"), str)
                            else ""
                        ),
                    )
                    if action.action in (
                        "session_expired",
                        "challenge",
                        "blocked",
                        "rate_limit",
                        "entitlement",
                        "fatal",
                    ):
                        self._append_action(action)
                        await self._process_network_actions()
                return None
            body = result.get("json")
            if isinstance(body, Mapping):
                return body
            body_text = result.get("text")
            if not isinstance(body_text, str):
                return None
            try:
                parsed = json.loads(body_text)
            except (json.JSONDecodeError, RecursionError):
                return None
            return parsed if isinstance(parsed, Mapping) else None

    async def _page_fetch_conversation(self) -> Mapping[str, Any] | None:
        conversation_id = self.network.conversation_id
        if conversation_id is None:
            return None
        session = await self._page_fetch_json(f"{TRUSTED_ORIGIN}/api/auth/session")
        if session is None:
            return None
        access_token = session.get("accessToken")
        if not isinstance(access_token, str) or not access_token:
            raise GptProSessionExpiredError(
                "the ChatGPT session response did not contain an access token"
            )
        return await self._page_fetch_json(
            f"{TRUSTED_ORIGIN}/backend-api/conversation/{conversation_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )

    async def _fetch_finished_turn(self) -> str | None:
        if self.network.conversation_id is None:
            return None
        for attempt in range(API_TURN_ATTEMPTS):
            conversation = await self._page_fetch_conversation()
            if conversation is not None:
                turn = extract_assistant_turn(conversation, self.marker)
                if turn is not None and turn.finished and turn.text:
                    return turn.text
            if attempt + 1 < API_TURN_ATTEMPTS:
                await self._pause_while_waiting(API_TURN_RETRY_SECONDS)
        return None

    async def _require_raw_turn(self) -> str:
        text = await self._fetch_finished_turn()
        if text is not None:
            return text
        raise GptProAskError(
            "no_raw_turn",
            "completion was observed, but the server raw assistant turn was "
            "unavailable",
        )

    async def _monitor_completion(self, locked_user_id: str) -> str:
        saw_stop = False
        recovery_active = False
        recovery_started_at = 0.0
        last_mutation_key = ""
        stable_mutation_polls = 0
        next_stable_fetch_at = 0.0
        last_activity_at = self.response_wait_started_at or _monotonic()

        while True:
            self._ensure_deadline()
            await self._process_network_actions()
            self._maybe_heartbeat()

            if (
                self.network.strong_signal_serial > self.handled_strong_signal
                and self.network.conversation_id is not None
            ):
                self.handled_strong_signal = self.network.strong_signal_serial
                self.completion_candidate_seen = True
                text = await self._fetch_finished_turn()
                if text is not None:
                    return text

            if (
                self.network.weak_signal_serial > self.handled_weak_signal
                and self.network.conversation_id is not None
            ):
                self.handled_weak_signal = self.network.weak_signal_serial
                self.completion_candidate_seen = True
                text = await self._fetch_finished_turn()
                if text is not None:
                    return text

            try:
                turn_state = await self._turn_state(locked_user_id)
            except Exception as exc:
                if not _is_navigation_destroyed_error(exc):
                    raise
                locked_user_id = await self._recover_navigation(locked_user_id)
                saw_stop = False
                recovery_active = True
                recovery_started_at = _monotonic()
                last_mutation_key = ""
                stable_mutation_polls = 0
                continue

            if not turn_state.anchor_present:
                relocked_id = await self._relock_user_echo(
                    ANCHOR_LOST_TIMEOUT_SECONDS
                )
                if relocked_id is None:
                    self.completion_candidate_seen = True
                    return await self._require_raw_turn()
                locked_user_id = relocked_id
                saw_stop = False
                last_mutation_key = ""
                stable_mutation_polls = 0
                continue

            mutation_changed = (
                turn_state.assistant_mutation_key != last_mutation_key
            )
            if turn_state.assistant_text_length > 0:
                if mutation_changed or not self.has_seen_assistant_text:
                    last_activity_at = _monotonic()
                self.has_seen_assistant_text = True
            if mutation_changed:
                last_mutation_key = turn_state.assistant_mutation_key
                stable_mutation_polls = 0
            elif turn_state.assistant_exists and turn_state.assistant_text_length > 0:
                stable_mutation_polls += 1
            else:
                stable_mutation_polls = 0
            if turn_state.has_stop:
                saw_stop = True

            stop_disappeared = (
                turn_state.assistant_exists
                and saw_stop
                and not turn_state.has_stop
                and not recovery_active
            )
            recovery_settled = (
                recovery_active
                and turn_state.assistant_exists
                and turn_state.assistant_text_length > 0
                and not turn_state.has_stop
                and stable_mutation_polls >= STABLE_POLLS_REQUIRED
                and _monotonic() - recovery_started_at >= RECOVERY_OBSERVE_SECONDS
            )
            if stop_disappeared or recovery_settled:
                self.completion_candidate_seen = True
                return await self._require_raw_turn()

            mutation_stable = (
                turn_state.assistant_exists
                and turn_state.assistant_text_length > 0
                and not turn_state.has_stop
                and not recovery_active
                and stable_mutation_polls >= STABLE_POLLS_REQUIRED
                and _monotonic() >= next_stable_fetch_at
            )
            if mutation_stable:
                self.completion_candidate_seen = True
                text = await self._fetch_finished_turn()
                if text is not None:
                    return text
                next_stable_fetch_at = _monotonic() + API_TURN_RETRY_SECONDS
                stable_mutation_polls = 0

            idle_grace_elapsed = (
                turn_state.assistant_exists
                and turn_state.assistant_text_length > 0
                and not turn_state.has_stop
                and not saw_stop
                and self.network.strong_signal_serial == 0
                and _monotonic() - last_activity_at >= IDLE_GRACE_SECONDS
            )
            if idle_grace_elapsed:
                self.completion_candidate_seen = True
                return await self._require_raw_turn()

            await self._pause(POLL_INTERVAL_SECONDS)

    async def run(self) -> AskOutcome:
        try:
            self._install_listeners()
            self._ensure_deadline()
            await self._navigate()
            await self._wait_for_composer()
            if self.attachment_paths:
                await self._attach_files()
            pre_submit_ids = await self._stable_pre_submit_user_ids()
            await self._fill_and_verify()
            await self._click_send()
            locked_user_id = await self._lock_user_echo(pre_submit_ids)
            self.has_locked_user_echo = True
            text = await self._monitor_completion(locked_user_id)
            return AskOutcome(
                text=text,
                marker=self.marker,
                conversation_id=self.network.conversation_id,
            )
        except _DeadlineExpired as exc:
            if self.network.saw_rate_limit and not self.has_locked_user_echo:
                raise GptProAskError(
                    "rate_limited_timeout",
                    "the ask deadline expired before a rate-limited user turn "
                    "appeared in ChatGPT",
                ) from exc
            if self.completion_candidate_seen:
                raise GptProAskError(
                    "no_raw_turn",
                    "the ask deadline expired after completion without a server "
                    "raw turn",
                ) from exc
            raise GptProAskError(
                "timeout", "the overall ChatGPT ask deadline expired"
            ) from exc
        except (GptProAskError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise GptProAskError(
                "error", "the ChatGPT ask failed unexpectedly"
            ) from exc
        finally:
            await self._remove_listeners()


async def execute_ask_outcome(
    page: Any,
    question: str,
    *,
    on_status: Callable[[str], None] | None = None,
    on_conversation_id: Callable[[str], None] | None = None,
    on_marker: Callable[[str], None] | None = None,
    conversation_id: str | None = None,
    timeout_seconds: float | None = None,
    attachment_paths: Sequence[str] | None = None,
) -> AskOutcome:
    """Submit one prompt and return its answer with tracking metadata.

    The caller owns ``page`` and remains responsible for closing it. Network
    listeners installed by this function are always removed before it returns.
    """
    return await _AskExecution(
        page,
        question,
        on_status,
        on_conversation_id,
        on_marker,
        conversation_id=conversation_id,
        timeout_seconds=timeout_seconds,
        attachment_paths=attachment_paths,
    ).run()


async def execute_ask(
    page: Any,
    question: str,
    *,
    on_status: Callable[[str], None] | None = None,
) -> str:
    """Submit one prompt and return its server raw markdown."""
    outcome = await execute_ask_outcome(page, question, on_status=on_status)
    return outcome.text
