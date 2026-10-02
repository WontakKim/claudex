"""Tests for the network-first gptpro ask runner with a fake page."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import ask, locators, selectors


@pytest.fixture(autouse=True)
def isolated_locator_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> locators.LocatorBook:
    book = locators.LocatorBook(tmp_path / "locators.json")
    monkeypatch.setattr(locators, "_default_book", book)
    return book


_CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"
_EVIL_CONVERSATION_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
_STREAM_URL = "https://chatgpt.com/backend-api/conversation"
_LAT_URL = "https://chatgpt.com/backend-api/lat/report"


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


class _FakeRequest:
    def __init__(self, url: str, method: str, post_data: str | None = None) -> None:
        self.url = url
        self.method = method
        self.post_data = post_data


class _FakeResponse:
    def __init__(
        self,
        url: str,
        status: int,
        *,
        headers: dict[str, str] | None = None,
        request: _FakeRequest | None = None,
    ) -> None:
        self.url = url
        self.status = status
        self.headers = headers or {}
        self.request = request or _FakeRequest(url, "POST")


class _FakePage:
    def __init__(
        self,
        *,
        signal: str = "weak",
        raw_text: str = "server **raw** markdown",
        api_payloads: list[str | None] | None = None,
        api_statuses: list[int] | None = None,
        turn_states: list[dict[str, object] | BaseException] | None = None,
        initial_response: _FakeResponse | None = None,
        readback_mismatch: bool = False,
        readback_reformatted: bool = False,
        swallow_first_click: bool = False,
        echo_never: bool = False,
        require_listeners_before_goto: bool = False,
        emit_second_lat_after_fetch: int | None = None,
        emit_lat_on_load_state: bool = False,
        access_token: str | None = "access-token",
        transient_session_failures: int = 0,
        transient_conversation_failures: int = 0,
        hang_operation: str | None = None,
    ) -> None:
        self.url = "https://chatgpt.com/"
        self.signal = signal
        self.api_payloads = list(api_payloads or [raw_text])
        self.api_statuses = list(api_statuses or [200])
        self.turn_states = list(
            turn_states
            or [
                {
                    "anchorPresent": True,
                    "assistantExists": False,
                    "assistantTextLength": 0,
                    "assistantMutationKey": "0:0",
                    "hasStop": False,
                }
            ]
        )
        self.initial_response = initial_response
        self.readback_mismatch = readback_mismatch
        self.readback_reformatted = readback_reformatted
        self.swallow_first_click = swallow_first_click
        self.echo_never = echo_never
        self.require_listeners_before_goto = require_listeners_before_goto
        self.emit_second_lat_after_fetch = emit_second_lat_after_fetch
        self.emit_lat_on_load_state = emit_lat_on_load_state
        self.access_token = access_token
        self.transient_session_failures = transient_session_failures
        self.transient_conversation_failures = transient_conversation_failures
        self.hang_operation = hang_operation
        self.has_emitted_second_lat = False
        self.listeners: dict[str, list[Callable[[Any], None]]] = {
            "request": [],
            "requestfinished": [],
            "response": [],
        }
        self.composer_value = ""
        self.filled_prompt = ""
        self.fill_actions: list[tuple[str, bool]] = []
        self.click_actions: list[tuple[str, bool]] = []
        self.click_count = 0
        self.fetch_count = 0
        self.session_fetch_count = 0
        self.fetch_arguments: list[dict[str, object]] = []
        self.load_state_calls = 0
        self.goto_urls: list[str] = []
        self.goto_checked_listeners = False
        self.relock_id = "user-relocked"
        self.call_order: list[str] = []

    def on(self, event: str, listener: Callable[[Any], None]) -> None:
        self.listeners[event].append(listener)

    def off(self, event: str, listener: Callable[[Any], None]) -> None:
        self.listeners[event].remove(listener)

    def _emit(self, event: str, value: Any) -> None:
        for listener in tuple(self.listeners[event]):
            listener(value)

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_urls.append(url)
        assert wait_until == "domcontentloaded"
        assert 0 < timeout <= ask.NAVIGATION_TIMEOUT_MS
        if self.require_listeners_before_goto:
            assert self.listeners["requestfinished"]
            assert self.listeners["response"]
            self.goto_checked_listeners = True
        if self.initial_response is not None:
            self._emit("response", self.initial_response)

    async def wait_for_selector(
        self, selector: str, *, state: str, timeout: int
    ) -> object:
        assert selector == selectors.COMPOSER_SELECTOR
        assert state == "visible"
        assert timeout > 0
        self.call_order.append("composer")
        return object()

    async def fill(self, selector: str, value: str, *, strict: bool = False) -> None:
        self.fill_actions.append((selector, strict))
        assert strict is True
        assert selector == selectors.COMPOSER_SELECTOR
        if self.hang_operation == "fill":
            await asyncio.Event().wait()
        self.call_order.append("fill")
        self.composer_value = value
        self.filled_prompt = value

    async def click(self, selector: str, *, timeout: int, strict: bool = False) -> None:
        self.click_actions.append((selector, strict))
        assert strict is True
        assert selector == selectors.SEND_BUTTON_SELECTOR
        assert timeout > 0
        self.click_count += 1
        if self.hang_operation == "click":
            await asyncio.Event().wait()
        if self.swallow_first_click and self.click_count == 1:
            return
        self.composer_value = ""
        post_data = json.dumps({"conversation_id": _CONVERSATION_ID, "messages": [{"content": {"parts": [self.filled_prompt]}}]})
        if self.signal == "weak":
            self._emit(
                "requestfinished", _FakeRequest(_STREAM_URL, "POST", post_data)
            )
        elif self.signal == "strong":
            self._emit("request", _FakeRequest(_STREAM_URL, "POST", post_data))
            self._emit("requestfinished", _FakeRequest(_LAT_URL, "POST", post_data))
        elif self.signal in ("weak_then_evil", "weak_then_other_trusted"):
            self._emit(
                "requestfinished", _FakeRequest(_STREAM_URL, "POST", post_data)
            )
            origin = (
                "https://evil.example"
                if self.signal == "weak_then_evil"
                else "https://chatgpt.com"
            )
            self._emit(
                "requestfinished",
                _FakeRequest(
                    f"{origin}/backend-api/conversation/"
                    f"{_EVIL_CONVERSATION_ID}",
                    "GET",
                ),
            )
        elif self.signal == "id_only":
            self.url = f"https://chatgpt.com/c/{_CONVERSATION_ID}"
            self._emit(
                "requestfinished",
                _FakeRequest(
                    f"https://chatgpt.com/backend-api/conversation/{_CONVERSATION_ID}",
                    "GET",
                ),
            )
        elif self.signal == "rate_limit":
            self._emit(
                "response",
                _FakeResponse(
                    _STREAM_URL,
                    429,
                    headers={"retry-after": "1"},
                ),
            )
            self._emit(
                "requestfinished", _FakeRequest(_STREAM_URL, "POST", post_data)
            )

    async def wait_for_load_state(
        self, state: str, *, timeout: int
    ) -> None:
        assert state == "domcontentloaded"
        assert timeout > 0
        self.load_state_calls += 1
        if self.emit_lat_on_load_state:
            self.emit_lat_on_load_state = False
            self._emit(
                "requestfinished",
                _FakeRequest(
                    _LAT_URL,
                    "POST",
                    json.dumps({"conversation_id": _CONVERSATION_ID}),
                ),
            )

    def _conversation(self, raw_text: str) -> dict[str, object]:
        marker = self.filled_prompt.splitlines()[0]
        return {
            "current_node": "assistant",
            "mapping": {
                "user": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [marker]},
                    },
                },
                "assistant": {
                    "parent": "user",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": [raw_text],
                        },
                        "status": "finished_successfully",
                        "end_turn": True,
                    },
                },
            },
        }

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        if (
            self.hang_operation == "evaluate"
            and expression == selectors.TOP_LEVEL_USER_IDS_PROBE_JS
        ):
            await asyncio.Event().wait()
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            return {
                "invalidSelector": False, "matched": 1, "eligible": 1,
                "candidates": 1, "lang": "en-US",
            }
        if expression == selectors.CHALLENGE_DOM_PROBE_JS:
            return []
        if expression == selectors.TOP_LEVEL_USER_IDS_PROBE_JS:
            return ["existing-user"]
        if expression == selectors.COMPOSER_READBACK_PROBE_JS:
            if self.readback_mismatch:
                return "changed by the page"
            if self.readback_reformatted:
                return self.composer_value.replace("\n\n", "\n")
            return self.composer_value
        if expression == selectors.DISMISS_MODAL_PROBE_JS:
            return "none"
        if expression == selectors.VISIBLE_MODAL_PROBE_JS:
            return False
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            return True
        if expression == ask.attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
            return {"ready": {"notes.txt": 1}, "processing": {}, "failed": {}, "unknown": {}}
        if expression == selectors.USER_ECHO_PROBE_JS:
            if self.echo_never:
                return None
            if self.swallow_first_click and self.click_count < 2:
                return None
            return "user-current"
        if expression == selectors.RELOCK_USER_ECHO_PROBE_JS:
            return self.relock_id
        if expression == selectors.TURN_STATE_PROBE_JS:
            state = self.turn_states[0]
            if len(self.turn_states) > 1:
                state = self.turn_states.pop(0)
            if isinstance(state, BaseException):
                raise state
            return state
        if expression == selectors.PAGE_FETCH_PROBE_JS:
            assert isinstance(argument, dict)
            self.fetch_arguments.append(argument)
            target_url = argument["url"]
            if target_url == "https://chatgpt.com/api/auth/session":
                self.session_fetch_count += 1
                if self.transient_session_failures > 0:
                    self.transient_session_failures -= 1
                    return {
                        "status": 0,
                        "headers": {},
                        "text": "",
                        "json": None,
                        "fetchError": "TypeError: Failed to fetch",
                        "timedOut": False,
                    }
                session_json = (
                    {"accessToken": self.access_token}
                    if self.access_token is not None
                    else {}
                )
                return {
                    "status": 200,
                    "headers": {},
                    "text": "",
                    "json": session_json,
                    "fetchError": None,
                    "timedOut": False,
                }

            self.fetch_count += 1
            if self.transient_conversation_failures > 0:
                self.transient_conversation_failures -= 1
                return {
                    "status": 0,
                    "headers": {},
                    "text": "",
                    "json": None,
                    "fetchError": "TypeError: Failed to fetch",
                    "timedOut": False,
                }
            status = self.api_statuses[0]
            if len(self.api_statuses) > 1:
                status = self.api_statuses.pop(0)
            if (
                self.emit_second_lat_after_fetch == self.fetch_count
                and not self.has_emitted_second_lat
            ):
                self.has_emitted_second_lat = True
                self._emit(
                    "requestfinished",
                    _FakeRequest(
                        _LAT_URL,
                        "POST",
                        json.dumps({"conversation_id": _CONVERSATION_ID}),
                    ),
                )
            payload = self.api_payloads[0]
            if len(self.api_payloads) > 1:
                payload = self.api_payloads.pop(0)
            return {
                "status": status,
                "headers": {},
                "text": "",
                "json": (
                    self._conversation(payload)
                    if status == 200 and payload is not None
                    else {}
                ),
                "fetchError": None,
                "timedOut": False,
            }
        raise AssertionError("unexpected page probe")


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(ask, "_monotonic", clock.monotonic)
    monkeypatch.setattr(locators._default_book, "monotonic", clock.monotonic)
    monkeypatch.setattr(ask, "_sleep", clock.sleep)
    monkeypatch.setattr(ask, "POLL_INTERVAL_SECONDS", 1.0)
    monkeypatch.setattr(ask, "rate_limit_delay_ms", lambda retry_after_ms: 1_000)
    return clock


def _run(page: _FakePage) -> str:
    return asyncio.run(ask.execute_ask(page, "Review this code"))


def _dom_state(*, has_stop: bool, length: int = 12) -> dict[str, object]:
    return {
        "anchorPresent": True,
        "assistantExists": True,
        "assistantTextLength": length,
        "assistantMutationKey": f"{length}:123",
        "hasStop": has_stop,
    }


def test_heartbeat_reports_generation_after_assistant_text_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    statuses: list[str] = []
    execution = ask._AskExecution(
        _FakePage(signal="none"),
        "Review this code",
        ask.AskCallbacks(on_status=statuses.append),
    )
    execution.response_wait_started_at = clock.monotonic()
    execution.last_heartbeat_at = clock.monotonic()
    execution.has_seen_assistant_text = True
    clock.value = 20.0

    asyncio.run(execution._pause_while_waiting(41.0))

    assert statuses == [
        "ChatGPT is still generating (30s elapsed)",
        "ChatGPT is still generating (60s elapsed)",
    ]


def test_heartbeat_preserves_waiting_message_before_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    statuses: list[str] = []
    execution = ask._AskExecution(
        _FakePage(signal="none"),
        "Review this code",
        ask.AskCallbacks(on_status=statuses.append),
    )
    execution.response_wait_started_at = clock.monotonic()
    execution.last_heartbeat_at = clock.monotonic()
    clock.value = 20.0

    asyncio.run(execution._pause_while_waiting(11.0))

    assert statuses == [
        "still waiting for the ChatGPT response (30s elapsed)"
    ]


def test_completion_candidate_status_is_emitted_once_for_repeated_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    statuses: list[str] = []
    page = _FakePage(
        signal="weak",
        api_payloads=[None, None, None, "raw after repeated signals"],
        emit_second_lat_after_fetch=3,
    )

    result = asyncio.run(
        ask.execute_ask(page, "Review this code", on_status=statuses.append)
    )

    assert result == "raw after repeated signals"
    assert page.has_emitted_second_lat
    assert statuses.count(
        "completion observed; fetching the server answer"
    ) == 1


def test_overall_timeout_defaults_when_environment_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPTPRO_OVERALL_TIMEOUT_SECONDS", raising=False)

    assert ask.overall_timeout_seconds() == ask.OVERALL_TIMEOUT_SECONDS


def test_overall_timeout_uses_valid_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPTPRO_OVERALL_TIMEOUT_SECONDS", "14400")

    assert ask.overall_timeout_seconds() == 14_400.0


@pytest.mark.parametrize("raw_value", ["invalid", "0", "-1"])
def test_overall_timeout_defaults_for_invalid_environment_value(
    monkeypatch: pytest.MonkeyPatch, raw_value: str
) -> None:
    monkeypatch.setenv("GPTPRO_OVERALL_TIMEOUT_SECONDS", raw_value)

    assert ask.overall_timeout_seconds() == ask.OVERALL_TIMEOUT_SECONDS


def test_happy_path_returns_finished_server_raw_markdown_and_removes_listeners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(require_listeners_before_goto=True)

    result = _run(page)

    assert result == "server **raw** markdown"
    assert page.goto_checked_listeners
    assert page.click_count == 1
    assert page.fetch_count == 1
    assert page.listeners == {"request": [], "requestfinished": [], "response": []}
    marker = page.filled_prompt.splitlines()[0]
    assert page.filled_prompt == f"{marker}\n\nReview this code\n\n{marker}"
    assert "echo" not in page.filled_prompt.lower()


def test_execute_ask_outcome_returns_tracking_metadata_and_notifies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "uuid4", lambda: "fixed-nonce")
    page = _FakePage(
        signal="weak_then_other_trusted",
        raw_text="server **raw** markdown",
    )
    captured_conversation_ids: list[str] = []

    def on_conversation_id(conversation_id: str) -> None:
        captured_conversation_ids.append(conversation_id)
        raise RuntimeError("observer failure")

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            callbacks=ask.AskCallbacks(
                on_conversation_id=on_conversation_id
            ),
        )
    )
    legacy_text = _run(_FakePage(raw_text="server **raw** markdown"))
    expected_marker = ask.build_nonce_marker("fixed-nonce")

    assert outcome.text == legacy_text
    assert outcome.marker == expected_marker
    assert page.filled_prompt == (
        f"{expected_marker}\n\nReview this code\n\n{expected_marker}"
    )
    assert outcome.conversation_id == _CONVERSATION_ID
    assert captured_conversation_ids == [_CONVERSATION_ID]


def test_execute_ask_outcome_notifies_marker_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "uuid4", lambda: "fixed-nonce")
    captured_markers: list[str] = []

    def on_marker(marker: str) -> None:
        captured_markers.append(marker)
        raise RuntimeError("observer failure")

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            _FakePage(raw_text="server **raw** markdown"),
            "Review this code",
            callbacks=ask.AskCallbacks(on_marker=on_marker),
        )
    )

    assert captured_markers == [ask.build_nonce_marker("fixed-nonce")]
    assert captured_markers == [outcome.marker]


@pytest.mark.parametrize(
    ("conversation_id", "expected_url"),
    [
        (None, "https://chatgpt.com/"),
        (
            _CONVERSATION_ID,
            f"https://chatgpt.com/c/{_CONVERSATION_ID}",
        ),
    ],
)
def test_navigation_targets_new_or_existing_conversation(
    monkeypatch: pytest.MonkeyPatch,
    conversation_id: str | None,
    expected_url: str,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            conversation_id=conversation_id,
        )
    )

    assert page.goto_urls == [expected_url]


def test_invalid_conversation_id_is_classified_as_error() -> None:
    page = _FakePage()

    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(
            ask.execute_ask_outcome(
                page,
                "Review this code",
                conversation_id=f"WEB:{_CONVERSATION_ID}",
            )
        )

    assert raised.value.failure == "error"
    assert "conversation_id" in str(raised.value)
    assert page.goto_urls == []


def test_provided_conversation_id_is_returned_without_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="weak_then_other_trusted")
    provided_conversation_id = _CONVERSATION_ID.upper()
    captured_conversation_ids: list[str] = []

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            conversation_id=provided_conversation_id,
            callbacks=ask.AskCallbacks(
                on_conversation_id=captured_conversation_ids.append
            ),
        )
    )

    assert outcome.conversation_id == provided_conversation_id
    assert captured_conversation_ids == []


def test_custom_timeout_uses_existing_timeout_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="none")

    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(
            ask.execute_ask_outcome(
                page,
                "Review this code",
                timeout_seconds=0.01,
            )
        )

    assert raised.value.failure == "timeout"


@pytest.mark.parametrize("signal", ["strong", "weak"])
def test_network_completion_signals_return_raw_turn(
    monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal=signal, raw_text=f"raw from {signal}")

    assert _run(page) == f"raw from {signal}"
    assert page.fetch_count == 1


def test_listener_is_installed_before_navigation_and_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(require_listeners_before_goto=True)

    _run(page)

    assert page.goto_checked_listeners
    assert page.click_count == 1


def test_attachment_upload_runs_after_composer_and_before_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    captured: list[tuple[tuple[str, ...], float | None]] = []

    async def attach_files(
        attached_page: _FakePage,
        attachment_paths: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, tuple[int, int]]:
        assert attached_page is page
        page.call_order.append("attach")
        captured.append((attachment_paths, timeout_seconds))
        return {"notes.txt": (0, 1)}

    monkeypatch.setattr(ask.attachments, "attach_files", attach_files)

    asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            attachment_paths=["notes.txt"],
        )
    )

    assert page.call_order.index("composer") < page.call_order.index("attach")
    assert page.call_order.index("attach") < page.call_order.index("fill")
    assert captured == [
        (("notes.txt",), ask.attachments.ATTACH_SETTLE_TIMEOUT_SECONDS)
    ]


def test_renamed_attachment_is_checked_again_before_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)

    class RenamedAttachmentPage(_FakePage):
        attachment_checks = 0

        async def evaluate(self, expression: str, argument: Any = None) -> Any:
            if expression == ask.attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
                self.attachment_checks += 1
                assert argument == ["effect(20260927-120843).ts"]
                return {
                    "ready": {"effect(20260927-120843).ts": 1}
                    if self.attachment_checks == 1 else {},
                    "processing": {}, "failed": {}, "unknown": {},
                }
            return await super().evaluate(expression, argument)

    page = RenamedAttachmentPage()

    async def attach_files(*_args: Any, **_kwargs: Any) -> dict[str, tuple[int, int]]:
        return {"effect(20260927-120843).ts": (0, 1)}

    monkeypatch.setattr(ask.attachments, "attach_files", attach_files)

    with pytest.raises(ask.GptProAskError, match="effect\\(20260927-120843\\).ts") as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "Review this code", attachment_paths=["effect.ts"],
        ))

    assert page.attachment_checks == 2
    assert page.click_count == 0
    assert raised.value.evidence.submission == "not_attempted"
    assert raised.value.evidence.ready_attachments == 0


def test_disappearing_attachment_blocks_enabled_send_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)

    class ReRenderedPage(_FakePage):
        async def evaluate(self, expression: str, argument: Any = None) -> Any:
            if expression == ask.attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
                return {
                    "ready": {} if self.filled_prompt else {"notes.txt": 1},
                    "processing": {}, "failed": {}, "unknown": {},
                }
            return await super().evaluate(expression, argument)

    page = ReRenderedPage()

    async def attach_files(*_args: Any, **_kwargs: Any) -> dict[str, tuple[int, int]]:
        return {"notes.txt": (0, 1)}

    monkeypatch.setattr(ask.attachments, "attach_files", attach_files)

    with pytest.raises(ask.GptProAskError, match="notes.txt") as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "Review this code", attachment_paths=["notes.txt"],
        ))

    assert page.click_count == 0
    assert raised.value.evidence is not None
    assert raised.value.evidence.submission == "not_attempted"
    assert raised.value.evidence.ready_attachments == 0


def test_attachment_disappearing_after_send_ready_still_blocks_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)

    class LateRerenderPage(_FakePage):
        attachment_checks = 0

        async def evaluate(self, expression: str, argument: Any = None) -> Any:
            if expression == ask.attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
                self.attachment_checks += 1
                return {
                    "ready": {"notes.txt": 1} if self.attachment_checks == 1 else {},
                    "processing": {}, "failed": {}, "unknown": {},
                }
            return await super().evaluate(expression, argument)

    page = LateRerenderPage()

    async def attach_files(*_args: Any, **_kwargs: Any) -> dict[str, tuple[int, int]]:
        return {"notes.txt": (0, 1)}

    monkeypatch.setattr(ask.attachments, "attach_files", attach_files)

    with pytest.raises(ask.GptProAskError, match="notes.txt") as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "Review this code", attachment_paths=["notes.txt"],
        ))

    assert page.attachment_checks == 2
    assert page.click_count == 0
    assert raised.value.evidence is not None
    assert raised.value.evidence.submission == "not_attempted"
    assert raised.value.evidence.ready_attachments == 0


@pytest.mark.parametrize("count", ["1", -1, True])
def test_invalid_composer_attachment_count_blocks_send(
    monkeypatch: pytest.MonkeyPatch, count: object,
) -> None:
    _install_clock(monkeypatch)

    class InvalidAttachmentPage(_FakePage):
        async def evaluate(self, expression: str, argument: Any = None) -> Any:
            if expression == ask.attachments.READ_COMPOSER_ATTACHMENT_STATE_JS:
                return {"ready": {"notes.txt": count}}
            return await super().evaluate(expression, argument)

    page = InvalidAttachmentPage()

    async def attach_files(*_args: Any, **_kwargs: Any) -> dict[str, tuple[int, int]]:
        return {"notes.txt": (0, 1)}

    monkeypatch.setattr(ask.attachments, "attach_files", attach_files)

    with pytest.raises(ask.GptProAskError, match="invalid state") as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "Review this code", attachment_paths=["notes.txt"],
        ))

    assert raised.value.failure == "submit_failed"
    assert raised.value.evidence.submission == "not_attempted"
    assert page.click_count == 0


@pytest.mark.parametrize("attachment_paths", [None, []])
def test_empty_attachment_paths_do_not_invoke_attachment_upload(
    monkeypatch: pytest.MonkeyPatch,
    attachment_paths: list[str] | None,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    async def fail_attach(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("attachment upload must not be invoked")

    monkeypatch.setattr(ask.attachments, "attach_files", fail_attach)

    asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            attachment_paths=attachment_paths,
        )
    )


def test_attachment_failure_is_classified_with_attachment_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    async def fail_attach(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("invalid UTF-8")

    monkeypatch.setattr(ask.attachments, "attach_files", fail_attach)

    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(
            ask.execute_ask_outcome(
                page,
                "Review this code",
                attachment_paths=["binary.zip"],
            )
        )

    assert raised.value.failure == "error"
    assert "attachment upload failed" in str(raised.value)
    assert "invalid UTF-8" in str(raised.value)
    assert "fill" not in page.call_order


def test_existing_thread_attachment_error_retains_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    async def fail_attach(*_args: Any, **_kwargs: Any) -> None:
        raise ask.attachments.AttachmentSettleTimeoutError(
            "attachment not ready", completed_file_create_responses=1,
            ready_attachments=0,
        )

    monkeypatch.setattr(ask.attachments, "attach_files", fail_attach)

    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "Review this code", conversation_id=_CONVERSATION_ID,
            attachment_paths=["notes.txt"],
        ))

    assert raised.value.evidence is not None
    assert raised.value.evidence.conversation_id == _CONVERSATION_ID
    assert raised.value.evidence.submission == "not_attempted"
    assert "conversation not established" not in str(raised.value)
    assert page.click_count == 0


def test_missing_echo_with_retained_composer_does_not_click_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A retained composer is not positive proof that the first click did not
    # submit; a second click could create a duplicate turn.
    clock = _install_clock(monkeypatch)
    page = _FakePage(swallow_first_click=True, readback_reformatted=True)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "echo_timeout"
    assert page.click_count == 1
    assert raised.value.evidence.submission == "uncertain"
    assert clock.value >= ask.ECHO_PROBE_TIMEOUT_SECONDS


def test_send_button_never_ready_does_not_attempt_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original_evaluate = page.evaluate

    async def never_ready(expression: str, argument: Any = None) -> Any:
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            return False
        return await original_evaluate(expression, argument)

    page.evaluate = never_ready  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "submit_failed"
    assert raised.value.evidence.submission == "not_attempted"
    assert raised.value.evidence.failure_stage == "submission"
    assert page.click_count == 0


def test_readback_mismatch_is_submit_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(readback_mismatch=True)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "submit_failed"
    assert page.click_count == 0


def test_backend_401_is_session_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        initial_response=_FakeResponse(_STREAM_URL, 401),
    )

    with pytest.raises(ask.GptProSessionExpiredError) as raised:
        _run(page)

    assert raised.value.failure == "session_expired"
    assert page.click_count == 0


@pytest.mark.parametrize("mitigation", ["challenge", "block"])
def test_cf_mitigation_is_typed_challenge(
    monkeypatch: pytest.MonkeyPatch, mitigation: str
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        initial_response=_FakeResponse(
            _STREAM_URL,
            403,
            headers={"cf-mitigated": mitigation},
        )
    )

    with pytest.raises(ask.GptProChallengeError) as raised:
        _run(page)

    assert raised.value.failure == "challenge"


def _challenge_interstitial_page(
    page: _FakePage,
    *,
    marker_probes_remaining: int | None,
) -> None:
    """Route the composer wait through a Cloudflare interstitial.

    ``marker_probes_remaining=None`` keeps the interstitial up forever;
    otherwise the challenge markers vanish once that many probes saw them.
    """
    original_wait_for_selector = page.wait_for_selector
    original_evaluate = page.evaluate
    probes = {"remaining": marker_probes_remaining}

    async def wait_while_challenged(
        selector: str, *, state: str, timeout: int
    ) -> object:
        if probes["remaining"] is None or probes["remaining"] > 0:
            raise RuntimeError("challenge interstitial is showing")
        return await original_wait_for_selector(
            selector, state=state, timeout=timeout
        )

    async def probe_while_challenged(
        expression: str, argument: Any = None
    ) -> Any:
        if expression == selectors.LOCATOR_CHECK_PROBE_JS and (
            probes["remaining"] is None or probes["remaining"] > 0
        ):
            return {"matched": 0, "eligible": 0, "candidates": 0}
        if expression == selectors.CHALLENGE_DOM_PROBE_JS and (
            probes["remaining"] is None or probes["remaining"] > 0
        ):
            if probes["remaining"] is not None:
                probes["remaining"] -= 1
            return ["cf-challenge", "challenge-platform"]
        return await original_evaluate(expression, argument)

    page.wait_for_selector = wait_while_challenged  # type: ignore[method-assign]
    page.evaluate = probe_while_challenged  # type: ignore[method-assign]


def test_transient_challenge_markup_self_resolves_within_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()
    _challenge_interstitial_page(page, marker_probes_remaining=2)

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1
    assert clock.value < ask.CHALLENGE_GRACE_SECONDS


def test_persistent_challenge_markup_fails_after_grace_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()
    _challenge_interstitial_page(page, marker_probes_remaining=None)

    with pytest.raises(ask.GptProChallengeError) as raised:
        _run(page)

    assert raised.value.failure == "challenge"
    assert "Cloudflare challenge markup was detected" in str(raised.value)
    assert "cf-challenge" in str(raised.value)
    assert clock.value >= ask.CHALLENGE_GRACE_SECONDS
    assert page.click_count == 0


def test_challenge_grace_is_bounded_by_ask_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "OVERALL_TIMEOUT_SECONDS", 4.0)
    page = _FakePage()
    _challenge_interstitial_page(page, marker_probes_remaining=None)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "timeout"
    assert clock.value <= 4.0


def test_rate_limit_waits_and_does_not_resubmit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage(signal="rate_limit")

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1
    assert 1.0 in clock.sleeps


def test_rate_limit_wait_that_exceeds_deadline_is_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "OVERALL_TIMEOUT_SECONDS", 0.9)
    page = _FakePage(signal="rate_limit")

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "rate_limited_timeout"
    assert page.click_count == 1


def test_navigation_destroy_relocks_and_uses_recovery_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage(
        signal="id_only",
        raw_text="raw after navigation",
        turn_states=[
            RuntimeError("Execution context was destroyed"),
            _dom_state(has_stop=False),
        ],
    )

    assert _run(page) == "raw after navigation"
    assert page.load_state_calls == 1
    assert page.fetch_count == 1
    assert clock.value >= ask.RECOVERY_OBSERVE_SECONDS


def test_dom_completion_observation_fetches_raw_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="id_only",
        raw_text="raw from DOM trigger",
        turn_states=[_dom_state(has_stop=True), _dom_state(has_stop=False)],
    )

    assert _run(page) == "raw from DOM trigger"
    assert page.fetch_count == 1


def test_dom_completion_without_raw_turn_is_no_raw_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="id_only",
        api_payloads=[None, None, None],
        turn_states=[_dom_state(has_stop=True), _dom_state(has_stop=False)],
    )

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "no_raw_turn"
    assert page.fetch_count == ask.API_TURN_ATTEMPTS


def test_expired_overall_deadline_is_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "OVERALL_TIMEOUT_SECONDS", 0.0)
    page = _FakePage()

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "timeout"
    assert page.click_count == 0


def test_untrusted_request_cannot_overwrite_captured_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="weak_then_evil")

    assert _run(page) == "server **raw** markdown"
    conversation_fetches = [
        arguments
        for arguments in page.fetch_arguments
        if "/backend-api/conversation/" in str(arguments["url"])
    ]
    assert conversation_fetches[-1]["url"].endswith(_CONVERSATION_ID)
    assert _EVIL_CONVERSATION_ID not in str(conversation_fetches[-1]["url"])


def test_backend_204_response_is_not_a_fatal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(initial_response=_FakeResponse(_STREAM_URL, 204))

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1


@pytest.mark.parametrize("status", [404, 418])
def test_fatal_conversation_fetch_status_fails_without_retrying(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(api_statuses=[status])

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "error"
    assert f"HTTP {status}" in str(raised.value)
    assert page.fetch_count == 1


def test_conversation_fetch_uses_session_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    assert _run(page) == "server **raw** markdown"
    conversation_fetch = next(
        arguments
        for arguments in page.fetch_arguments
        if "/backend-api/conversation/" in str(arguments["url"])
    )
    assert conversation_fetch["headers"] == {
        "Authorization": "Bearer access-token"
    }
    assert page.session_fetch_count == 1


def test_no_stop_mutation_stability_retries_until_raw_turn_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="id_only",
        api_payloads=[None, None, None, "raw after stable mutation"],
        turn_states=[_dom_state(has_stop=False)],
    )

    assert _run(page) == "raw after stable mutation"
    assert page.fetch_count == 4


def test_lat_signal_rearms_attempts_consumed_by_weak_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="weak",
        api_payloads=[None, None, None, "raw after lat"],
        emit_second_lat_after_fetch=3,
    )

    assert _run(page) == "raw after lat"
    assert page.fetch_count == 4


def test_top_level_role_predicate_is_shared_by_all_role_probes() -> None:
    predicate = selectors.TOP_LEVEL_ROLE_PREDICATE_JS

    assert predicate in selectors.TOP_LEVEL_USER_IDS_PROBE_JS
    assert predicate in selectors.USER_ECHO_PROBE_JS
    assert predicate in selectors.RELOCK_USER_ECHO_PROBE_JS
    assert predicate in selectors.TURN_STATE_PROBE_JS


def test_trusted_backend_request_cannot_replace_latched_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="weak_then_other_trusted")

    assert _run(page) == "server **raw** markdown"
    conversation_fetch = next(
        arguments
        for arguments in page.fetch_arguments
        if "/backend-api/conversation/" in str(arguments["url"])
    )
    assert conversation_fetch["url"].endswith(_CONVERSATION_ID)
    assert _EVIL_CONVERSATION_ID not in str(conversation_fetch["url"])


def test_unrelated_pre_submit_backend_error_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    unrelated_url = "https://chatgpt.com/backend-api/sidebar"
    page = _FakePage(
        initial_response=_FakeResponse(
            unrelated_url,
            403,
            request=_FakeRequest(unrelated_url, "GET"),
        )
    )

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1


def test_relevant_redirect_response_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(initial_response=_FakeResponse(_STREAM_URL, 302))

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1


@pytest.mark.parametrize("hang_operation", ["evaluate", "fill", "click"])
def test_hanging_page_operation_obeys_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
    hang_operation: str,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "PRE_SUBMIT_POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(ask, "OVERALL_TIMEOUT_SECONDS", 0.01)
    page = _FakePage(hang_operation=hang_operation)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "timeout"
    assert page.listeners == {"request": [], "requestfinished": [], "response": []}


def test_navigation_recovery_preserves_completion_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="id_only",
        raw_text="raw from recovery signal",
        turn_states=[
            RuntimeError("Execution context was destroyed"),
            {
                "anchorPresent": True,
                "assistantExists": False,
                "assistantTextLength": 0,
                "assistantMutationKey": "0:0",
                "hasStop": False,
            },
        ],
        emit_lat_on_load_state=True,
    )

    assert _run(page) == "raw from recovery signal"
    assert page.fetch_count == 1


def test_recovery_completion_waits_for_stable_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    changing_states = [
        {
            **_dom_state(has_stop=False),
            "assistantMutationKey": f"12:{index}",
        }
        for index in range(10)
    ]
    stable_state = {
        **_dom_state(has_stop=False),
        "assistantMutationKey": "12:stable",
    }
    page = _FakePage(
        signal="id_only",
        raw_text="raw after stable recovery",
        turn_states=[
            RuntimeError("Execution context was destroyed"),
            *changing_states,
            stable_state,
        ],
    )

    assert _run(page) == "raw after stable recovery"
    assert clock.value >= (
        ask.RECOVERY_OBSERVE_SECONDS + ask.STABLE_POLLS_REQUIRED
    )


def test_idle_grace_bounds_completion_without_network_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "IDLE_GRACE_SECONDS", 3.0)
    monkeypatch.setattr(ask, "STABLE_POLLS_REQUIRED", 1_000)
    page = _FakePage(
        signal="id_only",
        raw_text="raw after idle grace",
        turn_states=[_dom_state(has_stop=False)],
    )

    assert _run(page) == "raw after idle grace"
    assert page.fetch_count == 1
    assert clock.value >= 3.0


def test_missing_session_access_token_is_session_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(access_token=None)

    with pytest.raises(ask.GptProSessionExpiredError) as raised:
        _run(page)

    assert raised.value.failure == "session_expired"
    assert page.session_fetch_count == 1
    assert page.fetch_count == 0


@pytest.mark.parametrize(
    "failure_field",
    ["transient_session_failures", "transient_conversation_failures"],
)
def test_status_zero_fetch_failure_retries_with_bounded_backoff(
    monkeypatch: pytest.MonkeyPatch,
    failure_field: str,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage(**{failure_field: 2})

    assert _run(page) == "server **raw** markdown"
    assert 1.0 in clock.sleeps
    assert 2.0 in clock.sleeps
    if failure_field == "transient_session_failures":
        assert page.session_fetch_count == 3
    else:
        assert page.fetch_count == 3


def test_rate_limit_without_user_echo_is_rate_limited_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="rate_limit", echo_never=True)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "rate_limited_timeout"
    assert page.click_count == 1


def test_recovery_stop_transition_still_waits_for_stable_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    changing_states = [
        {
            **_dom_state(has_stop=index == 0),
            "assistantMutationKey": f"12:recovery-{index}",
        }
        for index in range(10)
    ]
    stable_state = {
        **_dom_state(has_stop=False),
        "assistantMutationKey": "12:stable-after-stop",
    }
    page = _FakePage(
        signal="id_only",
        raw_text="raw after stable recovery stop transition",
        turn_states=[
            RuntimeError("Execution context was destroyed"),
            *changing_states,
            stable_state,
        ],
    )

    assert _run(page) == "raw after stable recovery stop transition"
    assert clock.value >= (
        ask.RECOVERY_OBSERVE_SECONDS + ask.STABLE_POLLS_REQUIRED
    )


def test_echo_deadline_after_rate_limit_is_rate_limited_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "OVERALL_TIMEOUT_SECONDS", 2.5)
    page = _FakePage(signal="rate_limit", echo_never=True)

    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)

    assert raised.value.failure == "rate_limited_timeout"
    assert page.click_count == 1


def test_readback_with_reformatted_newlines_still_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(readback_reformatted=True)

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 1


def test_contention_detaches_completed_submission_and_returns_poller_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "uuid4", lambda: "fixed-nonce")
    page = _FakePage(signal="id_only")
    statuses: list[str] = []
    submissions: list[ask.AskSubmission] = []
    detached_outcome = ask.AskOutcome(
        text="answer from detached polling",
        marker=ask.build_nonce_marker("fixed-nonce"),
        conversation_id=_CONVERSATION_ID,
    )

    async def on_detach(submission: ask.AskSubmission) -> ask.AskOutcome:
        submissions.append(submission)
        return detached_outcome

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            callbacks=ask.AskCallbacks(on_status=statuses.append),
            should_detach=lambda: True,
            on_detach=on_detach,
        )
    )

    assert outcome is detached_outcome
    assert submissions == [
        ask.AskSubmission(
            marker=ask.build_nonce_marker("fixed-nonce"),
            conversation_id=_CONVERSATION_ID,
        )
    ]
    assert statuses[-1] == "detached; polling for the answer"
    assert page.fetch_count == 0
    assert page.listeners == {"request": [], "requestfinished": [], "response": []}


def test_detach_is_ignored_without_a_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="none")
    submissions: list[ask.AskSubmission] = []

    async def on_detach(submission: ask.AskSubmission) -> ask.AskOutcome:
        submissions.append(submission)
        raise AssertionError("an incomplete submission must not detach")

    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(
            ask.execute_ask_outcome(
                page,
                "Review this code",
                timeout_seconds=1.0,
                should_detach=lambda: True,
                on_detach=on_detach,
            )
        )

    assert raised.value.failure == "timeout"
    assert submissions == []
    assert page.listeners == {"request": [], "requestfinished": [], "response": []}


def test_detach_is_ignored_before_the_user_echo_is_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="none")
    submissions: list[ask.AskSubmission] = []

    async def on_detach(submission: ask.AskSubmission) -> ask.AskOutcome:
        submissions.append(submission)
        raise AssertionError("an unlocked submission must not detach")

    execution = ask._AskExecution(
        page,
        "Review this code",
        None,
        conversation_id=_CONVERSATION_ID,
        should_detach=lambda: True,
        on_detach=on_detach,
    )
    page.filled_prompt = execution.prompt
    execution.network.weak_signal_serial = 1

    completion = asyncio.run(execution._monitor_completion("user-current"))

    assert completion == "server **raw** markdown"
    assert submissions == []


def test_detach_callbacks_are_inert_unless_both_are_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    detach_checks = 0

    def should_detach() -> bool:
        nonlocal detach_checks
        detach_checks += 1
        return True

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            _FakePage(raw_text="normal monitored answer"),
            "Review this code",
            should_detach=should_detach,
        )
    )

    assert outcome.text == "normal monitored answer"
    assert detach_checks == 0


def test_request_detach_sets_the_monitor_detach_flag() -> None:
    page = _FakePage(signal="none")
    statuses: list[str] = []

    async def on_detach(submission: ask.AskSubmission) -> ask.AskOutcome:
        return ask.AskOutcome(
            text="detached answer",
            marker=submission.marker,
            conversation_id=submission.conversation_id,
        )

    execution = ask._AskExecution(
        page,
        "Review this code",
        ask.AskCallbacks(on_status=statuses.append),
        conversation_id=_CONVERSATION_ID,
        should_detach=lambda: False,
        on_detach=on_detach,
    )
    execution.has_locked_user_echo = True
    execution.request_detach()

    completion = asyncio.run(execution._monitor_completion("user-current"))

    assert completion == ask.AskSubmission(
        marker=execution.marker,
        conversation_id=_CONVERSATION_ID,
    )
    assert statuses == ["detached; polling for the answer"]


def test_truncated_prompt_with_nonce_is_not_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original_evaluate = page.evaluate

    async def truncated_readback(expression: str, argument: Any = None) -> Any:
        if expression == selectors.COMPOSER_READBACK_PROBE_JS:
            return page.filled_prompt.splitlines()[0] + "\ntruncated"
        return await original_evaluate(expression, argument)

    page.evaluate = truncated_readback  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == "submit_failed"
    assert page.click_count == 0


def test_correlated_outgoing_request_captures_id_before_requestfinished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(signal="none", echo_never=True)
    observed: list[str] = []
    original_click = page.click

    async def click_then_lose_confirmation(selector: str, *, timeout: int, strict: bool = False) -> None:
        page._emit(
            "request",
            _FakeRequest(
                _STREAM_URL,
                "POST",
                json.dumps({
                    "conversation_id": _CONVERSATION_ID,
                    "messages": [{"content": {"parts": [page.filled_prompt]}}],
                }),
            ),
        )
        await original_click(selector, timeout=timeout, strict=strict)
        raise RuntimeError("navigation destroyed click confirmation")

    page.click = click_then_lose_confirmation  # type: ignore[method-assign]
    page.listeners["request"] = []
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(
            ask.execute_ask_outcome(
                page, "question", callbacks=ask.AskCallbacks(
                    on_conversation_id=observed.append,
                ),
            )
        )
    assert raised.value.failure == "submit_failed"
    assert observed == [_CONVERSATION_ID]
    assert page.click_count == 1
    assert raised.value.evidence.submission == "uncertain"


def test_echo_confirms_submission_after_outgoing_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    observations: list[ask.AskEvidence] = []
    asyncio.run(ask.execute_ask_outcome(
        _FakePage(), "question",
        callbacks=ask.AskCallbacks(on_evidence=observations.append),
    ))
    assert any(evidence.submission == "confirmed" for evidence in observations)


def test_unrelated_outgoing_request_does_not_latch_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    execution = ask._AskExecution(_FakePage(), "target", None)
    execution.has_submitted = True
    execution._capture_request(_FakeRequest(
        _STREAM_URL, "POST", json.dumps({
            "conversation_id": _EVIL_CONVERSATION_ID,
            "messages": [{"content": {"parts": ["unrelated"]}}],
        }),
    ))
    assert execution.network.conversation_id is None


def test_unrelated_backend_fetch_after_correlated_send_does_not_latch_thread() -> None:
    execution = ask._AskExecution(_FakePage(), "target", None)
    execution.has_submitted = True
    execution._capture_request(_FakeRequest(
        _STREAM_URL, "POST", json.dumps({
            "messages": [{"content": {"parts": [execution.marker]}}],
        }),
    ))
    assert execution.has_correlated_request
    execution._capture_request(_FakeRequest(
        f"https://chatgpt.com/backend-api/conversation/{_EVIL_CONVERSATION_ID}",
        "GET",
    ))
    assert execution.network.conversation_id is None


def test_timeout_before_first_click_preserves_not_attempted_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(hang_operation="fill")
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "question", timeout_seconds=0.01,
            callbacks=ask.AskCallbacks(on_status=lambda _: None),
        ))
    assert raised.value.failure == "timeout"
    assert page.click_count == 0
    assert getattr(raised.value, "evidence", None) is not None
    assert raised.value.evidence.submission == "not_attempted"
    assert raised.value.evidence.failure_stage == "composer"


def test_failed_evidence_observer_logs_safely_and_retains_uncertain_submission(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(echo_never=True)
    question = "private question tail"
    cookie = "secret session cookie"
    observed: list[ask.AskEvidence] = []

    def on_evidence(evidence: ask.AskEvidence) -> None:
        observed.append(evidence)
        if evidence.submission == "uncertain":
            raise RuntimeError(f"{question}; {cookie}")

    with caplog.at_level(logging.WARNING, logger=ask.__name__):
        with pytest.raises(ask.GptProAskError) as raised:
            asyncio.run(ask.execute_ask_outcome(
                page, question, timeout_seconds=1.0,
                callbacks=ask.AskCallbacks(on_evidence=on_evidence),
            ))

    assert page.click_count == 1
    assert any(evidence.submission == "uncertain" for evidence in observed)
    assert raised.value.evidence.submission == "uncertain"
    assert raised.value.evidence.failure_stage == "echo"
    assert "evidence observer failed at submission (RuntimeError)" in caplog.messages
    assert question not in caplog.text
    assert cookie not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_late_navigation_error_keeps_nonce_correlated_thread_and_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage(
        signal="weak", api_payloads=[None],
        turn_states=[RuntimeError("DOM changed")],
    )
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == "error"
    assert raised.value.evidence.conversation_id == _CONVERSATION_ID
    assert raised.value.evidence.failure_stage == "answer"
    assert raised.value.evidence.submission == "confirmed"
    assert page.click_count == 1


def test_attachment_receipts_do_not_claim_composer_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()

    async def unsettled(*_args: Any, **_kwargs: Any) -> None:
        raise ask.attachments.AttachmentSettleTimeoutError(
            "one receipt, unready composer",
            completed_file_create_responses=1,
            ready_attachments=0,
        )

    monkeypatch.setattr(ask.attachments, "attach_files", unsettled)
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(ask.execute_ask_outcome(
            page, "question", attachment_paths=["pending.txt"],
        ))
    assert raised.value.failure == "error"
    assert raised.value.evidence.upload_receipts == 1
    assert raised.value.evidence.ready_attachments == 0
    assert raised.value.evidence.submission == "not_attempted"
    assert page.click_count == 0


_GENERATED_MESSAGE_ID = "11111111-2222-4333-8444-555555555555"


class _GeneratedFilePage(_FakePage):
    """Fake page whose final answer links one generated sandbox file."""

    def __init__(self, content: bytes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.file_requests: list[str] = []

    def _conversation(self, raw_text: str) -> dict[str, object]:
        conversation = super()._conversation(raw_text)
        mapping = conversation["mapping"]
        assert isinstance(mapping, dict)
        mapping["assistant"]["message"]["id"] = _GENERATED_MESSAGE_ID
        return conversation

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        url = argument.get("url") if isinstance(argument, dict) else None
        if (
            url == "https://chatgpt.com/api/auth/session"
            and expression != selectors.PAGE_FETCH_PROBE_JS
        ):
            result = await super().evaluate(selectors.PAGE_FETCH_PROBE_JS, argument)
            body = json.dumps(result["json"]).encode()
            return {
                **result, "bodyBase64": base64.b64encode(body).decode(),
                "byteLength": len(body), "tooLarge": False, "redirected": False,
                "url": url,
            }
        if isinstance(url, str) and "/interpreter/download" in url:
            self.file_requests.append(url)
            body = json.dumps({
                "status": "success",
                "download_url": (
                    "https://chatgpt.com/backend-api/estuary/content?id=f&sig=s"
                ),
                "mime_type": "text/markdown",
            }).encode()
            return {
                "status": 200, "headers": {}, "text": body.decode(),
                "json": json.loads(body), "fetchError": None, "timedOut": False,
                "bodyBase64": base64.b64encode(body).decode(),
                "byteLength": len(body), "tooLarge": False, "redirected": False,
                "url": url,
            }
        if isinstance(url, str) and "/backend-api/estuary/content" in url:
            self.file_requests.append(url)
            return {
                "status": 200, "headers": {}, "text": "", "json": None,
                "fetchError": None, "timedOut": False,
                "bodyBase64": base64.b64encode(self.content).decode(),
                "byteLength": len(self.content), "tooLarge": False,
                "redirected": False, "url": url,
            }
        return await super().evaluate(expression, argument)


def test_direct_ask_delivers_generated_files_after_final_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _install_clock(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    answer = "See [notes](sandbox:/mnt/data/notes.md)."
    page = _GeneratedFilePage(b"# notes\n", raw_text=answer)
    statuses: list[str] = []

    outcome = asyncio.run(
        ask.execute_ask_outcome(
            page,
            "Review this code",
            callbacks=ask.AskCallbacks(on_status=statuses.append),
        )
    )

    assert len(page.file_requests) == 2
    assert outcome.text == answer
    files = getattr(outcome, "files", None)
    assert files is not None, "AskOutcome does not expose generated files"
    (saved,) = files
    assert saved.status == "saved"
    assert saved.message_id == _GENERATED_MESSAGE_ID
    with open(saved.path, "rb") as handle:
        assert handle.read() == b"# notes\n"
    assert outcome.files_complete is True
    assert page.listeners == {"request": [], "requestfinished": [], "response": []}
    assert "downloading 1 generated file" in statuses


def test_direct_ask_without_file_links_makes_no_file_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    page = _GeneratedFilePage(b"unused", raw_text="plain answer")

    outcome = asyncio.run(ask.execute_ask_outcome(page, "Review this code"))

    assert page.file_requests == []
    assert outcome == ask.AskOutcome(
        text="plain answer",
        marker=outcome.marker,
        conversation_id=_CONVERSATION_ID,
    )
    assert getattr(outcome, "files", None) == ()


@pytest.mark.parametrize(
    ("candidates", "expected"), [(1, "locator_unresolved"), (0, "navigation_failed")],
)
def test_composer_locator_failure_diagnostic(
    monkeypatch: pytest.MonkeyPatch, candidates: int, expected: str,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    last_error = RuntimeError("composer wait timed out")
    original = page.evaluate

    async def wait(selector: str, *, state: str, timeout: int) -> object:
        raise last_error

    async def evaluate(expression: str, argument: Any = None) -> Any:
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            return {
                "matched": 0, "eligible": 0, "candidates": candidates, "lang": "ko-KR",
            }
        return await original(expression, argument)

    page.wait_for_selector = wait  # type: ignore[method-assign]
    page.evaluate = evaluate  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == expected
    assert raised.value.evidence.failure_stage == "composer"
    assert raised.value.evidence.submission == "not_attempted"
    assert f"page candidates={candidates}, lang=ko-KR" in str(raised.value)
    if candidates:
        assert isinstance(raised.value.__cause__, locators.HealerUnavailable)
    else:
        assert raised.value.__cause__ is last_error


@pytest.mark.parametrize("fault", [True, False])
def test_send_locator_fault_or_disabled_button(
    monkeypatch: pytest.MonkeyPatch, fault: bool,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original = page.evaluate

    async def evaluate(expression: str, argument: Any = None) -> Any:
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            return False
        if (
            expression == selectors.LOCATOR_CHECK_PROBE_JS
            and argument["target"] == "send"
        ):
            return {
                "matched": 0 if fault else 1, "eligible": 0 if fault else 1,
                "candidates": 1, "lang": "ko-KR",
            }
        return await original(expression, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == ("locator_unresolved" if fault else "submit_failed")
    assert "page candidates=1, lang=ko-KR" in str(raised.value)
    assert raised.value.evidence.submission == "not_attempted"
    assert page.click_count == 0


@pytest.mark.parametrize("target", ["composer", "send"])
@pytest.mark.parametrize("result", ["success", "absent", "second_fault", "action_refault", "invalid_css", "click_failure", "echo_timeout"])
def test_locator_recovery_first_use(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    target: locators.LocatorTarget, result: str,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    book = isolated_locator_book
    calls = []
    class Healer:
        async def complete(self, system, prompt, timeout_seconds):
            calls.append(prompt)
            return json.dumps({"status": "absent"} if result == "absent" else {"status": "selected", "candidate": 0})
    book.healer = Healer()
    original = page.evaluate
    recovered = ".recovered"
    async def evaluate(expression, argument=None):
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"lang": "ko", "candidates": [{"index": 0, "selector": recovered}]}
        if expression == selectors.LOCATOR_CHECK_PROBE_JS and argument["target"] == target:
            valid = argument["selector"] == recovered
            if result in ("second_fault", "action_refault") and valid:
                checks = getattr(page, "candidate_checks", 0) + 1
                page.candidate_checks = checks
                valid = checks < (2 if result == "second_fault" else 3)
            return {"matched": int(valid), "eligible": int(valid), "candidates": 1, "lang": "ko",
                    "invalidSelector": result == "invalid_css" and not valid}
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            return {"matched": 1, "eligible": 1, "candidates": 1, "lang": "ko"}
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS and target == "send":
            if result == "invalid_css" and argument["selector"] != recovered:
                raise RuntimeError("SyntaxError: invalid selector")
            return argument["selector"] == recovered and result != "second_fault"
        if expression == selectors.USER_ECHO_PROBE_JS and target == "send":
            assert not book.path.exists()
        if expression == selectors.USER_ECHO_PROBE_JS and result == "echo_timeout":
            return None
        if expression == selectors.COMPOSER_READBACK_PROBE_JS and target == "composer":
            assert argument["selector"] == recovered
            assert not book.path.exists()
        return await original(expression, argument)
    page.evaluate = evaluate
    if target == "composer":
        original_fill = page.fill
        async def fill(selector, value, *, strict=False):
            assert selector == recovered and not book.path.exists()
            await original_fill(selectors.COMPOSER_SELECTOR, value, strict=strict)
        page.fill = fill
    else:
        original_click = page.click
        async def click(selector, *, timeout, strict=False):
            assert selector == recovered and not book.path.exists()
            await original_click(selectors.SEND_BUTTON_SELECTOR, timeout=timeout, strict=strict)
        page.click = click
    if result == "click_failure":
        async def failed_click(selector, *, timeout, strict=False):
            raise RuntimeError("click failed")
        page.click = failed_click
    if result in ("success", "invalid_css"):
        _run(page)
        assert book.selector_for("ko", target) == recovered
        assert book.path.exists()
    else:
        with pytest.raises(ask.GptProAskError) as raised:
            _run(page)
        if result in ("click_failure", "echo_timeout"):
            assert raised.value.failure == ("submit_failed" if result == "click_failure" else "echo_timeout")
            assert raised.value.evidence.submission == "uncertain"
            if target == "send":
                assert not book.pending
                if result == "click_failure":
                    assert book._record("ko", target)["consecutive_failures"] == 1
                    assert book.refusal("ko", target)
                else:
                    assert not book.path.exists()
        else:
            assert raised.value.failure == "locator_unresolved"
            assert raised.value.evidence.failure_stage == ("composer" if target == "composer" else "submission")
            assert raised.value.evidence.submission == "not_attempted"
            assert page.click_count == 0
    assert len(calls) == 1
    if result in ("second_fault", "action_refault"):
        record = json.loads(book.path.read_text())["environments"]["ko"][target]
        assert record["consecutive_failures"] == 1
        assert record["selector"] is None
        assert record["last_failure"] == "the rediscovered locator failed the contract again"
        assert not book.pending


def test_healthy_locators_do_not_write(isolated_locator_book: locators.LocatorBook) -> None:
    class UnexpectedHealer:
        async def complete(self, *args, **kwargs):
            pytest.fail("healthy locators must not call the healer")
    isolated_locator_book.healer = UnexpectedHealer()
    _run(_FakePage())
    assert not isolated_locator_book.path.exists()


def test_composer_probe_exception_clears_previous_fault(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original = page.evaluate
    checks = []
    async def evaluate(expression, argument=None):
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            checks.append(argument)
            if len(checks) == 1:
                return {"matched": 0, "eligible": 0, "candidates": 1}
            raise RuntimeError("probe unavailable")
        return await original(expression, argument)
    page.evaluate = evaluate
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == "navigation_failed"
    assert "the locator check did not complete" in str(raised.value)
    assert str(raised.value.__cause__) == "probe unavailable"


@pytest.mark.parametrize("failure", ["fill", "readback"])
def test_unproven_composer_failure_is_recorded(
    isolated_locator_book: locators.LocatorBook, failure: str,
) -> None:
    page = _FakePage()
    execution = ask._AskExecution(page, "question", None, locator_book=isolated_locator_book)
    execution.locator_environment = "ko"
    attempt = locators.RediscoveredLocator(selectors.COMPOSER_SELECTOR, "test-attempt")
    execution.unproven_locators["composer"] = attempt
    isolated_locator_book.pending[("ko", "composer")] = attempt
    if failure == "fill":
        async def fill(selector, value, *, strict=False):
            raise RuntimeError("cannot fill")
        page.fill = fill
    else:
        original = page.evaluate
        async def evaluate(expression, argument=None):
            if expression == selectors.COMPOSER_READBACK_PROBE_JS:
                return "lost prompt"
            return await original(expression, argument)
        page.evaluate = evaluate
    with pytest.raises(ask.GptProAskError):
        asyncio.run(execution._fill_and_verify())
    assert isolated_locator_book.refusal("ko", "composer")
    assert not execution.unproven_locators


def test_optional_recording_error_does_not_fail_ask(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    caplog: pytest.LogCaptureFixture,
) -> None:
    execution = ask._AskExecution(_FakePage(), "question", None, locator_book=isolated_locator_book)
    execution.unproven_locators["composer"] = locators.RediscoveredLocator(".candidate")
    def fail(*args):
        raise OSError("read-only directory")
    monkeypatch.setattr(isolated_locator_book, "record_success", fail)
    execution._settle_rediscovered("composer", "success")
    assert "read-only directory" in caplog.text


@pytest.mark.parametrize("has_saved_record", [False, True])
def test_unproven_composer_timeout_leaves_book_unchanged(
    isolated_locator_book: locators.LocatorBook, has_saved_record: bool,
) -> None:
    book = isolated_locator_book
    if has_saved_record:
        book.record_success("ko", "composer", ".saved")
    before = book.path.read_bytes() if book.path.exists() else None
    page = _FakePage(hang_operation="fill")
    execution = ask._AskExecution(
        page, "question", None, timeout_seconds=0.01, locator_book=isolated_locator_book,
    )
    execution.locator_environment = "ko"
    execution.locator_selectors["composer"] = selectors.COMPOSER_SELECTOR
    attempt = locators.RediscoveredLocator(selectors.COMPOSER_SELECTOR, "test-attempt")
    execution.unproven_locators["composer"] = attempt
    isolated_locator_book.pending[("ko", "composer")] = attempt
    with pytest.raises(ask._DeadlineExpired):
        asyncio.run(execution._fill_and_verify())
    after = book.path.read_bytes() if book.path.exists() else None
    assert after == before
    assert book.refusal("ko", "composer") is None
    assert "composer" in execution.unproven_locators


@pytest.mark.parametrize("scenario", [
    "stale_composer", "composer_refault", "send_button_type", "send_duplicate",
    "send_other_form", "send_invalid_css", "send_refault", "healthy", "book_changes",
])
def test_actions_require_verified_locators(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    scenario: str,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    book = isolated_locator_book
    stale = ".stale-composer"
    if scenario == "stale_composer":
        book.record_success("en-US", "composer", stale)
    original = page.evaluate
    checked: list[tuple[str, str, bool]] = []
    send_checks = 0
    composer_checks = 0

    async def evaluate(expression: str, argument: Any = None) -> Any:
        nonlocal send_checks, composer_checks
        if expression == selectors.TOP_LEVEL_USER_IDS_PROBE_JS and scenario == "book_changes":
            book.record_success("en-US", "composer", stale)
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS and scenario == "send_invalid_css":
            raise RuntimeError("SyntaxError: invalid selector")
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            target, selector = argument["target"], argument["selector"]
            if target == "composer":
                composer_checks += 1
                valid = selector != stale and not (
                    scenario == "composer_refault" and composer_checks > 1
                )
            else:
                send_checks += 1
                valid = not scenario.startswith("send_") or (
                    scenario == "send_refault" and send_checks == 1
                )
            checked.append((target, selector, valid))
            return {
                "matched": 2 if scenario == "send_duplicate" and target == "send" else 1,
                "eligible": int(valid), "candidates": 1, "lang": "en-US",
                "invalidSelector": scenario == "send_invalid_css" and target == "send",
            }
        return await original(expression, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    if scenario in ("healthy", "stale_composer", "book_changes"):
        _run(page)
        assert page.fill_actions == [(selectors.COMPOSER_SELECTOR, True)]
        assert ("composer", page.fill_actions[0][0], True) in checked
        assert page.click_actions == [(selectors.SEND_BUTTON_SELECTOR, True)]
        assert ("send", page.click_actions[0][0], True) in checked
        if scenario == "stale_composer":
            assert ("composer", stale, False) in checked
    else:
        with pytest.raises(ask.GptProAskError) as raised:
            _run(page)
        assert raised.value.failure == "locator_unresolved"
        assert raised.value.evidence.submission == "not_attempted"
        assert page.click_count == 0
        if scenario == "composer_refault":
            assert page.fill_actions == []
        if scenario not in ("composer_refault", "send_refault"):
            assert isinstance(raised.value.__cause__, locators.HealerUnavailable)


@pytest.mark.parametrize("invalidates_before_click", [False, True])
def test_send_stops_when_filled_composer_becomes_ineligible(
    monkeypatch: pytest.MonkeyPatch, invalidates_before_click: bool,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original = page.evaluate
    is_composer_valid = True
    send_checks = 0

    async def evaluate(expression: str, argument: Any = None) -> Any:
        nonlocal is_composer_valid, send_checks
        if expression == selectors.COMPOSER_READBACK_PROBE_JS:
            result = await original(expression, argument)
            if not invalidates_before_click:
                is_composer_valid = False
            return result
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            if argument["target"] == "send":
                assert argument["composerSelector"] == page.fill_actions[0][0]
                send_checks += 1
                if invalidates_before_click and send_checks > 1:
                    is_composer_valid = False
                return {
                    "matched": 1, "eligible": int(is_composer_valid),
                    "candidates": int(is_composer_valid), "lang": "en-US",
                }
        return await original(expression, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == (
        "locator_unresolved" if invalidates_before_click else "submit_failed"
    )
    assert page.fill_actions == [(selectors.COMPOSER_SELECTOR, True)]
    assert send_checks > 0
    assert raised.value.evidence.submission == "not_attempted"
    assert page.click_count == 0
    assert page.click_actions == []


@pytest.mark.parametrize("contract_raises", [False, True])
def test_send_probe_failure_without_locator_fault_stays_submit_failed(
    monkeypatch: pytest.MonkeyPatch, contract_raises: bool,
) -> None:
    _install_clock(monkeypatch)
    page = _FakePage()
    original = page.evaluate

    async def evaluate(expression: str, argument: Any = None) -> Any:
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            raise RuntimeError("readiness probe unavailable")
        if (expression == selectors.LOCATOR_CHECK_PROBE_JS
                and argument["target"] == "send" and contract_raises):
            raise RuntimeError("contract probe unavailable")
        return await original(expression, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == "submit_failed"
    assert raised.value.evidence.submission == "not_attempted"
    assert page.click_actions == []


def _install_timed_locator_probes(
    page: _FakePage,
    clock: _FakeClock,
    book: locators.LocatorBook,
    *,
    target: locators.LocatorTarget,
    result_at: Callable[[float], dict[str, object]],
    markers_until: float = 0,
    ready_at: float | None = None,
    readiness_raises: bool = False,
) -> tuple[list[float], list[float]]:
    original = page.evaluate
    selector = locators.SEED_SELECTORS[target]
    check_times: list[float] = []
    healer_times: list[float] = []

    class Healer:
        async def complete(self, system, prompt, timeout_seconds):
            healer_times.append(clock.value)
            return json.dumps({"status": "absent"})

    book.healer = Healer()

    async def evaluate(expression: str, argument: Any = None) -> Any:
        if (expression == selectors.LOCATOR_CHECK_PROBE_JS
                and argument["target"] == target and argument["selector"] == selector):
            check_times.append(clock.value)
            return result_at(clock.value)
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"lang": "en-US", "candidates": [{"index": 0, "selector": ".candidate"}]}
        if expression == selectors.CHALLENGE_DOM_PROBE_JS:
            return ["cf-challenge"] if clock.value < markers_until else []
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            if readiness_raises:
                raise RuntimeError("SyntaxError: invalid selector")
            return ready_at is not None and clock.value >= ready_at
        return await original(expression, argument)

    page.evaluate = evaluate  # type: ignore[method-assign]
    return check_times, healer_times


def test_composer_invalid_selector_without_candidates_respects_challenge_grace(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()
    _, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="composer", markers_until=12,
        result_at=lambda _: {"invalidSelector": True, "candidates": 0},
    )
    with pytest.raises(ask.GptProChallengeError):
        _run(page)
    assert ask.CHALLENGE_GRACE_SECONDS <= clock.value < 12
    assert healer_times == []
    assert not isolated_locator_book.path.exists()


def test_composer_fault_timer_starts_after_challenge_disappears(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()
    _, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="composer", markers_until=6,
        result_at=lambda _: {"matched": 0, "eligible": 0, "candidates": 1},
    )
    with pytest.raises(ask.GptProAskError) as raised:
        _run(page)
    assert raised.value.failure == "locator_unresolved"
    assert healer_times == [6 + ask.LOCATOR_SETTLE_SECONDS]


@pytest.mark.parametrize("reset", ["valid", "wait", "invalid_without_candidates", "probe_error"])
def test_composer_unsustained_fault_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    reset: str,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()

    def result_at(now: float) -> dict[str, object]:
        if int(now) % 4 != 3:
            return {"matched": 0, "eligible": 0, "candidates": 1}
        if reset == "probe_error":
            raise RuntimeError("check unavailable")
        return {
            "matched": int(reset == "valid"), "eligible": int(reset == "valid"),
            "candidates": int(reset == "valid"),
            "invalidSelector": reset == "invalid_without_candidates",
        }

    _, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="composer", result_at=result_at,
    )
    execution = ask._AskExecution(page, "question", None, timeout_seconds=12)
    if reset == "valid":
        asyncio.run(execution._wait_for_composer())
        assert clock.value == 3
    else:
        with pytest.raises(ask._DeadlineExpired):
            asyncio.run(execution._wait_for_composer())
    assert healer_times == []
    assert not isolated_locator_book.path.exists()


def test_send_disabled_valid_button_waits_with_bounded_contract_checks(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
) -> None:
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "POLL_INTERVAL_SECONDS", 0.12)
    page = _FakePage()
    check_times, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="send", ready_at=10,
        result_at=lambda _: {"matched": 1, "eligible": 1, "candidates": 1},
    )
    execution = ask._AskExecution(page, "question", None)
    asyncio.run(execution._wait_for_send_ready())
    assert clock.value >= 10
    assert 9 <= len(check_times) <= 12
    assert healer_times == []
    assert not isolated_locator_book.path.exists()


@pytest.mark.parametrize("scenario", ["candidate_appears", "ready_fault", "invalid_css", "invalid_css_without_candidates"])
def test_send_rediscovery_requires_sustained_fault_with_candidates(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    scenario: str,
) -> None:
    clock = _install_clock(monkeypatch)
    monkeypatch.setattr(ask, "POLL_INTERVAL_SECONDS", 0.12)
    page = _FakePage()
    invalid_css = scenario.startswith("invalid_css")

    def result_at(now: float) -> dict[str, object]:
        candidates = int(now >= 4) if scenario == "candidate_appears" else 1
        if scenario == "invalid_css_without_candidates":
            candidates = 0
        return {"matched": 0, "eligible": 0, "candidates": candidates,
                "invalidSelector": invalid_css}

    check_times, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="send", result_at=result_at,
        ready_at=0 if scenario == "ready_fault" else None, readiness_raises=invalid_css,
    )
    execution = ask._AskExecution(page, "question", None)
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(execution._wait_for_send_ready())
    if scenario == "invalid_css_without_candidates":
        assert raised.value.failure == "submit_failed"
        assert clock.value >= ask.SEND_READY_TIMEOUT_SECONDS
        assert healer_times == []
        assert not isolated_locator_book.path.exists()
    else:
        assert raised.value.failure == "locator_unresolved"
        first_fault = next(now for now in check_times if scenario != "candidate_appears" or now >= 4)
        assert len(healer_times) == 1
        assert first_fault + ask.LOCATOR_SETTLE_SECONDS <= healer_times[0] <= first_fault + 5.2


@pytest.mark.parametrize("reset", ["valid", "wait", "invalid_without_candidates", "probe_error"])
def test_send_fault_timer_resets_on_nonrediscoverable_observation(
    monkeypatch: pytest.MonkeyPatch, isolated_locator_book: locators.LocatorBook,
    reset: str,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage()

    def result_at(now: float) -> dict[str, object]:
        if not 3 <= now < 6:
            return {"matched": 0, "eligible": 0, "candidates": 1}
        if reset == "probe_error":
            raise RuntimeError("contract probe unavailable")
        return {
            "matched": int(reset == "valid"), "eligible": int(reset == "valid"),
            "candidates": int(reset == "valid"),
            "invalidSelector": reset == "invalid_without_candidates",
        }

    _, healer_times = _install_timed_locator_probes(
        page, clock, isolated_locator_book, target="send", result_at=result_at,
    )
    execution = ask._AskExecution(page, "question", None)
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(execution._wait_for_send_ready())
    if reset == "probe_error":
        assert raised.value.failure == "submit_failed"
        assert healer_times == []
        assert not isolated_locator_book.path.exists()
    else:
        assert raised.value.failure == "locator_unresolved"
        assert healer_times == [6 + ask.LOCATOR_SETTLE_SECONDS]


@pytest.mark.parametrize("exit_path", ["late_not_ready", "click_error", "click_timeout", "echo_timeout", "cancel_echo"])
def test_rediscovered_send_exit_settlement(monkeypatch, isolated_locator_book, exit_path):
    clock = _install_clock(monkeypatch)
    book = isolated_locator_book
    page = _FakePage()
    calls = []
    class Healer:
        async def complete(self, *args, **kwargs):
            calls.append(clock.value)
            if exit_path == "late_not_ready":
                clock.value = 27
            return '{"status":"selected","candidate":0}'
    book.healer = Healer()
    original = page.evaluate
    echo_started = asyncio.Event()
    async def evaluate(expression, argument=None):
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"candidates": [{"selector": ".send"}]}
        if expression == selectors.LOCATOR_CHECK_PROBE_JS and argument["target"] == "send":
            valid = argument["selector"] == ".send"
            return {"matched": int(valid), "eligible": int(valid), "candidates": 1, "lang": "ko"}
        if expression == selectors.LOCATOR_CHECK_PROBE_JS:
            return {"matched": 1, "eligible": 1, "candidates": 1, "lang": "ko"}
        if expression == selectors.SEND_BUTTON_READY_PROBE_JS:
            return argument["selector"] == ".send" and exit_path != "late_not_ready"
        if expression == selectors.USER_ECHO_PROBE_JS:
            if exit_path == "cancel_echo":
                echo_started.set()
                await asyncio.Event().wait()
            return None
        return await original(expression, argument)
    page.evaluate = evaluate
    async def click(*args, **kwargs):
        page.click_count += 1
        if exit_path == "click_error":
            raise RuntimeError("click failed")
        if exit_path == "click_timeout":
            raise TimeoutError("click timed out")
    page.click = click
    async def run():
        execution = ask._AskExecution(page, "question", None, locator_book=book)
        if exit_path == "cancel_echo":
            task = asyncio.create_task(execution.run())
            await echo_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ask.GptProAskError) as raised:
                await execution.run()
            assert raised.value.failure == ("echo_timeout" if exit_path == "echo_timeout" else "timeout" if exit_path == "click_timeout" else "submit_failed")
            assert execution.evidence.submission == ("not_attempted" if exit_path == "late_not_ready" else "uncertain")
    asyncio.run(run())
    assert not book.pending
    assert page.click_count == (0 if exit_path == "late_not_ready" else 1)
    assert book._record("ko", "send").get("consecutive_failures", 0) == int(exit_path == "click_error")
    if exit_path == "click_error":
        assert book.refusal("ko", "send")
        with pytest.raises(ask.GptProAskError) as raised:
            _run(page)
        assert raised.value.failure == "locator_unresolved"
        assert len(calls) == 1


@pytest.mark.parametrize("target", ["composer", "send"])
@pytest.mark.parametrize("verdict", ["wait", "fault"])
def test_action_recheck_settlement(isolated_locator_book, verdict, target):
    page = _FakePage()
    book = isolated_locator_book
    class Healer:
        async def complete(self, *args, **kwargs):
            return '{"status":"selected","candidate":0}'
    book.healer = Healer()
    async def run():
        result = await book.rediscover(evaluate_candidate, target, "ko",
            failed_selector=locators.SEED_SELECTORS[target], composer_selector=selectors.COMPOSER_SELECTOR,
            deadline=ask._monotonic() + 20)
        execution = ask._AskExecution(page, "question", None, locator_book=book)
        execution.locator_environment = "ko"
        execution.locator_selectors[target] = result.selector
        execution.unproven_locators[target] = result
        async def evaluate(*args):
            return {"matched": 0, "eligible": 0, "candidates": int(verdict == "fault")}
        page.evaluate = evaluate
        with pytest.raises(ask.GptProAskError):
            if target == "composer":
                await execution._fill_and_verify()
            else:
                async def ready():
                    pass
                execution._dismiss_modal = ready
                execution._wait_for_send_ready = ready
                await execution._click_send()
    async def evaluate_candidate(expression, argument):
        if expression == selectors.LOCATOR_CANDIDATES_PROBE_JS:
            return {"candidates": [{"selector": ".composer"}]}
        return {"matched": 1, "eligible": 1}
    asyncio.run(run())
    assert not book.pending
    assert book._record("ko", target).get("consecutive_failures", 0) == int(verdict == "fault")


@pytest.mark.parametrize("overall", [False, True])
@pytest.mark.parametrize("blocked", ["lock", "healer"])
def test_runner_rediscovery_deadline(monkeypatch, isolated_locator_book, overall, blocked):
    monkeypatch.setattr(locators, "MINIMUM_REDISCOVERY_SECONDS", 0)
    book = isolated_locator_book
    class Healer:
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(1)
            return "late malformed reply"
    book.healer = Healer()
    async def run():
        execution = ask._AskExecution(_FakePage(), "question", None,
            timeout_seconds=.05 if overall else 1, locator_book=book)
        execution.locator_environment = "ko"
        async def evaluate(expression, args):
            return {"candidates": [{"selector": ".send"}]}
        execution.page.evaluate = evaluate
        lock = book._locks.setdefault(("ko", "send"), asyncio.Lock())
        if blocked == "lock":
            await lock.acquire()
        check = locators.classify_check({"candidates": 1, "lang": "ko"})
        start = ask._monotonic()
        try:
            with pytest.raises(ask._DeadlineExpired if overall else ask.GptProAskError) as raised:
                await execution._rediscover("send", check, start + .05)
            if not overall:
                assert raised.value.failure == "locator_unresolved"
                assert "rediscovery did not finish within the send wait budget" in str(raised.value)
            assert ask._monotonic() - start < .15
        finally:
            if blocked == "lock":
                lock.release()
    asyncio.run(run())
    assert not book.pending and not book.path.exists()


def test_unverifiable_composer_readback_abandons(isolated_locator_book):
    book = isolated_locator_book
    page = _FakePage()
    execution = ask._AskExecution(page, "question", None, locator_book=book)
    execution.locator_environment = "ko"
    attempt = locators.RediscoveredLocator(selectors.COMPOSER_SELECTOR, "attempt")
    execution.unproven_locators["composer"] = attempt
    book.pending[("ko", "composer")] = attempt
    original = page.evaluate
    async def evaluate(expression, args=None):
        if expression == selectors.COMPOSER_READBACK_PROBE_JS:
            raise RuntimeError("probe unavailable")
        return await original(expression, args)
    page.evaluate = evaluate
    with pytest.raises(ask.GptProAskError, match="could not verify"):
        asyncio.run(execution._fill_and_verify())
    assert not book.pending and not book.path.exists()


def test_playwright_click_timeout_abandons(isolated_locator_book):
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    book = isolated_locator_book
    page = _FakePage()
    execution = ask._AskExecution(page, "question", None, locator_book=book)
    execution.locator_environment = "ko"
    attempt = locators.RediscoveredLocator(selectors.SEND_BUTTON_SELECTOR, "attempt")
    execution.unproven_locators["send"] = attempt
    book.pending[("ko", "send")] = attempt
    async def click(*args, **kwargs):
        page.click_count += 1
        raise PlaywrightTimeoutError("click timed out")
    page.click = click
    with pytest.raises(ask.GptProAskError) as raised:
        asyncio.run(execution._click_send())
    assert raised.value.failure == "submit_failed"
    assert execution.evidence.submission == "uncertain"
    assert page.click_count == 1
    assert not book.pending and not book.path.exists()
