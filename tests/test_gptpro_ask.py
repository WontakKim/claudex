"""Tests for the network-first gptpro ask runner with a fake page."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from claudex.gptpro import ask, selectors

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
            "requestfinished": [],
            "response": [],
        }
        self.composer_value = ""
        self.filled_prompt = ""
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

    async def fill(self, selector: str, value: str) -> None:
        assert selector == selectors.COMPOSER_SELECTOR
        if self.hang_operation == "fill":
            await asyncio.Event().wait()
        self.call_order.append("fill")
        self.composer_value = value
        self.filled_prompt = value

    async def click(self, selector: str, *, timeout: int) -> None:
        assert selector == selectors.SEND_BUTTON_SELECTOR
        assert timeout > 0
        self.click_count += 1
        if self.hang_operation == "click":
            await asyncio.Event().wait()
        if self.swallow_first_click and self.click_count == 1:
            return
        self.composer_value = ""
        post_data = json.dumps({"conversation_id": _CONVERSATION_ID})
        if self.signal == "weak":
            self._emit(
                "requestfinished", _FakeRequest(_STREAM_URL, "POST", post_data)
            )
        elif self.signal == "strong":
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
    assert page.listeners == {"requestfinished": [], "response": []}
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
            on_conversation_id=on_conversation_id,
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
            on_conversation_id=captured_conversation_ids.append,
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
    ) -> None:
        assert attached_page is page
        page.call_order.append("attach")
        captured.append((attachment_paths, timeout_seconds))

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


def test_missing_echo_with_retained_composer_retries_click_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    page = _FakePage(swallow_first_click=True)

    assert _run(page) == "server **raw** markdown"
    assert page.click_count == 2
    assert clock.value >= ask.ECHO_PROBE_TIMEOUT_SECONDS


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
    assert page.listeners == {"requestfinished": [], "response": []}


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
