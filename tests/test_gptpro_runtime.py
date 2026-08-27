"""Tests for the warm gptpro ask runtime lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from claudex.gptpro import ask, browser, runtime


class _FakeLockHandle:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


class _FakePage:
    def __init__(self, *, user_agent: str = "Mozilla/5.0 Chrome/151.0") -> None:
        self.user_agent = user_agent
        self.extra_headers: dict[str, str] | None = None
        self.close_calls = 0

    async def evaluate(self, script: str) -> str:
        assert "navigator.userAgent" in script
        return self.user_agent

    async def set_extra_http_headers(self, headers: dict[str, str]) -> None:
        self.extra_headers = headers

    async def close(self) -> None:
        self.close_calls += 1


class _FakeContext:
    def __init__(self) -> None:
        self.pages: list[_FakePage] = []
        self.close_calls = 0

    async def new_page(self) -> _FakePage:
        page = _FakePage()
        self.pages.append(page)
        return page

    async def close(self) -> None:
        self.close_calls += 1


class _RuntimeFakes:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        contexts: list[_FakeContext],
    ) -> None:
        self._contexts = list(contexts)
        self.launch_calls: list[tuple[Path, bool]] = []
        self.close_calls: list[_FakeContext] = []
        self.lock_paths: list[Path] = []
        self.locks: list[_FakeLockHandle] = []
        self.sleep_calls: list[float] = []

        monkeypatch.setattr(
            runtime.session,
            "session_status",
            lambda: {"valid": True, "message": "valid"},
        )
        monkeypatch.setattr(runtime, "_sleep", self._sleep)
        monkeypatch.setattr(runtime.random, "uniform", lambda _low, _high: 1.5)
        monkeypatch.setattr(
            runtime.locking, "try_file_lock", self._try_file_lock
        )
        monkeypatch.setattr(
            runtime.browser, "launch_persistent_profile", self._launch
        )
        monkeypatch.setattr(
            runtime.browser, "close_playwright_resource", self._close
        )

    async def _sleep(self, interval: float) -> None:
        self.sleep_calls.append(interval)

    def _try_file_lock(self, path: Path) -> _FakeLockHandle:
        self.lock_paths.append(path)
        lock = _FakeLockHandle()
        self.locks.append(lock)
        return lock

    async def _launch(
        self, profile_dir: Path, *, headless: bool = False
    ) -> _FakeContext:
        self.launch_calls.append((profile_dir, headless))
        return self._contexts.pop(0)

    async def _close(self, context: _FakeContext) -> None:
        self.close_calls.append(context)
        await context.close()


def _outcome(text: str) -> ask.AskOutcome:
    return ask.AskOutcome(text=text, marker="marker", conversation_id=None)


def test_runtime_initializes_lazily_and_reuses_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])
    conversation_id_callbacks: list[Callable[[str], None] | None] = []
    marker_callbacks: list[Callable[[str], None] | None] = []

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status
        conversation_id_callbacks.append(on_conversation_id)
        marker_callbacks.append(on_marker)
        return _outcome(f"answer: {question}")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        captured_conversation_ids: list[str] = []
        callback = captured_conversation_ids.append
        captured_markers: list[str] = []
        marker_callback = captured_markers.append
        assert fakes.launch_calls == []
        first = await ask_runtime.ask(
            "first",
            on_conversation_id=callback,
            on_marker=marker_callback,
        )
        second = await ask_runtime.ask("second")
        assert first.text == "answer: first"
        assert second.text == "answer: second"
        assert conversation_id_callbacks == [callback, None]
        assert marker_callbacks == [marker_callback, None]
        assert len(fakes.launch_calls) == 1
        assert fakes.launch_calls[0][1] is True
        assert len(context.pages) == 2
        assert all(page.close_calls == 1 for page in context.pages)
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_propagates_conversation_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    captured_options: list[tuple[str | None, float | None]] = []

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        captured_options.append((conversation_id, timeout_seconds))
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        outcome = await ask_runtime.ask(
            "question",
            conversation_id="123e4567-e89b-12d3-a456-426614174000",
            timeout_seconds=12.5,
        )
        assert outcome.text == "question"
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert captured_options == [
        ("123e4567-e89b-12d3-a456-426614174000", 12.5)
    ]


def test_runtime_propagates_attachment_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    captured_paths: list[list[str] | None] = []

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
        attachment_paths: list[str] | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        captured_paths.append(attachment_paths)
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        outcome = await ask_runtime.ask(
            "question", attachment_paths=["notes.txt"]
        )
        assert outcome.text == "question"
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert captured_paths == [["notes.txt"]]


def test_runtime_limits_concurrent_asks_to_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    active = 0
    maximum_active = 0
    started: list[str] = []
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        nonlocal active, maximum_active
        del page, on_status, on_conversation_id, on_marker
        active += 1
        maximum_active = max(maximum_active, active)
        started.append(question)
        if len(started) == 2:
            two_started.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)
    monkeypatch.setenv("GPTPRO_MAX_CONCURRENT_ASKS", "2")

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        tasks = [
            asyncio.create_task(ask_runtime.ask(question))
            for question in ("one", "two", "three")
        ]
        await two_started.wait()
        await asyncio.sleep(0)
        assert len(started) == 2
        assert maximum_active == 2
        release.set()
        outcomes = await asyncio.gather(*tasks)
        assert [outcome.text for outcome in outcomes] == ["one", "two", "three"]
        assert maximum_active == 2
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_applies_submission_jitter_after_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    def uniform(low: float, high: float) -> float:
        assert (low, high) == (1.0, 2.0)
        return 1.75

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        return _outcome(question)

    monkeypatch.setattr(runtime.random, "uniform", uniform)
    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert (await ask_runtime.ask("question")).text == "question"
        assert fakes.sleep_calls == [1.75]
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_invalid_session_fails_before_browser_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime.session,
        "session_status",
        lambda: {
            "valid": False,
            "message": "run claudex-gateway gptpro login; session expired",
        },
    )

    def fail_lock(_path: Path) -> None:
        raise AssertionError("the profile lock must not be acquired")

    async def fail_launch(_profile_dir: Path, *, headless: bool) -> Any:
        raise AssertionError("the browser must not be launched")

    monkeypatch.setattr(runtime.locking, "try_file_lock", fail_lock)
    monkeypatch.setattr(runtime.browser, "launch_persistent_profile", fail_launch)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(runtime.GptProSessionExpiredError):
            await ask_runtime.ask("question")
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_closes_page_when_execute_ask_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    failure = ask.GptProAskError("timeout", "deadline expired")

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, on_status, on_conversation_id, on_marker
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.ask("question")
        assert raised.value is failure
        assert context.pages[0].close_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_closed_context_is_discarded_and_reinitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_context = _FakeContext()
    second_context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [first_context, second_context])
    calls = 0

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        nonlocal calls
        del page, on_status, on_conversation_id, on_marker
        calls += 1
        if calls == 1:
            closed_error = RuntimeError(
                "Target page, context or browser has been closed"
            )
            raise ask.GptProAskError(
                "error", "the ChatGPT ask failed unexpectedly"
            ) from closed_error
        return _outcome(f"answer: {question}")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(ask.GptProAskError):
            await ask_runtime.ask("first")
        assert first_context.close_calls == 1
        assert fakes.locks[0].release_calls == 1

        assert (await ask_runtime.ask("second")).text == "answer: second"
        assert len(fakes.launch_calls) == 2
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_aclose_cleans_up_playwright_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        await ask_runtime.ask("question")
        await ask_runtime.aclose()
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert fakes.close_calls == [context]
    assert context.close_calls == 1


def test_runtime_holds_profile_lock_until_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        await ask_runtime.ask("question")
        assert len(fakes.lock_paths) == 1
        assert fakes.lock_paths[0].name == "chrome-profile.lock"
        assert fakes.locks[0].release_calls == 0
        await ask_runtime.aclose()
        assert fakes.locks[0].release_calls == 1

    asyncio.run(scenario())


class _HeadlessUserAgentContext(_FakeContext):
    def __init__(self, user_agent: str) -> None:
        super().__init__()
        self._user_agent = user_agent

    async def new_page(self) -> _FakePage:
        page = _FakePage(user_agent=self._user_agent)
        self.pages.append(page)
        return page


def _install_ask_stub(monkeypatch: pytest.MonkeyPatch, context: _FakeContext) -> None:
    _RuntimeFakes(monkeypatch, [context])

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        return _outcome(f"answer: {question}")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)


def test_ask_hardens_headless_user_agent_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _HeadlessUserAgentContext(
        "Mozilla/5.0 HeadlessChrome/151.0.0.0"
    )
    _install_ask_stub(monkeypatch, context)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert (await ask_runtime.ask("question")).text == "answer: question"
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert context.pages[0].extra_headers == {
        "User-Agent": "Mozilla/5.0 Chrome/151.0.0.0"
    }


def test_ask_leaves_plain_user_agent_header_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _HeadlessUserAgentContext("Mozilla/5.0 Chrome/151.0.0.0")
    _install_ask_stub(monkeypatch, context)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert (await ask_runtime.ask("question")).text == "answer: question"
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert context.pages[0].extra_headers is None


_CONVERSATION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _detached_conversation(marker: str, text: str) -> dict[str, object]:
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
                    "content": {"content_type": "text", "parts": [text]},
                    "status": "finished_successfully",
                    "end_turn": True,
                },
            },
        },
    }


class _PollerFakePage(_FakePage):
    def __init__(
        self,
        marker: str,
        conversation_responses: list[
            tuple[int, dict[str, object] | None] | BaseException
        ],
        *,
        on_conversation_fetch: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.marker = marker
        self.conversation_responses = list(conversation_responses)
        self.on_conversation_fetch = on_conversation_fetch
        self.goto_urls: list[str] = []
        self.fetch_arguments: list[dict[str, object]] = []

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_urls.append(url)
        assert wait_until == "domcontentloaded"
        assert timeout == browser.NAVIGATION_TIMEOUT_MS

    async def evaluate(self, script: str, argument: Any = None) -> Any:
        if "navigator.userAgent" in script:
            return self.user_agent
        assert script == runtime.PAGE_FETCH_PROBE_JS
        assert isinstance(argument, dict)
        self.fetch_arguments.append(argument)
        if argument["url"] == "https://chatgpt.com/api/auth/session":
            return {
                "status": 200,
                "headers": {},
                "text": "",
                "json": {"accessToken": "access-token"},
                "fetchError": None,
                "timedOut": False,
            }
        if self.on_conversation_fetch is not None:
            self.on_conversation_fetch()
        response = self.conversation_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        status, body = response
        return {
            "status": status,
            "headers": {},
            "text": "",
            "json": body,
            "fetchError": None,
            "timedOut": False,
        }


class _PollerFakeContext:
    def __init__(self, page: _PollerFakePage) -> None:
        self.page = page
        self.new_page_calls = 0

    async def new_page(self) -> _PollerFakePage:
        self.new_page_calls += 1
        return self.page


async def _wait_for_poller_idle(poller: runtime.DetachPoller) -> None:
    while poller._task is not None:
        await asyncio.sleep(0)


def test_detach_poller_completes_finished_marker_turn_and_closes_tab() -> None:
    marker = "[gptpro-transport-nonce:poller-success]"
    page = _PollerFakePage(
        marker,
        [(200, _detached_conversation(marker, "detached raw answer"))],
    )
    context = _PollerFakeContext(page)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(
            _CONVERSATION_ID,
            marker,
            deadline=runtime._monotonic() + 100.0,
        )
        outcome = await future
        await _wait_for_poller_idle(poller)

        assert outcome == ask.AskOutcome(
            text="detached raw answer",
            marker=marker,
            conversation_id=_CONVERSATION_ID,
        )
        assert context.new_page_calls == 1
        assert page.goto_urls == ["https://chatgpt.com/"]
        assert page.fetch_arguments[-1]["headers"] == {
            "Authorization": "Bearer access-token"
        }
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


def test_detach_poller_backs_off_on_429_and_recovers_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:poller-backoff]"
    page = _PollerFakePage(
        marker,
        [
            (429, None),
            (200, _detached_conversation(marker, "answer after backoff")),
        ],
    )
    context = _PollerFakeContext(page)
    observed_backoffs: list[float] = []

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))

        async def sleep(_seconds: float) -> None:
            observed_backoffs.append(poller.current_backoff_seconds())

        monkeypatch.setattr(runtime, "_sleep", sleep)
        future = poller.register(
            _CONVERSATION_ID,
            marker,
            deadline=runtime._monotonic() + 100.0,
        )
        outcome = await future
        await _wait_for_poller_idle(poller)

        assert outcome.text == "answer after backoff"
        assert observed_backoffs == [90.0]
        assert poller.current_backoff_seconds() == 0.0
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


def test_detach_poller_retries_navigation_destroyed_fetch_on_next_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:poller-navigation-retry]"
    page = _PollerFakePage(
        marker,
        [
            RuntimeError("Execution context was destroyed during navigation"),
            (200, _detached_conversation(marker, "answer after navigation")),
        ],
    )
    context = _PollerFakeContext(page)
    sleep_calls: list[float] = []

    async def sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(runtime, "_sleep", sleep)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(
            _CONVERSATION_ID,
            marker,
            deadline=runtime._monotonic() + 100.0,
        )
        outcome = await future
        await _wait_for_poller_idle(poller)

        assert outcome.text == "answer after navigation"
        assert sleep_calls == [runtime.DETACH_POLL_INTERVAL_SECONDS]
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


def test_detach_poller_resets_backoff_when_registrations_become_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:poller-idle-reset]"
    page = _PollerFakePage(marker, [(429, None)])
    context = _PollerFakeContext(page)
    now = 0.0
    observed_backoffs: list[float] = []

    def monotonic() -> float:
        return now

    async def scenario() -> None:
        nonlocal now
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))

        async def sleep(seconds: float) -> None:
            nonlocal now
            observed_backoffs.append(poller.current_backoff_seconds())
            now += seconds

        monkeypatch.setattr(runtime, "_monotonic", monotonic)
        monkeypatch.setattr(runtime, "_sleep", sleep)
        future = poller.register(_CONVERSATION_ID, marker, deadline=1.0)
        with pytest.raises(ask.GptProAskError) as raised:
            await future
        await _wait_for_poller_idle(poller)

        assert raised.value.failure == "timeout"
        assert observed_backoffs == [90.0]
        assert poller.current_backoff_seconds() == 0.0
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


def test_detach_poller_times_out_immediately_when_fetch_exhausts_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:poller-fetch-deadline]"
    now = 0.0

    def expire_budget() -> None:
        nonlocal now
        now = 2.0

    page = _PollerFakePage(
        marker,
        [(200, _detached_conversation(marker, "late answer"))],
        on_conversation_fetch=expire_budget,
    )
    context = _PollerFakeContext(page)
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)

    async def fail_sleep(_seconds: float) -> None:
        raise AssertionError("an expired fetch must not wait for another cycle")

    monkeypatch.setattr(runtime, "_sleep", fail_sleep)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(_CONVERSATION_ID, marker, deadline=1.0)
        with pytest.raises(ask.GptProAskError) as raised:
            await future
        await _wait_for_poller_idle(poller)

        assert raised.value.failure == "timeout"
        assert str(raised.value) == (
            "the detached ask budget expired while polling for the answer"
        )
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 403])
def test_detach_poller_classifies_authorization_failure_as_session_expired(
    status: int,
) -> None:
    marker = "[gptpro-transport-nonce:poller-auth]"
    page = _PollerFakePage(marker, [(status, None)])
    context = _PollerFakeContext(page)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(
            _CONVERSATION_ID,
            marker,
            deadline=runtime._monotonic() + 100.0,
        )
        with pytest.raises(ask.GptProSessionExpiredError):
            await future
        await _wait_for_poller_idle(poller)
        assert page.close_calls == 1
        await poller.aclose()

    asyncio.run(scenario())


def test_detach_poller_sweeps_expired_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:poller-expired]"
    page = _PollerFakePage(marker, [])
    context = _PollerFakeContext(page)
    monkeypatch.setattr(runtime, "_monotonic", lambda: 10.0)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(_CONVERSATION_ID, marker, deadline=9.0)
        with pytest.raises(ask.GptProAskError) as raised:
            await future
        await _wait_for_poller_idle(poller)

        assert raised.value.failure == "timeout"
        assert str(raised.value) == (
            "the detached ask budget expired while polling for the answer"
        )
        assert context.new_page_calls == 0
        await poller.aclose()

    asyncio.run(scenario())


class _RuntimeDetachPollerFake:
    def __init__(self, _get_context: Callable[[], Awaitable[Any]]) -> None:
        self.future: asyncio.Future[ask.AskOutcome] | None = None
        self.registrations: list[tuple[str, str, float]] = []
        self.close_calls = 0
        self.backoff_seconds = 0.0

    def register(
        self, conversation_id: str, marker: str, deadline: float
    ) -> asyncio.Future[ask.AskOutcome]:
        self.registrations.append((conversation_id, marker, deadline))
        self.future = asyncio.get_running_loop().create_future()
        return self.future

    def current_backoff_seconds(self) -> float:
        return self.backoff_seconds

    async def aclose(self) -> None:
        self.close_calls += 1


def test_runtime_detaches_waiting_answer_when_submitter_contends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    pollers: list[_RuntimeDetachPollerFake] = []

    def create_poller(
        get_context: Callable[[], Awaitable[Any]],
    ) -> _RuntimeDetachPollerFake:
        poller = _RuntimeDetachPollerFake(get_context)
        pollers.append(poller)
        return poller

    monkeypatch.setattr(runtime, "DetachPoller", create_poller)
    monkeypatch.setenv("GPTPRO_MAX_CONCURRENT_ASKS", "1")
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker
        assert should_detach is not None
        assert on_detach is not None
        if question == "first":
            first_started.set()
            while not should_detach():
                await asyncio.sleep(0)
            return await on_detach(
                ask.AskSubmission(
                    marker="first-marker",
                    conversation_id=_CONVERSATION_ID,
                )
            )
        second_started.set()
        return _outcome("second answer")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        first_task = asyncio.create_task(ask_runtime.ask("first"))
        await first_started.wait()
        second_task = asyncio.create_task(ask_runtime.ask("second"))
        await second_started.wait()

        assert second_task.done()
        assert not first_task.done()
        assert len(context.pages) == 2
        assert context.pages[0].close_calls == 1
        assert pollers[0].registrations[0][:2] == (
            _CONVERSATION_ID,
            "first-marker",
        )

        assert pollers[0].future is not None
        pollers[0].future.set_result(
            ask.AskOutcome(
                text="first detached answer",
                marker="first-marker",
                conversation_id=_CONVERSATION_ID,
            )
        )
        first, second = await asyncio.gather(first_task, second_task)
        assert first.text == "first detached answer"
        assert second.text == "second answer"
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_keeps_monitor_path_without_contention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, on_status, on_conversation_id, on_marker, on_detach
        assert should_detach is not None
        assert not should_detach()
        return _outcome(f"monitored: {question}")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        outcome = await ask_runtime.ask("question")
        assert outcome.text == "monitored: question"
        assert context.pages[0].close_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_extends_submission_jitter_during_poller_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del (
            page,
            on_status,
            on_conversation_id,
            on_marker,
            should_detach,
            on_detach,
        )
        return _outcome(question)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        ask_runtime._poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller.backoff_seconds = 90.0
        await ask_runtime.ask("question")
        assert fakes.sleep_calls == [91.5]
        await ask_runtime.aclose()

    asyncio.run(scenario())
