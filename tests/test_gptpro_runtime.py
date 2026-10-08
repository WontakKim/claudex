"""Tests for the warm gptpro ask runtime lifecycle."""

from __future__ import annotations

import asyncio
import logging
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
        self.clear_cookies_calls: list[Any] = []

    async def clear_cookies(self, *, name: Any = None) -> None:
        self.clear_cookies_calls.append(name)

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
        *,
        default_user_agent: str | None = None,
    ) -> None:
        self._contexts = list(contexts)
        self.default_user_agent = default_user_agent
        self.launch_calls: list[tuple[Path, bool]] = []
        self.user_agent_overrides: list[str | None] = []
        self.user_agent_probe_calls = 0
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
            runtime.browser, "read_chromium_user_agent", self._read_user_agent
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

    async def _read_user_agent(self) -> str | None:
        self.user_agent_probe_calls += 1
        return self.default_user_agent

    async def _launch(
        self,
        profile_dir: Path,
        *,
        headless: bool = False,
        user_agent: str | None = None,
    ) -> _FakeContext:
        self.launch_calls.append((profile_dir, headless))
        self.user_agent_overrides.append(user_agent)
        return self._contexts.pop(0)

    async def _close(self, context: _FakeContext) -> None:
        self.close_calls.append(context)
        await context.close()


def _outcome(text: str) -> ask.AskOutcome:
    return ask.AskOutcome(text=text, marker="marker", conversation_id=None)


@pytest.mark.parametrize(
    ("configured_value", "expected_seconds"),
    [
        (None, runtime.DEFAULT_RAW_TURN_RECOVERY_SECONDS),
        ("invalid", runtime.DEFAULT_RAW_TURN_RECOVERY_SECONDS),
        ("0", 0.0),
        ("125.5", 125.5),
    ],
    ids=["default", "invalid", "disabled", "positive"],
)
def test_raw_turn_recovery_seconds_parses_environment(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str | None,
    expected_seconds: float,
) -> None:
    if configured_value is None:
        monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)
    else:
        monkeypatch.setenv(
            "GPTPRO_RAW_TURN_RECOVERY_SECONDS", configured_value
        )

    assert runtime.raw_turn_recovery_seconds() == expected_seconds


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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page
        assert callbacks is not None
        conversation_id_callbacks.append(callbacks.on_conversation_id)
        marker_callbacks.append(callbacks.on_marker)
        if callbacks.on_conversation_id is not None:
            callbacks.on_conversation_id("captured-conversation")
        if callbacks.on_marker is not None:
            callbacks.on_marker("captured-marker")
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
            callbacks=ask.AskCallbacks(
                on_conversation_id=callback,
                on_marker=marker_callback,
            ),
        )
        second = await ask_runtime.ask("second")
        assert first.text == "answer: first"
        assert second.text == "answer: second"
        assert conversation_id_callbacks[0] is not None
        assert conversation_id_callbacks[1] is None
        assert marker_callbacks[0] is not None
        assert marker_callbacks[1] is None
        assert captured_conversation_ids == ["captured-conversation"]
        assert captured_markers == ["captured-marker"]
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
        attachment_paths: list[str] | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        nonlocal active, maximum_active
        del page, callbacks
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

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime(max_concurrent_asks=2)
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


def test_runtime_default_ignores_concurrency_environment(
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        nonlocal active, maximum_active
        del page, callbacks
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
    monkeypatch.setenv("GPTPRO_MAX_CONCURRENT_ASKS", "1")

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        tasks = [
            asyncio.create_task(ask_runtime.ask(question))
            for question in ("one", "two", "three")
        ]
        await asyncio.wait_for(two_started.wait(), timeout=1)
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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


def test_runtime_clears_stale_cloudflare_cookies_after_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        returned = await ask_runtime._get_context()
        assert returned is context
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert len(fakes.launch_calls) == 1
    assert context.clear_cookies_calls == [
        runtime.session.CLOUDFLARE_COOKIE_NAME_PATTERN
    ]
    assert runtime.session.CLOUDFLARE_COOKIE_NAME_PATTERN.match("cf_clearance")


def test_runtime_launches_with_the_normalized_probe_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(
        monkeypatch,
        [context],
        default_user_agent="Mozilla/5.0 HeadlessChrome/140.0",
    )

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        returned = await ask_runtime._get_context()
        assert returned is context
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert fakes.user_agent_probe_calls == 1
    assert fakes.launch_calls[0][1] is True
    assert fakes.user_agent_overrides == ["Mozilla/5.0 Chrome/140.0"]


@pytest.mark.parametrize("probe_failure", ["raises", "returns_none"])
def test_runtime_launches_without_user_agent_when_probe_falls_back(
    monkeypatch: pytest.MonkeyPatch, probe_failure: str
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    if probe_failure == "raises":

        async def fail_probe() -> str | None:
            raise RuntimeError("probe failed")

        monkeypatch.setattr(
            runtime.browser, "read_chromium_user_agent", fail_probe
        )

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        returned = await ask_runtime._get_context()
        assert returned is context
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert fakes.launch_calls[0][1] is True
    assert fakes.user_agent_overrides == [None]


def test_runtime_releases_profile_lock_when_cookie_clear_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def fail_clear_cookies(*, name: Any = None) -> None:
        raise RuntimeError("cookie clear failed")

    context.clear_cookies = fail_clear_cookies

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(RuntimeError, match="cookie clear failed"):
            await ask_runtime._get_context()
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert len(fakes.launch_calls) == 1
    assert fakes.locks[0].release_calls == 1
    assert fakes.close_calls == [context]
    assert context.close_calls == 1


def test_runtime_releases_lock_and_reraises_when_context_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def fail_clear_cookies(*, name: Any = None) -> None:
        raise RuntimeError("cookie clear failed")

    close_calls = 0

    async def fail_close() -> None:
        nonlocal close_calls
        close_calls += 1
        raise RuntimeError("close failed")

    context.clear_cookies = fail_clear_cookies
    context.close = fail_close

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(RuntimeError, match="cookie clear failed"):
            await ask_runtime._get_context()
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert close_calls == 1
    assert fakes.locks[0].release_calls == 1


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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        nonlocal calls
        del page, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
    caplog: pytest.LogCaptureFixture,
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

    caplog.set_level(logging.INFO, logger="claudex.gptpro.runtime")
    asyncio.run(scenario())

    assert caplog.messages == [
        "gptpro detach poller rate-limited (interval=90s)",
        f"gptpro detached answer recovered (thread={_CONVERSATION_ID} "
        f"chars={len('answer after backoff')})",
        "gptpro detach poller interval restored (45s)",
    ]


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
    caplog: pytest.LogCaptureFixture,
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

    caplog.set_level(logging.WARNING, logger="claudex.gptpro.runtime")
    asyncio.run(scenario())

    assert caplog.messages == [
        f"gptpro detached ask hit an auth failure (thread={_CONVERSATION_ID})"
    ]


def test_read_only_recovery_exhausts_after_unmatched_server_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:expected-turn]"
    page = _PollerFakePage(
        marker,
        [(200, _detached_conversation(
            "[gptpro-transport-nonce:other-turn]", "other answer",
        ))],
    )
    context = _PollerFakeContext(page)
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "0.05")

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        ask_runtime._poller = runtime.DetachPoller(
            lambda: asyncio.sleep(0, result=context),
        )
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.recover(_CONVERSATION_ID, marker)
        assert raised.value.failure == "timeout"
        assert page.fetch_arguments
        await _wait_for_poller_idle(ask_runtime._poller)
        assert page.close_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_detach_poller_sweeps_expired_registration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
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

    caplog.set_level(logging.WARNING, logger="claudex.gptpro.runtime")
    asyncio.run(scenario())

    assert caplog.messages == [
        f"gptpro detached ask timed out (thread={_CONVERSATION_ID})"
    ]


class _RuntimeDetachPollerFake:
    def __init__(self, _get_context: Callable[[], Awaitable[Any]]) -> None:
        self.future: asyncio.Future[ask.AskOutcome] | None = None
        self.registrations: list[tuple[str, str, float]] = []
        self.delivery_deadlines: list[float | None] = []
        self.close_calls = 0
        self.backoff_seconds = 0.0

    def register(
        self, conversation_id: str, marker: str, deadline: float,
        *, delivery_deadline: float | None = None,
    ) -> asyncio.Future[ask.AskOutcome]:
        self.registrations.append((conversation_id, marker, deadline))
        self.delivery_deadlines.append(delivery_deadline)
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
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    detached_callback_calls = 0

    def capture_detached() -> None:
        nonlocal detached_callback_calls
        assert pollers[0].registrations == []
        detached_callback_calls += 1
        raise RuntimeError("callback failed")

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks
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
        ask_runtime = runtime.AskRuntime(max_concurrent_asks=1)
        first_task = asyncio.create_task(
            ask_runtime.ask(
                "first",
                callbacks=ask.AskCallbacks(on_detached=capture_detached),
            )
        )
        await first_started.wait()
        second_task = asyncio.create_task(ask_runtime.ask("second"))
        await second_started.wait()

        await second_task
        assert second_task.done()
        assert not first_task.done()
        assert len(context.pages) == 2
        assert context.pages[0].close_calls == 1
        assert detached_callback_calls == 1
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


def test_runtime_recovers_no_raw_turn_through_detach_poller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "120")
    now = 100.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    marker = "[gptpro-transport-nonce:raw-turn-recovery]"
    failure = ask.GptProAskError("no_raw_turn", "raw turn missing")
    captured_conversation_ids: list[str] = []
    captured_markers: list[str] = []
    detached_calls: list[None] = []
    status_messages: list[str] = []

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, should_detach, on_detach
        assert callbacks is not None
        assert callbacks.on_conversation_id is not None
        assert callbacks.on_marker is not None
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker(marker)
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime(max_concurrent_asks=1)
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        outcome_task = asyncio.create_task(
            ask_runtime.ask(
                "question",
                callbacks=ask.AskCallbacks(
                    on_status=status_messages.append,
                    on_conversation_id=captured_conversation_ids.append,
                    on_marker=captured_markers.append,
                    on_detached=lambda: detached_calls.append(None),
                ),
            )
        )
        while poller.future is None:
            await asyncio.sleep(0)

        assert poller.registrations == [
            (_CONVERSATION_ID, marker, now + 120.0)
        ]
        assert poller.registrations[0][2] > now
        assert captured_conversation_ids == [_CONVERSATION_ID]
        assert captured_markers == [marker]
        assert detached_calls == [None]
        assert status_messages == ["detached; polling for the answer"]
        assert not ask_runtime._ask_semaphore.locked()

        recovered_outcome = ask.AskOutcome(
            text="recovered answer",
            marker=marker,
            conversation_id=_CONVERSATION_ID,
        )
        poller.future.set_result(recovered_outcome)
        assert await outcome_task == recovered_outcome
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_preserves_no_raw_turn_when_recovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "120")
    marker = "[gptpro-transport-nonce:raw-turn-recovery-failure]"
    failure = ask.GptProAskError("no_raw_turn", "raw turn missing")

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, should_detach, on_detach
        assert callbacks is not None
        assert callbacks.on_conversation_id is not None
        assert callbacks.on_marker is not None
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker(marker)
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        outcome_task = asyncio.create_task(
            ask_runtime.ask("question", callbacks=ask.AskCallbacks())
        )
        while poller.future is None:
            await asyncio.sleep(0)

        recovery_failure = RuntimeError("poller failed")
        poller.future.set_exception(recovery_failure)
        with pytest.raises(ask.GptProAskError) as raised:
            await outcome_task

        assert raised.value is failure
        assert raised.value.failure == "no_raw_turn"
        assert raised.value.__cause__ is recovery_failure
        assert raised.value.evidence.recovery_failure == "error"
        assert raised.value.evidence.recovery_detail == "poller failed"
        assert "recovery error: poller failed" in str(raised.value)
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_auto_recovery_keeps_original_failure_and_reports_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "60")
    marker = "[gptpro-transport-nonce:auth-failure]"
    failure = ask.GptProAskError("timeout", "original answer timed out")
    failure.evidence = ask.AskEvidence(
        submission="uncertain", conversation_id=_CONVERSATION_ID,
        failure_stage="answer",
    )
    observations: list[ask.AskEvidence] = []

    async def execute_ask_outcome(
        page: _FakePage, question: str, *,
        callbacks: ask.AskCallbacks | None = None, **_kwargs: object,
    ) -> ask.AskOutcome:
        del page, question
        assert callbacks is not None
        assert callbacks.on_marker is not None
        assert callbacks.on_conversation_id is not None
        callbacks.on_marker(marker)
        callbacks.on_conversation_id(_CONVERSATION_ID)
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.ask(
            "question", callbacks=ask.AskCallbacks(on_evidence=observations.append),
        ))
        while poller.future is None:
            await asyncio.sleep(0)
        poller.future.set_exception(ask.GptProSessionExpiredError())
        with pytest.raises(ask.GptProAskError) as raised:
            await task
        assert raised.value.failure == "timeout"
        assert raised.value.evidence.failure_stage == "answer"
        assert raised.value.evidence.recovery_failure == "session_expired"
        assert raised.value.evidence.recovery == "unavailable"
        assert "sign in again" in raised.value.evidence.recovery_detail
        assert "recovery session_expired" in str(raised.value)
        assert observations[-1] == raised.value.evidence
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_recovers_lost_echo_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "120")
    failure = ask.GptProAskError("echo_timeout", "echo missing")

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, should_detach, on_detach
        assert callbacks is not None
        assert callbacks.on_conversation_id is not None
        assert callbacks.on_marker is not None
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker("marker")
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        outcome_task = asyncio.create_task(
            ask_runtime.ask("question", callbacks=ask.AskCallbacks())
        )
        while poller.future is None and not outcome_task.done():
            await asyncio.sleep(0)
        assert not outcome_task.done()
        assert poller.registrations[0][:2] == (_CONVERSATION_ID, "marker")
        poller.future.set_result(_outcome("recovered echo answer"))
        assert (await outcome_task).text == "recovered echo answer"
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_disables_no_raw_turn_recovery_with_zero_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "0")
    failure = ask.GptProAskError("no_raw_turn", "raw turn missing")

    async def execute_ask_outcome(
        page: _FakePage,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, question, should_detach, on_detach
        assert callbacks is not None
        assert callbacks.on_conversation_id is not None
        assert callbacks.on_marker is not None
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker("marker")
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.ask("question", callbacks=ask.AskCallbacks())

        assert raised.value is failure
        assert raised.value.failure == "no_raw_turn"
        assert poller.registrations == []
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del page, callbacks, on_detach
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
        callbacks: ask.AskCallbacks | None = None,
        should_detach: Callable[[], bool] | None = None,
        on_detach: Callable[
            [ask.AskSubmission], Awaitable[ask.AskOutcome]
        ]
        | None = None,
    ) -> ask.AskOutcome:
        del (
            page,
            callbacks,
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


@pytest.mark.parametrize("failure_code", ["timeout", "submit_failed", "navigation_failed"])
def test_runtime_recovers_uncertain_or_generating_turn_without_another_submit(
    monkeypatch: pytest.MonkeyPatch, failure_code: str,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "60")
    marker = "[gptpro-transport-nonce:uncertain]"
    provider_calls = 0
    observations: list[ask.AskEvidence] = []
    failure = ask.GptProAskError(failure_code, "confirmation lost")
    failure.evidence = ask.AskEvidence(
        submission="uncertain", conversation_id=_CONVERSATION_ID,
        generation_observed=failure_code == "timeout",
        answer_seen=failure_code == "timeout",
        failure_stage="answer" if failure_code == "timeout" else "submission",
    )

    async def execute_ask_outcome(
        page: _FakePage, question: str, *,
        callbacks: ask.AskCallbacks | None = None,
        **_kwargs: object,
    ) -> ask.AskOutcome:
        nonlocal provider_calls
        del page, question
        provider_calls += 1
        assert callbacks is not None
        assert callbacks.on_conversation_id is not None
        assert callbacks.on_marker is not None
        assert callbacks.on_evidence is not None
        callbacks.on_marker(marker)
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_evidence(failure.evidence)
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.ask(
            "question", callbacks=ask.AskCallbacks(on_evidence=observations.append),
        ))
        while poller.future is None and not task.done():
            await asyncio.sleep(0)
        assert not task.done()
        assert poller.registrations[0][:2] == (_CONVERSATION_ID, marker)
        assert observations[-1].recovery == "polling"
        poller.future.set_result(ask.AskOutcome("server answer", marker, _CONVERSATION_ID))
        assert (await task).text == "server answer"
        assert observations[-1].recovery == "recovered"
        assert observations[-1].submission == "confirmed"
        assert observations[-1].raw_extracted is True
        assert provider_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_runtime_never_polls_pre_submit_timeout_even_with_known_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    marker = "[gptpro-transport-nonce:never-sent]"
    failure = ask.GptProAskError("timeout", "composer timed out")
    failure.evidence = ask.AskEvidence(
        submission="not_attempted", conversation_id=_CONVERSATION_ID,
        failure_stage="composer",
    )

    async def execute_ask_outcome(
        page: _FakePage, question: str, *,
        callbacks: ask.AskCallbacks | None = None,
        **_kwargs: object,
    ) -> ask.AskOutcome:
        del page, question
        assert callbacks is not None
        assert callbacks.on_marker is not None
        callbacks.on_marker(marker)
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute_ask_outcome)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.ask(
                "question", conversation_id=_CONVERSATION_ID,
                callbacks=ask.AskCallbacks(),
            )
        assert raised.value is failure
        assert raised.value.evidence.failure_stage == "composer"
        assert poller.registrations == []
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_poller_does_not_resolve_on_finished_commentary_before_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:current]"
    commentary = _detached_conversation(marker, "Checking sources.")
    progress = commentary["mapping"]["assistant"]["message"]  # type: ignore[index]
    progress["channel"] = "commentary"
    progress["end_turn"] = False
    final = _detached_conversation(marker, "actual final answer")
    final["mapping"]["assistant"]["message"]["channel"] = "final"  # type: ignore[index]
    page = _PollerFakePage(marker, [(200, commentary), (200, final)])
    context = _PollerFakeContext(page)

    async def no_delay(seconds: float) -> None:
        return None

    monkeypatch.setattr(runtime, "_sleep", no_delay)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(
            _CONVERSATION_ID, marker, deadline=runtime._monotonic() + 100,
        )
        answer = await future
        assert answer.text == "actual final answer"
        await poller.aclose()

    asyncio.run(scenario())


def test_poller_waits_for_correct_finished_turn_across_other_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "[gptpro-transport-nonce:current]"
    other = "[gptpro-transport-nonce:other]"
    unfinished = _detached_conversation(marker, "still generating")
    assistant = unfinished["mapping"]["assistant"]["message"]  # type: ignore[index]
    assistant["status"] = "in_progress"
    assistant["end_turn"] = False
    page = _PollerFakePage(marker, [
        (200, _detached_conversation(other, "unrelated answer")),
        (200, unfinished),
        (200, _detached_conversation(marker, "actual final answer")),
    ])
    context = _PollerFakeContext(page)
    observed_pauses: list[float] = []

    async def no_delay(seconds: float) -> None:
        observed_pauses.append(seconds)

    monkeypatch.setattr(runtime, "_sleep", no_delay)

    async def scenario() -> None:
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        future = poller.register(
            _CONVERSATION_ID, marker, deadline=runtime._monotonic() + 100,
        )
        answer = await future
        assert answer.text == "actual final answer"
        assert len(observed_pauses) == 2
        assert page.fetch_arguments[-1]["url"].endswith(_CONVERSATION_ID)
        await poller.aclose()

    asyncio.run(scenario())


def test_recover_waits_for_file_delivery_after_the_recovery_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "0.05")
    marker = "[gptpro-transport-nonce:file-delivery]"

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.recover(_CONVERSATION_ID, marker))
        while poller.future is None:
            await asyncio.sleep(0)
        # The finished answer was found; its files are still downloading.
        await asyncio.sleep(0.1)
        assert not task.done(), "recovery gave up while files were downloading"
        outcome = ask.AskOutcome(
            text="recovered", marker=marker, conversation_id=_CONVERSATION_ID
        )
        poller.future.set_result(outcome)
        assert await task == outcome
        await ask_runtime.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("configured", [None, "inf", "nan", "999999"])
def test_recovery_configuration_has_finite_ninety_minute_cap(
    monkeypatch: pytest.MonkeyPatch, configured: str | None,
) -> None:
    if configured is None:
        monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)
    else:
        monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", configured)
    assert runtime.raw_turn_recovery_seconds() == 5400.0


@pytest.mark.parametrize("answer_at", [1.0, 3600.0, 5340.0, 5400.0, 5401.0])
def test_shared_deadline_survives_short_initial_watchdog(
    monkeypatch: pytest.MonkeyPatch, answer_at: float,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)
    now = 100.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    submissions: list[str] = []

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        nonlocal now
        submissions.append(question)
        callbacks = options["callbacks"]
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker("exact-marker")
        assert options["timeout_seconds"] == 30.0
        now += min(30.0, answer_at / 2)
        raise ask.GptProAskError("no_raw_turn", "answer not available")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)

    async def scenario() -> None:
        nonlocal now
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.ask(
            "review", timeout_seconds=30.0, callbacks=ask.AskCallbacks(),
        ))
        while poller.future is None:
            await asyncio.sleep(0)
        assert poller.registrations == [(_CONVERSATION_ID, "exact-marker", 5500.0)]
        assert ask_runtime._ask_semaphore._value == 2
        now = 100.0 + answer_at
        poller.future.set_result(_outcome("finished answer"))
        if answer_at >= 5400:
            with pytest.raises(ask.GptProAskError):
                await task
        else:
            assert (await task).text == "finished answer"
        assert submissions == ["review"]
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_capacity_queue_does_not_consume_shared_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    now = 0.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        callbacks = options["callbacks"]
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker("queued-marker")
        raise ask.GptProAskError("no_raw_turn", "answer not available")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)

    async def scenario() -> None:
        nonlocal now
        ask_runtime = runtime.AskRuntime(max_concurrent_asks=1)
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        await ask_runtime._ask_semaphore.acquire()
        task = asyncio.create_task(ask_runtime.ask("queued", callbacks=ask.AskCallbacks()))
        while ask_runtime._waiting_submitters != 1:
            await asyncio.sleep(0)
        now = 7200.0
        ask_runtime._ask_semaphore.release()
        while poller.future is None:
            await asyncio.sleep(0)
        assert poller.registrations[0][2] == 12600.0
        poller.future.set_result(_outcome("ready"))
        assert (await task).text == "ready"
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_initial_result_and_file_delivery_cannot_overrun_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    now = 0.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        nonlocal now
        now = 5400.0
        return _outcome("files finished too late")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.ask("review")
        assert raised.value.failure == "timeout"
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_manual_recovery_starts_new_window_and_rejects_late_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 7200.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)

    async def scenario() -> None:
        nonlocal now
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.recover(_CONVERSATION_ID, "source-marker"))
        while poller.future is None:
            await asyncio.sleep(0)
        assert poller.registrations == [(_CONVERSATION_ID, "source-marker", 12600.0)]
        now = 12600.0
        poller.future.set_result(_outcome("late files"))
        with pytest.raises(ask.GptProAskError) as raised:
            await task
        assert raised.value.failure == "timeout"
        await ask_runtime.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("delivery_at", [5399.0, 5400.0, 5401.0])
def test_detached_generated_files_share_total_deadline(
    monkeypatch: pytest.MonkeyPatch, delivery_at: float,
) -> None:
    now = 0.0
    marker = "[gptpro-transport-nonce:files-deadline]"
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)

    def answer_ready() -> None:
        nonlocal now
        now = 5390.0

    page = _PollerFakePage(marker, [(200, _detached_conversation(
        marker, "Finished [file](sandbox:/mnt/data/answer.txt)",
    ))], on_conversation_fetch=answer_ready)
    context = _PollerFakeContext(page)
    collections: list[Any] = []

    async def collect(
        evaluate: Any, conversation_id: str, references: Any,
    ) -> tuple[tuple[Any, ...], bool]:
        nonlocal now
        assert conversation_id == _CONVERSATION_ID
        assert len(references) == 1
        collections.append(references)
        now = delivery_at
        # A failed optional download still permits a timely finished answer.
        return (), False

    monkeypatch.setattr(runtime.generated_files, "collect_generated_files", collect)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        ask_runtime._poller = poller
        if delivery_at < 5400:
            outcome = await ask_runtime.recover(_CONVERSATION_ID, marker)
            assert outcome.text.startswith("Finished")
            assert outcome.files_complete is False
        else:
            with pytest.raises(ask.GptProAskError) as raised:
                await ask_runtime.recover(_CONVERSATION_ID, marker)
            assert raised.value.failure == "timeout"
        await _wait_for_poller_idle(poller)
        assert not poller._registrations
        assert not poller._deliveries
        assert len(collections) == 1
        assert page.close_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_shared_deadline_without_answer_cleans_poller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    marker = "[gptpro-transport-nonce:no-answer-deadline]"
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    monkeypatch.delenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", raising=False)

    def exhaust_execution() -> None:
        nonlocal now
        now = 5400.0

    page = _PollerFakePage(marker, [(200, _detached_conversation(
        "another-marker", "unrelated answer",
    ))], on_conversation_fetch=exhaust_execution)
    context = _PollerFakeContext(page)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        poller = runtime.DetachPoller(lambda: asyncio.sleep(0, result=context))
        ask_runtime._poller = poller
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.recover(_CONVERSATION_ID, marker)
        assert raised.value.failure == "timeout"
        await _wait_for_poller_idle(poller)
        assert not poller._registrations
        assert not poller._deliveries
        assert page.close_calls == 1
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_total_deadline_does_not_wait_for_resistant_browser_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    now = 0.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    original_wait = asyncio.wait
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        try:
            await asyncio.Future()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()

    async def simulated_wait(
        futures: set[asyncio.Future[Any]], *, timeout: float,
    ) -> tuple[set[asyncio.Future[Any]], set[asyncio.Future[Any]]]:
        nonlocal now
        # Let jitter/context finish normally, but expire the browser execution.
        future = next(iter(futures))
        await asyncio.sleep(0)
        if not future.done():
            assert timeout == 5400.0
            now = 5400.0
            return set(), set(futures)
        return await original_wait(futures, timeout=timeout)

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)
    monkeypatch.setattr(runtime.asyncio, "wait", simulated_wait)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(ask.GptProAskError) as raised:
            await ask_runtime.ask("review")
        assert raised.value.failure == "timeout"
        assert ask_runtime._ask_semaphore._value == 2
        await cleanup_started.wait()
        assert not release_cleanup.is_set()
        release_cleanup.set()
        while not context.pages[0].close_calls:
            await asyncio.sleep(0)
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_initial_detach_uses_total_not_browser_stage_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    now = 100.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        assert options["timeout_seconds"] == 30.0
        return await options["on_detach"](ask.AskSubmission(
            marker="detached-marker", conversation_id=_CONVERSATION_ID,
        ))

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)

    async def scenario() -> None:
        nonlocal now
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.ask("review", timeout_seconds=30.0))
        while poller.future is None:
            await asyncio.sleep(0)
        assert poller.registrations == [(_CONVERSATION_ID, "detached-marker", 5500.0)]
        assert context.pages[0].close_calls == 1
        assert ask_runtime._ask_semaphore._value == 2
        now = 5440.0
        poller.future.set_result(_outcome("89-minute answer"))
        assert (await task).text == "89-minute answer"
        await ask_runtime.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_seconds", [60.0, 5390.0])
def test_short_auto_recovery_file_allowance_cannot_extend_total(
    monkeypatch: pytest.MonkeyPatch, initial_seconds: float,
) -> None:
    context = _FakeContext()
    _RuntimeFakes(monkeypatch, [context])
    now = 0.0
    monkeypatch.setattr(runtime, "_monotonic", lambda: now)
    monkeypatch.setenv("GPTPRO_RAW_TURN_RECOVERY_SECONDS", "120")

    async def execute(
        page: _FakePage, question: str, **options: Any,
    ) -> ask.AskOutcome:
        nonlocal now
        callbacks = options["callbacks"]
        callbacks.on_conversation_id(_CONVERSATION_ID)
        callbacks.on_marker("short-window-marker")
        now = initial_seconds
        raise ask.GptProAskError("no_raw_turn", "answer not available")

    monkeypatch.setattr(runtime.ask, "execute_ask_outcome", execute)

    async def scenario() -> None:
        nonlocal now
        ask_runtime = runtime.AskRuntime()
        poller = _RuntimeDetachPollerFake(ask_runtime._get_context)
        ask_runtime._poller = poller
        task = asyncio.create_task(ask_runtime.ask("review", callbacks=ask.AskCallbacks()))
        while poller.future is None:
            await asyncio.sleep(0)
        polling_deadline = min(5400.0, initial_seconds + 120.0)
        delivery_deadline = min(
            5400.0, polling_deadline + runtime.generated_files.MAX_COLLECTION_SECONDS,
        )
        assert poller.registrations[0][2] == polling_deadline
        assert poller.delivery_deadlines == [delivery_deadline]
        # A finished answer was found in the polling window, but delivery is late.
        now = delivery_deadline
        poller.future.set_result(_outcome("late files"))
        with pytest.raises(ask.GptProAskError) as raised:
            await task
        assert raised.value.failure == "no_raw_turn"
        assert raised.value.evidence.recovery_failure == "timeout"
        await ask_runtime.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [0, -1])
def test_runtime_rejects_nonpositive_concurrency(limit: int) -> None:
    with pytest.raises(ValueError, match="max_concurrent_asks"):
        runtime.AskRuntime(max_concurrent_asks=limit)
