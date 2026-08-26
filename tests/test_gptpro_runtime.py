"""Tests for the warm gptpro ask runtime lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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
    def __init__(self, *, new_page_error: BaseException | None = None) -> None:
        self.new_page_error = new_page_error
        self.pages: list[_FakePage] = []
        self.close_calls = 0

    async def new_page(self) -> _FakePage:
        if self.new_page_error is not None:
            raise self.new_page_error
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


def test_runtime_initializes_lazily_and_reuses_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, on_status, deadline
        return f"answer: {question}"

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert fakes.launch_calls == []
        assert await ask_runtime.ask("first") == "answer: first"
        assert await ask_runtime.ask("second") == "answer: second"
        assert len(fakes.launch_calls) == 1
        assert fakes.launch_calls[0][1] is True
        assert len(context.pages) == 2
        assert all(page.close_calls == 1 for page in context.pages)
        await ask_runtime.aclose()

    asyncio.run(scenario())


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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        nonlocal active, maximum_active
        del page, on_status, deadline
        active += 1
        maximum_active = max(maximum_active, active)
        started.append(question)
        if len(started) == 2:
            two_started.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return question

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)
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
        assert await asyncio.gather(*tasks) == ["one", "two", "three"]
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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, on_status, deadline
        return question

    monkeypatch.setattr(runtime.random, "uniform", uniform)
    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert await ask_runtime.ask("question") == "question"
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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, question, on_status, deadline
        raise failure

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        nonlocal calls
        del page, on_status, deadline
        calls += 1
        if calls == 1:
            closed_error = RuntimeError(
                "Target page, context or browser has been closed"
            )
            raise ask.GptProAskError(
                "error", "the ChatGPT ask failed unexpectedly"
            ) from closed_error
        return f"answer: {question}"

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        with pytest.raises(ask.GptProAskError):
            await ask_runtime.ask("first")
        assert first_context.close_calls == 1
        assert fakes.locks[0].release_calls == 1

        assert await ask_runtime.ask("second") == "answer: second"
        assert len(fakes.launch_calls) == 2
        await ask_runtime.aclose()

    asyncio.run(scenario())


def test_aclose_cleans_up_playwright_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    fakes = _RuntimeFakes(monkeypatch, [context])

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, on_status, deadline
        return question

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, on_status, deadline
        return question

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)

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

    async def execute_ask(
        page: _FakePage,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        deadline: float | None = None,
    ) -> str:
        del page, on_status, deadline
        return f"answer: {question}"

    monkeypatch.setattr(runtime.ask, "execute_ask", execute_ask)


def test_ask_hardens_headless_user_agent_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _HeadlessUserAgentContext(
        "Mozilla/5.0 HeadlessChrome/151.0.0.0"
    )
    _install_ask_stub(monkeypatch, context)

    async def scenario() -> None:
        ask_runtime = runtime.AskRuntime()
        assert await ask_runtime.ask("question") == "answer: question"
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
        assert await ask_runtime.ask("question") == "answer: question"
        await ask_runtime.aclose()

    asyncio.run(scenario())

    assert context.pages[0].extra_headers is None
