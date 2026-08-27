"""Tests for the background gptpro ask job lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from claudex.gptpro import ask, jobs


async def _wait_for_state(
    service: jobs.AskJobService,
    ask_id: str,
    expected_state: jobs.AskJobState,
) -> jobs.AskJob:
    deadline = asyncio.get_running_loop().time() + 1.0
    while asyncio.get_running_loop().time() < deadline:
        job = service.status(ask_id)
        if job is not None and job.state == expected_state:
            return job
        await asyncio.sleep(0)
    raise AssertionError(f"job did not reach {expected_state}")


async def _wait_for_status_message(
    service: jobs.AskJobService,
    ask_id: str,
    expected_message: str,
) -> jobs.AskJob:
    deadline = asyncio.get_running_loop().time() + 1.0
    while asyncio.get_running_loop().time() < deadline:
        job = service.status(ask_id)
        if job is not None and job.status_message == expected_message:
            return job
        await asyncio.sleep(0)
    raise AssertionError(f"job did not report {expected_message!r}")


async def _wait_until_missing(
    service: jobs.AskJobService, ask_id: str
) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while asyncio.get_running_loop().time() < deadline:
        if service.status(ask_id) is None:
            return
        await asyncio.sleep(0.002)
    raise AssertionError("job was not removed")


def test_start_returns_running_job_before_successful_completion() -> None:
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    captured_thread_refs: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del on_status, on_conversation_id
        assert question == "question"
        provider_started.set()
        await release_provider.wait()
        return ask.AskOutcome(
            text="answer",
            marker="full nonce marker",
            conversation_id="conversation-from-outcome",
        )

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start(
            "question", on_thread_ref=captured_thread_refs.append
        )

        assert started_job.state == "running"
        assert started_job.answer is None
        assert started_job.thread_ref is None
        assert not provider_started.is_set()

        await provider_started.wait()
        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )

        assert completed_job.answer == "answer"
        assert completed_job.nonce_marker == "full nonce marker"
        assert completed_job.thread_ref == "conversation-from-outcome"
        assert captured_thread_refs == ["conversation-from-outcome"]
        assert completed_job.finished_at is not None
        assert service.result(started_job.ask_id) == completed_job
        assert started_job.state == "running"
        await service.aclose()

    asyncio.run(scenario())


def test_marker_callback_updates_running_job_and_preserves_success_marker() -> None:
    marker_updated = asyncio.Event()
    release_provider = asyncio.Event()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        del conversation_id, timeout_seconds
        assert on_marker is not None
        on_marker("full nonce marker")
        marker_updated.set()
        await release_provider.wait()
        return ask.AskOutcome("answer", "full nonce marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")

        await marker_updated.wait()
        running_job = service.status(started_job.ask_id)
        assert running_job is not None
        assert running_job.state == "running"
        assert running_job.nonce_marker == "full nonce marker"

        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.nonce_marker == "full nonce marker"
        await service.aclose()

    asyncio.run(scenario())


def test_marker_callback_preserves_marker_on_failure() -> None:
    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        del conversation_id, timeout_seconds
        assert on_marker is not None
        on_marker("failed nonce marker")
        raise ask.GptProAskError("error", "provider failed")

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")
        failed_job = await _wait_for_state(service, started_job.ask_id, "failed")

        assert failed_job.nonce_marker == "failed nonce marker"
        await service.aclose()

    asyncio.run(scenario())


def test_status_callback_updates_latest_status_message() -> None:
    status_updated = asyncio.Event()
    release_provider = asyncio.Event()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del question, on_conversation_id
        assert on_status is not None
        on_status("opening ChatGPT")
        on_status("waiting for answer")
        status_updated.set()
        await release_provider.wait()
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")

        await status_updated.wait()
        current_job = service.status(started_job.ask_id)
        assert current_job is not None
        assert current_job.status_message == "waiting for answer"

        release_provider.set()
        await _wait_for_state(service, started_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())


def test_conversation_id_callback_latches_first_thread_ref_while_running() -> None:
    conversation_id_updated = asyncio.Event()
    release_provider = asyncio.Event()
    captured_thread_refs: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del question, on_status
        assert on_conversation_id is not None
        on_conversation_id("first-conversation")
        on_conversation_id("second-conversation")
        conversation_id_updated.set()
        await release_provider.wait()
        return ask.AskOutcome("answer", "marker", "outcome-conversation")

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start(
            "question", on_thread_ref=captured_thread_refs.append
        )

        await conversation_id_updated.wait()
        running_job = service.status(started_job.ask_id)
        assert running_job is not None
        assert running_job.state == "running"
        assert running_job.thread_ref == "first-conversation"
        assert captured_thread_refs == ["first-conversation"]

        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.thread_ref == "first-conversation"
        await service.aclose()

    asyncio.run(scenario())


def test_existing_conversation_is_visible_and_passed_to_provider() -> None:
    captured_options: list[tuple[str | None, float | None]] = []
    captured_thread_refs: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        captured_options.append((conversation_id, timeout_seconds))
        return ask.AskOutcome("answer", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(provider, overall_timeout_seconds=5.0)
        started_job = service.start(
            "question",
            conversation_id="existing-conversation",
            on_thread_ref=captured_thread_refs.append,
        )

        assert started_job.thread_ref == "existing-conversation"
        assert captured_thread_refs == []

        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.thread_ref == "existing-conversation"
        assert captured_thread_refs == []
        assert len(captured_options) == 1
        captured_conversation_id, captured_timeout = captured_options[0]
        assert captured_conversation_id == "existing-conversation"
        assert captured_timeout is not None
        assert 0 < captured_timeout <= 5.0
        await service.aclose()

    asyncio.run(scenario())


def test_same_conversation_asks_run_in_submission_order() -> None:
    first_provider_started = asyncio.Event()
    release_first_provider = asyncio.Event()
    second_provider_started = asyncio.Event()
    started_questions: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del on_status, on_conversation_id, timeout_seconds
        assert conversation_id == "shared-conversation"
        started_questions.append(question)
        if question == "first":
            first_provider_started.set()
            await release_first_provider.wait()
        else:
            second_provider_started.set()
        return ask.AskOutcome(f"answer: {question}", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        first_job = service.start(
            "first", conversation_id="shared-conversation"
        )
        await first_provider_started.wait()

        second_job = service.start(
            "second", conversation_id="shared-conversation"
        )
        waiting_job = await _wait_for_status_message(
            service,
            second_job.ask_id,
            "waiting for the in-flight answer",
        )
        assert waiting_job.state == "running"
        assert not second_provider_started.is_set()
        assert started_questions == ["first"]

        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await second_provider_started.wait()
        await _wait_for_state(service, second_job.ask_id, "succeeded")
        assert started_questions == ["first", "second"]
        await service.aclose()

    asyncio.run(scenario())


def test_different_conversation_asks_run_concurrently() -> None:
    both_providers_started = asyncio.Event()
    release_providers = asyncio.Event()
    started_conversations: set[str | None] = set()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id, timeout_seconds
        started_conversations.add(conversation_id)
        if len(started_conversations) == 2:
            both_providers_started.set()
        await release_providers.wait()
        return ask.AskOutcome("answer", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        first_job = service.start("first", conversation_id="conversation-a")
        second_job = service.start("second", conversation_id="conversation-b")

        await both_providers_started.wait()
        assert started_conversations == {"conversation-a", "conversation-b"}

        release_providers.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await _wait_for_state(service, second_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())


def test_same_conversation_wait_expires_with_timeout_failure() -> None:
    first_provider_started = asyncio.Event()
    release_first_provider = asyncio.Event()
    provider_questions: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del on_status, on_conversation_id, timeout_seconds
        assert conversation_id == "shared-conversation"
        provider_questions.append(question)
        if question == "first":
            first_provider_started.set()
            await release_first_provider.wait()
        return ask.AskOutcome("answer", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, overall_timeout_seconds=0.01
        )
        first_job = service.start(
            "first", conversation_id="shared-conversation"
        )
        await first_provider_started.wait()
        second_job = service.start(
            "second", conversation_id="shared-conversation"
        )

        failed_job = await _wait_for_state(
            service, second_job.ask_id, "failed"
        )
        assert failed_job.failure == "timeout"
        assert failed_job.error_message == (
            "the ask deadline expired while waiting for the in-flight answer "
            "on this conversation"
        )
        assert provider_questions == ["first"]

        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())


def test_service_uses_environment_overall_timeout_for_start_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPTPRO_OVERALL_TIMEOUT_SECONDS", "14400")
    captured_timeouts: list[float | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id, conversation_id
        captured_timeouts.append(timeout_seconds)
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider, clock=lambda: 100.0)
        started_job = service.start("question")
        await _wait_for_state(service, started_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert captured_timeouts == [14_400.0]


def test_provider_receives_remaining_overall_timeout() -> None:
    captured_timeouts: list[float | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id, conversation_id
        captured_timeouts.append(timeout_seconds)
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, overall_timeout_seconds=12.5
        )
        started_job = service.start("question")
        await _wait_for_state(service, started_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert len(captured_timeouts) == 1
    captured_timeout = captured_timeouts[0]
    assert captured_timeout is not None
    assert 0 < captured_timeout <= 12.5


@pytest.mark.parametrize("should_fail", [False, True], ids=["success", "failure"])
def test_oversized_question_spills_to_temporary_attachment_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    should_fail: bool,
) -> None:
    monkeypatch.setattr(jobs, "QUESTION_SPILL_THRESHOLD_BYTES", 8)
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    captured_questions: list[str] = []
    captured_attachment_paths: list[Sequence[str] | None] = []
    original_question = "price: €"

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del on_status, on_conversation_id, conversation_id, timeout_seconds
        captured_questions.append(question)
        captured_attachment_paths.append(attachment_paths)
        provider_started.set()
        await release_provider.wait()
        if should_fail:
            raise RuntimeError("provider failed")
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start(original_question)

        await provider_started.wait()
        attachment_paths = captured_attachment_paths[0]
        assert attachment_paths is not None
        assert len(attachment_paths) == 1
        spill_path = Path(attachment_paths[0])
        assert spill_path.name.startswith("gptpro-spill-")
        assert spill_path.suffix == ".txt"
        assert spill_path.exists()
        assert spill_path.read_text(encoding="utf-8") == original_question
        assert captured_questions == [
            "The full question text is attached as "
            f"{spill_path.name}; read the attachment and answer it."
        ]

        release_provider.set()
        expected_state: jobs.AskJobState = (
            "failed" if should_fail else "succeeded"
        )
        await _wait_for_state(service, started_job.ask_id, expected_state)
        assert not spill_path.exists()
        await service.aclose()

    asyncio.run(scenario())


def test_question_at_spill_threshold_is_sent_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "QUESTION_SPILL_THRESHOLD_BYTES", 10)
    captured_questions: list[str] = []
    captured_attachment_paths: list[Sequence[str] | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del on_status, on_conversation_id, conversation_id, timeout_seconds
        captured_questions.append(question)
        captured_attachment_paths.append(attachment_paths)
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        question = "price: €"
        started_job = service.start(question)
        await _wait_for_state(service, started_job.ask_id, "succeeded")
        assert captured_questions == [question]
        assert captured_attachment_paths == [None]
        await service.aclose()

    asyncio.run(scenario())


def test_explicit_attachment_paths_are_passed_unchanged() -> None:
    captured_attachment_paths: list[Sequence[str] | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id, conversation_id
        del timeout_seconds
        captured_attachment_paths.append(attachment_paths)
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        explicit_attachment_paths = ["notes.txt", "context.txt"]
        started_job = service.start(
            "question", attachment_paths=explicit_attachment_paths
        )
        await _wait_for_state(service, started_job.ask_id, "succeeded")
        assert captured_attachment_paths == [explicit_attachment_paths]
        assert captured_attachment_paths[0] is explicit_attachment_paths
        await service.aclose()

    asyncio.run(scenario())


def test_thread_registry_returns_bound_thread_and_none_for_unknown_session() -> None:
    registry = jobs.ThreadRegistry()

    assert registry.lookup("unknown-session") is None
    registry.bind("session", "thread")
    assert registry.lookup("session") == "thread"


def test_thread_registry_removes_expired_binding_on_lookup() -> None:
    now = [100.0]
    registry = jobs.ThreadRegistry(clock=lambda: now[0])
    registry.bind("session", "thread")

    now[0] += jobs.THREAD_BINDING_TTL_SECONDS

    assert registry.lookup("session") is None


@pytest.mark.parametrize(
    ("provider_error", "expected_failure", "expected_message"),
    [
        (
            ask.GptProAskError("timeout", "deadline expired"),
            "timeout",
            "deadline expired",
        ),
        (RuntimeError("browser crashed"), "error", "RuntimeError: browser crashed"),
    ],
)
def test_provider_failure_transitions_job_to_failed(
    provider_error: Exception,
    expected_failure: str,
    expected_message: str,
) -> None:
    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del question, on_status, on_conversation_id
        raise provider_error

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")
        failed_job = await _wait_for_state(service, started_job.ask_id, "failed")

        assert failed_job.failure == expected_failure
        assert failed_job.error_message == expected_message
        assert failed_job.finished_at is not None
        await service.aclose()

    asyncio.run(scenario())


def test_unknown_ask_id_has_no_status_or_result() -> None:
    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del question, on_status, on_conversation_id
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        assert service.status("unknown") is None
        assert service.result("unknown") is None
        await service.aclose()

    asyncio.run(scenario())


def test_sweeper_removes_expired_finished_jobs_but_keeps_running_jobs() -> None:
    running_provider_started = asyncio.Event()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del on_status, on_conversation_id
        if question == "running":
            running_provider_started.set()
            await asyncio.Event().wait()
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider,
            retention_seconds=0.01,
            sweep_interval_seconds=0.005,
        )
        finished_job = service.start("finished")
        running_job = service.start("running")

        await running_provider_started.wait()
        await _wait_for_state(service, finished_job.ask_id, "succeeded")
        await _wait_until_missing(service, finished_job.ask_id)

        retained_job = service.status(running_job.ask_id)
        assert retained_job is not None
        assert retained_job.state == "running"
        await service.aclose()

    asyncio.run(scenario())


def test_aclose_cancels_running_job_and_sweeper_and_is_idempotent() -> None:
    provider_started = asyncio.Event()
    provider_cancelled = asyncio.Event()
    sweeper_started = asyncio.Event()
    sweeper_cancelled = asyncio.Event()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del conversation_id, timeout_seconds
        del question, on_status, on_conversation_id
        provider_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            provider_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def sleep(seconds: float) -> None:
        assert seconds == 300.0
        sweeper_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sweeper_cancelled.set()
            raise

    async def scenario() -> None:
        service = jobs.AskJobService(provider, sleep=sleep)
        service.start("question")
        await provider_started.wait()
        await sweeper_started.wait()

        await service.aclose()
        assert provider_cancelled.is_set()
        assert sweeper_cancelled.is_set()
        await service.aclose()

    asyncio.run(scenario())
