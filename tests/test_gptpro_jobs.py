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


def test_provider_without_detached_callback_transitions_through_lifecycle() -> None:
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

        assert started_job.state == "queued"
        assert started_job.answer is None
        assert started_job.thread_ref is None
        assert not provider_started.is_set()

        await provider_started.wait()
        running_job = service.status(started_job.ask_id)
        assert running_job is not None
        assert running_job.state == "running"

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
        assert started_job.state == "queued"
        await service.aclose()

    asyncio.run(scenario())


def test_detached_callback_transitions_job_then_completion_succeeds() -> None:
    detached = asyncio.Event()
    release_provider = asyncio.Event()

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_detached: Callable[[], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        del conversation_id, timeout_seconds
        assert on_detached is not None
        on_detached()
        detached.set()
        await release_provider.wait()
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")

        await detached.wait()
        detached_job = service.status(started_job.ask_id)
        assert detached_job is not None
        assert detached_job.state == "detached"
        assert detached_job.finished_at is None

        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.answer == "answer"
        assert completed_job.finished_at is not None
        await service.aclose()

    asyncio.run(scenario())


def test_late_detached_callback_does_not_change_completed_job() -> None:
    captured_callbacks: list[Callable[[], None]] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_detached: Callable[[], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        del conversation_id, timeout_seconds
        assert on_detached is not None
        captured_callbacks.append(on_detached)
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )

        assert len(captured_callbacks) == 1
        captured_callbacks[0]()
        assert service.status(started_job.ask_id) == completed_job
        await service.aclose()

    asyncio.run(scenario())


def test_success_emits_turn_finished_once_even_if_callback_raises() -> None:
    captured_events: list[jobs.TurnFinished] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, on_conversation_id
        del conversation_id, timeout_seconds
        return ask.AskOutcome("answer", "marker", None)

    def capture_turn_finished(event: jobs.TurnFinished) -> None:
        captured_events.append(event)
        raise RuntimeError("subscriber failed")

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, on_turn_finished=capture_turn_finished
        )
        started_job = service.start("question")
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )

        assert completed_job.answer == "answer"
        assert captured_events == [
            jobs.TurnFinished(
                ask_id=started_job.ask_id,
                thread_ref=None,
                answer="answer",
            )
        ]
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


def test_conversation_id_latches_thread_ref_but_notifies_on_success() -> None:
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
        assert captured_thread_refs == []

        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.thread_ref == "first-conversation"
        assert captured_thread_refs == ["first-conversation"]
        await service.aclose()

    asyncio.run(scenario())


def test_failed_job_does_not_notify_latched_thread_ref() -> None:
    captured_thread_refs: list[str] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del question, on_status, conversation_id, timeout_seconds
        assert on_conversation_id is not None
        on_conversation_id("failed-conversation")
        raise ask.GptProAskError("error", "provider failed")

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start(
            "question", on_thread_ref=captured_thread_refs.append
        )
        failed_job = await _wait_for_state(service, started_job.ask_id, "failed")

        assert failed_job.thread_ref == "failed-conversation"
        assert captured_thread_refs == []
        await service.aclose()

    asyncio.run(scenario())


def test_existing_conversation_notifies_thread_ref_on_success() -> None:
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
        assert captured_thread_refs == ["existing-conversation"]
        assert len(captured_options) == 1
        captured_conversation_id, captured_timeout = captured_options[0]
        assert captured_conversation_id == "existing-conversation"
        assert captured_timeout is not None
        assert 0 < captured_timeout <= 5.0
        await service.aclose()

    asyncio.run(scenario())


def test_same_conversation_queued_ask_runs_in_submission_order() -> None:
    first_provider_started = asyncio.Event()
    release_first_provider = asyncio.Event()
    second_provider_started = asyncio.Event()
    release_second_provider = asyncio.Event()
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
            await release_second_provider.wait()
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
        assert second_job.state == "queued"
        assert waiting_job.state == "queued"
        assert not second_provider_started.is_set()
        assert started_questions == ["first"]

        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await second_provider_started.wait()
        admitted_job = service.status(second_job.ask_id)
        assert admitted_job is not None
        assert admitted_job.state == "running"
        release_second_provider.set()
        await _wait_for_state(service, second_job.ask_id, "succeeded")
        assert started_questions == ["first", "second"]
        await service.aclose()

    asyncio.run(scenario())


def test_waiting_ask_runs_after_in_flight_ask_fails() -> None:
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
            raise ask.GptProAskError("error", "provider failed")
        second_provider_started.set()
        return ask.AskOutcome("second answer", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        first_job = service.start(
            "first", conversation_id="shared-conversation"
        )
        await first_provider_started.wait()
        second_job = service.start(
            "second", conversation_id="shared-conversation"
        )
        await _wait_for_status_message(
            service,
            second_job.ask_id,
            "waiting for the in-flight answer",
        )

        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "failed")
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


def test_queue_ttl_expires_waiting_ask_and_preserves_thread_ref() -> None:
    conversation_id = "123e4567-e89b-12d3-a456-426614174000"
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
        assert conversation_id == "123e4567-e89b-12d3-a456-426614174000"
        provider_questions.append(question)
        if question == "first":
            first_provider_started.set()
            await release_first_provider.wait()
        return ask.AskOutcome("answer", "marker", conversation_id)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider,
            overall_timeout_seconds=30.0,
            queue_ttl_seconds=0.01,
        )
        first_job = service.start("first", conversation_id=conversation_id)
        await first_provider_started.wait()
        second_job = service.start("second", conversation_id=conversation_id)

        failed_job = await _wait_for_state(
            service, second_job.ask_id, "failed"
        )
        assert failed_job.failure == "expired"
        assert failed_job.error_message == (
            "the queue TTL expired while waiting for the in-flight ask "
            "on this conversation"
        )
        assert failed_job.thread_ref == conversation_id
        assert provider_questions == ["first"]

        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())


def test_answer_watchdog_ignores_negative_duration_samples() -> None:
    watchdog = jobs.AnswerWatchdog(initial_budget_seconds=900.0)

    for _ in range(jobs.WATCHDOG_MIN_SAMPLES):
        watchdog.record(-1.0)

    assert watchdog.execution_budget_seconds() == 900.0


def test_successful_duration_samples_update_next_execution_budget() -> None:
    now = 0.0
    durations = [100.0, 200.0, 300.0, 400.0, 500.0, 0.0]
    captured_timeouts: list[float | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        nonlocal now
        del question, on_status, on_conversation_id, conversation_id
        captured_timeouts.append(timeout_seconds)
        now += durations[len(captured_timeouts) - 1]
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, overall_timeout_seconds=900.0, clock=lambda: now
        )
        for index in range(jobs.WATCHDOG_MIN_SAMPLES + 1):
            started_job = service.start(f"question {index}")
            await _wait_for_state(service, started_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert captured_timeouts[: jobs.WATCHDOG_MIN_SAMPLES] == [
        900.0
    ] * jobs.WATCHDOG_MIN_SAMPLES
    assert captured_timeouts[-1] == pytest.approx(720.0)


@pytest.mark.parametrize(
    ("duration_seconds", "expected_budget_seconds"),
    [(10.0, 60.0), (700.0, 900.0)],
    ids=["minimum", "initial-maximum"],
)
def test_answer_watchdog_clamps_measured_budget(
    duration_seconds: float, expected_budget_seconds: float
) -> None:
    watchdog = jobs.AnswerWatchdog(initial_budget_seconds=900.0)

    for _ in range(jobs.WATCHDOG_MIN_SAMPLES):
        watchdog.record(duration_seconds)

    assert watchdog.execution_budget_seconds() == expected_budget_seconds


def test_failed_job_does_not_update_answer_watchdog() -> None:
    now = 0.0
    captured_timeouts: list[float | None] = []
    watchdog = jobs.AnswerWatchdog(initial_budget_seconds=900.0)
    for duration_seconds in [100.0, 200.0, 300.0, 400.0, 500.0]:
        watchdog.record(duration_seconds)

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        nonlocal now
        del on_status, on_conversation_id, conversation_id
        captured_timeouts.append(timeout_seconds)
        if question == "failure":
            now += 2_000.0
            raise ask.GptProAskError("error", "provider failed")
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, watchdog=watchdog, clock=lambda: now
        )
        failed_job = service.start("failure")
        await _wait_for_state(service, failed_job.ask_id, "failed")
        successful_job = service.start("success")
        await _wait_for_state(service, successful_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert captured_timeouts == pytest.approx([720.0, 720.0])


def test_injected_overall_timeout_caps_measured_watchdog_budget() -> None:
    now = 0.0
    captured_timeouts: list[float | None] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        nonlocal now
        del question, on_status, on_conversation_id, conversation_id
        captured_timeouts.append(timeout_seconds)
        now += 300.0
        return ask.AskOutcome("answer", "marker", None)

    async def scenario() -> None:
        service = jobs.AskJobService(
            provider, overall_timeout_seconds=300.0, clock=lambda: now
        )
        for index in range(jobs.WATCHDOG_MIN_SAMPLES + 1):
            started_job = service.start(f"question {index}")
            await _wait_for_state(service, started_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert captured_timeouts == [300.0] * (jobs.WATCHDOG_MIN_SAMPLES + 1)


def test_service_uses_environment_overall_timeout_for_admission_budget(
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


def test_waiting_ask_receives_full_execution_budget_at_admission() -> None:
    now = 100.0
    first_provider_started = asyncio.Event()
    release_first_provider = asyncio.Event()
    captured_timeouts: list[tuple[str, float | None]] = []

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ask.AskOutcome:
        del on_status, on_conversation_id
        assert conversation_id == "shared-conversation"
        captured_timeouts.append((question, timeout_seconds))
        if question == "first":
            first_provider_started.set()
            await release_first_provider.wait()
        return ask.AskOutcome("answer", "marker", conversation_id)

    async def scenario() -> None:
        nonlocal now
        service = jobs.AskJobService(
            provider,
            overall_timeout_seconds=12.5,
            queue_ttl_seconds=20.0,
            clock=lambda: now,
        )
        first_job = service.start(
            "first", conversation_id="shared-conversation"
        )
        await first_provider_started.wait()
        second_job = service.start(
            "second", conversation_id="shared-conversation"
        )
        await _wait_for_status_message(
            service,
            second_job.ask_id,
            "waiting for the in-flight answer",
        )

        now += 10.0
        release_first_provider.set()
        await _wait_for_state(service, first_job.ask_id, "succeeded")
        await _wait_for_state(service, second_job.ask_id, "succeeded")
        await service.aclose()

    asyncio.run(scenario())

    assert captured_timeouts == [("first", 12.5), ("second", 12.5)]


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
