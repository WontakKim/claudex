"""Tests for the background gptpro ask job lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

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

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
    ) -> ask.AskOutcome:
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
        started_job = service.start("question")

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
        assert completed_job.finished_at is not None
        assert service.result(started_job.ask_id) == completed_job
        assert started_job.state == "running"
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
    ) -> ask.AskOutcome:
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

    async def provider(
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
    ) -> ask.AskOutcome:
        del question, on_status
        assert on_conversation_id is not None
        on_conversation_id("first-conversation")
        on_conversation_id("second-conversation")
        conversation_id_updated.set()
        await release_provider.wait()
        return ask.AskOutcome("answer", "marker", "outcome-conversation")

    async def scenario() -> None:
        service = jobs.AskJobService(provider)
        started_job = service.start("question")

        await conversation_id_updated.wait()
        running_job = service.status(started_job.ask_id)
        assert running_job is not None
        assert running_job.state == "running"
        assert running_job.thread_ref == "first-conversation"

        release_provider.set()
        completed_job = await _wait_for_state(
            service, started_job.ask_id, "succeeded"
        )
        assert completed_job.thread_ref == "first-conversation"
        await service.aclose()

    asyncio.run(scenario())


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
    ) -> ask.AskOutcome:
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
    ) -> ask.AskOutcome:
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
    ) -> ask.AskOutcome:
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
    ) -> ask.AskOutcome:
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
