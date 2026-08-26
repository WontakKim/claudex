"""In-memory lifecycle registry for background ChatGPT Pro asks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, Protocol
from uuid import uuid4

from claudex.gptpro.ask import AskOutcome, GptProAskError

JOB_RETENTION_SECONDS = 24.0 * 60 * 60
SWEEP_INTERVAL_SECONDS = 300.0

AskJobState = Literal["running", "succeeded", "failed"]


@dataclass(frozen=True)
class AskJob:
    ask_id: str
    state: AskJobState
    answer: str | None
    failure: str | None
    error_message: str | None
    status_message: str | None
    nonce_marker: str | None
    thread_ref: str | None
    created_at: float
    finished_at: float | None


class _AskCallable(Protocol):
    def __call__(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
    ) -> Awaitable[AskOutcome]: ...


class AskJobService:
    """Run asks in the background and retain immutable lifecycle snapshots."""

    def __init__(
        self,
        ask: _AskCallable,
        *,
        retention_seconds: float = JOB_RETENTION_SECONDS,
        sweep_interval_seconds: float = SWEEP_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._ask = ask
        self._retention_seconds = retention_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._jobs: dict[str, AskJob] = {}
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._sweeper_task: asyncio.Task[None] | None = None

    def start(self, question: str) -> AskJob:
        ask_id = uuid4().hex
        job = AskJob(
            ask_id=ask_id,
            state="running",
            answer=None,
            failure=None,
            error_message=None,
            status_message=None,
            nonce_marker=None,
            thread_ref=None,
            created_at=self._clock(),
            finished_at=None,
        )
        self._jobs[ask_id] = job

        task = asyncio.create_task(self._run_job(ask_id, question))
        self._job_tasks.add(task)
        task.add_done_callback(self._job_tasks.discard)

        if self._sweeper_task is None:
            self._sweeper_task = asyncio.create_task(self._run_sweeper())

        return job

    def status(self, ask_id: str) -> AskJob | None:
        return self._jobs.get(ask_id)

    def result(self, ask_id: str) -> AskJob | None:
        return self._jobs.get(ask_id)

    async def aclose(self) -> None:
        sweeper_task = self._sweeper_task
        self._sweeper_task = None
        job_tasks = tuple(self._job_tasks)
        tasks = job_tasks + ((sweeper_task,) if sweeper_task is not None else ())

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._job_tasks.difference_update(job_tasks)

    async def _run_job(self, ask_id: str, question: str) -> None:
        try:
            outcome = await self._ask(
                question,
                on_status=lambda message: self._on_status(ask_id, message),
                on_conversation_id=lambda conversation_id: (
                    self._on_conversation_id(ask_id, conversation_id)
                ),
            )
            job = self._jobs[ask_id]
            thread_ref = (
                job.thread_ref
                if job.thread_ref is not None
                else outcome.conversation_id
            )
            self._jobs[ask_id] = replace(
                job,
                state="succeeded",
                answer=outcome.text,
                nonce_marker=outcome.marker,
                thread_ref=thread_ref,
            )
        except GptProAskError as exc:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                state="failed",
                failure=exc.failure,
                error_message=str(exc),
            )
        except Exception as exc:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                state="failed",
                failure="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], finished_at=self._clock()
            )

    def _on_status(self, ask_id: str, message: str) -> None:
        self._jobs[ask_id] = replace(
            self._jobs[ask_id], status_message=message
        )

    def _on_conversation_id(self, ask_id: str, conversation_id: str) -> None:
        job = self._jobs[ask_id]
        if job.thread_ref is not None:
            return
        self._jobs[ask_id] = replace(job, thread_ref=conversation_id)

    async def _run_sweeper(self) -> None:
        while True:
            await self._sleep(self._sweep_interval_seconds)
            now = self._clock()
            expired_ask_ids = [
                ask_id
                for ask_id, job in self._jobs.items()
                if job.state != "running"
                and job.finished_at is not None
                and job.finished_at + self._retention_seconds <= now
            ]
            for ask_id in expired_ask_ids:
                del self._jobs[ask_id]
