"""In-memory lifecycle registry for background ChatGPT Pro asks.

Job and thread registries live only for the process lifetime. A daemon restart
loses pending jobs and session thread bindings, while callers retain thread_ref
values to revisit conversations.
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from claudex.gptpro import ask as ask_module
from claudex.gptpro.ask import AskOutcome, GptProAskError

JOB_RETENTION_SECONDS = 24.0 * 60 * 60
SWEEP_INTERVAL_SECONDS = 300.0
THREAD_BINDING_TTL_SECONDS = 23 * 60 * 60
QUESTION_SPILL_THRESHOLD_BYTES = 35_000

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
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> Awaitable[AskOutcome]: ...


class ThreadRegistry:
    """Retain MCP session bindings to ChatGPT conversation threads."""

    def __init__(
        self, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self._bindings: dict[str, tuple[str, float]] = {}

    def bind(self, session_id: str, thread_ref: str) -> None:
        expires_at = self._clock() + THREAD_BINDING_TTL_SECONDS
        self._bindings[session_id] = (thread_ref, expires_at)

    def lookup(self, session_id: str) -> str | None:
        binding = self._bindings.get(session_id)
        if binding is None:
            return None
        thread_ref, expires_at = binding
        if expires_at <= self._clock():
            del self._bindings[session_id]
            return None
        return thread_ref


class AskJobService:
    """Run asks in the background and retain immutable lifecycle snapshots."""

    def __init__(
        self,
        ask: _AskCallable,
        *,
        retention_seconds: float = JOB_RETENTION_SECONDS,
        sweep_interval_seconds: float = SWEEP_INTERVAL_SECONDS,
        overall_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._ask = ask
        ask_parameters = inspect.signature(ask).parameters
        self._supports_conversation_options = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in ask_parameters.values()
        ) or {"conversation_id", "timeout_seconds"}.issubset(ask_parameters)
        self._retention_seconds = retention_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._overall_timeout_seconds = (
            ask_module.overall_timeout_seconds()
            if overall_timeout_seconds is None
            else overall_timeout_seconds
        )
        self._clock = clock
        self._sleep = sleep
        self._jobs: dict[str, AskJob] = {}
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._sweeper_task: asyncio.Task[None] | None = None
        self._thread_locks: dict[str, asyncio.Lock] = {}

    def start(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> AskJob:
        ask_id = uuid4().hex
        created_at = self._clock()
        expires_at = created_at + self._overall_timeout_seconds
        job = AskJob(
            ask_id=ask_id,
            state="running",
            answer=None,
            failure=None,
            error_message=None,
            status_message=None,
            nonce_marker=None,
            thread_ref=conversation_id,
            created_at=created_at,
            finished_at=None,
        )
        self._jobs[ask_id] = job

        task = asyncio.create_task(
            self._run_job(
                ask_id,
                question,
                conversation_id,
                on_thread_ref,
                expires_at,
                attachment_paths,
            )
        )
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

    async def _run_job(
        self,
        ask_id: str,
        question: str,
        conversation_id: str | None,
        on_thread_ref: Callable[[str], None] | None,
        expires_at: float,
        attachment_paths: Sequence[str] | None,
    ) -> None:
        thread_lock: asyncio.Lock | None = None
        has_thread_lock = False
        spill_path: Path | None = None
        try:
            if conversation_id is not None:
                thread_lock = self._thread_locks.setdefault(
                    conversation_id, asyncio.Lock()
                )
                if thread_lock.locked():
                    self._on_status(
                        ask_id, "waiting for the in-flight answer"
                    )
                    remaining = expires_at - self._clock()
                    if remaining <= 0:
                        raise GptProAskError(
                            "timeout",
                            "the ask deadline expired while waiting for the "
                            "in-flight answer on this conversation",
                        )
                    try:
                        await asyncio.wait_for(
                            thread_lock.acquire(), timeout=remaining
                        )
                    except TimeoutError as exc:
                        raise GptProAskError(
                            "timeout",
                            "the ask deadline expired while waiting for the "
                            "in-flight answer on this conversation",
                        ) from exc
                else:
                    await thread_lock.acquire()
                has_thread_lock = True

            remaining = expires_at - self._clock()
            if remaining <= 0:
                raise GptProAskError(
                    "timeout", "the ask deadline expired before provider start"
                )

            def capture_status(message: str) -> None:
                self._on_status(ask_id, message)

            def capture_conversation_id(
                captured_conversation_id: str,
            ) -> None:
                self._on_conversation_id(
                    ask_id, captured_conversation_id, on_thread_ref
                )

            provider_question = question
            provider_attachment_paths = attachment_paths
            if len(question.encode("utf-8")) > QUESTION_SPILL_THRESHOLD_BYTES:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="gptpro-spill-",
                    suffix=".txt",
                    delete=False,
                ) as spill_file:
                    spill_path = Path(spill_file.name)
                    spill_file.write(question)
                provider_question = (
                    "The full question text is attached as "
                    f"{spill_path.name}; read the attachment and answer it."
                )
                provider_attachment_paths = [
                    str(spill_path),
                    *(attachment_paths or ()),
                ]

            ask_options: dict[str, Any] = {
                "on_status": capture_status,
                "on_conversation_id": capture_conversation_id,
            }
            if self._supports_conversation_options:
                ask_options.update(
                    conversation_id=conversation_id,
                    timeout_seconds=remaining,
                )
            if provider_attachment_paths is not None:
                ask_options["attachment_paths"] = provider_attachment_paths
            outcome = await self._ask(provider_question, **ask_options)
            job = self._jobs[ask_id]
            if job.thread_ref is None and outcome.conversation_id is not None:
                self._on_conversation_id(
                    ask_id, outcome.conversation_id, on_thread_ref
                )
                job = self._jobs[ask_id]
            self._jobs[ask_id] = replace(
                job,
                state="succeeded",
                answer=outcome.text,
                nonce_marker=outcome.marker,
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
            if has_thread_lock:
                assert thread_lock is not None
                thread_lock.release()
            if spill_path is not None:
                spill_path.unlink(missing_ok=True)
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], finished_at=self._clock()
            )

    def _on_status(self, ask_id: str, message: str) -> None:
        self._jobs[ask_id] = replace(
            self._jobs[ask_id], status_message=message
        )

    def _on_conversation_id(
        self,
        ask_id: str,
        conversation_id: str,
        on_thread_ref: Callable[[str], None] | None,
    ) -> None:
        job = self._jobs[ask_id]
        if job.thread_ref is not None:
            return
        self._jobs[ask_id] = replace(job, thread_ref=conversation_id)
        if on_thread_ref is None:
            return
        try:
            on_thread_ref(conversation_id)
        except Exception:
            return

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
