"""Lifecycle, queue, and ownership service for background gptpro asks."""

import asyncio
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from claudex.gptpro import ask as ask_module
from claudex.gptpro.ask import AskCallbacks, AskOutcome, GptProAskError

from .models import AskJob, TurnFinished
from .watchdog import AnswerWatchdog

# Keep the historical logger name so operator greps for these lifecycle
# lines keep matching after the module split.
logger = logging.getLogger("claudex.gptpro.jobs")

JOB_RETENTION_SECONDS = 24.0 * 60 * 60
SWEEP_INTERVAL_SECONDS = 300.0
QUEUE_TTL_SECONDS = 900.0
QUESTION_SPILL_THRESHOLD_BYTES = 35_000
ACTIVE_JOB_STATES = frozenset({"queued", "running", "detached"})


@dataclass(frozen=True)
class _ConversationOwnership:
    owner_ask_id: str
    released: asyncio.Event


class _AskCallable(Protocol):
    def __call__(
        self,
        question: str,
        *,
        callbacks: AskCallbacks | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> Awaitable[AskOutcome]: ...


class AskJobService:
    """Run asks in the background and retain immutable lifecycle snapshots."""

    def __init__(
        self,
        ask: _AskCallable,
        *,
        retention_seconds: float = JOB_RETENTION_SECONDS,
        sweep_interval_seconds: float = SWEEP_INTERVAL_SECONDS,
        overall_timeout_seconds: float | None = None,
        watchdog: AnswerWatchdog | None = None,
        queue_ttl_seconds: float | None = None,
        on_turn_finished: Callable[[TurnFinished], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._ask = ask
        self._retention_seconds = retention_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._overall_timeout_seconds = (
            ask_module.overall_timeout_seconds()
            if overall_timeout_seconds is None
            else overall_timeout_seconds
        )
        self._watchdog = (
            AnswerWatchdog(self._overall_timeout_seconds)
            if watchdog is None
            else watchdog
        )
        self._queue_ttl_seconds = (
            QUEUE_TTL_SECONDS
            if queue_ttl_seconds is None
            else queue_ttl_seconds
        )
        self._on_turn_finished = on_turn_finished
        self._clock = clock
        self._sleep = sleep
        self._jobs: dict[str, AskJob] = {}
        self._job_tasks: set[asyncio.Task[None]] = set()
        self._sweeper_task: asyncio.Task[None] | None = None
        self._conversation_owners: dict[str, _ConversationOwnership] = {}

    def start(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> AskJob:
        """Start an ask and notify `on_thread_ref` only after it succeeds.

        Completion-time notification makes concurrent session bindings follow
        successful completion order instead of conversation ID discovery order.
        """
        ask_id = uuid4().hex
        created_at = self._clock()
        queue_deadline = created_at + self._queue_ttl_seconds
        job = AskJob(
            ask_id=ask_id,
            state="queued",
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
        question_preview = " ".join(question.split())
        logger.info(
            'gptpro ask %.8s submitted (session=%.8s thread=%s '
            'question="%.40s…", chars=%d)',
            ask_id,
            session_id or "new",
            conversation_id or "new",
            question_preview,
            len(question),
        )

        task = asyncio.create_task(
            self._run_job(
                ask_id,
                question,
                conversation_id,
                on_thread_ref,
                queue_deadline,
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

    def has_active_jobs(self) -> bool:
        return any(
            job.state in ACTIVE_JOB_STATES for job in self._jobs.values()
        )

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
        queue_deadline: float,
        attachment_paths: Sequence[str] | None,
    ) -> None:
        spill_path: Path | None = None
        has_logged_queue_wait = False
        try:
            if conversation_id is not None:
                while ownership := self._conversation_owners.get(
                    conversation_id
                ):
                    self._on_status(
                        ask_id, "waiting for the in-flight answer"
                    )
                    if not has_logged_queue_wait:
                        logger.debug(
                            "gptpro ask %.8s waiting for the in-flight answer "
                            "(thread=%s)",
                            ask_id,
                            conversation_id,
                        )
                        has_logged_queue_wait = True
                    queue_wait_remaining = (
                        queue_deadline - self._clock()
                    )
                    if queue_wait_remaining <= 0:
                        raise GptProAskError(
                            "expired",
                            "the queue TTL expired while waiting for the "
                            "in-flight ask on this conversation",
                        )
                    try:
                        await asyncio.wait_for(
                            ownership.released.wait(),
                            timeout=queue_wait_remaining,
                        )
                    except TimeoutError as exc:
                        raise GptProAskError(
                            "expired",
                            "the queue TTL expired while waiting for the "
                            "in-flight ask on this conversation",
                        ) from exc
                self._conversation_owners[conversation_id] = (
                    _ConversationOwnership(ask_id, asyncio.Event())
                )

            self._jobs[ask_id] = replace(
                self._jobs[ask_id], state="running"
            )
            admitted_at = self._clock()
            remaining = self._watchdog.execution_budget_seconds()
            logger.info(
                "gptpro ask %.8s admitted (waited=%.1fs budget=%.0fs "
                "thread=%s)",
                ask_id,
                admitted_at - self._jobs[ask_id].created_at,
                remaining,
                self._jobs[ask_id].thread_ref or "new",
            )

            def capture_status(message: str) -> None:
                self._on_status(ask_id, message)

            def capture_conversation_id(
                captured_conversation_id: str,
            ) -> None:
                self._on_conversation_id(ask_id, captured_conversation_id)

            def capture_marker(marker: str) -> None:
                self._on_marker(ask_id, marker)

            def capture_detached() -> None:
                self._on_detached(ask_id)

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
                "callbacks": AskCallbacks(
                    on_status=capture_status,
                    on_conversation_id=capture_conversation_id,
                    on_marker=capture_marker,
                    on_detached=capture_detached,
                ),
                "conversation_id": conversation_id,
                "timeout_seconds": remaining,
            }
            if provider_attachment_paths is not None:
                ask_options["attachment_paths"] = provider_attachment_paths
            outcome = await self._ask(provider_question, **ask_options)
            job = self._jobs[ask_id]
            if job.thread_ref is None and outcome.conversation_id is not None:
                self._on_conversation_id(ask_id, outcome.conversation_id)
                job = self._jobs[ask_id]
            self._jobs[ask_id] = replace(
                job,
                state="succeeded",
                answer=outcome.text,
                nonce_marker=(
                    job.nonce_marker
                    if job.nonce_marker is not None
                    else outcome.marker
                ),
            )
            duration_seconds = self._clock() - admitted_at
            thread_ref = self._jobs[ask_id].thread_ref
            self._watchdog.record(duration_seconds)
            logger.info(
                "gptpro ask %.8s succeeded (duration=%.1fs thread=%s "
                "answer_chars=%d)",
                ask_id,
                duration_seconds,
                thread_ref or "new",
                len(outcome.text),
            )
            if on_thread_ref is not None and thread_ref is not None:
                try:
                    on_thread_ref(thread_ref)
                except Exception:
                    pass
            if self._on_turn_finished is not None:
                try:
                    self._on_turn_finished(
                        TurnFinished(
                            ask_id=ask_id,
                            thread_ref=thread_ref,
                            answer=outcome.text,
                        )
                    )
                except Exception:
                    pass
        except GptProAskError as exc:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                state="failed",
                failure=exc.failure,
                error_message=str(exc),
            )
            logger.warning(
                "gptpro ask %.8s failed (failure=%s thread=%s): %s",
                ask_id,
                exc.failure,
                self._jobs[ask_id].thread_ref or "new",
                exc,
            )
        except Exception as exc:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                state="failed",
                failure="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )
            logger.exception(
                "gptpro ask %.8s failed unexpectedly", ask_id
            )
        finally:
            owned_conversation_id = self._jobs[ask_id].thread_ref
            ownership = (
                self._conversation_owners.get(owned_conversation_id)
                if owned_conversation_id is not None
                else None
            )
            if ownership is not None and ownership.owner_ask_id == ask_id:
                del self._conversation_owners[owned_conversation_id]
                ownership.released.set()
            if spill_path is not None:
                spill_path.unlink(missing_ok=True)
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], finished_at=self._clock()
            )

    def _on_status(self, ask_id: str, message: str) -> None:
        self._jobs[ask_id] = replace(
            self._jobs[ask_id], status_message=message
        )

    def _on_detached(self, ask_id: str) -> None:
        job = self._jobs.get(ask_id)
        if job is None or job.state != "running":
            return
        self._jobs[ask_id] = replace(job, state="detached")
        logger.info(
            "gptpro ask %.8s detached (thread=%s) - polling for the answer",
            ask_id,
            job.thread_ref or "new",
        )

    def _on_marker(self, ask_id: str, marker: str) -> None:
        job = self._jobs[ask_id]
        if job.nonce_marker is not None:
            return
        self._jobs[ask_id] = replace(job, nonce_marker=marker)

    def _on_conversation_id(
        self,
        ask_id: str,
        conversation_id: str,
    ) -> None:
        job = self._jobs[ask_id]
        if job.thread_ref is not None:
            return
        self._jobs[ask_id] = replace(job, thread_ref=conversation_id)
        self._conversation_owners.setdefault(
            conversation_id,
            _ConversationOwnership(ask_id, asyncio.Event()),
        )

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
                job = self._jobs.pop(ask_id)
                logger.debug(
                    "gptpro ask %.8s record swept (state=%s age=%.0fs)",
                    ask_id,
                    job.state,
                    now - job.created_at,
                )
