"""Lifecycle, queue, and ownership service for background gptpro asks."""

import asyncio
import logging
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from claudex.gptpro import ask as ask_module
from claudex.gptpro.ask import AskCallbacks, AskEvidence, AskOutcome, GptProAskError
from claudex.gptpro.conversation import is_conversation_id

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
_NONCE_PATTERN = re.compile(r"^\[gptpro-transport-nonce:[^\]\r\n]{1,128}\]$")


def _spill_question(
    question: str, attachment_paths: Sequence[str] | None,
) -> tuple[str, list[str], Path]:
    spill_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="gptpro-spill-",
        suffix=".txt",
        delete=False,
    )
    spill_path = Path(spill_file.name)
    try:
        with spill_file:
            spill_file.write(question)
    except OSError:
        spill_path.unlink(missing_ok=True)
        raise
    provider_question = (
        "The full question text is attached as "
        f"{spill_path.name}; read the attachment and answer it."
    )
    return provider_question, [str(spill_path), *(attachment_paths or ())], spill_path


def _ownership_key(conversation_id: str) -> str:
    if is_conversation_id(conversation_id):
        return conversation_id.lower()
    return conversation_id


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


class _RecoverCallable(Protocol):
    def __call__(
        self, conversation_id: str, marker: str,
    ) -> Awaitable[AskOutcome]: ...


class AskJobService:
    """Run asks in the background and retain immutable lifecycle snapshots."""

    def __init__(
        self,
        ask: _AskCallable,
        *,
        recover: _RecoverCallable | None = None,
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
        self._recover = recover
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
            evidence=AskEvidence(conversation_id=conversation_id),
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

    def start_recovery(
        self,
        *,
        ask_id: str | None = None,
        conversation_id: str | None = None,
        marker: str | None = None,
    ) -> AskJob:
        """Poll an existing turn by retained ask ID or explicit identifiers."""
        if ask_id is not None and (conversation_id is not None or marker is not None):
            raise ValueError("provide ask_id or thread_ref and nonce_marker, not both")
        source: AskJob | None = None
        if ask_id is not None:
            source = self.status(ask_id)
            if source is None:
                raise ValueError(
                    "unknown or expired ask_id; provide saved thread_ref and "
                    "nonce_marker to recover after restart"
                )
            if source.state != "failed":
                raise ValueError("only a failed ask can start recovery; poll pending jobs")
            conversation_id, marker = source.thread_ref, source.nonce_marker
        if (
            not is_conversation_id(conversation_id)
            or not isinstance(marker, str)
            or not _NONCE_PATTERN.fullmatch(marker)
        ):
            raise ValueError(
                "recovery requires a known ChatGPT thread_ref UUID and "
                "nonce_marker; no prompt is submitted"
            )
        if self._recover is None:
            raise ValueError("read-only answer recovery is unavailable")
        for job in self._jobs.values():
            if (
                job.state in ACTIVE_JOB_STATES
                and job.thread_ref is not None
                and _ownership_key(job.thread_ref) == _ownership_key(conversation_id)
                and job.nonce_marker == marker
            ):
                raise ValueError(
                    "recovery already pending for this turn; poll its ask_id"
                )
        recovery_id = uuid4().hex
        created_at = self._clock()
        evidence = (
            source.evidence if source is not None else AskEvidence(
                conversation_id=conversation_id, submission="uncertain",
            )
        )
        job = AskJob(
            ask_id=recovery_id, state="queued", answer=None, failure=None,
            error_message=None, status_message=None, nonce_marker=marker,
            thread_ref=conversation_id, created_at=created_at, finished_at=None,
            evidence=replace(evidence, recovery="not_attempted"),
            source_ask_id=ask_id,
        )
        self._jobs[recovery_id] = job
        task = asyncio.create_task(self._run_recovery(
            recovery_id, created_at + self._queue_ttl_seconds,
        ))
        self._job_tasks.add(task)
        task.add_done_callback(self._job_tasks.discard)
        if self._sweeper_task is None:
            self._sweeper_task = asyncio.create_task(self._run_sweeper())
        return job

    async def _run_recovery(self, ask_id: str, queue_deadline: float) -> None:
        job = self._jobs[ask_id]
        assert job.thread_ref is not None and job.nonce_marker is not None
        try:
            await self._claim_conversation(ask_id, job.thread_ref, queue_deadline)
            job = replace(
                self._jobs[ask_id], state="detached",
                status_message="detached; polling for the answer",
                evidence=replace(self._jobs[ask_id].evidence, recovery="polling"),
            )
            self._jobs[ask_id] = job
            assert self._recover is not None
            outcome = await self._recover(job.thread_ref, job.nonce_marker)
            if (
                not outcome.text or outcome.marker != job.nonce_marker
                or not is_conversation_id(outcome.conversation_id)
                or _ownership_key(outcome.conversation_id) != _ownership_key(job.thread_ref)
            ):
                raise GptProAskError(
                    "no_raw_turn", "recovery did not return a finished "
                    "nonce-correlated answer for this conversation",
                )
            self._jobs[ask_id] = replace(
                job, state="succeeded", answer=outcome.text,
                files=outcome.files, files_complete=outcome.files_complete,
                evidence=replace(
                    job.evidence, recovery="recovered", submission="confirmed",
                    raw_extracted=True, answer_seen=True, generation_observed=True,
                ),
            )
        except asyncio.CancelledError:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], state="failed", failure="cancelled",
                error_message="the recovery was cancelled",
            )
            raise
        except Exception as exc:
            failure = (
                exc.failure if isinstance(exc, GptProAskError)
                else "timeout" if isinstance(exc, TimeoutError) else "error"
            )
            detail = str(exc) or (
                "the bounded answer recovery window expired"
                if isinstance(exc, TimeoutError) else type(exc).__name__
            )
            recovery_evidence = (
                exc.evidence if isinstance(exc, GptProAskError) else None
            )
            is_unavailable = (
                self._jobs[ask_id].state != "detached"
                or failure in {"session_expired", "challenge"}
                or recovery_evidence is not None
                and recovery_evidence.recovery == "unavailable"
            )
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], state="failed", failure=failure,
                error_message=detail,
                evidence=replace(
                    self._jobs[ask_id].evidence,
                    failure_stage=(
                        "queue" if self._jobs[ask_id].state == "queued"
                        else "recovery"
                    ),
                    recovery="unavailable" if is_unavailable else "exhausted",
                    recovery_failure=(
                        recovery_evidence.recovery_failure
                        if recovery_evidence is not None
                        and recovery_evidence.recovery_failure is not None
                        else failure
                    ),
                    recovery_detail=detail,
                ),
            )
        finally:
            self._release_conversation(ask_id)
            self._jobs[ask_id] = replace(
                self._jobs[ask_id], status_message=None, finished_at=self._clock(),
            )

    async def _claim_conversation(
        self, ask_id: str, conversation_id: str, queue_deadline: float,
    ) -> None:
        ownership_key = _ownership_key(conversation_id)
        has_logged_wait = False
        while ownership := self._conversation_owners.get(ownership_key):
            self._on_status(ask_id, "waiting for the in-flight answer")
            if not has_logged_wait:
                logger.debug(
                    "gptpro ask %.8s waiting for the in-flight answer (thread=%s)",
                    ask_id, conversation_id,
                )
                has_logged_wait = True
            remaining = queue_deadline - self._clock()
            if remaining <= 0:
                raise GptProAskError(
                    "expired", "the queue TTL expired while waiting for the "
                    "in-flight ask on this conversation",
                )
            try:
                await asyncio.wait_for(ownership.released.wait(), remaining)
            except TimeoutError as exc:
                raise GptProAskError(
                    "expired", "the queue TTL expired while waiting for the "
                    "in-flight ask on this conversation",
                ) from exc
            # An already-released event may return without yielding.
            await asyncio.sleep(0)
        self._conversation_owners[ownership_key] = _ConversationOwnership(
            ask_id, asyncio.Event()
        )

    def _release_conversation(self, ask_id: str) -> None:
        conversation_id = self._jobs[ask_id].thread_ref
        if conversation_id is None:
            return
        key = _ownership_key(conversation_id)
        ownership = self._conversation_owners.get(key)
        if ownership is not None and ownership.owner_ask_id == ask_id:
            del self._conversation_owners[key]
            ownership.released.set()

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

        for ask_id, job in self._jobs.items():
            if job.state in ACTIVE_JOB_STATES:
                self._jobs[ask_id] = replace(
                    job,
                    state="failed",
                    failure="cancelled",
                    error_message="the ask was cancelled",
                    status_message=None,
                    finished_at=self._clock(),
                )
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
        try:
            if conversation_id is not None:
                await self._claim_conversation(
                    ask_id, conversation_id, queue_deadline
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

            def capture_evidence(evidence: AskEvidence) -> None:
                self._on_evidence(ask_id, evidence)

            provider_question = question
            provider_attachment_paths = attachment_paths
            if len(question.encode("utf-8")) > QUESTION_SPILL_THRESHOLD_BYTES:
                provider_question, provider_attachment_paths, spill_path = (
                    _spill_question(question, attachment_paths)
                )

            ask_options: dict[str, Any] = {
                "callbacks": AskCallbacks(
                    on_status=capture_status,
                    on_conversation_id=capture_conversation_id,
                    on_marker=capture_marker,
                    on_detached=capture_detached,
                    on_evidence=capture_evidence,
                ),
                "conversation_id": conversation_id,
                "timeout_seconds": remaining,
            }
            if provider_attachment_paths is not None:
                ask_options["attachment_paths"] = provider_attachment_paths
            try:
                outcome = await self._ask(provider_question, **ask_options)
            except GptProAskError as exc:
                if (
                    exc.failure != "submit_failed"
                    or exc.evidence is None
                    or exc.evidence.failure_stage != "composer"
                    or exc.evidence.submission != "not_attempted"
                    or spill_path is not None
                ):
                    raise
                retry_budget = remaining - (self._clock() - admitted_at)
                if retry_budget <= 0:
                    raise
                logger.warning(
                    "gptpro ask %.8s composer did not retain the inline question "
                    "(%s); resubmitting it as an attachment",
                    ask_id,
                    str(exc),
                )
                capture_status(
                    "composer did not retain the inline question; "
                    "resubmitting it as an attachment"
                )
                job = self._jobs[ask_id]
                # Recovery must use the nonce of the attempt that submits.
                self._jobs[ask_id] = replace(
                    job, nonce_marker=None,
                    evidence=replace(job.evidence, failure_stage=None),
                )
                provider_question, provider_attachment_paths, spill_path = (
                    _spill_question(question, attachment_paths)
                )
                ask_options["attachment_paths"] = provider_attachment_paths
                ask_options["timeout_seconds"] = retry_budget
                outcome = await self._ask(provider_question, **ask_options)
            job = self._jobs[ask_id]
            if job.thread_ref is None and outcome.conversation_id is not None:
                self._on_conversation_id(ask_id, outcome.conversation_id)
                job = self._jobs[ask_id]
            self._jobs[ask_id] = replace(
                job,
                state="succeeded",
                answer=outcome.text,
                files=outcome.files,
                files_complete=outcome.files_complete,
                evidence=replace(
                    job.evidence, submission="confirmed",
                    generation_observed=True, answer_seen=True,
                    raw_extracted=True,
                ),
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
        except asyncio.CancelledError:
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                state="failed",
                failure="cancelled",
                error_message="the ask was cancelled",
            )
            raise
        except GptProAskError as exc:
            if exc.evidence is not None:
                self._on_evidence(ask_id, exc.evidence)
            job = self._jobs[ask_id]
            evidence = job.evidence
            stage = evidence.failure_stage or {
                "echo_timeout": "echo", "no_raw_turn": "answer",
                "submit_failed": "submission", "navigation_failed": "navigation",
                "expired": "queue",
            }.get(exc.failure, "provider")
            submission = evidence.submission
            if submission == "not_attempted" and exc.failure in {
                "echo_timeout", "no_raw_turn",
            }:
                submission = "uncertain"
            self._jobs[ask_id] = replace(
                job,
                state="failed",
                failure=exc.failure,
                error_message=str(exc),
                evidence=replace(
                    evidence, failure_stage=stage, submission=submission,
                    conversation_id=job.thread_ref,
                ),
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
            self._release_conversation(ask_id)
            if spill_path is not None:
                spill_path.unlink(missing_ok=True)
            self._jobs[ask_id] = replace(
                self._jobs[ask_id],
                status_message=None,
                finished_at=self._clock(),
            )

    def _on_evidence(self, ask_id: str, evidence: AskEvidence) -> None:
        job = self._jobs[ask_id]
        if job.state not in ACTIVE_JOB_STATES:
            return
        if job.thread_ref is None and evidence.conversation_id is not None:
            self._on_conversation_id(ask_id, evidence.conversation_id)
            job = self._jobs[ask_id]
        previous = job.evidence
        submission_order = ("not_attempted", "uncertain", "confirmed")
        merged = replace(
            evidence,
            submission=max(
                (previous.submission, evidence.submission),
                key=submission_order.index,
            ),
            conversation_id=(
                evidence.conversation_id or job.thread_ref
                or previous.conversation_id
            ),
            upload_receipts=max(
                (value for value in (
                    previous.upload_receipts, evidence.upload_receipts
                ) if value is not None),
                default=None,
            ),
            ready_attachments=max(
                (value for value in (
                    previous.ready_attachments, evidence.ready_attachments
                ) if value is not None),
                default=None,
            ),
            generation_observed=(
                previous.generation_observed or evidence.generation_observed
            ),
            answer_seen=previous.answer_seen or evidence.answer_seen,
            raw_extracted=previous.raw_extracted or evidence.raw_extracted,
            failure_stage=evidence.failure_stage or previous.failure_stage,
            recovery=(
                evidence.recovery if evidence.recovery != "not_attempted"
                else previous.recovery
            ),
            recovery_failure=(
                evidence.recovery_failure or previous.recovery_failure
            ),
            recovery_detail=evidence.recovery_detail or previous.recovery_detail,
        )
        self._jobs[ask_id] = replace(job, evidence=merged)

    def _on_status(self, ask_id: str, message: str) -> None:
        job = self._jobs[ask_id]
        if job.state not in ACTIVE_JOB_STATES:
            return
        self._jobs[ask_id] = replace(job, status_message=message)

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
        self._jobs[ask_id] = replace(
            job, thread_ref=conversation_id,
            evidence=replace(job.evidence, conversation_id=conversation_id),
        )
        self._conversation_owners.setdefault(
            _ownership_key(conversation_id),
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
