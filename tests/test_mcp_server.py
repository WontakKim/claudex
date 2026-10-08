"""Protocol and optional-dependency tests for the ChatGPT Pro MCP endpoint."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from claudex.config import GatewayConfig
from claudex.gptpro import ask, jobs
from claudex.mcp_server import LazyAskRuntime, McpEndpoint

_PROTOCOL_VERSION = "2025-06-18"
_REQUEST_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
_CONVERSATION_A = "11111111-1111-4111-8111-111111111111"
_CONVERSATION_B = "22222222-2222-4222-8222-222222222222"


class FakeAskRuntime:
    def __init__(
        self,
        *,
        answer: str = "# Answer\n\nSettled Markdown.",
        conversation_id: str | None = "conversation-123",
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.conversation_id = conversation_id
        self.error = error
        self.questions: list[str] = []
        self.provider_conversation_ids: list[str | None] = []
        self.provider_attachment_paths: list[Sequence[str] | None] = []
        self.submitted_session_ids: list[str | None] = []
        self._conversation_id_callbacks: list[
            Callable[[str], None] | None
        ] = []
        self._provider_calls_changed = asyncio.Condition()
        self._provider_release_events: list[asyncio.Event] = []
        self._submitted_ask_ids: list[str] = []
        self._job_service = jobs.AskJobService(self._ask, recover=self._recover)
        self._thread_registry = jobs.ThreadRegistry()

    async def _ask(
        self,
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del timeout_seconds
        release_provider = asyncio.Event()
        async with self._provider_calls_changed:
            self.questions.append(question)
            self.provider_conversation_ids.append(conversation_id)
            self.provider_attachment_paths.append(attachment_paths)
            self._conversation_id_callbacks.append(
                callbacks.on_conversation_id
                if callbacks is not None
                else None
            )
            self._provider_release_events.append(release_provider)
            self._provider_calls_changed.notify_all()
        if callbacks is not None and callbacks.on_status is not None:
            callbacks.on_status("waiting for ChatGPT Pro")
        await release_provider.wait()
        if self.error is not None:
            if callbacks is not None and callbacks.on_marker is not None:
                callbacks.on_marker("[gptpro-transport-nonce:fake]")
            raise self.error
        return ask.AskOutcome(
            text=self.answer,
            marker="nonce-marker",
            conversation_id=self.conversation_id,
        )

    async def _recover(
        self, conversation_id: str, marker: str,
    ) -> ask.AskOutcome:
        return ask.AskOutcome(self.answer, marker, conversation_id)

    def start_recovery(
        self, *, ask_id: str | None = None,
        conversation_id: str | None = None, marker: str | None = None,
    ) -> jobs.AskJob:
        return self._job_service.start_recovery(
            ask_id=ask_id, conversation_id=conversation_id, marker=marker,
        )

    def start_ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
        session_id: str | None = None,
    ) -> jobs.AskJob:
        job = self._job_service.start(
            question,
            conversation_id=conversation_id,
            on_thread_ref=on_thread_ref,
            attachment_paths=attachment_paths,
            session_id=session_id,
        )
        self.submitted_session_ids.append(session_id)
        self._submitted_ask_ids.append(job.ask_id)
        return job

    def lookup_thread(self, session_id: str) -> str | None:
        return self._thread_registry.lookup(session_id)

    def bind_thread(self, session_id: str, thread_ref: str) -> None:
        self._thread_registry.bind(session_id, thread_ref)

    def job_status(self, ask_id: str) -> jobs.AskJob | None:
        return self._job_service.status(ask_id)

    def job_result(self, ask_id: str) -> jobs.AskJob | None:
        return self._job_service.result(ask_id)

    async def wait_for_provider_calls(self, count: int) -> None:
        async with self._provider_calls_changed:
            await self._provider_calls_changed.wait_for(
                lambda: len(self.provider_conversation_ids) >= count
            )

    async def latch_thread(self, call_index: int, thread_ref: str) -> None:
        await self.wait_for_provider_calls(call_index + 1)
        callback = self._conversation_id_callbacks[call_index]
        assert callback is not None
        callback(thread_ref)

    async def finish_job(self, ask_id: str) -> None:
        call_index = self._submitted_ask_ids.index(ask_id)
        await self.wait_for_provider_calls(call_index + 1)
        self._provider_release_events[call_index].set()
        while True:
            job = self._job_service.status(ask_id)
            if job is None or job.state != "running":
                return
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        await self._job_service.aclose()


@contextlib.contextmanager
def _mcp_client(
    runtime: FakeAskRuntime,
    *,
    local_token: str | None = None,
) -> Iterator[TestClient]:
    endpoint = McpEndpoint()

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> Any:
        try:
            yield
        finally:
            await endpoint.aclose()
            await runtime.aclose()

    app = Starlette(routes=[Route("/mcp", endpoint)], lifespan=lifespan)
    app.state.config = GatewayConfig(local_token=local_token)
    app.state.gptpro_ask_runtime = runtime
    with TestClient(app, base_url="http://localhost") as client:
        yield client


def _initialize(client: TestClient) -> tuple[dict[str, Any], dict[str, str]]:
    response = client.post(
        "/mcp",
        headers=_REQUEST_HEADERS,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    session_headers = {
        **_REQUEST_HEADERS,
        "mcp-session-id": session_id,
        "mcp-protocol-version": _PROTOCOL_VERSION,
    }
    initialized = client.post(
        "/mcp",
        headers=session_headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized.status_code == 202
    return response.json(), session_headers


def _call_tool(
    client: TestClient,
    headers: dict[str, str],
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert response.status_code == 200
    return response.json()["result"]


def _json_tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    payload = json.loads(result["content"][0]["text"])
    assert isinstance(payload, dict)
    return payload


def _wait_for_provider_calls(
    client: TestClient, runtime: FakeAskRuntime, count: int
) -> None:
    assert client.portal is not None
    client.portal.call(runtime.wait_for_provider_calls, count)


def _latch_thread(
    client: TestClient,
    runtime: FakeAskRuntime,
    call_index: int,
    thread_ref: str,
) -> None:
    assert client.portal is not None
    client.portal.call(runtime.latch_thread, call_index, thread_ref)


def _finish_job(
    client: TestClient, runtime: FakeAskRuntime, ask_id: str
) -> None:
    assert client.portal is not None
    client.portal.call(runtime.finish_job, ask_id)


def test_initialize_handshake_advertises_tools() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        payload, _headers = _initialize(client)

    assert payload["result"]["protocolVersion"] == _PROTOCOL_VERSION
    assert payload["result"]["serverInfo"]["name"] == "claudex-gateway-gptpro"
    assert payload["result"]["capabilities"]["tools"] == {"listChanged": False}


def test_tools_list_exposes_job_tools_schemas_and_usage_guidance() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    assert set(tools) == {
        "ask_gpt_pro",
        "ask_gpt_pro_status",
        "ask_gpt_pro_result",
        "recover_gpt_pro",
    }
    assert tools["recover_gpt_pro"]["inputSchema"]["additionalProperties"] is False
    assert tools["ask_gpt_pro"]["inputSchema"] == {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "thread": {
                "type": "string",
                "description": (
                    "'new' forces a fresh conversation; a conversation UUID (a "
                    "previous thread_ref) continues that conversation; omit to "
                    "continue this session's most recently completed "
                    "conversation."
                ),
            },
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional plain-text file paths to attach (UTF-8 only; at "
                    "most 10 files and 1.2 MB total; questions over ~35 KB are "
                    "spilled into an attachment automatically, so send full "
                    "text)."
                ),
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }
    ask_id_schema = {
        "type": "object",
        "properties": {"ask_id": {"type": "string"}},
        "required": ["ask_id"],
        "additionalProperties": False,
    }
    assert tools["ask_gpt_pro_status"]["inputSchema"] == ask_id_schema
    assert tools["ask_gpt_pro_result"]["inputSchema"] == ask_id_schema
    ask_description = tools["ask_gpt_pro"]["description"]
    assert "return immediately with {ask_id, thread_ref}" in ask_description
    assert "ask_gpt_pro_status(ask_id)" in ask_description
    assert "ask_gpt_pro_result(ask_id)" in ask_description
    assert "state is queued, running, or detached" in ask_description
    assert "detached asks remain recoverable" in ask_description
    assert "An expired failure" in ask_description
    assert "recover_gpt_pro" in ask_description
    assert "self-contained whenever possible" in ask_description
    assert "up to 10 UTF-8 plain-text files totaling 1.2 MB" in ask_description
    assert "Questions over ~35 KB" in ask_description
    assert "send the full text" in ask_description
    assert (
        "Omitting thread continues this MCP session's most recently completed "
        "conversation"
        in ask_description
    )
    assert "ChatGPT can see previous turns" in ask_description
    assert 'Set thread to "new" to force a fresh conversation' in ask_description
    assert "conversation UUID from a previous thread_ref" in ask_description
    assert "Asks in the same conversation are serialized" in ask_description
    assert "status reports that an ask is waiting" in ask_description
    assert "GPTPRO_CONTEXT_REQUEST_V1" in ask_description
    assert "omitting thread continues the conversation" in ask_description
    assert "explicitly requests ChatGPT Pro" in ask_description
    assert "consequential judgment" in ask_description
    status_description = tools["ask_gpt_pro_status"]["description"]
    assert "current state" in status_description
    assert "thread_ref is preserved throughout the job lifecycle" in (
        status_description
    )
    assert "binding only after the ask succeeds" in status_description
    assert "queued means awaiting admission" in status_description
    assert "asks in other conversations can continue in parallel" in (
        status_description
    )
    assert "running means navigation, submission, or answer observation" in (
        status_description
    )
    assert "submission must be read from evidence" in status_description
    assert "detached means read-only server polling" in status_description
    assert "succeeded and failed are terminal" in status_description
    assert "failure=expired" in status_description
    assert "any available nonce marker are preserved" in status_description
    assert '"waiting for the in-flight answer"' in status_description
    assert '"detached; polling for the answer"' in status_description
    result_description = tools["ask_gpt_pro_result"]["description"]
    assert "settled result" in result_description
    assert "queued, running, and detached are still in progress" in (
        result_description
    )
    assert "failure=expired" in result_description


def test_tool_descriptions_enumerate_failure_recovery_actions() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    ask_description = tools["ask_gpt_pro"]["description"]
    status_description = tools["ask_gpt_pro_status"]["description"]
    result_description = tools["ask_gpt_pro_result"]["description"]
    failure_action_labels = (
        "- expired:",
        "- no_raw_turn:",
        "- timeout / echo_timeout:",
        "- rate_limited_timeout:",
        "- session_expired / challenge:",
        "- locator_unresolved:",
        "- submit_failed / navigation_failed / error:",
    )
    for description in (status_description, result_description):
        for failure_action_label in failure_action_labels:
            assert failure_action_label in description
        # Unknown submission must not be described as safe to retry.
        assert "recover_gpt_pro" in description
        assert "Unknown outcome is not proof" in description

    assert (
        "An expired failure means same-conversation queue admission ended "
        "before this ask submitted" in ask_description
    )
    assert "any available nonce marker are preserved" in status_description
    assert '"detached; polling for the answer"' in status_description


def test_lazy_runtime_forwards_detached_callback() -> None:
    captured_callbacks: list[Callable[[], None] | None] = []

    class ProviderRuntime:
        async def ask(
            self,
            question: str,
            *,
            callbacks: ask.AskCallbacks | None = None,
            conversation_id: str | None = None,
            timeout_seconds: float | None = None,
            attachment_paths: Sequence[str] | None = None,
        ) -> ask.AskOutcome:
            del conversation_id, timeout_seconds, attachment_paths
            assert question == "question"
            captured_callbacks.append(
                callbacks.on_detached if callbacks is not None else None
            )
            return ask.AskOutcome("answer", "marker", None)

        async def aclose(self) -> None:
            return None

    async def scenario() -> None:
        lazy_runtime = LazyAskRuntime()
        lazy_runtime._runtime = ProviderRuntime()
        callback = lambda: None

        outcome = await lazy_runtime.ask(
            "question", callbacks=ask.AskCallbacks(on_detached=callback)
        )

        assert outcome.text == "answer"
        assert captured_callbacks == [callback]
        await lazy_runtime.aclose()

    asyncio.run(scenario())


def test_lazy_runtime_passes_session_id_to_job_service() -> None:
    returned_job = object()

    class JobServiceSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def start(self, question: str, **options: Any) -> Any:
            self.calls.append((question, options))
            return returned_job

    lazy_runtime = LazyAskRuntime()
    job_service = JobServiceSpy()
    lazy_runtime._job_service = job_service

    result = lazy_runtime.start_ask("question", session_id="session-123")

    assert result is returned_job
    assert job_service.calls == [
        (
            "question",
            {
                "conversation_id": None,
                "on_thread_ref": None,
                "attachment_paths": None,
                "session_id": "session-123",
            },
        )
    ]


def test_lazy_runtime_has_active_jobs_delegates_to_job_service() -> None:
    class JobServiceSpy:
        def __init__(self, result: bool) -> None:
            self.result = result
            self.calls = 0

        def has_active_jobs(self) -> bool:
            self.calls += 1
            return self.result

    runtime = LazyAskRuntime()
    assert runtime.has_active_jobs() is False

    inactive_service = JobServiceSpy(False)
    runtime._job_service = inactive_service
    assert runtime.has_active_jobs() is False
    assert inactive_service.calls == 1

    active_service = JobServiceSpy(True)
    runtime._job_service = active_service
    assert runtime.has_active_jobs() is True
    assert active_service.calls == 1


def test_release_runtime_closes_warm_runtime_and_preserves_jobs() -> None:
    class RuntimeSpy:
        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1

    async def provider(
        question: str,
        *,
        callbacks: ask.AskCallbacks | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del callbacks, conversation_id, timeout_seconds, attachment_paths
        raise AssertionError(f"unexpected provider call: {question}")

    async def scenario() -> None:
        lazy_runtime = LazyAskRuntime()
        warm_runtime = RuntimeSpy()
        job_service = jobs.AskJobService(provider)
        snapshot = jobs.AskJob(
            ask_id="preserved-id",
            state="succeeded",
            answer="answer",
            failure=None,
            error_message=None,
            status_message=None,
            nonce_marker="marker",
            thread_ref=_CONVERSATION_A,
            created_at=1.0,
            finished_at=2.0,
        )
        job_service._jobs[snapshot.ask_id] = snapshot
        lazy_runtime._runtime = warm_runtime
        lazy_runtime._job_service = job_service

        await lazy_runtime.release_runtime()

        assert warm_runtime.close_calls == 1
        assert lazy_runtime._runtime is None
        assert lazy_runtime._job_service is job_service
        assert lazy_runtime.job_status(snapshot.ask_id) == snapshot
        assert lazy_runtime.job_result(snapshot.ask_id) == snapshot
        await lazy_runtime.aclose()

    asyncio.run(scenario())


def test_release_runtime_is_a_no_op_without_a_warm_runtime() -> None:
    async def scenario() -> None:
        lazy_runtime = LazyAskRuntime()

        await lazy_runtime.release_runtime()
        await lazy_runtime.release_runtime()

        assert lazy_runtime._runtime is None
        assert lazy_runtime._job_service is None

    asyncio.run(scenario())


@pytest.mark.parametrize("state", ["queued", "detached"])
def test_status_exposes_nonterminal_job_states(
    monkeypatch: pytest.MonkeyPatch,
    state: jobs.AskJobState,
) -> None:
    runtime = FakeAskRuntime()
    snapshot = jobs.AskJob(
        ask_id="state-id",
        state=state,
        answer=None,
        failure=None,
        error_message=None,
        status_message="progress",
        nonce_marker="nonce-marker",
        thread_ref=_CONVERSATION_A,
        created_at=1.0,
        finished_at=None,
    )
    monkeypatch.setattr(runtime, "job_status", lambda _ask_id: snapshot)

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        status = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_status",
                {"ask_id": snapshot.ask_id},
            )
        )

    assert status == {
        "ask_id": snapshot.ask_id,
        "state": state,
        "status_message": "progress",
        "thread_ref": _CONVERSATION_A,
        "nonce_marker": "nonce-marker",
        "failure": None,
        "error_message": None,
        "evidence": {
            "submission": "not_attempted", "conversation_id": None,
            "upload_receipts": None, "ready_attachments": None,
            "generation_observed": False, "answer_seen": False,
            "raw_extracted": False, "failure_stage": None,
            "recovery_failure": None, "recovery_detail": None,
            "recovery": "not_attempted",
        },
        "source_ask_id": None,
        "recovery_guidance": None,
    }


@pytest.mark.parametrize("nonce_marker", [None, "nonce-marker"])
def test_status_includes_nullable_nonce_marker(
    monkeypatch: pytest.MonkeyPatch,
    nonce_marker: str | None,
) -> None:
    runtime = FakeAskRuntime()
    snapshot = jobs.AskJob(
        ask_id="status-nonce-id",
        state="running",
        answer=None,
        failure=None,
        error_message=None,
        status_message=None,
        nonce_marker=nonce_marker,
        thread_ref=_CONVERSATION_A,
        created_at=1.0,
        finished_at=None,
    )
    monkeypatch.setattr(runtime, "job_status", lambda _ask_id: snapshot)

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        status = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_status",
                {"ask_id": snapshot.ask_id},
            )
        )

    assert "nonce_marker" in status
    assert status["nonce_marker"] == nonce_marker


def test_submit_returns_immediately_and_status_transitions_to_succeeded() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Review this decision."},
            )
        )
        ask_id = submitted["ask_id"]

        assert submitted == {"ask_id": ask_id, "thread_ref": None}
        assert len(ask_id) == 32
        int(ask_id, 16)
        running = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_status",
                {"ask_id": ask_id},
            )
        )
        assert running["ask_id"] == ask_id
        assert running["state"] == "running"
        assert running["status_message"] == "waiting for ChatGPT Pro"
        assert running["thread_ref"] is None
        assert running["nonce_marker"] is None
        assert running["failure"] is None
        assert running["evidence"]["submission"] == "not_attempted"
        assert running["evidence"]["raw_extracted"] is False

        _finish_job(client, runtime, ask_id)
        succeeded = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_status",
                {"ask_id": ask_id},
            )
        )

    assert succeeded["ask_id"] == ask_id
    assert succeeded["state"] == "succeeded"
    assert succeeded["status_message"] is None
    assert succeeded["thread_ref"] == "conversation-123"
    assert succeeded["nonce_marker"] == "nonce-marker"
    assert succeeded["evidence"]["raw_extracted"] is True
    assert succeeded["evidence"]["answer_seen"] is True
    assert succeeded["failure"] is None
    assert runtime.questions == ["Review this decision."]


def test_submit_passes_mcp_session_id_to_runtime() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Trace this session."},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)
        _finish_job(client, runtime, submitted["ask_id"])

    assert runtime.submitted_session_ids == [headers["mcp-session-id"]]


def test_submit_passes_attachment_paths_to_provider() -> None:
    runtime = FakeAskRuntime()
    attachment_paths = ["notes.txt", "context.txt"]
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {
                    "question": "Review the attached context.",
                    "attachments": attachment_paths,
                },
            )
        )
        _wait_for_provider_calls(client, runtime, 1)
        _finish_job(client, runtime, submitted["ask_id"])

    assert runtime.provider_attachment_paths == [attachment_paths]


@pytest.mark.parametrize(
    "attachments",
    ["notes.txt", ["notes.txt", 42]],
    ids=["not-an-array", "non-string-item"],
)
def test_submit_rejects_invalid_attachments(attachments: Any) -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro",
            {"question": "Question", "attachments": attachments},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": "attachments must be an array of strings",
            }
        ],
        "isError": True,
    }
    assert runtime.provider_attachment_paths == []


def test_submit_rejects_thread_ref_argument() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro",
            {"question": "Question", "thread_ref": _CONVERSATION_A},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": (
                    "Unknown argument(s) for ask_gpt_pro: thread_ref — "
                    "expected parameters: question, thread, attachments"
                ),
            }
        ],
        "isError": True,
    }
    assert runtime.questions == []


def test_submit_rejects_unknown_argument_before_question_validation() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro",
            {"unexpected": True},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": (
                    "Unknown argument(s) for ask_gpt_pro: unexpected — "
                    "expected parameters: question, thread, attachments"
                ),
            }
        ],
        "isError": True,
    }
    assert runtime.questions == []


@pytest.mark.parametrize(
    "tool_name", ["ask_gpt_pro_status", "ask_gpt_pro_result"]
)
def test_status_and_result_reject_unknown_arguments_before_ask_id_validation(
    tool_name: str,
) -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            tool_name,
            {"unexpected": True},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Unknown argument(s) for {tool_name}: unexpected — "
                    "expected parameters: ask_id"
                ),
            }
        ],
        "isError": True,
    }


def test_result_returns_settled_answer_and_thread_ref() -> None:
    runtime = FakeAskRuntime(
        answer="## Result\n\nThe settled answer.",
        conversation_id="conversation-from-outcome",
    )
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Question"},
            )
        )
        ask_id = submitted["ask_id"]
        _finish_job(client, runtime, ask_id)
        result = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_result",
                {"ask_id": ask_id},
            )
        )

    assert result["ask_id"] == ask_id
    assert result["answer"] == "## Result\n\nThe settled answer."
    assert result["thread_ref"] == "conversation-from-outcome"
    assert result["nonce_marker"] == "nonce-marker"
    assert result["source_ask_id"] is None
    assert result["evidence"]["raw_extracted"] is True


def test_succeeded_result_includes_nonce_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeAskRuntime()
    snapshot = jobs.AskJob(
        ask_id="result-nonce-id",
        state="succeeded",
        answer="answer",
        failure=None,
        error_message=None,
        status_message=None,
        nonce_marker="result-nonce-marker",
        thread_ref=_CONVERSATION_A,
        created_at=1.0,
        finished_at=2.0,
    )
    monkeypatch.setattr(runtime, "job_result", lambda _ask_id: snapshot)

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_result",
                {"ask_id": snapshot.ask_id},
            )
        )

    assert result["nonce_marker"] == "result-nonce-marker"


def test_result_reports_when_ask_is_still_running() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Question"},
            )
        )
        ask_id = submitted["ask_id"]
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro_result",
            {"ask_id": ask_id},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Ask {ask_id} is still running; poll ask_gpt_pro_status."
                ),
            }
        ],
        "isError": True,
    }


@pytest.mark.parametrize(
    "provider_error",
    [
        ask.GptProSessionExpiredError("expired"),
        ask.GptProAskError("challenge", "blocked"),
        ask.GptProAskError("rate_limited_timeout", "limited"),
        ask.GptProAskError("timeout", "deadline"),
        ask.GptProAskError("echo_timeout", "missing echo"),
        ask.GptProAskError("submit_failed", "button unavailable"),
    ],
)
def test_failed_result_preserves_domain_error_mapping(
    provider_error: ask.GptProAskError,
) -> None:
    runtime = FakeAskRuntime(error=provider_error)
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Question"},
            )
        )
        ask_id = submitted["ask_id"]
        _finish_job(client, runtime, ask_id)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro_result",
            {"ask_id": ask_id},
        )

    # Previous retry-only messages concealed uncertainty; preserve the error
    # flag and require factual classification plus structured diagnostics.
    assert result["isError"] is True
    assert f"[{provider_error.failure}]" in result["content"][0]["text"]
    assert str(provider_error) in result["content"][0]["text"]
    detail = json.loads(result["content"][1]["text"])
    assert detail["failure"] == provider_error.failure
    assert detail["error_message"] == str(provider_error)
    assert detail["ask_id"] == ask_id
    assert detail["evidence"]["raw_extracted"] is False


def test_no_raw_turn_result_explains_thread_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeAskRuntime()
    snapshot = jobs.AskJob(
        ask_id="no-raw-turn-id",
        state="failed",
        answer=None,
        failure="no_raw_turn",
        error_message="no recoverable assistant turn",
        status_message=None,
        nonce_marker="[gptpro-transport-nonce:failed]",
        thread_ref=_CONVERSATION_A,
        created_at=1.0,
        finished_at=2.0,
        evidence=ask.AskEvidence(
            submission="confirmed", conversation_id=_CONVERSATION_A,
            failure_stage="answer", recovery="exhausted",
        ),
    )
    monkeypatch.setattr(runtime, "job_result", lambda _ask_id: snapshot)

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro_result",
            {"ask_id": snapshot.ask_id},
        )

    assert result["isError"] is True
    message = result["content"][0]["text"]
    assert "[no_raw_turn] at answer" in message
    assert "recover_gpt_pro" in message
    assert _CONVERSATION_A in message
    detail = json.loads(result["content"][1]["text"])
    assert detail["evidence"]["recovery"] == "exhausted"
    assert detail["evidence"]["raw_extracted"] is False


def test_omitted_thread_continues_completed_session_conversation() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        session_id = headers["mcp-session-id"]
        first = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Start a conversation."},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)

        assert first["thread_ref"] is None
        assert runtime.provider_conversation_ids == [None]

        _latch_thread(client, runtime, 0, _CONVERSATION_A)
        assert runtime.lookup_thread(session_id) is None

        _finish_job(client, runtime, first["ask_id"])
        assert runtime.lookup_thread(session_id) == _CONVERSATION_A
        second = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Follow up."},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)

    assert second["thread_ref"] == _CONVERSATION_A
    assert runtime.provider_conversation_ids == [None, _CONVERSATION_A]


def test_parallel_fresh_asks_bind_only_the_completed_thread() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        session_id = headers["mcp-session-id"]
        first = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "First.", "thread": "new"},
            )
        )
        second = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Second.", "thread": "new"},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)
        _latch_thread(client, runtime, 0, _CONVERSATION_A)
        _latch_thread(client, runtime, 1, _CONVERSATION_B)

        assert runtime.lookup_thread(session_id) is None

        _finish_job(client, runtime, second["ask_id"])
        first_job = runtime.job_status(first["ask_id"])
        assert first_job is not None
        assert first_job.state == "running"
        assert runtime.lookup_thread(session_id) == _CONVERSATION_B

        continued = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Continue the completed conversation."},
            )
        )
        _wait_for_provider_calls(client, runtime, 3)

    assert continued["thread_ref"] == _CONVERSATION_B
    assert runtime.provider_conversation_ids == [None, None, _CONVERSATION_B]


def test_parallel_fresh_asks_bind_the_last_completed_thread() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        session_id = headers["mcp-session-id"]
        first = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "First.", "thread": "new"},
            )
        )
        second = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Second.", "thread": "new"},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)
        _latch_thread(client, runtime, 0, _CONVERSATION_A)
        _latch_thread(client, runtime, 1, _CONVERSATION_B)

        _finish_job(client, runtime, second["ask_id"])
        assert runtime.lookup_thread(session_id) == _CONVERSATION_B

        _finish_job(client, runtime, first["ask_id"])
        assert runtime.lookup_thread(session_id) == _CONVERSATION_A


def test_failed_ask_preserves_previous_session_binding() -> None:
    runtime = FakeAskRuntime(error=RuntimeError("provider failed"))
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        session_id = headers["mcp-session-id"]
        runtime.bind_thread(session_id, _CONVERSATION_A)
        submitted = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "This will fail.", "thread": "new"},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)
        _latch_thread(client, runtime, 0, _CONVERSATION_B)

        assert runtime.lookup_thread(session_id) == _CONVERSATION_A

        _finish_job(client, runtime, submitted["ask_id"])
        failed_job = runtime.job_status(submitted["ask_id"])
        assert failed_job is not None
        assert failed_job.state == "failed"
        assert runtime.lookup_thread(session_id) == _CONVERSATION_A


def test_new_thread_forces_fresh_conversation_despite_session_binding() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        bound = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Bind this session.", "thread": _CONVERSATION_A},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)
        _finish_job(client, runtime, bound["ask_id"])
        fresh = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Start over.", "thread": "new"},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)

    assert bound["thread_ref"] == _CONVERSATION_A
    assert fresh["thread_ref"] is None
    assert runtime.provider_conversation_ids == [_CONVERSATION_A, None]


def test_explicit_thread_updates_binding_after_successful_completion() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        session_id = headers["mcp-session-id"]
        runtime.bind_thread(session_id, _CONVERSATION_A)
        explicit = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Revisit this.", "thread": _CONVERSATION_B},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)

        assert explicit["thread_ref"] == _CONVERSATION_B
        assert runtime.provider_conversation_ids == [_CONVERSATION_B]
        assert runtime.lookup_thread(session_id) == _CONVERSATION_A

        _finish_job(client, runtime, explicit["ask_id"])
        assert runtime.lookup_thread(session_id) == _CONVERSATION_B
        continued = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro",
                {"question": "Continue."},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)

    assert continued["thread_ref"] == _CONVERSATION_B
    assert runtime.provider_conversation_ids == [
        _CONVERSATION_B,
        _CONVERSATION_B,
    ]


@pytest.mark.parametrize(
    "thread_ref",
    ["not-a-uuid", f"WEB:{_CONVERSATION_A}"],
)
def test_submit_rejects_invalid_thread_reference(thread_ref: str) -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            "ask_gpt_pro",
            {"question": "Question", "thread": thread_ref},
        )

    assert result == {
        "content": [
            {
                "type": "text",
                "text": (
                    'Invalid thread reference: it must be "new" or a '
                    "conversation UUID (a previous thread_ref)."
                ),
            }
        ],
        "isError": True,
    }
    assert runtime.provider_conversation_ids == []


def test_mcp_sessions_keep_thread_bindings_independent() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, session_a_headers = _initialize(client)
        _payload, session_b_headers = _initialize(client)
        assert session_a_headers["mcp-session-id"] != session_b_headers[
            "mcp-session-id"
        ]

        session_a = _json_tool_payload(
            _call_tool(
                client,
                session_a_headers,
                "ask_gpt_pro",
                {"question": "Session A.", "thread": _CONVERSATION_A},
            )
        )
        _wait_for_provider_calls(client, runtime, 1)
        _finish_job(client, runtime, session_a["ask_id"])
        session_b = _json_tool_payload(
            _call_tool(
                client,
                session_b_headers,
                "ask_gpt_pro",
                {"question": "Session B."},
            )
        )
        _wait_for_provider_calls(client, runtime, 2)

    assert session_a["thread_ref"] == _CONVERSATION_A
    assert session_b["thread_ref"] is None
    assert runtime.provider_conversation_ids == [_CONVERSATION_A, None]


@pytest.mark.parametrize(
    "tool_name", ["ask_gpt_pro_status", "ask_gpt_pro_result"]
)
def test_status_and_result_reject_unknown_ask_id(tool_name: str) -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        result = _call_tool(
            client,
            headers,
            tool_name,
            {"ask_id": "missing-id"},
        )

    assert result == {
        "content": [
            {"type": "text", "text": "Unknown or expired ask_id: missing-id"}
        ],
        "isError": True,
    }


def test_mcp_requires_configured_local_token() -> None:
    with _mcp_client(FakeAskRuntime(), local_token="secret") as client:
        response = client.post(
            "/mcp",
            headers=_REQUEST_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["message"] == "Missing or invalid bearer token"


@pytest.mark.parametrize("module_name", ["mcp", None])
def test_mcp_missing_module_returns_installation_recovery_guidance(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str | None,
) -> None:
    async def missing_manager(_endpoint: McpEndpoint, _app: Any) -> Any:
        raise ModuleNotFoundError("Missing required module", name=module_name)

    monkeypatch.setattr(McpEndpoint, "_get_manager", missing_manager)
    with _mcp_client(FakeAskRuntime()) as client:
        response = client.post(
            "/mcp",
            headers=_REQUEST_HEADERS,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "message": (
                "A required module is missing from the gateway installation; "
                "run uv sync in a source checkout or reinstall the latest "
                "release tarball."
            ),
            "type": "service_unavailable_error",
            "param": None,
            "code": None,
        }
    }
    assert "gptpro extra" not in response.text


def test_server_import_and_app_creation_do_not_import_optional_mcp(
    tmp_path: Path,
) -> None:
    code = """
import importlib.abc
import sys

class BlockOptionalMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "mcp" or fullname.startswith("mcp.") or fullname == "mcp_types":
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOptionalMcp())
from starlette.testclient import TestClient
from claudex.config import GatewayConfig
from claudex.server import create_app

app = create_app(GatewayConfig())
assert any(getattr(route, "path", None) == "/mcp" for route in app.routes)
with TestClient(app, base_url="http://localhost") as client:
    assert client.get("/api/hello").status_code == 200
assert "mcp" not in sys.modules
assert "mcp_types" not in sys.modules
"""
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_recovery_tool_rejects_missing_identifiers_without_submitting() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _response, headers = _initialize(client)
        result = _call_tool(client, headers, "recover_gpt_pro", {})
        assert result["isError"] is True
        assert "ask_id or both thread_ref and nonce_marker" in result["content"][0]["text"]
        assert runtime.questions == []


def test_recovery_tool_accepts_explicit_identifiers_after_restart_without_prompt() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _response, headers = _initialize(client)
        result = _call_tool(client, headers, "recover_gpt_pro", {
            "thread_ref": _CONVERSATION_A,
            "nonce_marker": "[gptpro-transport-nonce:existing]",
        })
        payload = _json_tool_payload(result)
        assert payload["thread_ref"] == _CONVERSATION_A
        assert isinstance(payload["ask_id"], str)
        assert runtime.questions == []


def test_failed_status_and_result_include_diagnostics_and_identifiers() -> None:
    runtime = FakeAskRuntime(
        error=ask.GptProAskError("echo_timeout", "echo unavailable"),
        conversation_id=_CONVERSATION_A,
    )
    with _mcp_client(runtime) as client:
        _response, headers = _initialize(client)
        submitted = _json_tool_payload(_call_tool(client, headers, "ask_gpt_pro", {
            "question": "one", "thread": _CONVERSATION_A,
        }))
        ask_id = submitted["ask_id"]
        _finish_job(client, runtime, ask_id)
        status = _json_tool_payload(_call_tool(client, headers, "ask_gpt_pro_status", {
            "ask_id": ask_id,
        }))
        result = _call_tool(client, headers, "ask_gpt_pro_result", {
            "ask_id": ask_id,
        })
        assert status["failure"] == "echo_timeout"
        assert status["thread_ref"] == _CONVERSATION_A
        assert status["evidence"]["failure_stage"] == "echo"
        assert result["isError"] is True
        detail = json.loads(result["content"][1]["text"])
        assert detail["failure"] == "echo_timeout"
        assert detail["thread_ref"] == _CONVERSATION_A
        assert detail["evidence"]["submission"] != "not_attempted"
        assert "recover_gpt_pro" in detail["recovery_guidance"]


def test_recover_failed_ask_read_only_and_poll_new_job() -> None:
    runtime = FakeAskRuntime(
        error=ask.GptProAskError("echo_timeout", "confirmation lost"),
        conversation_id=_CONVERSATION_A,
    )
    with _mcp_client(runtime) as client:
        _response, headers = _initialize(client)
        original = _json_tool_payload(_call_tool(client, headers, "ask_gpt_pro", {
            "question": "original", "thread": _CONVERSATION_A,
        }))
        _finish_job(client, runtime, original["ask_id"])
        recovered = _json_tool_payload(_call_tool(
            client, headers, "recover_gpt_pro", {"ask_id": original["ask_id"]},
        ))
        assert recovered["source_ask_id"] == original["ask_id"]
        assert recovered["thread_ref"] == _CONVERSATION_A
        assert client.portal is not None
        async def wait_for_recovery() -> None:
            while runtime.job_status(recovered["ask_id"]).state != "succeeded":
                await asyncio.sleep(0)
        client.portal.call(wait_for_recovery)
        result = _json_tool_payload(_call_tool(client, headers, "ask_gpt_pro_result", {
            "ask_id": recovered["ask_id"],
        }))
        original_status = _json_tool_payload(_call_tool(client, headers, "ask_gpt_pro_status", {
            "ask_id": original["ask_id"],
        }))
        assert result["answer"] == runtime.answer
        assert result["source_ask_id"] == original["ask_id"]
        assert result["evidence"]["recovery"] == "recovered"
        assert original_status["state"] == "failed"
        assert runtime.questions == ["original"]


@pytest.mark.parametrize("arguments, error_fragment", [
    ({"ask_id": "unknown"}, "unknown or expired ask_id"),
    ({"thread_ref": _CONVERSATION_A}, "both thread_ref and nonce_marker"),
    ({"ask_id": "x", "thread_ref": _CONVERSATION_A}, "not both"),
    ({"thread_ref": "new", "nonce_marker": "[gptpro-transport-nonce:saved]"},
     "thread_ref UUID"),
    ({"thread_ref": _CONVERSATION_A, "nonce_marker": "arbitrary"},
     "nonce_marker"),
    ({"thread_ref": _CONVERSATION_A,
      "nonce_marker": "[gptpro-transport-nonce:saved]", "question": "oops"},
     "Unknown argument(s)"),
])
def test_recovery_validation_never_submits(
    arguments: dict[str, str], error_fragment: str,
) -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _response, headers = _initialize(client)
        result = _call_tool(client, headers, "recover_gpt_pro", arguments)
        assert result["isError"] is True
        assert error_fragment in result["content"][0]["text"]
        assert runtime.questions == []


def test_succeeded_result_without_files_reports_complete_empty_files() -> None:
    runtime = FakeAskRuntime()

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        submitted = _json_tool_payload(
            _call_tool(client, headers, "ask_gpt_pro", {"question": "Question"})
        )
        _finish_job(client, runtime, submitted["ask_id"])
        result = _json_tool_payload(
            _call_tool(
                client, headers, "ask_gpt_pro_result",
                {"ask_id": submitted["ask_id"]},
            )
        )

    assert result.get("files") == []
    assert result.get("files_complete") is True
    assert result["answer"] == runtime.answer


def test_succeeded_result_lists_generated_files_and_incompleteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    @dataclass(frozen=True)
    class _File:
        name: str
        status: str
        path: str | None
        error: str | None

    runtime = FakeAskRuntime()
    snapshot = SimpleNamespace(
        ask_id="result-files-id", state="succeeded", answer="answer",
        failure=None, error_message=None, status_message=None,
        nonce_marker="marker", thread_ref=_CONVERSATION_A,
        created_at=1.0, finished_at=2.0, evidence=ask.AskEvidence(),
        source_ask_id=None,
        files=(
            _File("report.zip", "saved", "/gateway/outputs/report.zip", None),
            _File("notes.md", "failed", None, "HTTP 403"),
        ),
        files_complete=False,
    )
    monkeypatch.setattr(runtime, "job_result", lambda _ask_id: snapshot)

    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        result = _json_tool_payload(
            _call_tool(
                client, headers, "ask_gpt_pro_result", {"ask_id": snapshot.ask_id}
            )
        )

    assert result.get("files") == [
        {"name": "report.zip", "status": "saved",
         "path": "/gateway/outputs/report.zip", "error": None},
        {"name": "notes.md", "status": "failed", "path": None,
         "error": "HTTP 403"},
    ]
    assert result.get("files_complete") is False
    assert result["answer"] == "answer"


def test_result_tool_description_explains_gateway_host_file_paths() -> None:
    from claudex import mcp_tools

    description = mcp_tools.ASK_GPT_PRO_RESULT_DESCRIPTION
    assert "files" in description
    assert "gateway host" in description
    assert "files_complete" in description
    assert "sandbox" in description


def test_lazy_runtime_default_concurrency() -> None:
    assert LazyAskRuntime().max_concurrent_asks == 2


def test_lazy_runtime_passes_configured_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from claudex.gptpro import runtime

    captured: list[dict[str, Any]] = []

    class RuntimeSpy:
        def __init__(self, **options: Any) -> None:
            captured.append(options)

    monkeypatch.setattr(runtime, "AskRuntime", RuntimeSpy)
    lazy_runtime = LazyAskRuntime()
    lazy_runtime.set_max_concurrent_asks(3)
    assert lazy_runtime.max_concurrent_asks == 3
    created = asyncio.run(lazy_runtime._get_runtime())
    assert isinstance(created, RuntimeSpy)
    assert captured == [{"max_concurrent_asks": 3}]
