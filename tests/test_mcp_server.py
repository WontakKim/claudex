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
from claudex.mcp_server import McpEndpoint

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
        self._conversation_id_callbacks: list[
            Callable[[str], None] | None
        ] = []
        self._provider_calls_changed = asyncio.Condition()
        self._release_provider = asyncio.Event()
        self._job_service = jobs.AskJobService(self._ask)
        self._thread_registry = jobs.ThreadRegistry()

    async def _ask(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> ask.AskOutcome:
        del timeout_seconds
        async with self._provider_calls_changed:
            self.questions.append(question)
            self.provider_conversation_ids.append(conversation_id)
            self.provider_attachment_paths.append(attachment_paths)
            self._conversation_id_callbacks.append(on_conversation_id)
            self._provider_calls_changed.notify_all()
        if on_status is not None:
            on_status("waiting for ChatGPT Pro")
        await self._release_provider.wait()
        if self.error is not None:
            raise self.error
        return ask.AskOutcome(
            text=self.answer,
            marker="nonce-marker",
            conversation_id=self.conversation_id,
        )

    def start_ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> jobs.AskJob:
        return self._job_service.start(
            question,
            conversation_id=conversation_id,
            on_thread_ref=on_thread_ref,
            attachment_paths=attachment_paths,
        )

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
        self._release_provider.set()
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
    }
    assert tools["ask_gpt_pro"]["inputSchema"] == {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "thread": {
                "type": "string",
                "description": (
                    "'new' forces a fresh conversation; a conversation UUID (a "
                    "previous thread_ref) continues that conversation; omit to "
                    "continue this session's current conversation."
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
    assert "self-contained whenever possible" in ask_description
    assert "up to 10 UTF-8 plain-text files totaling 1.2 MB" in ask_description
    assert "Questions over ~35 KB" in ask_description
    assert "send the full text" in ask_description
    assert "Omitting thread continues this MCP session's current conversation" in (
        ask_description
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
    assert '"waiting for the in-flight answer"' in status_description
    assert "settled result" in tools["ask_gpt_pro_result"]["description"]


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
        assert running == {
            "ask_id": ask_id,
            "state": "running",
            "status_message": "waiting for ChatGPT Pro",
            "thread_ref": None,
        }

        _finish_job(client, runtime, ask_id)
        succeeded = _json_tool_payload(
            _call_tool(
                client,
                headers,
                "ask_gpt_pro_status",
                {"ask_id": ask_id},
            )
        )

    assert succeeded == {
        "ask_id": ask_id,
        "state": "succeeded",
        "status_message": "waiting for ChatGPT Pro",
        "thread_ref": "conversation-123",
    }
    assert runtime.questions == ["Review this decision."]


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

    assert result == {
        "ask_id": ask_id,
        "answer": "## Result\n\nThe settled answer.",
        "thread_ref": "conversation-from-outcome",
    }


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
    ("provider_error", "expected_message"),
    [
        (
            ask.GptProSessionExpiredError("expired"),
            "ChatGPT Pro session expired; run claudex-gateway gptpro login, then retry.",
        ),
        (
            ask.GptProAskError("challenge", "blocked"),
            "ChatGPT Pro browser challenge blocked the request; complete the "
            "challenge with claudex-gateway gptpro login, then retry.",
        ),
        (
            ask.GptProAskError("rate_limited_timeout", "limited"),
            "ChatGPT Pro remained rate limited until timeout; retry later.",
        ),
        (
            ask.GptProAskError("timeout", "deadline"),
            "ChatGPT Pro request timed out; check ChatGPT and the network, then retry.",
        ),
        (
            ask.GptProAskError("echo_timeout", "missing echo"),
            "ChatGPT Pro request timed out; check ChatGPT and the network, then retry.",
        ),
        (
            ask.GptProAskError("submit_failed", "button unavailable"),
            "ChatGPT Pro request failed [submit_failed]: button unavailable",
        ),
    ],
)
def test_failed_result_preserves_domain_error_mapping(
    provider_error: Exception,
    expected_message: str,
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

    assert result == {
        "content": [{"type": "text", "text": expected_message}],
        "isError": True,
    }


def test_omitted_thread_continues_latched_session_conversation() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
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
        _finish_job(client, runtime, first["ask_id"])
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


def test_explicit_thread_updates_session_binding_for_next_omitted_thread() -> None:
    runtime = FakeAskRuntime()
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
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

        _finish_job(client, runtime, explicit["ask_id"])
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
