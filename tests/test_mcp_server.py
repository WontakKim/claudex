"""Protocol and optional-dependency tests for the ChatGPT Pro MCP endpoint."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from claudex.config import GatewayConfig
from claudex.gptpro.ask import GptProSessionExpiredError
from claudex.mcp_server import McpEndpoint

_PROTOCOL_VERSION = "2025-06-18"
_REQUEST_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}


class FakeAskRuntime:
    def __init__(
        self,
        *,
        answer: str = "# Answer\n\nSettled Markdown.",
        error: Exception | None = None,
    ) -> None:
        self.answer = answer
        self.error = error
        self.questions: list[str] = []

    async def ask(
        self,
        question: str,
        *,
        on_status: Any = None,
    ) -> str:
        self.questions.append(question)
        if on_status is not None:
            on_status("waiting for ChatGPT Pro")
        if self.error is not None:
            raise self.error
        return self.answer


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


def test_initialize_handshake_advertises_tools() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        payload, _headers = _initialize(client)

    assert payload["result"]["protocolVersion"] == _PROTOCOL_VERSION
    assert payload["result"]["serverInfo"]["name"] == "claudex-gateway-gptpro"
    assert payload["result"]["capabilities"]["tools"] == {"listChanged": False}


def test_tools_list_exposes_ask_gpt_pro_schema_and_usage_guidance() -> None:
    with _mcp_client(FakeAskRuntime()) as client:
        _payload, headers = _initialize(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )

    assert response.status_code == 200
    tools = response.json()["result"]["tools"]
    assert len(tools) == 1
    tool = tools[0]
    assert tool["name"] == "ask_gpt_pro"
    assert tool["inputSchema"] == {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    }
    assert "self-contained" in tool["description"]
    assert "GPTPRO_CONTEXT_REQUEST_V1" in tool["description"]
    assert "explicitly requests ChatGPT Pro" in tool["description"]


def test_tools_call_returns_settled_markdown() -> None:
    runtime = FakeAskRuntime(answer="## Result\n\nThe settled answer.")
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ask_gpt_pro",
                    "arguments": {"question": "Review this decision."},
                },
            },
        )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result == {
        "content": [{"type": "text", "text": "## Result\n\nThe settled answer."}],
        "isError": False,
    }
    assert runtime.questions == ["Review this decision."]


def test_tools_call_converts_domain_error_to_error_result() -> None:
    runtime = FakeAskRuntime(error=GptProSessionExpiredError("expired"))
    with _mcp_client(runtime) as client:
        _payload, headers = _initialize(client)
        response = client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "ask_gpt_pro",
                    "arguments": {"question": "Question"},
                },
            },
        )

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["content"][0]["type"] == "text"
    assert "run claudex-gateway gptpro login" in result["content"][0]["text"]


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
