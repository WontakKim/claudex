"""Lazy streamable HTTP MCP exposure for the ChatGPT Pro ask runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

ASK_GPT_PRO_DESCRIPTION = (
    "Send a self-contained question to ChatGPT Pro and return its settled "
    "Markdown answer. Include all necessary code, logs, metadata, and context "
    "inline in question. If the response begins with GPTPRO_CONTEXT_REQUEST_V1, "
    "gather the requested material and call this tool again with it included. "
    "Use this tool only when the user explicitly requests ChatGPT Pro or for a "
    "consequential judgment where a second opinion materially helps; do not use "
    "it routinely."
)
ASK_GPT_PRO_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
    "additionalProperties": False,
}


class LazyAskRuntime:
    """Create the optional ChatGPT Pro runtime on its first ask."""

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._lock = asyncio.Lock()

    async def ask(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        runtime = await self._get_runtime()
        return await runtime.ask(question, on_status=on_status)

    async def aclose(self) -> None:
        async with self._lock:
            runtime = self._runtime
            self._runtime = None
            if runtime is not None:
                await runtime.aclose()

    async def _get_runtime(self) -> Any:
        async with self._lock:
            if self._runtime is None:
                from claudex.gptpro.runtime import AskRuntime

                self._runtime = AskRuntime()
            return self._runtime


class McpEndpoint:
    """Start and serve the optional MCP transport only after its first request."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._manager: Any | None = None
        self._manager_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._startup_error: BaseException | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive=receive)
        from claudex.admin.common import _admin_guard

        denied = _admin_guard(request)
        if denied is not None:
            await denied(scope, receive, send)
            return

        try:
            manager = await self._get_manager(request.app)
        except ModuleNotFoundError:
            response = JSONResponse(
                {
                    "error": {
                        "message": (
                            "MCP support is not installed; install the gptpro extra"
                        ),
                        "type": "service_unavailable_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)

    async def aclose(self) -> None:
        async with self._lock:
            task = self._manager_task
            stop = self._stop
            if task is None or stop is None:
                return
            stop.set()
        try:
            await task
        finally:
            async with self._lock:
                if self._manager_task is task:
                    self._manager = None
                    self._manager_task = None
                    self._ready = None
                    self._stop = None
                    self._startup_error = None

    async def _get_manager(self, app: Any) -> Any:
        async with self._lock:
            if self._manager_task is None:
                manager = _create_session_manager(app)
                self._manager = manager
                self._ready = asyncio.Event()
                self._stop = asyncio.Event()
                self._manager_task = asyncio.create_task(
                    self._run_manager(manager, self._ready, self._stop)
                )
            ready = self._ready

        assert ready is not None
        await ready.wait()
        if self._startup_error is not None:
            raise self._startup_error
        assert self._manager is not None
        return self._manager

    async def _run_manager(
        self,
        manager: Any,
        ready: asyncio.Event,
        stop: asyncio.Event,
    ) -> None:
        try:
            async with manager.run():
                ready.set()
                await stop.wait()
        except BaseException as exc:
            self._startup_error = exc
            ready.set()
            raise


def _create_session_manager(app: Any) -> Any:
    """Build the MCP SDK server and transport without importing it at boot."""
    import mcp_types as types
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    async def list_tools(_context: Any, _params: Any) -> Any:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="ask_gpt_pro",
                    description=ASK_GPT_PRO_DESCRIPTION,
                    inputSchema=ASK_GPT_PRO_INPUT_SCHEMA,
                )
            ]
        )

    async def call_tool(context: Any, params: Any) -> Any:
        if params.name != "ask_gpt_pro":
            return _tool_error(types, f"Unknown tool: {params.name}")
        arguments = params.arguments
        question = arguments.get("question") if isinstance(arguments, dict) else None
        if not isinstance(question, str):
            return _tool_error(types, "question must be a string")

        progress_tasks: list[asyncio.Task[None]] = []
        progress = 0

        def on_status(message: str) -> None:
            nonlocal progress
            progress += 1
            task = asyncio.create_task(
                context.session.report_progress(float(progress), message=message)
            )
            progress_tasks.append(task)

        try:
            runtime = app.state.gptpro_ask_runtime
            answer = await runtime.ask(question, on_status=on_status)
        except Exception as exc:
            result = _gptpro_error_result(types, exc)
            if result is None:
                raise
            return result
        finally:
            if progress_tasks:
                await asyncio.gather(*progress_tasks, return_exceptions=True)

        return types.CallToolResult(content=[types.TextContent(text=answer)])

    server = Server(
        "claudex-gateway-gptpro",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    return StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=False,
    )


def _gptpro_error_result(types: Any, exc: Exception) -> Any | None:
    from claudex.gptpro.ask import GptProAskError

    if not isinstance(exc, GptProAskError):
        return None

    failure = exc.failure
    if failure == "session_expired":
        message = (
            "ChatGPT Pro session expired; run claudex-gateway gptpro login, then retry."
        )
    elif failure == "challenge":
        message = (
            "ChatGPT Pro browser challenge blocked the request; complete the "
            "challenge with claudex-gateway gptpro login, then retry."
        )
    elif failure == "rate_limited_timeout":
        message = "ChatGPT Pro remained rate limited until timeout; retry later."
    elif failure in {"timeout", "echo_timeout"}:
        message = (
            "ChatGPT Pro request timed out; check ChatGPT and the network, then retry."
        )
    else:
        message = f"ChatGPT Pro request failed [{failure}]: {exc}"
    return _tool_error(types, message)


def _tool_error(types: Any, message: str) -> Any:
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        isError=True,
    )
