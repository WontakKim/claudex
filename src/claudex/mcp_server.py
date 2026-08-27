"""Lazy streamable HTTP MCP exposure for the ChatGPT Pro ask runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from claudex.gptpro import conversation, jobs

ASK_GPT_PRO_DESCRIPTION = (
    "Send a self-contained question to ChatGPT Pro as a background job and return "
    "immediately with {ask_id, thread_ref}. Poll "
    "ask_gpt_pro_status(ask_id) until state is succeeded or failed, then fetch "
    "ask_gpt_pro_result(ask_id) for the settled Markdown answer. Include all "
    "necessary code, logs, metadata, and context inline so the question is "
    "self-contained whenever possible. Attach up to 10 UTF-8 plain-text "
    "files totaling 1.2 MB with attachments. Questions over ~35 KB are "
    "spilled into an attachment automatically, so send the full text. "
    "Omitting thread continues this MCP session's most recently completed "
    "conversation, or starts a new one when the session is "
    "unbound, so ChatGPT can see previous turns and short follow-up questions "
    "may rely on them. Set thread to \"new\" to force a fresh conversation, or "
    "pass a conversation UUID from a previous thread_ref to revisit that "
    "conversation; separate MCP sessions share a conversation only when they "
    "explicitly pass the same UUID. Asks in the same conversation are serialized, "
    "and status reports that an ask is waiting while the previous ask finishes. "
    "If the answer begins with GPTPRO_CONTEXT_REQUEST_V1, gather the requested "
    "material and call again — omitting thread continues the conversation. Use "
    "this tool only when the user explicitly requests ChatGPT Pro or for a "
    "consequential judgment where a second opinion materially helps; do not use "
    "it routinely."
)
ASK_GPT_PRO_STATUS_DESCRIPTION = (
    "Poll a background ChatGPT Pro ask and return its current state, latest "
    "status message, and thread_ref. The thread_ref is preserved throughout the "
    "job lifecycle. A thread_ref can appear while an ask is running, but it "
    "becomes this MCP session's binding only after the ask succeeds. A "
    "status_message of \"waiting for the in-flight answer\" means the previous "
    "ask in the same conversation is still finishing."
)
ASK_GPT_PRO_RESULT_DESCRIPTION = (
    "Fetch the settled result of a background ChatGPT Pro ask after its status "
    "is succeeded or failed."
)
_ATTACHMENTS_DESCRIPTION = (
    "Optional plain-text file paths to attach (UTF-8 only; at most 10 files "
    "and 1.2 MB total; questions over ~35 KB are spilled into an attachment "
    "automatically, so send full text)."
)
_THREAD_DESCRIPTION = (
    "'new' forces a fresh conversation; a conversation UUID (a previous "
    "thread_ref) continues that conversation; omit to continue this session's "
    "most recently completed conversation."
)
ASK_GPT_PRO_ALLOWED_ARGUMENT_KEYS = ("question", "thread", "attachments")
ASK_GPT_PRO_STATUS_ALLOWED_ARGUMENT_KEYS = ("ask_id",)
ASK_GPT_PRO_RESULT_ALLOWED_ARGUMENT_KEYS = ("ask_id",)
ASK_GPT_PRO_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "thread": {"type": "string", "description": _THREAD_DESCRIPTION},
        "attachments": {
            "type": "array",
            "items": {"type": "string"},
            "description": _ATTACHMENTS_DESCRIPTION,
        },
    },
    "required": ["question"],
    "additionalProperties": False,
}
ASK_GPT_PRO_STATUS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ask_id": {"type": "string"}},
    "required": ["ask_id"],
    "additionalProperties": False,
}
ASK_GPT_PRO_RESULT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ask_id": {"type": "string"}},
    "required": ["ask_id"],
    "additionalProperties": False,
}


class LazyAskRuntime:
    """Create the optional ChatGPT Pro runtime on its first background ask."""

    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._job_service: jobs.AskJobService | None = None
        self._thread_registry = jobs.ThreadRegistry()
        self._lock = asyncio.Lock()

    async def ask(
        self,
        question: str,
        *,
        on_status: Callable[[str], None] | None = None,
        on_conversation_id: Callable[[str], None] | None = None,
        on_marker: Callable[[str], None] | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> Any:
        runtime = await self._get_runtime()
        return await runtime.ask(
            question,
            on_status=on_status,
            on_conversation_id=on_conversation_id,
            on_marker=on_marker,
            conversation_id=conversation_id,
            timeout_seconds=timeout_seconds,
            attachment_paths=attachment_paths,
        )

    def start_ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        on_thread_ref: Callable[[str], None] | None = None,
        attachment_paths: Sequence[str] | None = None,
    ) -> jobs.AskJob:
        if self._job_service is None:
            self._job_service = jobs.AskJobService(self.ask)
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
        if self._job_service is None:
            return None
        return self._job_service.status(ask_id)

    def job_result(self, ask_id: str) -> jobs.AskJob | None:
        if self._job_service is None:
            return None
        return self._job_service.result(ask_id)

    async def aclose(self) -> None:
        async with self._lock:
            job_service = self._job_service
            self._job_service = None
            if job_service is not None:
                await job_service.aclose()

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
                ),
                types.Tool(
                    name="ask_gpt_pro_status",
                    description=ASK_GPT_PRO_STATUS_DESCRIPTION,
                    inputSchema=ASK_GPT_PRO_STATUS_INPUT_SCHEMA,
                ),
                types.Tool(
                    name="ask_gpt_pro_result",
                    description=ASK_GPT_PRO_RESULT_DESCRIPTION,
                    inputSchema=ASK_GPT_PRO_RESULT_INPUT_SCHEMA,
                ),
            ]
        )

    async def call_tool(context: Any, params: Any) -> Any:
        arguments = params.arguments
        if not isinstance(arguments, dict):
            arguments = {}
        runtime = app.state.gptpro_ask_runtime

        if params.name == "ask_gpt_pro":
            unknown_argument_error = _get_unknown_argument_error(
                types,
                tool_name="ask_gpt_pro",
                arguments=arguments,
                allowed_argument_keys=ASK_GPT_PRO_ALLOWED_ARGUMENT_KEYS,
            )
            if unknown_argument_error is not None:
                return unknown_argument_error
            question = arguments.get("question")
            if not isinstance(question, str):
                return _tool_error(types, "question must be a string")
            attachment_paths = None
            if "attachments" in arguments:
                requested_attachments = arguments["attachments"]
                if not isinstance(requested_attachments, list) or not all(
                    isinstance(attachment_path, str)
                    for attachment_path in requested_attachments
                ):
                    return _tool_error(
                        types, "attachments must be an array of strings"
                    )
                attachment_paths = requested_attachments
            session_id = _get_mcp_session_id(context)
            if "thread" not in arguments:
                conversation_id = (
                    runtime.lookup_thread(session_id)
                    if session_id is not None
                    else None
                )
            elif arguments["thread"] == "new":
                conversation_id = None
            elif not conversation.is_conversation_id(arguments["thread"]):
                return _tool_error(
                    types,
                    "Invalid thread reference: it must be \"new\" or a "
                    "conversation UUID (a previous thread_ref).",
                )
            else:
                conversation_id = arguments["thread"]

            on_thread_ref = None
            if session_id is not None:

                def on_thread_ref(thread_ref: str) -> None:
                    runtime.bind_thread(session_id, thread_ref)

            job = runtime.start_ask(
                question,
                conversation_id=conversation_id,
                on_thread_ref=on_thread_ref,
                attachment_paths=attachment_paths,
            )
            return _json_result(
                types,
                {"ask_id": job.ask_id, "thread_ref": job.thread_ref},
            )

        if params.name == "ask_gpt_pro_status":
            unknown_argument_error = _get_unknown_argument_error(
                types,
                tool_name="ask_gpt_pro_status",
                arguments=arguments,
                allowed_argument_keys=ASK_GPT_PRO_STATUS_ALLOWED_ARGUMENT_KEYS,
            )
            if unknown_argument_error is not None:
                return unknown_argument_error
            ask_id = arguments.get("ask_id")
            if not isinstance(ask_id, str):
                return _tool_error(types, "ask_id must be a string")
            job = runtime.job_status(ask_id)
            if job is None:
                return _tool_error(types, f"Unknown or expired ask_id: {ask_id}")
            return _json_result(
                types,
                {
                    "ask_id": job.ask_id,
                    "state": job.state,
                    "status_message": job.status_message,
                    "thread_ref": job.thread_ref,
                },
            )

        if params.name == "ask_gpt_pro_result":
            unknown_argument_error = _get_unknown_argument_error(
                types,
                tool_name="ask_gpt_pro_result",
                arguments=arguments,
                allowed_argument_keys=ASK_GPT_PRO_RESULT_ALLOWED_ARGUMENT_KEYS,
            )
            if unknown_argument_error is not None:
                return unknown_argument_error
            ask_id = arguments.get("ask_id")
            if not isinstance(ask_id, str):
                return _tool_error(types, "ask_id must be a string")
            job = runtime.job_result(ask_id)
            if job is None:
                return _tool_error(types, f"Unknown or expired ask_id: {ask_id}")
            if job.state == "running":
                return _tool_error(
                    types,
                    f"Ask {ask_id} is still running; poll ask_gpt_pro_status.",
                )
            if job.state == "failed":
                return _gptpro_error_result(
                    types, job.failure, job.error_message
                )
            return _json_result(
                types,
                {
                    "ask_id": job.ask_id,
                    "answer": job.answer,
                    "thread_ref": job.thread_ref,
                },
            )

        return _tool_error(types, f"Unknown tool: {params.name}")

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


def _get_mcp_session_id(context: Any) -> str | None:
    try:
        request = getattr(context, "request", None)
        headers = getattr(request, "headers", None)
        session_id = headers.get("mcp-session-id") if headers is not None else None
    except (AttributeError, TypeError):
        return None
    return session_id if isinstance(session_id, str) and session_id else None


def _gptpro_error_result(
    types: Any, failure: str | None, error_message: str | None
) -> Any:
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
        failure_name = failure if failure is not None else "error"
        detail = error_message if error_message is not None else "unknown error"
        message = f"ChatGPT Pro request failed [{failure_name}]: {detail}"
    return _tool_error(types, message)


def _json_result(types: Any, payload: dict[str, Any]) -> Any:
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(payload))]
    )


def _get_unknown_argument_error(
    types: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    allowed_argument_keys: Sequence[str],
) -> Any | None:
    unknown_argument_keys = sorted(set(arguments) - set(allowed_argument_keys))
    if not unknown_argument_keys:
        return None
    return _tool_error(
        types,
        f"Unknown argument(s) for {tool_name}: "
        f"{', '.join(unknown_argument_keys)} — expected parameters: "
        f"{', '.join(allowed_argument_keys)}",
    )


def _tool_error(types: Any, message: str) -> Any:
    return types.CallToolResult(
        content=[types.TextContent(text=message)],
        isError=True,
    )
