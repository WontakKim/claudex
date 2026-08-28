"""MCP tool definitions and call handling for the ChatGPT Pro ask runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from claudex.gptpro import conversation

ASK_GPT_PRO_DESCRIPTION = (
    "Send a self-contained question to ChatGPT Pro as a background job and return "
    "immediately with {ask_id, thread_ref}. Poll "
    "ask_gpt_pro_status(ask_id) while state is queued, running, or detached; "
    "detached asks remain recoverable and normally return through polling. When "
    "state is succeeded or failed, fetch ask_gpt_pro_result(ask_id) for the "
    "settled Markdown answer or failure. An expired failure means same-conversation "
    "queue admission timed out; use the preserved thread_ref, and nonce marker "
    "when available, to revisit the conversation and attempt answer recovery. "
    "Include all necessary code, logs, metadata, and context inline so the "
    "question is self-contained whenever possible. Attach up to 10 UTF-8 "
    "plain-text files totaling 1.2 MB with attachments. Questions over ~35 KB "
    "are spilled into an attachment automatically, so send the full text. "
    "Omitting thread continues this MCP session's most recently completed "
    "conversation, or starts a new one when the session is unbound, so ChatGPT "
    "can see previous turns and short follow-up questions may rely on them. Set "
    "thread to \"new\" to force a fresh conversation, or pass a conversation "
    "UUID from a previous thread_ref to revisit that conversation; separate MCP "
    "sessions share a conversation only when they explicitly pass the same UUID. "
    "Asks in the same conversation are serialized, and status reports that an "
    "ask is waiting while the previous ask finishes without blocking asks in "
    "other conversations. If the answer begins with GPTPRO_CONTEXT_REQUEST_V1, "
    "gather the requested material and call again — omitting thread continues "
    "the conversation. Use this tool only when the user explicitly requests "
    "ChatGPT Pro or for a consequential judgment where a second opinion "
    "materially helps; do not use it routinely."
)
ASK_GPT_PRO_STATUS_DESCRIPTION = (
    "Poll a background ChatGPT Pro ask and return its current state, latest "
    "status message, and thread_ref. The thread_ref is preserved throughout the "
    "job lifecycle. queued means awaiting admission, normally "
    "because the previous ask in the same conversation is still generating; "
    "asks in other conversations can continue in parallel. running means the "
    "ask was submitted and ChatGPT is generating the answer. detached means "
    "server-side generation continues while the gateway polls to recover the "
    "answer, which normally returns through the job. succeeded and failed are "
    "terminal. For failed jobs, failure=expired means same-conversation queue "
    "waiting reached its TTL; thread_ref and any available nonce marker are "
    "preserved so callers can revisit that conversation with thread and attempt "
    "answer recovery. A thread_ref can appear before completion, but it becomes "
    "this MCP session's binding only after the ask succeeds. status_message "
    "values such as \"waiting for the in-flight answer\" and \"detached; "
    "polling for the answer\" remain supplemental progress details."
)
ASK_GPT_PRO_RESULT_DESCRIPTION = (
    "Fetch the settled result of a background ChatGPT Pro ask only after its "
    "status is succeeded or failed; queued, running, and detached are still in "
    "progress. For failure=expired, use the preserved thread_ref and any nonce "
    "marker to revisit the conversation and attempt answer recovery."
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


def build_gptpro_server(app: Any) -> Any:
    """Build the MCP SDK server without importing it at boot."""
    import mcp_types as types
    from mcp.server import Server

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
                session_id=session_id,
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
            if job.state in {"queued", "running", "detached"}:
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
    return server


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
