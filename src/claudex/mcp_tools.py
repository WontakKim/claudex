"""MCP tool definitions and call handling for the ChatGPT Pro ask runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from claudex.gptpro import conversation
from claudex.gptpro.jobs.models import AskJob

ASK_GPT_PRO_DESCRIPTION = (
    "Send a self-contained question to ChatGPT Pro as a background job and return "
    "immediately with {ask_id, thread_ref}. Poll "
    "ask_gpt_pro_status(ask_id) while state is queued, running, or detached; "
    "detached asks remain recoverable and normally return through polling. When "
    "state is succeeded or failed, fetch ask_gpt_pro_result(ask_id) for the "
    "settled Markdown answer or failure. An expired failure means same-conversation "
    "queue admission ended before this ask submitted; a recovery job that expires "
    "in queue may still target an existing server turn. For an uncertain submitted "
    "turn, use recover_gpt_pro with a failed ask_id or a saved thread_ref and "
    "nonce_marker; recovery never submits another prompt. "
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
_GPTPRO_FAILURE_ACTIONS_DESCRIPTION = (
    "Failure actions:\n"
    "- expired: queue admission ended before this job ran; a recovery job may "
    "still refer to an earlier submitted turn.\n"
    "- no_raw_turn: recovery polling did not obtain a finished nonce-correlated answer.\n"
    "- timeout / echo_timeout: submission may have reached ChatGPT; use read-only recovery.\n"
    "- rate_limited_timeout: wait, then use read-only recovery if submitted.\n"
    "- session_expired / challenge: run claudex-gateway gptpro login; "
    "recover a known turn rather than resubmitting blindly.\n"
    "- locator_unresolved: composer or send button was not found before submission; "
    "nothing was sent, retrying is safe, and repeated failures need a gateway update.\n"
    "- submit_failed / navigation_failed / error: inspect evidence; recover if "
    "submission may have happened.\n"
    "Use recover_gpt_pro with a failed ask_id, or saved thread_ref and nonce_marker "
    "after restart. Unknown outcome is not proof that no answer exists."
)
ASK_GPT_PRO_STATUS_DESCRIPTION = (
    "Poll a background ChatGPT Pro ask and return its current state, latest "
    "status message, thread_ref, nonce_marker, and structured evidence. The "
    "thread_ref is preserved throughout the job lifecycle. queued means awaiting "
    "admission, normally behind an in-flight answer in the same conversation; "
    "asks in other conversations can continue in parallel. running means "
    "navigation, submission, or answer observation is in progress; submission "
    "must be read from evidence. detached means read-only server polling is in "
    "progress, not that generation is finished. succeeded and failed are "
    "terminal. For failed jobs, failure=expired means same-conversation queue "
    "waiting reached its TTL before this job ran; a recovery job may still "
    "refer to an earlier submitted turn. thread_ref and any "
    "available nonce marker are preserved for diagnosis. A "
    "thread_ref can appear before completion, but it becomes this MCP session's "
    "binding only after the ask succeeds. status_message values such as "
    "\"waiting for the in-flight answer\" and \"detached; polling for the "
    "answer\" remain supplemental progress details.\n"
    + _GPTPRO_FAILURE_ACTIONS_DESCRIPTION
)
ASK_GPT_PRO_RESULT_DESCRIPTION = (
    "Fetch the settled result of a background ChatGPT Pro ask only after its "
    "status is succeeded or failed; queued, running, and detached are still in "
    "progress. Successful results include nonce_marker and evidence; failed "
    "results retain isError and include both human guidance and structured "
    "failure diagnostics. failure=expired is a pre-execution queue timeout; "
    "an expired recovery job may still target an existing turn. Successful "
    "results also include files and files_complete. files lists every "
    "sandbox: file link in the final answer, each with name, sandbox_path, "
    "message_id, status (saved or failed), path, size_bytes, mime_type, "
    "sha256, and error. A saved file's contents live at path on the gateway "
    "host: read the file there. path is not a URL, and the sandbox: link in "
    "the answer is not downloadable by the caller; a client on another "
    "machine needs a shared filesystem or other out-of-band access to the "
    "gateway host. files_complete is false when any linked file was not "
    "saved; the answer text is still complete.\n"
    + _GPTPRO_FAILURE_ACTIONS_DESCRIPTION
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
RECOVER_GPT_PRO_DESCRIPTION = (
    "Read-only recovery of an existing failed ChatGPT Pro ask. Provide ask_id "
    "for a retained failed job, OR thread_ref and nonce_marker saved before a "
    "gateway restart. Never sends a prompt or attaches files. Returns a new "
    "recovery ask_id and optional source_ask_id; poll its status and result. "
    "Pending asks cannot be recovered twice, and missing identifiers cannot "
    "identify a server turn. Recovery is bounded and requires a finished "
    "nonce-correlated raw answer."
)
RECOVER_GPT_PRO_ALLOWED_ARGUMENT_KEYS = (
    "ask_id", "thread_ref", "nonce_marker",
)
RECOVER_GPT_PRO_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ask_id": {"type": "string"},
        "thread_ref": {"type": "string"},
        "nonce_marker": {"type": "string"},
    },
    "additionalProperties": False,
}
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
                types.Tool(
                    name="recover_gpt_pro",
                    description=RECOVER_GPT_PRO_DESCRIPTION,
                    inputSchema=RECOVER_GPT_PRO_INPUT_SCHEMA,
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

        if params.name == "recover_gpt_pro":
            unknown_argument_error = _get_unknown_argument_error(
                types, tool_name="recover_gpt_pro", arguments=arguments,
                allowed_argument_keys=RECOVER_GPT_PRO_ALLOWED_ARGUMENT_KEYS,
            )
            if unknown_argument_error is not None:
                return unknown_argument_error
            ask_id = arguments.get("ask_id")
            thread_ref = arguments.get("thread_ref")
            marker = arguments.get("nonce_marker")
            if ask_id is not None:
                if not isinstance(ask_id, str) or not ask_id:
                    return _tool_error(types, "ask_id must be a nonempty string")
                if "thread_ref" in arguments or "nonce_marker" in arguments:
                    return _tool_error(
                        types, "provide ask_id or thread_ref and nonce_marker, not both",
                    )
            elif not isinstance(thread_ref, str) or not isinstance(marker, str):
                return _tool_error(
                    types, "provide ask_id or both thread_ref and nonce_marker "
                    "for read-only recovery",
                )
            try:
                job = runtime.start_recovery(
                    ask_id=ask_id, conversation_id=thread_ref, marker=marker,
                )
            except ValueError as exc:
                return _tool_error(types, str(exc))
            return _json_result(types, {
                "ask_id": job.ask_id, "source_ask_id": job.source_ask_id,
                "thread_ref": job.thread_ref,
            })

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
                    "nonce_marker": job.nonce_marker,
                    "failure": job.failure,
                    "error_message": job.error_message,
                    "evidence": asdict(job.evidence),
                    "source_ask_id": job.source_ask_id,
                    "recovery_guidance": _recovery_guidance(job),
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
                return _gptpro_error_result(types, job)
            return _json_result(
                types,
                {
                    "ask_id": job.ask_id,
                    "answer": job.answer,
                    "thread_ref": job.thread_ref,
                    "nonce_marker": job.nonce_marker,
                    "evidence": asdict(job.evidence),
                    "source_ask_id": job.source_ask_id,
                    "files": [asdict(item) for item in job.files],
                    "files_complete": job.files_complete,
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


def _recovery_guidance(job: AskJob) -> str | None:
    if job.state != "failed":
        return None
    if job.evidence.failure_stage == "queue" and job.evidence.recovery == "unavailable":
        return (
            "Recovery queue admission expired before read-only polling. The "
            "original server turn may still exist; use saved thread_ref and "
            "nonce_marker to retry recovery after the in-flight ask settles."
        )
    if job.evidence.recovery_failure == "session_expired":
        return (
            "The recovery poll encountered an expired ChatGPT session. Sign in "
            "again, then recover by saved thread_ref and nonce_marker; do not "
            "resubmit the uncertain prompt."
        )
    if job.evidence.submission == "not_attempted":
        return (
            "No send click was attempted in this ask; inspect the reported "
            "failure stage before deciding whether to submit."
        )
    if job.thread_ref is not None and job.nonce_marker is not None:
        return (
            f"Call recover_gpt_pro with ask_id={job.ask_id} for read-only "
            "nonce-correlated polling; after restart use saved thread_ref "
            "and nonce_marker. Do not resubmit while the outcome is uncertain."
        )
    return (
        "Read-only recovery needs both thread_ref and nonce_marker. If either "
        "is unavailable, inspect the ChatGPT conversation before considering "
        "another submission; the missing observation does not prove no answer."
    )


def _gptpro_error_result(types: Any, job: AskJob) -> Any:
    failure = job.failure or "error"
    detail = job.error_message or "unknown error"
    guidance = _recovery_guidance(job)
    message = (
        f"ChatGPT Pro request failed [{failure}] at "
        f"{job.evidence.failure_stage or 'unknown'}: {detail}. "
        f"thread_ref={job.thread_ref}; nonce_marker={job.nonce_marker}. "
        f"{guidance}"
    )
    return types.CallToolResult(
        content=[
            types.TextContent(text=message),
            types.TextContent(text=json.dumps({
                "ask_id": job.ask_id, "state": job.state,
                "failure": job.failure, "error_message": job.error_message,
                "thread_ref": job.thread_ref, "nonce_marker": job.nonce_marker,
                "evidence": asdict(job.evidence),
                "source_ask_id": job.source_ask_id,
                "recovery_guidance": guidance,
            })),
        ],
        isError=True,
    )


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
