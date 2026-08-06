"""Translate Codex Responses API events into Anthropic Messages API events.

Ported from CLIProxyAPI's codex/claude response translator. The streaming
translator is a state machine that turns the flat Responses event stream into
Claude's block-oriented SSE protocol:

- one Codex ``reasoning`` item -> one Claude ``thinking`` block; its
  ``encrypted_content`` is emitted as a ``signature_delta`` so the client can
  replay it on the next turn
- ``output_text`` deltas -> a ``text`` block
- ``function_call`` items -> ``tool_use`` blocks fed by ``input_json_delta``;
  items that arrive without a name are parked as pending and hydrated when the
  name shows up (``output_item.done`` or the terminal ``response.completed``)
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from claudex_gateway.translate.claude_to_codex import build_tool_name_shortening_map, shorten_call_id

logger = logging.getLogger(__name__)

# Claude tool_use ids only allow this alphabet.
_TOOL_ID_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")
_FALLBACK_TOOL_ID_COUNTER = itertools.count(1)

# Consecutive reasoning summary parts of one Codex reasoning item are joined
# inside a single thinking block with a blank line.
_THINKING_SUMMARY_PART_SEPARATOR = "\n\n"

# Claude Code triggers its context compaction by matching the literal Anthropic
# error text "prompt is too long" (error codes are ignored), so Codex context
# overflow errors must be rewritten to carry that phrase. Detection mirrors
# CLIProxyAPI's codex_executor_terminal.go classification.
_CONTEXT_OVERFLOW_CODES = frozenset({"context_length_exceeded", "context_too_large"})
_CONTEXT_OVERFLOW_PHRASES = (
    "context window",
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
)
_CLAUDE_PROMPT_TOO_LONG = "prompt is too long"

# Claude Code client parsing contract: it extracts the actual/limit token
# counts from a matching error message and trims exactly `actual - limit`
# leading tokens before retrying compaction. Without a match it falls back to
# trimming 20% per retry with a 3-retry budget, which dead-ends long
# conversations, so any numeric pair the gateway emits must satisfy this
# pattern and be semantically valid (actual > limit >= 1). ASCII digit classes
# only: the client regex runs in JavaScript, where \d never matches the
# Unicode decimal digits Python's \d would accept.
_PROMPT_TOO_LONG_NUMBERS_RE = re.compile(
    r"prompt is too long[^0-9]*([0-9]+)\s*tokens?\s*>\s*([0-9]+)", re.IGNORECASE
)


def _has_valid_overflow_pair(match: re.Match[str]) -> bool:
    # int() rejects decimal strings beyond sys.get_int_max_str_digits(); an
    # unparseable capture from an untrusted provider message is treated as an
    # invalid pair, never an exception.
    try:
        actual = int(match.group(1))
        limit = int(match.group(2))
    except ValueError:
        return False
    return actual > limit >= 1

ClaudeEvent = tuple[str, dict[str, Any]]


def is_context_overflow_error(code: Any, message: Any) -> bool:
    text = message if isinstance(message, str) else ""
    lowered = text.lower()
    return (
        _CLAUDE_PROMPT_TOO_LONG in lowered
        or (isinstance(code, str) and code in _CONTEXT_OVERFLOW_CODES)
        or any(phrase in lowered for phrase in _CONTEXT_OVERFLOW_PHRASES)
    )


def estimate_overflow_prompt_tokens(claude_request: dict[str, Any]) -> int:
    """Estimate the actual prompt token count to report for overflow recovery.

    This is the recovery-biased reporting divisor (ceil(chars / 3.2), based on
    observed code-heavy payloads and deliberately safer than the display-path
    /4 estimate, though not a mathematical worst-case bound). It exists solely
    to synthesize a numeric ``prompt is too long`` pair for overflow recovery
    and is unrelated to the count_tokens display estimate.
    """
    chars = len(json.dumps(claude_request, ensure_ascii=False))
    return (chars * 5 + 15) // 16


def _neutralize_client_pairs(text: str) -> str:
    """Break every client-regex match by replacing its `>` separator with `/`.

    Claude Code parses the first matching pair, so any pair the gateway did
    not vouch for must be made unparseable rather than left in place.
    """
    neutralized = text
    while True:
        match = _PROMPT_TOO_LONG_NUMBERS_RE.search(neutralized)
        if match is None:
            return neutralized
        start, end = match.span()
        neutralized = neutralized[:start] + neutralized[start:end].replace(">", "/", 1) + neutralized[end:]


def rewrite_context_overflow_message(
    code: Any,
    message: Any,
    *,
    estimated_tokens: int | None = None,
    context_window: int | None = None,
) -> str | None:
    """Return a Claude-compatible message if the error is a context overflow, else None."""
    text = message if isinstance(message, str) else ""
    if not is_context_overflow_error(code, text):
        return None

    match = _PROMPT_TOO_LONG_NUMBERS_RE.search(text)
    if match is not None and _has_valid_overflow_pair(match):
        return text

    if estimated_tokens is not None and context_window is not None and context_window >= 1:
        floor = (context_window * 110 + 99) // 100
        reported = max(estimated_tokens, floor)
        rewritten = f"{_CLAUDE_PROMPT_TOO_LONG}: {reported} tokens > {context_window}"
        # Neutralize any pair inside the appended original text so the
        # synthesized message carries exactly one parseable pair — the one
        # the gateway vouches for.
        return f"{rewritten} ({_neutralize_client_pairs(text)})" if text else rewritten

    if _CLAUDE_PROMPT_TOO_LONG in text.lower():
        candidate = text
    else:
        candidate = f"{_CLAUDE_PROMPT_TOO_LONG}: {text}" if text else _CLAUDE_PROMPT_TOO_LONG

    candidate_match = _PROMPT_TOO_LONG_NUMBERS_RE.search(candidate)
    if candidate_match is None or _has_valid_overflow_pair(candidate_match):
        return candidate

    # The legacy prefix minted an unverified/invalid numeric pair (e.g. the
    # original message already had one, or contained a phrase-less "X tokens
    # > Y" sequence). Neutralize so Claude Code's client regex can't misparse
    # it and instead falls back to its conservative 20%-per-retry trimming.
    neutralized = _neutralize_client_pairs(candidate)
    logger.warning("Neutralized invalid overflow token pair in message: %s", text[:200])
    return neutralized


def sanitize_claude_tool_id(tool_id: str) -> str:
    sanitized = _TOOL_ID_SANITIZER.sub("_", tool_id)
    if not sanitized:
        sanitized = f"toolu_gateway_{next(_FALLBACK_TOOL_ID_COUNTER)}"
    return shorten_call_id(sanitized)


def map_codex_stop_reason_to_claude(stop_reason: str, has_tool_call: bool) -> str:
    if has_tool_call:
        return "tool_use"
    if stop_reason in ("", "stop", "completed", "tool_use", "tool_calls", "function_call"):
        return "end_turn"
    if stop_reason in ("max_tokens", "max_output_tokens"):
        return "max_tokens"
    if stop_reason in ("end_turn", "stop_sequence", "pause_turn", "refusal", "model_context_window_exceeded"):
        return stop_reason
    if stop_reason == "content_filter":
        return "refusal"
    return "end_turn"


def extract_codex_stop_reason(response_data: dict[str, Any]) -> str:
    stop_reason = response_data.get("stop_reason")
    stop_sequence = response_data.get("stop_sequence")
    if stop_reason:
        if stop_reason == "stop" and stop_sequence:
            return "stop_sequence"
        return stop_reason
    incomplete = response_data.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason"):
        return incomplete["reason"]
    if stop_sequence:
        return "stop_sequence"
    return ""


def extract_responses_usage(usage: Any) -> tuple[int, int, int]:
    """Return (input_tokens excluding cached, output_tokens, cached_tokens)."""
    if not isinstance(usage, dict):
        return 0, 0, 0
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    details = usage.get("input_tokens_details")
    cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    if cached_tokens > 0:
        input_tokens = max(input_tokens - cached_tokens, 0)
    return input_tokens, output_tokens, cached_tokens


def _build_reverse_tool_name_map(claude_request: dict[str, Any]) -> dict[str, str]:
    return {short: original for original, short in build_tool_name_shortening_map(claude_request).items()}


@dataclass
class _PendingFunctionCall:
    call_id: str = ""
    arguments: str = ""
    has_received_arguments_delta: bool = False
    start_emitted: bool = False


@dataclass
class CodexToClaudeStreamTranslator:
    """Stateful translator; feed it Codex SSE data payloads in stream order."""

    claude_request: dict[str, Any]
    context_window: int | None = None

    _short_to_original: dict[str, str] = field(init=False)
    _block_index: int = 0
    _text_block_open: bool = False
    _thinking_block_open: bool = False
    _thinking_signature: str = ""
    _thinking_summary_seen: bool = False
    _function_call_block_open: bool = False
    _function_call_block_call_id: str = ""
    _function_call_block_index: int = 0
    _has_received_arguments_delta: bool = False
    _has_text_delta: bool = False
    _has_emitted_tool_use: bool = False
    _pending_calls: dict[str, _PendingFunctionCall] = field(default_factory=dict)
    _last_pending_key: str = ""
    _web_search_tool_use_ids: set[str] = field(default_factory=set)
    _web_search_tool_result_ids: set[str] = field(default_factory=set)
    _last_web_search_tool_use_id: str = ""

    def __post_init__(self) -> None:
        self._short_to_original = _build_reverse_tool_name_map(self.claude_request)

    def translate_event(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        event_type = event.get("type", "")
        handler = getattr(self, "_on_" + event_type.replace("response.", "").replace(".", "_"), None)
        if handler is None:
            return []
        return handler(event)

    # --- lifecycle events ---------------------------------------------------

    def _on_created(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        response = event.get("response") or {}
        # Answer as the model the client asked for: Claude Code keys behavior
        # on the model name, and the Codex target is a routing detail.
        requested_model = self.claude_request.get("model")
        if not (isinstance(requested_model, str) and requested_model):
            requested_model = response.get("model", "")
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": response.get("id", ""),
                        "type": "message",
                        "role": "assistant",
                        "model": requested_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _on_error(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        error = event.get("error") or {}
        error_type = error.get("type") or event.get("error_type") or "api_error"
        message = error.get("message") or event.get("message") or error.get("code") or error_type
        if error.get("code") == "cyber_policy" or error_type == "invalid_request":
            error_type = "invalid_request_error"
        rewritten = self._rewrite_overflow_message(error.get("code"), message)
        if rewritten is not None:
            error_type = "invalid_request_error"
            message = rewritten
        return [("error", {"type": "error", "error": {"type": error_type, "message": message}})]

    def _on_failed(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        response = event.get("response") or {}
        error = response.get("error") or {}
        message = error.get("message") or "Codex response failed"
        error_type = "api_error"
        rewritten = self._rewrite_overflow_message(error.get("code"), message)
        if rewritten is not None:
            error_type = "invalid_request_error"
            message = rewritten
        return [("error", {"type": "error", "error": {"type": error_type, "message": message}})]

    def _rewrite_overflow_message(self, code: Any, message: Any) -> str | None:
        """Rewrite an overflow error message, enriching it with a numeric pair.

        Estimation only runs for context-overflow errors, and only when this
        translator carries a `context_window`; non-overflow errors and
        translators without a configured window are untouched.
        """
        text = message if isinstance(message, str) else ""
        if not is_context_overflow_error(code, text):
            return None
        if self.context_window is None:
            return rewrite_context_overflow_message(code, text)
        estimated_tokens = estimate_overflow_prompt_tokens(self.claude_request)
        return rewrite_context_overflow_message(
            code, text, estimated_tokens=estimated_tokens, context_window=self.context_window
        )

    def _on_completed(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        response = event.get("response") or {}
        events: list[ClaudeEvent] = []
        events.extend(self._hydrate_open_function_call_from_terminal(response))
        events.extend(self._finalize_open_blocks())
        events.extend(self._flush_pending_calls_from_terminal(response))

        input_tokens, output_tokens, cached_tokens = extract_responses_usage(response.get("usage"))
        usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        if cached_tokens > 0:
            usage["cache_read_input_tokens"] = cached_tokens

        stop_reason = map_codex_stop_reason_to_claude(
            extract_codex_stop_reason(response), self._has_emitted_tool_use
        )
        delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
        if response.get("stop_sequence"):
            delta["stop_sequence"] = response["stop_sequence"]

        events.append(("message_delta", {"type": "message_delta", "delta": delta, "usage": usage}))
        events.append(("message_stop", {"type": "message_stop"}))
        return events

    _on_incomplete = _on_completed

    # --- reasoning (thinking) events ----------------------------------------

    def _on_reasoning_summary_part_added(self, _event: dict[str, Any]) -> list[ClaudeEvent]:
        if self._thinking_block_open:
            self._thinking_summary_seen = True
            return self._thinking_delta(_THINKING_SUMMARY_PART_SEPARATOR)
        self._thinking_summary_seen = True
        return self._start_thinking_block()

    def _on_reasoning_summary_text_delta(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        events = self._start_thinking_block()
        events.extend(self._thinking_delta(event.get("delta", "")))
        return events

    def _on_reasoning_summary_part_done(self, _event: dict[str, Any]) -> list[ClaudeEvent]:
        # The thinking block stays open: only output_item.done carries the
        # reasoning item's final encrypted_content.
        return []

    # --- text events ---------------------------------------------------------

    def _on_content_part_added(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        events = self._finalize_thinking_block()
        part = event.get("part") or {}
        if part.get("type") == "output_text":
            events.extend(self._start_text_block())
        return events

    def _on_output_text_delta(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        self._has_text_delta = True
        events = self._finalize_thinking_block()
        events.extend(self._start_text_block())
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "text_delta", "text": event.get("delta", "")},
                },
            )
        )
        return events

    def _on_content_part_done(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        part = event.get("part") or {}
        if part.get("type") == "output_text":
            return self._stop_text_block()
        return []

    # --- output item events ---------------------------------------------------

    def _on_output_item_added(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        item = event.get("item") or {}
        item_type = item.get("type")

        if item_type == "function_call":
            events = self._finalize_thinking_block()
            events.extend(self._stop_text_block())
            self._has_received_arguments_delta = False

            call_id = item.get("call_id", "")
            name = item.get("name", "")
            if not name:
                self._record_pending_call(event, item)
                return events

            pending, alias_keys = self._pending_call_for_done(event, item)
            if pending is not None:
                self._delete_pending_aliases(alias_keys)

            block_index = self._block_index
            events.extend(self._function_call_start(call_id, name, block_index))
            self._has_emitted_tool_use = True
            events.extend(self._function_call_arguments_delta("", block_index))
            self._function_call_block_open = True
            self._function_call_block_call_id = call_id
            self._function_call_block_index = block_index
            return events

        if item_type == "reasoning":
            # A previous reasoning item that never saw output_item.done must not
            # leak its still-open block into this one.
            events = self._finalize_thinking_block()
            self._thinking_summary_seen = False
            # Pre-content snapshot; output_item.done delivers the final value.
            self._thinking_signature = item.get("encrypted_content") or ""
            return events

        return []

    def _on_output_item_done(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        item = event.get("item") or {}
        item_type = item.get("type")

        if item_type == "message":
            return self._emit_message_item_fallback(item)

        if item_type == "function_call":
            pending, alias_keys = self._pending_call_for_done(event, item)
            if pending is not None and not pending.start_emitted:
                name = item.get("name", "")
                if not name:
                    return []
                call_id = pending.call_id or item.get("call_id", "")
                block_index = self._block_index
                events = self._function_call_start(call_id, name, block_index)
                self._has_emitted_tool_use = True
                pending.start_emitted = True

                arguments = pending.arguments or item.get("arguments") or ""
                if arguments:
                    events.extend(self._function_call_arguments_delta(arguments, block_index))
                events.extend(self._function_call_stop(block_index))
                self._block_index += 1
                self._delete_pending_aliases(alias_keys)
                return events

            if self._function_call_block_open:
                events: list[ClaudeEvent] = []
                if not self._has_received_arguments_delta and item.get("arguments"):
                    events.extend(
                        self._function_call_arguments_delta(
                            item["arguments"], self._function_call_block_index
                        )
                    )
                    self._has_received_arguments_delta = True
                events.extend(self._stop_open_function_call_block())
                return events
            return []

        if item_type == "reasoning":
            if item.get("encrypted_content"):
                self._thinking_signature = item["encrypted_content"]
            if self._thinking_summary_seen:
                events = self._finalize_thinking_block()
            else:
                events = self._finalize_signature_only_thinking_block()
            self._thinking_signature = ""
            self._thinking_summary_seen = False
            return events

        if item_type == "web_search_call":
            return self._append_web_search_tool_result(event, item)

        return []

    def _emit_message_item_fallback(self, item: dict[str, Any]) -> list[ClaudeEvent]:
        """Emit the full message text when the stream never sent text deltas."""
        if self._has_text_delta:
            return []
        content = item.get("content")
        if not isinstance(content, list):
            return []
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "output_text"
        )
        if not text:
            return []

        events = self._finalize_thinking_block()
        events.extend(self._start_text_block())
        events.append(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
        )
        events.extend(self._stop_text_block())
        self._has_text_delta = True
        return events

    # --- web search (server tool) events --------------------------------------

    def _web_search_tool_use_id(self, event: dict[str, Any], item: dict[str, Any]) -> str:
        for key in ("id", "output_item_id", "call_id"):
            for source in (item, event):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if self._last_web_search_tool_use_id:
            return self._last_web_search_tool_use_id
        for source in (item, event):
            value = source.get("item_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
        generated = f"web_search_{self._block_index}"
        self._last_web_search_tool_use_id = generated
        return generated

    @staticmethod
    def _web_search_query(event: dict[str, Any], item: dict[str, Any]) -> str:
        for path in (("action", "query"), ("query",), ("input", "query")):
            for source in (item, event):
                value: Any = source
                for key in path:
                    value = value.get(key) if isinstance(value, dict) else None
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @staticmethod
    def _web_search_result_content(
        event: dict[str, Any], item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        results = item.get("results")
        if not isinstance(results, list):
            results = event.get("results")
        if not isinstance(results, list):
            return []
        content: list[dict[str, Any]] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = (result.get("url") or "").strip()
            if not url:
                continue
            title = (result.get("title") or "").strip() or url
            content.append(
                {"type": "web_search_result", "title": title, "url": url, "page_age": None}
            )
        return content

    def _append_web_search_server_tool_use(
        self, event: dict[str, Any], item: dict[str, Any]
    ) -> list[ClaudeEvent]:
        tool_use_id = self._web_search_tool_use_id(event, item)
        if not tool_use_id or tool_use_id in self._web_search_tool_use_ids:
            return []

        events = self._finalize_thinking_block()
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": tool_use_id,
                        "name": "web_search",
                        "input": {},
                    },
                },
            )
        )
        query = self._web_search_query(event, item)
        if query:
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps({"query": query}, ensure_ascii=False),
                        },
                    },
                )
            )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": self._block_index})
        )
        self._web_search_tool_use_ids.add(tool_use_id)
        self._block_index += 1
        return events

    def _append_web_search_tool_result(
        self, event: dict[str, Any], item: dict[str, Any]
    ) -> list[ClaudeEvent]:
        tool_use_id = self._web_search_tool_use_id(event, item)
        if not tool_use_id:
            return []
        events = self._append_web_search_server_tool_use(event, item)
        if tool_use_id in self._web_search_tool_result_ids:
            return events

        query = self._web_search_query(event, item)
        content = self._web_search_result_content(event, item)
        if not query and not content and "action" not in item:
            return events

        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    },
                },
            )
        )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": self._block_index})
        )
        self._web_search_tool_result_ids.add(tool_use_id)
        self._block_index += 1
        if tool_use_id == self._last_web_search_tool_use_id:
            self._last_web_search_tool_use_id = ""
        return events

    # --- function call argument events ---------------------------------------

    def _on_function_call_arguments_delta(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        delta = event.get("delta", "")
        pending = self._pending_call_for_arguments(event)
        if pending is not None and not pending.start_emitted:
            pending.has_received_arguments_delta = True
            pending.arguments += delta
            return []

        self._has_received_arguments_delta = True
        return self._function_call_arguments_delta(delta, self._block_index)

    def _on_function_call_arguments_done(self, event: dict[str, Any]) -> list[ClaudeEvent]:
        pending = self._pending_call_for_arguments(event)
        if pending is not None and not pending.start_emitted:
            if not pending.has_received_arguments_delta:
                pending.arguments = event.get("arguments", "")
            return []

        if not self._has_received_arguments_delta and event.get("arguments"):
            self._has_received_arguments_delta = True
            return self._function_call_arguments_delta(event["arguments"], self._block_index)
        return []

    # --- pending function call bookkeeping ------------------------------------

    @staticmethod
    def _pending_key(event: dict[str, Any], item: dict[str, Any]) -> str:
        if "output_index" in event:
            return f"output:{event['output_index']}"
        if item.get("call_id"):
            return f"call:{item['call_id']}"
        return "last"

    def _record_pending_call(self, event: dict[str, Any], item: dict[str, Any]) -> None:
        pending = _PendingFunctionCall(call_id=item.get("call_id", ""))
        key = self._pending_key(event, item)
        self._pending_calls[key] = pending
        if pending.call_id:
            self._pending_calls[f"call:{pending.call_id}"] = pending
        self._last_pending_key = key

    def _pending_call_for_arguments(self, event: dict[str, Any]) -> _PendingFunctionCall | None:
        if "output_index" in event:
            key = f"output:{event['output_index']}"
        else:
            key = self._last_pending_key
        return self._pending_calls.get(key) if key else None

    def _pending_call_for_done(
        self, event: dict[str, Any], item: dict[str, Any]
    ) -> tuple[_PendingFunctionCall | None, list[str]]:
        keys = [self._pending_key(event, item)]
        call_id = item.get("call_id", "")
        if call_id:
            key = f"call:{call_id}"
            if key not in keys:
                keys.append(key)
        elif "output_index" not in event and self._last_pending_key:
            if self._last_pending_key not in keys:
                keys.append(self._last_pending_key)

        for key in keys:
            pending = self._pending_calls.get(key)
            if pending is not None:
                return pending, self._aliases_of(pending)
        return None, []

    def _aliases_of(self, pending: _PendingFunctionCall) -> list[str]:
        return [key for key, candidate in self._pending_calls.items() if candidate is pending]

    def _delete_pending_aliases(self, keys: list[str]) -> None:
        for key in keys:
            self._pending_calls.pop(key, None)
            if self._last_pending_key == key:
                self._last_pending_key = ""

    def _hydrate_open_function_call_from_terminal(
        self, response_data: dict[str, Any]
    ) -> list[ClaudeEvent]:
        """Backfill arguments for an open tool_use block from the terminal response."""
        if not self._function_call_block_open or self._has_received_arguments_delta:
            return []
        for item in response_data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue
            if item.get("call_id") != self._function_call_block_call_id:
                continue
            if item.get("arguments"):
                self._has_received_arguments_delta = True
                return self._function_call_arguments_delta(
                    item["arguments"], self._function_call_block_index
                )
            break
        return []

    def _flush_pending_calls_from_terminal(self, response_data: dict[str, Any]) -> list[ClaudeEvent]:
        """Emit complete tool_use blocks for calls that never got a named stream item."""
        if not self._pending_calls:
            return []

        events: list[ClaudeEvent] = []
        for output_index, item in enumerate(response_data.get("output") or []):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue

            pending, alias_keys = self._pending_call_for_terminal_item(output_index, item)
            if pending is None:
                continue
            if pending.start_emitted:
                self._delete_pending_aliases(alias_keys)
                continue

            name = item.get("name", "")
            if not name:
                self._delete_pending_aliases(alias_keys)
                continue
            call_id = pending.call_id or item.get("call_id", "")

            block_index = self._block_index
            events.extend(self._function_call_start(call_id, name, block_index))
            self._has_emitted_tool_use = True
            pending.start_emitted = True

            arguments = item.get("arguments") or pending.arguments
            if arguments:
                events.extend(self._function_call_arguments_delta(arguments, block_index))
            events.extend(self._function_call_stop(block_index))
            self._block_index += 1
            self._delete_pending_aliases(alias_keys)

        self._pending_calls.clear()
        self._last_pending_key = ""
        return events

    def _pending_call_for_terminal_item(
        self, output_index: int, item: dict[str, Any]
    ) -> tuple[_PendingFunctionCall | None, list[str]]:
        keys: list[str] = []
        if item.get("call_id"):
            keys.append(f"call:{item['call_id']}")
        if "output_index" in item:
            keys.append(f"output:{item['output_index']}")
        keys.append(f"output:{output_index}")

        for key in keys:
            pending = self._pending_calls.get(key)
            if pending is not None:
                return pending, self._aliases_of(pending)
        return None, []

    # --- block emit helpers ----------------------------------------------------

    def _start_text_block(self) -> list[ClaudeEvent]:
        if self._text_block_open:
            return []
        self._text_block_open = True
        return [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        ]

    def _stop_text_block(self) -> list[ClaudeEvent]:
        if not self._text_block_open:
            return []
        self._text_block_open = False
        events: list[ClaudeEvent] = [
            ("content_block_stop", {"type": "content_block_stop", "index": self._block_index})
        ]
        self._block_index += 1
        return events

    def _start_thinking_block(self) -> list[ClaudeEvent]:
        if self._thinking_block_open:
            return []
        self._thinking_block_open = True
        return [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        ]

    def _thinking_delta(self, text: str) -> list[ClaudeEvent]:
        if not text:
            return []
        return [
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                },
            )
        ]

    def _finalize_thinking_block(self) -> list[ClaudeEvent]:
        if not self._thinking_block_open:
            return []
        events: list[ClaudeEvent] = []
        if self._thinking_signature:
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": self._thinking_signature,
                        },
                    },
                )
            )
        events.append(
            ("content_block_stop", {"type": "content_block_stop", "index": self._block_index})
        )
        self._block_index += 1
        self._thinking_block_open = False
        return events

    def _finalize_signature_only_thinking_block(self) -> list[ClaudeEvent]:
        """Emit an empty thinking block carrying only the signature for replay."""
        if not self._thinking_signature:
            return []
        events = self._start_thinking_block()
        events.extend(self._finalize_thinking_block())
        return events

    def _function_call_start(self, call_id: str, name: str, block_index: int) -> list[ClaudeEvent]:
        return [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": sanitize_claude_tool_id(call_id),
                        "name": self._short_to_original.get(name, name),
                        "input": {},
                    },
                },
            )
        ]

    def _function_call_arguments_delta(self, partial_json: str, block_index: int) -> list[ClaudeEvent]:
        return [
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": partial_json},
                },
            )
        ]

    def _function_call_stop(self, block_index: int) -> list[ClaudeEvent]:
        return [("content_block_stop", {"type": "content_block_stop", "index": block_index})]

    def _stop_open_function_call_block(self) -> list[ClaudeEvent]:
        if not self._function_call_block_open:
            return []
        block_index = self._function_call_block_index
        events = self._function_call_stop(block_index)
        if self._block_index <= block_index:
            self._block_index = block_index + 1
        self._function_call_block_open = False
        self._function_call_block_call_id = ""
        self._function_call_block_index = 0
        return events

    def _finalize_open_blocks(self) -> list[ClaudeEvent]:
        events = self._finalize_thinking_block()
        events.extend(self._stop_text_block())
        events.extend(self._stop_open_function_call_block())
        return events


def assemble_claude_message(claude_events: list[ClaudeEvent]) -> dict[str, Any] | None:
    """Assemble a non-streaming Claude message from translated stream events.

    The Codex backend returns an empty ``response.output`` when ``store`` is
    false, so a non-streaming Claude response must be aggregated from the same
    event stream the streaming path emits.
    """
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}

    for event_name, payload in claude_events:
        if event_name == "message_start":
            started = payload["message"]
            message = {
                "id": started.get("id", ""),
                "type": "message",
                "role": "assistant",
                "model": started.get("model", ""),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": dict(started.get("usage") or {"input_tokens": 0, "output_tokens": 0}),
            }
        elif event_name == "content_block_start":
            index = payload["index"]
            blocks[index] = dict(payload["content_block"])
            partial_json[index] = ""
        elif event_name == "content_block_delta":
            block = blocks.get(payload["index"])
            if block is None:
                continue
            delta = payload["delta"]
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta_type == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta_type == "input_json_delta":
                partial_json[payload["index"]] += delta["partial_json"]
        elif event_name == "message_delta" and message is not None:
            message["stop_reason"] = payload["delta"].get("stop_reason")
            message["stop_sequence"] = payload["delta"].get("stop_sequence")
            message["usage"] = payload.get("usage") or message["usage"]

    if message is None:
        return None

    for index in sorted(blocks):
        block = blocks[index]
        if block.get("type") in ("tool_use", "server_tool_use"):
            raw_arguments = partial_json.get(index, "")
            try:
                tool_input = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError:
                tool_input = {}
            block["input"] = tool_input if isinstance(tool_input, dict) else {}
        message["content"].append(block)
    return message
