"""Detect, rewrite, and estimate request context-overflow errors."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

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
