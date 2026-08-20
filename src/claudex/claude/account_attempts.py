"""Privacy-safe observability primitives for Claude account attempts."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_REDACTION = "[redacted]"
_IDENTIFIER_CAP = 128
_VALID_MODES = frozenset(
    {
        "disabled",
        "fallback",
        "balanced_stateless",
        "balanced_pinned",
        "balanced_count_tokens",
    }
)
_VALID_RESULTS = frozenset({"success", "rate_limited", "failed", "exception"})
_OCCURRED_AT_UTC_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"
)
_TOKEN_CHARACTERS = r"[A-Za-z0-9._~+/=-]"
# A credential token run stops only before a complete embedded Bearer start
# (boundary + "Bearer " + at least one token character). Bearer is the one
# prefix whose secret is separated by a non-token space, so absorbing its
# prefix would expose the secret; inline prefixes (sk-, ghp_, AIza, AKIA, …)
# are still redacted wholesale when absorbed into an enclosing run, and an
# incomplete Bearer (no token argument) must not suppress the enclosing match.
_TOKEN_RUN_STOP = rf"(?<![A-Za-z0-9])Bearer +(?={_TOKEN_CHARACTERS})"
_CREDENTIAL_RUN = rf"(?:(?!{_TOKEN_RUN_STOP}){_TOKEN_CHARACTERS})+"
_CREDENTIAL_PATTERNS = (
    re.compile(rf"(?<![A-Za-z0-9])Bearer +{_CREDENTIAL_RUN}", re.IGNORECASE),
    re.compile(rf"(?<![A-Za-z0-9])sk-{_CREDENTIAL_RUN}", re.IGNORECASE),
    re.compile(
        rf"(?<![A-Za-z0-9])(?:"
        rf"xox[baprs]-|gh[pousr]_|github_pat_|glpat-|npm_|pypi-|hf_|xai-|ya29\."
        rf"){_CREDENTIAL_RUN}",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![A-Za-z0-9])(?:AIza[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16})"),
)
_OPAQUE_RUN_PATTERN = re.compile(rf"{_TOKEN_CHARACTERS}{{32,}}")

# Session literals and credential matches are both masked with this
# token-compatible sentinel until the opaque-run pass has run. Substituting
# "[redacted]" directly would insert brackets — not token characters — and
# split a surrounding credential or opaque run, leaving its outer fragments
# unredacted. The sentinel keeps such runs contiguous for the intervening
# passes and is converted to "[redacted]" before truncation; a collision with
# real text only ever over-redacts. Length >= 32 so a standalone sentinel is
# also opaque-redacted in the full profile. The trailing "-" is a token
# character but not a word character, so a credential prefix immediately
# following a sentinel keeps its lookbehind start boundary.
_REDACTION_SENTINEL = "claudex0session0literal0sentinel0redacted-"


def _normalize_text_prefix(text: str) -> str:
    replaced = text.encode("utf-8", errors="replace").decode(
        "utf-8", errors="replace"
    )
    normalized = unicodedata.normalize("NFC", replaced)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )


def sanitize_external_text(
    text,
    *,
    cap: int,
    session_literals: tuple[str, ...] = (),
    redact_opaque_runs: bool = True,
) -> str:
    """Return normalized, redacted external text without propagating failures."""
    try:
        if not isinstance(cap, int) or cap <= 0:
            return ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        elif not isinstance(text, str):
            text = str(text)

        sanitized = _normalize_text_prefix(text)
        normalized_literals = {
            _normalize_text_prefix(literal)
            for literal in session_literals
            if isinstance(literal, str) and literal
        }
        for literal in sorted(normalized_literals, key=len, reverse=True):
            if literal:
                sanitized = sanitized.replace(literal, _REDACTION_SENTINEL)
        for credential_pattern in _CREDENTIAL_PATTERNS:
            sanitized = credential_pattern.sub(_REDACTION_SENTINEL, sanitized)
        if redact_opaque_runs:
            sanitized = _OPAQUE_RUN_PATTERN.sub(_REDACTION, sanitized)
        return sanitized.replace(_REDACTION_SENTINEL, _REDACTION)[:cap]
    except Exception:
        return ""


def request_shape_fields(
    parsed_body, raw_body: bytes, pin_created: bool | None
) -> dict[str, Any]:
    """Build the content-free shape fields for the client request."""
    messages = parsed_body.get("messages") if isinstance(parsed_body, dict) else None
    tools = parsed_body.get("tools") if isinstance(parsed_body, dict) else None
    return {
        "body_bytes": len(raw_body),
        "message_count": len(messages) if isinstance(messages, list) else None,
        "tool_count": len(tools) if isinstance(tools, list) else None,
        "pin_created": pin_created,
    }


@dataclass(frozen=True)
class AccountLegContext:
    """Request-scoped timing and privacy context for one account leg."""

    mode: str
    ordinal: int
    pin_created: bool | None
    first_started_monotonic: float
    started_monotonic: float
    previous_started_monotonic: float | None
    session_literals: tuple[str, ...] = field(repr=False)

    def attempt_fields(self) -> dict[str, int | None]:
        """Build one-based attempt timing fields in nonnegative milliseconds."""
        elapsed_ms_since_first = int(
            max(0.0, self.started_monotonic - self.first_started_monotonic) * 1000
        )
        gap_ms_since_previous = (
            None
            if self.previous_started_monotonic is None
            else int(
                max(0.0, self.started_monotonic - self.previous_started_monotonic)
                * 1000
            )
        )
        return {
            "ordinal": self.ordinal,
            "elapsed_ms_since_first": elapsed_ms_since_first,
            "gap_ms_since_previous": gap_ms_since_previous,
        }


class AccountLegTracker:
    """Allocate account-leg contexts from one request-scoped monotonic clock."""

    def __init__(
        self,
        mode: str,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        session_literals: tuple[str, ...] = (),
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"unsupported account-leg mode: {mode!r}")
        self._mode = mode
        self._monotonic = monotonic
        self._session_literals = tuple(session_literals)
        self._ordinal = 0
        self._first_started_monotonic: float | None = None
        self._previous_started_monotonic: float | None = None

    def begin_leg(self, pin_created: bool | None) -> AccountLegContext:
        """Allocate the next leg after exactly one monotonic clock read."""
        started_monotonic = self._monotonic()
        self._ordinal += 1
        if self._first_started_monotonic is None:
            self._first_started_monotonic = started_monotonic

        context = AccountLegContext(
            mode=self._mode,
            ordinal=self._ordinal,
            pin_created=pin_created,
            first_started_monotonic=self._first_started_monotonic,
            started_monotonic=started_monotonic,
            previous_started_monotonic=self._previous_started_monotonic,
            session_literals=self._session_literals,
        )
        self._previous_started_monotonic = started_monotonic
        return context


def try_begin_account_leg(
    tracker: AccountLegTracker | None, pin_created: bool | None
) -> AccountLegContext | None:
    """Allocate optional observability context without affecting relay behavior."""
    if tracker is None:
        return None
    try:
        return tracker.begin_leg(pin_created)
    except Exception:
        return None


def emit_account_leg_log(
    logger,
    context: AccountLegContext,
    *,
    account_id: str,
    model: str | None,
    result: str,
    parsed_body,
    raw_body: bytes,
    occurred_at_utc: str,
) -> None:
    """Emit one canonical account-leg envelope without affecting relay behavior."""
    try:
        if context.mode not in _VALID_MODES:
            raise ValueError(f"unsupported account-leg mode: {context.mode!r}")
        if result not in _VALID_RESULTS:
            raise ValueError(f"unsupported account-leg result: {result!r}")
        if _OCCURRED_AT_UTC_PATTERN.fullmatch(occurred_at_utc) is None:
            raise ValueError("occurred_at_utc must use ISO-8601 milliseconds and Z")

        envelope = {
            "v": 1,
            "event": "claude_account_leg",
            "occurred_at_utc": occurred_at_utc,
            "mode": context.mode,
            "account_id": sanitize_external_text(
                account_id,
                cap=_IDENTIFIER_CAP,
                session_literals=context.session_literals,
                redact_opaque_runs=False,
            ),
            "model": (
                None
                if model is None
                else sanitize_external_text(
                    model,
                    cap=_IDENTIFIER_CAP,
                    session_literals=context.session_literals,
                    redact_opaque_runs=False,
                )
            ),
            "result": result,
            "attempt": context.attempt_fields(),
            "request_shape": request_shape_fields(
                parsed_body, raw_body, context.pin_created
            ),
        }
        payload = json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        logger.info(payload)
    except Exception:
        try:
            logger.warning("failed to emit Claude account-leg log")
        except Exception:
            pass
