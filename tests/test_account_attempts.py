"""Tests for privacy-safe Claude account-attempt observability primitives."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from claudex.claude import account_attempts
from claudex.claude.account_attempts import (
    AccountLegContext,
    AccountLegTracker,
    emit_account_leg_log,
    request_shape_fields,
    sanitize_external_text,
)

_RAW_SESSION = "550E8400-E29B-41D4-A716-446655440000"
_CANONICAL_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SESSION_LITERALS = (_RAW_SESSION, _CANONICAL_SESSION)
_OCCURRED_AT_UTC = "2026-08-21T01:02:03.456Z"
_OPAQUE_ID = "rQ7mV2xK9pL4nD8sF1hJ6cB3wZ0tG5yU2eA7iO4kN9vP"


class RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def info(self, *args: Any, **kwargs: Any) -> None:
        self.info_calls.append((args, kwargs))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warning_calls.append((args, kwargs))


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self._values)


def _context(
    *,
    session_literals: tuple[str, ...] = (),
    pin_created: bool | None = True,
) -> AccountLegContext:
    return AccountLegContext(
        mode="fallback",
        ordinal=1,
        pin_created=pin_created,
        first_started_monotonic=100.0,
        started_monotonic=100.0,
        previous_started_monotonic=None,
        session_literals=session_literals,
    )


def _emit(
    logger: RecordingLogger,
    context: AccountLegContext,
    *,
    account_id: str = "account-1",
    model: str | None = "claude-sonnet",
    parsed_body: Any = None,
    raw_body: bytes = b"client-body",
) -> str:
    emit_account_leg_log(
        logger,
        context,
        account_id=account_id,
        model=model,
        result="success",
        parsed_body={} if parsed_body is None else parsed_body,
        raw_body=raw_body,
        occurred_at_utc=_OCCURRED_AT_UTC,
    )
    return logger.info_calls[0][0][0]


def test_sanitize_external_text_strips_controls_redacts_tokens_caps_length() -> None:
    bearer_token = "Bearer abcdefghijklmnopqrstuvwxyz012345"
    source = (
        "visible\x00\n​ sk-ant-secret-value "
        f"{bearer_token} {_OPAQUE_ID} trailing"
    )

    sanitized = sanitize_external_text(source, cap=512)

    assert all(
        unicodedata.category(character) not in {"Cc", "Cf"}
        for character in sanitized
    )
    assert "sk-ant-secret-value" not in sanitized
    assert bearer_token not in sanitized
    assert _OPAQUE_ID not in sanitized
    assert "[redacted]" in sanitized
    assert sanitize_external_text("é" * 10, cap=4) == "é" * 4


def test_sanitize_external_text_normalizes_invalid_unicode() -> None:
    sanitized = sanitize_external_text("\ud800é", cap=10)

    assert sanitized == "?é"
    assert unicodedata.is_normalized("NFC", sanitized)
    assert sanitize_external_text(b"\xffe\xcc\x81", cap=10) == "�é"


def test_sanitize_external_text_redacts_raw_and_canonical_session_literals() -> None:
    sanitized = sanitize_external_text(
        f"raw={_RAW_SESSION} canonical={_CANONICAL_SESSION}",
        cap=256,
        session_literals=_SESSION_LITERALS,
    )

    assert _RAW_SESSION not in sanitized
    assert _CANONICAL_SESSION not in sanitized
    assert sanitized.count("[redacted]") == 2


def test_sanitize_external_text_identifier_profile_preserves_opaque_ids() -> None:
    account_id = f"acct-{_OPAQUE_ID}"

    assert (
        sanitize_external_text(account_id, cap=128, redact_opaque_runs=False)
        == account_id
    )


def test_sanitize_external_text_identifier_profile_still_redacts_credentials_and_session_literals() -> None:
    bearer_token = "Bearer abcdefghijklmnopqrstuvwxyz012345"
    source = (
        f"acct-{_OPAQUE_ID}-{_RAW_SESSION} "
        f"sk-ant-account-secret {bearer_token}"
    )

    sanitized = sanitize_external_text(
        source,
        cap=256,
        session_literals=_SESSION_LITERALS,
        redact_opaque_runs=False,
    )

    assert _OPAQUE_ID in sanitized
    assert _RAW_SESSION not in sanitized
    assert "sk-ant-account-secret" not in sanitized
    assert bearer_token not in sanitized


def test_sanitize_external_text_literal_inside_credential_leaves_no_fragments() -> None:
    for credential in (
        f"Bearer pre{_CANONICAL_SESSION}post",
        f"sk-pre{_RAW_SESSION}post",
    ):
        sanitized = sanitize_external_text(
            credential,
            cap=256,
            session_literals=_SESSION_LITERALS,
            redact_opaque_runs=False,
        )

        assert sanitized == "[redacted]"
        assert "pre" not in sanitized
        assert "post" not in sanitized


def test_sanitize_external_text_literal_inside_opaque_run_redacts_whole_run() -> None:
    source = f"{'x' * 20}{_CANONICAL_SESSION}{'y' * 20}"

    sanitized = sanitize_external_text(
        source,
        cap=256,
        session_literals=_SESSION_LITERALS,
        redact_opaque_runs=True,
    )

    assert sanitized == "[redacted]"
    assert "x" not in sanitized
    assert "y" not in sanitized


def test_sanitize_external_text_literal_adjacent_to_credential_keeps_boundary() -> None:
    cases = (
        ("Bearer opaque-token", "opaque-token"),
        ("sk-ant-value", "sk-ant-value"),
        ("ghp_tokenvalue", "ghp_tokenvalue"),
        ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    )
    for credential, value_fragment in cases:
        for redact_opaque_runs in (True, False):
            sanitized = sanitize_external_text(
                f"{_CANONICAL_SESSION}{credential}",
                cap=256,
                session_literals=_SESSION_LITERALS,
                redact_opaque_runs=redact_opaque_runs,
            )

            assert value_fragment not in sanitized
            assert _CANONICAL_SESSION not in sanitized


def test_sanitize_external_text_credential_run_stops_before_embedded_prefix() -> None:
    source = f"Bearer pre{_CANONICAL_SESSION}Bearer secret"

    for redact_opaque_runs in (True, False):
        sanitized = sanitize_external_text(
            source,
            cap=256,
            session_literals=_SESSION_LITERALS,
            redact_opaque_runs=redact_opaque_runs,
        )

        assert "secret" not in sanitized
        assert _CANONICAL_SESSION not in sanitized


def test_request_shape_fields_counts_client_bytes_and_list_lengths() -> None:
    raw_body = "client-é".encode()

    assert request_shape_fields(
        {"messages": [{}, {}], "tools": [{}]}, raw_body, True
    ) == {
        "body_bytes": len(raw_body),
        "message_count": 2,
        "tool_count": 1,
        "pin_created": True,
    }


def test_request_shape_fields_null_counts_for_non_list_or_non_dict() -> None:
    assert request_shape_fields(
        {"messages": {}, "tools": None}, b"x", None
    ) == {
        "body_bytes": 1,
        "message_count": None,
        "tool_count": None,
        "pin_created": None,
    }
    assert request_shape_fields([], b"", False) == {
        "body_bytes": 0,
        "message_count": None,
        "tool_count": None,
        "pin_created": False,
    }


def test_tracker_allocates_one_based_ordinals_and_monotonic_gaps() -> None:
    clock = FakeClock(100.0, 100.125, 99.0)
    tracker = AccountLegTracker(
        "fallback", monotonic=clock, session_literals=_SESSION_LITERALS
    )

    assert clock.calls == 0
    first = tracker.begin_leg(True)
    second = tracker.begin_leg(None)
    third = tracker.begin_leg(False)

    assert clock.calls == 3
    assert [first.ordinal, second.ordinal, third.ordinal] == [1, 2, 3]
    assert second.attempt_fields() == {
        "ordinal": 2,
        "elapsed_ms_since_first": 125,
        "gap_ms_since_previous": 125,
    }
    assert third.attempt_fields() == {
        "ordinal": 3,
        "elapsed_ms_since_first": 0,
        "gap_ms_since_previous": 0,
    }
    assert second.previous_started_monotonic == 100.0
    assert first.session_literals == _SESSION_LITERALS


def test_tracker_first_leg_has_null_gap_and_zero_elapsed() -> None:
    clock = FakeClock(42.0)
    tracker = AccountLegTracker("balanced_pinned", monotonic=clock)

    first = tracker.begin_leg(True)

    assert first.attempt_fields() == {
        "ordinal": 1,
        "elapsed_ms_since_first": 0,
        "gap_ms_since_previous": None,
    }
    assert first.first_started_monotonic == first.started_monotonic == 42.0
    with pytest.raises(FrozenInstanceError):
        first.ordinal = 2


def test_account_leg_log_envelope_fields_and_canonical_payload() -> None:
    logger = RecordingLogger()
    raw_body = "client-é".encode()
    parsed_body = {"messages": [{}, {}], "tools": [{}]}

    payload = _emit(logger, _context(), parsed_body=parsed_body, raw_body=raw_body)
    envelope = json.loads(payload)

    assert envelope == {
        "v": 1,
        "event": "claude_account_leg",
        "occurred_at_utc": _OCCURRED_AT_UTC,
        "mode": "fallback",
        "account_id": "account-1",
        "model": "claude-sonnet",
        "result": "success",
        "attempt": {
            "ordinal": 1,
            "elapsed_ms_since_first": 0,
            "gap_ms_since_previous": None,
        },
        "request_shape": {
            "body_bytes": len(raw_body),
            "message_count": 2,
            "tool_count": 1,
            "pin_created": True,
        },
    }
    assert payload == json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert len(logger.info_calls) == 1
    assert logger.info_calls[0][1] == {}
    assert "\n" not in payload


def test_account_leg_log_sanitizes_model_and_account_id() -> None:
    logger = RecordingLogger()
    bearer_token = "Bearer abcdefghijklmnopqrstuvwxyz012345"
    account_id = f"acct-{_OPAQUE_ID}-{_RAW_SESSION}-sk-ant-account-secret"
    model = f"claude-é-​-{_CANONICAL_SESSION}-{bearer_token}"

    payload = _emit(
        logger,
        _context(session_literals=_SESSION_LITERALS),
        account_id=account_id,
        model=model,
    )
    envelope = json.loads(payload)

    assert _OPAQUE_ID in envelope["account_id"]
    assert "é" in envelope["model"]
    for forbidden in (
        _RAW_SESSION,
        _CANONICAL_SESSION,
        "sk-ant-account-secret",
        bearer_token,
        "​",
    ):
        assert forbidden not in payload


def test_account_leg_log_never_serializes_session_literals() -> None:
    context = _context(session_literals=_SESSION_LITERALS)
    logger = RecordingLogger()

    payload = _emit(
        logger,
        context,
        account_id=f"account-{_RAW_SESSION}",
        model=f"model-{_CANONICAL_SESSION}",
    )

    assert _RAW_SESSION not in payload
    assert _CANONICAL_SESSION not in payload
    assert _RAW_SESSION not in repr(context)
    assert _CANONICAL_SESSION not in repr(context)
    assert "session_literals" not in json.loads(payload)


def test_account_leg_log_contains_no_prompt_content() -> None:
    prompt = "PROMPT-CONTENT-MUST-NOT-APPEAR"
    tool_argument = "TOOL-ARGUMENT-MUST-NOT-APPEAR"
    parsed_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "tool_use", "input": {"value": tool_argument}},
                ],
            }
        ],
        "tools": [{"name": "fixture", "description": tool_argument}],
    }
    logger = RecordingLogger()

    payload = _emit(logger, _context(), parsed_body=parsed_body)

    assert prompt not in payload
    assert tool_argument not in payload
    assert json.loads(payload)["request_shape"] == {
        "body_bytes": len(b"client-body"),
        "message_count": 1,
        "tool_count": 1,
        "pin_created": True,
    }


def test_emit_account_leg_log_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialization_logger = RecordingLogger()

    def fail_serialization(*_args: Any, **_kwargs: Any) -> str:
        raise TypeError("serialization failed")

    monkeypatch.setattr(account_attempts.json, "dumps", fail_serialization)
    emit_account_leg_log(
        serialization_logger,
        _context(),
        account_id="account",
        model=None,
        result="failed",
        parsed_body={},
        raw_body=b"",
        occurred_at_utc=_OCCURRED_AT_UTC,
    )
    assert serialization_logger.warning_calls
    monkeypatch.undo()

    class InfoFailingLogger(RecordingLogger):
        def info(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("info failed")

    info_failing_logger = InfoFailingLogger()
    emit_account_leg_log(
        info_failing_logger,
        _context(),
        account_id="account",
        model=None,
        result="exception",
        parsed_body={},
        raw_body=b"",
        occurred_at_utc=_OCCURRED_AT_UTC,
    )
    assert info_failing_logger.warning_calls

    class AllLoggingFailingLogger:
        def info(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("info failed")

        def warning(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("warning failed")

    emit_account_leg_log(
        AllLoggingFailingLogger(),
        _context(),
        account_id="account",
        model=None,
        result="exception",
        parsed_body={},
        raw_body=b"",
        occurred_at_utc=_OCCURRED_AT_UTC,
    )
