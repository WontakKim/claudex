"""Tests for Claude quota incident records and bounded JSONL persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from claudex.claude import quota_429
from claudex.claude.account_attempts import AccountLegContext
from claudex.claude.quota_429 import (
    Quota429IncidentWriter,
    build_quota_429_record,
    capture_upstream_headers,
    finalize_quota_429_record,
    parse_upstream_error_body,
)

_RAW_SESSION = "550E8400-E29B-41D4-A716-446655440000"
_CANONICAL_SESSION = "550e8400-e29b-41d4-a716-446655440000"
_SESSION_LITERALS = (_RAW_SESSION, _CANONICAL_SESSION)
_OPAQUE_ID = "req_rQ7mV2xK9pL4nD8sF1hJ6cB3wZ0tG5yU2eA7iO4kN9vP"
_OCCURRED_AT_UTC = "2026-08-21T01:02:03.456Z"


class OccurrenceHeaders:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def multi_items(self) -> list[tuple[str, str]]:
        return list(self._items)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _context() -> AccountLegContext:
    return AccountLegContext(
        mode="balanced_stateless",
        ordinal=2,
        pin_created=None,
        first_started_monotonic=10.0,
        started_monotonic=10.125,
        previous_started_monotonic=10.0,
        session_literals=_SESSION_LITERALS,
    )


def _build_record() -> dict[str, Any]:
    return build_quota_429_record(
        occurred_at_utc=_OCCURRED_AT_UTC,
        mode="balanced",
        account_id="account-1",
        model="claude-sonnet",
        cooldown_seconds=60.0,
        cooldown_source="default_60",
        context=_context(),
        parsed_body={"messages": [{}], "tools": []},
        raw_body=b"client-body",
        response_headers={"request-id": "request-1"},
        response_body=b'{"error":{"type":"rate_limit_error"}}',
    )


def test_capture_upstream_headers_allowlists_and_lowercases() -> None:
    headers = OccurrenceHeaders(
        [
            ("Retry-After", "60"),
            ("Request-ID", "request-1"),
            ("Anthropic-RateLimit-Reset", "2026-08-21T01:03:03Z"),
            ("Authorization", "Bearer forbidden-token"),
            ("Cookie", "forbidden=value"),
            ("X-API-Key", "sk-ant-forbidden"),
        ]
    )

    captured = capture_upstream_headers(headers)

    assert captured == {
        "anthropic-ratelimit-reset": "2026-08-21T01:03:03Z",
        "request-id": "request-1",
        "retry-after": "60",
    }


def test_capture_upstream_headers_enforces_per_field_and_aggregate_caps() -> None:
    headers = OccurrenceHeaders(
        [
            (f"Anthropic-RateLimit-Metric-{index:03d}-{'n' * 80}", "é" * 300)
            for index in range(100)
        ]
    )

    captured = capture_upstream_headers(headers)
    encoded = _canonical(captured).encode("utf-8")

    assert list(captured) == sorted(captured)
    assert all(len(name) <= 64 for name in captured)
    assert all(len(value) <= 256 for value in captured.values())
    assert len(encoded) <= 2048


def test_capture_upstream_headers_first_duplicate_wins() -> None:
    headers = OccurrenceHeaders(
        [("Request-ID", "first"), ("request-id", "second")]
    )

    assert capture_upstream_headers(headers)["request-id"] == "first"


def test_capture_upstream_headers_preserves_opaque_identifier_values() -> None:
    captured = capture_upstream_headers({"Request-ID": _OPAQUE_ID})

    assert captured["request-id"] == _OPAQUE_ID


def test_parse_upstream_error_body_extracts_type_message_request_id() -> None:
    credential = "Bearer abcdefghijklmnopqrstuvwxyz012345"
    opaque_run = "A" * 40
    body = _canonical(
        {
            "error": {
                "type": "rate_limit_error",
                "message": f"quota {_RAW_SESSION} {opaque_run} {credential}",
            },
            "request_id": _OPAQUE_ID,
        }
    ).encode()

    parsed = parse_upstream_error_body(body, session_literals=_SESSION_LITERALS)

    assert parsed["error_type"] == "rate_limit_error"
    assert parsed["request_id"] == _OPAQUE_ID
    assert parsed["body_bytes"] == len(body)
    assert parsed["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert "[redacted]" in parsed["message"]
    for forbidden in (_RAW_SESSION, opaque_run, credential):
        assert forbidden not in parsed["message"]


@pytest.mark.parametrize("body", [b"{", b"not-json", b"\xff"])
def test_parse_upstream_error_body_handles_malformed_and_nonjson(body: bytes) -> None:
    parsed = parse_upstream_error_body(body)

    assert parsed == {
        "body_bytes": len(body),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "error_type": None,
        "message": None,
        "request_id": None,
    }


def test_parse_upstream_error_body_hashes_empty_body() -> None:
    parsed = parse_upstream_error_body(b"")

    assert parsed["body_bytes"] == 0
    assert parsed["body_sha256"] == hashlib.sha256(b"").hexdigest()
    unavailable = parse_upstream_error_body(None)
    assert unavailable["body_bytes"] is None
    assert unavailable["body_sha256"] is None


def test_parse_upstream_error_body_preserves_opaque_request_id() -> None:
    parsed = parse_upstream_error_body(
        _canonical({"request_id": _OPAQUE_ID}).encode()
    )

    assert parsed["request_id"] == _OPAQUE_ID


def test_build_record_sanitizes_account_id_and_model() -> None:
    credential = "Bearer abcdefghijklmnopqrstuvwxyz012345"
    prompt = "PROMPT-CONTENT-MUST-NOT-APPEAR"
    record = build_quota_429_record(
        occurred_at_utc=_OCCURRED_AT_UTC,
        mode="balanced",
        account_id=f"acct\x00-{_OPAQUE_ID}-{_RAW_SESSION}",
        model=f"claude-é-​-{_CANONICAL_SESSION}-{credential}",
        cooldown_seconds=60.0,
        cooldown_source="default_60",
        context=_context(),
        parsed_body={"messages": [{"content": prompt}], "tools": []},
        raw_body=b"client-body",
        response_headers={"Authorization": credential, "Request-ID": _OPAQUE_ID},
        response_body=b"",
    )
    canonical = finalize_quota_429_record(record)

    assert record["v"] == 1
    assert record["event"] == "claude_quota_429"
    assert record["attempt"] == _context().attempt_fields()
    assert record["request_shape"] == {
        "body_bytes": len(b"client-body"),
        "message_count": 1,
        "tool_count": 0,
        "pin_created": None,
    }
    assert _OPAQUE_ID in record["account_id"]
    assert record["upstream"]["headers"] == {"request-id": _OPAQUE_ID}
    for enrichment_key in (
        "installed_scope",
        "quota_family",
        "family_gate",
        "observed_scope",
        "scope_rationale",
        "session_fingerprint",
    ):
        assert record[enrichment_key] is None
    assert "record_degraded" not in record
    assert "degradation_reason" not in record
    for forbidden in (_RAW_SESSION, _CANONICAL_SESSION, credential, prompt, "​"):
        assert forbidden not in canonical


def test_finalize_record_produces_canonical_stable_json() -> None:
    record = {"z": "é", "a": {"second": 2, "first": 1}}

    finalized = finalize_quota_429_record(record)

    assert finalized == '{"a":{"first":1,"second":2},"z":"é"}'
    assert finalized == finalize_quota_429_record(record)
    assert json.loads(finalize_quota_429_record({"unexpected": object()}))


def test_observed_scope_unknown_for_family_gate_failure() -> None:
    record = _build_record()
    reason = "fable_weekly_not_saturated"
    record["family_gate"] = {
        "scope": "family",
        "reason": reason,
        "family_deadline_utc": None,
    }
    record["observed_scope"] = "unknown"
    record["scope_rationale"] = reason

    persisted = json.loads(finalize_quota_429_record(record))

    assert persisted["observed_scope"] == "unknown"
    assert persisted["observed_scope"] != "account"
    assert persisted["scope_rationale"] == reason


def test_incident_writer_appends_one_record_per_line(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    writer = Quota429IncidentWriter(path)
    first = _canonical({"id": 1})
    second = _canonical({"id": 2})

    asyncio.run(writer.append_record(first))
    asyncio.run(writer.append_record(second))

    assert path.read_bytes() == f"{first}\n{second}\n".encode()


def test_incident_writer_repairs_truncated_tail_before_append(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    first = _canonical({"id": 1})
    second = _canonical({"id": 2})
    path.write_bytes(f'{first}\n{{"partial":'.encode())
    writer = Quota429IncidentWriter(path)

    asyncio.run(writer.append_record(second))

    assert path.read_bytes() == f"{first}\n{second}\n".encode()


def test_incident_writer_compacts_to_cap_keeping_newest_and_new_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incidents.jsonl"
    records = [
        _canonical({"id": identifier, "padding": "x" * 16})
        for identifier in (1, 2, 3)
    ]
    line_size = len(records[0].encode()) + 1
    writer = Quota429IncidentWriter(path, max_bytes=line_size * 2)

    for record in records:
        asyncio.run(writer.append_record(record))

    assert path.read_text(encoding="utf-8").splitlines() == records[-2:]
    assert path.stat().st_size <= line_size * 2


def test_incident_writer_substitutes_truncated_record_instead_of_dropping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incidents.jsonl"
    response_body = b"response-body"
    record = _build_record()
    record["session_fingerprint"] = "fingerprint-value"
    record["upstream"]["body_bytes"] = len(response_body)
    record["upstream"]["body_sha256"] = hashlib.sha256(response_body).hexdigest()
    record["oversized"] = "z" * 40000

    asyncio.run(Quota429IncidentWriter(path).append_record(_canonical(record)))

    encoded = path.read_bytes()
    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert len(encoded[:-1]) <= 32768
    truncated = json.loads(encoded)
    assert truncated["record_truncated"] is True
    for field in (
        "v",
        "event",
        "occurred_at_utc",
        "mode",
        "account_id",
        "model",
        "cooldown_seconds",
        "cooldown_source",
        "attempt",
        "session_fingerprint",
    ):
        assert truncated[field] == record[field]
    assert truncated["upstream"]["body_bytes"] == len(response_body)
    assert truncated["upstream"]["body_sha256"] == hashlib.sha256(
        response_body
    ).hexdigest()
    assert "z" * 100 not in encoded.decode()


def test_incident_writer_complete_write_under_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "incidents.jsonl"
    records = [
        _canonical({"id": identifier, "padding": "x" * 16})
        for identifier in (1, 2, 3)
    ]
    line_size = len(records[0].encode()) + 1
    writer = Quota429IncidentWriter(path, max_bytes=line_size * 2)
    real_write = quota_429.os.write
    write_calls = 0

    def short_write(file_descriptor: int, payload) -> int:
        nonlocal write_calls
        write_calls += 1
        shortened = bytes(payload[:7])
        return real_write(file_descriptor, shortened)

    monkeypatch.setattr(quota_429.os, "write", short_write)

    for record in records:
        asyncio.run(writer.append_record(record))

    assert write_calls > 3
    assert path.read_text(encoding="utf-8").splitlines() == records[-2:]


def test_incident_writer_failure_warns_and_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory_target = tmp_path / "not-a-file"
    directory_target.mkdir()
    writer = Quota429IncidentWriter(directory_target)

    asyncio.run(writer.append_record(_canonical({"id": "must-not-raise"})))

    assert "failed to append Claude 429 incident record" in caplog.text


def test_incident_writer_sets_0600_permissions(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    writer = Quota429IncidentWriter(path)

    asyncio.run(writer.append_record(_canonical({"id": 1})))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
