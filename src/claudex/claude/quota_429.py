"""Privacy-safe Claude 429 incident records and bounded JSONL persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .account_attempts import (
    AccountLegContext,
    request_shape_fields,
    sanitize_external_text,
)

logger = logging.getLogger(__name__)

_HEADER_BYTES_CAP = 2048
_RECORD_BYTES_CAP = 32768
_DEFAULT_RETENTION_BYTES = 1048576
_IDENTIFIER_CAP = 128
_HEADER_NAME_CAP = 64
_HEADER_VALUE_CAP = 256
_ERROR_TYPE_CAP = 64
_MESSAGE_CAP = 256
_READ_CHUNK_BYTES = 65536
_INCIDENT_WRITE_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class Quota429Mark:
    """One marked quota response and its privacy-sensitive session literals."""

    cooldown_seconds: float
    cooldown_source: str
    record: dict[str, Any]
    session_literals: tuple[str, str] | None = field(repr=False)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _record_session_literals(
    context: AccountLegContext, session_literals: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            literal
            for literal in (*context.session_literals, *session_literals)
            if isinstance(literal, str) and literal
        )
    )


def _header_items(headers: Any) -> Iterable[Any]:
    multi_items = getattr(headers, "multi_items", None)
    if callable(multi_items):
        return multi_items()

    raw_items = getattr(headers, "raw", None)
    if raw_items is not None:
        return raw_items() if callable(raw_items) else raw_items

    items = getattr(headers, "items", None)
    if callable(items):
        return items()
    return headers


def _is_allowlisted_header(name: str) -> bool:
    return name in {"retry-after", "request-id"} or name.startswith(
        "anthropic-ratelimit-"
    )


def capture_upstream_headers(
    headers, *, session_literals: tuple[str, ...] = ()
) -> dict[str, str]:
    """Capture bounded, allowlisted upstream headers in deterministic order."""
    first_values: dict[str, str] = {}
    try:
        for item in _header_items(headers):
            try:
                name, value = item
            except (TypeError, ValueError):
                continue

            sanitized_name = sanitize_external_text(
                name,
                cap=_HEADER_NAME_CAP,
                session_literals=session_literals,
                redact_opaque_runs=False,
            ).lower()
            if (
                not _is_allowlisted_header(sanitized_name)
                or sanitized_name in first_values
            ):
                continue

            first_values[sanitized_name] = sanitize_external_text(
                value,
                cap=_HEADER_VALUE_CAP,
                session_literals=session_literals,
                redact_opaque_runs=False,
            )
    except Exception:
        pass

    captured: dict[str, str] = {}
    for name in sorted(first_values):
        candidate = {**captured, name: first_values[name]}
        if len(_canonical_json(candidate).encode("utf-8")) <= _HEADER_BYTES_CAP:
            captured[name] = first_values[name]
    return captured


def _sanitize_optional_text(
    value: Any,
    *,
    cap: int,
    session_literals: tuple[str, ...],
    redact_opaque_runs: bool,
) -> str | None:
    if value is None:
        return None
    return sanitize_external_text(
        value,
        cap=cap,
        session_literals=session_literals,
        redact_opaque_runs=redact_opaque_runs,
    )


def parse_upstream_error_body(
    body: bytes | None, *, session_literals: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Extract bounded 429 error metadata without propagating parse failures."""
    result: dict[str, Any] = {
        "body_bytes": None,
        "body_sha256": None,
        "error_type": None,
        "message": None,
        "request_id": None,
    }
    if body is None:
        return result

    try:
        result["body_bytes"] = len(body)
        result["body_sha256"] = hashlib.sha256(body).hexdigest()
    except Exception:
        return result

    try:
        parsed = json.loads(body)
    except Exception:
        return result
    if not isinstance(parsed, dict):
        return result

    error = parsed.get("error")
    if isinstance(error, dict):
        result["error_type"] = _sanitize_optional_text(
            error.get("type"),
            cap=_ERROR_TYPE_CAP,
            session_literals=session_literals,
            redact_opaque_runs=False,
        )
        result["message"] = _sanitize_optional_text(
            error.get("message"),
            cap=_MESSAGE_CAP,
            session_literals=session_literals,
            redact_opaque_runs=True,
        )
    result["request_id"] = _sanitize_optional_text(
        parsed.get("request_id"),
        cap=_IDENTIFIER_CAP,
        session_literals=session_literals,
        redact_opaque_runs=False,
    )
    return result


def build_quota_429_record(
    *,
    occurred_at_utc: str,
    mode: str,
    account_id: str,
    model: str | None,
    cooldown_seconds: float,
    cooldown_source: str,
    context: AccountLegContext,
    parsed_body: Any,
    raw_body: bytes,
    response_headers,
    response_body: bytes | None,
    session_literals: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the base v1 Claude quota incident record for later enrichment."""
    record_literals = _record_session_literals(context, session_literals)
    upstream = parse_upstream_error_body(
        response_body, session_literals=record_literals
    )
    upstream["headers"] = capture_upstream_headers(
        response_headers, session_literals=record_literals
    )

    return {
        "v": 1,
        "event": "claude_quota_429",
        "occurred_at_utc": occurred_at_utc,
        "mode": mode,
        "account_id": sanitize_external_text(
            account_id,
            cap=_IDENTIFIER_CAP,
            session_literals=record_literals,
            redact_opaque_runs=False,
        ),
        "model": (
            None
            if model is None
            else sanitize_external_text(
                model,
                cap=_IDENTIFIER_CAP,
                session_literals=record_literals,
                redact_opaque_runs=False,
            )
        ),
        "cooldown_seconds": cooldown_seconds,
        "cooldown_source": cooldown_source,
        "installed_scope": None,
        "quota_family": None,
        "family_gate": None,
        "observed_scope": None,
        "scope_rationale": None,
        "upstream": upstream,
        "attempt": context.attempt_fields(),
        "session_fingerprint": None,
        "request_shape": request_shape_fields(
            parsed_body, raw_body, context.pin_created
        ),
    }


def enrich_record_with_family_gate(
    record: dict[str, Any],
    *,
    scope: str,
    reason: str,
    family_deadline_utc: str | None,
    quota_family: str,
    session_fingerprint: str | None,
) -> dict[str, Any]:
    """Add a balanced family-gate outcome to a base incident record."""
    record["installed_scope"] = scope
    record["quota_family"] = quota_family
    record["family_gate"] = {
        "scope": scope,
        "reason": reason,
        "family_deadline_utc": family_deadline_utc,
    }
    record["observed_scope"] = "family" if scope == "family" else "unknown"
    record["scope_rationale"] = reason
    record["session_fingerprint"] = session_fingerprint
    return record


def enrich_record_fallback(
    record: dict[str, Any], *, session_fingerprint: str | None
) -> dict[str, Any]:
    """Add fallback-mode account cooldown facts to a base incident record."""
    record["installed_scope"] = "account"
    record["quota_family"] = None
    record["family_gate"] = None
    record["observed_scope"] = "unknown"
    record["scope_rationale"] = "fallback_no_family_gate"
    record["session_fingerprint"] = session_fingerprint
    return record


def enrich_record_degraded(
    record: dict[str, Any],
    *,
    installed_scope: str,
    quota_family: str,
    family_gate: dict[str, Any] | None,
    degradation_reason: str = "evidence_enrichment_failed",
) -> dict[str, Any]:
    """Preserve known cooldown facts when balanced evidence enrichment fails."""
    record["installed_scope"] = installed_scope
    record["quota_family"] = quota_family
    record["family_gate"] = family_gate
    if family_gate is None:
        record["observed_scope"] = "unknown"
        record["scope_rationale"] = "evidence_classification_unavailable"
    else:
        record["observed_scope"] = (
            "family" if family_gate.get("scope") == "family" else "unknown"
        )
        record["scope_rationale"] = family_gate.get("reason")
    record["session_fingerprint"] = None
    record["record_degraded"] = True
    record["degradation_reason"] = degradation_reason
    return record


def finalize_quota_429_record(record: Mapping[str, Any]) -> str:
    """Return canonical JSON, degrading safely for unexpected values."""
    try:
        return _canonical_json(dict(record))
    except Exception:
        return (
            '{"degradation_reason":"record_serialization_failed",'
            '"record_degraded":true}'
        )


def _bounded_scalar(value: Any, *, string_cap: int = 256) -> Any:
    if isinstance(value, str):
        return value[:string_cap]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def _bounded_fields(
    value: Any, field_caps: Mapping[str, int]
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        field: _bounded_scalar(value.get(field), string_cap=cap)
        for field, cap in field_caps.items()
    }


def _truncated_record(canonical_record: str) -> str:
    try:
        source = json.loads(canonical_record)
    except (json.JSONDecodeError, TypeError, ValueError):
        source = {}
    if not isinstance(source, dict):
        source = {}

    truncated: dict[str, Any] = {
        "v": _bounded_scalar(source.get("v", 1)),
        "event": _bounded_scalar(
            source.get("event", "claude_quota_429"), string_cap=64
        ),
        "occurred_at_utc": _bounded_scalar(
            source.get("occurred_at_utc"), string_cap=32
        ),
        "mode": _bounded_scalar(source.get("mode"), string_cap=16),
        "account_id": _bounded_scalar(source.get("account_id"), string_cap=128),
        "model": _bounded_scalar(source.get("model"), string_cap=128),
        "cooldown_seconds": _bounded_scalar(source.get("cooldown_seconds")),
        "cooldown_source": _bounded_scalar(
            source.get("cooldown_source"), string_cap=32
        ),
        "installed_scope": _bounded_scalar(
            source.get("installed_scope"), string_cap=16
        ),
        "quota_family": _bounded_scalar(source.get("quota_family"), string_cap=16),
        "family_gate": _bounded_fields(
            source.get("family_gate"),
            {
                "scope": 16,
                "reason": 128,
                "family_deadline_utc": 32,
            },
        ),
        "observed_scope": _bounded_scalar(
            source.get("observed_scope"), string_cap=16
        ),
        "scope_rationale": _bounded_scalar(
            source.get("scope_rationale"), string_cap=128
        ),
        "attempt": _bounded_fields(
            source.get("attempt"),
            {
                "ordinal": 32,
                "elapsed_ms_since_first": 32,
                "gap_ms_since_previous": 32,
            },
        ),
        "session_fingerprint": _bounded_scalar(
            source.get("session_fingerprint"), string_cap=128
        ),
        "request_shape": _bounded_fields(
            source.get("request_shape"),
            {
                "body_bytes": 32,
                "message_count": 32,
                "tool_count": 32,
                "pin_created": 8,
            },
        ),
        "record_truncated": True,
    }

    upstream = source.get("upstream")
    if isinstance(upstream, dict):
        truncated_upstream = _bounded_fields(
            upstream,
            {
                "body_bytes": 32,
                "body_sha256": 64,
                "error_type": 64,
                "request_id": 128,
            },
        )
        headers = upstream.get("headers")
        if isinstance(headers, dict) and truncated_upstream is not None:
            truncated_upstream["headers"] = {
                "request-id": _bounded_scalar(
                    headers.get("request-id"), string_cap=128
                )
            }
        truncated["upstream"] = truncated_upstream
    else:
        truncated["upstream"] = None

    if source.get("record_degraded") is True:
        truncated["record_degraded"] = True
        truncated["degradation_reason"] = _bounded_scalar(
            source.get("degradation_reason"), string_cap=256
        )

    encoded = _canonical_json(truncated).encode("utf-8")
    if len(encoded) <= _RECORD_BYTES_CAP:
        return encoded.decode("utf-8")
    return '{"event":"claude_quota_429","record_truncated":true,"v":1}'


def _write_all(file_descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError("incident write made no progress")
        remaining = remaining[written:]


def _repair_incomplete_tail(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
    file_descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(file_descriptor, 0o600)
        size = os.fstat(file_descriptor).st_size
        if size == 0:
            return 0

        os.lseek(file_descriptor, size - 1, os.SEEK_SET)
        if os.read(file_descriptor, 1) == b"\n":
            return size

        position = size
        truncate_at = 0
        while position > 0:
            chunk_start = max(0, position - _READ_CHUNK_BYTES)
            os.lseek(file_descriptor, chunk_start, os.SEEK_SET)
            chunk = os.read(file_descriptor, position - chunk_start)
            newline_index = chunk.rfind(b"\n")
            if newline_index >= 0:
                truncate_at = chunk_start + newline_index + 1
                break
            position = chunk_start
        os.ftruncate(file_descriptor, truncate_at)
        return truncate_at
    finally:
        os.close(file_descriptor)


def _read_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    file_descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(file_descriptor)


def _append_line(path: Path, line: bytes) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    file_descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(file_descriptor, 0o600)
        _write_all(file_descriptor, line)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    file_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _replace_with_compacted_file(path: Path, payload: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    file_descriptor = os.open(temporary_path, flags, 0o600)
    try:
        try:
            os.fchmod(file_descriptor, 0o600)
            _write_all(file_descriptor, payload)
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


class Quota429IncidentWriter:
    """Append complete incidents while retaining a bounded newest JSONL suffix."""

    def __init__(
        self, path: Path, *, max_bytes: int = _DEFAULT_RETENTION_BYTES
    ) -> None:
        if not isinstance(max_bytes, int) or max_bytes < _RECORD_BYTES_CAP + 1:
            raise ValueError(
                f"max_bytes must be at least {_RECORD_BYTES_CAP + 1}"
            )
        self._path = path
        self._max_bytes = max_bytes
        self._lock = _INCIDENT_WRITE_LOCK

    async def append_record(self, canonical_record: str) -> None:
        """Append one incident without allowing persistence failures to escape."""
        try:
            encoded_record = canonical_record.encode("utf-8")
            if len(encoded_record) > _RECORD_BYTES_CAP:
                encoded_record = _truncated_record(canonical_record).encode("utf-8")
            line = encoded_record + b"\n"

            async with self._lock:
                current_size = _repair_incomplete_tail(self._path)
                if current_size + len(line) <= self._max_bytes:
                    _append_line(self._path, line)
                    _fsync_directory(self._path.parent)
                    return

                complete_lines = _read_file(self._path).splitlines(keepends=True)
                remaining_bytes = max(0, self._max_bytes - len(line))
                retained_reversed: list[bytes] = []
                for complete_line in reversed(complete_lines):
                    if len(complete_line) > remaining_bytes:
                        break
                    retained_reversed.append(complete_line)
                    remaining_bytes -= len(complete_line)
                retained_reversed.reverse()
                _replace_with_compacted_file(
                    self._path, b"".join(retained_reversed) + line
                )
        except Exception:
            try:
                logger.warning("failed to append Claude 429 incident record")
            except Exception:
                pass
