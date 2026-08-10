"""Static, captured-wire recognition table for Anthropic's unified
`anthropic-ratelimit-*` response headers (T-15), plus the pure functions that
parse a response's headers against it.

Header names and their (window, field) meaning are committed here as LITERAL
constants, generated once from a live two-call capture (T-14's
`scripts/capture_unified_headers.py`) and hand-derived by a human from that
capture's own output -- never guessed, and never re-derived at runtime.

To regenerate this table: run `python3 scripts/capture_unified_headers.py`
against a registered account, inspect the resulting capture's
`ratelimit_headers` entries, hand-edit `RECOGNIZED_HEADERS` below to match
what was actually observed, and re-commit this file.

This module has no import-time or call-time dependency on that capture's
output at all -- only on this file's own already-committed constants. It
never opens a file, reads a path, or loads a package resource; nothing here
is environment-dependent. `RECOGNIZED_HEADERS` is the explicit, literal empty
table this build commits because no capture output was available to
hand-derive entries from at commit time: every parse call below is inert
(recognizes nothing) until a future regeneration replaces it with real
entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from claudex_gateway.usage import _reset_epoch_seconds

HeaderField = Literal["used_percent", "status", "reset"]


@dataclass(frozen=True)
class HeaderDescriptor:
    """One recognized header name's wire meaning: which binding window
    (`five_hour` / `seven_day` / `fable_weekly`) it reports on, and which
    merge-relevant field of that window it carries.
    """

    window: str
    field: HeaderField


# The captured wire table itself. LITERAL -- never computed, never loaded
# from a path, never guessed. Empty because no capture output was available
# to hand-derive entries from at commit time (see module docstring);
# regenerate per the instructions above once one is, and re-commit this
# constant with real entries.
RECOGNIZED_HEADERS: Mapping[str, HeaderDescriptor] = {}


@dataclass(frozen=True)
class ParsedWindowHeaders:
    """One window's recognized-header values pulled out of a single response,
    each `None` when that field's header was absent or failed to parse.
    """

    used_percent: float | None = None
    status: str | None = None
    reset_epoch: float | None = None


def _parse_used_percent(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN never a valid percent.
        return None
    return min(100.0, max(0.0, value))


def _parse_status(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    status = raw.strip().lower()
    return status or None


def parse_unified_headers(
    headers: Mapping[str, str],
    *,
    recognized: Mapping[str, HeaderDescriptor] = RECOGNIZED_HEADERS,
) -> dict[str, ParsedWindowHeaders]:
    """Pure parse: verbatim response headers -> `{window: ParsedWindowHeaders}`.

    Only header names present in `recognized` (matched case-insensitively)
    contribute anything at all; every other header -- including any
    `anthropic-ratelimit-*` header while `recognized` is the committed empty
    table above -- is ignored outright, and the result is `{}`. A field whose
    raw value fails to parse is simply left `None` on its
    `ParsedWindowHeaders` rather than raising; deciding whether a resulting
    incomplete/missing window may still update stored state is the caller's
    job (the balanced router's own merge rules), not this function's.
    """
    lowered_headers = {name.lower(): value for name, value in headers.items()}
    raw_by_window: dict[str, dict[str, Any]] = {}
    for name, descriptor in recognized.items():
        value = lowered_headers.get(name.lower())
        if value is None:
            continue
        bucket = raw_by_window.setdefault(descriptor.window, {})
        if descriptor.field == "used_percent":
            parsed_percent = _parse_used_percent(value)
            if parsed_percent is not None:
                bucket["used_percent"] = parsed_percent
        elif descriptor.field == "status":
            parsed_status = _parse_status(value)
            if parsed_status is not None:
                bucket["status"] = parsed_status
        elif descriptor.field == "reset":
            epoch = _reset_epoch_seconds(value)
            if epoch is not None:
                bucket["reset_epoch"] = epoch

    return {
        window: ParsedWindowHeaders(
            used_percent=fields.get("used_percent"),
            status=fields.get("status"),
            reset_epoch=fields.get("reset_epoch"),
        )
        for window, fields in raw_by_window.items()
    }
