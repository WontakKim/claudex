"""Shape validation for the captured unified-header wire-table fixture.

`tests/fixtures/unified_ratelimit_headers.json` is only published by a fully
successful run of `scripts/capture_unified_headers.py` (both `/v1/messages`
calls 2xx, both exposing a ratelimit header). A graceful no-fixture outcome
-- exhausted quota, offline operation, or refresh failure -- is an
explicitly accepted result of that script's Step 3/4, not a failure, so this
test skips explicitly when the fixture is absent instead of failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "unified_ratelimit_headers.json"

_CALL_KEYS = {"model", "status", "captured_at_utc", "ratelimit_headers"}


def test_fixture_shape_or_skip_when_absent() -> None:
    if not _FIXTURE_PATH.exists():
        pytest.skip(
            f"{_FIXTURE_PATH} was not published (graceful no-fixture outcome is accepted)"
        )

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert set(payload.keys()) == {"calls"}
    calls = payload["calls"]
    assert isinstance(calls, list)
    assert len(calls) == 2

    for call in calls:
        assert isinstance(call, dict)
        assert set(call.keys()) == _CALL_KEYS

        assert isinstance(call["model"], str) and call["model"]

        status = call["status"]
        assert isinstance(status, int) and not isinstance(status, bool)
        assert 200 <= status < 300

        assert isinstance(call["captured_at_utc"], str) and call["captured_at_utc"]

        headers = call["ratelimit_headers"]
        assert isinstance(headers, dict) and headers
        for name, value in headers.items():
            assert isinstance(name, str) and name == name.lower()
            assert name.startswith("anthropic-ratelimit-")
            assert isinstance(value, str)
