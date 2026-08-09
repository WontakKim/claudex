"""Mocked unit tests for `scripts/capture_unified_headers.py`.

Every test here runs against `httpx.MockTransport` and never contacts
Anthropic. They prove: exactly two requests per probe run, the probe-only
`--include-fable` flag selects the right model pair, every request body
carries `max_tokens: 1`, a failed second call never triggers a third
request, a failing run publishes no fixture, a fully successful run
publishes one atomically (with no leftover staging file), and neither the
captured stdout/stderr nor the published fixture ever contain the account's
access token or the probe's message content.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex_gateway import claude_accounts
from scripts import capture_unified_headers


def _ratelimit_response(status: int, model: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={"id": "msg_1", "model": model, "content": []},
        headers={"anthropic-ratelimit-requests-remaining": "999"},
    )


def _write_account_dir(account_dir: Path, access_token: str) -> None:
    account_dir.mkdir(parents=True, exist_ok=True)
    oauth_blob: dict[str, Any] = {
        "accessToken": access_token,
        "refreshToken": "refresh-1",
        "expiresAt": (time.time() + 3600) * 1000,
    }
    (account_dir / "credentials.json").write_text(
        json.dumps({"claudeAiOauth": oauth_blob}), encoding="utf-8"
    )
    (account_dir / "oauth-account.json").write_text(
        json.dumps({"accountUuid": "11111111-1111-1111-1111-111111111111"}), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Probe-only flag / model selection
# --------------------------------------------------------------------------


def test_default_model_pair_is_sonnet_twice() -> None:
    assert capture_unified_headers.select_model_pair(False) == ("claude-sonnet-5", "claude-sonnet-5")


def test_include_fable_flag_selects_sonnet_then_fable() -> None:
    assert capture_unified_headers.select_model_pair(True) == ("claude-sonnet-5", "claude-fable-5")


def test_cli_include_fable_flag_defaults_false_and_is_settable() -> None:
    assert capture_unified_headers._parse_args([]).include_fable is False
    assert capture_unified_headers._parse_args(["--include-fable"]).include_fable is True


# --------------------------------------------------------------------------
# probe_unified_headers -- exactly two requests, no retry, no third call
# --------------------------------------------------------------------------


def test_probe_issues_exactly_two_requests_with_max_tokens_one() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        return _ratelimit_response(200, body["model"])

    async def scenario() -> tuple[capture_unified_headers.CallResult, capture_unified_headers.CallResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.probe_unified_headers(
                http_client, "token-value", include_fable=False
            )

    first, second = asyncio.run(scenario())

    assert len(seen) == 2
    bodies = [json.loads(request.content) for request in seen]
    assert [body["model"] for body in bodies] == ["claude-sonnet-5", "claude-sonnet-5"]
    assert all(body["max_tokens"] == 1 for body in bodies)
    assert first.status == 200
    assert second.status == 200


def test_probe_include_fable_calls_sonnet_then_fable() -> None:
    seen_models: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_models.append(body["model"])
        return _ratelimit_response(200, body["model"])

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await capture_unified_headers.probe_unified_headers(
                http_client, "token-value", include_fable=True
            )

    asyncio.run(scenario())

    assert seen_models == ["claude-sonnet-5", "claude-fable-5"]


def test_probe_sends_the_required_headers() -> None:
    seen_headers: list[httpx.Headers] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        return _ratelimit_response(200, "claude-sonnet-5")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await capture_unified_headers.probe_unified_headers(
                http_client, "secret-token-abc", include_fable=False
            )

    asyncio.run(scenario())

    assert len(seen_headers) == 2
    for headers in seen_headers:
        assert headers["authorization"] == "Bearer secret-token-abc"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "oauth-2025-04-20" in headers["anthropic-beta"]


def test_no_third_request_follows_a_failed_second_call() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return _ratelimit_response(200, "claude-sonnet-5")
        return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})

    async def scenario() -> tuple[capture_unified_headers.CallResult, capture_unified_headers.CallResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.probe_unified_headers(
                http_client, "token-value", include_fable=False
            )

    first, second = asyncio.run(scenario())

    assert len(seen) == 2
    assert first.ok is True
    assert second.ok is False
    assert second.status == 429


def test_second_call_still_attempted_after_first_call_network_failure() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ConnectError("boom", request=request)
        return _ratelimit_response(200, "claude-sonnet-5")

    async def scenario() -> tuple[capture_unified_headers.CallResult, capture_unified_headers.CallResult]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.probe_unified_headers(
                http_client, "token-value", include_fable=False
            )

    first, second = asyncio.run(scenario())

    assert len(seen) == 2
    assert first.status is None
    assert first.ok is False
    assert second.status == 200


# --------------------------------------------------------------------------
# run_capture -- publish-on-success, no-fixture-on-failure, atomic write
# --------------------------------------------------------------------------


def test_full_success_publishes_fixture_atomically(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"
    _write_account_dir(account_dir, "secret-access-token-value")
    fixture_path = tmp_path / "fixtures" / "unified_ratelimit_headers.json"

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return _ratelimit_response(200, body["model"])

    async def scenario() -> capture_unified_headers.CaptureOutcome:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.run_capture(
                http_client, account_dir, include_fable=False, fixture_path=fixture_path
            )

    outcome = asyncio.run(scenario())

    assert outcome.success is True
    assert fixture_path.exists()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"calls"}
    assert len(payload["calls"]) == 2
    for call in payload["calls"]:
        assert set(call.keys()) == {"model", "status", "captured_at_utc", "ratelimit_headers"}
        assert call["status"] == 200
        assert call["ratelimit_headers"] == {"anthropic-ratelimit-requests-remaining": "999"}
    # No leftover staging artifact from the atomic publish step.
    assert list(fixture_path.parent.glob(".*.tmp-*")) == []


def test_failure_publishes_no_fixture(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"
    _write_account_dir(account_dir, "secret-access-token-value")
    fixture_path = tmp_path / "fixtures" / "unified_ratelimit_headers.json"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})

    async def scenario() -> capture_unified_headers.CaptureOutcome:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.run_capture(
                http_client, account_dir, include_fable=False, fixture_path=fixture_path
            )

    outcome = asyncio.run(scenario())

    assert outcome.success is False
    assert not fixture_path.exists()


def test_missing_credentials_reports_diagnostic_without_any_request(tmp_path: Path) -> None:
    account_dir = tmp_path / "account"  # never created: no credentials.json
    fixture_path = tmp_path / "fixtures" / "unified_ratelimit_headers.json"
    request_count = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        request_count["n"] += 1
        return _ratelimit_response(200, "claude-sonnet-5")

    async def scenario() -> capture_unified_headers.CaptureOutcome:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.run_capture(
                http_client, account_dir, include_fable=False, fixture_path=fixture_path
            )

    outcome = asyncio.run(scenario())

    assert outcome.success is False
    assert request_count["n"] == 0
    assert not fixture_path.exists()


# --------------------------------------------------------------------------
# No credential or message-content leakage
# --------------------------------------------------------------------------


def test_no_secret_or_prompt_content_in_output_or_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret_token = "SECRET-ACCESS-TOKEN-4F91"
    account_dir = tmp_path / "account"
    _write_account_dir(account_dir, secret_token)
    fixture_path = tmp_path / "fixtures" / "unified_ratelimit_headers.json"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert secret_token not in str(request.url)
        body = json.loads(request.content)
        return _ratelimit_response(200, body["model"])

    async def scenario() -> capture_unified_headers.CaptureOutcome:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await capture_unified_headers.run_capture(
                http_client, account_dir, include_fable=False, fixture_path=fixture_path
            )

    outcome = asyncio.run(scenario())
    captured = capsys.readouterr()

    assert secret_token not in captured.out
    assert secret_token not in captured.err
    assert secret_token not in outcome.diagnostic
    assert capture_unified_headers._PROBE_PROMPT not in captured.out
    assert capture_unified_headers._PROBE_PROMPT not in captured.err

    fixture_text = fixture_path.read_text(encoding="utf-8")
    assert secret_token not in fixture_text
    assert capture_unified_headers._PROBE_PROMPT not in fixture_text


def test_report_outcome_prints_diagnostic_without_leaking(
    capsys: pytest.CaptureFixture[str],
) -> None:
    outcome = capture_unified_headers.CaptureOutcome(
        success=False,
        calls=None,
        diagnostic="could not load account credentials (ClaudeAccountAuthError)",
    )

    exit_code = capture_unified_headers._report_outcome(outcome)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "capture failed" in captured.err
    assert "Bearer" not in captured.err


# --------------------------------------------------------------------------
# First ready registered account lookup
# --------------------------------------------------------------------------


def test_first_ready_account_id_returns_none_without_accounts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert capture_unified_headers._first_ready_account_id() is None


def test_first_ready_account_id_returns_the_ready_account(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    record = claude_accounts.add_account(
        "user@example.com",
        "org-1",
        "Org One",
        {"accessToken": "at-1"},
        {"accountUuid": "22222222-2222-2222-2222-222222222222"},
    )

    assert capture_unified_headers._first_ready_account_id() == record.id
