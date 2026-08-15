"""Tests for the subscription usage probes behind the dashboard's Usage tab."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from claudex_gateway.claude import usage as claude_usage
from claudex_gateway.usage import envelope as usage_envelope
from claudex_gateway.usage import providers as provider_usage
from claudex_gateway.claude.auth import (
    ClaudeAccountAuthError,
    ClaudeAccountCredentials,
    ClaudeAccountReauthRequiredError,
)
from claudex_gateway.providers.codex_auth import CodexAuthError, CodexCredentials
from claudex_gateway.providers.kimi_auth import KimiAuthError, KimiCredentials
from claudex_gateway.providers.grok_auth import GrokAuthError, GrokCredentials


def _run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


def _mock_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _unused_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.url}")

    return _mock_client(handler)


class ChatGPTCredentials:
    def __init__(self, credentials: CodexCredentials) -> None:
        self._credentials = credentials

    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        return self._credentials


class MissingCredentials:
    async def get_credentials(self, force_refresh: bool = False) -> CodexCredentials:
        raise CodexAuthError("missing Codex credentials")


# ---------------------------------------------------------------------------
# Claude credential resolution
# ---------------------------------------------------------------------------


def test_parse_claude_credentials_extracts_access_token() -> None:
    raw = json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat-1", "refreshToken": "r"}})
    assert claude_usage._parse_claude_credentials(raw) == "sk-ant-oat-1"


def test_parse_claude_credentials_rejects_garbage() -> None:
    assert claude_usage._parse_claude_credentials("not json") is None
    assert claude_usage._parse_claude_credentials(json.dumps({"claudeAiOauth": {}})) is None
    assert claude_usage._parse_claude_credentials(json.dumps({"other": 1})) is None


def test_claude_keychain_services_scoped_when_config_dir_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/custom-claude")
    services = claude_usage._claude_keychain_services()
    assert len(services) == 2
    assert services[0].startswith("Claude Code-credentials-")
    assert services[1] == "Claude Code-credentials"

    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert claude_usage._claude_keychain_services() == ["Claude Code-credentials"]


def test_resolve_claude_token_reads_credentials_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    async def no_keychain(service: str, account: str) -> None:
        return None

    monkeypatch.setattr(claude_usage, "_keychain_password", no_keychain)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "file-token"}})
    )
    assert _run(claude_usage._resolve_claude_oauth_token()) == "file-token"


def test_resolve_claude_token_prefers_keychain(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    async def keychain(service: str, account: str) -> str:
        return json.dumps({"claudeAiOauth": {"accessToken": "keychain-token"}})

    monkeypatch.setattr(claude_usage, "_keychain_password", keychain)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "file-token"}})
    )
    assert _run(claude_usage._resolve_claude_oauth_token()) == "keychain-token"


# ---------------------------------------------------------------------------
# Claude usage fetch
# ---------------------------------------------------------------------------


def test_fetch_claude_usage_maps_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stub_token() -> str:
        return "tok"

    monkeypatch.setattr(claude_usage, "_resolve_claude_oauth_token", stub_token)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx.Response(
            200,
            json={
                "five_hour": {"utilization": 42.5, "resets_at": "2026-07-31T15:00:00Z"},
                "seven_day": {"utilization": 120, "resets_at": 1754500000},
            },
        )

    result = _run(claude_usage.fetch_claude_usage(_mock_client(handler)))

    assert seen["authorization"] == "Bearer tok"
    assert seen["beta"] == "oauth-2025-04-20"
    assert result["status"] == "ok"
    assert result["session"]["used_percent"] == 42.5
    assert result["session"]["window_minutes"] == 300
    # ISO resets_at is normalized to epoch seconds.
    expected = datetime(2026, 7, 31, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    assert result["session"]["resets_at"] == expected
    # Out-of-range percentages are clamped.
    assert result["weekly"]["used_percent"] == 100.0
    assert result["weekly"]["resets_at"] == 1754500000.0


def test_fetch_claude_usage_401_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stub_token() -> str:
        return "tok"

    monkeypatch.setattr(claude_usage, "_resolve_claude_oauth_token", stub_token)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid token"}})

    result = _run(claude_usage.fetch_claude_usage(_mock_client(handler)))
    assert result["status"] == "error"
    assert "401" in result["error"]


def test_fetch_claude_usage_maps_fable_weekly_from_scoped_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stub_token() -> str:
        return "tok"

    monkeypatch.setattr(claude_usage, "_resolve_claude_oauth_token", stub_token)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "five_hour": {"utilization": 22, "resets_at": 1754000000},
                "seven_day": {"utilization": 67, "resets_at": 1754600000},
                "limits": [
                    {"kind": "session", "percent": 22, "resets_at": 1754000000, "scope": None},
                    {"kind": "weekly_all", "percent": 67, "resets_at": 1754600000, "scope": None},
                    {
                        "kind": "weekly_scoped",
                        "percent": 99,
                        "resets_at": "2026-08-04T20:59:59+00:00",
                        "scope": {"model": {"id": None, "display_name": "Fable"}},
                        "is_active": True,
                    },
                ],
            },
        )

    result = _run(claude_usage.fetch_claude_usage(_mock_client(handler)))

    assert result["status"] == "ok"
    assert result["fable_weekly"]["used_percent"] == 99.0
    assert result["fable_weekly"]["window_minutes"] == 10080
    expected = datetime(2026, 8, 4, 20, 59, 59, tzinfo=timezone.utc).timestamp()
    assert result["fable_weekly"]["resets_at"] == expected


@pytest.mark.parametrize(
    "field_name", ["fable_weekly", "fable_seven_day", "seven_day_fable"]
)
def test_flat_fable_quota_fields_are_ignored(field_name: str) -> None:
    data = {field_name: {"utilization": 55, "resets_at": 1754600000}}

    assert claude_usage._map_fable_weekly_window(data) is None


def test_map_fable_weekly_window_ignores_malformed_or_nonmatching_limits() -> None:
    assert claude_usage._map_fable_weekly_window({"limits": "malformed"}) is None
    data = {
        "limits": [
            None,
            {"kind": "weekly_scoped", "percent": 10, "scope": {}},
            {
                "kind": "weekly_scoped",
                "percent": 10,
                "scope": {"model": {"display_name": "Opus"}},
            },
        ]
    }

    assert claude_usage._map_fable_weekly_window(data) is None


def test_fetch_claude_usage_without_credentials_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_token() -> None:
        return None

    monkeypatch.setattr(claude_usage, "_resolve_claude_oauth_token", no_token)
    result = _run(claude_usage.fetch_claude_usage(_unused_client()))
    assert result["status"] == "unavailable"
    assert result["session"] is None


# ---------------------------------------------------------------------------
# Per-account Claude usage fetch
# ---------------------------------------------------------------------------


class _AccountCredentialsStub:
    """Duck-typed ClaudeAccountAuthManager double handing out fixed tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.calls: list[bool] = []

    async def get_credentials(self, force_refresh: bool = False) -> ClaudeAccountCredentials:
        self.calls.append(force_refresh)
        token = self._tokens[min(len(self.calls) - 1, len(self._tokens) - 1)]
        return ClaudeAccountCredentials(access_token=token, account_uuid=None)


class _FailingAccountCredentials:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_credentials(self, force_refresh: bool = False) -> ClaudeAccountCredentials:
        raise self._error


def test_fetch_claude_account_usage_sends_the_account_bearer() -> None:
    manager = _AccountCredentialsStub(["acct-tok"])
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["user_agent"] = request.headers.get("user-agent")
        return httpx.Response(
            200, json={"five_hour": {"utilization": 10, "resets_at": 1754500000}}
        )

    result, retry_after = _run(claude_usage.fetch_claude_account_usage(_mock_client(handler), manager))

    assert seen["authorization"] == "Bearer acct-tok"
    assert seen["beta"] == "oauth-2025-04-20"
    assert seen["user_agent"] == "claude-code/2.1.0"
    assert result["status"] == "ok"
    assert result["session"]["used_percent"] == 10.0
    assert retry_after is None


def test_fetch_claude_account_usage_429_returns_retry_after_seconds() -> None:
    manager = _AccountCredentialsStub(["acct-tok"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"})

    result, retry_after = _run(claude_usage.fetch_claude_account_usage(_mock_client(handler), manager))
    assert result["status"] == "error"
    assert "429" in result["error"]
    assert retry_after == 30.0


def test_fetch_claude_account_usage_429_parses_http_date_retry_after() -> None:
    manager = _AccountCredentialsStub(["acct-tok"])
    retry_at = datetime.now(timezone.utc).timestamp() + 120

    def handler(request: httpx.Request) -> httpx.Response:
        http_date = datetime.fromtimestamp(retry_at, tz=timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        return httpx.Response(429, headers={"Retry-After": http_date})

    _result, retry_after = _run(claude_usage.fetch_claude_account_usage(_mock_client(handler), manager))
    assert retry_after is not None
    assert 0 < retry_after <= 121


def test_fetch_claude_account_usage_retries_once_with_a_forced_refresh() -> None:
    manager = _AccountCredentialsStub(["stale-tok", "fresh-tok"])
    tokens_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tokens_seen.append(request.headers.get("authorization", ""))
        if len(tokens_seen) == 1:
            return httpx.Response(401)
        return httpx.Response(
            200, json={"five_hour": {"utilization": 5, "resets_at": 1754500000}}
        )

    result, _retry_after = _run(claude_usage.fetch_claude_account_usage(_mock_client(handler), manager))

    assert tokens_seen == ["Bearer stale-tok", "Bearer fresh-tok"]
    assert manager.calls == [False, True]
    assert result["status"] == "ok"


def test_fetch_claude_account_usage_persistent_401_is_an_error() -> None:
    manager = _AccountCredentialsStub(["tok-1", "tok-2"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    result, _retry_after = _run(claude_usage.fetch_claude_account_usage(_mock_client(handler), manager))
    assert result["status"] == "error"
    assert "after refresh" in result["error"]


def test_fetch_claude_account_usage_auth_error_is_unavailable() -> None:
    manager = _FailingAccountCredentials(ClaudeAccountAuthError("no credentials file"))
    result, retry_after = _run(claude_usage.fetch_claude_account_usage(_unused_client(), manager))
    assert result["status"] == "unavailable"
    assert "no credentials file" in result["error"]
    assert retry_after is None


def test_fetch_claude_account_usage_reauth_required_propagates() -> None:
    manager = _FailingAccountCredentials(ClaudeAccountReauthRequiredError("invalid_grant"))
    with pytest.raises(ClaudeAccountReauthRequiredError):
        _run(claude_usage.fetch_claude_account_usage(_unused_client(), manager))


# ---------------------------------------------------------------------------
# Codex usage fetch
# ---------------------------------------------------------------------------


def test_fetch_codex_usage_maps_windows_and_plan() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["account"] = request.headers.get("chatgpt-account-id")
        seen["originator"] = request.headers.get("originator")
        return httpx.Response(
            200,
            json={
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 12,
                        "limit_window_seconds": 18000,
                        "reset_at": 1754000000,
                    },
                    "secondary_window": {
                        "used_percent": 34,
                        "limit_window_seconds": 604800,
                        "reset_at": 1754600000,
                    },
                },
                "rate_limit_reset_credits": {"available_count": 2},
            },
        )

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id="acc-1"))
    result = _run(provider_usage.fetch_codex_usage(_mock_client(handler), auth))

    assert seen["account"] == "acc-1"
    assert seen["originator"] == "Codex Desktop"
    assert result["status"] == "ok"
    assert result["plan_type"] == "plus"
    assert result["reset_credits_available"] == 2
    assert result["session"]["used_percent"] == 12
    assert result["session"]["window_minutes"] == 300
    assert result["session"]["resets_at"] == 1754000000.0
    assert result["weekly"]["used_percent"] == 34
    assert result["weekly"]["window_minutes"] == 10080


def test_fetch_codex_usage_classifies_unknown_durations_legacy_way() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {"used_percent": 5, "reset_at": 1754000000},
                    "secondary_window": {"used_percent": 7, "reset_at": 1754600000},
                },
            },
        )

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id=None))
    result = _run(provider_usage.fetch_codex_usage(_mock_client(handler), auth))

    assert result["session"]["used_percent"] == 5
    assert result["weekly"]["used_percent"] == 7


def test_fetch_codex_usage_api_key_mode_is_unavailable() -> None:
    auth = ChatGPTCredentials(
        CodexCredentials(access_token="sk-key", account_id=None, is_api_key=True)
    )
    result = _run(provider_usage.fetch_codex_usage(_unused_client(), auth))
    assert result["status"] == "unavailable"
    assert "API key" in result["error"]


def test_fetch_codex_usage_missing_credentials_is_unavailable() -> None:
    result = _run(provider_usage.fetch_codex_usage(_unused_client(), MissingCredentials()))
    assert result["status"] == "unavailable"
    assert "missing Codex credentials" in result["error"]


def test_fetch_codex_usage_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id=None))
    result = _run(provider_usage.fetch_codex_usage(_mock_client(handler), auth))
    assert result["status"] == "error"
    assert "500" in result["error"]


# ---------------------------------------------------------------------------
# Codex reset credit consumption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code", ["reset", "nothing_to_reset", "no_credit", "already_redeemed"]
)
def test_consume_codex_reset_credit_reports_every_backend_outcome(code: str) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["account"] = request.headers.get("chatgpt-account-id")
        return httpx.Response(200, json={"code": code})

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id="acc-1"))
    result = _run(provider_usage.consume_codex_reset_credit(_mock_client(handler), auth, "key-1"))

    assert seen["method"] == "POST"
    assert seen["url"].endswith("/wham/rate-limit-reset-credits/consume")
    assert seen["body"] == {"redeem_request_id": "key-1"}
    assert seen["account"] == "acc-1"
    # Every recognised code is a settled attempt, even the ones that spent nothing.
    assert result == {"status": "ok", "outcome": code, "error": None}


def test_consume_codex_reset_credit_treats_unknown_outcome_as_unsettled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "teleported"})

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id=None))
    result = _run(provider_usage.consume_codex_reset_credit(_mock_client(handler), auth, "key-1"))

    # The credit may have been spent, so this must not settle the attempt.
    assert result["status"] == "error"
    assert result["outcome"] is None
    assert "teleported" in result["error"]


def test_consume_codex_reset_credit_upstream_error_is_unsettled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="nope")

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id=None))
    result = _run(provider_usage.consume_codex_reset_credit(_mock_client(handler), auth, "key-1"))
    assert result["status"] == "error"
    assert "503" in result["error"]


def test_consume_codex_reset_credit_transport_error_is_unsettled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    auth = ChatGPTCredentials(CodexCredentials(access_token="tok", account_id=None))
    result = _run(provider_usage.consume_codex_reset_credit(_mock_client(handler), auth, "key-1"))
    assert result["status"] == "error"
    assert result["outcome"] is None


def test_consume_codex_reset_credit_never_calls_out_without_usable_credentials() -> None:
    api_key = ChatGPTCredentials(
        CodexCredentials(access_token="sk-key", account_id=None, is_api_key=True)
    )
    assert _run(
        provider_usage.consume_codex_reset_credit(_unused_client(), api_key, "key-1")
    )["status"] == "unavailable"
    assert _run(
        provider_usage.consume_codex_reset_credit(_unused_client(), MissingCredentials(), "key-1")
    )["status"] == "unavailable"


# ---------------------------------------------------------------------------
# Kimi usage fetch
# ---------------------------------------------------------------------------


class KimiCredentialsStore:
    def __init__(self, access_token: str = "kimi-tok") -> None:
        self._credentials = KimiCredentials(access_token=access_token, device_id="device-1")

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        return self._credentials


class MissingKimiCredentials:
    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        raise KimiAuthError("missing Kimi credentials")


def test_fetch_kimi_usage_maps_windows_and_plan() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        # The real payload shape from GET /coding/v1/usages.
        return httpx.Response(
            200,
            json={
                "user": {"membership": {"level": "LEVEL_STANDARD"}},
                "usage": {
                    "limit": "100",
                    "used": "5",
                    "remaining": "95",
                    "resetTime": "2026-08-05T02:19:09.474659Z",
                },
                "limits": [
                    {
                        "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
                        "detail": {
                            "limit": "100",
                            "used": "4",
                            "remaining": "96",
                            "resetTime": "2026-07-31T19:19:09.474659Z",
                        },
                    }
                ],
            },
        )

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))

    assert seen["authorization"] == "Bearer kimi-tok"
    assert result["status"] == "ok"
    assert result["plan_type"] == "standard"
    assert result["session"]["used_percent"] == 4.0
    assert result["session"]["window_minutes"] == 300
    expected_session = datetime(2026, 7, 31, 19, 19, 9, 474659, tzinfo=timezone.utc).timestamp()
    assert result["session"]["resets_at"] == expected_session
    assert result["weekly"]["used_percent"] == 5.0
    assert result["weekly"]["window_minutes"] == 10080
    expected_weekly = datetime(2026, 8, 5, 2, 19, 9, 474659, tzinfo=timezone.utc).timestamp()
    assert result["weekly"]["resets_at"] == expected_weekly


def test_fetch_kimi_usage_derives_used_from_remaining() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"usage": {"limit": "100", "remaining": "90", "resetAt": 1754600000}},
        )

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))

    assert result["status"] == "ok"
    assert result["weekly"]["used_percent"] == 10.0
    assert result["weekly"]["resets_at"] == 1754600000.0


def test_fetch_kimi_usage_picks_window_closest_to_five_hours() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "limits": [
                    {
                        "window": {"duration": 1, "timeUnit": "TIME_UNIT_DAY"},
                        "detail": {"limit": "50", "used": "25"},
                    },
                    {
                        "window": {"duration": 5, "timeUnit": "TIME_UNIT_HOUR"},
                        "detail": {"limit": "100", "used": "4"},
                    },
                ]
            },
        )

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))

    assert result["session"]["used_percent"] == 4.0
    assert result["session"]["window_minutes"] == 300


def test_fetch_kimi_usage_missing_credentials_is_unavailable() -> None:
    result = _run(provider_usage.fetch_kimi_usage(_unused_client(), MissingKimiCredentials()))
    assert result["status"] == "unavailable"
    assert "missing Kimi credentials" in result["error"]


def test_fetch_kimi_usage_401_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid token"}})

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))
    assert result["status"] == "error"
    assert "401" in result["error"]


def test_fetch_kimi_usage_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))
    assert result["status"] == "error"
    assert "500" in result["error"]


def test_fetch_kimi_usage_without_windows_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"user": {}})

    result = _run(provider_usage.fetch_kimi_usage(_mock_client(handler), KimiCredentialsStore()))
    assert result["status"] == "error"
    assert "quota windows" in result["error"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_usage_envelope_reset_time_normalizes_units() -> None:
    assert usage_envelope.reset_epoch_seconds(1754000000) == 1754000000.0
    assert usage_envelope.reset_epoch_seconds(1754000000000) == 1754000000.0
    assert usage_envelope.reset_epoch_seconds("1754000000") == 1754000000.0
    expected = datetime.fromisoformat("2026-07-31T15:00:00+00:00").timestamp()
    assert usage_envelope.reset_epoch_seconds("2026-07-31T15:00:00+00:00") == expected
    assert usage_envelope.reset_epoch_seconds("2026-07-31T15:00:00Z") == expected
    assert usage_envelope.reset_epoch_seconds("not a date") is None
    assert usage_envelope.reset_epoch_seconds(None) is None
    assert usage_envelope.reset_epoch_seconds(True) is None


# ---------------------------------------------------------------------------
# Grok usage probe (Grok chat-proxy billing endpoint)
# ---------------------------------------------------------------------------


class GrokCredentialsStore:
    def __init__(self, access_token: str = "grok-tok") -> None:
        self._credentials = GrokCredentials(
            access_token=access_token, email=None, user_id="user-1"
        )

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        return self._credentials


class MissingGrokCredentials:
    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        raise GrokAuthError("missing Grok credentials")


def test_fetch_grok_usage_maps_weekly_credits_and_tier() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["token_auth"] = request.headers.get("x-xai-token-auth")
        seen["userid"] = request.headers.get("x-userid")
        return httpx.Response(
            200,
            json={
                "config": {
                    "creditUsagePercent": 42.5,
                    "subscriptionTier": "SUPERGROK",
                    "currentPeriod": {
                        "type": "USAGE_PERIOD_TYPE_WEEKLY",
                        "start": "2026-07-30T00:00:00Z",
                        "end": "2026-08-06T00:00:00Z",
                    },
                }
            },
        )

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))

    assert seen["url"].endswith("/v1/billing?format=credits")
    assert seen["authorization"] == "Bearer grok-tok"
    assert seen["token_auth"] == "xai-grok-cli"
    assert seen["userid"] == "user-1"
    assert result["status"] == "ok"
    assert result["plan_type"] == "supergrok"
    assert result["weekly"]["used_percent"] == 42.5
    assert result["weekly"]["window_minutes"] == 10080
    expected = datetime(2026, 8, 6, tzinfo=timezone.utc).timestamp()
    assert result["weekly"]["resets_at"] == expected
    assert result["monthly"] is None


def test_fetch_grok_usage_accepts_top_level_config() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"creditUsagePercent": 10})

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))

    assert result["status"] == "ok"
    assert result["weekly"]["used_percent"] == 10.0


def test_fetch_grok_usage_confirmed_weekly_period_means_zero_used() -> None:
    # Grok omits the protobuf zero, so a missing percent with matching billing
    # bounds is a full weekly budget, not absent data (mirrors Orca).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "config": {
                    "currentPeriod": {
                        "type": "USAGE_PERIOD_TYPE_WEEKLY",
                        "start": "2026-07-30T00:00:00Z",
                        "end": "2026-08-06T00:00:00Z",
                    },
                    "billingPeriodStart": "2026-07-30T00:00:00Z",
                    "billingPeriodEnd": "2026-08-06T00:00:00Z",
                }
            },
        )

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))

    assert result["status"] == "ok"
    assert result["weekly"]["used_percent"] == 0.0


def test_fetch_grok_usage_falls_back_to_monthly_budget() -> None:
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if "format=credits" in str(request.url):
            # Unified-billing account: the credits view carries no percent.
            return httpx.Response(200, json={"config": {"subscriptionTier": "PREMIUM"}})
        return httpx.Response(
            200,
            json={
                "config": {
                    "monthlyLimit": {"val": "100.0"},
                    "used": {"val": "25.0"},
                    "billingPeriodEnd": "2026-09-01T00:00:00Z",
                }
            },
        )

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))

    assert len(urls) == 2
    assert result["status"] == "ok"
    assert result["weekly"] is None
    assert result["monthly"]["used_percent"] == 25.0
    assert result["monthly"]["window_minutes"] == 43200
    assert result["plan_type"] == "premium"


def test_fetch_grok_usage_without_credit_usage_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"config": {}})

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))

    assert result["status"] == "unavailable"


def test_fetch_grok_usage_missing_credentials_is_unavailable() -> None:
    result = _run(provider_usage.fetch_grok_usage(_unused_client(), MissingGrokCredentials()))
    assert result["status"] == "unavailable"
    assert "missing Grok credentials" in result["error"]


def test_fetch_grok_usage_401_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid token"})

    result = _run(provider_usage.fetch_grok_usage(_mock_client(handler), GrokCredentialsStore()))
    assert result["status"] == "error"
    assert "401" in result["error"]
