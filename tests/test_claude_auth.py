"""Tests for registered Claude account credential loading, refresh, and coalescing."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex.claude.auth import (
    CLAUDE_CLIENT_ID,
    CLAUDE_TOKEN_URL,
    ClaudeAccountAuthError,
    ClaudeAccountAuthManager,
    ClaudeAccountReauthRequiredError,
)


def _write_account_dir(
    account_dir: Path,
    access_token: str,
    expires_at_ms: float | None,
    *,
    scopes: list[str] | None = None,
    account_uuid: str | None = "acct-uuid-1",
) -> None:
    oauth_blob: dict[str, Any] = {
        "accessToken": access_token,
        "refreshToken": "refresh-1",
        "subscriptionType": "max",
    }
    if expires_at_ms is not None:
        oauth_blob["expiresAt"] = expires_at_ms
    if scopes is not None:
        oauth_blob["scopes"] = scopes
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "credentials.json").write_text(
        json.dumps({"claudeAiOauth": oauth_blob}), encoding="utf-8"
    )
    oauth_account: dict[str, Any] = {"emailAddress": "a@example.com"}
    if account_uuid is not None:
        oauth_account["accountUuid"] = account_uuid
    (account_dir / "oauth-account.json").write_text(json.dumps(oauth_account), encoding="utf-8")


def _refresh_transport(counter: dict[str, int], new_access_token: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CLAUDE_TOKEN_URL
        payload = json.loads(request.content)
        assert payload["client_id"] == CLAUDE_CLIENT_ID
        assert payload["grant_type"] == "refresh_token"
        assert payload["refresh_token"] == "refresh-1"
        counter["posts"] += 1
        # Suspend while holding the refresh lock so a concurrent caller
        # genuinely waits on it instead of racing past.
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": new_access_token,
                "refresh_token": "refresh-2",
                "expires_in": 900,
            },
        )

    return httpx.MockTransport(handler)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _in_ms(seconds_from_now: float) -> float:
    return (time.time() + seconds_from_now) * 1000


def test_concurrent_forced_refreshes_rotate_only_once(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(3600))
    counter = {"posts": 0}

    async def scenario() -> list[Any]:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = ClaudeAccountAuthManager(tmp_path, http_client)
            # Two requests saw a 401 with the same stale token and retry.
            return list(
                await asyncio.gather(
                    manager.get_credentials(force_refresh=True),
                    manager.get_credentials(force_refresh=True),
                )
            )

    first, second = _run(scenario())

    assert counter["posts"] == 1
    assert first.access_token == "rotated-token"
    assert second.access_token == "rotated-token"
    persisted = json.loads((tmp_path / "credentials.json").read_text())["claudeAiOauth"]
    assert persisted["accessToken"] == "rotated-token"
    assert persisted["refreshToken"] == "refresh-2"
    # Claude Code's record shape: absolute expiry in epoch milliseconds.
    assert persisted["expiresAt"] > time.time() * 1000
    # Untouched blob fields survive the rewrite.
    assert persisted["subscriptionType"] == "max"


def test_sequential_forced_refreshes_each_rotate(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(3600))
    counter = {"posts": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        counter["posts"] += 1
        # The second rotation must spend the refresh token the first one wrote.
        assert payload["refresh_token"] == f"refresh-{counter['posts']}"
        return httpx.Response(
            200,
            json={
                "access_token": f"rotated-{counter['posts']}",
                "refresh_token": f"refresh-{counter['posts'] + 1}",
                "expires_in": 900,
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            manager = ClaudeAccountAuthManager(tmp_path, http_client)
            await manager.get_credentials(force_refresh=True)
            # A later 401 against the rotated token is a new stale generation.
            await manager.get_credentials(force_refresh=True)

    _run(scenario())

    assert counter["posts"] == 2


def test_expiring_token_is_refreshed_without_force(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(60))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 1
    assert credentials.access_token == "rotated-token"


def test_valid_token_is_returned_without_refresh(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "valid-token", _in_ms(3600))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.access_token == "valid-token"
    assert credentials.account_uuid == "acct-uuid-1"


def test_missing_expiry_counts_as_valid(tmp_path: Path) -> None:
    # Refreshing on absence would loop forever when the token endpoint omits
    # expires_in; the 401 retry path covers silent expiry instead.
    _write_account_dir(tmp_path, "valid-token", None)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).access_token == "valid-token"
    assert counter["posts"] == 0


def test_stored_scopes_are_sent_space_joined(tmp_path: Path) -> None:
    _write_account_dir(
        tmp_path, "stale-token", _in_ms(60), scopes=["user:inference", "user:profile"]
    )
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"access_token": "rotated", "refresh_token": "refresh-2", "expires_in": 900}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    _run(scenario())

    assert seen["scope"] == "user:inference user:profile"


def test_missing_credentials_file_raises_with_guidance(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountAuthError, match="account add"):
        _run(scenario())


def test_missing_claude_ai_oauth_object_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "credentials.json").write_text(json.dumps({"other": 1}), encoding="utf-8")

    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountAuthError, match="claudeAiOauth"):
        _run(scenario())


def test_missing_oauth_account_file_yields_no_account_uuid(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "valid-token", _in_ms(3600))
    (tmp_path / "oauth-account.json").unlink()

    async def scenario() -> Any:
        async with httpx.AsyncClient() as http_client:
            return await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    assert _run(scenario()).account_uuid is None


def test_refresh_failure_surfaces_status_and_body(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(60))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountAuthError, match="status 400.*invalid_grant"):
        _run(scenario())


def test_invalid_grant_raises_reauth_subtype_without_the_response_body(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(60))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": "invalid_grant", "error_description": "SECRET-BODY-DETAIL"}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountReauthRequiredError) as excinfo:
        _run(scenario())
    assert "SECRET-BODY-DETAIL" not in str(excinfo.value)


def test_non_invalid_grant_refresh_failure_raises_the_base_class_only(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(60))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountAuthError) as excinfo:
        _run(scenario())
    assert not isinstance(excinfo.value, ClaudeAccountReauthRequiredError)


def test_missing_refresh_token_raises_with_guidance(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "stale-token", _in_ms(60))
    file_data = json.loads((tmp_path / "credentials.json").read_text())
    del file_data["claudeAiOauth"]["refreshToken"]
    (tmp_path / "credentials.json").write_text(json.dumps(file_data), encoding="utf-8")

    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await ClaudeAccountAuthManager(tmp_path, http_client).get_credentials()

    with pytest.raises(ClaudeAccountAuthError, match="no refresh token"):
        _run(scenario())
