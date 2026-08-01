"""Tests for Codex credential loading and refresh coalescing."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex_gateway.codex_auth import OAUTH_TOKEN_URL, CodexAuthError, CodexAuthManager


def _fake_jwt(exp: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def _write_auth_file(path: Path, access_token: str) -> None:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-1",
                    "account_id": "acct-1",
                }
            }
        ),
        encoding="utf-8",
    )


def _refresh_transport(counter: dict[str, int], new_access_token: str) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == OAUTH_TOKEN_URL
        counter["posts"] += 1
        # Suspend while holding the refresh lock so a concurrent caller
        # genuinely waits on it instead of racing past.
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={"access_token": new_access_token, "refresh_token": "refresh-2"},
        )

    return httpx.MockTransport(handler)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_concurrent_forced_refreshes_rotate_only_once(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, "stale-token")
    counter = {"posts": 0}

    async def scenario() -> list[Any]:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = CodexAuthManager(auth_file, http_client)
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
    persisted = json.loads(auth_file.read_text(encoding="utf-8"))
    assert persisted["tokens"]["access_token"] == "rotated-token"
    assert persisted["tokens"]["refresh_token"] == "refresh-2"


def test_sequential_forced_refreshes_each_rotate(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, "stale-token")
    counter = {"posts": 0}

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            manager = CodexAuthManager(auth_file, http_client)
            await manager.get_credentials(force_refresh=True)
            # A later 401 against the rotated token is a new stale generation.
            await manager.get_credentials(force_refresh=True)

    _run(scenario())

    assert counter["posts"] == 2


def test_expired_jwt_is_refreshed_without_force(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    _write_auth_file(auth_file, _fake_jwt(time.time() - 60))
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "rotated-token")
        ) as http_client:
            return await CodexAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 1
    assert credentials.access_token == "rotated-token"
    assert credentials.account_id == "acct-1"


def test_valid_jwt_is_returned_without_refresh(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    valid_token = _fake_jwt(time.time() + 3600)
    _write_auth_file(auth_file, valid_token)
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await CodexAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.access_token == valid_token


def test_api_key_auth_bypasses_refresh(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"OPENAI_API_KEY": "sk-test"}), encoding="utf-8")
    counter = {"posts": 0}

    async def scenario() -> Any:
        async with httpx.AsyncClient(
            transport=_refresh_transport(counter, "unused")
        ) as http_client:
            return await CodexAuthManager(auth_file, http_client).get_credentials()

    credentials = _run(scenario())

    assert counter["posts"] == 0
    assert credentials.is_api_key is True
    assert credentials.access_token == "sk-test"


def test_missing_auth_file_raises_with_guidance(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as http_client:
            await CodexAuthManager(tmp_path / "auth.json", http_client).get_credentials()

    with pytest.raises(CodexAuthError, match="codex login"):
        _run(scenario())
