"""Tests for registered Claude account credential loading, refresh, and coalescing."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from claudex import paths
from claudex.claude import account_store, accounts as claude_accounts
from claudex.claude.account_profile import load_account_profile_fingerprint
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


@pytest.mark.parametrize("force_refresh", [False, True])
@pytest.mark.parametrize(
    "new_uuid",
    ["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"],
    ids=["same-uuid", "changed-uuid"],
)
def test_inflight_refresh_cannot_replace_reauthentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_uuid: str, force_refresh: bool
) -> None:
    monkeypatch.setattr(paths, "runtime_dir", lambda: tmp_path / "runtime")
    original_uuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    original = claude_accounts.add_account(
        "synthetic@example.com", None, None,
        {"claudeAiOauth": {"accessToken": "login-A", "refreshToken": "refresh-A", "expiresAt": _in_ms(-60)}},
        {"accountUuid": original_uuid},
    )
    account_dir = paths.accounts_dir("claude") / original.id
    original_fingerprint = load_account_profile_fingerprint(account_dir)
    new_payload = {"claudeAiOauth": {"accessToken": "login-B", "refreshToken": "refresh-B", "expiresAt": _in_ms(3600)}}

    async def scenario() -> Any:
        started, release = asyncio.Event(), asyncio.Event()
        posts = []

        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == CLAUDE_TOKEN_URL
            posts.append(json.loads(request.content)["refresh_token"])
            started.set()
            await release.wait()
            return httpx.Response(200, json={"access_token": "rotated-A", "refresh_token": "rotated-refresh-A", "expires_in": 900})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            manager = ClaudeAccountAuthManager(account_dir, client)
            pending = asyncio.create_task(manager.get_credentials(force_refresh=force_refresh))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                updated = await asyncio.wait_for(asyncio.to_thread(
                    claude_accounts.update_account_credentials,
                    "synthetic@example.com", None, None, new_payload, {"accountUuid": new_uuid},
                ), timeout=5)
                assert manager._refresh_lock.locked()
                observed = await manager.get_credentials()
                assert (observed.access_token, observed.account_uuid) == ("login-B", new_uuid)
                assert updated.id == original.id
                assert (updated.account_incarnation_id == original.account_incarnation_id) == (new_uuid == original_uuid)
                assert (load_account_profile_fingerprint(account_dir) == original_fingerprint) == (new_uuid == original_uuid)
            finally:
                release.set()
            returned = await pending
            assert posts == ["refresh-A"]
            return returned

    returned = _run(scenario())
    assert json.loads((account_dir / "credentials.json").read_text()) == new_payload
    assert (returned.access_token, returned.account_uuid) == ("login-B", new_uuid)
    [current] = claude_accounts.load_registry()
    assert current.id == original.id
    assert current.state == "ready"


def test_refresh_uses_private_unique_atomic_staging(tmp_path: Path) -> None:
    _write_account_dir(tmp_path, "login-A", _in_ms(-60))
    victim = tmp_path / "synthetic-victim.json"
    victim.write_text("synthetic sentinel", encoding="utf-8")
    (tmp_path / "credentials.json.tmp").symlink_to(victim)

    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == CLAUDE_TOKEN_URL
            return httpx.Response(200, json={"access_token": "rotated-A", "refresh_token": "refresh-2"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            returned = await ClaudeAccountAuthManager(tmp_path, client).get_credentials()
            assert returned.access_token == "rotated-A"

    _run(scenario())
    assert victim.read_text() == "synthetic sentinel"
    assert not (tmp_path / "credentials.json").is_symlink()
    assert stat.S_IMODE((tmp_path / "credentials.json").stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".credentials.json.tmp-*"))


@pytest.mark.parametrize("new_access_token", ["rotated-token", "shared-token"])
def test_event_controlled_forced_refreshes_coalesce_one_file_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_access_token: str
) -> None:
    _write_account_dir(tmp_path, "shared-token", _in_ms(3600))
    snapshot_read = threading.Event()

    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        posts = []

        async def handler(request: httpx.Request) -> httpx.Response:
            posts.append(json.loads(request.content)["refresh_token"])
            started.set()
            await release.wait()
            assert posts == ["refresh-1"]
            return httpx.Response(200, json={"access_token": new_access_token, "refresh_token": "refresh-2", "expires_in": 900})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            manager = ClaudeAccountAuthManager(tmp_path, client)
            real_load = manager._load_credentials_file
            reads = 0

            def observed_load() -> Any:
                nonlocal reads
                result = real_load()
                reads += 1
                if reads == 3:
                    snapshot_read.set()
                return result

            monkeypatch.setattr(manager, "_load_credentials_file", observed_load)
            first = asyncio.create_task(manager.get_credentials(force_refresh=True))
            await asyncio.wait_for(started.wait(), timeout=5)
            second = asyncio.create_task(manager.get_credentials(force_refresh=True))
            try:
                assert await asyncio.to_thread(snapshot_read.wait, 5)
            finally:
                release.set()
            results = await asyncio.gather(first, second)
            assert [result.access_token for result in results] == [new_access_token] * 2
            assert posts == ["refresh-1"]

    _run(scenario())
    assert json.loads((tmp_path / "credentials.json").read_text())["claudeAiOauth"]["refreshToken"] == "refresh-2"


@pytest.mark.parametrize("repeat_cancellation", [False, True])
@pytest.mark.parametrize("publication_fails", [False, True])
def test_cancelled_refresh_keeps_generation_owned_until_publication_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeat_cancellation: bool,
    publication_fails: bool,
) -> None:
    _write_account_dir(tmp_path, "login-A", _in_ms(3600))
    publication_started = threading.Event()
    release_publication = threading.Event()
    worker_finished = threading.Event()

    async def scenario() -> None:
        second_waiting, second_acquired = asyncio.Event(), asyncio.Event()
        duplicate_post = asyncio.Event()
        posts: list[str] = []

        class ObservedRefreshLock(asyncio.Lock):
            acquisitions = 0

            async def acquire(self) -> bool:
                self.acquisitions += 1
                is_second = self.acquisitions == 2
                if is_second:
                    second_waiting.set()
                result = await super().acquire()
                if is_second:
                    second_acquired.set()
                return result

        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == CLAUDE_TOKEN_URL
            posts.append(json.loads(request.content)["refresh_token"])
            if len(posts) > 1 and not release_publication.is_set():
                duplicate_post.set()
            return httpx.Response(200, json={
                "access_token": f"rotated-{len(posts)}",
                "refresh_token": f"refresh-{len(posts) + 1}",
                "expires_in": 900,
            })

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            manager = ClaudeAccountAuthManager(tmp_path, client)
            manager._refresh_lock = ObservedRefreshLock()
            real_publish = manager._publish_refresh
            publications = 0

            def paused_publish(*args: Any) -> Any:
                nonlocal publications
                publications += 1
                if publications != 1:
                    return real_publish(*args)
                try:
                    publication_started.set()
                    assert release_publication.wait(timeout=5)
                    if publication_fails:
                        raise ClaudeAccountAuthError("synthetic publication failure")
                    return real_publish(*args)
                finally:
                    worker_finished.set()

            monkeypatch.setattr(manager, "_publish_refresh", paused_publish)
            first = asyncio.create_task(manager.get_credentials(force_refresh=True))
            second = None
            try:
                assert await asyncio.to_thread(publication_started.wait, 5)
                assert posts == ["refresh-1"]
                first.cancel()
                second = asyncio.create_task(manager.get_credentials(force_refresh=True))
                await asyncio.wait_for(second_waiting.wait(), timeout=5)
                if repeat_cancellation:
                    first.cancel()
                    checkpoint = asyncio.Event()
                    asyncio.get_running_loop().call_soon(checkpoint.set)
                    await checkpoint.wait()
                if second_acquired.is_set():
                    await asyncio.wait_for(duplicate_post.wait(), timeout=5)
                assert posts == ["refresh-1"], "unresolved publication must own its single-use generation"
                assert not second_acquired.is_set()
                assert not worker_finished.is_set()
            finally:
                release_publication.set()
                callers = [first] + ([second] if second is not None else [])
                outcomes = await asyncio.wait_for(
                    asyncio.gather(*callers, return_exceptions=True), timeout=5
                )
                if publication_started.is_set():
                    assert await asyncio.to_thread(worker_finished.wait, 5)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert second is not None and second_acquired.is_set()
            assert outcomes[1].access_token == ("rotated-2" if publication_fails else "rotated-1")
            assert posts == (["refresh-1"] * 2 if publication_fails else ["refresh-1"])

    _run(scenario())


def test_refresh_http_cancellation_does_not_start_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_account_dir(tmp_path, "login-A", _in_ms(3600))
    original = (tmp_path / "credentials.json").read_bytes()

    async def scenario() -> None:
        started, release, http_cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == CLAUDE_TOKEN_URL
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                http_cancelled.set()
                raise
            raise AssertionError("cancelled HTTP must not produce a successful response")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as client:
            manager = ClaudeAccountAuthManager(tmp_path, client)

            def reject_publication(*args: Any) -> Any:
                raise AssertionError("cancelled HTTP must not start publication")

            monkeypatch.setattr(manager, "_publish_refresh", reject_publication)
            pending = asyncio.create_task(manager.get_credentials(force_refresh=True))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pending, timeout=5)
                assert http_cancelled.is_set()
                assert not manager._refresh_lock.locked()
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)

    _run(scenario())
    assert (tmp_path / "credentials.json").read_bytes() == original


@pytest.mark.parametrize("failed_filename", ["oauth-account.json", "registry.json"])
def test_inflight_refresh_preserves_partially_published_reauthentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_filename: str
) -> None:
    monkeypatch.setattr(paths, "runtime_dir", lambda: tmp_path / "runtime")
    profile = {"accountUuid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
    original = claude_accounts.add_account(
        "synthetic@example.com", None, None,
        {"claudeAiOauth": {"accessToken": "login-A", "refreshToken": "refresh-A", "expiresAt": 1}}, profile,
    )
    account_dir = paths.accounts_dir("claude") / original.id
    new_payload = {"claudeAiOauth": {"accessToken": "login-B", "refreshToken": "refresh-B"}}
    real_write = account_store._write_json_atomic

    def fail_later_publication(path: Path, data: Any) -> None:
        if path.name == failed_filename:
            raise OSError("synthetic later publication failure")
        real_write(path, data)

    monkeypatch.setattr(account_store, "_write_json_atomic", fail_later_publication)

    async def scenario() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["refresh_token"] == "refresh-A"
            started.set()
            await release.wait()
            return httpx.Response(200, json={"access_token": "rotated-A", "refresh_token": "rotated-refresh-A"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            pending = asyncio.create_task(ClaudeAccountAuthManager(account_dir, client).get_credentials())
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                with pytest.raises(claude_accounts.AccountRegistryError):
                    await asyncio.to_thread(claude_accounts.update_account_credentials,
                        "synthetic@example.com", None, None, new_payload, profile)
                assert json.loads((account_dir / "credentials.json").read_text()) == new_payload
            finally:
                release.set()
            returned = await pending
            assert (returned.access_token, returned.account_uuid) == ("login-B", profile["accountUuid"])

    _run(scenario())
    assert json.loads((account_dir / "credentials.json").read_text()) == new_payload


@pytest.mark.parametrize("failure_boundary", ["before-replace", "after-replace"])
def test_refresh_publication_failure_never_rolls_back_committed_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_boundary: str
) -> None:
    _write_account_dir(tmp_path, "login-A", _in_ms(-60))

    def fail(*args: Any) -> None:
        raise OSError("synthetic publication failure")

    if failure_boundary == "before-replace":
        monkeypatch.setattr(os, "replace", fail)
    else:
        monkeypatch.setattr(account_store, "_fsync_directory", fail)

    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"access_token": "rotated-A", "refresh_token": "refresh-2"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            with pytest.raises(ClaudeAccountAuthError, match="persist|durability"):
                await ClaudeAccountAuthManager(tmp_path, client).get_credentials()

    _run(scenario())
    persisted = json.loads((tmp_path / "credentials.json").read_text())["claudeAiOauth"]
    assert persisted["accessToken"] == ("login-A" if failure_boundary == "before-replace" else "rotated-A")
    assert persisted["refreshToken"] == ("refresh-1" if failure_boundary == "before-replace" else "refresh-2")
    assert not list(tmp_path.glob(".credentials.json.tmp-*"))
