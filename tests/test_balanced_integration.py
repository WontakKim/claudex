"""End-to-end integration tests for the balanced router's restart/drain/
corruption gates (T-17): real temporary `ClaudePoolRuntimeStateStore` SQLite
files throughout, `TestClient` for the scenarios that need the full HTTP
dispatch stack, and a deterministic fake clock (`_FakeClock`, mirroring
`test_balanced_router.py`'s own helper) wherever precise
freshness/TTL timing matters.

Every scenario here is locally reproducible with exactly ONE registered
account -- unlike the deferred gates in `test_balanced_deferred_gates.py`,
which need a second live account to observe a real cross-account race or
failover.

Reservation-aware LRU eviction and restored-cooldown no-repeat-burst
behavior are intentionally NOT duplicated in this file: both are already
exercised end-to-end by earlier task tests --
`test_balanced_router.py::test_migration_reserved_pin_survives_eviction_pressure_and_becomes_evictable_once_resolved`
(plus its sibling `test_lru_eviction_*` tests) for reservation-aware LRU,
and
`test_server.py::test_balanced_restart_restores_the_family_cooldown_without_a_repeat_429_burst`
for the restored-cooldown case.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from claudex_gateway import paths
from claudex_gateway.claude import accounts as claude_accounts
from claudex_gateway.claude_ambient_account import AmbientClaudeAuthManager
from claudex_gateway.balanced.router import CAPABILITY_CLASSIFIER_VERSION, ClaudeBalancedRouter
from claudex_gateway.balanced.runtime import ClaudeBalancedRuntime
from claudex_gateway.balanced.selection import AccountCandidate, SessionKey, derive_session_key
from claudex_gateway.balanced.state_model import SCHEMA_VERSION, RestoreValidationContext
from claudex_gateway.balanced.state_store import ClaudePoolRuntimeStateStore
from claudex_gateway.config import GatewayConfig
from claudex_gateway.providers.grok_auth import GrokCredentials
from claudex_gateway.providers.kimi_auth import KimiCredentials

import claudex_gateway.admin.settings as admin_settings
import claudex_gateway.relay.balanced as relay_balanced
import claudex_gateway.server as server
import claudex_gateway.server_support as server_support

# ---------------------------------------------------------------------------
# Fakes/fixtures mirroring test_server.py's `_create_test_client` exactly --
# this file is deliberately self-contained (no cross-test-file imports, per
# this codebase's convention).
# ---------------------------------------------------------------------------


class _AvailableCodexAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            is_api_key=False, account_id="account", access_token="codex", email="codex@example.com"
        )


class _FakeCodexClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None


class _AvailableKimiAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        return KimiCredentials(access_token="kimi-token", device_id="device-1", account="kimi-user-1")


class _FakeKimiClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class _AvailableGrokAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        return GrokCredentials(access_token="grok-token", email="user@example.com")


class _FakeGrokClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None


def _create_test_client(
    monkeypatch: pytest.MonkeyPatch, *, config: GatewayConfig | None = None, base_url: str = "http://testserver"
) -> TestClient:
    monkeypatch.setattr(server, "CodexAuthManager", _AvailableCodexAuthManager)
    monkeypatch.setattr(server, "CodexClient", _FakeCodexClient)
    monkeypatch.setattr(server, "KimiAuthManager", _AvailableKimiAuthManager)
    monkeypatch.setattr(server, "KimiClient", _FakeKimiClient)
    monkeypatch.setattr(server, "GrokAuthManager", _AvailableGrokAuthManager)
    monkeypatch.setattr(server, "GrokClient", _FakeGrokClient)
    return TestClient(server.create_app(config or GatewayConfig()), base_url=base_url)


_BALANCED_ACCOUNT_UUID = "11111111-2222-3333-4444-555555555555"


def _balanced_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate HOME under `tmp_path` and clear the env-lock override, exactly
    like `test_server.py`'s own helper -- registry/pool state then lives
    entirely under this test's own `.claudex` directory.
    """
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _register_balanced_ready_account(
    *, email: str = "balanced@example.com", account_uuid: str | None = _BALANCED_ACCOUNT_UUID
) -> str:
    """Register one ready account carrying a valid T-3 profile fingerprint,
    so it can activate balanced routing."""
    oauth_account: dict[str, Any] = {"emailAddress": email}
    if account_uuid is not None:
        oauth_account["accountUuid"] = account_uuid
    record = claude_accounts.add_account(
        email=email,
        organization_uuid="org-1",
        organization_name="Example Org",
        credentials_json={
            "claudeAiOauth": {
                "accessToken": "balanced-access-1",
                "refreshToken": "balanced-refresh-1",
                "expiresAt": (time.time() + 3600) * 1000,
                "scopes": ["user:inference", "user:profile"],
            }
        },
        oauth_account_json=oauth_account,
    )
    return record.id


def _write_ambient_login(
    tmp_path: Path,
    *,
    email: str = "ambient@example.com",
    organization_uuid: str = "org-ambient",
    expires_at: float | None = None,
) -> None:
    home = tmp_path / "home"
    claude_home = home / ".claude"
    claude_home.mkdir(parents=True, exist_ok=True)
    account_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    (home / ".claude.json").write_text(
        json.dumps(
            {
                "oauthAccount": {
                    "emailAddress": email,
                    "organizationUuid": organization_uuid,
                    "accountUuid": account_uuid,
                    "organizationType": "claude_max",
                    "organizationRateLimitTier": "default_claude_max_5x",
                }
            }
        ),
        encoding="utf-8",
    )
    (claude_home / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "ambient-access-1",
                    "expiresAt": (
                        (time.time() + 3600) * 1000
                        if expires_at is None
                        else expires_at
                    ),
                }
            }
        ),
        encoding="utf-8",
    )


def _capture_balanced_candidate_ids(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> set[str]:
    captured: set[str] = set()

    async def capture_records(
        _request: Any,
        _raw_body: bytes,
        _parsed_body: Any,
        _runtime: ClaudeBalancedRuntime,
        records_by_id: dict[str, Any],
        _session_key: SessionKey,
        _model: str,
    ) -> Any:
        captured.update(records_by_id)
        return JSONResponse({"captured": True})

    monkeypatch.setattr(
        relay_balanced, "_serve_balanced_pinned_message", capture_records
    )
    response = client.post("/v1/messages", json=_balanced_body(str(uuid.uuid4())))
    assert response.status_code == 200
    return captured


def _message_body(model: str) -> dict[str, Any]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}


def _balanced_body(session_id: str, *, model: str = "claude-sonnet-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {
        "user_id": json.dumps(
            {"device_id": "d" * 64, "account_uuid": "client-account-uuid", "session_id": session_id},
            separators=(",", ":"),
        )
    }
    return body


def _enable_balanced(client: TestClient, handler: Any) -> ClaudeBalancedRuntime:
    routing_put = next(
        route
        for route in client.app.routes
        if route.path == "/admin/providers/claude/pool/routing"
        and route.methods == {"PUT"}
    )
    assert routing_put.endpoint is admin_settings._handle_admin_claude_routing_put
    client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
    assert response.status_code == 200, response.text
    return client.app.state.claude_balanced_runtime


class _FakeClock:
    """A manually advanced monotonic-like clock for deterministic freshness/TTL
    assertions -- mirrors `test_balanced_router.py`'s own helper."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


# ===========================================================================
# Ambient local login membership
# ===========================================================================


def test_distinct_ambient_login_joins_balanced_candidates_and_serves_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    registered_id = _register_balanced_ready_account()
    _write_ambient_login(tmp_path)
    message_authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/oauth/usage":
            return httpx.Response(200, json={})
        message_authorizations.append(request.headers["authorization"])
        return httpx.Response(200, json={"id": "msg_ambient"})

    with _create_test_client(monkeypatch, base_url="http://127.0.0.1:8787") as client:
        runtime = _enable_balanced(client, handler)
        provider = client.app.state.claude_ambient_accounts
        member = provider.pool_member()
        assert member is not None

        records = claude_accounts.load_registry()
        records_by_id = {record.id: record for record in records}
        records_by_id[member.record.id] = member.record
        candidates = relay_balanced._balanced_candidates(
            records_by_id.values(), runtime.router, family="default", now=time.monotonic()
        )
        assert {candidate.account_id for candidate in candidates} == {
            registered_id,
            member.record.id,
        }

        for _ in range(100):
            body = _balanced_body(str(uuid.uuid4()))
            session_key = derive_session_key(body, runtime.epoch_seed, "default")
            selected_id = relay_balanced._balanced_pick_account(
                runtime.router,
                session_key_digest=session_key.scoring_digest_or_default,
                model=body["model"],
                candidates=candidates,
                seed=runtime.epoch_seed,
            )
            if selected_id == member.record.id:
                break
        else:
            pytest.fail("could not derive a session routed to the ambient account")

        response = client.post("/v1/messages", json=body)
        assert response.status_code == 200
        assert message_authorizations == ["Bearer ambient-access-1"]
        pin = runtime.router.get_pin(session_key.digest)
        assert pin is not None and pin.account_id == member.record.id


def test_registered_duplicate_identity_suppresses_ambient_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    registered_id = _register_balanced_ready_account(email="same@example.com")
    _write_ambient_login(
        tmp_path, email="same@example.com", organization_uuid="org-1"
    )

    with _create_test_client(
        monkeypatch, base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, lambda _request: httpx.Response(200, json={}))
        provider = client.app.state.claude_ambient_accounts
        member = provider.pool_member()
        assert member is not None

        candidate_ids = _capture_balanced_candidate_ids(client, monkeypatch)
        assert candidate_ids == {registered_id}
        assert member.record.id not in candidate_ids


def test_include_local_login_false_suppresses_ambient_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    registered_id = _register_balanced_ready_account()
    _write_ambient_login(tmp_path)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "claude_account.routing": {
                    "mode": "disabled",
                    "include_local_login": False,
                }
            }
        ),
        encoding="utf-8",
    )

    with _create_test_client(
        monkeypatch,
        config=GatewayConfig.load(settings_file),
        base_url="http://127.0.0.1:8787",
    ) as client:
        assert client.app.state.claude_ambient_accounts is None
        _enable_balanced(client, lambda _request: httpx.Response(200, json={}))
        candidate_ids = _capture_balanced_candidate_ids(client, monkeypatch)
        assert candidate_ids == {registered_id}


def test_stale_ambient_credentials_do_not_create_balanced_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    registered_id = _register_balanced_ready_account()
    _write_ambient_login(tmp_path, expires_at=(time.time() - 60) * 1000)

    with _create_test_client(
        monkeypatch, base_url="http://127.0.0.1:8787"
    ) as client:
        _enable_balanced(client, lambda _request: httpx.Response(200, json={}))
        provider = client.app.state.claude_ambient_accounts
        assert provider.pool_member() is None
        assert _capture_balanced_candidate_ids(client, monkeypatch) == {registered_id}


def test_auth_manager_routes_ambient_id_without_caching_directory_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _balanced_env(monkeypatch, tmp_path)
    _write_ambient_login(tmp_path)
    app = server.create_app(GatewayConfig())
    provider = app.state.claude_ambient_accounts
    member = provider.pool_member()
    assert member is not None

    manager = server_support._claude_account_auth_manager(app.state, member.record.id)

    assert isinstance(manager, AmbientClaudeAuthManager)
    assert app.state.claude_account_auth_managers == {}


# ===========================================================================
# Restart continuity through the full HTTP stack (TestClient)
# ===========================================================================


def test_graceful_restart_preserves_the_same_session_pin_across_two_app_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Design v2 §5.2: a session pinned before a clean process shutdown
    (`shutdown_preserving_epoch`) is FOLLOWED -- never re-placed -- by a
    second, independent app instance restoring the same epoch/pin from the
    on-disk store, and keeps being served without creating a second pin.
    """
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_ready_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    settings_file = tmp_path / "settings.json"
    body = _balanced_body(str(uuid.uuid4()))

    with _create_test_client(
        monkeypatch, config=GatewayConfig(settings_file=settings_file), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        response = client.post("/v1/messages", json=body)
        assert response.status_code == 200
        session_key = derive_session_key(body, runtime.epoch_seed, "default")
        pinned_account_id = runtime.router.get_pin(session_key.digest).account_id
        epoch_id_1 = runtime.epoch_id

    second_config = GatewayConfig.load(settings_file)
    with _create_test_client(
        monkeypatch, config=second_config, base_url="http://127.0.0.1:8787"
    ) as client:
        runtime2 = client.app.state.claude_balanced_runtime
        assert runtime2.epoch_id == epoch_id_1
        session_key2 = derive_session_key(body, runtime2.epoch_seed, "default")
        pin = runtime2.router.get_pin(session_key2.digest)
        assert pin is not None
        assert pin.account_id == pinned_account_id

        client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = client.post("/v1/messages", json=body)
        assert response.status_code == 200
        # Followed, not re-placed: still exactly one pin.
        assert runtime2.router.pin_count() == 1


def test_mode_exit_rotates_the_epoch_so_a_later_reentry_never_restores_the_old_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Design v2 Step 4/9: an INTENTIONAL `balanced` -> `disabled` exit
    rotates the epoch (a fresh seed and a wiped pin map), unlike a graceful
    process shutdown -- a later re-entry starts a brand new epoch whose seed
    makes even the identical request body hash to a different digest.
    """
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_ready_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    with _create_test_client(
        monkeypatch, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        old_epoch_id = runtime.epoch_id
        body = _balanced_body(str(uuid.uuid4()))
        response = client.post("/v1/messages", json=body)
        assert response.status_code == 200
        old_session_key = derive_session_key(body, runtime.epoch_seed, "default")
        assert runtime.router.get_pin(old_session_key.digest) is not None

        exited = client.put("/admin/providers/claude/pool/routing", json={"mode": "disabled"})
        assert exited.status_code == 200

        reentered = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
        assert reentered.status_code == 200
        new_runtime = client.app.state.claude_balanced_runtime
        assert new_runtime.epoch_id != old_epoch_id
        assert new_runtime.router.pin_count() == 0

        new_session_key = derive_session_key(body, new_runtime.epoch_seed, "default")
        assert new_session_key.digest != old_session_key.digest
        assert new_runtime.router.get_pin(new_session_key.digest) is None


def test_newer_schema_runtime_db_is_refused_at_enable_time_and_left_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Design v2 §5.5: a runtime-state database whose `schema_version` is
    newer than this build supports is refused (never touched) even when
    reached through the full enable path -- balanced routing simply fails to
    activate (a graceful 500, not a crash), and the file's bytes never
    change.
    """
    _balanced_env(monkeypatch, tmp_path)

    runtime_db_path = paths.claude_account_pool_runtime_db()
    seed_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
    seed_store.close()
    conn = sqlite3.connect(str(runtime_db_path))
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()
    before = runtime_db_path.read_bytes()

    with _create_test_client(
        monkeypatch, config=GatewayConfig(settings_file=tmp_path / "settings.json"), base_url="http://127.0.0.1:8787"
    ) as client:
        response = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
        assert response.status_code == 500
        assert "could not enable balanced routing" in response.json()["error"]["message"]
        assert client.app.state.claude_balanced_runtime.status == "disabled"
        mode_after = client.get("/admin/providers/claude/pool/routing").json()
        assert mode_after == {"mode": "disabled", "env_locked": False}

    after = runtime_db_path.read_bytes()
    assert before == after


def test_abrupt_crash_analog_leaves_the_durably_committed_pin_intact_without_graceful_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The full HTTP stack's own durability guarantee, proven without ever
    invoking the clean-shutdown path: `synchronous=FULL` WAL commits are
    durable on disk the instant they complete, independent of a graceful
    `close()` -- a process that dies before its shutdown handler ever runs
    (kill -9, OOM, ...) still leaves every already-committed pin recoverable.
    """
    _balanced_env(monkeypatch, tmp_path)
    _register_balanced_ready_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    client = _create_test_client(
        monkeypatch, config=GatewayConfig(settings_file=tmp_path / "settings.json"), base_url="http://127.0.0.1:8787"
    )
    client.__enter__()  # deliberately never __exit__()'d: this IS the crash
    runtime = _enable_balanced(client, handler)
    body = _balanced_body(str(uuid.uuid4()))
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200
    session_key = derive_session_key(body, runtime.epoch_seed, "default")
    pinned_account_id = runtime.router.get_pin(session_key.digest).account_id

    # No `shutdown_preserving_epoch`, no WAL checkpoint, no released pool
    # lease -- the process is gone. Inspect the on-disk store directly, the
    # way a freshly restarted process would.
    inspect_store = ClaudePoolRuntimeStateStore.open_(paths.claude_account_pool_runtime_db())
    try:
        restore_result = inspect_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert session_key.digest in restore_result.pins
        assert restore_result.pins[session_key.digest].account_id == pinned_account_id
    finally:
        inspect_store.close()


# ===========================================================================
# Crash-window disposition: the durability barrier's exact commit boundary
# (real temporary store, direct router+store construction for precise
# timing control)
# ===========================================================================


def test_pre_commit_crash_analog_loses_the_never_persisted_pin_and_a_later_request_re_places_cleanly(
    tmp_path: Path,
) -> None:
    """The crash lands BEFORE `submit_new_pin_durability` ever runs -- e.g.
    the process dies in the gap between `place_session`'s in-memory insert
    and its caller ever scheduling the background durability task. The pin
    was never written to disk, so a restart correctly has no memory of it;
    the session safely gets a fresh placement instead of a dangling
    reference to a lost generation.
    """
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    clock = _FakeClock()
    router = ClaudeBalancedRouter(balanced_epoch_id=store.balanced_epoch_id, store=store, clock=clock)
    session_key = SessionKey(digest=b"\x33" * 32, kind="content_hash")
    candidates = [AccountCandidate(account_id="acct-a", account_incarnation_id="inc-a")]

    placement = router.place_session(
        session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
    )
    assert placement.created
    assert placement.durability_barrier is not None and not placement.durability_barrier.is_resolved
    # The crash happens right here -- `submit_new_pin_durability` is never
    # invoked at all.

    fresh_store = ClaudePoolRuntimeStateStore.open_(db_path)
    try:
        restore_result = fresh_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert session_key.digest not in restore_result.pins

        fresh_router = ClaudeBalancedRouter(balanced_epoch_id=fresh_store.balanced_epoch_id, store=fresh_store)
        fresh_router.restore_from_store(restore_result)
        assert fresh_router.pin_count() == 0

        replacement = fresh_router.place_session(
            session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
        )
        assert replacement.created
        assert replacement.generation == 0
    finally:
        fresh_store.close()


def test_post_commit_crash_analog_retains_the_pin_that_finished_committing_before_the_crash(
    tmp_path: Path,
) -> None:
    """The crash lands strictly AFTER `submit_new_pin_durability`'s durable
    write has been awaited to completion (§5.3's barrier already resolved
    with success) -- the row is on disk, so a restart recovers it exactly.
    """
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    clock = _FakeClock()
    router = ClaudeBalancedRouter(balanced_epoch_id=store.balanced_epoch_id, store=store, clock=clock)
    session_key = SessionKey(digest=b"\x22" * 32, kind="content_hash")
    candidates = [AccountCandidate(account_id="acct-a", account_incarnation_id="inc-a")]

    placement = router.place_session(
        session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
    )
    asyncio.run(router.submit_new_pin_durability(session_key.digest))
    assert placement.durability_barrier is not None and placement.durability_barrier.is_resolved
    assert router.persistence_degraded is False
    # The crash happens right here -- after the barrier resolved, before any
    # graceful `store.close()`.

    fresh_store = ClaudePoolRuntimeStateStore.open_(db_path)
    try:
        restore_result = fresh_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert session_key.digest in restore_result.pins
        assert restore_result.pins[session_key.digest].account_id == "acct-a"
    finally:
        fresh_store.close()


# ===========================================================================
# Restored observation bucketing: fresh / stale / unknown (store-level
# restore, since neither the router's live `ObservationView` nor the
# in-memory-only `ClaudeAccountUsageCache` has a restore path for
# `usage_observations` -- that table's only restore-time consumer is the
# store's own freshness classification exercised below)
# ===========================================================================


def test_restored_fresh_usage_observation_is_kept_after_a_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    now_utc = time.time()
    store.upsert_usage_observation(
        account_id="acct-a",
        window="five_hour",
        account_incarnation_id="inc-a",
        account_profile_fingerprint="fp-a",
        used_percent=37.5,
        reset_identity="reset-1",
        reset_at_utc=now_utc + 3600,
        observed_at_utc=now_utc - 30.0,
        source="usage_api",
    ).wait(timeout=5.0)

    restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
    restored = restore_result.usage_observations[("acct-a", "five_hour")]
    assert restored.used_percent == pytest.approx(37.5)
    assert restore_result.skip_counts.get("usage_observations.stale_reset", 0) == 0
    store.close()

    fresh_store = ClaudePoolRuntimeStateStore.open_(db_path)
    try:
        assert fresh_store.get_usage_observation("acct-a", "five_hour") is not None
    finally:
        fresh_store.close()


def test_restored_stale_usage_observation_is_deleted_and_skipped_after_a_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    now_utc = time.time()
    store.upsert_usage_observation(
        account_id="acct-a",
        window="seven_day",
        account_incarnation_id="inc-a",
        account_profile_fingerprint="fp-a",
        used_percent=88.0,
        reset_identity="reset-2",
        reset_at_utc=now_utc - 5.0,  # this window's reset already passed
        observed_at_utc=now_utc - 3600.0,
        source="usage_api",
    ).wait(timeout=5.0)

    restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
    assert ("acct-a", "seven_day") not in restore_result.usage_observations
    assert restore_result.skip_counts.get("usage_observations.stale_reset") == 1
    store.close()

    fresh_store = ClaudePoolRuntimeStateStore.open_(db_path)
    try:
        assert fresh_store.get_usage_observation("acct-a", "seven_day") is None
    finally:
        fresh_store.close()


def test_restored_unknown_usage_observation_has_no_row_to_restore_at_all(tmp_path: Path) -> None:
    """An account/window pair the store has never observed restores as
    nothing at all -- UNKNOWN by the total absence of a row, not by any
    special sentinel value, and never blocks the OTHER windows of the same
    account from restoring normally.
    """
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    now_utc = time.time()
    store.upsert_usage_observation(
        account_id="acct-a",
        window="five_hour",
        account_incarnation_id="inc-a",
        account_profile_fingerprint="fp-a",
        used_percent=10.0,
        reset_identity="reset-1",
        reset_at_utc=now_utc + 3600,
        observed_at_utc=now_utc,
        source="usage_api",
    ).wait(timeout=5.0)

    try:
        restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
        assert ("acct-a", "five_hour") in restore_result.usage_observations
        assert ("acct-a", "seven_day") not in restore_result.usage_observations
        assert store.get_usage_observation("acct-a", "seven_day") is None
    finally:
        store.close()


# ===========================================================================
# Incarnation vs. fingerprint invalidation (restored capability evidence)
# ===========================================================================


def test_restored_capability_evidence_is_invalidated_by_a_changed_incarnation_id(tmp_path: Path) -> None:
    """§5.5's restore backstop is enforced LAZILY, at query time, against the
    live candidate the caller supplies (same pattern pins rely on): a
    capability row restored from disk for an old incarnation must never be
    trusted for a NEW incarnation of the same `account_id` -- e.g. the
    account was removed and re-added while the process was down.
    """
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    now_utc = time.time()
    store.upsert_capability_evidence(
        account_id="acct-a",
        capability_key="opus",
        account_incarnation_id="inc-old",
        account_profile_fingerprint="fp-a",
        state="eligible",
        evidence_source="probe",
        classifier_version=CAPABILITY_CLASSIFIER_VERSION,
        observed_at_utc=now_utc,
        expires_at_utc=now_utc + 3600,
    ).wait(timeout=5.0)

    try:
        restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
        clock = _FakeClock(1_000.0)
        router = ClaudeBalancedRouter(balanced_epoch_id=store.balanced_epoch_id, store=store, clock=clock)
        router.restore_from_store(restore_result, now=clock.value, wall_now=now_utc)

        assert router.is_capability_eligible(
            "acct-a", "opus", account_incarnation_id="inc-old", account_profile_fingerprint="fp-a"
        )
        assert not router.is_capability_eligible(
            "acct-a", "opus", account_incarnation_id="inc-new", account_profile_fingerprint="fp-a"
        )
    finally:
        store.close()


def test_restored_capability_evidence_is_invalidated_by_a_changed_profile_fingerprint(tmp_path: Path) -> None:
    """Same backstop, the other axis: a restored row's `account_incarnation_id`
    can still match while its `account_profile_fingerprint` (plan/seat/org
    tier) has changed -- e.g. a plan change while the process was down --
    and that alone invalidates the restored evidence too.
    """
    db_path = tmp_path / "runtime.sqlite3"
    store = ClaudePoolRuntimeStateStore.open_(db_path, debounce_seconds=0.0)
    now_utc = time.time()
    store.upsert_capability_evidence(
        account_id="acct-a",
        capability_key="fable",
        account_incarnation_id="inc-a",
        account_profile_fingerprint="fp-old",
        state="eligible",
        evidence_source="probe",
        classifier_version=CAPABILITY_CLASSIFIER_VERSION,
        observed_at_utc=now_utc,
        expires_at_utc=now_utc + 3600,
    ).wait(timeout=5.0)

    try:
        restore_result = store.restore(RestoreValidationContext(now_utc=now_utc))
        clock = _FakeClock(1_000.0)
        router = ClaudeBalancedRouter(balanced_epoch_id=store.balanced_epoch_id, store=store, clock=clock)
        router.restore_from_store(restore_result, now=clock.value, wall_now=now_utc)

        assert router.is_capability_eligible(
            "acct-a", "fable", account_incarnation_id="inc-a", account_profile_fingerprint="fp-old"
        )
        assert not router.is_capability_eligible(
            "acct-a", "fable", account_incarnation_id="inc-a", account_profile_fingerprint="fp-new"
        )
    finally:
        store.close()


# ===========================================================================
# Persistence degradation / recovery (real temporary store, controlled
# transient failures via the store's own `fault_injector` test seam)
# ===========================================================================


def test_persistence_degrades_under_transient_failures_but_recovers_and_durably_lands_the_pin(
    tmp_path: Path,
) -> None:
    """Combines the store's own transient-failure retry/backoff with the
    router's pin-durability flow: while the store is degraded, the
    in-memory pin already serves the session (a request never blocks past
    the barrier resolving); once the transient failures stop, the retry
    succeeds on its own, `persistence_degraded` clears, and a restart-
    simulating reopen finds the pin durably on disk -- self-healing with no
    operator intervention.
    """
    db_path = tmp_path / "runtime.sqlite3"
    fault_state = {"remaining_failures": 2}

    def fault_injector(_scope_key: Any, _payload_sequence: int) -> None:
        if fault_state["remaining_failures"] > 0:
            fault_state["remaining_failures"] -= 1
            raise sqlite3.OperationalError("simulated transient disk failure")

    store = ClaudePoolRuntimeStateStore.open_(
        db_path,
        debounce_seconds=0.0,
        retry_backoff_initial_seconds=0.02,
        retry_backoff_max_seconds=0.05,
        fault_injector=fault_injector,
    )
    router = ClaudeBalancedRouter(balanced_epoch_id=store.balanced_epoch_id, store=store)
    session_key = SessionKey(digest=b"\x44" * 32, kind="content_hash")
    candidates = [AccountCandidate(account_id="acct-a", account_incarnation_id="inc-a")]

    placement = router.place_session(
        session_key=session_key, model="claude-sonnet-5", candidates=candidates, seed=b"seed"
    )
    assert placement.created

    async def scenario() -> None:
        durability_task = asyncio.create_task(router.submit_new_pin_durability(session_key.digest))

        deadline = time.monotonic() + 5.0
        saw_degraded = False
        while time.monotonic() < deadline and not durability_task.done():
            if store.persistence_degraded:
                saw_degraded = True
                break
            await asyncio.sleep(0.005)
        assert saw_degraded, "expected to observe persistence_degraded=True during the transient retries"

        # The in-memory pin already serves the session while the durable
        # write is still retrying in the background.
        pin = router.get_pin(session_key.digest)
        assert pin is not None and pin.account_id == "acct-a"

        await asyncio.wait_for(durability_task, timeout=5.0)

    asyncio.run(scenario())

    assert store.persistence_degraded is False
    assert router.persistence_degraded is False  # no hard (IntegrityError) failure ever occurred
    store.close()

    fresh_store = ClaudePoolRuntimeStateStore.open_(db_path)
    try:
        restore_result = fresh_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert session_key.digest in restore_result.pins
        assert restore_result.pins[session_key.digest].account_id == "acct-a"
    finally:
        fresh_store.close()


# ===========================================================================
# Disable-while-streaming drain behavior (direct `ClaudeBalancedRuntime`
# construction against a real temporary store, mirroring
# test_server.py::test_balanced_request_during_controlled_exit_awaits_and_serves_under_target_mode)
# ===========================================================================


def test_disable_while_streaming_drains_the_in_flight_stream_before_epoch_rotation_invalidates_pins(
    tmp_path: Path,
) -> None:
    """`ClaudeBalancedRuntime.exit_mode`'s documented sequence (Step 4/9):
    mark draining, drain in-flight requests, THEN rotate the epoch, THEN
    publish. An admin `disabled` transition arriving while an SSE stream is
    still relaying bytes (`begin_request` still holding a slot open) must
    not invalidate that stream's pin or rotate the epoch until the stream
    finishes.
    """
    from claudex_gateway.claude.accounts import AccountRecord

    account_id = "22222222-3333-4444-5555-666666666666"
    accounts_root = tmp_path / "accounts"
    (accounts_root / account_id).mkdir(parents=True)
    (accounts_root / account_id / "oauth-account.json").write_text(
        json.dumps({"accountUuid": _BALANCED_ACCOUNT_UUID}), encoding="utf-8"
    )
    account = AccountRecord(
        id=account_id,
        email="balanced@example.com",
        organization_uuid=None,
        organization_name=None,
        created_at=0,
        updated_at=0,
        last_authenticated_at=0,
        state="ready",
        account_incarnation_id="incarnation-1",
        upstream_account_uuid=_BALANCED_ACCOUNT_UUID,
    )
    runtime_db_path = tmp_path / "runtime.sqlite3"
    published = {"mode": "disabled"}

    async def scenario() -> None:
        runtime = ClaudeBalancedRuntime()
        await runtime.prepare_and_publish(
            accounts=[account],
            accounts_root=accounts_root,
            runtime_db_path=runtime_db_path,
            persist=lambda: published.__setitem__("mode", "balanced"),
            entry="admin_enable",
        )
        assert runtime.router is not None
        session_key = SessionKey(digest=b"\x55" * 32, kind="content_hash")
        runtime.router.place_session(
            session_key=session_key,
            model="claude-sonnet-5",
            candidates=[AccountCandidate(account_id=account_id, account_incarnation_id="incarnation-1")],
            seed=runtime.epoch_seed,
        )
        await runtime.router.submit_new_pin_durability(session_key.digest)
        pre_exit_epoch_id = runtime.epoch_id

        assert runtime.begin_request()  # a slot held open, as if an SSE stream is still relaying bytes

        exit_task = asyncio.create_task(
            runtime.exit_mode("disabled", publish=lambda: published.__setitem__("mode", "disabled"))
        )
        await asyncio.sleep(0)
        assert runtime.status == "draining"
        # The stream hasn't finished yet -- the pin and epoch must be
        # untouched.
        assert runtime.epoch_id == pre_exit_epoch_id
        assert runtime.router.get_pin(session_key.digest) is not None

        runtime.end_request()  # the stream finishes -> the drain can complete
        await asyncio.wait_for(exit_task, timeout=5.0)

        assert runtime.status == "disabled"
        assert published["mode"] == "disabled"

        # After the drain completed, the epoch WAS rotated (invalidating the
        # pin) -- confirmed durably, not just on the now-discarded runtime.
        inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
        try:
            assert inspect_store.balanced_epoch_id != pre_exit_epoch_id
            restore_result = inspect_store.restore(RestoreValidationContext(now_utc=time.time()))
            assert restore_result.pins == {}
        finally:
            inspect_store.close()

    asyncio.run(scenario())
