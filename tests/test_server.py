"""Integration tests for the gateway HTTP routes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

import claudex.admin.settings as admin_settings
import claudex.admin.system as admin_system
import claudex.relay.endpoints as relay_endpoints
import claudex.relay.kimi as relay_kimi
import claudex.relay.openai_backend as relay_openai_backend
import claudex.server as server
import claudex.server_support as server_support
from claudex import compaction, paths
from claudex.claude import accounts as claude_accounts
from claudex.claude.account_usage_cache import ClaudeAccountUsageCache
from claudex.balanced.polling import ClaudeUsagePollCoordinator, UsagePollAccount
from claudex.balanced.router import ClaudeBalancedRouter
from claudex.balanced.runtime import ClaudeBalancedRuntime
from claudex.balanced.selection import derive_session_key
from claudex.balanced.state_model import RestoreValidationContext
from claudex.balanced.state_store import ClaudePoolRuntimeStateStore
from claudex.providers.backends import AnthropicBackend, ResponsesBackend
from claudex.providers.codex_client import CODEX_FAST_TIER_WIRE_VALUE, CodexUpstreamError
from claudex.config import (
    AnthropicCompatibleProvider,
    ConfigError,
    GatewayConfig,
    OpenAICompatibleProvider,
)
from claudex.providers.anthropic_compatible_client import (
    AnthropicCompatibleClient,
)
from claudex.providers.kimi_auth import KimiCredentials
from claudex.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex.providers.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleUpstreamError,
)
from claudex.providers.grok_auth import GrokCredentials
from claudex.providers.grok_client import GrokUpstreamError
from claudex.translate.codex_to_claude import estimate_overflow_prompt_tokens


class AvailableCodexAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            is_api_key=False,
            account_id="account",
            access_token="codex",
            email="codex@example.com",
        )


class MissingCodexAuthManager(AvailableCodexAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> SimpleNamespace:
        raise server.CodexAuthError("missing Codex credentials")


class FakeCodexClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None

    async def supports_fast_tier(self, model: str) -> bool:
        return False


class AvailableKimiAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        return KimiCredentials(
            access_token="kimi-token", device_id="device-1", account="kimi-user-1"
        )


class MissingKimiAuthManager(AvailableKimiAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> KimiCredentials:
        raise server.KimiAuthError("no Kimi credentials; run `kimi login` first")


class FakeKimiClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def count_tokens(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return httpx.Response(200, json={"input_tokens": 1})

    async def list_models(self) -> Any:
        return {"data": []}


class AvailableGrokAuthManager:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        return GrokCredentials(access_token="grok-token", email="user@example.com")


class MissingGrokAuthManager(AvailableGrokAuthManager):
    async def get_credentials(self, force_refresh: bool = False) -> GrokCredentials:
        raise server.GrokAuthError("no Grok credentials; run `grok login` first")


class FakeGrokClient:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def context_window(self, model: str) -> int | None:
        return None


_CUSTOM_API_KEY = "sk-custom-secret"
_ANTHROPIC_CUSTOM_API_KEY = "sk-anthropic-custom-secret"


def _custom_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        wire_api="responses",
        base_url="https://models.example/api/v1",
        api_key=_CUSTOM_API_KEY,
    )


def _anthropic_custom_provider() -> AnthropicCompatibleProvider:
    return AnthropicCompatibleProvider(
        base_url="https://messages.example/api/v1",
        api_key=_ANTHROPIC_CUSTOM_API_KEY,
    )


class FakeOpenAICompatibleClient:
    def __init__(
        self,
        name: str,
        provider: OpenAICompatibleProvider,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        self.name = name
        self.provider = provider

    async def list_models(self) -> list[str]:
        return ["gpt-5.5"]

    async def context_window(self, model: str) -> int | None:
        return None


class FailingOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def list_models(self) -> list[str]:
        raise OpenAICompatibleUpstreamError(
            503, '{"error":{"message":"catalog unavailable"}}', self.name
        )


class SecretBearingStartupClient(FakeOpenAICompatibleClient):
    def __init__(
        self,
        name: str,
        provider: OpenAICompatibleProvider,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        super().__init__(name, provider)
        self.catalog_calls = 0

    async def list_models(self) -> list[str]:
        self.catalog_calls += 1
        raise httpx.ConnectError(
            f"startup catalog failed with {self.provider.api_key}"
        )


# The real HOME this process started with, captured before any test ever
# monkeypatches it -- lets `_create_test_client` tell a still-real HOME
# (a naive test that never isolated it) apart from one a caller already
# isolated itself (e.g. `_balanced_env`, `_admin_client`, under its own
# `tmp_path` subpath), so it isolates HOME without clobbering a caller's
# own choice.
_REAL_HOME = Path.home()


def _create_test_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    config: GatewayConfig | None = None,
    codex_auth: type = AvailableCodexAuthManager,
    codex_client: type = FakeCodexClient,
    kimi_auth: type = AvailableKimiAuthManager,
    kimi_client: type = FakeKimiClient,
    grok_auth: type = AvailableGrokAuthManager,
    grok_client: type = FakeGrokClient,
    custom_client: type | None = FakeOpenAICompatibleClient,
    base_url: str = "http://testserver",
) -> TestClient:
    # T-9's lifespan acquires the process-lifetime claude account pool lease
    # (paths.runtime_dir() == Path.home() / ".claudex") for every routing
    # mode, so a naive lifespan test run against the REAL HOME would contend
    # with a live claudex-gateway daemon holding that lock (G-6). Isolate
    # HOME to this test's own tmp_path before the lifespan ever runs, unless
    # the caller already isolated it to somewhere else first.
    if Path.home() == _REAL_HOME:
        monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(server, "CodexAuthManager", codex_auth)
    monkeypatch.setattr(server, "CodexClient", codex_client)
    monkeypatch.setattr(server, "KimiAuthManager", kimi_auth)
    monkeypatch.setattr(server, "KimiClient", kimi_client)
    monkeypatch.setattr(server, "GrokAuthManager", grok_auth)
    monkeypatch.setattr(server, "GrokClient", grok_client)
    if custom_client is not None:
        monkeypatch.setattr(server, "OpenAICompatibleClient", custom_client)
    return TestClient(server.create_app(config or GatewayConfig()), base_url=base_url)


def test_route_ownership_matches_surface_modules() -> None:
    def route_methods(*methods: str) -> frozenset[str]:
        result = set(methods)
        if "GET" in result:
            result.add("HEAD")
        return frozenset(result)

    expected_admin_routes = {
        ("/", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_dashboard",
        ),
        ("/dashboard.css", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_dashboard_css",
        ),
        ("/dashboard.js", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_dashboard_js",
        ),
        ("/favicon.ico", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_favicon",
        ),
        ("/api/hello", route_methods("GET")): (
            "claudex.admin.common",
            "_handle_hello",
        ),
        ("/health", route_methods("GET")): (
            "claudex.admin.common",
            "_handle_health",
        ),
        ("/admin/settings/mapping", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_mapping_get",
        ),
        ("/admin/settings/mapping", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_mapping_put",
        ),
        ("/admin/settings/log-level", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_log_level_get",
        ),
        ("/admin/settings/log-level", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_log_level_put",
        ),
        ("/admin/settings/compaction", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_compaction_get",
        ),
        ("/admin/settings/compaction", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_compaction_put",
        ),
        ("/admin/settings/codex", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_codex_get",
        ),
        ("/admin/settings/codex", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_codex_put",
        ),
        ("/admin/providers/codex/models", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_codex_models",
        ),
        ("/admin/providers/codex/reset-credit", route_methods("POST")): (
            "claudex.admin.system",
            "_handle_admin_codex_reset_credit",
        ),
        ("/admin/providers/kimi/models", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_kimi_models",
        ),
        ("/admin/providers/grok/models", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_grok_models",
        ),
        ("/admin/providers/custom/{name}/models", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_custom_models",
        ),
        ("/admin/providers/claude/local", route_methods("GET")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_local_get",
        ),
        ("/admin/providers/claude/accounts", route_methods("GET")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_accounts_get",
        ),
        ("/admin/providers/claude/accounts/{account_id}", route_methods("DELETE")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_account_delete",
        ),
        ("/admin/providers/claude/login", route_methods("GET")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_login_get",
        ),
        ("/admin/providers/claude/login", route_methods("POST")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_login_post",
        ),
        ("/admin/providers/claude/login", route_methods("DELETE")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_login_delete",
        ),
        ("/admin/providers/claude/login/code", route_methods("POST")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_login_code_post",
        ),
        ("/admin/providers/claude/login/replace", route_methods("POST")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_login_replace_post",
        ),
        ("/admin/providers/claude/pool/serving", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_claude_serving_get",
        ),
        ("/admin/providers/claude/pool/serving", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_claude_serving_put",
        ),
        ("/admin/providers/claude/pool/serving", route_methods("DELETE")): (
            "claudex.admin.settings",
            "_handle_admin_claude_serving_delete",
        ),
        ("/admin/providers/claude/pool/routing", route_methods("GET")): (
            "claudex.admin.settings",
            "_handle_admin_claude_routing_get",
        ),
        ("/admin/providers/claude/pool/routing", route_methods("PUT")): (
            "claudex.admin.settings",
            "_handle_admin_claude_routing_put",
        ),
        ("/admin/providers/claude/pool/status", route_methods("GET")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_pool_status",
        ),
        ("/admin/providers/claude/pool/usage", route_methods("GET")): (
            "claudex.admin.accounts",
            "_handle_admin_claude_accounts_usage",
        ),
        ("/admin/logs", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_logs",
        ),
        ("/admin/gptpro/session", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_session",
        ),
        ("/admin/gptpro/login", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_login_get",
        ),
        ("/admin/gptpro/login", route_methods("POST")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_login_post",
        ),
        ("/admin/gptpro/login", route_methods("DELETE")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_login_delete",
        ),
        ("/admin/gptpro/doctor", route_methods("POST")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_doctor",
        ),
        ("/admin/gptpro/mcp", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_mcp",
        ),
        ("/admin/gptpro/connect", route_methods("POST")): (
            "claudex.admin.system",
            "_handle_admin_gptpro_connect",
        ),
        ("/admin/usage", route_methods("GET")): (
            "claudex.admin.system",
            "_handle_admin_usage",
        ),
        ("/admin/test", route_methods("POST")): (
            "claudex.admin.system",
            "_handle_admin_connection_test",
        ),
    }

    app = server.create_app(GatewayConfig())
    admin_paths = {path for path, _methods in expected_admin_routes}
    admin_routes = [
        route
        for route in app.routes
        if hasattr(route, "endpoint") and route.path in admin_paths
    ]
    assert len(admin_routes) == len(expected_admin_routes)
    assert {
        (route.path, frozenset(route.methods)): (
            route.endpoint.__module__,
            route.endpoint.__name__,
        )
        for route in admin_routes
    } == expected_admin_routes

    relay_routes = {
        route.path: route.endpoint.__module__
        for route in app.routes
        if hasattr(route, "endpoint")
        and route.path in {"/v1/messages", "/v1/messages/count_tokens"}
    }
    assert relay_routes == {
        "/v1/messages": "claudex.relay.endpoints",
        "/v1/messages/count_tokens": "claudex.relay.endpoints",
    }


def test_lifespan_builds_complete_parallel_route_backend_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ConfigurableFastTierCodexClient(FakeCodexClient):
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.is_fast_tier_supported = True
            self.fast_tier_models: list[str] = []

        async def supports_fast_tier(self, model: str) -> bool:
            self.fast_tier_models.append(model)
            return self.is_fast_tier_supported

    config = GatewayConfig(
        custom_providers={
            "wrtn": _custom_provider(),
            "gemini-primary": _custom_provider(),
        }
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        codex_client=ConfigurableFastTierCodexClient,
    ) as client:
        state = client.app.state
        route_backends = state.route_backends

        assert set(route_backends) == set(config.route_providers)
        assert set(route_backends) == {
            "codex",
            "kimi",
            "grok",
            "wrtn",
            "gemini-primary",
        }
        assert "claude" not in route_backends
        assert "anthropic" not in route_backends

        codex_backend = route_backends["codex"]
        assert isinstance(codex_backend, ResponsesBackend)
        assert codex_backend.transport is state.codex_client
        assert codex_backend.adapt_probe_payload is server._adapt_identity_probe_payload
        assert not inspect.iscoroutinefunction(codex_backend.adapt_probe_payload)
        assert codex_backend.signature_namespace is None

        kimi_backend = route_backends["kimi"]
        assert isinstance(kimi_backend, AnthropicBackend)
        assert kimi_backend.transport is state.kimi_client
        assert kimi_backend.header_policy is relay_kimi._kimi_request_headers
        assert kimi_backend.error_policy is relay_kimi._kimi_error_to_claude
        assert kimi_backend.token_counter is not None
        assert kimi_backend.token_counter.__self__ is state.kimi_client
        assert kimi_backend.token_counter.__func__ is FakeKimiClient.count_tokens
        assert kimi_backend.catalog_loader is not None
        assert kimi_backend.catalog_loader.__self__ is state.kimi_client
        assert kimi_backend.catalog_loader.__func__ is FakeKimiClient.list_models

        grok_backend = route_backends["grok"]
        assert isinstance(grok_backend, ResponsesBackend)
        assert grok_backend.transport is state.grok_client
        assert grok_backend.adapt_probe_payload is server.sanitize_grok_payload
        assert not inspect.iscoroutinefunction(grok_backend.adapt_probe_payload)
        assert grok_backend.signature_namespace is None

        for name in ("wrtn", "gemini-primary"):
            custom_backend = route_backends[name]
            assert isinstance(custom_backend, ResponsesBackend)
            assert custom_backend.transport is state.custom_provider_clients[name]
            assert (
                custom_backend.adapt_probe_payload
                is server._adapt_identity_probe_payload
            )
            assert not inspect.iscoroutinefunction(custom_backend.adapt_probe_payload)
            assert custom_backend.signature_namespace == name
            assert custom_backend.catalog_loader is not None
            assert custom_backend.catalog_loader.__self__ is state.custom_provider_clients[name]
            assert custom_backend.catalog_loader.__func__ is FakeOpenAICompatibleClient.list_models

        disabled_payload = {"model": "gpt-5.6-sol"}
        disabled_result = asyncio.run(
            codex_backend.adapt_payload(disabled_payload, "gpt-5.6-sol")
        )
        assert disabled_result is disabled_payload
        assert "service_tier" not in disabled_payload
        assert state.codex_client.fast_tier_models == []

        state.config = GatewayConfig(
            custom_providers=config.custom_providers, codex_service_tier="fast"
        )
        probe_payload = {"model": "gpt-5.6-sol"}
        probe_result = codex_backend.adapt_probe_payload(
            probe_payload, "gpt-5.6-sol"
        )
        assert probe_result is probe_payload
        assert "service_tier" not in probe_payload
        assert state.codex_client.fast_tier_models == []

        fast_payload = {"model": "gpt-5.6-sol"}
        fast_result = asyncio.run(
            codex_backend.adapt_payload(fast_payload, "gpt-5.6-sol")
        )
        assert fast_result is fast_payload
        assert fast_payload["service_tier"] == CODEX_FAST_TIER_WIRE_VALUE
        assert state.codex_client.fast_tier_models == ["gpt-5.6-sol"]

        state.codex_client.is_fast_tier_supported = False
        fallback_payload = {"model": "unknown-model"}
        fallback_result = asyncio.run(
            codex_backend.adapt_payload(fallback_payload, "unknown-model")
        )
        assert fallback_result is fallback_payload
        assert "service_tier" not in fallback_payload
        assert state.codex_client.fast_tier_models == [
            "gpt-5.6-sol",
            "unknown-model",
        ]

        grok_probe_payload = {
            "model": "grok-4.5",
            "reasoning": {"effort": "future-effort"},
            "service_tier": "priority",
        }
        grok_probe_result = grok_backend.adapt_probe_payload(
            grok_probe_payload, "grok-4.5"
        )
        assert grok_probe_result is not grok_probe_payload
        assert "service_tier" not in grok_probe_result
        assert grok_probe_result["reasoning"] is grok_probe_payload["reasoning"]
        assert grok_probe_payload["reasoning"]["effort"] == "medium"

        grok_payload = {
            "model": "grok-4.5",
            "reasoning": {"effort": "future-effort"},
            "service_tier": "priority",
        }
        grok_result = asyncio.run(
            grok_backend.adapt_payload(grok_payload, "grok-4.5")
        )
        assert grok_result is not grok_payload
        assert "service_tier" not in grok_result
        assert grok_result["reasoning"] is grok_payload["reasoning"]
        assert grok_result["reasoning"]["effort"] == "medium"

        custom_payload = {"model": "gemini-2.5-pro", "input": []}
        custom_backend = route_backends["gemini-primary"]
        custom_probe_result = custom_backend.adapt_probe_payload(
            custom_payload, "gemini-2.5-pro"
        )
        assert custom_probe_result is custom_payload
        custom_result = asyncio.run(
            custom_backend.adapt_payload(custom_payload, "gemini-2.5-pro")
        )
        assert custom_result is custom_payload
        assert custom_payload == {"model": "gemini-2.5-pro", "input": []}


def test_lifespan_binds_mixed_custom_provider_families_by_config_type_without_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request_count = 0

    async def count_request(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    async_client_type = httpx.AsyncClient
    transport = httpx.MockTransport(count_request)

    def create_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return async_client_type(*args, **kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", create_async_client)
    responses_provider = _custom_provider()
    anthropic_provider = _anthropic_custom_provider()
    config = GatewayConfig(
        custom_providers={
            "anthropic-by-name": responses_provider,
            "openai-by-name": anthropic_provider,
        },
        model_map={"haiku": "openai-by-name:upstream-haiku"},
    )

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=None,
    ) as client:
        state = client.app.state
        responses_client = state.custom_provider_clients["anthropic-by-name"]
        anthropic_client = state.custom_provider_clients["openai-by-name"]
        assert isinstance(responses_client, OpenAICompatibleClient)
        assert isinstance(anthropic_client, AnthropicCompatibleClient)
        assert responses_client._http_client is state.http_client
        assert anthropic_client._http_client is state.http_client
        if responses_client._api_key != responses_provider.api_key:
            pytest.fail(
                "OpenAI-compatible client did not retain its configured credential"
            )
        if anthropic_client._api_key != anthropic_provider.api_key:
            pytest.fail(
                "Anthropic-compatible client did not retain its configured credential"
            )

        responses_backend = state.route_backends["anthropic-by-name"]
        assert isinstance(responses_backend, ResponsesBackend)
        assert responses_backend.transport is responses_client
        assert responses_backend.adapt_payload is server._adapt_identity_payload
        assert (
            responses_backend.adapt_probe_payload
            is server._adapt_identity_probe_payload
        )
        assert responses_backend.signature_namespace == "anthropic-by-name"
        assert responses_backend.catalog_loader is not None
        assert responses_backend.catalog_loader.__self__ is responses_client
        assert responses_backend.catalog_loader.__func__ is OpenAICompatibleClient.list_models

        anthropic_backend = state.route_backends["openai-by-name"]
        assert isinstance(anthropic_backend, AnthropicBackend)
        assert anthropic_backend.transport is anthropic_client
        assert (
            anthropic_backend.header_policy
            is server._anthropic_compatible_request_headers
        )
        assert (
            anthropic_backend.error_policy
            is server._anthropic_compatible_error_to_claude
        )
        assert anthropic_backend.token_counter is None
        assert anthropic_backend.catalog_loader is None
        assert set(state.route_backends) == set(config.route_providers)

    assert request_count == 0


def test_startup_redacts_secret_bearing_custom_provider_http_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:upstream-model"},
        custom_providers={"wrtn": _custom_provider()},
    )
    caplog.set_level(logging.WARNING, logger="claudex.server")

    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=SecretBearingStartupClient,
    ) as client:
        selected_client = client.app.state.custom_provider_clients["wrtn"]

    if _CUSTOM_API_KEY in caplog.text:
        pytest.fail("a configured custom-provider credential was exposed in logs")
    assert "custom provider unavailable" in caplog.text
    assert "ConnectError" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert selected_client.catalog_calls == 1


@pytest.mark.parametrize(
    ("config", "custom_provider_clients", "expected_message"),
    [
        (
            GatewayConfig(custom_providers={"wrtn": _custom_provider()}),
            {},
            "route backend registry mismatch: "
            "configured=['codex', 'grok', 'kimi', 'wrtn']; "
            "clients=['codex', 'grok', 'kimi']; "
            "backends=['codex', 'grok', 'kimi']",
        ),
        (
            GatewayConfig(),
            {
                "wrtn": FakeOpenAICompatibleClient(
                    "wrtn", _custom_provider()
                )
            },
            "route backend registry mismatch: "
            "configured=['codex', 'grok', 'kimi']; "
            "clients=['codex', 'grok', 'kimi', 'wrtn']; "
            "backends=['codex', 'grok', 'kimi']",
        ),
        (
            GatewayConfig(custom_providers={"codex": _custom_provider()}),
            {
                "codex": FakeOpenAICompatibleClient(
                    "codex", _custom_provider()
                )
            },
            "route backend registry mismatch: "
            "configured=['codex', 'codex', 'grok', 'kimi']; "
            "clients=['codex', 'codex', 'grok', 'kimi']; "
            "backends=['codex', 'codex', 'grok', 'kimi']",
        ),
    ],
)
def test_route_backend_registry_mismatch_fails_at_boot(
    config: GatewayConfig,
    custom_provider_clients: dict[str, FakeOpenAICompatibleClient],
    expected_message: str,
) -> None:
    app = server.create_app(config)
    app.state.config = config

    with pytest.raises(RuntimeError, match="route backend registry mismatch") as exc_info:
        server._assemble_route_backends(
            app,
            config,
            FakeCodexClient(),
            FakeKimiClient(),
            FakeGrokClient(),
            custom_provider_clients,
        )

    assert str(exc_info.value) == expected_message


def test_responses_dispatch_reads_route_registry_without_provider_client_branches() -> None:
    relay_source = inspect.getsource(
        relay_openai_backend._relay_via_responses_backend
    )
    assert "route_backends" in relay_source
    for legacy_dispatch in (
        "codex_client",
        "grok_client",
        "custom_provider_clients",
        "sanitize_grok_payload",
        'provider == "codex"',
        'provider == "grok"',
    ):
        assert legacy_dispatch not in relay_source

    endpoint_source = inspect.getsource(relay_endpoints)
    assert "route_backends" in endpoint_source
    assert 'route.provider == "kimi"' not in endpoint_source
    assert "kimi_client" not in endpoint_source
    assert "route_backends" not in inspect.getsource(relay_kimi)


def test_admin_connection_probe_selects_transport_and_payload_policy_from_binding() -> None:
    handler_source = inspect.getsource(admin_system._handle_admin_connection_test)
    responses_probe_source = inspect.getsource(admin_system._probe_responses_route)
    registry_source = inspect.getsource(admin_system._get_route_backend)
    assert "route_backends" in registry_source
    assert responses_probe_source.count("backend.adapt_probe_payload(") == 1
    assert "backend.transport.stream_responses(" in responses_probe_source
    assert "list_models" not in responses_probe_source
    assert "adapt_probe_payload" not in inspect.getsource(
        relay_openai_backend._relay_via_responses_backend
    )
    for legacy_selection in (
        "request.app.state.codex_client",
        "request.app.state.kimi_client",
        "request.app.state.grok_client",
        "request.app.state.custom_provider_clients",
        'route.provider == "codex"',
        'route.provider == "grok"',
        'route.provider == "kimi"',
        "sanitize_grok_payload",
    ):
        assert legacy_selection not in handler_source
        assert legacy_selection not in responses_probe_source


def test_messages_routes_enforce_local_bearer_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        messages = client.post("/v1/messages", json={"messages": []})
        count_tokens = client.post("/v1/messages/count_tokens", json={"messages": []})

    for response in (messages, count_tokens):
        assert response.status_code == 401
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "authentication_error"


def test_removed_responses_direction_routes_return_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.post("/v1/responses", json={"input": "Hello"}).status_code == 404
        assert client.get("/v1/models").status_code == 404


class StubCodexClient:
    """Records translated payloads, then fails with a recognizable error."""

    def __init__(
        self,
        context_window: int | None = None,
        error: CodexUpstreamError | None = None,
        supports_fast_tier: bool = False,
    ) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.context_window_calls: list[str] = []
        self._context_window = context_window
        self._error = error or CodexUpstreamError(503, "stub codex upstream")
        self._supports_fast_tier = supports_fast_tier

    async def context_window(self, model: str) -> int | None:
        self.context_window_calls.append(model)
        return self._context_window

    async def supports_fast_tier(self, model: str) -> bool:
        return self._supports_fast_tier

    async def stream_responses(self, payload: dict[str, Any], session_id: str):
        self.payloads.append(payload)
        raise self._error
        yield  # unreachable; makes this an async generator like the real client


def _gateway(
    config: GatewayConfig,
    anthropic_handler,
    kimi_handler=None,
    kimi_auth: Any | None = None,
    grok_client: Any | None = None,
    custom_provider_clients: dict[str, Any] | None = None,
    codex_context_window: int | None = None,
    codex_error: CodexUpstreamError | None = None,
    codex_supports_fast_tier: bool = False,
) -> tuple[TestClient, StubCodexClient]:
    app = server.create_app(config)
    # The lifespan requires real Codex credentials, so set the state directly
    # instead of entering the TestClient context manager.
    app.state.config = config
    app.state.compaction_last_reroute = None
    app.state.compaction_reroute_sequence = 0
    stub = StubCodexClient(
        codex_context_window, codex_error, codex_supports_fast_tier
    )
    app.state.codex_client = stub
    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(anthropic_handler)
    )
    kimi_backend_client: Any = FakeKimiClient()
    if kimi_handler is not None or kimi_auth is not None:
        kimi_backend_client = KimiClient(
            kimi_auth or AvailableKimiAuthManager(),
            httpx.AsyncClient(
                transport=httpx.MockTransport(
                    kimi_handler or (lambda request: httpx.Response(500))
                )
            ),
        )
    grok_backend_client = grok_client or FakeGrokClient()
    custom_backend_clients = custom_provider_clients or {}
    app.state.kimi_client = kimi_backend_client
    app.state.grok_client = grok_backend_client
    app.state.custom_provider_clients = custom_backend_clients
    app.state.route_backends = server._assemble_route_backends(
        app,
        config,
        stub,
        kimi_backend_client,
        grok_backend_client,
        custom_backend_clients,
    )
    return TestClient(app), stub


def _message_body(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _compaction_signal_text() -> str:
    return (
        f"{compaction.SIGNAL_A_PREFIX} Some filler context so the message resembles a "
        f"real conversation. {compaction.SIGNAL_A_MARKER} some more filler detail."
    )


def _compaction_body(
    model: str, *, max_tokens: int = 16, thinking_block: bool = False
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if thinking_block:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "internal reasoning that must never reach Anthropic",
                        "signature": "sig-from-a-different-backend",
                    },
                    {"type": "text", "text": "visible reply"},
                ],
            }
        )
    messages.append({"role": "user", "content": _compaction_signal_text()})
    return {"model": model, "max_tokens": max_tokens, "messages": messages}


def _pinned_reroute_record_keys() -> set[str]:
    return {
        "outcome",
        "timestamp",
        "target_model",
        "mapped_model",
        "estimated_prompt_tokens",
        "context_window",
        "detail",
    }


_ANTHROPIC_CREDENTIAL_HEADERS = {"x-api-key": "sk-ant-real-key"}


def _quota_429(marker: str = "Error") -> httpx.Response:
    """The empirically observed OAuth quota rejection: no reset signal at all."""
    return httpx.Response(
        429,
        json={"type": "error", "error": {"type": "rate_limit_error", "message": marker}},
        headers={"x-should-retry": "true"},
    )


class TestAdminMappingApi:
    """GET/PUT /admin/settings/mapping — runtime map changes persisted to settings.json."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        config = GatewayConfig(
            settings_file=tmp_path / "settings.json", **config_kwargs
        )
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_admin_requires_local_token_when_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/settings/mapping").status_code == 401
            response = client.get(
                "/admin/settings/mapping", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200


class TestAdminCompactionApi:
    """GET/PUT /admin/settings/compaction — compaction reroute target, mirroring /admin/settings/mapping."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_COMPACTION_MODEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    # --- GET: fresh/configured state, diagnostics schema -------------------

    def test_get_reports_last_reroute_with_exact_pinned_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            relay_openai_backend._assign_compaction_reroute(
                client.app.state,
                outcome="rerouted",
                target_model="claude-opus-5",
                mapped_model="codex:gpt-5.1-codex-max",
                estimated_prompt_tokens=4096,
                context_window=4000,
                detail=None,
            )
            payload = client.get("/admin/settings/compaction").json()

        last_reroute = payload["last_reroute"]
        assert set(last_reroute) == _pinned_reroute_record_keys()
        assert "sequence" not in last_reroute
        assert last_reroute["outcome"] == "rerouted"
        assert last_reroute["target_model"] == "claude-opus-5"
        assert last_reroute["mapped_model"] == "codex:gpt-5.1-codex-max"
        assert last_reroute["estimated_prompt_tokens"] == 4096
        assert last_reroute["context_window"] == 4000
        assert last_reroute["detail"] is None
        assert isinstance(last_reroute["timestamp"], str) and last_reroute["timestamp"]

    # --- PUT: persistence, hot-swap, disable, enable/disable trigger -------

    def test_put_enable_then_disable_changes_the_live_reroute_trigger(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        body = _compaction_body("claude-opus-4-6")
        window = estimate_overflow_prompt_tokens(body) - 1
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={"id": "msg_1", "type": "message", "model": "claude-opus-5"},
            )

        config = GatewayConfig(
            settings_file=tmp_path / "settings.json",
            model_map={"opus": "codex:gpt-5.1-codex-max"},
        )
        client, stub = _gateway(config, handler, codex_context_window=window)
        client.app.state.admin_lock = asyncio.Lock()
        client.base_url = "http://127.0.0.1:8787"

        enable = client.put("/admin/settings/compaction", json={"model": "claude:claude-opus-5"})
        assert enable.status_code == 200

        rerouted = client.post(
            "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
        )
        assert rerouted.status_code == 200
        assert len(captured) == 1
        assert stub.payloads == []

        disable = client.put("/admin/settings/compaction", json={"model": None})
        assert disable.status_code == 200

        not_rerouted = client.post(
            "/v1/messages", json=body, headers=_ANTHROPIC_CREDENTIAL_HEADERS
        )
        # Reroute is now off: falls through to the mapped Codex backend
        # (the stub, which always fails), never a second Anthropic attempt.
        assert not_rerouted.status_code == 503
        assert len(captured) == 1
        assert len(stub.payloads) == 1



def _record_reset_keys(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[dict[str, Any]]
) -> list[str]:
    """Stub the consume call, answering `outcomes` in order and logging the keys."""
    keys: list[str] = []
    remaining = list(outcomes)

    async def fake_consume(
        http_client: Any, auth_manager: Any, redeem_request_id: str
    ) -> dict[str, Any]:
        keys.append(redeem_request_id)
        return remaining.pop(0)

    monkeypatch.setattr(admin_system, "consume_codex_reset_credit", fake_consume)
    return keys


def _compaction_section(page: str) -> str:
    """Slice out only the Compaction card's own markup, by its own markers.

    `claude-haiku` must never appear as a curated compaction option, but the
    Router tab's `CLAUDE_KEYS` array legitimately mentions bare "haiku"
    elsewhere in the same document — so absence has to be asserted against
    this scoped slice, never the whole page.
    """
    start = page.index("<!-- compaction-section:start -->")
    end = page.index("<!-- compaction-section:end -->")
    return page[start:end]


def _compaction_apply_fn(page: str) -> str:
    start = page.index("function applyCompaction(){")
    end = page.index(
        'document.getElementById("comp-select").addEventListener("change"', start
    )
    return page[start:end]


def _routing_section(page: str) -> str:
    """Slice the routing card's own markup — "balanced" legitimately appears
    elsewhere in the document (API test prose never, but future copy might),
    so absence is asserted against this scoped slice."""
    start = page.index("<!-- routing-section:start -->")
    end = page.index("<!-- routing-section:end -->")
    return page[start:end]


def _codex_section(page: str) -> str:
    start = page.index("<!-- codex-section:start -->")
    end = page.index("<!-- codex-section:end -->")
    return page[start:end]


class CatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        return ["gpt-5.6-sol", "gpt-5.5"]


class FailingCatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        raise CodexUpstreamError(400, '{"error":{"message":"unsupported client"}}')


class ProbeCodexClient(FakeCodexClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class RejectingCodexClient(FakeCodexClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise CodexUpstreamError(400, '{"error":{"message":"model_not_found"}}')


class CatalogKimiClient(FakeKimiClient):
    async def list_models(self) -> Any:
        return {"data": [{"id": "k2.5"}, {"id": "k3"}]}


class FailingCatalogKimiClient(FakeKimiClient):
    async def list_models(self) -> Any:
        raise KimiUpstreamError(401, '{"error":{"message":"token expired"}}')


class ProbeKimiClient(FakeKimiClient):
    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        # The probe must send the raw model with the prefix already removed.
        assert json.loads(body)["model"] == "k3"
        return httpx.Response(200, json={"type": "message", "model": "k3"})


class RejectingKimiClient(FakeKimiClient):
    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        raise KimiUpstreamError(
            404, '{"type":"error","error":{"type":"not_found_error","message":"model not found"}}'
        )


class ProbeGrokClient(FakeGrokClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class CatalogGrokClient(FakeGrokClient):
    async def list_models(self) -> list[str]:
        return ["grok-4.5", "grok-4.3"]


class FailingCatalogGrokClient(FakeGrokClient):
    async def list_models(self) -> list[str]:
        raise GrokUpstreamError(401, '{"error":{"message":"token expired"}}')


class RejectingGrokClient(FakeGrokClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise GrokUpstreamError(400, '{"error":{"message":"model_not_found"}}')


class CatalogOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def list_models(self) -> list[str]:
        return ["gpt-5.5", "gemini-3.1-pro"]


class ProbeOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class RejectingOpenAICompatibleClient(FakeOpenAICompatibleClient):
    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        if False:
            yield {}
        raise OpenAICompatibleUpstreamError(
            400, '{"error":{"message":"model_not_found"}}', self.name
        )


# ---------------------------------------------------------------------------
# Anthropic passthrough served with a registered Claude account
# ---------------------------------------------------------------------------


def _register_serving_account(
    *,
    email: str = "pool@example.com",
    account_uuid: str | None = "serving-account-uuid",
    access_token: str = "pool-access-1",
    refresh_token: str = "pool-refresh-1",
    expires_in_seconds: float = 3600,
) -> str:
    """Register one ready account under the (HOME-isolated) registry.

    Distinct emails register distinct accounts (the registry deduplicates on
    the (email, organizationUuid) pair), which is how pool tests build a
    multi-account chain.
    """
    oauth_account: dict[str, Any] = {"emailAddress": email}
    if account_uuid is not None:
        oauth_account["accountUuid"] = account_uuid
    record = claude_accounts.add_account(
        email=email,
        organization_uuid="org-1",
        organization_name="Example Org",
        credentials_json={
            "claudeAiOauth": {
                "accessToken": access_token,
                "refreshToken": refresh_token,
                "expiresAt": (time.time() + expires_in_seconds) * 1000,
                "scopes": ["user:inference", "user:profile"],
            }
        },
        oauth_account_json=oauth_account,
    )
    return record.id


def _claude_code_user_id(account_uuid: str = "client-account-uuid") -> str:
    return json.dumps(
        {
            "device_id": "d" * 64,
            "account_uuid": account_uuid,
            "session_id": "11111111-2222-4333-8444-555555555555",
        },
        separators=(",", ":"),
    )


def _account_body(model: str = "claude-fable-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {"user_id": _claude_code_user_id()}
    return body



# ---------------------------------------------------------------------------
# Admin claude-accounts surface (dashboard account management)
# ---------------------------------------------------------------------------


_ADMIN_BASE = "http://127.0.0.1:8787"


class TestAdminClaudeAccountsApi:
    @staticmethod
    def _client(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **config_kwargs: Any
    ) -> TestClient:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        return _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(
                settings_file=tmp_path / "settings.json", **config_kwargs
            ),
            base_url=_ADMIN_BASE,
        )

    def test_pool_status_reports_ready_cooldown_and_unavailable_members(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        millis = iter(range(1_700_000_000_000, 1_700_000_000_100))
        monkeypatch.setattr(claude_accounts, "_now_millis", lambda: next(millis))
        ready_id = _register_serving_account()
        cooling_id = _register_serving_account(
            email="pool2@example.com",
            account_uuid="second-account-uuid",
            access_token="pool-access-2",
            refresh_token="pool-refresh-2",
        )
        dead_id = _register_serving_account(
            email="pool3@example.com",
            account_uuid="third-account-uuid",
            access_token="pool-access-3",
            refresh_token="pool-refresh-3",
        )
        claude_accounts.mark_account_needs_reauth(dead_id)
        client.app.state.claude_account_cooldowns.mark(cooling_id, 120.0)

        with client:
            response = client.get("/admin/providers/claude/pool/status")

        assert response.status_code == 200
        members = {member["account_id"]: member for member in response.json()["members"]}
        assert members[ready_id] == {"account_id": ready_id, "routing_state": "ready"}
        cooling = members[cooling_id]
        assert cooling["routing_state"] == "cooldown"
        # Epoch ms, like every other registry timestamp in the payload.
        expected = (time.time() + 120.0) * 1000
        assert abs(cooling["cooldown_until"] - expected) < 5_000
        assert members[dead_id] == {
            "account_id": dead_id,
            "routing_state": "unavailable",
            "reason": "needs-reauth",
        }

    def test_account_delete_removes_the_registry_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        client.app.state.claude_account_cooldowns.mark(account_id, 120.0)

        with client:
            # Prime the cached auth manager so the delete has one to drop.
            client.app.state.claude_account_auth_managers[account_id] = object()
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert response.status_code == 204
        assert response.content == b""
        assert claude_accounts.list_accounts() == []
        assert account_id not in client.app.state.claude_account_auth_managers
        assert not client.app.state.claude_account_cooldowns.is_cooling(account_id)


# ---------------------------------------------------------------------------
# Claude account pool lease: acquired at lifespan startup and held for the
# process lifetime, regardless of routing mode (T-9)
# ---------------------------------------------------------------------------


def test_lifespan_holds_the_claude_pool_lease_for_the_process_lifetime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lock_path = paths.claude_account_pool_lock()

    with _create_test_client(monkeypatch, tmp_path) as client:
        assert lock_path.is_file()
        # Contended while the lifespan-backed client is open.
        assert server.try_file_lock(lock_path) is None
        assert client.get("/api/hello").status_code == 200

    # Released once the client (and its lifespan) has shut down.
    reacquired = server.try_file_lock(lock_path)
    assert reacquired is not None
    reacquired.release()


def test_lifespan_raises_the_pinned_message_when_the_pool_lock_is_contended(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    lock_path = paths.claude_account_pool_lock()

    holder = server.try_file_lock(lock_path)
    assert holder is not None
    try:
        client = _create_test_client(monkeypatch, tmp_path)
        with pytest.raises(RuntimeError) as exc_info:
            with client:
                pass
        assert str(exc_info.value) == (
            "claude account pool is already served by another process "
            "(balanced-router.lock held)"
        )
    finally:
        holder.release()


# ---------------------------------------------------------------------------
# Balanced routing lifecycle: transactional mode changes, restart-preserving
# shutdown, fail-closed dispatch (T-10)
# ---------------------------------------------------------------------------

_BALANCED_ACCOUNT_UUID = "11111111-2222-3333-4444-555555555555"


def _balanced_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate HOME under `tmp_path` and clear the env-lock override, exactly
    like `TestAdminClaudeRoutingApi._admin_client` -- so registry/pool state
    lives under an isolated, per-test `.claudex` directory."""
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _register_balanced_ready_account(
    *, email: str = "balanced@example.com", account_uuid: str | None = _BALANCED_ACCOUNT_UUID
) -> str:
    """Register one ready account. With the default canonical `account_uuid`, it
    carries a valid T-3 profile fingerprint and can enable balanced routing; a
    caller after `account_uuid=None` (or any non-UUID string, e.g.
    `_register_serving_account`'s own default) gets one that cannot.
    """
    return _register_serving_account(email=email, account_uuid=account_uuid)


class TestBalancedRoutingEnable:
    """PUT .../pool/routing {"mode": "balanced"} -- prepares the complete runtime
    (store, epoch, router, T-3 fingerprint verification) while the old mode keeps
    serving, and persists settings only once every check passes (T-10 Steps 1-7).
    """

    def test_put_balanced_enables_with_valid_fingerprints(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            runtime = client.app.state.claude_balanced_runtime
            config_after = client.app.state.config
            # Read while the runtime is still live -- lifespan shutdown resets
            # `status` to "disabled" on `with`-block exit (shutdown_preserving_epoch).
            status_while_active = runtime.status
            epoch_id_while_active = runtime.epoch_id
            router_while_active = runtime.router

        assert response.status_code == 200
        assert response.json() == {"mode": "balanced", "env_locked": False}
        assert config_after.claude_account_routing_mode == "balanced"
        assert status_while_active == "active"
        assert epoch_id_while_active is not None
        assert router_while_active is not None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"claude_account": {"routing": {"mode": "balanced"}}}

    def test_put_balanced_rejects_invalid_account_uuid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        # The default account_uuid ("serving-account-uuid") is not a UUID, so
        # this account carries no T-3 profile fingerprint.
        _register_balanced_ready_account(account_uuid="serving-account-uuid")
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            runtime = client.app.state.claude_balanced_runtime
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "profile_fingerprint" in response.json()["error"]["message"]
        assert config_after.claude_account_routing_mode == "disabled"
        assert runtime.status == "disabled"
        assert not settings_file.exists()

    def test_put_balanced_rejects_store_open_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)

        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated store-open failure")

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "open_", classmethod(_boom))
                response = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
                )
            runtime = client.app.state.claude_balanced_runtime

        assert response.status_code == 500
        assert "could not enable balanced routing" in response.json()["error"]["message"]
        assert runtime.status == "disabled"
        assert not settings_file.exists()

    def test_put_balanced_tears_down_before_settings_persist_on_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Publication ordering (Context): a failure at the persist() commit point
        tears preparation down and leaves the old mode -- and a clean retry
        (store closed, no lingering lock) then succeeds.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file)

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("simulated disk-full settings write")

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            with monkeypatch.context() as fault:
                fault.setattr(admin_settings, "update_settings_file", _boom)
                failed = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
                )
            runtime = client.app.state.claude_balanced_runtime
            status_after_failure = runtime.status
            config_after_failure = client.app.state.config
            settings_missing_after_failure = not settings_file.exists()

            retried = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            status_after_retry = runtime.status

        assert failed.status_code == 500
        assert status_after_failure == "disabled"
        assert config_after_failure.claude_account_routing_mode == "disabled"
        assert settings_missing_after_failure
        assert retried.status_code == 200
        assert status_after_retry == "active"

    def test_lifespan_activates_persisted_balanced_mode_at_startup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.routing": {"mode": "balanced"}}),
            encoding="utf-8",
        )
        config = GatewayConfig.load(settings_file)
        assert config.claude_account_routing_mode == "balanced"

        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = client.app.state.claude_balanced_runtime
            payload = client.get("/admin/providers/claude/pool/routing").json()
            status_while_active = runtime.status
            epoch_id_while_active = runtime.epoch_id

        assert status_while_active == "active"
        assert epoch_id_while_active is not None
        assert payload == {"mode": "balanced", "env_locked": False}

    def test_get_reports_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_ready_account()
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            payload = client.get("/admin/providers/claude/pool/routing").json()

        assert enable.status_code == 200
        assert payload == {"mode": "balanced", "env_locked": False}



class TestBalancedRoutingExit:
    """Intentional balanced -> fallback/disabled exit (T-10 Step 8): drains
    in-flight dispatch, persists+publishes the target mode, invalidates the
    current epoch's pins, and starts a fresh epoch on any later re-entry --
    contrast `test_graceful_shutdown_preserves_epoch_and_pin_for_restart`
    below, where a process shutdown preserves that very same state instead.
    """

    def test_exit_drains_persists_invalidates_pins_and_reentry_gets_new_epoch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            epoch_id_1 = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1

            runtime_db_path = paths.claude_account_pool_runtime_db()

            exited = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            served_after_exit = client.post("/v1/messages", json=_account_body())

            assert exited.status_code == 200
            assert exited.json() == {"mode": "disabled", "env_locked": False}
            assert runtime.status == "disabled"
            # Draining resolved cleanly: the in-flight-created pin's request, and
            # a fresh one after exit (now served single-account, "disabled"), both
            # completed rather than erroring.
            assert served_after_exit.status_code == 200
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert "claude_account" not in saved

            # The exited epoch's pins are gone at the persistence layer too --
            # not just discarded in memory with this runtime instance.
            inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
            try:
                restore_result = inspect_store.restore(
                    RestoreValidationContext(now_utc=time.time())
                )
                epoch_id_after_exit = inspect_store.balanced_epoch_id
            finally:
                inspect_store.close()
            assert epoch_id_after_exit != epoch_id_1
            assert restore_result.pins == {}

            reentered = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert reentered.status_code == 200
            assert runtime.status == "active"
            assert runtime.epoch_id != epoch_id_1
            # T-22 (fix for gap G-7): the admin re-entry path durably mints its
            # OWN fresh epoch, independent of the one `exit_mode` already
            # rotated to above -- it never simply reuses whatever the store
            # happens to hold.
            assert runtime.epoch_id != epoch_id_after_exit
            assert runtime.router is not None
            assert runtime.router.pin_count() == 0

    def test_exit_persistence_failure_aborts_and_leaves_epoch_and_pins_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Crash contract (T-20, fix for gap G-3): a settings-persistence
        failure during exit must abort BEFORE anything balanced-only is
        touched -- the PUT returns 500, the runtime resumes "active", and
        the epoch id and its durable pins are exactly as they were, still
        serving requests under balanced with the same pins.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise ConfigError("simulated disk-full settings write")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            epoch_id_before = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1

            with monkeypatch.context() as fault:
                fault.setattr(admin_settings, "update_settings_file", _boom)
                failed = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )

            assert failed.status_code == 500
            assert runtime.status == "active"
            assert runtime.epoch_id == epoch_id_before
            assert client.app.state.config.claude_account_routing_mode == "balanced"
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert saved["claude_account"]["routing"] == {"mode": "balanced"}

            # The durable pin the persistence failure was supposed to leave
            # untouched is still there, both in memory and at the durable
            # layer -- a subsequent request keeps being served under
            # balanced, reusing it rather than a fresh placement.
            assert runtime._store is not None
            session_key = derive_session_key(_account_body(), runtime.epoch_seed, "fable")
            assert session_key is not None
            assert runtime.router.get_pin(session_key.digest) is not None
            assert runtime._store.get_pin(session_key.digest) is not None

            served = client.post("/v1/messages", json=_account_body())
            assert served.status_code == 200
            assert runtime.router.pin_count() == 1

    def test_exit_epoch_rotation_failure_after_persistence_surfaces_persistence_degraded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Crash contract (T-20, fix for gap G-3): once `persist()` has
        already committed the target mode, a subsequent epoch-rotation
        failure must NOT roll back to balanced -- the exit still completes
        under the target mode, with the cleanup failure only surfaced as
        persistence_degraded.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom_rotate(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated durable epoch rotation failure")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        caplog.set_level(logging.WARNING, logger="claudex.balanced.runtime")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime

            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "rotate_epoch", _boom_rotate)
                exited = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )

            assert exited.status_code == 200
            assert exited.json() == {"mode": "disabled", "env_locked": False}
            assert runtime.status == "disabled"
            saved = json.loads(settings_file.read_text(encoding="utf-8"))
            assert "claude_account" not in saved

        assert any(
            "persistence_degraded" in record.getMessage() for record in caplog.records
        )

    def test_admin_reenable_after_degraded_exit_rotation_mints_a_fresh_epoch_and_drops_the_old_pin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Gap G-7(b) regression: `exit_mode`'s own epoch rotation can fail
        (`persistence_degraded`, exercised above) and leave the runtime DB
        holding the exited epoch's id/seed and its durable pin. A later admin
        re-enable must never resurrect that epoch or pin --
        `prepare_and_publish`'s `entry="admin_enable"` branch durably mints a
        fresh epoch (wiping pins) before it ever restores from the store,
        regardless of what the store still contains.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "msg_1"})

        def _boom_rotate(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated durable epoch rotation failure")

        settings_file = tmp_path / "settings.json"
        config = GatewayConfig(settings_file=settings_file, claude_account_id=account_id)
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            )
            enable = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert enable.status_code == 200
            runtime = client.app.state.claude_balanced_runtime
            old_epoch_id = runtime.epoch_id

            pinned = client.post("/v1/messages", json=_account_body())
            assert pinned.status_code == 200
            assert runtime.router is not None
            assert runtime.router.pin_count() == 1
            session_key = derive_session_key(_account_body(), runtime.epoch_seed, "fable")
            assert session_key is not None
            old_pin_digest = session_key.digest

            with monkeypatch.context() as fault:
                fault.setattr(ClaudePoolRuntimeStateStore, "rotate_epoch", _boom_rotate)
                exited = client.put(
                    "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
                )
            assert exited.status_code == 200
            assert runtime.status == "disabled"

            # The degraded exit's own rotation never ran -- the runtime DB
            # still durably holds the "exited" epoch id and its pin, exactly
            # the resurrection precondition gap G-7(b) describes.
            runtime_db_path = paths.claude_account_pool_runtime_db()
            inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
            try:
                assert inspect_store.balanced_epoch_id == old_epoch_id
                assert inspect_store.get_pin(old_pin_digest) is not None
            finally:
                inspect_store.close()

            reentered = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
            )
            assert reentered.status_code == 200
            new_runtime = client.app.state.claude_balanced_runtime
            assert new_runtime.status == "active"
            # A durably fresh epoch -- never the exited one the degraded
            # rotation left behind.
            assert new_runtime.epoch_id != old_epoch_id
            assert new_runtime.router is not None
            assert new_runtime.router.pin_count() == 0
            assert new_runtime.router.get_pin(old_pin_digest) is None
            assert new_runtime._store is not None
            assert new_runtime._store.get_pin(old_pin_digest) is None


def test_balanced_request_during_controlled_exit_awaits_and_serves_under_target_mode(
    tmp_path: Path,
) -> None:
    """Step 6/8's spec-gate ruling, exercised directly against
    `ClaudeBalancedRuntime` (the exact primitives
    `relay._passthrough_with_claude_balanced` dispatches through): a request
    arriving while `exit_mode` is draining awaits the transition, then observes
    the ALREADY-published target mode -- never a 503 merely because the
    controlled transition was in flight.
    """
    from claudex.claude.accounts import AccountRecord
    from claudex.balanced.runtime import ClaudeBalancedRuntime

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
    published = {"mode": "disabled"}

    async def scenario() -> str:
        runtime = ClaudeBalancedRuntime()
        await runtime.prepare_and_publish(
            accounts=[account],
            accounts_root=accounts_root,
            runtime_db_path=tmp_path / "runtime.sqlite3",
            persist=lambda: published.__setitem__("mode", "balanced"),
            entry="admin_enable",
        )
        assert runtime.status == "active"

        # Simulate one in-flight balanced dispatch holding a slot open.
        assert runtime.begin_request()

        exit_task = asyncio.create_task(
            runtime.exit_mode(
                "fallback", publish=lambda: published.__setitem__("mode", "fallback")
            )
        )
        await asyncio.sleep(0)
        assert runtime.status == "draining"

        async def waiting_dispatch() -> str:
            # Mirrors relay._passthrough_with_claude_balanced's transition-await
            # branch: wait, then re-read the published mode.
            await runtime.wait_for_transition()
            return published["mode"]

        waiter_task = asyncio.create_task(waiting_dispatch())
        await asyncio.sleep(0)
        assert not waiter_task.done()  # still awaiting -- draining hasn't resolved

        runtime.end_request()  # the in-flight request finishes -> drain completes
        target_mode_seen = await asyncio.wait_for(waiter_task, timeout=5.0)
        await asyncio.wait_for(exit_task, timeout=5.0)
        assert runtime.status == "disabled"
        return target_mode_seen

    assert asyncio.run(scenario()) == "fallback"


def test_balanced_graceful_shutdown_preserves_epoch_and_pin_for_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Step 9: two lifespan-backed app instances over the same pool directory.
    Closing the first (settings still "balanced") preserves the epoch id, seed,
    and durable pin for the second's automatic startup restoration -- contrast
    `TestBalancedRoutingExit`, where an INTENTIONAL exit invalidates that very
    same state instead.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
    _register_balanced_ready_account()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_1"})

    settings_file = tmp_path / "settings.json"
    first_config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=first_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        enable = client.put(
            "/admin/providers/claude/pool/routing", json={"mode": "balanced"}
        )
        assert enable.status_code == 200
        first_runtime = client.app.state.claude_balanced_runtime
        epoch_id_1 = first_runtime.epoch_id
        seed_1 = first_runtime.epoch_seed

        pinned = client.post("/v1/messages", json=_account_body())
        assert pinned.status_code == 200
        assert first_runtime.router is not None
        assert first_runtime.router.pin_count() == 1

    # The first instance's lifespan has shut down (shutdown_preserving_epoch,
    # then the T-9 lease released) -- a second, independent app instance loads
    # the SAME settings file and pool directory (HOME unchanged) and restores
    # them automatically at startup.
    second_config = GatewayConfig.load(settings_file)
    assert second_config.claude_account_routing_mode == "balanced"
    with _create_test_client(
        monkeypatch, tmp_path, config=second_config, base_url="http://127.0.0.1:8787"
    ) as client:
        second_runtime = client.app.state.claude_balanced_runtime
        assert second_runtime.status == "active"
        assert second_runtime.epoch_id == epoch_id_1
        assert second_runtime.epoch_seed == seed_1
        assert second_runtime.router is not None
        assert second_runtime.router.pin_count() == 1
        payload = client.get("/admin/providers/claude/pool/routing").json()

    assert payload == {"mode": "balanced", "env_locked": False}


# ---------------------------------------------------------------------------
# Balanced serve-path chain runner: commit-at-headers protocol and §6.5
# exhaustion responses (T-11). The Step 3 fallback-regression guard is every
# `test_pool_*`/`test_account_passthrough_*` test above, left byte-for-byte
# unmodified and still green in this same run.
# ---------------------------------------------------------------------------


def _register_balanced_accounts(count: int) -> list[tuple[str, str]]:
    """Register `count` balanced-ready accounts, each with a distinct valid T-3
    fingerprint and a distinct access token, in registration order.
    """
    accounts = []
    for index in range(count):
        access_token = f"balanced-access-{index}"
        account_id = _register_serving_account(
            email=f"balanced{index}@example.com",
            account_uuid=str(uuid.uuid4()),
            access_token=access_token,
            refresh_token=f"balanced-refresh-{index}",
        )
        accounts.append((account_id, access_token))
    return accounts


_USAGE_PROBE_PATH = "/api/oauth/usage"  # claude_usage._CLAUDE_USAGE_URL's path component


def _usage_probe_intercepting_handler(anthropic_handler: Any) -> Any:
    """Wrap `anthropic_handler` so the T-18 background usage poll driver's own
    automatic probe (`GET .../api/oauth/usage`) is answered directly with a
    no-window "ok" envelope, without ever reaching `anthropic_handler` --
    once balanced routing is active, that driver immediately runs its own
    warm-up poll in the background, and most tests here only care about
    `/v1/messages` traffic (call counts, retry/migration sequencing, ...).
    """

    def transport_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == _USAGE_PROBE_PATH:
            return httpx.Response(200, json={})
        return anthropic_handler(request)

    return transport_handler


def _enable_balanced(
    client: TestClient, anthropic_handler: Any, *, intercept_usage_probe: bool = True
) -> Any:
    """Swap in a mock Anthropic transport and enable balanced routing.

    `intercept_usage_probe` (default `True`, T-18) wraps `anthropic_handler`
    with `_usage_probe_intercepting_handler`; pass `False` for the few tests
    that deliberately want to observe or control the driver's own automatic
    probe's HTTP behavior.

    Returns the resulting `ClaudeBalancedRuntime`.
    """
    handler = (
        _usage_probe_intercepting_handler(anthropic_handler)
        if intercept_usage_probe
        else anthropic_handler
    )
    client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
    assert response.status_code == 200, response.text
    return client.app.state.claude_balanced_runtime


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _balanced_body(session_id: str, *, model: str = "claude-fable-5") -> dict[str, Any]:
    body = _message_body(model)
    body["metadata"] = {
        "user_id": json.dumps(
            {
                "device_id": "d" * 64,
                "account_uuid": "client-account-uuid",
                "session_id": session_id,
            },
            separators=(",", ":"),
        )
    }
    return body


_UNPINNABLE_BODY: dict[str, Any] = {
    "model": "claude-fable-5",
    "max_tokens": 16,
    "messages": [],
}



# ---------------------------------------------------------------------------
# Durable cooldowns, Fable family gate, capability evidence, removal cleanup
# (T-12, design v2 §6.4/§5.5/§5.7, adjudication G)
# ---------------------------------------------------------------------------


def _ingest_gate_satisfying_fable_observations(router: ClaudeBalancedRouter, account_id: str) -> None:
    """Fresh, REAL readings satisfying every non-status/non-family §6.4 gate
    condition: `fable_weekly` >=99%, `five_hour`/`seven_day` <=70%, all <=15 min
    old, with a valid future Fable reset."""
    router.ingest_observation(
        account_id,
        "fable_weekly",
        used_percent=99.0,
        source="usage_api",
        age_seconds=0.0,
        reset_in_seconds=3600.0,
        reset_identity="fable-reset-1",
    )
    router.ingest_observation(account_id, "five_hour", used_percent=40.0, source="usage_api", age_seconds=0.0)
    router.ingest_observation(account_id, "seven_day", used_percent=40.0, source="usage_api", age_seconds=0.0)



def test_balanced_restart_restores_the_family_cooldown_without_a_repeat_429_burst(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A daemon restart restores durable cooldowns (§5.4/§5.5): the second
    process instance skips the still-cooling account for its own Fable-family
    request WITHOUT ever calling upstream again -- no repeat-429 burst.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["authorization"])
        return _quota_429()

    settings_file = tmp_path / "settings.json"
    first_config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=first_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_usage_probe_intercepting_handler(handler))
        )
        enable = client.put("/admin/providers/claude/pool/routing", json={"mode": "balanced"})
        assert enable.status_code == 200
        first_runtime = client.app.state.claude_balanced_runtime
        router = first_runtime.router
        assert router is not None
        _ingest_gate_satisfying_fable_observations(router, account_id)

        first_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert first_response.status_code == 429
        assert len(calls) == 1
        assert router.family_cooldown_deadline(account_id, "fable", now=time.monotonic()) is not None

    # A second, independent process instance over the same pool directory
    # (mirrors `test_balanced_graceful_shutdown_preserves_epoch_and_pin_for_restart`).
    second_config = GatewayConfig.load(settings_file)
    assert second_config.claude_account_routing_mode == "balanced"
    with _create_test_client(
        monkeypatch, tmp_path, config=second_config, base_url="http://127.0.0.1:8787"
    ) as client:
        client.app.state.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_usage_probe_intercepting_handler(handler))
        )
        second_runtime = client.app.state.claude_balanced_runtime
        assert second_runtime.status == "active"
        assert second_runtime.router is not None
        assert second_runtime.router.family_cooldown_deadline(
            account_id, "fable", now=time.monotonic()
        ) is not None

        second_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))

        assert second_response.status_code == 429
        assert len(calls) == 1  # no repeat-429 burst: the restored cooldown skipped upstream entirely


def test_balanced_account_removal_clears_its_durable_rows_for_that_incarnation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removing an account while balanced is active runs the router's own
    removal matrix and deletes every durable row of that incarnation (§5.7) --
    verified directly against the persisted store after the process shuts down.
    """
    _balanced_env(monkeypatch, tmp_path)
    account_id, _access_token = _register_balanced_accounts(1)[0]

    def handler(request: httpx.Request) -> httpx.Response:
        return _quota_429()

    settings_file = tmp_path / "settings.json"
    config = GatewayConfig(settings_file=settings_file)
    with _create_test_client(
        monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        runtime = _enable_balanced(client, handler)
        router = runtime.router
        assert router is not None

        incarnation = next(
            record.account_incarnation_id
            for record in claude_accounts.list_accounts()
            if record.id == account_id
        )

        # One 429 leaves both a pin (still pointing at the cooling account) and
        # a durable cooldown row -- both durably written and awaited already.
        cooling_response = client.post("/v1/messages", json=_balanced_body(_new_session_id()))
        assert cooling_response.status_code == 429
        assert router.pin_count() == 1
        assert router.account_cooldown_deadline(account_id, now=time.monotonic()) is not None

        delete_response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert delete_response.status_code == 204
        assert router.pin_count() == 0
        assert router.account_cooldown_deadline(account_id, now=time.monotonic()) is None

    runtime_db_path = paths.claude_account_pool_runtime_db()
    inspect_store = ClaudePoolRuntimeStateStore.open_(runtime_db_path)
    try:
        restore_result = inspect_store.restore(RestoreValidationContext(now_utc=time.time()))
        assert not any(row.account_incarnation_id == incarnation for row in restore_result.pins.values())
        assert not any(row.account_incarnation_id == incarnation for row in restore_result.cooldowns.values())
        assert not any(
            row.account_incarnation_id == incarnation
            for row in restore_result.usage_observations.values()
        )
        assert not any(
            row.account_incarnation_id == incarnation
            for row in restore_result.capability_evidence.values()
        )
    finally:
        inspect_store.close()




# ---------------------------------------------------------------------------
# Balanced usage poll coordinator, refresh isolation, mode-safe usage API
# (T-13): budget/fairness/cooldown/anti-starvation fake-clock unit tests
# (Step 7), plus HTTP-level balanced/fallback/disabled endpoint tests (Steps
# 4-6, 8-9).
# ---------------------------------------------------------------------------


class _FakeMonotonicClock:
    """A controllable stand-in for `time.monotonic` (and `time.time`) --
    no real sleeping. Reused as both `clock` and `wall_clock` below: the
    coordinator's tests only need consistent, deterministic numbers, not a
    realistic monotonic/wall split.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeUsagePollFetch:
    """Records fetch order and replays per-account queued responses -- the
    coordinator's own fake stand-in for the real per-account usage probe.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses: dict[str, list[tuple[dict[str, Any], float | None]]] = {}

    def queue(self, account_id: str, result: dict[str, Any], retry_after: float | None = None) -> None:
        self.responses.setdefault(account_id, []).append((result, retry_after))

    async def __call__(self, account_id: str) -> tuple[dict[str, Any], float | None]:
        self.calls.append(account_id)
        return self.responses[account_id].pop(0)


def _fake_usage_ok(*, session_percent: float = 10.0, weekly_percent: float = 20.0) -> dict[str, Any]:
    return {
        "provider": "claude",
        "status": "ok",
        "error": None,
        "session": {"used_percent": session_percent, "resets_at": None},
        "weekly": {"used_percent": weekly_percent, "resets_at": None},
        "fable_weekly": None,
    }


def _fake_usage_err(message: str) -> dict[str, Any]:
    return {"provider": "claude", "status": "error", "error": message, "session": None}


class _RecordingUsageObservationStore:
    """Duck-typed stand-in for `ClaudePoolRuntimeStateStore.upsert_usage_observation`
    -- records what the coordinator submits, with none of the real store's
    background-thread persistence timing.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upsert_usage_observation(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _make_usage_poll_coordinator(
    *, clock: _FakeMonotonicClock, fetch: _FakeUsagePollFetch, store: Any = None
) -> tuple[ClaudeUsagePollCoordinator, ClaudeAccountUsageCache, ClaudeBalancedRouter]:
    cache = ClaudeAccountUsageCache(fetch, clock=clock)
    router = ClaudeBalancedRouter(balanced_epoch_id="epoch-1", clock=clock, wall_clock=clock)
    coordinator = ClaudeUsagePollCoordinator(
        cache=cache,
        router=router,
        store=store,
        clock=clock,
        wall_clock=clock,
        poll_interval_seconds=30.0,
    )
    return coordinator, cache, router


class TestUsagePollCoordinatorScheduling:
    """Fake-clock coordinator unit tests (T-13 Step 7): budget enforcement,
    missing-window priority, fairness after failure, global cooldown, manual
    coalescing/rate limiting, and full-sweep progress under manual refreshes.
    """

    def test_coordinator_enforces_the_actual_call_budget(self) -> None:
        # Two DIFFERENT accounts, both never observed -- this isolates the
        # coordinator's test-pinned 30s call budget from the cache's own (120s,
        # unrelated) per-account TTL: "b" stays due the whole time, so its
        # own due-ness can never explain a budget_wait here.
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok())
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        first = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        second = asyncio.run(coordinator.run_due_poll(["a", "b"]))

        assert first.outcome == "fetched"
        assert first.account_id == "a"
        assert second.outcome == "budget_wait"
        assert fetch.calls == ["a"]

        clock.advance(30.0)
        fetch.queue("b", _fake_usage_ok())
        third = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert third.outcome == "fetched"
        assert third.account_id == "b"
        assert fetch.calls == ["a", "b"]

    def test_missing_window_accounts_are_serviced_before_already_observed_ones(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("warm", _fake_usage_ok())
        coordinator, cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        asyncio.run(cache.get(["warm"]))  # seeds "warm" with a fresh observation
        fetch.queue("cold", _fake_usage_ok())

        tick = asyncio.run(coordinator.run_due_poll(["warm", "cold"]))

        assert tick.outcome == "fetched"
        assert tick.account_id == "cold"

    def test_a_failing_account_does_not_starve_the_rest_of_the_pool(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        for _ in range(4):
            fetch.queue("a", _fake_usage_err("usage API returned 500: boom"))
        fetch.queue("b", _fake_usage_ok())
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        serviced: list[str | None] = []
        for _ in range(4):
            serviced.append(asyncio.run(coordinator.run_due_poll(["a", "b"])).account_id)
            clock.advance(30.0)

        assert "b" in serviced
        assert fetch.calls.count("b") == 1
        assert fetch.calls.count("a") >= 1

    def test_global_cooldown_is_reported_and_blocks_the_whole_tick(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_err("usage API rate-limited (429); try again shortly"), 45.0)
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        first = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert first.outcome == "fetched"  # a's own call opened the cooldown

        clock.advance(30.0)
        second = asyncio.run(coordinator.run_due_poll(["a", "b"]))
        assert second.outcome == "cooldown"
        assert fetch.calls == ["a"]

    def test_manual_refresh_coalesces_per_account_and_is_globally_rate_limited(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        assert coordinator.request_manual_refresh("a") is True
        assert coordinator.request_manual_refresh("a") is True  # coalesced, not a new enqueue
        assert coordinator.diagnostics().manual_enqueued_count == 1

        assert coordinator.request_manual_refresh("b") is False  # globally rate-limited
        assert coordinator.diagnostics().manual_rate_limited_count == 1

        clock.advance(300.0)
        assert coordinator.request_manual_refresh("b") is True
        assert coordinator.diagnostics().manual_enqueued_count == 2

    def test_manual_refresh_never_fetches_inline(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        assert coordinator.request_manual_refresh("a") is True

        assert fetch.calls == []
        assert coordinator.is_manual_refresh_pending("a") is True

    def test_manual_refresh_only_consumes_a_slot_after_an_automatic_account_is_serviced(
        self,
    ) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("auto", _fake_usage_ok())
        coordinator, cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        asyncio.run(cache.get(["manual_target"]))  # already fresh -- never automatically due
        coordinator.request_manual_refresh("manual_target")

        # Nothing automatic has been serviced by THIS coordinator yet -- the
        # very first tick must claim the one genuinely due account ("auto"),
        # not the pending manual one.
        first = asyncio.run(coordinator.run_due_poll(["auto", "manual_target"]))
        assert first.outcome == "fetched"
        assert first.account_id == "auto"
        assert first.manual is False

        # Now that an automatic account has been serviced, and nothing else
        # is due (both accounts are within their own TTL), the next tick
        # claims the pending manual one -- forcing a fresh fetch despite it.
        fetch.queue("manual_target", _fake_usage_ok())
        clock.advance(30.0)
        second = asyncio.run(coordinator.run_due_poll(["auto", "manual_target"]))
        assert second.outcome == "fetched"
        assert second.account_id == "manual_target"
        assert second.manual is True
        assert coordinator.is_manual_refresh_pending("manual_target") is False

    def test_full_sweep_still_makes_progress_under_repeated_manual_refresh_requests(
        self,
    ) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)
        accounts = ["a", "b", "c"]
        for account_id in accounts:
            fetch.queue(account_id, _fake_usage_ok())
        coordinator.request_manual_refresh("a")

        # Exactly one tick per account -- within the cache's own 120s TTL, so
        # nothing here is re-fetched merely because it aged out.
        for _ in range(len(accounts)):
            asyncio.run(coordinator.run_due_poll(accounts))
            clock.advance(30.0)
            coordinator.request_manual_refresh("b")  # rate-limited every time; harmless either way

        # The full automatic sweep completed -- every ready account actually
        # got polled, not just whichever one a manual request kept naming.
        assert set(fetch.calls) == set(accounts)
        assert coordinator.diagnostics().fetched_count == len(accounts)

    def test_a_successful_fetch_feeds_the_router_observation_view(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok(session_percent=42.0, weekly_percent=17.0))
        coordinator, _cache, router = _make_usage_poll_coordinator(clock=clock, fetch=fetch)

        tick = asyncio.run(coordinator.run_due_poll(["a"]))

        assert tick.outcome == "fetched"
        assert router.observations.window_pressure("a", "five_hour", now=clock.now) == 42.0
        assert router.observations.window_pressure("a", "seven_day", now=clock.now) == 17.0

    def test_a_successful_fetch_persists_a_durable_usage_observation_row(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok(session_percent=55.0, weekly_percent=12.0))
        store = _RecordingUsageObservationStore()
        coordinator, _cache, _router = _make_usage_poll_coordinator(clock=clock, fetch=fetch, store=store)
        accounts = {
            "a": UsagePollAccount(
                account_id="a", account_incarnation_id="inc-1", account_profile_fingerprint="fp-1"
            )
        }

        tick = asyncio.run(coordinator.run_due_poll(["a"], accounts=accounts))

        assert tick.outcome == "fetched"
        by_window = {call["window"]: call for call in store.calls}
        assert by_window["five_hour"]["used_percent"] == 55.0
        assert by_window["five_hour"]["account_incarnation_id"] == "inc-1"
        assert by_window["five_hour"]["account_profile_fingerprint"] == "fp-1"
        assert by_window["seven_day"]["used_percent"] == 12.0

    def test_durable_persistence_is_skipped_without_a_profile_fingerprint(self) -> None:
        clock = _FakeMonotonicClock()
        fetch = _FakeUsagePollFetch()
        fetch.queue("a", _fake_usage_ok())
        store = _RecordingUsageObservationStore()
        coordinator, _cache, router = _make_usage_poll_coordinator(clock=clock, fetch=fetch, store=store)
        accounts = {
            "a": UsagePollAccount(
                account_id="a", account_incarnation_id="inc-1", account_profile_fingerprint=None
            )
        }

        asyncio.run(coordinator.run_due_poll(["a"], accounts=accounts))

        assert store.calls == []
        # The router's in-memory observation view is still fed regardless.
        assert router.observations.window_pressure("a", "five_hour", now=clock.now) is not None


class TestBalancedUsageCoordinatorEndpoints:
    """HTTP-level coverage (T-13 Steps 4-6, 8-9): active balanced usage reads
    are cache-only and never call upstream, manual refresh returns `queued`
    without fetching inline, and fallback/disabled mode keeps the pre-existing
    fetch path/envelope and never surfaces a stranded `queued` indication.
    """

    def test_pool_usage_active_balanced_mode_never_calls_upstream(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        # T-18: the background driver already warmed this account up with one
        # automatic poll the moment balanced routing activated -- this
        # endpoint's OWN read is still cache-only and adds no second call.
        assert payload["accounts"][account_id]["status"] == "ok"
        assert payload["accounts"][account_id]["windows"] == {}
        assert payload["accounts"][account_id]["queued"] is False
        assert calls == [account_id]

    def test_manual_refresh_returns_queued_without_fetching_in_active_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler)
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": account_id, "refresh": "1"},
            )
            assert runtime.usage_poll_coordinator is not None
            pending = runtime.usage_poll_coordinator.is_manual_refresh_pending(account_id)

        assert response.status_code == 200
        payload = response.json()
        assert payload["queued"] is True
        assert payload["accounts"][account_id]["queued"] is True
        assert pending is True
        # T-18: the background driver's own automatic warm-up poll is the
        # only call recorded -- `?refresh` itself never fetches inline, it
        # only enqueues the manual refresh the driver will service later.
        assert calls == [account_id]

    def test_manual_refresh_is_inert_outside_active_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        with client:
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": account_id, "refresh": "1"},
            )

        assert response.status_code == 200
        payload = response.json()
        assert "queued" not in payload
        assert "queued" not in payload["accounts"][account_id]

    def test_pool_usage_disabled_mode_preserves_the_existing_envelope_and_fetch_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"accounts", "fetched_at"}
        assert payload["accounts"][account_id] == {
            "provider": "claude",
            "status": "ok",
            "error": None,
        }
        assert calls == [account_id]  # an equivalent uncached read DOES hit upstream

    def test_pool_usage_fallback_mode_preserves_the_existing_envelope_and_fetch_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(
                settings_file=tmp_path / "settings.json", claude_account_routing_mode="fallback"
            ),
            base_url=_ADMIN_BASE,
        )
        account_id = _register_serving_account()
        calls: list[str] = []

        async def spy_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", spy_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"accounts", "fetched_at"}
        envelope = payload["accounts"][account_id]
        assert "queued" not in envelope
        assert "windows" not in envelope
        assert calls == [account_id]

    def test_pool_status_usage_freshness_is_none_outside_balanced_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )
        _register_serving_account()

        with client:
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] is None
        assert payload["usage_diagnostics"] is None

    def test_pool_status_reports_degraded_usage_freshness_before_any_observation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "degraded"
        assert payload["usage_diagnostics"]["persistence_degraded"] is False
        # T-18: the background driver already made its one automatic warm-up
        # attempt (the mock transport's no-window "ok" stub, same as every
        # other `_enable_balanced` caller) -- still "degraded" since it
        # carried no actual window data, but no longer the pre-driver "never
        # even tried" baseline of 0.
        assert payload["usage_diagnostics"]["coordinator"]["fetched_count"] == 1

    def test_pool_status_reports_fresh_usage_freshness_once_every_window_is_recent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": {"used_percent": 5.0, "resets_at": None},
                    "fable_weekly": None,
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            # Populate the cache through the ordinary (still-disabled-mode)
            # fetch path first -- the SAME cache instance survives the mode
            # switch below, so this seeds a fresh observation without a live
            # poll coordinator loop.
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "fresh"

    def test_pool_status_reports_partial_usage_freshness_when_a_binding_window_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """G-2 regression: `fresh` requires BOTH binding windows (`session`
        AND `weekly`) present -- an account with only `session` recently
        observed, and `weekly` never observed, must not count as fresh even
        though its one observed window is well within the 5-minute bound.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": None,
                    "fable_weekly": None,
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "partial"

    def test_pool_status_reports_partial_usage_freshness_when_fable_weekly_present_but_weekly_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """G-2 regression: `fable_weekly` is a scoped, Fable-only extra
        window and must never substitute for a missing `weekly` binding
        window -- an account with a fresh `session` and a fresh
        `fable_weekly`, but no `weekly`, still does not count as fresh.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id = _register_balanced_ready_account()

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            return (
                {
                    "provider": "claude",
                    "status": "ok",
                    "error": None,
                    "session": {"used_percent": 10.0, "resets_at": None},
                    "weekly": None,
                    "fable_weekly": {"used_percent": 1.0, "resets_at": None},
                },
                None,
            )

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("no /v1/messages traffic in this test")

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            seeded = client.get("/admin/providers/claude/pool/usage")
            assert seeded.json()["accounts"][account_id]["status"] == "ok"

            _enable_balanced(client, handler)
            response = client.get("/admin/providers/claude/pool/status")

        payload = response.json()
        assert payload["usage_freshness"] == "partial"


# ---------------------------------------------------------------------------
# Usage poll driver (T-18, fix for gap G-1): the background task that
# actually calls `ClaudeUsagePollCoordinator.run_due_poll` while balanced
# routing is active -- no production code path ever did before this.
# ---------------------------------------------------------------------------


def _wait_until(client: TestClient, predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    """Poll `predicate`, giving the app's background event loop another turn
    (a real `GET /health` round trip) between checks, until it holds or
    `timeout` elapses. Avoids a fixed real-time `sleep` in the test itself
    while staying deterministic: a working driver satisfies `predicate`
    almost immediately, and a genuinely broken/cancelled one fails fast
    instead of hanging until the timeout.
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met within timeout")
        client.get("/health")


class TestUsagePollDriver:
    """T-18 (fix for gap G-1): `ClaudeBalancedRuntime` now actually drives
    `usage_poll_coordinator.run_due_poll` itself -- automatically once
    balanced routing activates, cancelled on exit/shutdown strictly before
    the store closes, and eventually draining a queued manual refresh --
    instead of leaving the coordinator dormant outside of tests.
    """

    def test_usage_poll_driver_starts_automatically_and_polls_without_any_manual_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Merely activating balanced routing starts the driver: it performs
        one automatic upstream poll entirely on its own -- this test never
        calls `run_due_poll` or requests a manual refresh itself.
        """
        _balanced_env(monkeypatch, tmp_path)
        account_id, _access_token = _register_balanced_accounts(1)[0]
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != _USAGE_PROBE_PATH:
                raise AssertionError("no /v1/messages traffic in this test")
            calls.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "five_hour": {"utilization": 42.0, "resets_at": None},
                    "seven_day": {"utilization": 17.0, "resets_at": None},
                },
            )

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            _enable_balanced(client, handler, intercept_usage_probe=False)
            _wait_until(client, lambda: len(calls) >= 1)

            response = client.get("/admin/providers/claude/pool/usage")

        assert calls == [_USAGE_PROBE_PATH]
        payload = response.json()
        assert payload["accounts"][account_id]["status"] == "ok"
        assert payload["accounts"][account_id]["windows"]["session"]["state"] == "fresh"
        assert payload["accounts"][account_id]["windows"]["weekly"]["state"] == "fresh"

    def test_usage_poll_driver_is_cancelled_by_an_intentional_exit_and_polls_no_further(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Step 2: disabling balanced routing (`exit_mode`) cancels+awaits the
        driver strictly before the store closes -- no further automatic poll
        happens afterward, even though a short scheduling budget would
        otherwise have allowed several more within this test's own
        wall-clock time.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={})

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01
            _wait_until(client, lambda: len(calls) >= 1)
            assert calls == [_USAGE_PROBE_PATH]

            disabled = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            assert disabled.status_code == 200
            assert runtime.status == "disabled"
            assert runtime._usage_poll_driver_task is None

            # Several more short-budget ticks would fit in this window if the
            # driver were still running -- none did.
            client.get("/health")
            client.get("/health")
            assert calls == [_USAGE_PROBE_PATH]

    def test_usage_poll_driver_is_cancelled_on_lifespan_shutdown_and_polls_no_further(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Step 2: process shutdown (`shutdown_preserving_epoch`) gives the
        same cancel-before-close guarantee as an intentional exit, via the
        lifespan's own teardown path instead of an explicit disable.
        """
        _balanced_env(monkeypatch, tmp_path)
        _register_balanced_accounts(1)
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(200, json={})

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01
            _wait_until(client, lambda: len(calls) >= 1)
            assert calls == [_USAGE_PROBE_PATH]
            assert runtime.status == "active"

        # The `with` block's exit ran `shutdown_preserving_epoch` to
        # completion -- the driver is cancelled+awaited and the store closed
        # before it returned; no additional automatic poll occurred despite
        # the short scheduling budget.
        assert runtime.status == "disabled"
        assert runtime._usage_poll_driver_task is None
        assert calls == [_USAGE_PROBE_PATH]

    def test_usage_poll_driver_drains_a_pending_manual_refresh_without_an_extra_upstream_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A manual refresh (T-13 Step 5) never fetches inline -- it is only
        ever consumed by THIS driver's own next tick, once every automatic
        candidate has yielded no fetch that tick. Total calls stay exactly
        one per account attempt: the pre-seed, the automatic poll, and the
        one deliberate manual-forced poll -- never a redundant extra one.
        """
        _balanced_env(monkeypatch, tmp_path)
        (auto_id, auto_token), (manual_id, manual_token) = _register_balanced_accounts(2)
        auto_bearer = f"Bearer {auto_token}"
        manual_bearer = f"Bearer {manual_token}"
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path != _USAGE_PROBE_PATH:
                raise AssertionError("no /v1/messages traffic in this test")
            calls.append(request.headers["authorization"])
            return httpx.Response(
                200,
                json={
                    "five_hour": {"utilization": 10.0, "resets_at": None},
                    "seven_day": {"utilization": 5.0, "resets_at": None},
                },
            )

        with _create_test_client(
            monkeypatch, tmp_path, config=GatewayConfig(), base_url="http://127.0.0.1:8787"
        ) as client:
            client.app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            # Pre-seed the manual-target account as already fresh (still in
            # the pre-existing, ordinary fetch-through mode) -- it is never
            # automatically due once balanced routing activates below.
            seeded = client.get(
                "/admin/providers/claude/pool/usage", params={"account": manual_id}
            )
            assert seeded.json()["accounts"][manual_id]["status"] == "ok"
            assert calls == [manual_bearer]

            runtime = _enable_balanced(client, handler, intercept_usage_probe=False)
            assert runtime.usage_poll_coordinator is not None
            runtime.usage_poll_coordinator._poll_interval_seconds = 0.01

            # Tick 1: the only genuinely due account is `auto_id`.
            _wait_until(client, lambda: len(calls) >= 2)
            assert calls == [manual_bearer, auto_bearer]

            runtime.usage_poll_coordinator.request_manual_refresh(manual_id)
            # Tick 2+: nothing automatic is due anymore -- the driver drains
            # the pending manual refresh instead.
            _wait_until(client, lambda: len(calls) >= 3)
            assert runtime.usage_poll_coordinator.is_manual_refresh_pending(manual_id) is False

        assert calls == [manual_bearer, auto_bearer, manual_bearer]
