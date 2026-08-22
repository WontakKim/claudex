"""Integration tests for the gateway admin API routes."""

from __future__ import annotations

import ast
import asyncio
import importlib.resources
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

import claudex.admin.accounts as admin_accounts
import claudex.admin.common as admin_common
import claudex.admin.settings as admin_settings
import claudex.admin.system as admin_system
import claudex.server as server
import claudex.server_support as server_support
from claudex import compaction, paths
from claudex.claude import accounts as claude_accounts
from claudex.providers.codex_client import (
    CODEX_MODELS_URL,
    CODEX_RESPONSES_URL,
    CodexClient,
    CodexUpstreamError,
)
from claudex.config import (
    AnthropicCompatibleProvider,
    ConfigError,
    GatewayConfig,
    OpenAICompatibleProvider,
)
from claudex.providers.anthropic_compatible_client import AnthropicCompatibleClient
from claudex.providers.backends import AnthropicBackend, ResponsesBackend
from claudex.providers.grok_auth import GrokCredentials
from claudex.providers.grok_client import GrokClient, GrokUpstreamError
from claudex.providers.kimi_auth import KimiCredentials
from claudex.providers.kimi_client import KimiClient, KimiUpstreamError
from claudex.providers.openai_compatible_client import (
    OpenAICompatibleClient,
    OpenAICompatibleUpstreamError,
)


_ADMIN_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "claudex" / "admin"
_ADMIN_MODULE_NAMES = ("common", "settings", "accounts", "system")
_ADMIN_MODULE_PATHS = {
    "common": "claudex.admin.common",
    "settings": "claudex.admin.settings",
    "accounts": "claudex.admin.accounts",
    "system": "claudex.admin.system",
}
_ADMIN_IMPORTED_MODULES = {
    "common": admin_common,
    "settings": admin_settings,
    "accounts": admin_accounts,
    "system": admin_system,
}
_ADMIN_FUNCTION_MANIFEST = {
    "common": {
        "_get_custom_provider_binding",
        "_redact_configured_credential",
        "_safe_custom_provider_exception_detail",
        "_safe_custom_provider_upstream_detail",
        "_safe_route_target_error_detail",
        "_read_json_object",
        "_handle_hello",
        "_handle_health",
        "_admin_guard",
        "_require_json_content_type",
    },
    "settings": {
        "_mapping_payload",
        "_handle_admin_mapping_get",
        "_handle_admin_mapping_put",
        "_apply_log_level",
        "_log_level_payload",
        "_handle_admin_log_level_get",
        "_handle_admin_log_level_put",
        "_compaction_payload",
        "_handle_admin_compaction_get",
        "_handle_admin_compaction_put",
        "_codex_payload",
        "_handle_admin_codex_get",
        "_handle_admin_codex_put",
        "_claude_account_payload",
        "_handle_admin_claude_serving_get",
        "_claude_account_env_locked",
        "_handle_admin_claude_serving_put",
        "_handle_admin_claude_serving_delete",
        "_claude_routing_payload",
        "_handle_admin_claude_routing_get",
        "_persist_claude_routing_mode",
        "_handle_admin_claude_routing_put",
    },
    "accounts": {
        "_active_balanced_runtime",
        "_usage_window_state",
        "_compute_usage_freshness",
        "_handle_admin_claude_pool_status",
        "_require_login_attempt",
        "_local_claude_login_fields",
        "_account_plan_fields",
        "_handle_admin_claude_accounts_get",
        "_handle_admin_claude_local_get",
        "_handle_admin_claude_account_delete",
        "_cooling_down_until_millis",
        "_handle_admin_claude_accounts_usage",
        "_handle_admin_claude_login_get",
        "_handle_admin_claude_login_post",
        "_handle_admin_claude_login_code_post",
        "_handle_admin_claude_login_replace_post",
        "_handle_admin_claude_login_delete",
    },
    "system": {
        "_handle_admin_logs",
        "_handle_admin_usage",
        "_handle_admin_codex_reset_credit",
        "_serve_dashboard_asset",
        "_handle_dashboard",
        "_handle_dashboard_css",
        "_handle_dashboard_js",
        "_handle_favicon",
        "_handle_admin_codex_models",
        "_handle_admin_grok_models",
        "_handle_admin_custom_models",
        "_handle_admin_kimi_models",
        "_get_route_backend",
        "_probe_responses_route",
        "_probe_anthropic_route",
        "_handle_admin_connection_test",
    },
}
_ADMIN_CONSTANT_MANIFEST = {
    "settings": {
        "_ADMIN_MAP_KEYS",
        "_LOG_LEVEL_LOGGER_NAMES",
        "_COMPACTION_KEYS",
        "_CODEX_KEYS",
        "_CLAUDE_ACCOUNT_KEYS",
        "_CLAUDE_ROUTING_KEYS",
    },
    "accounts": {
        "_USAGE_WINDOW_FRESH_MAX_AGE_SECONDS",
        "_USAGE_WINDOW_AGING_MAX_AGE_SECONDS",
        "_USAGE_FRESHNESS_BINDING_WINDOWS",
        "_CLAUDE_LOGIN_CODE_KEYS",
        "_CLAUDE_LOGIN_REPLACE_KEYS",
        "_LOGIN_ATTEMPT_HEADER",
        "_CONTROL_CHARACTER_PATTERN",
    },
    "system": {
        "_STATUS_TO_OPENAI_ERROR_TYPE",
        "_FAVICON_SVG",
        "_CONNECTION_TEST_TIMEOUT",
    },
}


def _admin_module_trees() -> dict[str, ast.Module]:
    return {
        module_name: ast.parse(
            (_ADMIN_SOURCE_ROOT / f"{module_name}.py").read_text(encoding="utf-8")
        )
        for module_name in _ADMIN_MODULE_NAMES
    }


def _admin_top_level_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _admin_top_level_assignments(tree: ast.Module) -> set[str]:
    definitions: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            definitions.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions.add(node.target.id)
    return definitions


def _admin_sibling_dependencies(tree: ast.Module) -> set[str]:
    package_name = "claudex.admin"
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported_base = node.module or ""
            if imported_base == package_name:
                dependencies.update(
                    alias.name for alias in node.names if alias.name in _ADMIN_MODULE_NAMES
                )
            imported_modules = [imported_base]
        else:
            continue
        for imported_module in imported_modules:
            prefix = f"{package_name}."
            if not imported_module.startswith(prefix):
                continue
            sibling_name = imported_module.removeprefix(prefix).partition(".")[0]
            if sibling_name in _ADMIN_MODULE_NAMES:
                dependencies.add(sibling_name)
    return dependencies


def _admin_imported_modules(tree: ast.Module) -> set[str]:
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    return imported_modules


def _assert_admin_imports_are_used(module_name: str, tree: ast.Module) -> None:
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases = [
                (alias.asname or alias.name.split(".")[0], alias.name)
                for alias in node.names
            ]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            aliases = [(alias.asname or alias.name, alias.name) for alias in node.names]
        else:
            continue
        for local_name, imported_name in aliases:
            assert imported_name != "*", f"{module_name}: wildcard import"
            assert local_name in loaded_names, f"{module_name}: unused import {imported_name}"


def test_admin_symbols_have_one_canonical_owner() -> None:
    trees = _admin_module_trees()
    function_definitions = {
        module_name: _admin_top_level_functions(tree)
        for module_name, tree in trees.items()
    }
    assignment_definitions = {
        module_name: _admin_top_level_assignments(tree)
        for module_name, tree in trees.items()
    }

    assert function_definitions == _ADMIN_FUNCTION_MANIFEST
    for expected_owner, symbols in _ADMIN_CONSTANT_MANIFEST.items():
        for symbol in symbols:
            actual_owners = [
                module_name
                for module_name, definitions in assignment_definitions.items()
                if symbol in definitions
            ]
            assert actual_owners == [expected_owner], (
                symbol,
                expected_owner,
                actual_owners,
            )

    init_tree = ast.parse((_ADMIN_SOURCE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    init_body = list(init_tree.body)
    if (
        init_body
        and isinstance(init_body[0], ast.Expr)
        and isinstance(init_body[0].value, ast.Constant)
        and isinstance(init_body[0].value.value, str)
    ):
        init_body.pop(0)
    assert not init_body
    assert {
        module_name: module.__name__
        for module_name, module in _ADMIN_IMPORTED_MODULES.items()
    } == _ADMIN_MODULE_PATHS


def test_admin_import_inventory_and_dependency_directions() -> None:
    trees = _admin_module_trees()
    edges: dict[str, set[str]] = {}
    for module_name, tree in trees.items():
        _assert_admin_imports_are_used(module_name, tree)
        edges[module_name] = _admin_sibling_dependencies(tree)
        assert "claudex.server" not in _admin_imported_modules(tree)

    assert edges == {
        "common": set(),
        "settings": {"common"},
        "accounts": {"common"},
        "system": {"common"},
    }


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


def _assert_secret_absent(secret: str, *values: str) -> None:
    if any(secret in value for value in values):
        pytest.fail("a configured custom-provider credential was exposed")


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
    custom_client: type = FakeOpenAICompatibleClient,
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
    monkeypatch.setattr(server, "OpenAICompatibleClient", custom_client)
    return TestClient(server.create_app(config or GatewayConfig()), base_url=base_url)


def test_health_reports_ok_with_codex_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "providers": {
            "codex": {
                "status": "ok",
                "auth_mode": "chatgpt",
                "account": "account",
                "email": "codex@example.com",
            },
            "kimi": {"status": "ok", "required": False, "account": "kimi-user-1"},
            "grok": {
                "status": "ok",
                "required": False,
                "auth_mode": "oauth",
                "account": "user@example.com",
            },
        },
    }


def test_health_reports_error_without_codex_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, codex_auth=MissingCodexAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["codex"]["status"] == "error"


def test_health_stays_ok_without_kimi_credentials_when_map_has_no_kimi_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, kimi_auth=MissingKimiAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is False


def test_health_reports_error_without_kimi_credentials_when_map_routes_to_kimi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "kimi:k2.5"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, kimi_auth=MissingKimiAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["kimi"]["status"] == "error"
    assert health.json()["providers"]["kimi"]["required"] is True


def test_health_stays_ok_without_grok_credentials_when_map_has_no_grok_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path, grok_auth=MissingGrokAuthManager) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is False


def test_health_reports_error_without_grok_credentials_when_map_routes_to_grok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(model_map={"opus": "grok:grok-4.5"})
    with _create_test_client(
        monkeypatch, tmp_path, config=config, grok_auth=MissingGrokAuthManager
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["grok"]["status"] == "error"
    assert health.json()["providers"]["grok"]["required"] is True


def test_health_reports_ready_custom_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["providers"]["wrtn"] == {
        "status": "ok",
        "required": True,
    }


def test_health_reports_error_when_required_custom_provider_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=FailingOpenAICompatibleClient,
    ) as client:
        health = client.get("/health")

    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"]["wrtn"]["status"] == "error"
    assert health.json()["providers"]["wrtn"]["required"] is True


def test_health_stays_ready_when_unrequired_custom_provider_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        custom_client=FailingOpenAICompatibleClient,
    ) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["providers"]["wrtn"]["status"] == "error"
    assert health.json()["providers"]["wrtn"]["required"] is False


def test_health_uses_selected_route_transport_for_custom_provider_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:gpt-5.5"},
        custom_providers={"wrtn": _custom_provider()},
    )
    selected_transport = SelectedResponsesModelTransport(["gpt-5.5"])
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        _select_route_transport(client, "wrtn", selected_transport)
        client.app.state.custom_provider_clients["wrtn"] = (
            FailingOpenAICompatibleClient("wrtn", _custom_provider())
        )
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["providers"]["wrtn"] == {
        "status": "ok",
        "required": True,
    }
    assert selected_transport.calls == 1


def test_health_without_custom_providers_keeps_builtin_provider_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        health = client.get("/health")

    assert list(health.json()["providers"]) == ["codex", "kimi", "grok"]


def test_anthropic_custom_provider_health_and_missing_catalog_perform_no_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request_count = 0
    async_client_type = httpx.AsyncClient

    async def count_request(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    transport = httpx.MockTransport(count_request)

    def create_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return async_client_type(*args, **kwargs)

    monkeypatch.setattr(server.httpx, "AsyncClient", create_async_client)
    config = GatewayConfig(
        model_map={"opus": "messages-api:claude-upstream"},
        custom_providers={"messages-api": _anthropic_custom_provider()},
    )
    with _create_test_client(
        monkeypatch,
        tmp_path,
        config=config,
        base_url="http://127.0.0.1:8787",
    ) as client:
        catalog = client.get("/admin/providers/custom/messages-api/models")
        health = client.get("/health")

    assert catalog.status_code == 404
    assert catalog.json()["error"]["type"] == "not_found_error"
    assert health.status_code == 200
    assert health.json()["providers"]["messages-api"] == {
        "status": "ok",
        "required": True,
    }
    assert request_count == 0


@pytest.mark.parametrize(
    ("name", "provider", "wrong_backend_name"),
    [
        pytest.param("wrtn", _custom_provider(), "kimi", id="openai-compatible"),
        pytest.param(
            "messages-api",
            _anthropic_custom_provider(),
            "codex",
            id="anthropic-compatible",
        ),
    ],
)
@pytest.mark.parametrize("binding_state", ["missing", "mismatched"])
def test_health_fails_closed_for_invalid_required_custom_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    provider: OpenAICompatibleProvider | AnthropicCompatibleProvider,
    wrong_backend_name: str,
    binding_state: str,
) -> None:
    forbidden_catalog_calls = 0

    async def forbidden_catalog() -> list[str]:
        nonlocal forbidden_catalog_calls
        forbidden_catalog_calls += 1
        return []

    config = GatewayConfig(
        model_map={"opus": f"{name}:upstream-model"},
        custom_providers={name: provider},
    )
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        if binding_state == "missing":
            client.app.state.route_backends.pop(name)
        else:
            wrong_backend = client.app.state.route_backends[wrong_backend_name]
            client.app.state.route_backends[name] = replace(
                wrong_backend, catalog_loader=forbidden_catalog
            )
        health = client.get("/health")

    _assert_secret_absent(provider.api_key, health.text)
    assert health.status_code == 503
    assert health.json()["status"] == "error"
    assert health.json()["providers"][name] == {
        "status": "error",
        "detail": (
            "custom provider binding is missing"
            if binding_state == "missing"
            else "custom provider binding does not match its configured family"
        ),
        "required": True,
    }
    assert forbidden_catalog_calls == 0


def test_health_redacts_secret_bearing_http_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig(
        model_map={"opus": "wrtn:upstream-model"},
        custom_providers={"wrtn": _custom_provider()},
    )
    selected_transport = SecretBearingCatalogTransport(_CUSTOM_API_KEY)
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        _select_route_transport(client, "wrtn", selected_transport)
        health = client.get("/health")

    _assert_secret_absent(_CUSTOM_API_KEY, health.text, caplog.text)
    assert health.status_code == 503
    assert health.json()["providers"]["wrtn"]["status"] == "error"
    assert "ConnectError" in health.json()["providers"]["wrtn"]["detail"]
    assert "[REDACTED]" in health.json()["providers"]["wrtn"]["detail"]
    assert selected_transport.calls == 1


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

    def test_get_returns_current_mapping(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, model_map={"haiku": "codex:gpt-5.6-luna"}
        ) as client:
            response = client.get("/admin/settings/mapping")

        assert response.status_code == 200
        assert response.json()["model_map"] == {"haiku": "codex:gpt-5.6-luna"}

    def test_get_includes_custom_provider_metadata_without_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch,
            tmp_path,
            custom_providers={"wrtn": _custom_provider()},
        ) as client:
            response = client.get("/admin/settings/mapping")

        _assert_secret_absent(_CUSTOM_API_KEY, response.text)
        assert response.status_code == 200
        assert response.json()["custom_providers"] == [
            {
                "name": "wrtn",
                "family": "openai_compatible",
                "wire_kind": "responses",
                "base_url": "https://models.example/api/v1",
                "catalog_available": True,
            }
        ]
        assert "api_key" not in response.text

    def test_get_includes_deterministic_mixed_family_metadata_without_api_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        responses_provider = OpenAICompatibleProvider(
            wire_api="responses",
            base_url="https://responses.example/v1",
            api_key=_CUSTOM_API_KEY,
        )
        messages_provider = AnthropicCompatibleProvider(
            base_url="https://messages.example/v1",
            api_key=_ANTHROPIC_CUSTOM_API_KEY,
        )
        with self._admin_client(
            monkeypatch,
            tmp_path,
            custom_providers={
                "z-responses": responses_provider,
                "a-messages": messages_provider,
            },
        ) as client:
            response = client.get("/admin/settings/mapping")

        _assert_secret_absent(
            _CUSTOM_API_KEY, response.text, repr(responses_provider)
        )
        _assert_secret_absent(
            _ANTHROPIC_CUSTOM_API_KEY, response.text, repr(messages_provider)
        )
        assert response.status_code == 200
        assert response.json()["custom_providers"] == [
            {
                "name": "a-messages",
                "family": "anthropic_compatible",
                "wire_kind": "anthropic_messages",
                "base_url": "https://messages.example/v1",
                "catalog_available": False,
            },
            {
                "name": "z-responses",
                "family": "openai_compatible",
                "wire_kind": "responses",
                "base_url": "https://responses.example/v1",
                "catalog_available": True,
            },
        ]
        assert "api_key" not in response.text

    @pytest.mark.parametrize(
        ("name", "provider", "wrong_backend_name"),
        [
            pytest.param("wrtn", _custom_provider(), "kimi", id="openai-compatible"),
            pytest.param(
                "messages-api",
                _anthropic_custom_provider(),
                "codex",
                id="anthropic-compatible",
            ),
        ],
    )
    @pytest.mark.parametrize("binding_state", ["missing", "mismatched"])
    def test_get_rejects_invalid_custom_provider_binding_without_io(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        name: str,
        provider: OpenAICompatibleProvider | AnthropicCompatibleProvider,
        wrong_backend_name: str,
        binding_state: str,
    ) -> None:
        forbidden_catalog_calls = 0

        async def forbidden_catalog() -> list[str]:
            nonlocal forbidden_catalog_calls
            forbidden_catalog_calls += 1
            return []

        with self._admin_client(
            monkeypatch, tmp_path, custom_providers={name: provider}
        ) as client:
            if binding_state == "missing":
                client.app.state.route_backends.pop(name)
            else:
                wrong_backend = client.app.state.route_backends[wrong_backend_name]
                client.app.state.route_backends[name] = replace(
                    wrong_backend, catalog_loader=forbidden_catalog
                )
            response = client.get("/admin/settings/mapping")

        _assert_secret_absent(provider.api_key, response.text)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"
        assert (
            response.json()["error"]["message"]
            == "custom provider metadata is unavailable"
        )
        assert forbidden_catalog_calls == 0

    @pytest.mark.parametrize("overlap_field", ["name", "base_url"])
    def test_get_fails_closed_when_openai_credential_overlaps_public_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        overlap_field: str,
    ) -> None:
        name = "legacy-provider"
        base_url = "https://responses.example/v1"
        if overlap_field == "name":
            name = f"legacy-{_CUSTOM_API_KEY}"
        else:
            base_url = f"https://responses.example/{_CUSTOM_API_KEY}/v1"
        provider = OpenAICompatibleProvider(
            wire_api="responses",
            base_url=base_url,
            api_key=_CUSTOM_API_KEY,
        )
        with self._admin_client(
            monkeypatch, tmp_path, custom_providers={name: provider}
        ) as client:
            response = client.get("/admin/settings/mapping")

        _assert_secret_absent(_CUSTOM_API_KEY, response.text, repr(provider))
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"
        assert (
            response.json()["error"]["message"]
            == "custom provider metadata is unavailable"
        )

    def test_get_fails_closed_for_cross_provider_credential_overlap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cross_provider_secret = "cross-provider-sensitive-key"
        providers = {
            "first": OpenAICompatibleProvider(
                wire_api="responses",
                base_url="https://first.example/v1",
                api_key=cross_provider_secret,
            ),
            "second": OpenAICompatibleProvider(
                wire_api="responses",
                base_url=f"https://second.example/{cross_provider_secret}/v1",
                api_key=_CUSTOM_API_KEY,
            ),
        }
        with self._admin_client(
            monkeypatch, tmp_path, custom_providers=providers
        ) as client:
            response = client.get("/admin/settings/mapping")

        _assert_secret_absent(cross_provider_secret, response.text)
        _assert_secret_absent(_CUSTOM_API_KEY, response.text)
        assert response.status_code == 500
        assert (
            response.json()["error"]["message"]
            == "custom provider metadata is unavailable"
        )

    def test_get_without_custom_providers_keeps_existing_payload_shape(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.get("/admin/settings/mapping")

        assert "custom_providers" not in response.json()

    def test_put_updates_runtime_and_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )
            # The swap is live for later requests ...
            reread = client.get("/admin/settings/mapping")

        assert response.status_code == 200
        assert response.json()["model_map"] == {"opus": "codex:gpt-5.6-sol"}
        assert reread.json()["model_map"] == {"opus": "codex:gpt-5.6-sol"}
        # ... and persisted without clobbering unrelated settings keys.
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317, "model_map": {"opus": "codex:gpt-5.6-sol"}}

    def test_put_rejected_when_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "codex:gpt-5.6-luna"}')
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "codex:gpt-5.6-sol"}}
            )

        assert response.status_code == 409
        assert "CLAUDEX_MODEL_MAP" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/mapping",
                content='{"model_map": {}}',
                headers={"content-type": "text/plain"},
            )
        assert response.status_code == 415

    def test_put_validates_map_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/settings/mapping", json={"model_map": {"": "x"}})
        assert response.status_code == 400
        assert "non-empty strings" in response.json()["error"]["message"]

    def test_put_accepts_and_validates_provider_prefixes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        with self._admin_client(monkeypatch, tmp_path) as client:
            accepted = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kimi:k2.5"}}
            )
            bare_value = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "gpt-5.6-sol"}}
            )
            empty_model = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kimi:"}}
            )
            unknown_prefix = client.put(
                "/admin/settings/mapping", json={"model_map": {"opus": "kim:k2.5"}}
            )

        assert accepted.status_code == 200
        assert accepted.json()["model_map"] == {"opus": "kimi:k2.5"}
        assert bare_value.status_code == 400
        assert "no provider prefix" in bare_value.json()["error"]["message"]
        assert empty_model.status_code == 400
        assert "names no model" in empty_model.json()["error"]["message"]
        assert unknown_prefix.status_code == 400
        assert "unknown provider prefix" in unknown_prefix.json()["error"]["message"]

    def test_put_accepts_custom_provider_prefix_and_rejects_unknown_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDEX_MODEL_MAP", raising=False)
        with self._admin_client(
            monkeypatch,
            tmp_path,
            custom_providers={"wrtn": _custom_provider()},
        ) as client:
            accepted = client.put(
                "/admin/settings/mapping",
                json={"model_map": {"opus": "wrtn:gpt-5.5"}},
            )
            rejected = client.put(
                "/admin/settings/mapping",
                json={"model_map": {"opus": "other:gpt-5.5"}},
            )

        assert accepted.status_code == 200
        assert accepted.json()["model_map"] == {"opus": "wrtn:gpt-5.5"}
        assert rejected.status_code == 400
        assert "unknown provider prefix" in rejected.json()["error"]["message"]

    def test_put_rejects_unknown_and_empty_bodies(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/settings/mapping", json={"bogus": {}}).status_code == 400
            assert client.put("/admin/settings/mapping", json={}).status_code == 400

    def test_admin_refuses_foreign_host_header(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.get("/admin/settings/mapping")
        assert response.status_code == 403
        assert "DNS-rebinding" in response.json()["error"]["message"]


    def test_get_reports_dashboard_facts_and_env_locks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "codex:gpt-5.6-luna"}')
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/mapping").json()

        assert payload["env_locked"] == {"model_map": "CLAUDEX_MODEL_MAP"}
        assert payload["codex_home"].endswith(".codex")
        assert payload["kimi_code_home"].endswith(".kimi-code")
        assert payload["grok_home"].endswith(".grok")


class TestAdminLogLevel:
    """GET/PUT /admin/settings/log-level — runtime log level applied and persisted."""

    @staticmethod
    def _admin_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        monkeypatch.delenv("CLAUDEX_LOG_LEVEL", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_current_level(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/log-level").json()

        assert payload == {
            "log_level": "info",
            "choices": ["debug", "info", "warning", "error"],
            "env_locked": None,
        }

    def test_put_applies_and_persists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        saved_levels = {
            name: logging.getLogger(name).level
            for name in admin_settings._LOG_LEVEL_LOGGER_NAMES
        }
        try:
            with self._admin_client(monkeypatch, tmp_path) as client:
                response = client.put("/admin/settings/log-level", json={"log_level": "debug"})
                reread = client.get("/admin/settings/log-level")

            assert response.status_code == 200
            assert response.json()["log_level"] == "debug"
            assert reread.json()["log_level"] == "debug"
            assert logging.getLogger().level == logging.DEBUG
            assert logging.getLogger("uvicorn").level == logging.DEBUG
            saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
            assert saved == {"log_level": "debug"}
        finally:
            for name, level in saved_levels.items():
                logging.getLogger(name).setLevel(level)

    def test_put_rejected_when_env_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        monkeypatch.setenv("CLAUDEX_LOG_LEVEL", "info")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/settings/log-level", json={"log_level": "debug"})

        assert response.status_code == 409
        assert "CLAUDEX_LOG_LEVEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_put_validates_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            assert client.put("/admin/settings/log-level", json={"log_level": "loud"}).status_code == 400
            assert client.put("/admin/settings/log-level", json={}).status_code == 400


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

    def test_get_returns_fresh_state_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload == {"model": None, "env_locked": False, "last_reroute": None}

    def test_get_returns_configured_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }


    def test_get_reports_env_locked_true_for_nonempty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "claude:claude-env-5")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload["env_locked"] is True

    def test_get_reports_env_locked_true_for_empty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            payload = client.get("/admin/settings/compaction").json()

        assert payload["env_locked"] is True

    # --- PUT: persistence, hot-swap, disable, enable/disable trigger -------

    def test_put_persists_without_clobbering_unrelated_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
            )
            reread = client.get("/admin/settings/compaction")

        assert response.status_code == 200
        assert response.json() == {
            "model": "claude:claude-opus-5",
            "env_locked": False,
            "last_reroute": None,
        }
        assert reread.json()["model"] == "claude:claude-opus-5"
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {
            "port": 9317,
            "compaction": {"model": "claude:claude-opus-5"},
        }

    def test_put_hot_swaps_live_config_for_subsequent_requests(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            client.put("/admin/settings/compaction", json={"model": "claude:claude-opus-5"})
            assert client.app.state.config.compaction_model == "claude:claude-opus-5"
            reread = client.get("/admin/settings/compaction")

        assert reread.json()["model"] == "claude:claude-opus-5"

    def test_put_disables_and_removes_the_settings_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"compaction.model": "claude:claude-old-5", "port": 1234}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-old-5"
        ) as client:
            response = client.put("/admin/settings/compaction", json={"model": None})
            reread = client.get("/admin/settings/compaction")

        assert response.status_code == 200
        assert response.json() == {"model": None, "env_locked": False, "last_reroute": None}
        assert reread.json()["model"] is None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 1234}
        assert "compaction" not in saved


    # --- PUT: request-shape validation --------------------------------------

    def test_put_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put(
                "/admin/settings/compaction",
                content='{"model": null}',
                headers={"content-type": "text/plain"},
            )
            config_after = client.app.state.config
        assert response.status_code == 415
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_missing_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put("/admin/settings/compaction", json={})
            config_after = client.app.state.config
        assert response.status_code == 400
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_unknown_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put(
                "/admin/settings/compaction",
                json={"model": "claude:claude-opus-5", "bogus": True},
            )
            config_after = client.app.state.config
        assert response.status_code == 400
        assert "bogus" in response.json()["error"]["message"]
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    def test_put_rejects_invalid_string(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/settings/compaction", json={"model": "gpt-5"})
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "claude:" in response.json()["error"]["message"]
        assert not settings_file.exists()
        assert config_after.compaction_model is None

    @pytest.mark.parametrize(
        "value", [True, False, 5, 1.5, ["claude:claude-opus-5"], {"model": "claude:x"}]
    )
    def test_put_rejects_every_non_string_non_null_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Any
    ) -> None:
        settings_file = tmp_path / "settings.json"
        configured = {"compaction.model": "claude:claude-opus-5", "port": 9317}
        settings_file.write_text(json.dumps(configured), encoding="utf-8")
        with self._admin_client(
            monkeypatch, tmp_path, compaction_model="claude:claude-opus-5"
        ) as client:
            response = client.put("/admin/settings/compaction", json={"model": value})
            config_after = client.app.state.config
        assert response.status_code == 400
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configured
        assert config_after.compaction_model == "claude:claude-opus-5"

    # --- Security/lock: bearer + Host, for both GET and PUT ----------------

    def test_get_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            assert client.get("/admin/settings/compaction").status_code == 401

    def test_get_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer wrong"}
            )
        assert response.status_code == 401

    def test_get_succeeds_with_correct_bearer_on_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 200

    def test_get_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        # Default base_url keeps the Host header at "testserver".
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.get(
                "/admin/settings/compaction", headers={"Authorization": "Bearer secret"}
            )
        assert response.status_code == 403

    def test_put_requires_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put("/admin/settings/compaction", json={"model": None})
            config_after = client.app.state.config
        assert response.status_code == 401
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_rejects_wrong_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put(
                "/admin/settings/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer wrong"},
            )
            config_after = client.app.state.config
        assert response.status_code == 401
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_succeeds_with_correct_bearer_on_allowed_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            response = client.put(
                "/admin/settings/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer secret"},
            )
        assert response.status_code == 200

    def test_put_refuses_foreign_host_even_with_correct_bearer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(settings_file=tmp_path / "settings.json", local_token="secret")
        with _create_test_client(monkeypatch, tmp_path, config=config) as client:
            response = client.put(
                "/admin/settings/compaction",
                json={"model": "claude:claude-opus-5"},
                headers={"Authorization": "Bearer secret"},
            )
            config_after = client.app.state.config
        assert response.status_code == 403
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    # --- Environment lock: 409 before the lock, empty value still locks ----

    def test_put_rejected_when_env_locked_with_nonempty_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "claude:claude-env-5")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
            )
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_COMPACTION_MODEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    def test_put_rejected_when_env_locked_with_empty_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            response = client.put("/admin/settings/compaction", json={"model": None})
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_COMPACTION_MODEL" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.compaction_model is None

    # --- Persistence failure: 500, no hot-swap, no false success envelope --

    def test_put_persistence_failure_returns_500_without_mutating_runtime(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise ConfigError("disk full")

        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setattr(admin_settings, "update_settings_file", boom)
            response = client.put(
                "/admin/settings/compaction", json={"model": "claude:claude-opus-5"}
            )
            config_after = client.app.state.config

        assert response.status_code == 500
        body = response.json()
        assert "error" in body
        assert "last_reroute" not in body
        assert config_after.compaction_model is None
        assert not (tmp_path / "settings.json").exists()


class TestAdminCodexApi:
    """GET/PUT /admin/settings/codex — Codex Fast service tier."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CODEX_SERVICE_TIER", raising=False)
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_default_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": None, "env_locked": False}

    def test_put_fast_persists_and_hot_swaps_live_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )
            config_after = client.app.state.config
            reread = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": "fast", "env_locked": False}
        assert reread.json() == {"service_tier": "fast", "env_locked": False}
        assert config_after.codex_service_tier == "fast"
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "port": 9317,
            "codex": {"service_tier": "fast"},
        }

    def test_put_null_removes_key_and_hot_swaps_live_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"port": 9317, "codex": {"service_tier": "fast"}}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, codex_service_tier="fast"
        ) as client:
            response = client.put(
                "/admin/settings/codex", json={"service_tier": None}
            )
            config_after = client.app.state.config
            reread = client.get("/admin/settings/codex")

        assert response.status_code == 200
        assert response.json() == {"service_tier": None, "env_locked": False}
        assert reread.json() == {"service_tier": None, "env_locked": False}
        assert config_after.codex_service_tier is None
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317}
        assert "codex" not in saved

    @pytest.mark.parametrize(
        "body",
        [
            {"service_tier": "priority"},
            {"service_tier": 123},
            {},
        ],
    )
    def test_put_rejects_invalid_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        body: dict[str, Any],
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/settings/codex", json=body)

        assert response.status_code == 400
        assert "error" in response.json()
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize("env_value", ["fast", ""])
    def test_env_override_locks_get_and_put(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        env_value: str,
    ) -> None:
        monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", env_value)
        config = GatewayConfig(settings_file=tmp_path / "settings.json")
        with _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        ) as client:
            get_response = client.get("/admin/settings/codex")
            put_response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )

        assert get_response.json() == {"service_tier": None, "env_locked": True}
        assert put_response.status_code == 409
        assert "CLAUDEX_CODEX_SERVICE_TIER" in put_response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    def test_get_and_put_require_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            get_response = client.get("/admin/settings/codex")
            put_response = client.put(
                "/admin/settings/codex", json={"service_tier": "fast"}
            )

        assert get_response.status_code == 401
        assert put_response.status_code == 401
        assert not (tmp_path / "settings.json").exists()


class TestAdminClaudeAccountApi:
    """GET/PUT /admin/providers/claude/pool/serving — serving-account selection, mirroring
    /admin/settings/compaction. The shared bearer/Host guard paths are exhaustively
    covered by the compaction tests; here one auth check pins the guard is
    wired at all, and the rest exercises the account-specific logic."""

    _ACCOUNT_ID = "0a1b2c3d-4e5f-4678-9abc-def012345678"

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_returns_fresh_state_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload == {"account_id": None, "env_locked": False}

    def test_get_returns_configured_account(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload == {"account_id": self._ACCOUNT_ID, "env_locked": False}

    def test_get_reports_env_locked_even_for_empty_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", "")
            payload = client.get("/admin/providers/claude/pool/serving").json()

        assert payload["env_locked"] is True

    def test_put_registered_account_persists_and_hot_swaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"port": 9317}', encoding="utf-8")
        with self._admin_client(monkeypatch, tmp_path) as client:
            account_id = _register_serving_account()
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": account_id}
            )
            reread = client.get("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"account_id": account_id, "env_locked": False}
        assert reread.json()["account_id"] == account_id
        assert config_after.claude_account_id == account_id
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"port": 9317, "claude_account": {"id": account_id}}

    def test_put_null_is_rejected_pointing_at_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.id": self._ACCOUNT_ID}), encoding="utf-8"
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            response = client.put("/admin/providers/claude/pool/serving", json={"account_id": None})
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "DELETE" in response.json()["error"]["message"]
        assert config_after.claude_account_id == self._ACCOUNT_ID
        assert "claude_account.id" in json.loads(settings_file.read_text())

    def test_delete_clears_the_setting_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.id": self._ACCOUNT_ID}), encoding="utf-8"
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            response = client.delete("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"account_id": None, "env_locked": False}
        assert config_after.claude_account_id is None
        assert "claude_account" not in json.loads(settings_file.read_text())

    def test_delete_when_already_clear_is_a_no_op_200(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.delete("/admin/providers/claude/pool/serving")

        assert response.status_code == 200
        assert response.json() == {"account_id": None, "env_locked": False}

    def test_put_unregistered_account_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": self._ACCOUNT_ID}
            )
            config_after = client.app.state.config

        assert response.status_code == 400
        assert "no account registered" in response.json()["error"]["message"]
        assert config_after.claude_account_id is None
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize(
        "value", ["not-a-uuid", "0A1B2C3D-4E5F-4678-9ABC-DEF012345678", 1, True, [], {}]
    )
    def test_put_invalid_account_id_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: Any
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/providers/claude/pool/serving", json={"account_id": value})

        assert response.status_code == 400

    def test_put_rejects_unknown_and_missing_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            unknown = client.put("/admin/providers/claude/pool/serving", json={"account": "x"})
            missing = client.put("/admin/providers/claude/pool/serving", json={})

        assert unknown.status_code == 400
        assert missing.status_code == 400

    def test_put_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            account_id = _register_serving_account()
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", self._ACCOUNT_ID)
            response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": account_id}
            )
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.claude_account_id is None

    def test_delete_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_id=self._ACCOUNT_ID
        ) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", self._ACCOUNT_ID)
            response = client.delete("/admin/providers/claude/pool/serving")
            config_after = client.app.state.config

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert config_after.claude_account_id == self._ACCOUNT_ID

    def test_endpoints_require_local_token_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path, local_token="secret") as client:
            get_response = client.get("/admin/providers/claude/pool/serving")
            put_response = client.put(
                "/admin/providers/claude/pool/serving", json={"account_id": None}
            )

        assert get_response.status_code == 401
        assert put_response.status_code == 401
        assert not (tmp_path / "settings.json").exists()


class TestAdminClaudeRoutingApi:
    """GET/PUT /admin/providers/claude/pool/routing — the pool policy document."""

    @staticmethod
    def _admin_client(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        **config_kwargs: Any,
    ) -> TestClient:
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        config = GatewayConfig(settings_file=tmp_path / "settings.json", **config_kwargs)
        return _create_test_client(
            monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
        )

    def test_get_defaults_to_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            payload = client.get("/admin/providers/claude/pool/routing").json()

        assert payload == {"mode": "disabled", "env_locked": False}

    def test_put_fallback_persists_the_policy_document_and_hot_swaps(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "fallback"}
            )
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"mode": "fallback", "env_locked": False}
        assert config_after.claude_account_routing_mode == "fallback"
        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        assert saved == {"claude_account": {"routing": {"mode": "fallback"}}}

    def test_put_disabled_removes_the_settings_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"claude_account.routing": {"mode": "fallback"}}),
            encoding="utf-8",
        )
        with self._admin_client(
            monkeypatch, tmp_path, claude_account_routing_mode="fallback"
        ) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            config_after = client.app.state.config

        assert response.status_code == 200
        assert response.json() == {"mode": "disabled", "env_locked": False}
        assert config_after.claude_account_routing_mode == "disabled"
        assert "claude_account" not in json.loads(settings_file.read_text())

    def test_put_balanced_with_unknown_keys_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # "balanced" is a fully valid mode now (T-10); a future-shaped
        # document still reports the real "unknown keys" reason, not a stale
        # "not implemented" one, and the exact mode-envelope shape is
        # unchanged.
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put(
                "/admin/providers/claude/pool/routing",
                json={"mode": "balanced", "balanced": {"window": "session"}},
            )

        assert response.status_code == 400
        assert "unknown keys: balanced" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"mode": "round-robin"},
            {"mode": None},
            {"mode": True},
            {"mode": "fallback", "weights": [1]},
        ],
    )
    def test_put_garbage_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict[str, Any]
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            response = client.put("/admin/providers/claude/pool/routing", json=body)

        assert response.status_code == 400, body
        assert not (tmp_path / "settings.json").exists()

    def test_put_rejected_when_env_locked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._admin_client(monkeypatch, tmp_path) as client:
            monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", '{"mode": "fallback"}')
            response = client.put(
                "/admin/providers/claude/pool/routing", json={"mode": "disabled"}
            )
            locked = client.get("/admin/providers/claude/pool/routing").json()

        assert response.status_code == 409
        assert "CLAUDEX_CLAUDE_ACCOUNT_ROUTING" in response.json()["error"]["message"]
        assert not (tmp_path / "settings.json").exists()
        assert locked["env_locked"] is True


def test_admin_logs_returns_recent_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        logging.getLogger("claudex.test").warning("hello %s", "world")
        response = client.get("/admin/logs")

    assert response.status_code == 200
    entries = [e for e in response.json()["logs"] if e["message"] == "hello world"]
    assert entries and entries[0]["level"] == "WARNING"
    assert entries[0]["logger"] == "claudex.test"


def test_admin_logs_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.get("/admin/logs").status_code == 403


def test_admin_usage_returns_all_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    async def fake_claude(http_client: Any) -> dict[str, Any]:
        calls.append("claude")
        return {"provider": "claude", "status": "ok", "error": None}

    async def fake_codex(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("codex")
        return {"provider": "codex", "status": "unavailable", "error": "no creds"}

    async def fake_kimi(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("kimi")
        return {"provider": "kimi", "status": "ok", "error": None}

    async def fake_grok(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        calls.append("grok")
        return {"provider": "grok", "status": "ok", "error": None}

    monkeypatch.setattr(admin_system, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(admin_system, "fetch_codex_usage", fake_codex)
    monkeypatch.setattr(admin_system, "fetch_kimi_usage", fake_kimi)
    monkeypatch.setattr(admin_system, "fetch_grok_usage", fake_grok)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage")

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert body["codex"]["status"] == "unavailable"
    assert body["kimi"]["status"] == "ok"
    assert body["grok"]["status"] == "ok"
    assert body["fetched_at"] > 0
    assert calls == ["claude", "codex", "kimi", "grok"]


def test_admin_usage_refuses_foreign_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.get("/admin/usage").status_code == 403


def test_admin_usage_single_provider_skips_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_claude(http_client: Any) -> dict[str, Any]:
        return {"provider": "claude", "status": "ok", "error": None}

    async def codex_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("codex probe ran for ?provider=claude")

    async def kimi_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("kimi probe ran for ?provider=claude")

    async def grok_must_not_run(http_client: Any, auth_manager: Any) -> dict[str, Any]:
        raise AssertionError("grok probe ran for ?provider=claude")

    monkeypatch.setattr(admin_system, "fetch_claude_usage", fake_claude)
    monkeypatch.setattr(admin_system, "fetch_codex_usage", codex_must_not_run)
    monkeypatch.setattr(admin_system, "fetch_kimi_usage", kimi_must_not_run)
    monkeypatch.setattr(admin_system, "fetch_grok_usage", grok_must_not_run)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "claude"})

    assert response.status_code == 200
    body = response.json()
    assert body["claude"]["status"] == "ok"
    assert "codex" not in body
    assert "kimi" not in body
    assert "grok" not in body


def test_admin_usage_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.get("/admin/usage", params={"provider": "gemini"})

    assert response.status_code == 400


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


def test_admin_reset_credit_returns_the_backend_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    keys = _record_reset_keys(
        monkeypatch, [{"status": "ok", "outcome": "reset", "error": None}]
    )
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        response = client.post("/admin/providers/codex/reset-credit", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "outcome": "reset", "error": None}
    assert len(keys) == 1 and keys[0]


def test_admin_reset_credit_reuses_the_key_until_an_attempt_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A timed-out attempt may or may not have burned the credit, so the retry
    # must carry the same idempotency key and let the backend deduplicate;
    # only a settled outcome may mint a fresh one.
    keys = _record_reset_keys(
        monkeypatch,
        [
            {"status": "error", "outcome": None, "error": "timeout"},
            {"status": "unavailable", "outcome": None, "error": "no creds"},
            {"status": "ok", "outcome": "reset", "error": None},
            {"status": "ok", "outcome": "no_credit", "error": None},
        ],
    )
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        for _ in range(4):
            assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 200

    assert keys[0] == keys[1] == keys[2], "unsettled attempts must retry the same key"
    assert keys[3] != keys[2], "a settled attempt must not reuse its key"


def test_admin_reset_credit_is_guarded_like_every_other_admin_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a guarded request reached the ChatGPT backend")

    monkeypatch.setattr(admin_system, "consume_codex_reset_credit", must_not_run)

    # Foreign Host header (DNS-rebinding guard).
    with _create_test_client(monkeypatch, tmp_path) as client:
        assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 403
    # A form post would dodge the CORS preflight, so JSON is required.
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        assert client.post("/admin/providers/codex/reset-credit", content="x").status_code == 415
    # And the local bearer token still applies.
    config = GatewayConfig(local_token="local-secret")
    with _create_test_client(
        monkeypatch, tmp_path, config=config, base_url="http://127.0.0.1:8787"
    ) as client:
        assert client.post("/admin/providers/codex/reset-credit", json={}).status_code == 401


def test_admin_reset_credit_is_never_reachable_by_GET(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Spending a credit must not be possible by navigating to a URL.
    def must_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a GET spent a reset credit")

    monkeypatch.setattr(admin_system, "consume_codex_reset_credit", must_not_run)
    with _create_test_client(monkeypatch, tmp_path, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/admin/providers/codex/reset-credit").status_code == 405


def _dashboard_sources(client: TestClient) -> dict[str, str]:
    routes = {
        "html": "/",
        "css": "/dashboard.css",
        "javascript": "/dashboard.js",
    }
    responses = {name: client.get(route) for name, route in routes.items()}
    assert all(response.status_code == 200 for response in responses.values())
    return {name: response.text for name, response in responses.items()}


def test_dashboard_usage_merged_into_status_cards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Usage renders inside the Status tab's provider cards: a per-provider
    # body hook, the fetch on entering Status, and no separate tab.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert 'data-t="usage"' not in page
    assert 'id="tab-usage"' not in page
    assert 'id="usage-body-claude"' in page
    assert 'id="usage-body-codex"' in page
    assert 'id="usage-body-kimi"' in page
    assert "/admin/usage" in javascript
    # The usage hooks sit inside the Status section, not a sibling tab.
    assert page.index('id="tab-status"') < page.index('id="usage-body-codex"')
    assert page.index('id="usage-body-claude"') < page.index('id="tab-log"')
    # Claude leads the provider cards as a Status card like the others.
    assert "<h2>Claude Status" in page
    assert "<h2>Claude Usage" not in page
    assert page.index('id="usage-body-claude"') < page.index('id="codex-stat"')


def test_dashboard_settings_tab_leads_and_holds_compaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings tab is the settings home: it leads the tab bar, opens by
    # default, and holds the Compact Reroute row behind the category rail;
    # the provider status cards live in the Status tab.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert (
        page.index('data-t="settings"')
        < page.index('data-t="status"')
        < page.index('data-t="map"')
        < page.index('data-t="log"')
    )
    assert '<body data-tab="settings">' in page
    assert 'const TAB_NAMES=["settings","status","map","log"]' in javascript
    # The category rail leads the pane and deep-links its single category.
    assert 'href="#settings/general"' in page
    assert (
        page.index('class="rail-item on"')
        < page.index('id="compaction-card"')
    )
    # The compaction row sits inside the Settings section (before the next
    # sibling section), not in Status where it used to render.
    assert (
        page.index('id="tab-settings"')
        < page.index('id="compaction-card"')
        < page.index('id="tab-map"')
    )


def test_dashboard_optional_providers_hidden_until_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Kimi and Grok are extensions: their Status cards and the Router
    # add-node provider buttons ship hidden and are revealed from /health
    # only when a local login is detected — or when the map already routes
    # to them, where hiding a required-login error would mislead. The gating
    # is cosmetic only; routing, settings.json, and the admin API are
    # untouched.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert '<div class="card provider-hidden" id="card-kimi">' in page
    assert '<div class="card provider-hidden" id="card-grok">' in page
    # Codex is built in and never hides.
    assert 'id="card-codex"' not in page
    assert "function setProviderVisibility(" in javascript
    assert 'info.status==="ok"||info.required===true' in javascript
    # The Router provider picker builds optional providers hidden too.
    assert '''(p==="codex"?"":' class="provider-hidden"')''' in javascript
    # Bulk usage refresh only probes visible cards.
    assert "PROVIDER_VISIBLE[p]!==false" in javascript


def test_dashboard_plan_and_credits_read_inside_the_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    # Plan and credit chips used to crowd the card headings; both now read as
    # part of the card they qualify, so the hooks and the chip style are gone.
    assert all("usage-chips" not in source for source in sources.values())
    assert all("uplan" not in source for source in sources.values())
    for provider in ("Claude", "Codex", "Kimi"):
        assert f"<h2>{provider} Status</h2>" in page

    # The plan always precedes the quota bars it qualifies.
    for provider in ("claude", "codex", "kimi"):
        assert f'id="usage-plan-{provider}"' in page
        assert page.index(f'id="usage-plan-{provider}"') < page.index(
            f'id="usage-body-{provider}"'
        )
    # Codex and Kimi read it inside the login-status box, above the state
    # line. That line is its own element precisely so a health render can
    # rewrite the status without wiping the plan the usage probe put there.
    for provider in ("codex", "kimi"):
        assert (
            page.index(f'id="{provider}-stat"')
            < page.index(f'id="usage-plan-{provider}"')
            < page.index(f'id="{provider}-statline"')
        )
    # Claude has no login-status box, so its plan reads under the description.
    assert page.index('id="usage-plan-claude"') > page.index("<h2>Claude Status</h2>")

    # The credit count is stated by the reset line, which owns the spend
    # button too, so there is one place credits are reported.
    assert 'class="uact"' in javascript
    assert "리셋 크레딧" in javascript


def test_dashboard_status_cards_load_as_skeletons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        stylesheet = sources["css"]
        javascript = sources["javascript"]

    # The cards used to render a one-line "Checking…" status that loaded content
    # then pushed apart. They now ship a skeleton of the same shape instead,
    # so nothing moves when the probes answer.
    assert all("확인 중" not in source for source in sources.values())
    assert 'class="sk"' in page
    # A skeleton carries the text it stands in for, painted transparent, so
    # its line box matches the line that replaces it.
    assert ".sk{" in stylesheet and "color:transparent" in stylesheet
    # Both status boxes seed one before any request goes out, and the usage
    # bodies are filled synchronously at boot rather than starting empty.
    for provider in ("codex", "kimi"):
        assert f'id="{provider}-statline"><span class="sk">' in page
    assert 'renderUsageProvider(p,null)' in javascript


def test_dashboard_served_at_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Claudex Gateway" in response.text
    assert 'rel="stylesheet" href="/dashboard.css"' in response.text
    assert 'src="/dashboard.js"' in response.text


@pytest.mark.parametrize(
    ("route", "asset_name", "content_type"),
    [
        ("/dashboard.css", "dashboard.css", "text/css"),
        ("/dashboard.js", "dashboard.js", "application/javascript"),
    ],
)
def test_dashboard_assets_match_packaged_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    asset_name: str,
    content_type: str,
) -> None:
    expected = (
        importlib.resources.files("claudex")
        .joinpath("dashboard", asset_name)
        .read_text(encoding="utf-8")
    )
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get(route)

    assert response.status_code == 200
    assert response.text == expected
    assert response.headers["content-type"].startswith(content_type)


@pytest.mark.parametrize(
    ("route", "asset_name"),
    [
        ("/dashboard.css", "dashboard.css"),
        ("/dashboard.js", "dashboard.js"),
    ],
)
def test_dashboard_asset_missing_returns_named_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    asset_name: str,
) -> None:
    def missing_package_files(_package: str) -> Any:
        raise FileNotFoundError(asset_name)

    monkeypatch.setattr(importlib.resources, "files", missing_package_files)
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get(route)

    assert response.status_code == 500
    assert response.json() == server_support._openai_error_body(
        "server_error", f"{asset_name} is missing from the package"
    )


def test_favicon_served_for_browser_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "max-age" in response.headers["cache-control"]
    assert response.text.startswith("<svg")


def test_dashboard_port_has_an_enlarged_invisible_hit_zone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The 14px visible port dot is too small to drag from: the board widens
    # the grab area to the node's full height plus margins without changing
    # the visual. These markers are the whole mechanism, so pin them.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["css"]

    assert ".node.src .port::after" in page
    assert "pointer-events:auto" in page


def test_hello_reports_identity_and_auth_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        body = client.get("/api/hello").json()

    assert body["hello"] == "claudex-gateway"
    assert body["local_auth_required"] is False
    assert isinstance(body["pid"], int)


def test_hello_reports_auth_required_without_leaking_the_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = GatewayConfig(local_token="secret-token-value")
    with _create_test_client(monkeypatch, tmp_path, config=config) as client:
        response = client.get("/api/hello")

    assert response.json()["local_auth_required"] is True
    assert "secret-token-value" not in response.text


def test_dashboard_keeps_the_local_token_in_memory_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    # The dashboard bootstraps from the safe hello flag and attaches the
    # bearer token to admin calls from a closure variable only.
    assert "local_auth_required" in page
    assert '"Authorization":"Bearer "+localToken' in page
    # The token must never be persisted or interpolated anywhere durable.
    assert "localStorage" not in page
    assert "sessionStorage" not in page
    assert "document.cookie" not in page


def test_dashboard_has_no_hardcoded_codex_model_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    # Catalogs are fetched live; a baked-in model list goes stale and misleads
    # when the catalog request fails. Nodes already in the mapping still render
    # via buildColumns.
    assert "CODEX_FALLBACK" not in page
    assert "gpt-5" not in page
    assert "CATALOG={codex:[],kimi:[],grok:[]}" in page


def test_dashboard_board_shows_only_referenced_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]
        html = sources["html"]

    # With several providers a catalog dump is unusable as a board, so target
    # nodes are only what the map references plus what the add-node box
    # stages; the catalogs survive purely as autocomplete for that box.
    assert "concat(Object.values(DIR.mapping),addedTargets)" in page
    assert 'list="add-catalog"' in html
    # All provider catalogs feed it, so the dashboard depends on the Kimi
    # and Grok endpoints too — not just the Codex one.
    assert '"/admin/providers/kimi/models"' in page
    assert '"/admin/providers/grok/models"' in page


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


def _lockable_apply_fn(page: str) -> str:
    start = page.index("function applyLockableSetting(config){")
    end = page.index("function applyCompaction(){", start)
    return page[start:end]


def _codex_apply_fn(page: str) -> str:
    start = page.index("function applyCodex(){")
    end = page.index(
        'document.getElementById("codex-fast").addEventListener("change"', start
    )
    return page[start:end]


def _routing_section(page: str) -> str:
    """Slice the routing card's own markup — "balanced" legitimately appears
    elsewhere in the document (API test prose never, but future copy might),
    so absence is asserted against this scoped slice."""
    start = page.index("<!-- routing-section:start -->")
    end = page.index("<!-- routing-section:end -->")
    return page[start:end]


def _routing_apply_fn(page: str) -> str:
    start = page.index("function applyRouting(){")
    end = page.index(
        'document.getElementById("routing-select").addEventListener("change"', start
    )
    return page[start:end]


def test_dashboard_compaction_section_marker_and_endpoint_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert "<!-- compaction-section:start -->" in page
    assert "<!-- compaction-section:end -->" in page
    section = _compaction_section(page)
    assert 'id="compaction-card"' in section
    assert "/admin/settings/compaction" in javascript


def test_dashboard_compaction_options_in_pinned_order_without_haiku(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    section = _compaction_section(page)
    assert (
        section.index(">Disabled<")
        < section.index("claude-sonnet-5 (recommended)")
        < section.index(">claude-opus-5<")
        < section.index(">claude-fable-5<")
        < section.index(">Custom<")
    )
    assert "claude-haiku" not in section
    # The literal string still exists in the JavaScript asset (see
    # compactionDraftFromModel's own comment), proving this is a scoped
    # assertion and not a whole-dashboard absence check.
    assert "claude-haiku" in javascript


def test_dashboard_compaction_custom_input_labeled_unverified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]

    section = _compaction_section(page)
    assert "unverified until first use" in section
    assert 'id="comp-custom-input"' in section


def test_dashboard_compaction_credentials_disclosure_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The card must state which credentials rerouted requests run on, so the
    # user knows their own Claude account is being used.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]

    section = _compaction_section(page)
    assert "장치에 저장된 Claude 기본 자격증명" in section


def test_dashboard_compaction_fetched_in_parallel_boot_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    boot_start = page.index("function boot(){")
    promise_all = page.index("Promise.all([", boot_start)
    promise_all_end = page.index("]);", promise_all)
    parallel_calls = page[promise_all:promise_all_end]
    assert 'jfetch("/admin/settings/compaction")' in parallel_calls


def test_dashboard_compaction_keeps_configured_custom_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    # A configured model that is not one of the three curated ids renders as
    # Custom with its raw id filled in, instead of being dropped.
    assert "COMPACTION_CURATED_MODELS.indexOf(id)>=0" in page
    assert '{kind:"custom",custom:id}' in page


def test_dashboard_compaction_diagnostics_ui_removed_by_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings redesign dropped the last-reroute record from the page:
    # diagnostics stay reachable through GET /admin/settings/compaction only. Guard
    # against the UI quietly returning.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert 'id="comp-diagnostics"' not in page
    assert "renderCompactionDiagnostics" not in javascript
    assert "아직 재라우팅이 시도되지 않았습니다" not in javascript


def test_dashboard_compaction_apply_body_matches_pinned_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    apply_fn = _compaction_apply_fn(page)
    assert 'jfetch("/admin/settings/compaction",{' in apply_fn
    assert 'method:"PUT"' in apply_fn
    assert "headers:JSON_HEADERS" in apply_fn
    assert "JSON.stringify({model:model})" in apply_fn
    # Disabled sends exactly {"model": null}; curated/custom selections carry
    # the "claude:" prefix parse_compaction_model expects.
    assert "model=null" in apply_fn
    assert '"claude:"+raw' in apply_fn
    assert '"claude:"+COMP.draftKind' in apply_fn


def test_dashboard_compaction_custom_submission_is_trimmed_and_guarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    apply_fn = _compaction_apply_fn(page)
    assert "input.value.trim()" in apply_fn
    assert "if(!raw)return;" in apply_fn


def test_dashboard_compaction_409_branch_refreshes_via_get_not_error_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    shared_apply = _lockable_apply_fn(page)
    branch_start = shared_apply.index("r.status===409")
    branch_end = shared_apply.index("if(!r.ok){", branch_start)
    locked_branch = shared_apply[branch_start:branch_end]
    apply_fn = _compaction_apply_fn(page)

    # The shared flow locks and renders the admin error before refreshing.
    lock_at = locked_branch.index("config.lock()")
    error_at = locked_branch.index("errDetail(r.body)")
    refresh_at = locked_branch.index("config.refresh()")
    assert lock_at < error_at < refresh_at
    # The compaction wrapper actually routes through the shared helper and
    # binds the lock mutation and authenticated GET to the config object; the
    # 409 response body itself is never passed to the state renderer.
    assert "applyLockableSetting({" in apply_fn
    assert "lock:function(){COMP.locked=true}" in apply_fn
    assert 'refresh:function(){return jfetch("/admin/settings/compaction")}' in apply_fn
    assert "renderCompactionState(g.body)" in apply_fn


def test_dashboard_compaction_409_refresh_failure_stays_locked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # If the post-409 refresh GET itself fails or returns a malformed
    # envelope, its body must not be rendered as state and the card must
    # remain locked.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    shared_apply = _lockable_apply_fn(page)
    apply_fn = _compaction_apply_fn(page)

    # All refresh invariants live inside the 409 branch itself: refresh, then
    # the freshness guard, then the ONLY adopt in the branch, then a failure
    # arm that re-locks and re-renders instead of adopting the malformed body.
    branch_start = shared_apply.index("r.status===409")
    branch_end = shared_apply.index("if(!r.ok){", branch_start)
    locked_branch = shared_apply[branch_start:branch_end]
    refresh_at = locked_branch.index("config.refresh()")
    fresh_guard_at = locked_branch.index("if(config.isFresh(g)){", refresh_at)
    adopt_at = locked_branch.index("config.adopt(g)", fresh_guard_at)
    assert locked_branch.index("config.adopt(") == adopt_at
    assert locked_branch.count("config.adopt(") == 1
    failure_arm = locked_branch[adopt_at:]
    assert "config.lock()" in failure_arm
    assert "config.render()" in failure_arm
    # The compaction wrapper routes through the shared helper and binds the
    # envelope guard to isFresh and the state renderer to adopt, so a
    # malformed refresh can never reach renderCompactionState.
    assert "applyLockableSetting({" in apply_fn
    is_fresh_body = apply_fn[
        apply_fn.index("isFresh:function(g){") : apply_fn.index("adopt:function(g){")
    ]
    assert "g.ok" in is_fresh_body
    assert 'typeof g.body.env_locked==="boolean"' in is_fresh_body
    adopt_body = apply_fn[
        apply_fn.index("adopt:function(g){") : apply_fn.index("envName:")
    ]
    assert "renderCompactionState(g.body)" in adopt_body
    assert "lock:function(){COMP.locked=true}" in apply_fn
    locked_sources = shared_apply + apply_fn
    assert "r.body.model" not in locked_sources
    assert "r.body.env_locked" not in locked_sources
    assert "r.body.last_reroute" not in locked_sources


def _codex_section(page: str) -> str:
    start = page.index("<!-- codex-section:start -->")
    end = page.index("<!-- codex-section:end -->")
    return page[start:end]


def test_dashboard_codex_fast_card_wires_apply_flow_and_env_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    section = _codex_section(page)
    assert page.index('id="compaction-card"') < page.index('id="codex-card"')
    assert 'type="checkbox" id="codex-fast"' in section
    assert 'id="codex-apply"' in section
    assert "~1.5x speed" in section
    assert "~2–2.5x usage burn" in section
    assert "silently stay standard" in section
    assert "CLAUDEX_CODEX_SERVICE_TIER" in javascript
    assert 'jfetch("/admin/settings/codex")' in javascript
    apply_fn = _codex_apply_fn(javascript)
    assert 'jfetch("/admin/settings/codex",{' in apply_fn
    assert 'method:"PUT"' in apply_fn
    assert "headers:JSON_HEADERS" in apply_fn
    assert 'JSON.stringify({service_tier:CODEX.draft?"fast":null})' in apply_fn
    assert "checkbox.disabled=CODEX.locked" in javascript
    assert (
        "btn.disabled=CODEX.locked||CODEX.draft===(CODEX.serviceTier===\"fast\")"
        in javascript
    )


def test_dashboard_codex_fast_fetched_in_parallel_boot_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    boot_start = page.index("function boot(){")
    promise_all = page.index("Promise.all([", boot_start)
    promise_all_end = page.index("]);", promise_all)
    parallel_calls = page[promise_all:promise_all_end]
    assert 'jfetch("/admin/settings/codex")' in parallel_calls


def test_dashboard_settings_rail_switches_categories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The Settings rail is real category switching now: one .scard visible at
    # a time, gated by data-cat on the section, deep-linkable per category.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        stylesheet = sources["css"]
        javascript = sources["javascript"]

    assert '<section id="tab-settings" data-cat="general">' in page
    assert 'href="#settings/general"' in page
    assert 'href="#settings/accounts"' in page
    assert ".scard{display:none}" in stylesheet
    assert 'id="scard-general"' in page
    assert 'id="scard-accounts"' in page
    assert "function setSettingsCat(" in javascript
    # General leads the rail and the card order; the accounts card follows.
    assert page.index('href="#settings/general"') < page.index(
        'href="#settings/accounts"'
    )
    assert page.index('id="scard-general"') < page.index('id="scard-accounts"')
    # A #settings/accounts deep link lands on the category at boot and on
    # hash changes.
    assert (
        'if(bootTab==="settings"&&bootParts[1])setSettingsCat(bootParts[1])'
        in javascript
    )
    assert (
        'if(parts[0]==="settings")setSettingsCat(parts[1]||"general")'
        in javascript
    )


def test_dashboard_accounts_card_mirrors_the_final_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The local CLI hero leads as the only boxed area, then the registered accounts
    # caption with the add
    # button, then dense flat rows that expand independently.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert 'class="lhero"' in page
    assert "로컬 CLI 로그인" in page
    assert "게이트웨이 서빙과 무관" in javascript
    assert 'id="btn-local-refresh"' in page
    assert 'id="btn-add-account"' in page
    assert (
        page.index('id="local-sec"')
        < page.index('id="acct-count"')
        < page.index('id="acct-list"')
    )
    # Collapsed rows carry status text only (no chips, no mini bars); the
    # right edge is the plan text.
    assert "서빙 중" in javascript
    assert "재로그인 필요" in javascript
    assert 'class="plan-txt"' in javascript
    # Expansion is independent per-row state, never an accordion.
    assert "ACCT.open[id]=!ACCT.open[id]" in javascript
    # Plan pill mapping: claude_max -> MAX, claude_pro -> PRO, null -> dash.
    assert "function planLabel(" in javascript
    assert 'replace(/^claude_/,"").toUpperCase()' in javascript


def test_dashboard_accounts_fetch_paints_registry_before_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The registry GET paints rows immediately; the cache-backed usage GET
    # and the local hero's ambient usage fill in async afterwards. No force
    # parameter exists — the UI shows data age instead.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    assert 'jfetch("/admin/providers/claude/accounts")' in page
    assert 'jfetch("/admin/providers/claude/pool/usage")' in page
    assert 'jfetch("/admin/usage?provider=claude")' in page
    assert "force" not in page.split('jfetch("/admin/providers/claude/pool/usage")')[1][:200]
    fetch_fn = page[page.index("function fetchAccounts(") :]
    assert fetch_fn.index("renderAcctList()") < fetch_fn.index("fetchAccountUsage()")
    # Data age renders from each result's updated_at.
    assert "function fmtAgo(" in page
    assert "기준" in page


def test_dashboard_serving_selection_reuses_the_singular_admin_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Serving and unserving go through the existing PUT /admin/providers/claude/pool/serving; a 409
    # env-lock renders the lockband and disables the buttons.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]
        html = sources["html"]
        stylesheet = sources["css"]

    assert 'jfetch("/admin/providers/claude/pool/serving",{' in page
    assert "JSON.stringify({account_id:accountId})" in page
    assert "CLAUDEX_CLAUDE_ACCOUNT_ID" in page
    assert 'id="acct-lockband"' in html
    assert "#scard-accounts.locked .acctlock{display:block}" in stylesheet
    assert "이 계정으로 서빙" in page
    assert "서빙 해제" in page
    # Removal uses the account endpoint; the serving pin guard stays visible in the UI.
    assert 'jfetch("/admin/providers/claude/accounts/"+encodeURIComponent(accountId),{method:"DELETE"})' in page


def test_dashboard_routing_section_wires_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["html"]
        javascript = sources["javascript"]

    assert "<!-- routing-section:start -->" in page
    assert "<!-- routing-section:end -->" in page
    # Policy row sits in General, directly after the compaction row.
    assert page.index('id="compaction-card"') < page.index('id="routing-card"')
    # Boot GET and the apply PUT both target the pool/routing endpoint, and
    # the PUT body is pinned to exactly {"mode": ...}.
    assert 'jfetch("/admin/providers/claude/pool/routing")' in javascript
    apply_fn = _routing_apply_fn(javascript)
    assert 'jfetch("/admin/providers/claude/pool/routing",{' in apply_fn
    assert 'method:"PUT"' in apply_fn
    assert "headers:JSON_HEADERS" in apply_fn
    assert "JSON.stringify({mode:ROUTING.draft})" in apply_fn
    section = _routing_section(page)
    assert ">Disabled<" in section
    assert ">Fallback<" in section
    # balanced is a fully valid server-side mode (config.py's
    # VALID_CLAUDE_ACCOUNT_ROUTING_MODES) and is offered here too.
    assert 'value="balanced"' in section
    assert ">Balanced<" in section
    assert "계정별 라우팅 상태 보기" in section
    # The mode-envelope validator adopts a balanced boot GET/apply response
    # instead of rejecting it as malformed.
    assert 'body.mode==="balanced"' in javascript


def test_dashboard_accounts_surface_pool_usage_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Balanced-mode usage reads are cache-only (T-13): the accounts screen
    # renders pool/status's usage_freshness chip plus each window's
    # pool/usage observation age/source, and a queued manual refresh renders
    # its own indication instead of claiming a completed refresh.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]
        html = sources["html"]

    assert "ACCT.usageFreshness" in page
    assert "statusResp.body.usage_freshness" in page
    assert 'id="pool-fresh-pill"' in html
    assert "function renderPoolFreshness(" in page
    assert "meta.age_seconds" in page
    assert "meta.source" in page
    assert "u.queued" in page
    assert "대기 중" in page


def test_dashboard_routing_locked_renders_readonly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A 409 flips the local lock; the lockband names the env var and the
    # control disables — same discipline as the compaction card.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]
        html = sources["html"]
        stylesheet = sources["css"]

    assert "CLAUDEX_CLAUDE_ACCOUNT_ROUTING" in page
    assert 'id="routing-lock-env"' in html
    assert (
        "#compaction-card.locked .complock,#codex-card.locked .complock,"
        "#routing-card.locked .complock{display:block}" in stylesheet
    )
    shared_apply = _lockable_apply_fn(page)
    apply_fn = _routing_apply_fn(page)
    branch_start = shared_apply.index("r.status===409")
    branch_end = shared_apply.index("if(!r.ok){", branch_start)
    assert "config.lock()" in shared_apply[branch_start:branch_end]
    # The routing wrapper routes through the shared helper and binds its lock,
    # refresh, and adopt callbacks to the config object.
    assert "applyLockableSetting({" in apply_fn
    assert "lock:function(){ROUTING.locked=true}" in apply_fn
    assert (
        'refresh:function(){return jfetch("/admin/providers/claude/pool/routing")}'
        in apply_fn
    )
    assert "adopt:function(g){renderRoutingState(g.body)}" in apply_fn


def test_dashboard_accounts_surface_routing_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Row badges come from pool/status; a failed status GET renders no badges
    # (never stale), and the cooldown row explains itself in the detail pane.
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]
        html = sources["html"]

    assert 'jfetch("/admin/providers/claude/pool/status")' in page
    assert "라우팅 준비" in page
    assert "라우팅 불가" in page
    assert "쿨다운 · " in page
    assert "coolnote" in page
    assert "function fmtCooldownUntil(" in page
    # The accounts card links back to the policy row in General.
    assert (
        '라우팅 정책은 <a class="route-link" href="#settings/general">General</a>에서 설정합니다.'
        in html
    )


def test_dashboard_login_modal_drives_the_login_endpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with _create_test_client(monkeypatch, tmp_path) as client:
        sources = _dashboard_sources(client)
        page = sources["javascript"]

    # All five login endpoints are wired: start, poll, code, confirm, cancel.
    assert 'const JSON_HEADERS={"Content-Type":"application/json"}' in page
    login_start = page.index("function openLoginModal(){")
    login_end = page.index("function pollLogin(){", login_start)
    login_start_flow = page[login_start:login_end]
    assert 'jfetch("/admin/providers/claude/login",{' in login_start_flow
    assert 'method:"POST",headers:JSON_HEADERS,body:"{}"' in login_start_flow
    assert page.count("attemptHeaders(attempt,JSON_HEADERS)") == 2
    assert 'jfetch("/admin/providers/claude/login")' in page
    assert "/admin/providers/claude/login/code" in page
    assert "/admin/providers/claude/login/replace" in page
    assert 'method:"DELETE"' in page
    # 409 login-active attaches to the running session instead of erroring.
    assert 'code==="login-active"' in page
    # The replace confirmation shows exactly the CLI's two choices.
    assert "교체 안 함" in page
    assert 'data-lact="replace"' in page
    assert 'data-lact="decline"' in page
    # The URL opens in a new tab with rel=noopener and an https-only guard.
    assert 'rel="noopener"' in page
    assert "/^https:\\/\\//.test(st.url||" in page
    # Anti-churn: the body re-renders only when the session state changes.
    assert "if(key===LOGIN.lastKey)return" in page
    # Explicit cancel only — neither a backdrop click nor Escape ever closes
    # the login modal (a silent close would leak a live CLI login child).
    assert "closeLoginModal()" in page
    assert "target===this)closeLoginModal()" not in page
    assert 'Escape")closeLoginModal()' not in page


class CatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        return ["gpt-5.6-sol", "gpt-5.5"]


class FailingCatalogCodexClient(FakeCodexClient):
    async def list_models(self) -> list[str]:
        raise CodexUpstreamError(400, '{"error":{"message":"unsupported client"}}')


class ProbeCodexClient(FakeCodexClient):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.payloads.append(payload)
        yield {"type": "response.created", "response": {"model": payload["model"]}}


class FailingFastTierLookupCodexClient(FakeCodexClient):
    async def supports_fast_tier(self, model: str) -> bool:
        raise AssertionError("admin probe must not query Codex fast-tier capability")


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


class TrackedProbeStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.read_calls = 0
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.read_calls += 1
        yield self.content

    async def aclose(self) -> None:
        self.close_calls += 1


class TrackedProbeResponse(httpx.Response):
    def __init__(self, status_code: int, stream: httpx.AsyncByteStream) -> None:
        super().__init__(status_code, stream=stream)
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1
        await super().aclose()


class ReadAndCloseFailingProbeResponse(httpx.Response):
    def __init__(self, credential: str) -> None:
        super().__init__(200, stream=TrackedProbeStream(b""))
        self.credential = credential
        self.aclose_calls = 0

    async def aread(self) -> bytes:
        raise httpx.ReadError(
            f"primary read failed with {self.credential}",
            request=httpx.Request("POST", "https://messages.example/v1/messages"),
        )

    async def aclose(self) -> None:
        self.aclose_calls += 1
        raise httpx.ReadError(
            f"cleanup close failed with {self.credential}",
            request=httpx.Request("POST", "https://messages.example/v1/messages"),
        )


class ReadAndCloseFailingAnthropicTransport:
    def __init__(self, response: ReadAndCloseFailingProbeResponse) -> None:
        self.response = response

    async def send_messages(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return self.response


class ProbeAnthropicTransport:
    def __init__(self, response: TrackedProbeResponse) -> None:
        self.response = response

    async def send_messages(
        self, body: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        return self.response


class ProbeKimiResponse(httpx.Response):
    def __init__(self) -> None:
        super().__init__(200, json={"type": "message", "model": "k3"})
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1
        await super().aclose()


class ProbeKimiClient(FakeKimiClient):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.headers: dict[str, str] | None = None
        self.response: ProbeKimiResponse | None = None

    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        # The probe must send the raw model with the prefix already removed.
        assert json.loads(body)["model"] == "k3"
        self.headers = headers
        self.response = ProbeKimiResponse()
        return self.response


class RejectingKimiClient(FakeKimiClient):
    async def send_messages(self, body: bytes, headers: dict[str, str]) -> httpx.Response:
        raise KimiUpstreamError(
            404, '{"type":"error","error":{"type":"not_found_error","message":"model not found"}}'
        )


class ProbeGrokClient(FakeGrokClient):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.payloads.append(payload)
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
    def __init__(
        self,
        name: str,
        provider: OpenAICompatibleProvider,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        super().__init__(name, provider)
        self.payloads: list[dict[str, Any]] = []

    async def stream_responses(
        self, payload: dict[str, Any], session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.payloads.append(payload)
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


class SelectedResponsesModelTransport:
    def __init__(self, models: list[str]) -> None:
        self.models = models
        self.calls = 0

    async def list_models(self) -> list[str]:
        self.calls += 1
        return self.models


class SelectedAnthropicModelTransport:
    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self.calls = 0

    async def list_models(self) -> Any:
        self.calls += 1
        return self.catalog


class SecretBearingCatalogTransport:
    def __init__(self, credential: str) -> None:
        self.credential = credential
        self.calls = 0

    async def list_models(self) -> list[str]:
        self.calls += 1
        raise httpx.ConnectError(
            f"catalog connection failed with {self.credential}"
        )


def _select_route_transport(
    client: TestClient, provider: str, transport: Any
) -> None:
    backend = client.app.state.route_backends[provider]
    if isinstance(backend, ResponsesBackend):
        client.app.state.route_backends[provider] = ResponsesBackend(
            transport=transport,
            adapt_payload=backend.adapt_payload,
            adapt_probe_payload=backend.adapt_probe_payload,
            signature_namespace=backend.signature_namespace,
            catalog_loader=(
                transport.list_models if backend.catalog_loader is not None else None
            ),
        )
        return
    assert isinstance(backend, AnthropicBackend)
    client.app.state.route_backends[provider] = AnthropicBackend(
        transport=transport,
        header_policy=backend.header_policy,
        error_policy=backend.error_policy,
        token_counter=backend.token_counter,
        catalog_loader=(
            transport.list_models if backend.catalog_loader is not None else None
        ),
    )


def test_codex_client_list_models_filters_hidden_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["client_version"] == "0.146.0"
        assert request.headers["Chatgpt-Account-Id"] == "account"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.6-sol", "visibility": "list"},
                    {"slug": "codex-auto-review", "visibility": "hide"},
                    {"slug": "gpt-5.5"},
                ]
            },
        )

    async def run() -> list[str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            return await CodexClient(AvailableCodexAuthManager(), http_client).list_models()

    assert asyncio.run(run()) == ["gpt-5.6-sol", "gpt-5.5"]


class TestAdminDashboardApi:
    """Codex catalog proxy and connection-test endpoints behind the admin guard."""

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any) -> TestClient:
        return _create_test_client(
            monkeypatch, tmp_path, base_url="http://127.0.0.1:8787", **kwargs
        )

    def test_codex_models_returns_visible_slugs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        selected_transport = SelectedResponsesModelTransport(
            ["gpt-5.6-sol", "gpt-5.5"]
        )
        with self._client(monkeypatch, tmp_path, codex_client=CatalogCodexClient) as client:
            _select_route_transport(client, "codex", selected_transport)
            assert selected_transport is not client.app.state.codex_client
            response = client.get("/admin/providers/codex/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["gpt-5.6-sol", "gpt-5.5"]}
        assert selected_transport.calls == 1

    def test_codex_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=FailingCatalogCodexClient) as client:
            response = client.get("/admin/providers/codex/models")

        assert response.status_code == 400
        assert response.json()["error"]["message"] == "unsupported client"

    def test_codex_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, codex_client=CatalogCodexClient) as client:
            assert client.get("/admin/providers/codex/models").status_code == 403

    def test_kimi_models_relays_catalog_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The catalog passes through unshaped: the map bypasses model IDs
        # untouched, so the raw backend answer is the preset source.
        selected_transport = SelectedAnthropicModelTransport(
            {"data": [{"id": "k2.5"}, {"id": "k3"}]}
        )
        with self._client(monkeypatch, tmp_path, kimi_client=CatalogKimiClient) as client:
            _select_route_transport(client, "kimi", selected_transport)
            assert selected_transport is not client.app.state.kimi_client
            response = client.get("/admin/providers/kimi/models")

        assert response.status_code == 200
        assert response.json() == {"data": [{"id": "k2.5"}, {"id": "k3"}]}
        assert selected_transport.calls == 1

    def test_kimi_models_without_catalog_capability_performs_no_io(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        selected_transport = SelectedAnthropicModelTransport({"data": []})
        with self._client(monkeypatch, tmp_path) as client:
            backend = client.app.state.route_backends["kimi"]
            assert isinstance(backend, AnthropicBackend)
            client.app.state.route_backends["kimi"] = AnthropicBackend(
                transport=selected_transport,
                header_policy=backend.header_policy,
                error_policy=backend.error_policy,
                token_counter=backend.token_counter,
                catalog_loader=None,
            )
            response = client.get("/admin/providers/kimi/models")

        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found_error"
        assert selected_transport.calls == 0

    def test_kimi_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, kimi_client=FailingCatalogKimiClient) as client:
            response = client.get("/admin/providers/kimi/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_kimi_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, kimi_client=CatalogKimiClient) as client:
            assert client.get("/admin/providers/kimi/models").status_code == 403

    def test_grok_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        selected_transport = SelectedResponsesModelTransport(
            ["grok-4.5", "grok-4.3"]
        )
        with self._client(monkeypatch, tmp_path, grok_client=CatalogGrokClient) as client:
            _select_route_transport(client, "grok", selected_transport)
            assert selected_transport is not client.app.state.grok_client
            response = client.get("/admin/providers/grok/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["grok-4.5", "grok-4.3"]}
        assert selected_transport.calls == 1

    def test_grok_models_relays_upstream_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=FailingCatalogGrokClient) as client:
            response = client.get("/admin/providers/grok/models")

        assert response.status_code == 401
        assert response.json()["error"]["message"] == "token expired"

    def test_grok_models_refuses_foreign_host(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with _create_test_client(monkeypatch, tmp_path, grok_client=CatalogGrokClient) as client:
            assert client.get("/admin/providers/grok/models").status_code == 403

    def test_custom_provider_models_returns_catalog_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        selected_transport = SelectedResponsesModelTransport(
            ["gpt-5.5", "gemini-3.1-pro"]
        )
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            custom_client=CatalogOpenAICompatibleClient,
        ) as client:
            _select_route_transport(client, "wrtn", selected_transport)
            assert selected_transport is not client.app.state.custom_provider_clients["wrtn"]
            response = client.get("/admin/providers/custom/wrtn/models")

        assert response.status_code == 200
        assert response.json() == {"models": ["gpt-5.5", "gemini-3.1-pro"]}
        assert selected_transport.calls == 1

    def test_custom_provider_models_preserves_api_key_redaction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.headers.get("Authorization") != f"Bearer {_CUSTOM_API_KEY}":
                pytest.fail("the custom catalog request used the wrong credential")
            return httpx.Response(
                503,
                json={"error": {"message": f"credential {_CUSTOM_API_KEY} rejected"}},
            )

        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        selected_transport = OpenAICompatibleClient(
            "wrtn", _custom_provider(), http_client
        )
        try:
            with self._client(monkeypatch, tmp_path, config=config) as client:
                _select_route_transport(client, "wrtn", selected_transport)
                response = client.get("/admin/providers/custom/wrtn/models")
        finally:
            asyncio.run(http_client.aclose())

        _assert_secret_absent(_CUSTOM_API_KEY, response.text)
        assert response.status_code == 503
        assert response.json()["error"]["message"] == "credential [REDACTED] rejected"

    def test_custom_provider_models_redacts_raw_secret_bearing_http_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        selected_transport = SecretBearingCatalogTransport(_CUSTOM_API_KEY)
        with self._client(monkeypatch, tmp_path, config=config) as client:
            _select_route_transport(client, "wrtn", selected_transport)
            response = client.get("/admin/providers/custom/wrtn/models")

        _assert_secret_absent(_CUSTOM_API_KEY, response.text, caplog.text)
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "server_error"
        assert "ConnectError" in response.json()["error"]["message"]
        assert "[REDACTED]" in response.json()["error"]["message"]
        assert selected_transport.calls == 1

    def test_custom_provider_models_returns_404_for_unknown_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(monkeypatch, tmp_path, config=config) as client:
            response = client.get("/admin/providers/custom/other/models")

        assert response.status_code == 404
        assert "not configured" in response.json()["error"]["message"]

    def test_connection_test_ok(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        selected_transport = ProbeCodexClient()
        config = GatewayConfig(codex_service_tier="fast")
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            codex_client=FailingFastTierLookupCodexClient,
        ) as client:
            _select_route_transport(client, "codex", selected_transport)
            assert selected_transport is not client.app.state.codex_client
            response = client.post(
                "/admin/test",
                json={"target": "codex:gpt-5.6-luna"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "gpt-5.6-luna"
        assert isinstance(result["latency_ms"], int)
        assert len(selected_transport.payloads) == 1
        assert "service_tier" not in selected_transport.payloads[0]

    def test_connection_test_rejects_bare_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=ProbeCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "gpt-5.6-luna"},
            )

        assert response.status_code == 400
        assert "no provider prefix" in response.json()["error"]["message"]

    def test_connection_test_kimi_target_probes_kimi(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        selected_transport = ProbeKimiClient()
        with self._client(monkeypatch, tmp_path) as client:
            _select_route_transport(client, "kimi", selected_transport)
            assert selected_transport is not client.app.state.kimi_client
            response = client.post(
                "/admin/test",
                json={"target": "kimi:k3"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "k3"
        assert selected_transport.headers is not None
        assert "oauth-2025-04-20" in selected_transport.headers["anthropic-beta"]
        assert selected_transport.response is not None
        assert selected_transport.response.aclose_calls == 1

    def test_connection_test_static_anthropic_uses_policy_and_bearer_auth(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        outbound_headers: httpx.Headers | None = None
        outbound_url: str | None = None
        response_stream = TrackedProbeStream(
            json.dumps({"type": "message", "model": "claude-upstream"}).encode()
        )
        upstream_response = TrackedProbeResponse(200, response_stream)

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal outbound_headers, outbound_url
            outbound_headers = request.headers
            outbound_url = str(request.url)
            return upstream_response

        provider = _anthropic_custom_provider()
        config = GatewayConfig(custom_providers={"openai-by-name": provider})
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AnthropicCompatibleClient(
            "openai-by-name", provider, http_client
        )
        try:
            with self._client(monkeypatch, tmp_path, config=config) as client:
                _select_route_transport(client, "openai-by-name", transport)
                response = client.post(
                    "/admin/test",
                    json={"target": "openai-by-name:configured-target"},
                    headers={
                        "anthropic-beta": "oauth-2025-04-20,feature-2026-01-01"
                    },
                )
        finally:
            asyncio.run(http_client.aclose())

        _assert_secret_absent(
            _ANTHROPIC_CUSTOM_API_KEY, response.text, outbound_url or ""
        )
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "claude-upstream"
        assert outbound_headers is not None
        if outbound_headers.get("authorization") != (
            f"Bearer {_ANTHROPIC_CUSTOM_API_KEY}"
        ):
            pytest.fail("the static probe did not apply its configured Bearer credential")
        assert outbound_headers.get("anthropic-beta") == "feature-2026-01-01"
        assert response_stream.read_calls == 1
        assert response_stream.close_calls == 1
        assert upstream_response.aclose_calls == 1
        assert outbound_url is not None

    @pytest.mark.parametrize(
        "upstream_body",
        [
            pytest.param(b"", id="empty-body"),
            pytest.param(b"not JSON", id="non-json-body"),
            pytest.param(b'"message"', id="json-scalar"),
            pytest.param(b"[]", id="json-list"),
            pytest.param(
                b'{"type":"completion","model":"claude-upstream"}',
                id="wrong-type",
            ),
            pytest.param(
                b'{"model":"claude-upstream"}',
                id="missing-type",
            ),
            pytest.param(b'{"type":"message"}', id="missing-model"),
            pytest.param(
                b'{"type":"message","model":123}',
                id="non-string-model",
            ),
            pytest.param(
                b'{"type":"message","model":""}',
                id="empty-model",
            ),
            pytest.param(
                b'{"type":"message","model":"   "}',
                id="whitespace-model",
            ),
            pytest.param(
                b'{"type":"message","model":"claude-upstream","count":'
                + b"9" * 5000
                + b"}",
                id="numeric-parser-limit",
            ),
            pytest.param(
                b"[" * 5000 + b"0" + b"]" * 5000,
                id="recursion-parser-limit",
            ),
        ],
    )
    def test_connection_test_rejects_malformed_anthropic_success_envelope(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        upstream_body: bytes,
    ) -> None:
        response_stream = TrackedProbeStream(upstream_body)
        upstream_response = TrackedProbeResponse(200, response_stream)
        selected_transport = ProbeAnthropicTransport(upstream_response)
        config = GatewayConfig(
            custom_providers={"messages-api": _anthropic_custom_provider()}
        )

        with self._client(monkeypatch, tmp_path, config=config) as client:
            _select_route_transport(client, "messages-api", selected_transport)
            response = client.post(
                "/admin/test",
                json={"target": "messages-api:claude-upstream"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 502
        assert result["response_model"] is None
        assert (
            result["detail"]
            == "upstream returned an invalid Anthropic Messages response"
        )
        assert response_stream.read_calls == 1
        assert response_stream.close_calls == 1
        assert upstream_response.aclose_calls == 1

    def test_connection_test_rejects_proprietary_anthropic_200_without_disclosure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        provider = _anthropic_custom_provider()
        proprietary_message = f"unsupported target with credential {provider.api_key}"
        provider_key = "private upstream provider key"
        generated_content = "private generated probe output"
        upstream_body = json.dumps(
            {
                "code": 400,
                "msg": proprietary_message,
                "success": False,
                "provider_key": provider_key,
                "content": generated_content,
            }
        ).encode()
        response_stream = TrackedProbeStream(upstream_body)
        upstream_response = TrackedProbeResponse(200, response_stream)
        selected_transport = ProbeAnthropicTransport(upstream_response)
        config = GatewayConfig(custom_providers={"glm": provider})

        with self._client(monkeypatch, tmp_path, config=config) as client:
            _select_route_transport(client, "glm", selected_transport)
            response = client.post(
                "/admin/test",
                json={"target": "glm:glm-5.anything"},
            )

        _assert_secret_absent(provider.api_key, response.text, caplog.text)
        _assert_secret_absent(proprietary_message, response.text, caplog.text)
        _assert_secret_absent(provider_key, response.text, caplog.text)
        _assert_secret_absent(generated_content, response.text, caplog.text)
        assert response.status_code == 200
        assert response.json() == {
            "ok": False,
            "status": 502,
            "latency_ms": response.json()["latency_ms"],
            "target": "glm:glm-5.anything",
            "response_model": None,
            "detail": "upstream returned an invalid Anthropic Messages response",
        }
        assert isinstance(response.json()["latency_ms"], int)
        assert response_stream.read_calls == 1
        assert response_stream.close_calls == 1
        assert upstream_response.aclose_calls == 1

    @pytest.mark.parametrize("upstream_status", [401, 503])
    def test_connection_test_static_non_success_is_transport_closed_and_redacted(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        upstream_status: int,
    ) -> None:
        response_stream = TrackedProbeStream(
            json.dumps(
                {
                    "error": {
                        "message": (
                            f"credential {_ANTHROPIC_CUSTOM_API_KEY} rejected"
                        )
                    }
                }
            ).encode()
        )
        upstream_response = TrackedProbeResponse(upstream_status, response_stream)

        async def handler(_request: httpx.Request) -> httpx.Response:
            return upstream_response

        provider = _anthropic_custom_provider()
        config = GatewayConfig(custom_providers={"messages-api": provider})
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        transport = AnthropicCompatibleClient("messages-api", provider, http_client)
        try:
            with self._client(monkeypatch, tmp_path, config=config) as client:
                _select_route_transport(client, "messages-api", transport)
                response = client.post(
                    "/admin/test",
                    json={"target": "messages-api:claude-upstream"},
                )
        finally:
            asyncio.run(http_client.aclose())

        _assert_secret_absent(
            _ANTHROPIC_CUSTOM_API_KEY, response.text, caplog.text
        )
        result = response.json()
        assert result["ok"] is False
        assert result["status"] == upstream_status
        assert result["detail"] == "credential [REDACTED] rejected"
        assert response_stream.read_calls == 1
        assert response_stream.close_calls == 1
        assert upstream_response.aclose_calls == 1

    def test_connection_test_preserves_read_error_when_close_also_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        provider = _anthropic_custom_provider()
        config = GatewayConfig(custom_providers={"messages-api": provider})
        upstream_response = ReadAndCloseFailingProbeResponse(provider.api_key)
        selected_transport = ReadAndCloseFailingAnthropicTransport(
            upstream_response
        )
        with self._client(monkeypatch, tmp_path, config=config) as client:
            _select_route_transport(client, "messages-api", selected_transport)
            response = client.post(
                "/admin/test",
                json={"target": "messages-api:claude-upstream"},
            )

        _assert_secret_absent(provider.api_key, response.text, caplog.text)
        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is False
        assert result["status"] is None
        assert "ReadError" in result["detail"]
        assert "primary read failed" in result["detail"]
        assert "cleanup close failed" not in result["detail"]
        assert upstream_response.aclose_calls == 1

    def test_connection_test_reports_kimi_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, kimi_client=RejectingKimiClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kimi:k3"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 404
        assert "model not found" in result["detail"]

    def test_connection_test_grok_target_probes_grok(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        translated_payload = {
            "model": "grok-4.5",
            "reasoning": {"effort": "future-effort"},
            "service_tier": "priority",
            "prompt_cache_key": "grok-probe",
        }
        monkeypatch.setattr(
            admin_system,
            "translate_claude_request_to_codex",
            lambda *_args: translated_payload,
        )
        selected_transport = ProbeGrokClient()
        with self._client(monkeypatch, tmp_path) as client:
            _select_route_transport(client, "grok", selected_transport)
            assert selected_transport is not client.app.state.grok_client
            response = client.post(
                "/admin/test",
                json={"target": "grok:grok-4.5"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "grok-4.5"
        assert len(selected_transport.payloads) == 1
        probe_payload = selected_transport.payloads[0]
        assert probe_payload is not translated_payload
        assert "service_tier" not in probe_payload
        assert probe_payload["reasoning"] is translated_payload["reasoning"]
        assert translated_payload["reasoning"]["effort"] == "medium"

    def test_connection_test_reports_grok_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, grok_client=RejectingGrokClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "grok:grok-nope"},
            )

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert "model_not_found" in result["detail"]

    def test_connection_test_custom_target_probes_custom_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        translated_payload = {
            "model": "gpt-5.5",
            "prompt_cache_key": "custom-probe",
        }
        monkeypatch.setattr(
            admin_system,
            "translate_claude_request_to_codex",
            lambda *_args: translated_payload,
        )
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        selected_transport = ProbeOpenAICompatibleClient("wrtn", _custom_provider())
        with self._client(monkeypatch, tmp_path, config=config) as client:
            _select_route_transport(client, "wrtn", selected_transport)
            assert selected_transport is not client.app.state.custom_provider_clients["wrtn"]
            response = client.post("/admin/test", json={"target": "wrtn:gpt-5.5"})

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["response_model"] == "gpt-5.5"
        assert selected_transport.payloads == [translated_payload]
        assert selected_transport.payloads[0] is translated_payload

    def test_connection_test_reports_custom_provider_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config = GatewayConfig(custom_providers={"wrtn": _custom_provider()})
        with self._client(
            monkeypatch,
            tmp_path,
            config=config,
            custom_client=RejectingOpenAICompatibleClient,
        ) as client:
            response = client.post("/admin/test", json={"target": "wrtn:gpt-nope"})

        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert result["detail"] == "model_not_found"

    def test_connection_test_rejects_unknown_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            response = client.post(
                "/admin/test",
                json={"target": "kim:k3"},
            )

        assert response.status_code == 400
        assert "unknown provider prefix" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        ("target", "expected_detail"),
        [
            pytest.param(
                " ", "target must be a non-empty string", id="empty-target"
            ),
            pytest.param(
                "missing:model", "route target is invalid", id="unknown-prefix"
            ),
        ],
    )
    def test_connection_target_errors_hide_legacy_provider_name_overlap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        target: str,
        expected_detail: str,
    ) -> None:
        credential = "route-sensitive-key"
        unsafe_suffix = "unsafe-provider-detail"
        provider_name = f"legacy-{credential}-{unsafe_suffix}"
        provider = OpenAICompatibleProvider(
            wire_api="responses",
            base_url="https://responses.example/v1",
            api_key=credential,
        )
        config = GatewayConfig(custom_providers={provider_name: provider})
        with self._client(monkeypatch, tmp_path, config=config) as client:
            response = client.post("/admin/test", json={"target": target})

        _assert_secret_absent(credential, response.text, caplog.text)
        if unsafe_suffix in response.text or unsafe_suffix in caplog.text:
            pytest.fail("a legacy custom-provider routing detail was exposed")
        assert response.status_code == 400
        assert response.json()["error"]["message"] == expected_detail

    def test_connection_test_reports_unknown_codex_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path, codex_client=RejectingCodexClient) as client:
            response = client.post(
                "/admin/test",
                json={"target": "codex:gpt-nope"},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["ok"] is False
        assert result["status"] == 400
        assert result["detail"] == "model_not_found"

    def test_connection_test_validates_input(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        with self._client(monkeypatch, tmp_path) as client:
            empty_target = client.post(
                "/admin/test",
                json={"target": " "},
            )
            wrong_content_type = client.post(
                "/admin/test",
                content='{"target": "codex:x"}',
                headers={"content-type": "text/plain"},
            )

        assert empty_target.status_code == 400
        assert wrong_content_type.status_code == 415


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

    def test_list_returns_only_registry_rows_without_secrets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()

        with client:
            response = client.get("/admin/providers/claude/accounts")

        assert response.status_code == 200
        payload = response.json()
        # The collection alone: local login, serving pin, and cooldown
        # telemetry each live at their own endpoint now.
        assert set(payload) == {"accounts"}
        [row] = payload["accounts"]
        assert set(row) == {
            "id",
            "email",
            "organizationUuid",
            "organizationName",
            "createdAt",
            "updatedAt",
            "lastAuthenticatedAt",
            "state",
            "accountIncarnationId",
            "upstreamAccountUuid",
            "planType",
            "rateLimitTier",
        }
        assert row["id"] == account_id
        assert row["state"] == "ready"
        # _register_serving_account's oauth-account fixture has no plan
        # fields, so the derived plan metadata degrades to nulls.
        assert row["planType"] is None
        assert row["rateLimitTier"] is None
        # The registry response must never leak credential material.
        assert "accessToken" not in response.text
        assert "refreshToken" not in response.text
        assert "pool-access" not in response.text
        assert "pool-refresh" not in response.text


    def test_list_derives_plan_fields_from_the_captured_oauth_account(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        claude_accounts.add_account(
            email="max@example.com",
            organization_uuid="org-max",
            organization_name="Max Org",
            credentials_json={"claudeAiOauth": {"accessToken": "at"}},
            oauth_account_json={
                "emailAddress": "max@example.com",
                "organizationType": "claude_max",
                "organizationRateLimitTier": "default_claude_max_20x",
            },
        )

        with client:
            [row] = client.get("/admin/providers/claude/accounts").json()["accounts"]

        assert row["planType"] == "claude_max"
        assert row["rateLimitTier"] == "default_claude_max_20x"

    def test_local_reports_the_ambient_login_identity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The dashboard's local CLI login hero reads this block: identity and
        # plan metadata from the CLI's own ~/.claude.json — never secrets.
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        client = self._client(monkeypatch, tmp_path)
        home = tmp_path / "home"
        home.mkdir(parents=True, exist_ok=True)
        (home / ".claude.json").write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "accountUuid": "11111111-2222-3333-4444-555555555555",
                        "emailAddress": "local@example.com",
                        "organizationName": "Local Org",
                        "organizationType": "claude_max",
                        "organizationRateLimitTier": "default_claude_max_20x",
                    }
                }
            ),
            encoding="utf-8",
        )

        with client:
            payload = client.get("/admin/providers/claude/local").json()

        assert payload["local"] == {
            "accountUuid": "11111111-2222-3333-4444-555555555555",
            "email": "local@example.com",
            "organizationName": "Local Org",
            "planType": "claude_max",
            "rateLimitTier": "default_claude_max_20x",
        }

    def test_local_degrades_to_null_without_a_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        client = self._client(monkeypatch, tmp_path)

        with client:
            payload = client.get("/admin/providers/claude/local").json()

        assert payload == {"local": None}

    def test_usage_serves_from_the_cache_after_the_first_fetch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        calls: list[str] = []

        async def fake_fetch(http_client: Any, manager: Any) -> tuple[dict[str, Any], None]:
            calls.append(account_id)
            return ({"provider": "claude", "status": "ok", "error": None}, None)

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fake_fetch)

        with client:
            first = client.get("/admin/providers/claude/pool/usage")
            second = client.get("/admin/providers/claude/pool/usage")

        assert first.status_code == 200
        assert first.json()["accounts"][account_id]["status"] == "ok"
        assert second.json()["accounts"][account_id]["status"] == "ok"
        assert calls == [account_id]

    def test_usage_skips_needs_reauth_accounts_without_fetching(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        account_id = _register_serving_account()
        claude_accounts.mark_account_needs_reauth(account_id)

        async def fail_fetch(http_client: Any, manager: Any) -> None:
            raise AssertionError("needs-reauth accounts must not fetch usage")

        monkeypatch.setattr(server_support, "fetch_claude_account_usage", fail_fetch)

        with client:
            response = client.get("/admin/providers/claude/pool/usage")

        assert response.status_code == 200
        result = response.json()["accounts"][account_id]
        assert result["status"] == "unavailable"
        assert "re-authentication" in result["error"]

    def test_usage_unknown_account_filter_is_a_400(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        _register_serving_account()
        with client:
            response = client.get(
                "/admin/providers/claude/pool/usage",
                params={"account": "99999999-9999-4999-8999-999999999999"},
            )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/admin/providers/claude/accounts"),
            ("GET", "/admin/providers/claude/pool/usage"),
            ("GET", "/admin/providers/claude/login"),
            ("POST", "/admin/providers/claude/login"),
            ("DELETE", "/admin/providers/claude/login"),
            ("POST", "/admin/providers/claude/login/code"),
            ("POST", "/admin/providers/claude/login/replace"),
        ],
    )
    def test_foreign_host_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        method: str,
        path: str,
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        client = _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url="http://evil.example",
        )
        with client:
            response = client.request(method, path, json={})
        assert response.status_code == 403

    def test_login_post_requires_json_content_type(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # code/replace content-type handling is covered in the lifecycle
        # class — those endpoints check the attempt guard first, so they
        # need an attached session to reach the 415.
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.post(
                "/admin/providers/claude/login",
                content="{}",
                headers={"content-type": "text/plain"},
            )
        assert response.status_code == 415

    def test_login_get_without_a_session_is_idle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            assert client.get("/admin/providers/claude/login").json() == {"status": "idle"}

    def test_login_commands_without_a_session_are_409_stale_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # With no session there is no attempt any command could name — the
        # guard answers before body or state validation ever runs.
        client = self._client(monkeypatch, tmp_path)
        with client:
            code = client.post("/admin/providers/claude/login/code", json={"code": "x"})
            confirm = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
            )
            cancel = client.delete("/admin/providers/claude/login")
        for response in (code, confirm, cancel):
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "stale_login"

    def test_login_post_rejects_a_non_empty_body(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.post("/admin/providers/claude/login", json={"mode": "x"})
        assert response.status_code == 400

    def test_login_post_conflicts_with_a_held_capture_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        from claudex.claude.login_session import capture_lock_path
        from claudex.locking import try_file_lock as _try_lock

        handle = _try_lock(capture_lock_path())
        assert handle is not None
        try:
            with client:
                response = client.post("/admin/providers/claude/login", json={})
        finally:
            handle.release()
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "login-locked"


    def test_account_delete_unknown_id_is_404(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        client = self._client(monkeypatch, tmp_path)
        with client:
            response = client.delete(
                "/admin/providers/claude/accounts/99999999-9999-4999-8999-999999999999"
            )
        assert response.status_code == 404

    def test_account_delete_serving_account_is_409(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        account_id = _register_serving_account()
        client = self._client(monkeypatch, tmp_path, claude_account_id=account_id)

        with client:
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")

        assert response.status_code == 409
        assert "pool/serving" in response.json()["error"]["message"]
        assert [record.id for record in claude_accounts.list_accounts()] == [account_id]

    def test_account_delete_requires_the_admin_guard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        account_id = "99999999-9999-4999-8999-999999999999"
        client = self._client(monkeypatch, tmp_path, local_token="secret")
        with client:
            response = client.delete(f"/admin/providers/claude/accounts/{account_id}")
        assert response.status_code == 401


# a request to an old path must 404, never silently hit a moved handler.
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/mapping"),
        ("PUT", "/admin/mapping"),
        ("GET", "/admin/log-level"),
        ("PUT", "/admin/log-level"),
        ("GET", "/admin/compaction"),
        ("PUT", "/admin/compaction"),
        ("GET", "/admin/claude-account"),
        ("PUT", "/admin/claude-account"),
        ("GET", "/admin/claude-accounts"),
        ("GET", "/admin/claude-accounts/usage"),
        ("GET", "/admin/claude-accounts/login"),
        ("POST", "/admin/claude-accounts/login"),
        ("DELETE", "/admin/claude-accounts/login"),
        ("POST", "/admin/claude-accounts/login/code"),
        ("POST", "/admin/claude-accounts/login/confirm"),
        ("GET", "/admin/codex/models"),
        ("POST", "/admin/codex/reset-credit"),
        ("GET", "/admin/kimi/models"),
        ("GET", "/admin/grok/models"),
    ],
)
def test_pre_reorg_admin_paths_are_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str, path: str
) -> None:
    with _create_test_client(monkeypatch, tmp_path, base_url=_ADMIN_BASE) as client:
        response = client.request(method, path, json={})
    assert response.status_code == 404


class TestAdminClaudeLoginLifecycle:
    """End-to-end login sessions against the PATH-prepended fake claude.

    Every test runs the client as a context manager: the login driver task
    lives on the lifespan portal's event loop, which only exists while the
    `with` block is open.
    """

    @staticmethod
    def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
        import sys as _sys

        from fake_claude import prepend_fake_claude

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("CLAUDEX_CLAUDE_ACCOUNT_ID", raising=False)
        monkeypatch.setattr(_sys, "platform", "linux")
        prepend_fake_claude(monkeypatch, tmp_path)
        return _create_test_client(
            monkeypatch, tmp_path,
            config=GatewayConfig(settings_file=tmp_path / "settings.json"),
            base_url=_ADMIN_BASE,
        )

    @staticmethod
    def _poll_until(client: TestClient, predicate: Any, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = client.get("/admin/providers/claude/login").json()
            if predicate(status):
                return status
            time.sleep(0.05)
        raise AssertionError(f"login status never satisfied the predicate: {status}")

    def test_full_login_flow_with_code_submission(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-url-code")
        client = self._client(monkeypatch, tmp_path)

        with client:
            started = client.post("/admin/providers/claude/login", json={})
            assert started.status_code == 201
            envelope = started.json()
            assert envelope["status"] == "starting"
            attempt = envelope["attempt_id"]
            attempt_header = {"X-Login-Attempt": attempt}

            status = self._poll_until(
                client, lambda s: s["status"] == "awaiting-browser"
            )
            assert status["attempt_id"] == attempt
            assert status["url"].startswith("https://claude.com/cai/oauth/authorize")
            assert status["code_prompt_detected"] is True
            assert status["expires_at"] is not None

            submitted = client.post(
                "/admin/providers/claude/login/code",
                json={"code": "good-code"},
                headers=attempt_header,
            )
            assert submitted.status_code == 200

            status = self._poll_until(client, lambda s: s["status"] == "succeeded")
            assert status["account"]["email"] == "fixture@example.com"

            rows = client.get("/admin/providers/claude/accounts").json()["accounts"]
            assert [row["email"] for row in rows] == ["fixture@example.com"]

    def test_second_login_while_active_is_409_then_cancel_converges(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            attempt_header = {"X-Login-Attempt": attempt}
            conflict = client.post("/admin/providers/claude/login", json={})
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "login-active"

            cancelled = client.delete(
                "/admin/providers/claude/login", headers=attempt_header
            )
            assert cancelled.json() == {"status": "cancelling"}
            self._poll_until(client, lambda s: s["status"] == "cancelled")

            # A terminal session clears on DELETE and frees the slot.
            assert client.delete(
                "/admin/providers/claude/login", headers=attempt_header
            ).json() == {"status": "idle"}
            assert client.get("/admin/providers/claude/login").json() == {
                "status": "idle"
            }

    def test_stale_attempt_commands_are_409_stale_login(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            stale_header = {"X-Login-Attempt": "0" * 32}

            for response in (
                client.get("/admin/providers/claude/login", headers=stale_header),
                client.post(
                    "/admin/providers/claude/login/code",
                    json={"code": "late-code"},
                    headers=stale_header,
                ),
                client.post(
                    "/admin/providers/claude/login/replace",
                    json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
                    headers=stale_header,
                ),
                client.delete("/admin/providers/claude/login", headers=stale_header),
            ):
                assert response.status_code == 409
                assert response.json()["error"]["code"] == "stale_login"

            # A bare GET is discovery: it still answers, exposing the live
            # attempt so a fresh tab can re-attach.
            bare = client.get("/admin/providers/claude/login")
            assert bare.status_code == 200
            assert bare.json()["attempt_id"] == attempt

            # The stale commands drove nothing: the session is still alive.
            client.delete(
                "/admin/providers/claude/login", headers={"X-Login-Attempt": attempt}
            )
            self._poll_until(client, lambda s: s["status"] == "cancelled")

    def test_duplicate_login_confirm_replace_via_the_api(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-autocomplete")
        client = self._client(monkeypatch, tmp_path)
        original = claude_accounts.add_account(
            "fixture@example.com",
            None,
            None,
            {"claudeAiOauth": {"accessToken": "old-token"}},
            None,
        )

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            status = self._poll_until(
                client, lambda s: s["status"] == "awaiting-replace"
            )
            assert status["email"] == "fixture@example.com"
            assert status["existing_account_id"] == original.id

            mismatched = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": "0a1b2c3d-4e5f-4678-9abc-def012345678"},
                headers={"X-Login-Attempt": attempt},
            )
            assert mismatched.status_code == 409
            assert "does not match" in mismatched.json()["error"]["message"]

            confirmed = client.post(
                "/admin/providers/claude/login/replace",
                json={"existing_account_id": original.id},
                headers={"X-Login-Attempt": attempt},
            )
            assert confirmed.status_code == 200
            status = self._poll_until(client, lambda s: s["status"] == "succeeded")
            assert status["account"]["id"] == original.id

        [record] = claude_accounts.load_registry()
        assert record.id == original.id
        assert record.state == "ready"

    def test_attached_login_code_body_validation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Body validation sits behind the attempt guard, so it is exercised
        # through an attached session.
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "hang")
        client = self._client(monkeypatch, tmp_path)

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            attempt_header = {"X-Login-Attempt": attempt}

            wrong_type = client.post(
                "/admin/providers/claude/login/code",
                content="{}",
                headers={**attempt_header, "content-type": "text/plain"},
            )
            assert wrong_type.status_code == 415
            wrong_type = client.post(
                "/admin/providers/claude/login/replace",
                content="{}",
                headers={**attempt_header, "content-type": "text/plain"},
            )
            assert wrong_type.status_code == 415

            for bad_body in (
                {},
                {"code": ""},
                {"code": "  "},
                {"code": "line\nbreak"},
                {"code": 7},
                {"code": "ok", "extra": 1},
            ):
                response = client.post(
                    "/admin/providers/claude/login/code",
                    json=bad_body,
                    headers=attempt_header,
                )
                assert response.status_code == 400, bad_body

            for bad_body in (
                {},
                {"existing_account_id": ""},
                {"existing_account_id": None},
                {"existing_account_id": 7},
                {"replace": True},
                {"existing_account_id": "x", "extra": 1},
            ):
                response = client.post(
                    "/admin/providers/claude/login/replace",
                    json=bad_body,
                    headers=attempt_header,
                )
                assert response.status_code == 400, bad_body

            client.delete("/admin/providers/claude/login", headers=attempt_header)
            self._poll_until(client, lambda s: s["status"] == "cancelled")

    def test_duplicate_login_decline_is_delete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CLAUDEX_FAKE_CLAUDE_MODE", "piped-autocomplete")
        client = self._client(monkeypatch, tmp_path)
        original = claude_accounts.add_account(
            "fixture@example.com",
            None,
            None,
            {"claudeAiOauth": {"accessToken": "old-token"}},
            None,
        )

        with client:
            attempt = client.post("/admin/providers/claude/login", json={}).json()[
                "attempt_id"
            ]
            self._poll_until(client, lambda s: s["status"] == "awaiting-replace")

            declined = client.delete(
                "/admin/providers/claude/login",
                headers={"X-Login-Attempt": attempt},
            )
            assert declined.json() == {"status": "cancelling"}
            self._poll_until(client, lambda s: s["status"] == "cancelled")

        [record] = claude_accounts.load_registry()
        assert record == original
