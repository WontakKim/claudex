"""Tests for gateway configuration sources and model routing."""

import json
from pathlib import Path

import pytest

from claudex.config import (
    BUILTIN_ROUTE_PROVIDERS,
    AnthropicCompatibleProvider,
    ConfigError,
    GatewayConfig,
    OpenAICompatibleProvider,
    RouteTarget,
    parse_claude_account_id,
    parse_claude_account_routing,
    parse_compaction_model,
    parse_route_target,
    update_settings_file,
)


def test_max_reasoning_effort_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_REASONING_EFFORT", "max")
    assert GatewayConfig.from_env().reasoning_effort_override == "max"


def test_codex_fast_service_tier_is_accepted_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", "fast")
    assert GatewayConfig.from_env().codex_service_tier == "fast"


def test_invalid_codex_service_tier_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", "priority")
    with pytest.raises(ConfigError, match="CLAUDEX_CODEX_SERVICE_TIER"):
        GatewayConfig.from_env()


def test_empty_codex_service_tier_env_disables_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", "")
    assert GatewayConfig.from_env().codex_service_tier is None


def test_log_level_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_LOG_LEVEL", "DEBUG")
    assert GatewayConfig.from_env().log_level == "debug"


def test_invalid_log_level_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_LOG_LEVEL", "verbose")
    with pytest.raises(ConfigError, match="CLAUDEX_LOG_LEVEL"):
        GatewayConfig.from_env()


@pytest.mark.parametrize("port", ["0", "65536"])
def test_port_must_be_in_tcp_range(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    monkeypatch.setenv("CLAUDEX_PORT", port)
    with pytest.raises(ConfigError, match="between 1 and 65535"):
        GatewayConfig.from_env()


def test_non_loopback_bind_requires_local_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_HOST", "0.0.0.0")
    monkeypatch.delenv("CLAUDEX_LOCAL_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="CLAUDEX_LOCAL_TOKEN"):
        GatewayConfig.from_env()


def test_non_loopback_bind_accepts_local_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_HOST", "0.0.0.0")
    monkeypatch.setenv("CLAUDEX_LOCAL_TOKEN", "secret")
    assert GatewayConfig.from_env().host == "0.0.0.0"


def test_model_names_must_not_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"":"codex:gpt-5.6-sol"}')
    with pytest.raises(ConfigError, match="CLAUDEX_MODEL_MAP"):
        GatewayConfig.from_env()


def test_context_window_map_is_accepted_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDEX_CONTEXT_WINDOW_MAP", '{"codex:gpt-5.6-sol": 872000}')
    assert GatewayConfig.from_env().context_window_map == {"codex:gpt-5.6-sol": 872000}


def test_malformed_context_window_map_json_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDEX_CONTEXT_WINDOW_MAP", "{not json")
    with pytest.raises(ConfigError, match="CLAUDEX_CONTEXT_WINDOW_MAP"):
        GatewayConfig.from_env()


@pytest.mark.parametrize("raw", ["[]", '"gpt-5.6-sol"', "872000", "null"])
def test_non_object_context_window_map_fails(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("CLAUDEX_CONTEXT_WINDOW_MAP", raw)
    with pytest.raises(ConfigError, match="CLAUDEX_CONTEXT_WINDOW_MAP"):
        GatewayConfig.from_env()


@pytest.mark.parametrize("value", [None, "872000", True, 0, -1, 872000.5])
def test_context_window_map_values_must_be_positive_integers(
    monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    monkeypatch.setenv(
        "CLAUDEX_CONTEXT_WINDOW_MAP", json.dumps({"codex:gpt-5.6-sol": value})
    )
    with pytest.raises(ConfigError, match="CLAUDEX_CONTEXT_WINDOW_MAP"):
        GatewayConfig.from_env()


def test_context_window_map_keys_must_be_strings(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="context_window_map"):
        GatewayConfig._from_sources(
            {"context_window_map": {1: 872000}}, tmp_path / "settings.json"
        )


@pytest.mark.parametrize("target", ["code:gpt-5.6-sol", "gpt-5.6-sol"])
def test_context_window_map_keys_require_known_provider_prefix(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    monkeypatch.setenv(
        "CLAUDEX_CONTEXT_WINDOW_MAP", json.dumps({target: 872000})
    )

    with pytest.raises(ConfigError) as error:
        GatewayConfig.from_env()

    message = str(error.value)
    assert f"model target {target!r}" in message
    assert "known providers: codex, kimi, grok" in message


class TestSettingsFile:
    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    def test_values_apply_with_native_types(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "port": 9090,
                "model_map": {"haiku": "codex:gpt-5.6-luna"},
            },
        )

        config = GatewayConfig.load(settings_file)

        assert config.port == 9090
        assert config.model_map == {"haiku": "codex:gpt-5.6-luna"}

    def test_context_window_map_applies_with_native_values(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"context_window_map": {"codex:gpt-5.6-sol": 872000}}
        )

        assert GatewayConfig.load(settings_file).context_window_map == {
            "codex:gpt-5.6-sol": 872000
        }

    def test_context_window_map_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"context_window_map": {"codex:gpt-5.6-sol": 272000}}
        )
        monkeypatch.setenv(
            "CLAUDEX_CONTEXT_WINDOW_MAP", '{"codex:gpt-5.6-sol": 872000}'
        )

        assert GatewayConfig.load(settings_file).context_window_map == {
            "codex:gpt-5.6-sol": 872000
        }

    def test_empty_context_window_map_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"context_window_map": {"codex:gpt-5.6-sol": 872000}}
        )
        monkeypatch.setenv("CLAUDEX_CONTEXT_WINDOW_MAP", "")

        assert GatewayConfig.load(settings_file).context_window_map == {}

    @pytest.mark.parametrize(
        "value", [["gpt-5.6-sol"], {"codex:gpt-5.6-sol": True}]
    )
    def test_invalid_context_window_map_setting_fails(
        self, tmp_path: Path, value: object
    ) -> None:
        settings_file = self._write(tmp_path, {"context_window_map": value})

        with pytest.raises(ConfigError, match='settings.json key "context_window_map"'):
            GatewayConfig.load(settings_file)

    def test_null_context_window_map_setting_means_no_overrides(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"context_window_map": None})

        assert GatewayConfig.load(settings_file).context_window_map == {}

    def test_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"reasoning_effort": "low"})
        monkeypatch.setenv("CLAUDEX_REASONING_EFFORT", "high")
        assert GatewayConfig.load(settings_file).reasoning_effort_override == "high"

    def test_codex_fast_service_tier_applies_from_settings_file(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"codex.service_tier": "fast"})
        assert GatewayConfig.load(settings_file).codex_service_tier == "fast"

    def test_nested_registry_groups_apply_from_settings_file(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "codex": {"service_tier": "fast"},
                "compaction": {"model": "claude:claude-opus-5"},
                "claude_account": {"routing": {"mode": "fallback"}},
            },
        )

        config = GatewayConfig.load(settings_file)

        assert config.codex_service_tier == "fast"
        assert config.compaction_model == "claude:claude-opus-5"
        assert config.claude_account_routing_mode == "fallback"

    def test_duplicate_flat_and_nested_registry_key_fails(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "codex.service_tier": "fast",
                "codex": {"service_tier": "fast"},
            },
        )

        with pytest.raises(ConfigError, match="codex\\.service_tier"):
            GatewayConfig.load(settings_file)

    def test_unknown_nested_registry_key_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"codex": {"tier": "fast"}})

        with pytest.raises(
            ConfigError, match="unknown keys: codex\\.tier.*codex\\.service_tier"
        ):
            GatewayConfig.load(settings_file)

    def test_non_object_registry_group_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"codex": "fast"})

        with pytest.raises(ConfigError, match='key "codex" must contain a JSON object'):
            GatewayConfig.load(settings_file)

    def test_update_renders_dotted_key_as_nested_group(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {})

        update_settings_file(settings_file, {"codex.service_tier": "fast"})

        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "codex": {"service_tier": "fast"}
        }

    def test_update_migrates_legacy_dotted_keys_and_preserves_other_keys(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "codex.service_tier": "fast",
                "compaction.model": "claude:claude-opus-5",
                "port": 9090,
            },
        )

        update_settings_file(settings_file, {"log_level": "debug"})

        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "codex": {"service_tier": "fast"},
            "compaction": {"model": "claude:claude-opus-5"},
            "port": 9090,
            "log_level": "debug",
        }

    def test_deleting_last_group_key_removes_group(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path, {"compaction": {"model": "claude:claude-opus-5"}}
        )

        update_settings_file(settings_file, {}, deletions=("compaction.model",))

        assert json.loads(settings_file.read_text(encoding="utf-8")) == {}

    def test_codex_service_tier_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"codex.service_tier": "invalid"})
        monkeypatch.setenv("CLAUDEX_CODEX_SERVICE_TIER", "fast")
        assert GatewayConfig.load(settings_file).codex_service_tier == "fast"

    def test_empty_env_still_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"model_map": {"haiku": "codex:gpt-5.6-luna"}})
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", "")
        assert GatewayConfig.load(settings_file).model_map == {}

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        config = GatewayConfig.load(tmp_path / "settings.json")
        assert config.port == 8787
        assert config.model_map == {}
        assert config.log_level == "info"

    def test_default_location_is_gateway_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        gateway_home = tmp_path / ".claudex"
        gateway_home.mkdir()
        (gateway_home / "settings.json").write_text('{"port": 9317}', encoding="utf-8")
        assert GatewayConfig.load().port == 9317

    def test_invalid_json_fails(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            GatewayConfig.load(settings_file)

    def test_non_object_file_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, ["port"])
        with pytest.raises(ConfigError, match="JSON object"):
            GatewayConfig.load(settings_file)

    def test_unknown_key_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"model_mapp": {}})
        with pytest.raises(ConfigError, match="model_mapp"):
            GatewayConfig.load(settings_file)

    def test_non_numeric_port_fails_with_file_label(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"port": "not-a-port"})
        with pytest.raises(ConfigError, match='settings.json key "port"'):
            GatewayConfig.load(settings_file)

    def test_bool_port_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"port": True})
        with pytest.raises(ConfigError, match="must be an integer"):
            GatewayConfig.load(settings_file)

    def test_non_object_model_map_fails(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"model_map": ["haiku"]})
        with pytest.raises(ConfigError, match="JSON object mapping model names"):
            GatewayConfig.load(settings_file)

    def test_paths_expand_user(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"codex_home": "~/.codex-alt"})
        config = GatewayConfig.load(settings_file)
        assert config.codex_home == Path.home() / ".codex-alt"


class TestCustomProviders:
    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    @staticmethod
    def _entry(**overrides: object) -> dict[str, object]:
        entry: dict[str, object] = {
            "wire_api": "responses",
            "base_url": "https://model.example/api/v1",
            "api_key": "secret-key",
        }
        entry.update(overrides)
        return entry

    @staticmethod
    def _anthropic_entry(**overrides: object) -> dict[str, object]:
        entry: dict[str, object] = {
            "base_url": "https://messages.example/api",
            "api_key": "anthropic-secret-key",
        }
        entry.update(overrides)
        return entry

    @classmethod
    def _payload(
        cls,
        *,
        name: str = "wrtn",
        entry: object | None = None,
    ) -> dict[str, object]:
        if entry is None:
            entry = cls._entry()
        return {
            "custom_providers": {
                "openai_compatible": {
                    name: entry,
                }
            }
        }

    @classmethod
    def _anthropic_payload(
        cls,
        *,
        name: str = "messages-api",
        entry: object | None = None,
    ) -> dict[str, object]:
        if entry is None:
            entry = cls._anthropic_entry()
        return {
            "custom_providers": {
                "anthropic_compatible": {
                    name: entry,
                }
            }
        }

    def test_file_dict_form_is_parsed(self, tmp_path: Path) -> None:
        config = GatewayConfig.load(self._write(tmp_path, self._payload()))

        assert config.custom_providers == {
            "wrtn": OpenAICompatibleProvider(
                wire_api="responses",
                base_url="https://model.example/api/v1",
                api_key="secret-key",
            )
        }
        assert config.route_providers == (*BUILTIN_ROUTE_PROVIDERS, "wrtn")

    def test_context_window_map_accepts_builtin_and_custom_provider_prefixes(
        self, tmp_path: Path
    ) -> None:
        payload = self._payload()
        payload["context_window_map"] = {
            "grok:shared-model": 500000,
            "wrtn:shared-model": 600000,
        }

        config = GatewayConfig.load(self._write(tmp_path, payload))

        assert config.context_window_map == {
            "grok:shared-model": 500000,
            "wrtn:shared-model": 600000,
        }

    def test_env_json_string_form_is_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = self._payload()["custom_providers"]
        monkeypatch.setenv("CLAUDEX_CUSTOM_PROVIDERS", json.dumps(document))

        assert GatewayConfig.from_env().custom_providers == {
            "wrtn": OpenAICompatibleProvider(
                wire_api="responses",
                base_url="https://model.example/api/v1",
                api_key="secret-key",
            )
        }

    def test_mixed_family_env_json_string_is_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = {
            "anthropic_compatible": {"messages-api": self._anthropic_entry()},
            "openai_compatible": {"responses-api": self._entry()},
        }
        monkeypatch.setenv("CLAUDEX_CUSTOM_PROVIDERS", json.dumps(document))

        config = GatewayConfig.from_env()

        assert isinstance(
            config.custom_providers["responses-api"], OpenAICompatibleProvider
        )
        assert isinstance(
            config.custom_providers["messages-api"], AnthropicCompatibleProvider
        )

    def test_empty_env_means_no_custom_providers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, self._payload())
        monkeypatch.setenv("CLAUDEX_CUSTOM_PROVIDERS", "")

        assert GatewayConfig.load(settings_file).custom_providers == {}

    def test_both_families_are_parsed_into_one_route_namespace(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "custom_providers": {
                "openai_compatible": {"responses-api": self._entry()},
                "anthropic_compatible": {
                    "messages-api": self._anthropic_entry()
                },
            },
            "model_map": {"haiku": "messages-api:claude-haiku"},
            "context_window_map": {
                "responses-api:gpt-model": 128000,
                "messages-api:claude-haiku": 200000,
            },
        }

        config = GatewayConfig.load(self._write(tmp_path, payload))

        assert config.custom_providers == {
            "responses-api": OpenAICompatibleProvider(
                wire_api="responses",
                base_url="https://model.example/api/v1",
                api_key="secret-key",
            ),
            "messages-api": AnthropicCompatibleProvider(
                base_url="https://messages.example/api",
                api_key="anthropic-secret-key",
            ),
        }
        assert config.route_providers == (
            *BUILTIN_ROUTE_PROVIDERS,
            "responses-api",
            "messages-api",
        )
        assert config.mapped_route("claude-haiku-4-5") == RouteTarget(
            "messages-api", "claude-haiku"
        )
        assert config.context_window_map == {
            "responses-api:gpt-model": 128000,
            "messages-api:claude-haiku": 200000,
        }

    def test_cross_family_duplicate_provider_name_fails_at_boot(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "custom_providers": {
                "openai_compatible": {
                    "shared": self._entry(api_key="openai-sensitive-key")
                },
                "anthropic_compatible": {
                    "shared": self._anthropic_entry(
                        api_key="anthropic-sensitive-key"
                    )
                },
            }
        }

        with pytest.raises(ConfigError) as error:
            GatewayConfig.load(self._write(tmp_path, payload))

        message = str(error.value)
        assert "provider name 'shared' is configured in both" in message
        assert "openai_compatible" in message
        assert "anthropic_compatible" in message
        assert "openai-sensitive-key" not in message
        assert "anthropic-sensitive-key" not in message

    def test_family_order_is_deterministic(self, tmp_path: Path) -> None:
        payload = {
            "custom_providers": {
                "anthropic_compatible": {
                    "messages-api": self._anthropic_entry()
                },
                "openai_compatible": {"responses-api": self._entry()},
            }
        }

        config = GatewayConfig.load(self._write(tmp_path, payload))

        assert tuple(config.custom_providers) == ("responses-api", "messages-api")

    def test_unknown_family_fails_at_boot(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {"custom_providers": {"other_compatible": {}}},
        )

        with pytest.raises(
            ConfigError,
            match=(
                "unknown families: other_compatible.*"
                "valid families: openai_compatible, anthropic_compatible"
            ),
        ):
            GatewayConfig.load(settings_file)

    @pytest.mark.parametrize(
        ("name", "overrides", "message"),
        [
            pytest.param(
                "messages-api",
                {"wire_api": "responses"},
                "unknown keys: wire_api.*valid keys: api_key, base_url",
                id="wire-api-is-unknown",
            ),
            pytest.param(
                "messages-api",
                {"base_url": "http://model.example/api"},
                "https is required except for http loopback",
                id="invalid-url",
            ),
            pytest.param(
                "messages-api",
                {"api_key": ""},
                "api_key must be a non-empty string",
                id="empty-api-key",
            ),
            pytest.param(
                "anthropic",
                {},
                "custom provider name 'anthropic' is reserved",
                id="reserved-name",
            ),
        ],
    )
    def test_anthropic_compatible_validation_reuses_existing_rules(
        self,
        tmp_path: Path,
        name: str,
        overrides: dict[str, object],
        message: str,
    ) -> None:
        entry = self._anthropic_entry(api_key="sensitive-api-key")
        entry.update(overrides)
        payload = self._anthropic_payload(name=name, entry=entry)

        with pytest.raises(ConfigError, match=message) as error:
            GatewayConfig.load(self._write(tmp_path, payload))

        assert "sensitive-api-key" not in str(error.value)

    def test_unknown_entry_key_fails_at_boot(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(entry=self._entry(timeout_seconds=30)),
        )

        with pytest.raises(ConfigError, match="unknown keys: timeout_seconds"):
            GatewayConfig.load(settings_file)

    def test_missing_required_field_fails_at_boot(self, tmp_path: Path) -> None:
        entry = self._entry()
        del entry["api_key"]
        settings_file = self._write(tmp_path, self._payload(entry=entry))

        with pytest.raises(ConfigError, match="missing required keys: api_key"):
            GatewayConfig.load(settings_file)

    def test_openai_compatible_still_requires_responses_wire_api(
        self, tmp_path: Path
    ) -> None:
        missing_wire_api = self._entry()
        del missing_wire_api["wire_api"]

        with pytest.raises(ConfigError, match="missing required keys: wire_api"):
            GatewayConfig.load(
                self._write(tmp_path, self._payload(entry=missing_wire_api))
            )

        with pytest.raises(
            ConfigError, match="wire_api must be exactly 'responses'"
        ):
            GatewayConfig.load(
                self._write(
                    tmp_path,
                    self._payload(entry=self._entry(wire_api="messages")),
                )
            )

    def test_chat_wire_api_fails_with_specific_message(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(entry=self._entry(wire_api="chat")),
        )

        with pytest.raises(ConfigError) as error:
            GatewayConfig.load(settings_file)

        message = str(error.value)
        assert "chat completions upstreams are not supported" in message
        assert "only 'responses' is valid" in message

    def test_non_https_non_loopback_base_url_fails_at_boot(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(
                entry=self._entry(base_url="http://model.example/api/v1")
            ),
        )

        with pytest.raises(
            ConfigError, match="https is required except for http loopback"
        ):
            GatewayConfig.load(settings_file)

    def test_http_loopback_base_url_is_allowed(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(
                entry=self._entry(base_url="http://127.0.0.1:8080/api/v1")
            ),
        )

        provider = GatewayConfig.load(settings_file).custom_providers["wrtn"]

        assert provider.base_url == "http://127.0.0.1:8080/api/v1"

    def test_trailing_slashes_are_stripped(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(
                entry=self._entry(base_url="https://model.example/api/v1///")
            ),
        )

        provider = GatewayConfig.load(settings_file).custom_providers["wrtn"]

        assert provider.base_url == "https://model.example/api/v1"

    def test_empty_api_key_fails_at_boot(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            self._payload(entry=self._entry(api_key="")),
        )

        with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
            GatewayConfig.load(settings_file)

    def test_reserved_provider_name_fails_at_boot(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, self._payload(name="codex"))

        with pytest.raises(
            ConfigError, match="custom provider name 'codex' is reserved"
        ):
            GatewayConfig.load(settings_file)

    def test_invalid_provider_name_fails_at_boot(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, self._payload(name="Wrtn"))

        with pytest.raises(
            ConfigError, match="custom provider name 'Wrtn' is invalid"
        ):
            GatewayConfig.load(settings_file)

    def test_model_map_unknown_prefix_fails_at_boot(self, tmp_path: Path) -> None:
        payload = self._payload()
        payload["model_map"] = {"haiku": "missing:gpt-5.5"}
        settings_file = self._write(tmp_path, payload)

        with pytest.raises(ConfigError, match="unknown provider prefix 'missing'"):
            GatewayConfig.load(settings_file)

    def test_model_map_custom_prefix_succeeds(self, tmp_path: Path) -> None:
        payload = self._payload()
        payload["model_map"] = {"haiku": "wrtn:gpt-5.5"}
        config = GatewayConfig.load(self._write(tmp_path, payload))

        assert config.mapped_route("claude-haiku-4-5") == RouteTarget(
            "wrtn", "gpt-5.5"
        )
        assert config.maps_to_provider("wrtn")


class TestMappedRoute:
    def test_bare_value_is_rejected(self) -> None:
        # A programmatically built config bypasses load-time validation, so
        # the bare target surfaces as a parse error at route time instead.
        config = GatewayConfig(model_map={"claude-fable-5": "gpt-5.6-sol"})
        with pytest.raises(ConfigError, match="no provider prefix"):
            config.mapped_route("claude-fable-5")

    def test_codex_prefix_is_accepted(self) -> None:
        config = GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"})
        assert config.mapped_route("claude-haiku-4-5") == RouteTarget("codex", "gpt-5.6-luna")

    def test_kimi_prefix_routes_to_kimi(self) -> None:
        config = GatewayConfig(model_map={"opus": "kimi:k2.5"})
        assert config.mapped_route("claude-opus-5") == RouteTarget("kimi", "k2.5")

    def test_grok_prefix_routes_to_grok(self) -> None:
        config = GatewayConfig(model_map={"opus": "grok:grok-4.5"})
        assert config.mapped_route("claude-opus-5") == RouteTarget("grok", "grok-4.5")
        assert config.maps_to_provider("grok")

    def test_substring_match(self) -> None:
        config = GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"})
        assert config.mapped_route("claude-haiku-4-5-20251001") == RouteTarget(
            "codex", "gpt-5.6-luna"
        )

    def test_longest_substring_key_wins_over_map_order(self) -> None:
        config = GatewayConfig(
            model_map={"claude": "codex:gpt-5.5", "claude-haiku": "kimi:k2.5"}
        )
        assert config.mapped_route("claude-haiku-4-5") == RouteTarget("kimi", "k2.5")
        assert config.mapped_route("claude-fable-5") == RouteTarget("codex", "gpt-5.5")

    def test_exact_match_beats_substring_keys(self) -> None:
        config = GatewayConfig(
            model_map={"fable": "codex:gpt-5.6-sol", "claude-fable-5": "codex:gpt-5.5"}
        )
        assert config.mapped_route("claude-fable-5") == RouteTarget("codex", "gpt-5.5")

    def test_unmapped_model_returns_none(self) -> None:
        config = GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"})
        assert config.mapped_route("claude-sonnet-4-6") is None

    def test_missing_model_returns_none(self) -> None:
        config = GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"})
        assert config.mapped_route(None) is None
        assert config.mapped_route("") is None

    def test_maps_to_provider(self) -> None:
        config = GatewayConfig(model_map={"opus": "kimi:k2.5", "haiku": "codex:gpt-5.6-luna"})
        assert config.maps_to_provider("kimi")
        assert config.maps_to_provider("codex")
        assert not GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"}).maps_to_provider("kimi")


class TestRouteTargetValidation:
    def test_bare_value_fails_at_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"haiku": "gpt-5.6-luna"}')
        with pytest.raises(ConfigError, match="no provider prefix"):
            GatewayConfig.from_env()

    def test_unknown_prefix_fails_at_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"opus": "kim:k2.5"}')
        with pytest.raises(ConfigError, match="unknown provider prefix 'kim'"):
            GatewayConfig.from_env()

    def test_empty_model_after_prefix_fails_at_load(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps({"model_map": {"opus": "kimi:"}}), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="names no model after the provider prefix"):
            GatewayConfig.load(settings_file)

    def test_cli_credential_locations(self) -> None:
        config = GatewayConfig()
        assert config.kimi_code_home == Path.home() / ".kimi-code"
        assert config.grok_home == Path.home() / ".grok"

    def test_kimi_code_home_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIMI_CODE_HOME", "~/kimi-code-alt")
        assert GatewayConfig.from_env().kimi_code_home == Path.home() / "kimi-code-alt"


class TestParseCompactionModel:
    def test_returns_canonical_model_id(self) -> None:
        assert parse_compaction_model("claude:claude-opus-5") == "claude-opus-5"

    @pytest.mark.parametrize(
        "value",
        ["sonnet-5", "codex:x", "claude:", "claude: ", "anthropic:claude-sonnet-5"],
    )
    def test_invalid_values_are_rejected(self, value: str) -> None:
        with pytest.raises(ConfigError):
            parse_compaction_model(value)

    def test_claude_prefix_is_still_rejected_as_a_route_target(self) -> None:
        # Locked contract: "claude" must never become a valid model_map route
        # provider, so parse_route_target keeps rejecting it.
        with pytest.raises(ConfigError, match="unknown provider prefix 'claude'"):
            parse_route_target("claude:claude-opus-5", BUILTIN_ROUTE_PROVIDERS)


class TestCompactionModelSetting:
    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    def test_valid_value_populates_the_raw_field(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        config = GatewayConfig.load(settings_file)
        # The field holds the raw "claude:<id>" value, not the canonical id
        # returned by parse_compaction_model.
        assert config.compaction_model == "claude:claude-opus-5"

    def test_absent_key_disables_the_feature(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {})
        assert GatewayConfig.load(settings_file).compaction_model is None

    def test_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "claude:claude-sonnet-6")
        config = GatewayConfig.load(settings_file)
        assert config.compaction_model == "claude:claude-sonnet-6"

    def test_empty_env_forces_disabled_even_with_file_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", "")
        assert GatewayConfig.load(settings_file).compaction_model is None

    @pytest.mark.parametrize("value", [None, True, 1, [], {}])
    def test_non_string_file_value_fails_to_load(
        self, tmp_path: Path, value: object
    ) -> None:
        settings_file = self._write(tmp_path, {"compaction.model": value})
        with pytest.raises(ConfigError, match='compaction.model" must be a string'):
            GatewayConfig.load(settings_file)

    @pytest.mark.parametrize(
        "value",
        ["sonnet-5", "codex:x", "claude:", "claude: ", "anthropic:claude-sonnet-5"],
    )
    def test_invalid_string_value_fails_to_load(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("CLAUDEX_COMPACTION_MODEL", value)
        with pytest.raises(ConfigError, match="CLAUDEX_COMPACTION_MODEL"):
            GatewayConfig.from_env()


_ACCOUNT_ID = "8f9c2a4e-1234-4a5b-9c6d-0e1f2a3b4c5d"


class TestParseClaudeAccountId:
    def test_canonical_uuid_is_returned_unchanged(self) -> None:
        assert parse_claude_account_id(_ACCOUNT_ID) == _ACCOUNT_ID

    @pytest.mark.parametrize(
        "value",
        [
            "not-a-uuid",
            _ACCOUNT_ID.upper(),  # canonical form only, no case variants
            "{" + _ACCOUNT_ID + "}",
            _ACCOUNT_ID.replace("-", ""),
            "user@example.com",  # email resolution belongs to the CLI
        ],
    )
    def test_non_canonical_values_are_rejected(self, value: str) -> None:
        with pytest.raises(ConfigError, match="canonical account UUID"):
            parse_claude_account_id(value)


class TestClaudeAccountIdSetting:
    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    def test_valid_value_populates_the_field(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"claude_account.id": _ACCOUNT_ID})
        assert GatewayConfig.load(settings_file).claude_account_id == _ACCOUNT_ID

    def test_absent_key_disables_the_feature(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {})
        assert GatewayConfig.load(settings_file).claude_account_id is None

    def test_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        other_id = "0a1b2c3d-4e5f-4678-9abc-def012345678"
        settings_file = self._write(tmp_path, {"claude_account.id": _ACCOUNT_ID})
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", other_id)
        assert GatewayConfig.load(settings_file).claude_account_id == other_id

    def test_empty_env_forces_disabled_even_with_file_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"claude_account.id": _ACCOUNT_ID})
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", "")
        assert GatewayConfig.load(settings_file).claude_account_id is None

    @pytest.mark.parametrize("value", [None, True, 1, [], {}])
    def test_non_string_file_value_fails_to_load(
        self, tmp_path: Path, value: object
    ) -> None:
        settings_file = self._write(tmp_path, {"claude_account.id": value})
        with pytest.raises(ConfigError, match='claude_account.id" must be a string'):
            GatewayConfig.load(settings_file)

    def test_invalid_string_value_fails_to_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ID", "not-a-uuid")
        with pytest.raises(ConfigError, match="CLAUDEX_CLAUDE_ACCOUNT_ID"):
            GatewayConfig.from_env()


class TestClaudeAccountRoutingSetting:
    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    def test_absent_key_defaults_to_disabled(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {})
        config = GatewayConfig.load(settings_file)
        assert config.claude_account_routing_mode == "disabled"
        assert config.claude_account_include_local_login is True

    def test_file_document_sets_the_mode(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path, {"claude_account.routing": {"mode": "fallback"}}
        )
        assert (
            GatewayConfig.load(settings_file).claude_account_routing_mode == "fallback"
        )

    def test_env_json_string_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"claude_account.routing": {"mode": "disabled"}}
        )
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", '{"mode": "fallback"}')
        assert (
            GatewayConfig.load(settings_file).claude_account_routing_mode == "fallback"
        )

    def test_empty_env_forces_disabled_even_with_file_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"claude_account.routing": {"mode": "fallback"}}
        )
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", "")
        assert (
            GatewayConfig.load(settings_file).claude_account_routing_mode == "disabled"
        )

    def test_balanced_mode_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", '{"mode": "balanced"}')
        assert GatewayConfig.from_env().claude_account_routing_mode == "balanced"

    def test_explicit_false_excludes_local_login(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "claude_account.routing": {
                    "mode": "balanced",
                    "include_local_login": False,
                }
            },
        )
        config = GatewayConfig.load(settings_file)
        assert config.claude_account_routing_mode == "balanced"
        assert config.claude_account_include_local_login is False

    def test_balanced_mode_still_rejects_unknown_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A future-shaped document must name the real (unknown-key) reason,
        # not a stale "not implemented" one.
        monkeypatch.setenv(
            "CLAUDEX_CLAUDE_ACCOUNT_ROUTING",
            '{"mode": "balanced", "balanced": {"window": "session"}}',
        )
        with pytest.raises(ConfigError, match="unknown keys: balanced"):
            GatewayConfig.from_env()

    def test_unknown_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", '{"mode": "round-robin"}')
        with pytest.raises(
            ConfigError, match="must be one of disabled, fallback, balanced"
        ):
            GatewayConfig.from_env()

    def test_unknown_document_keys_are_rejected(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {"claude_account.routing": {"mode": "fallback", "weights": [1, 2]}},
        )
        with pytest.raises(ConfigError, match="unknown keys: weights"):
            GatewayConfig.load(settings_file)

    def test_non_boolean_include_local_login_is_rejected(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {
                "claude_account.routing": {
                    "mode": "balanced",
                    "include_local_login": "yes",
                }
            },
        )
        with pytest.raises(ConfigError, match="must be a JSON boolean"):
            GatewayConfig.load(settings_file)

    def test_missing_mode_key_is_rejected(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"claude_account.routing": {}})
        with pytest.raises(
            ConfigError, match="must be one of disabled, fallback, balanced"
        ):
            GatewayConfig.load(settings_file)

    @pytest.mark.parametrize("value", [None, True, 1, [], "fallback"])
    def test_non_object_file_value_fails_to_load(
        self, tmp_path: Path, value: object
    ) -> None:
        settings_file = self._write(tmp_path, {"claude_account.routing": value})
        with pytest.raises(ConfigError, match="claude_account.routing"):
            GatewayConfig.load(settings_file)

    def test_invalid_env_json_fails_to_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDEX_CLAUDE_ACCOUNT_ROUTING", "fallback")
        with pytest.raises(ConfigError, match="CLAUDEX_CLAUDE_ACCOUNT_ROUTING"):
            GatewayConfig.from_env()

    def test_parse_returns_disabled_for_explicit_disabled_document(self) -> None:
        policy = parse_claude_account_routing({"mode": "disabled"})
        assert policy.mode == "disabled"

    def test_parse_returns_balanced_policy_with_local_login_default(self) -> None:
        policy = parse_claude_account_routing({"mode": "balanced"})
        assert policy.mode == "balanced"
        assert policy.include_local_login is True

    def test_parse_returns_balanced_policy_without_local_login(self) -> None:
        policy = parse_claude_account_routing(
            {"mode": "balanced", "include_local_login": False}
        )
        assert policy.mode == "balanced"
        assert policy.include_local_login is False

    @staticmethod
    def _write(tmp_path: Path, payload: object) -> Path:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps(payload), encoding="utf-8")
        return settings_file

    def test_deleting_an_existing_key_removes_it(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        update_settings_file(settings_file, {}, deletions=("compaction.model",))
        written = json.loads(settings_file.read_text(encoding="utf-8"))
        assert "compaction.model" not in written

    def test_deleting_the_last_key_leaves_a_valid_empty_object(
        self, tmp_path: Path
    ) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        update_settings_file(settings_file, {}, deletions=("compaction.model",))
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {}

    def test_deleting_an_absent_key_is_a_no_op(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"port": 9090})
        update_settings_file(settings_file, {}, deletions=("compaction.model",))
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {"port": 9090}

    def test_unknown_deletion_key_raises(self, tmp_path: Path) -> None:
        settings_file = self._write(tmp_path, {"port": 9090})
        with pytest.raises(ConfigError, match="unknown settings keys"):
            update_settings_file(settings_file, {}, deletions=("not_a_key",))

    def test_combined_updates_and_deletions_apply_both(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path, {"compaction.model": "claude:claude-opus-5"}
        )
        update_settings_file(
            settings_file,
            {"port": 9317},
            deletions=("compaction.model",),
        )
        written = json.loads(settings_file.read_text(encoding="utf-8"))
        assert written == {"port": 9317}

    def test_unrelated_keys_survive_a_deletion(self, tmp_path: Path) -> None:
        settings_file = self._write(
            tmp_path,
            {"compaction.model": "claude:claude-opus-5", "port": 9090},
        )
        update_settings_file(settings_file, {}, deletions=("compaction.model",))
        written = json.loads(settings_file.read_text(encoding="utf-8"))
        assert written == {"port": 9090}
