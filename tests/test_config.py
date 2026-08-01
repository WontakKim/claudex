"""Tests for gateway configuration sources and model routing."""

import json
from pathlib import Path

import pytest

from claudex_gateway.config import ConfigError, GatewayConfig, RouteTarget


def test_max_reasoning_effort_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDEX_REASONING_EFFORT", "max")
    assert GatewayConfig.from_env().reasoning_effort_override == "max"


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
    monkeypatch.setenv("CLAUDEX_MODEL_MAP", '{"":"gpt-5.6-sol"}')
    with pytest.raises(ConfigError, match="CLAUDEX_MODEL_MAP"):
        GatewayConfig.from_env()


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
                "model_map": {"haiku": "gpt-5.6-luna"},
            },
        )

        config = GatewayConfig.load(settings_file)

        assert config.port == 9090
        assert config.model_map == {"haiku": "gpt-5.6-luna"}

    def test_env_overrides_settings_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"reasoning_effort": "low"})
        monkeypatch.setenv("CLAUDEX_REASONING_EFFORT", "high")
        assert GatewayConfig.load(settings_file).reasoning_effort_override == "high"

    def test_empty_env_still_overrides(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        settings_file = self._write(tmp_path, {"model_map": {"haiku": "gpt-5.6-luna"}})
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


class TestMappedRoute:
    def test_bare_value_routes_to_codex(self) -> None:
        config = GatewayConfig(model_map={"claude-fable-5": "gpt-5.6-sol"})
        assert config.mapped_route("claude-fable-5") == RouteTarget("codex", "gpt-5.6-sol")

    def test_codex_prefix_is_accepted(self) -> None:
        config = GatewayConfig(model_map={"haiku": "codex:gpt-5.6-luna"})
        assert config.mapped_route("claude-haiku-4-5") == RouteTarget("codex", "gpt-5.6-luna")

    def test_kimi_prefix_routes_to_kimi(self) -> None:
        config = GatewayConfig(model_map={"opus": "kimi:k2.5"})
        assert config.mapped_route("claude-opus-5") == RouteTarget("kimi", "k2.5")

    def test_substring_match(self) -> None:
        config = GatewayConfig(model_map={"haiku": "gpt-5.6-luna"})
        assert config.mapped_route("claude-haiku-4-5-20251001") == RouteTarget(
            "codex", "gpt-5.6-luna"
        )

    def test_longest_substring_key_wins_over_map_order(self) -> None:
        config = GatewayConfig(
            model_map={"claude": "gpt-5.5", "claude-haiku": "kimi:k2.5"}
        )
        assert config.mapped_route("claude-haiku-4-5") == RouteTarget("kimi", "k2.5")
        assert config.mapped_route("claude-fable-5") == RouteTarget("codex", "gpt-5.5")

    def test_exact_match_beats_substring_keys(self) -> None:
        config = GatewayConfig(
            model_map={"fable": "gpt-5.6-sol", "claude-fable-5": "gpt-5.5"}
        )
        assert config.mapped_route("claude-fable-5") == RouteTarget("codex", "gpt-5.5")

    def test_unmapped_model_returns_none(self) -> None:
        config = GatewayConfig(model_map={"haiku": "gpt-5.6-luna"})
        assert config.mapped_route("claude-sonnet-4-6") is None

    def test_missing_model_returns_none(self) -> None:
        config = GatewayConfig(model_map={"haiku": "gpt-5.6-luna"})
        assert config.mapped_route(None) is None
        assert config.mapped_route("") is None

    def test_maps_to_provider(self) -> None:
        config = GatewayConfig(model_map={"opus": "kimi:k2.5", "haiku": "gpt-5.6-luna"})
        assert config.maps_to_provider("kimi")
        assert config.maps_to_provider("codex")
        assert not GatewayConfig(model_map={"haiku": "gpt-5.6-luna"}).maps_to_provider("kimi")


class TestRouteTargetValidation:
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

    def test_default_kimi_auth_file_location(self) -> None:
        assert GatewayConfig().kimi_auth_file == Path.home() / ".claudex" / "kimi-auth.json"
