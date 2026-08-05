"""Tests for the shared ~/.claudex path resolver module."""

import ast
from pathlib import Path

import pytest

from claudex_gateway import paths

_PACKAGE_DIR = Path(paths.__file__).parent


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def test_runtime_dir_is_dot_claudex_under_home(tmp_path: Path) -> None:
    assert paths.runtime_dir() == tmp_path / ".claudex"


def test_settings_file_is_under_runtime_dir(tmp_path: Path) -> None:
    assert paths.settings_file() == tmp_path / ".claudex" / "settings.json"


def test_daemon_record_file_keeps_the_historical_pid_filename(tmp_path: Path) -> None:
    assert paths.daemon_record_file() == tmp_path / ".claudex" / "gateway.pid"


def test_log_file_is_under_runtime_dir(tmp_path: Path) -> None:
    assert paths.log_file() == tmp_path / ".claudex" / "gateway.log"


def test_accounts_dir_is_under_the_accounts_root(tmp_path: Path) -> None:
    assert paths.accounts_dir("claude") == tmp_path / ".claudex" / "accounts" / "claude"


@pytest.mark.parametrize(
    "provider", ["", ".", "..", "a/b", "a\\b", "C:", "C:claude", "D:claude"]
)
def test_provider_rejects_unsafe_names(provider: str) -> None:
    with pytest.raises(ValueError):
        paths.accounts_dir(provider)


def test_do_not_create_the_runtime_directory(tmp_path: Path) -> None:
    paths.runtime_dir()
    paths.settings_file()
    paths.daemon_record_file()
    paths.log_file()
    paths.accounts_dir("claude")
    assert not (tmp_path / ".claudex").exists()


def test_imports_only_stdlib() -> None:
    allowed_modules = {"__future__", "pathlib"}
    source = (_PACKAGE_DIR / "paths.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in allowed_modules
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed_modules


@pytest.mark.parametrize("module_name", ["__main__.py", "config.py"])
def test_consumers_use_shared_paths_module(module_name: str) -> None:
    source = (_PACKAGE_DIR / module_name).read_text(encoding="utf-8")
    assert 'Path.home() / ".claudex"' not in source
