"""Tests for the Darwin release asset dependency boundary."""

from __future__ import annotations

import importlib.metadata
import subprocess
import tomllib
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_gptpro_dependencies_are_required_by_default() -> None:
    configuration = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = configuration["project"]["dependencies"]
    optional_dependencies = configuration["project"].get(
        "optional-dependencies", {}
    )
    assert {"mcp>=2.1.1", "playwright>=1.62.0"} <= set(dependencies)
    assert "gptpro" not in optional_dependencies


def test_default_export_includes_gptpro_dependencies() -> None:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    exported_names = {
        line.partition("==")[0]
        for line in result.stdout.splitlines()
        if "==" in line
    }

    assert {"mcp", "playwright", "greenlet", "cryptography"} <= exported_names


def test_darwin_asset_script_installs_default_dependencies_and_checks_architecture() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )

    assert "uv export --frozen --no-dev --no-emit-project --no-hashes" in script
    assert "--extra" not in script
    assert "--all-extras" not in script
    assert '"dist/claudex_gateway-${VERSION}-py3-none-any.whl"' in script
    assert "-name '*.so' -o -name '*.dylib'" in script
    assert 'lipo "${native_file}" -verify_arch arm64' in script
    assert '"$(uname -s)" = "Darwin"' in script
    assert '"$(uname -m)" = "arm64"' in script


def test_darwin_asset_smoke_starts_bundled_mcp_and_playwright_driver() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )

    assert "from claudex.mcp_tools import build_gptpro_server" in script
    assert "build_gptpro_server(None)" in script
    assert "async_playwright().start()" in script
    assert '"${STAGE}/python/bin/python3"' in script
    assert "PYTHONNOUSERSITE=1" in script


def test_release_workflow_builds_on_arm64_macos() -> None:
    workflow = (_REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "runs-on: macos-15" in workflow
    assert "      - 'pyproject.toml'" in workflow
    assert "      - 'uv.lock'" in workflow
    assert "uv sync --frozen\n" in workflow
    assert "uv run --frozen pytest" in workflow
    assert "--extra" not in workflow


def test_release_version_matches_lock_and_installed_metadata() -> None:
    configuration = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lockfile = tomllib.loads(
        (_REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    )
    project_version = configuration["project"]["version"]
    project_package = next(
        package
        for package in lockfile["package"]
        if package.get("source") == {"editable": "."}
    )

    assert project_package["version"] == project_version
    assert importlib.metadata.version("claudex-gateway") == project_version


def test_gptpro_setup_documents_both_installations() -> None:
    documentation = (_REPOSITORY_ROOT / "docs" / "gptpro.md").read_text(
        encoding="utf-8"
    )

    assert "./bin/claudex-gateway gptpro login" in documentation
    assert "./bin/claudex-gateway gptpro doctor" in documentation
    assert "uv sync" in documentation
    assert "uv run claudex-gateway gptpro login" in documentation
    assert "latest release tarball" in documentation
    assert "--extra" not in documentation
    assert "sync the project dependencies first" in documentation
    normalized_documentation = " ".join(documentation.split())
    assert (
        "downloads the matching Playwright Chromium automatically"
        in normalized_documentation
    )
    assert "normal per-user cache" in normalized_documentation
    assert (
        "does not write a browser into the extracted release directory"
        in normalized_documentation
    )


def test_gptpro_copy_describes_default_dependencies() -> None:
    browser_source = (
        _REPOSITORY_ROOT / "src" / "claudex" / "gptpro" / "browser.py"
    ).read_text(encoding="utf-8")
    login_source = (
        _REPOSITORY_ROOT / "src" / "claudex" / "gptpro" / "login_session.py"
    ).read_text(encoding="utf-8")
    dashboard = (
        _REPOSITORY_ROOT / "src" / "claudex" / "dashboard" / "dashboard.html"
    ).read_text(encoding="utf-8")

    assert "`uv sync` in a source checkout" in browser_source
    assert "Raised when the Playwright dependency is unavailable." in browser_source
    assert "without importing Playwright into the daemon" in login_source
    assert "browser profile, and browser dependencies." in dashboard
