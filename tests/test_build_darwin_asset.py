"""Tests for the Darwin release asset dependency boundary."""

from __future__ import annotations

import re
import shlex
import subprocess
import tomllib
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_gptpro_dependencies_are_only_in_the_optional_extra() -> None:
    configuration = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = configuration["project"]["dependencies"]
    gptpro_dependencies = configuration["project"]["optional-dependencies"][
        "gptpro"
    ]
    assert not any(
        dependency.startswith(("mcp", "playwright")) for dependency in dependencies
    )
    assert gptpro_dependencies == ["mcp>=2.1.1", "playwright>=1.62.0"]


def test_darwin_asset_export_includes_gptpro_dependencies() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )
    export_command = re.search(
        r"^uv export (.+) -o build/requirements.txt$", script, re.MULTILINE
    )
    assert export_command is not None
    result = subprocess.run(
        ["uv", "export", *shlex.split(export_command.group(1))],
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


def test_darwin_asset_script_installs_gptpro_and_checks_native_architecture() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )

    assert (
        "uv export --frozen --extra gptpro --no-dev --no-emit-project --no-hashes"
        in script
    )
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


def test_gptpro_setup_documents_both_installations() -> None:
    documentation = (_REPOSITORY_ROOT / "docs" / "gptpro.md").read_text(
        encoding="utf-8"
    )

    assert "./bin/claudex-gateway gptpro login" in documentation
    assert "./bin/claudex-gateway gptpro doctor" in documentation
    assert "uv sync --extra gptpro" in documentation
    assert "latest release tarball" in documentation
