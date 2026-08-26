"""Tests for the Darwin release asset dependency boundary."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_playwright_is_only_in_the_gptpro_optional_extra() -> None:
    configuration = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependencies = configuration["project"]["dependencies"]
    gptpro_dependencies = configuration["project"]["optional-dependencies"][
        "gptpro"
    ]
    assert not any(dependency.startswith("playwright") for dependency in dependencies)
    assert gptpro_dependencies == ["playwright>=1.62.0"]


def test_darwin_asset_export_excludes_gptpro_native_dependencies() -> None:
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

    assert exported_names.isdisjoint({"playwright", "greenlet", "pyee"})


def test_darwin_asset_script_installs_without_optional_extras() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )

    assert (
        "uv export --frozen --no-dev --no-emit-project --no-hashes" in script
    )
    assert "--all-extras" not in script
    assert "--extra gptpro" not in script
    assert '"dist/claudex_gateway-${VERSION}-py3-none-any.whl"' in script
    assert "-name '*.so' -o -name '*.dylib'" in script
