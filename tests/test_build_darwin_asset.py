"""Tests for the Darwin release asset dependency boundary."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

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


def test_darwin_asset_smoke_starts_bundled_mcp_and_playwright_driver() -> None:
    script = (_REPOSITORY_ROOT / "scripts" / "build-darwin-asset.sh").read_text(
        encoding="utf-8"
    )

    assert "from claudex.mcp_tools import build_gptpro_server" in script
    assert "build_gptpro_server(None)" in script
    assert "async_playwright().start()" in script
    assert '"${STAGE}/python/bin/python3"' in script
    assert "PYTHONNOUSERSITE=1" in script


@pytest.mark.parametrize(
    "runner", [None, "ubuntu-24.04-arm", "common-platform-arm", "[self-hosted, Linux, ARM64]"]
)
def test_release_workflow_uses_frozen_dependencies_and_asset_builder(runner: str | None) -> None:
    workflow = (_REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    if runner is not None:
        workflow = re.sub(r"runs-on:.*", f"runs-on: {runner}", workflow)

    assert "uv sync --frozen" in workflow
    assert "uv run --frozen pytest" in workflow
    assert "./scripts/build-darwin-asset.sh" in workflow
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


_ARM64 = 0x0100000C
_X86_64 = 0x01000007


def _thin_macho(cpu_type: int = _ARM64, byte_order: str = "<") -> bytes:
    return struct.pack(f"{byte_order}8I", 0xFEEDFACF, cpu_type, 0, 6, 0, 0, 0, 0)


def _fat_macho(
    architectures: tuple[int, ...], byte_order: str = ">", is_fat64: bool = False
) -> bytes:
    record_format = f"{byte_order}IIQQII" if is_fat64 else f"{byte_order}5I"
    offset = 8 + len(architectures) * struct.calcsize(record_format)
    records = []
    slices = []
    for cpu_type in architectures:
        payload = _thin_macho(cpu_type)
        record = (cpu_type, 0, offset, len(payload), 0)
        if is_fat64:
            record += (0,)
        records.append(struct.pack(record_format, *record))
        slices.append(payload)
        offset += len(payload)
    magic = 0xCAFEBABF if is_fat64 else 0xCAFEBABE
    return (
        struct.pack(f"{byte_order}II", magic, len(architectures))
        + b"".join(records)
        + b"".join(slices)
    )


def _verify_native_files(*paths: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(_REPOSITORY_ROOT / "scripts/verify_darwin_architecture.py")],
        input=b"".join(os.fsencode(path) + b"\0" for path in paths),
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_portable_architecture_accepts_thin_arm64(tmp_path: Path, byte_order: str) -> None:
    native_file = tmp_path / "native library\narm64.so"
    native_file.write_bytes(_thin_macho(byte_order=byte_order))
    assert _verify_native_files(native_file).returncode == 0


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("is_fat64", [False, True])
@pytest.mark.parametrize("architectures", [(_ARM64, _X86_64), (_X86_64, _ARM64)])
def test_portable_architecture_accepts_fat_arm64(
    tmp_path: Path, byte_order: str, is_fat64: bool, architectures: tuple[int, ...]
) -> None:
    native_file = tmp_path / "universal.dylib"
    native_file.write_bytes(_fat_macho(architectures, byte_order, is_fat64))
    assert _verify_native_files(native_file).returncode == 0


def _invalid_macho_files() -> list[bytes]:
    fat = _fat_macho((_ARM64,))
    return [
        b"",
        b"\x7fELF" + b"\0" * 60,
        b"not a native executable",
        _thin_macho(_X86_64),
        _thin_macho()[:8],
        _thin_macho()[:20] + struct.pack("<I", 1) + _thin_macho()[24:],
        _fat_macho((_ARM64,), is_fat64=True)[:20],
        _fat_macho((_X86_64,)),
        struct.pack(">II", 0xCAFEBABE, 0),
        struct.pack(">II", 0xCAFEBABE, 0xFFFFFFFF),
        fat[:20],
        fat[:-1],
        fat[:16] + struct.pack(">I", 4) + fat[20:],
        fat[:20] + struct.pack(">I", 0xFFFFFFFF) + fat[24:],
        fat[:28] + _thin_macho(_X86_64),
        fat[:28] + b"\x7fELF" + fat[32:],
    ]


@pytest.mark.parametrize("payload", _invalid_macho_files())
def test_portable_architecture_rejects_invalid_files(tmp_path: Path, payload: bytes) -> None:
    native_file = tmp_path / "invalid.so"
    native_file.write_bytes(payload)
    result = _verify_native_files(native_file)
    assert result.returncode != 0
    assert os.fsencode(native_file) in result.stderr


def test_portable_architecture_checks_every_file_and_follows_links(tmp_path: Path) -> None:
    valid = tmp_path / "valid.so"
    valid.write_bytes(_thin_macho())
    link = tmp_path / "link.dylib"
    link.symlink_to(valid)
    assert _verify_native_files(valid, link).returncode == 0
    invalid = tmp_path / "invalid.so"
    invalid.write_bytes(_thin_macho(_X86_64))
    result = _verify_native_files(valid, link, invalid)
    assert result.returncode != 0
    assert os.fsencode(invalid) in result.stderr
    valid.unlink()
    assert _verify_native_files(link).returncode != 0


def test_portable_architecture_requires_existing_files_and_nonempty_input(tmp_path: Path) -> None:
    assert _verify_native_files().returncode != 0
    missing = tmp_path / "missing-node"
    result = _verify_native_files(missing)
    assert result.returncode != 0
    assert os.fsencode(missing) in result.stderr


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _asset_fixture(
    tmp_path: Path, host: str = "Linux", invalid_runtime: str | None = None
) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "build checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    tools = tmp_path / "tools"
    tools.mkdir()
    for command in (
        "bash", "cat", "dirname", "sed", "head", "rm", "mkdir", "awk", "tar", "find",
        "chmod", "grep", "ls", "shasum", "sha256sum", "gzip", "od", "tr", "env",
    ):
        executable = shutil.which(command)
        if executable:
            (tools / command).symlink_to(executable)

    runtime = tmp_path / "runtime" / "python"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib").mkdir()
    (runtime / "lib" / "runtime.dylib").write_bytes(_thin_macho())
    if host == "Darwin":
        _write_executable(
            runtime / "bin" / "python3",
            f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps({"runtime": sys.argv[1:], "environment": {
        key: os.environ[key] for key in (
            "HOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
            "CLAUDEX_PORT", "PLAYWRIGHT_NODEJS_PATH",
        ) if key in os.environ
    }}) + "\\n")
if sys.argv[1:] == ["-"]:
    source = sys.stdin.read()
    assert "build_gptpro_server(None)" in source
    assert "async_playwright().start()" in source
    raise SystemExit(int(os.environ.get("RUNTIME_EXIT", "0")))
print(os.environ.get("USAGE_OUTPUT", "usage: claudex-gateway"), file=sys.stderr)
raise SystemExit(int(os.environ.get("USAGE_EXIT", "2")))
''',
        )
    else:
        (runtime / "bin" / "python3").write_bytes(_thin_macho())
        (runtime / "bin" / "python3").chmod(0o755)
    if invalid_runtime is not None:
        (runtime / invalid_runtime).write_bytes(_thin_macho(_X86_64))
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(runtime, arcname="python")
    script = (_REPOSITORY_ROOT / "scripts/build-darwin-asset.sh").read_text()
    script = re.sub(
        r'PBS_SHA256="[0-9a-f]+"',
        f'PBS_SHA256="{hashlib.sha256(archive.read_bytes()).hexdigest()}"',
        script,
    )
    (scripts / "build-darwin-asset.sh").write_text(script)
    shutil.copy(_REPOSITORY_ROOT / "scripts/verify_darwin_architecture.py", scripts)
    (root / "pyproject.toml").write_text('version = "0.17.0"\n')

    packages = tmp_path / "packages"
    (packages / "claudex").mkdir(parents=True)
    (packages / "claudex" / "__init__.py").write_text("")
    (packages / "claudex" / "__main__.py").write_text('''import json, os, sys
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps({"host_smoke": sys.argv, "environment": {
        key: os.environ[key] for key in (
            "HOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
            "CLAUDEX_PORT", "PLAYWRIGHT_NODEJS_PATH",
        ) if key in os.environ
    },
                         "no_site": sys.flags.no_site, "module": __file__}) + "\\n")
print(os.environ.get("USAGE_OUTPUT", "usage: claudex-gateway"), file=sys.stderr)
raise SystemExit(int(os.environ.get("USAGE_EXIT", "2")))
''')
    (packages / "native.so").write_bytes(_fat_macho((_X86_64, _ARM64)))
    (packages / "native.dylib").write_bytes(_thin_macho())
    (packages / "linked.so").symlink_to("native.so")
    (packages / "playwright" / "driver").mkdir(parents=True)
    node = packages / "playwright" / "driver" / "node"
    node.write_bytes(_thin_macho())
    node.chmod(0o755)

    architecture = "arm64" if host == "Darwin" else "aarch64"
    _write_executable(
        tools / "uname",
        f'#!/bin/sh\ncase "$1" in -s) echo {host};; -m) echo {architecture};; esac\n',
    )
    _write_executable(tools / "curl", f"#!{sys.executable}\n" + '''import os, shutil, sys
shutil.copyfile(os.environ["RUNTIME_ARCHIVE"], sys.argv[sys.argv.index("-o") + 1])
''')
    _write_executable(tools / "uv", f"#!{sys.executable}\n" + '''import json
import os
import pathlib
import shutil
import sys
arguments = sys.argv[1:]
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps({"uv": arguments}) + "\\n")
if arguments[0] == "run":
    if os.environ.get("HOST_PYTHON_FAILURE"):
        raise SystemExit("host Python unavailable")
    print(sys.executable)
elif arguments[0] == "build":
    pathlib.Path("dist").mkdir()
    pathlib.Path("dist/claudex_gateway-0.17.0-py3-none-any.whl").touch()
elif arguments[0] == "export":
    requirements = pathlib.Path(arguments[arguments.index("-o") + 1])
    requirements.write_text("mcp==2.1.1\\nplaywright==1.62.0\\n")
elif arguments[:2] == ["pip", "install"]:
    target = arguments[arguments.index("--target") + 1]
    shutil.copytree(os.environ["PACKAGE_FIXTURE"], target, symlinks=True)
else:
    raise SystemExit("unexpected uv invocation")
''')
    if host == "Darwin":
        _write_executable(tools / "lipo", f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps({"lipo": sys.argv[1:]}) + "\\n")
if os.environ.get("LIPO_REJECT") and os.environ["LIPO_REJECT"] in sys.argv[1]:
    raise SystemExit(1)
''')
    environment = dict(os.environ)
    environment.update(
        PATH=str(tools),
        BUILD_LOG=str(tmp_path / "build-log.jsonl"),
        RUNTIME_ARCHIVE=str(archive),
        PACKAGE_FIXTURE=str(packages),
        CLAUDEX_PORT="invalid-inherited-port",
        PLAYWRIGHT_NODEJS_PATH="/not/the/bundled/node",
    )
    return root, environment


def _run_asset_fixture(root: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(Path(environment["PATH"]) / "bash"), str(root / "scripts/build-darwin-asset.sh")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _build_events(environment: dict[str, str]) -> list[dict]:
    return [json.loads(line) for line in Path(environment["BUILD_LOG"]).read_text().splitlines()]


def test_linux_build_uses_portable_verification_and_host_smoke(tmp_path: Path) -> None:
    root, environment = _asset_fixture(tmp_path)
    result = _run_asset_fixture(root, environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Skipping bundled Darwin runtime verification" in result.stdout
    events = _build_events(environment)
    assert not any("lipo" in event or "runtime" in event for event in events)
    selection = next(event["uv"] for event in events if event.get("uv", [None])[0] == "run")
    assert "--no-project" in selection
    assert selection[selection.index("--python") + 1] == "cpython@3.12"
    smoke = next(event for event in events if "host_smoke" in event)
    assert smoke["no_site"] == 1
    assert smoke["module"].startswith(str(root / "build/stage/python"))
    assert smoke["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert smoke["environment"]["PYTHONNOUSERSITE"] == "1"
    assert smoke["environment"]["HOME"] == str(root / "build/smoke-home")
    assert "CLAUDEX_PORT" not in smoke["environment"]
    assert "PLAYWRIGHT_NODEJS_PATH" not in smoke["environment"]
    with tarfile.open(root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz") as archive:
        members = {entry.name.removeprefix("./"): entry for entry in archive.getmembers()}
        assert {"bin/claudex-gateway", "bin/claudex", "python/bin/python3"} <= members.keys()
        assert members["bin/claudex-gateway"].mode & 0o111
        assert not any(name.endswith(".pyc") or "verify_darwin" in name for name in members)
        launcher = archive.extractfile(members["bin/claudex-gateway"]).read().decode()
        assert 'exec "$DIR/python/bin/python3" -m claudex "$@"' in launcher


@pytest.mark.parametrize("candidate", ["native.so", "native.dylib", "playwright/driver/node"])
def test_linux_build_rejects_wrong_architecture(tmp_path: Path, candidate: str) -> None:
    root, environment = _asset_fixture(tmp_path)
    (Path(environment["PACKAGE_FIXTURE"]) / "linked.so").unlink()
    (Path(environment["PACKAGE_FIXTURE"]) / candidate).write_bytes(_thin_macho(_X86_64))
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert candidate in result.stderr
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


@pytest.mark.parametrize("failure", ["missing", "not-executable"])
def test_linux_build_requires_playwright_node(tmp_path: Path, failure: str) -> None:
    root, environment = _asset_fixture(tmp_path)
    node = Path(environment["PACKAGE_FIXTURE"]) / "playwright/driver/node"
    if failure == "missing":
        node.unlink()
    else:
        node.chmod(0o644)
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert "playwright/driver/node" in result.stderr


def test_darwin_build_keeps_lipo_and_both_bundled_smokes(tmp_path: Path) -> None:
    root, environment = _asset_fixture(tmp_path, "Darwin")
    result = _run_asset_fixture(root, environment)
    assert result.returncode == 0, result.stdout + result.stderr
    events = _build_events(environment)
    checked = [event["lipo"][0] for event in events if "lipo" in event]
    for suffix in (
        "python/bin/python3", "runtime.dylib", "native.so", "native.dylib",
        "linked.so", "playwright/driver/node",
    ):
        assert any(path.endswith(suffix) for path in checked)
    assert all(
        event["lipo"][1:] == ["-verify_arch", "arm64"]
        for event in events if "lipo" in event
    )
    smokes = [event for event in events if "runtime" in event]
    assert [event["runtime"] for event in smokes] == [
        ["-m", "claudex", "definitely-not-a-subcommand"], ["-"]
    ]
    assert not any("host_smoke" in event or event.get("uv", [None])[0] == "run" for event in events)
    for event in smokes:
        assert event["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert event["environment"]["PYTHONPATH"] == ""
        assert "CLAUDEX_PORT" not in event["environment"]
        assert "PLAYWRIGHT_NODEJS_PATH" not in event["environment"]


@pytest.mark.parametrize("failure", ["missing-lipo", "wrong-node", "runtime-smoke"])
def test_darwin_build_propagates_verification_failures(tmp_path: Path, failure: str) -> None:
    root, environment = _asset_fixture(tmp_path, "Darwin")
    if failure == "missing-lipo":
        (Path(environment["PATH"]) / "lipo").unlink()
    elif failure == "wrong-node":
        environment["LIPO_REJECT"] = "playwright/driver/node"
    else:
        environment["RUNTIME_EXIT"] = "1"
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


@pytest.mark.parametrize("host", ["Linux", "Darwin"])
@pytest.mark.parametrize("failure", [{"USAGE_EXIT": "0"}, {"USAGE_OUTPUT": "not usage"}])
def test_build_requires_usage_exit_code_and_output(
    tmp_path: Path, host: str, failure: dict[str, str]
) -> None:
    root, environment = _asset_fixture(tmp_path, host)
    environment.update(failure)
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert "smoke test" in result.stderr
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


def test_usage_smoke_does_not_import_native_browser_dependencies(tmp_path: Path) -> None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("CLAUDEX_")
    }
    environment["HOME"] = str(tmp_path)
    script = """
import builtins
import runpy
import sys
real_import = builtins.__import__
blocked = {"mcp", "mcp_types", "playwright", "greenlet", "cryptography", "pydantic",
           "pydantic_core", "_cffi_backend", "rpds"}
def import_without_native_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition(".")[0] in blocked:
        raise AssertionError(f"usage smoke imported {name}")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = import_without_native_dependencies
sys.argv = ["claudex-gateway", "definitely-not-a-subcommand"]
runpy.run_module("claudex", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "usage: claudex-gateway" in result.stderr



@pytest.mark.parametrize("candidate", ["bin/python3", "lib/runtime.dylib"])
def test_linux_build_checks_native_files_outside_site_packages(
    tmp_path: Path, candidate: str
) -> None:
    root, environment = _asset_fixture(tmp_path, invalid_runtime=candidate)
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert candidate in result.stderr
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


def test_linux_build_rejects_broken_native_symlinks(tmp_path: Path) -> None:
    root, environment = _asset_fixture(tmp_path)
    (Path(environment["PACKAGE_FIXTURE"]) / "native.so").unlink()
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert "linked.so" in result.stderr
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


def test_linux_build_fails_if_native_file_enumeration_fails(tmp_path: Path) -> None:
    root, environment = _asset_fixture(tmp_path)
    find = Path(environment["PATH"]) / "find"
    find.unlink()
    _write_executable(find, "#!/bin/sh\nexit 1\n")
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert not (root / "build/claudex-gateway-0.17.0-darwin-arm64.tar.gz").exists()


def test_linux_build_requires_host_python_before_assembly(tmp_path: Path) -> None:
    root, environment = _asset_fixture(tmp_path)
    environment["HOST_PYTHON_FAILURE"] = "1"
    result = _run_asset_fixture(root, environment)
    assert result.returncode != 0
    assert "could not obtain host CPython 3.12" in result.stderr
    assert not (root / "build").exists()
