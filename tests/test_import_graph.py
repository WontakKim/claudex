"""Architectural checks for the internal source-module import graph."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_PACKAGE_NAME = "claudex_gateway"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / _PACKAGE_NAME

ImportGraph = dict[str, set[str]]


def _discover_source_modules(package_root: Path) -> tuple[dict[str, Path], set[str]]:
    source_modules: dict[str, Path] = {}
    package_modules: set[str] = set()

    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = source_path.relative_to(package_root.parent).with_suffix("")
        module_parts = relative_path.parts
        if source_path.name == "__init__.py":
            module_parts = module_parts[:-1]

        module_name = ".".join(module_parts)
        source_modules[module_name] = source_path
        if source_path.name == "__init__.py":
            package_modules.add(module_name)

    return source_modules, package_modules


def _resolve_import_from_base(
    importing_module: str,
    imported_module: str | None,
    level: int,
    package_modules: set[str],
) -> str | None:
    if level == 0:
        return imported_module

    importing_package = (
        importing_module
        if importing_module in package_modules
        else importing_module.rpartition(".")[0]
    )
    package_parts = importing_package.split(".") if importing_package else []
    if level > len(package_parts):
        return None

    base_parts = package_parts[: len(package_parts) - level + 1]
    if imported_module:
        base_parts.extend(imported_module.split("."))
    return ".".join(base_parts)


def _is_internal_module(module_name: str) -> bool:
    return module_name == _PACKAGE_NAME or module_name.startswith(f"{_PACKAGE_NAME}.")


def _extract_internal_imports(
    source: str,
    importing_module: str,
    source_modules: set[str],
    package_modules: set[str],
) -> set[str]:
    dependencies: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for imported_name in node.names:
                if (
                    _is_internal_module(imported_name.name)
                    and imported_name.name in source_modules
                ):
                    dependencies.add(imported_name.name)
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        imported_base = _resolve_import_from_base(
            importing_module,
            node.module,
            node.level,
            package_modules,
        )
        if imported_base is None or not _is_internal_module(imported_base):
            continue

        if imported_base in source_modules and not (
            importing_module in package_modules
            and imported_base == importing_module
        ):
            dependencies.add(imported_base)

        for imported_name in node.names:
            member_module = f"{imported_base}.{imported_name.name}"
            if member_module in source_modules:
                dependencies.add(member_module)

    return dependencies


def _build_import_graph(package_root: Path) -> ImportGraph:
    source_module_paths, package_modules = _discover_source_modules(package_root)
    source_modules = set(source_module_paths)
    return {
        module_name: _extract_internal_imports(
            source_path.read_text(encoding="utf-8"),
            module_name,
            source_modules,
            package_modules,
        )
        for module_name, source_path in source_module_paths.items()
    }


def _find_strongly_connected_components(graph: ImportGraph) -> list[tuple[str, ...]]:
    next_index = 0
    indexes: dict[str, int] = {}
    low_links: dict[str, int] = {}
    stack: list[str] = []
    stacked_modules: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(module_name: str) -> None:
        nonlocal next_index

        indexes[module_name] = next_index
        low_links[module_name] = next_index
        next_index += 1
        stack.append(module_name)
        stacked_modules.add(module_name)

        for dependency in sorted(graph[module_name]):
            if dependency not in graph:
                continue
            if dependency not in indexes:
                visit(dependency)
                low_links[module_name] = min(
                    low_links[module_name], low_links[dependency]
                )
            elif dependency in stacked_modules:
                low_links[module_name] = min(
                    low_links[module_name], indexes[dependency]
                )

        if low_links[module_name] != indexes[module_name]:
            return

        component: list[str] = []
        while True:
            dependency = stack.pop()
            stacked_modules.remove(dependency)
            component.append(dependency)
            if dependency == module_name:
                break
        components.append(tuple(sorted(component)))

    for module_name in sorted(graph):
        if module_name not in indexes:
            visit(module_name)

    return sorted(components)


def _find_cyclic_components(graph: ImportGraph) -> list[tuple[str, ...]]:
    return [
        component
        for component in _find_strongly_connected_components(graph)
        if len(component) > 1 or component[0] in graph[component[0]]
    ]


def _find_path(
    graph: ImportGraph,
    current: str,
    target: str,
    allowed_modules: set[str],
    visited_modules: set[str],
) -> tuple[str, ...] | None:
    if current == target:
        return (current,)

    visited_modules.add(current)
    for dependency in sorted(graph[current] & allowed_modules):
        if dependency in visited_modules and dependency != target:
            continue
        remaining_path = _find_path(
            graph,
            dependency,
            target,
            allowed_modules,
            visited_modules,
        )
        if remaining_path is not None:
            return (current, *remaining_path)
    return None


def _find_cycle_path(graph: ImportGraph, component: tuple[str, ...]) -> tuple[str, ...]:
    component_modules = set(component)
    for start in component:
        for dependency in sorted(graph[start] & component_modules):
            if dependency == start:
                return (start, start)
            return_path = _find_path(
                graph,
                dependency,
                start,
                component_modules,
                {start},
            )
            if return_path is not None:
                return (start, *return_path)

    raise AssertionError(f"No cycle path found in strongly connected component: {component}")


def _format_cycle_failure(
    graph: ImportGraph, cyclic_components: list[tuple[str, ...]]
) -> str:
    details = ["Internal import graph contains cycles:"]
    for component in cyclic_components:
        cycle_path = _find_cycle_path(graph, component)
        details.append(f"component: {', '.join(component)}")
        details.append(f"cycle: {' -> '.join(cycle_path)}")
    return "\n".join(details)


_SOURCE_MODULE_PATHS, _PACKAGE_MODULES = _discover_source_modules(_PACKAGE_ROOT)
_SOURCE_MODULES = set(_SOURCE_MODULE_PATHS)


def test_resolves_absolute_package_member_import() -> None:
    source = "from claudex_gateway import paths"

    dependencies = _extract_internal_imports(
        source,
        "claudex_gateway.config",
        _SOURCE_MODULES,
        _PACKAGE_MODULES,
    )

    assert "claudex_gateway.paths" in dependencies


def test_resolves_relative_package_member_import() -> None:
    source = "from . import paths"

    dependencies = _extract_internal_imports(
        source,
        "claudex_gateway.config",
        _SOURCE_MODULES,
        _PACKAGE_MODULES,
    )

    assert "claudex_gateway.paths" in dependencies


def test_package_initializer_member_import_adds_no_self_edge() -> None:
    source = "from . import claude_to_codex"

    dependencies = _extract_internal_imports(
        source,
        "claudex_gateway.translate",
        _SOURCE_MODULES,
        _PACKAGE_MODULES,
    )

    assert "claudex_gateway.translate.claude_to_codex" in dependencies
    assert "claudex_gateway.translate" not in dependencies


def test_module_explicit_self_import_keeps_self_edge() -> None:
    source = "import claudex_gateway.config"

    dependencies = _extract_internal_imports(
        source,
        "claudex_gateway.config",
        _SOURCE_MODULES,
        _PACKAGE_MODULES,
    )

    assert "claudex_gateway.config" in dependencies


@pytest.mark.parametrize(
    ("source", "expected_dependency"),
    [
        ("import claudex_gateway.paths", "claudex_gateway.paths"),
        (
            "from claudex_gateway.paths import runtime_dir",
            "claudex_gateway.paths",
        ),
    ],
)
def test_resolves_direct_internal_imports(
    source: str, expected_dependency: str
) -> None:
    dependencies = _extract_internal_imports(
        source,
        "claudex_gateway.config",
        _SOURCE_MODULES,
        _PACKAGE_MODULES,
    )

    assert expected_dependency in dependencies


@pytest.fixture
def multi_module_cycle_graph() -> ImportGraph:
    return {
        "claudex_gateway.alpha": {"claudex_gateway.bravo"},
        "claudex_gateway.bravo": {"claudex_gateway.charlie"},
        "claudex_gateway.charlie": {"claudex_gateway.alpha"},
        "claudex_gateway.independent": set(),
    }


@pytest.fixture
def self_cycle_graph() -> ImportGraph:
    return {"claudex_gateway.self_reference": {"claudex_gateway.self_reference"}}


def test_detects_multi_module_cycle(multi_module_cycle_graph: ImportGraph) -> None:
    assert _find_cyclic_components(multi_module_cycle_graph) == [
        (
            "claudex_gateway.alpha",
            "claudex_gateway.bravo",
            "claudex_gateway.charlie",
        )
    ]


def test_detects_self_cycle(self_cycle_graph: ImportGraph) -> None:
    assert _find_cyclic_components(self_cycle_graph) == [
        ("claudex_gateway.self_reference",)
    ]


def test_cycle_failure_lists_complete_component_and_path(
    multi_module_cycle_graph: ImportGraph,
) -> None:
    cyclic_components = _find_cyclic_components(multi_module_cycle_graph)

    failure = _format_cycle_failure(multi_module_cycle_graph, cyclic_components)

    assert (
        "component: claudex_gateway.alpha, claudex_gateway.bravo, "
        "claudex_gateway.charlie"
    ) in failure
    assert (
        "cycle: claudex_gateway.alpha -> claudex_gateway.bravo -> "
        "claudex_gateway.charlie -> claudex_gateway.alpha"
    ) in failure


def test_internal_import_graph_is_acyclic() -> None:
    graph = _build_import_graph(_PACKAGE_ROOT)
    cyclic_components = _find_cyclic_components(graph)

    assert not cyclic_components, _format_cycle_failure(graph, cyclic_components)


@pytest.mark.parametrize("module_name", sorted(_SOURCE_MODULES))
def test_every_source_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
