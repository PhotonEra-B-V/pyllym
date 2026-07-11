"""Dependency inspection + cross-checking for the BDD/TDG builder."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyllm.tdg import build
from pyllm.tdg.builder import PlanValidationError
from pyllm.tdg.deps import check_dependencies, inspect_dependencies


def _write_module(tmp_path: Path, name: str, body: str) -> None:
    """Write a real importable module into an on-sys.path package dir."""
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


# -- static discovery --------------------------------------------------------


def test_static_discovers_imports_and_members(pkg: Path) -> None:
    _write_module(
        pkg,
        "retrymod",
        """
        from __future__ import annotations
        import os
        import httpx
        from dataclasses import dataclass

        @dataclass
        class RetryPolicy:
            attempts: int = 3
        """,
    )
    deps = inspect_dependencies("retrymod")
    assert deps.source == "static"
    assert "os" in deps.imports
    assert "httpx" in deps.imports
    assert "dataclasses" in deps.imports
    assert "dataclass" in deps.members
    assert not deps.warnings


def test_static_ignores_relative_imports(pkg: Path) -> None:
    (pkg / "relpkg").mkdir()
    (pkg / "relpkg" / "__init__.py").write_text("", encoding="utf-8")
    _write_module(
        pkg / "relpkg",
        "mod",
        """
        from . import sibling
        import json
        """,
    )
    deps = inspect_dependencies("relpkg.mod")
    assert "json" in deps.imports
    # relative import contributed no top-level module name
    assert "relpkg" not in deps.imports


# -- runtime fallback --------------------------------------------------------


def test_runtime_fallback_when_no_source() -> None:
    # a C-extension / builtin module has no readable .py source
    deps = inspect_dependencies("math")
    assert deps.source == "runtime"


def test_unimportable_module_degrades_to_warning() -> None:
    deps = inspect_dependencies("does_not_exist_anywhere_xyz")
    assert deps.source == "runtime"
    assert deps.warnings
    assert not deps.imports


# -- cross-checking ----------------------------------------------------------


def test_check_flags_import_from_undeclared_module(pkg: Path) -> None:
    _write_module(pkg, "onlyos", "import os\n")
    deps = inspect_dependencies("onlyos")
    problems = check_dependencies(["from httpx import AsyncClient"], deps)
    assert any("httpx" in p and "does not depend on" in p for p in problems)


def test_check_passes_when_declared_matches_real(pkg: Path) -> None:
    _write_module(pkg, "usesos", "import os\nfrom json import dumps\n")
    deps = inspect_dependencies("usesos")
    problems = check_dependencies(["import os", "from json import dumps"], deps)
    assert problems == []


def test_check_flags_missing_member_from_target(pkg: Path) -> None:
    _write_module(pkg, "surface", "from json import dumps\n")
    deps = inspect_dependencies("surface")
    # 'loads' is not actually imported by the target module
    problems = check_dependencies(["from surface import loads"], deps)
    assert any("loads" in p and "not present" in p for p in problems)


def test_warning_is_surfaced_as_problem() -> None:
    deps = inspect_dependencies("nope_missing_module_abc")
    problems = check_dependencies([], deps)
    assert any("could not import" in p for p in problems)


# -- builder integration (plan mode → no LLM) --------------------------------


PLAN_SPEC = """
[meta]
feature_name = "Uses OS"
mode = "plan"

[api]
module = "{module}"
imports = ["from {module} import {member}"]
signatures = ["def run() -> int"]

[[rules]]
id = "rule_1"
text = "returns an int"

[[cases]]
test_name = "test_run_returns_int"
scenario = "runs"
when = "run()"
then = ["isinstance(result, int)"]
covers = ["rule_1"]
"""


async def test_build_strict_fails_on_dep_mismatch(pkg: Path, tmp_path: Path) -> None:
    _write_module(pkg, "planmod", "import os\n\ndef run() -> int:\n    return 0\n")
    spec = PLAN_SPEC.format(module="planmod", member="nonexistent_thing")
    with pytest.raises(PlanValidationError) as exc:
        await build(spec, tmp_path / "out", strict=True)
    assert any("planmod" in p for p in exc.value.problems)


async def test_build_reports_dep_mismatch_in_brief_when_not_strict(
    pkg: Path, tmp_path: Path
) -> None:
    _write_module(pkg, "planmod2", "import os\n\ndef run() -> int:\n    return 0\n")
    spec = PLAN_SPEC.format(module="planmod2", member="nonexistent_thing")
    results = await build(spec, tmp_path / "out", strict=False)
    brief = results[0].brief_path.read_text(encoding="utf-8")
    assert "not present" in brief or "does not depend on" in brief


async def test_build_no_check_deps_skips_inspection(pkg: Path, tmp_path: Path) -> None:
    _write_module(pkg, "planmod3", "import os\n\ndef run() -> int:\n    return 0\n")
    spec = PLAN_SPEC.format(module="planmod3", member="nonexistent_thing")
    # would raise under strict if deps were checked; check_deps=False skips it
    results = await build(spec, tmp_path / "out", strict=True, check_deps=False)
    assert results[0].plan.cases
