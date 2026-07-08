"""Orchestrate the pipeline: parse -> plan -> render tests + brief to disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brief import render_brief
from .checks import validate_plan
from .planner import plan_from_spec
from .renderer import render_conftest, render_tests
from .safety import scan_plan
from .schema import TestPlan
from .toml_spec import TomlSpec
from .toml_spec import parse as parse_toml
from .toml_spec import parse_file as parse_toml_file


class PlanValidationError(ValueError):
    """The LLM-produced (or hand-written) plan failed the deterministic
    cross-checks."""

    def __init__(self, source_name: str, problems: list[str]) -> None:
        self.source_name = source_name
        self.problems = problems
        details = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            f"plan for {source_name!r} failed validation "
            f"(the plan JSON was still written for inspection):\n{details}"
        )


class PlanSafetyError(ValueError):
    """A plan's executable strings tripped the AST safety gate.

    Raised before rendering — regardless of ``strict`` — so no runnable suite
    is ever written from a plan containing imports, dangerous builtins/names,
    dunder reflection, or non-Python. This is defense-in-depth, not a sandbox
    (see :mod:`~pyllm.bdd.safety`): a human must still review generated tests
    before they are run in a privileged environment.
    """

    def __init__(self, source_name: str, problems: list[str]) -> None:
        self.source_name = source_name
        self.problems = problems
        details = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            f"plan for {source_name!r} failed the safety gate "
            f"(nothing runnable was written; the plan JSON was kept for "
            f"inspection):\n{details}"
        )


@dataclass(frozen=True)
class BuildResult:
    feature: TomlSpec | None
    plan: TestPlan
    test_path: Path
    plan_path: Path
    brief_path: Path


def _discover(source: str | Path) -> list[TomlSpec]:
    if isinstance(source, str) and "\n" in source:
        return [parse_toml(source)]
    path = Path(source)
    if path.is_dir():
        tomls = sorted(path.rglob("*.toml"))
        if not tomls:
            raise FileNotFoundError(f"no .toml specs under {path}")
        return [parse_toml_file(t) for t in tomls]
    return [parse_toml_file(path)]


async def build(
    source: str | Path,
    out_dir: str | Path,
    *,
    api_hint: str | None = None,
    strict: bool = True,
    toml_mode: str = "auto",
    **chat_kwargs: Any,
) -> list[BuildResult]:
    """Turn TOML specs into a red pytest suite plus build briefs.

    ``source`` is raw TOML text, a ``.toml`` spec file, or a directory of them.
    For each spec, writes into ``out_dir``:

    - ``test_<slug>.py`` — the failing suite (the specification)
    - ``<slug>.plan.json`` — the reviewable plan the tests were rendered from
    - ``BRIEF_<slug>.md`` — the building instruction for the implementer
    - ``conftest.py`` — registers the ``bdd_pending`` marker (written once)

    TOML is the single front-end. A spec with hand-written ``[[cases]]`` is
    'plan' mode (``spec.to_test_plan()`` — no LLM); a loose spec with only
    rules/edge_cases is 'request' mode (:func:`~pyllm.bdd.planner.
    plan_from_spec` fills the plan via the model). Mode is auto-detected from
    the presence of ``[[cases]]`` unless ``[meta].mode`` (or ``toml_mode``) is
    set. Specs may absorb mermaid ``sequenceDiagram`` blocks via ``[[sequences]]``
    tables (parsed by :mod:`~pyllm.bdd.sequence`). The plan is then cross-checked
    deterministically (:func:`~pyllm.bdd.checks.validate_plan`): every scenario,
    every diagram message, and every required rule/edge id must be covered by at
    least one test case. With ``strict=True`` (default) a failing check raises
    :class:`PlanValidationError` after writing the plan JSON for inspection;
    with ``strict=False`` problems are listed in the brief instead.

    Before rendering, every plan (both modes, regardless of ``strict``) passes
    an AST safety gate (:func:`~pyllm.bdd.safety.scan_plan`): a case's
    ``given``/``when``/``then`` and fixture bodies become executable Python, so
    imports, dangerous builtins/names, dunder reflection, or non-Python raise
    :class:`PlanSafetyError` and nothing runnable is written. This is
    defense-in-depth, not a sandbox — a human must still review the generated
    suite before running it anywhere with secrets or network access.

    Review the plan and tests before pointing an implementer at the brief:
    once the suite is treated as the spec, a misread scenario is locked in.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conftest = out / "conftest.py"
    if not conftest.exists():
        conftest.write_text(render_conftest(), encoding="utf-8")

    results: list[BuildResult] = []
    for spec in _discover(source):
        if _spec_mode(spec, toml_mode) == "plan":
            test_plan = spec.to_test_plan()  # NO LLM
        else:  # request mode
            test_plan = await plan_from_spec(spec, api_hint=api_hint, **chat_kwargs)
        slug = spec.slug
        source_name = spec.meta.feature_name
        scenario_names = spec.scenario_names()
        message_ids = spec.message_ids()
        required_ids = spec.required_ids()

        test_path = out / f"test_{slug}.py"
        plan_path = out / f"{slug}.plan.json"
        brief_path = out / f"BRIEF_{slug}.md"
        plan_path.write_text(json.dumps(test_plan.model_dump(), indent=2) + "\n", encoding="utf-8")
        # Safety gate first, and unconditionally: an unsafe executable string
        # must never be downgraded to a warning the way a coverage gap can.
        safety_problems = scan_plan(test_plan)
        if safety_problems:
            raise PlanSafetyError(source_name, safety_problems)
        problems = validate_plan(
            test_plan,
            scenario_names=scenario_names,
            message_ids=message_ids,
            required_ids=required_ids,
        )
        if problems and strict:
            raise PlanValidationError(source_name, problems)
        test_path.write_text(render_tests(test_plan), encoding="utf-8")
        brief_path.write_text(
            render_brief(spec, test_plan, tests_path=str(test_path), problems=problems),
            encoding="utf-8",
        )
        results.append(BuildResult(spec, test_plan, test_path, plan_path, brief_path))
    return results


def _spec_mode(spec: TomlSpec, toml_mode: str = "auto") -> str:
    """Resolve the build mode for a TOML spec.

    An explicit ``toml_mode`` of ``plan`` or ``request`` (from ``--toml-mode``)
    overrides everything. Otherwise fall back to the spec's own auto-detection
    (``[meta].mode`` if set, else ``plan`` iff it carries ``[[cases]]``).
    """
    if toml_mode in ("plan", "request"):
        return toml_mode
    return spec.mode
