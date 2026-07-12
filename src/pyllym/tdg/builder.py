"""Orchestrate the pipeline: parse -> plan -> render tests + brief to disk."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .brief import render_brief
from .checks import validate_plan
from .deps import check_dependencies, inspect_dependencies
from .planner import plan_from_spec
from .renderer import render_conftest, render_tests
from .runs import (
    read_latest,
    reusable,
    run_stamp,
    spec_hash,
    utc_now,
    write_done_marker,
    write_latest_pointer,
)
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
    (see :mod:`~pyllym.tdg.safety`): a human must still review generated tests
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
    reused: bool = False


def _discover(source: str | Path) -> list[tuple[TomlSpec, str]]:
    """Return ``(spec, canonical_text)`` pairs; the text feeds the run hash."""
    if isinstance(source, str) and "\n" in source:
        return [(parse_toml(source), source)]
    path = Path(source)
    if path.is_dir():
        tomls = sorted(path.rglob("*.toml"))
        if not tomls:
            raise FileNotFoundError(f"no .toml specs under {path}")
        return [(parse_toml_file(t), t.read_text(encoding="utf-8")) for t in tomls]
    return [(parse_toml_file(path), path.read_text(encoding="utf-8"))]


async def build(
    source: str | Path,
    out_dir: str | Path,
    *,
    api_hint: str | None = None,
    strict: bool = True,
    toml_mode: str = "auto",
    check_deps: bool = True,
    use_runs: bool = True,
    now: datetime | None = None,
    **chat_kwargs: Any,
) -> list[BuildResult]:
    """Turn TOML specs into a red pytest suite plus build briefs.

    ``source`` is raw TOML text, a ``.toml`` spec file, or a directory of them.
    For each spec, writes into ``out_dir``:

    - ``test_<slug>.py`` — the failing suite (the specification)
    - ``<slug>.plan.json`` — the reviewable plan the tests were rendered from
    - ``BRIEF_<slug>.md`` — the building instruction for the implementer
    - ``conftest.py`` — registers the pending-test marker (written once)

    TOML is the single front-end. A spec with hand-written ``[[cases]]`` is
    'plan' mode (``spec.to_test_plan()`` — no LLM); a loose spec with only
    rules/edge_cases is 'request' mode (:func:`~pyllym.tdg.planner.
    plan_from_spec` fills the plan via the model). Mode is auto-detected from
    the presence of ``[[cases]]`` unless ``[meta].mode`` (or ``toml_mode``) is
    set. Specs may absorb mermaid ``sequenceDiagram`` blocks via ``[[sequences]]``
    tables (parsed by :mod:`~pyllym.tdg.sequence`). The plan is then cross-checked
    deterministically (:func:`~pyllym.tdg.checks.validate_plan`): every scenario,
    every diagram message, and every required rule/edge id must be covered by at
    least one test case. With ``strict=True`` (default) a failing check raises
    :class:`PlanValidationError` after writing the plan JSON for inspection;
    with ``strict=False`` problems are listed in the brief instead.

    When ``check_deps`` is true (default), the module under test is inspected
    for its real dependencies (:func:`~pyllym.tdg.deps.inspect_dependencies` —
    static AST read, runtime-import fallback). In request mode those discovered
    dependencies are handed to the planner so generated tests target the
    module's actual collaborators; in both modes a spec whose declared
    ``[api].imports`` disagree with the module's real dependencies yields
    dependency problems that are listed in the brief and, under ``strict``,
    raise :class:`PlanValidationError` alongside coverage gaps.

    Before rendering, every plan (both modes, regardless of ``strict``) passes
    an AST safety gate (:func:`~pyllym.tdg.safety.scan_plan`): a case's
    ``given``/``when``/``then`` and fixture bodies become executable Python, so
    imports, dangerous builtins/names, dunder reflection, or non-Python raise
    :class:`PlanSafetyError` and nothing runnable is written. This is
    defense-in-depth, not a sandbox — a human must still review the generated
    suite before running it anywhere with secrets or network access.

    Review the plan and tests before pointing an implementer at the brief:
    once the suite is treated as the spec, a misread scenario is locked in.

    With ``use_runs`` (default), each call writes into a timestamped run dir
    ``out_dir/<YYYY-MM-DD-HH-MM-SS>/`` and, on success, drops a ``_DONE.json``
    marker there plus updates ``out_dir/latest.json``. The next build reads that
    pointer and skips any spec whose content hash matches the latest completed
    run (copying its artifacts forward — ``BuildResult.reused`` is ``True``);
    changed or new specs regenerate. Pass ``now`` to pin the run timestamp
    (used by tests for determinism). Set ``use_runs=False`` to write flat into
    ``out_dir`` with no run bookkeeping.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Each build lands in its own timestamped run dir; latest.json (written on
    # success) lets the *next* build skip specs it already generated unchanged.
    if use_runs:
        stamp = run_stamp(now or utc_now())
        run_dir = out / stamp
        latest = read_latest(out)
    else:  # flat, single-directory mode — no run bookkeeping
        run_dir = out
    run_dir.mkdir(parents=True, exist_ok=True)

    conftest = run_dir / "conftest.py"
    if not conftest.exists():
        conftest.write_text(render_conftest(), encoding="utf-8")

    results: list[BuildResult] = []
    hashes: dict[str, str] = {}
    for spec, spec_text in _discover(source):
        slug = spec.slug
        digest = spec_hash(spec_text)
        hashes[slug] = digest

        # Skip-if-unchanged: if the latest completed run generated this exact
        # spec, copy its artifacts into this run instead of regenerating.
        reuse = reusable(latest, out, slug, digest) if use_runs else None
        if reuse is not None:
            results.append(_copy_forward(spec, reuse.source_run_dir, run_dir, slug))
            continue

        results.append(
            await _generate_spec(
                spec,
                run_dir,
                api_hint=api_hint,
                strict=strict,
                toml_mode=toml_mode,
                check_deps=check_deps,
                **chat_kwargs,
            )
        )

    # Mark the run complete only after every spec succeeded, then publish it as
    # the latest. A run that raised above never gets a marker → never reused.
    if use_runs:
        write_done_marker(run_dir, stamp, hashes)
        write_latest_pointer(out, run_dir, stamp, hashes)
    return results


async def _generate_spec(
    spec: TomlSpec,
    run_dir: Path,
    *,
    api_hint: str | None,
    strict: bool,
    toml_mode: str,
    check_deps: bool,
    **chat_kwargs: Any,
) -> BuildResult:
    """Plan, validate, and render one spec into ``run_dir`` (the freshly-built
    path). Raises :class:`PlanSafetyError` / :class:`PlanValidationError` on a
    gate failure exactly as before."""
    # Discover the module's real dependencies before planning. In request mode
    # the discovered surface is fed to the planner (via api_hint) so generated
    # tests reflect the module's actual collaborators, not just the declared
    # surface. In both modes, a spec whose declared imports disagree with
    # reality is a dependency problem (folded into the plan problems below →
    # reported in the brief, fatal under strict).
    dep_problems: list[str] = []
    planner_hint = api_hint
    if check_deps:
        deps = inspect_dependencies(spec.api.module)
        dep_problems = check_dependencies(list(spec.api.imports), deps)
        dep_block = deps.to_prompt_block()
        planner_hint = f"{api_hint}\n{dep_block}" if api_hint else dep_block

    if _spec_mode(spec, toml_mode) == "plan":
        test_plan = spec.to_test_plan()  # NO LLM
    else:  # request mode
        test_plan = await plan_from_spec(spec, api_hint=planner_hint, **chat_kwargs)
    slug = spec.slug
    source_name = spec.meta.feature_name

    test_path = run_dir / f"test_{slug}.py"
    plan_path = run_dir / f"{slug}.plan.json"
    brief_path = run_dir / f"BRIEF_{slug}.md"
    plan_path.write_text(json.dumps(test_plan.model_dump(), indent=2) + "\n", encoding="utf-8")
    # Safety gate first, and unconditionally: an unsafe executable string must
    # never be downgraded to a warning the way a coverage gap can.
    safety_problems = scan_plan(test_plan)
    if safety_problems:
        raise PlanSafetyError(source_name, safety_problems)
    problems = dep_problems + validate_plan(
        test_plan,
        scenario_names=spec.scenario_names(),
        message_ids=spec.message_ids(),
        required_ids=spec.required_ids(),
    )
    if problems and strict:
        raise PlanValidationError(source_name, problems)
    test_path.write_text(render_tests(test_plan), encoding="utf-8")
    brief_path.write_text(
        render_brief(spec, test_plan, tests_path=str(test_path), problems=problems),
        encoding="utf-8",
    )
    return BuildResult(spec, test_plan, test_path, plan_path, brief_path)


def _copy_forward(spec: TomlSpec, source_run_dir: Path, run_dir: Path, slug: str) -> BuildResult:
    """Reuse an unchanged spec's artifacts by copying them from the prior run
    into the current run dir, so each run dir is self-contained."""
    test_path = run_dir / f"test_{slug}.py"
    plan_path = run_dir / f"{slug}.plan.json"
    brief_path = run_dir / f"BRIEF_{slug}.md"
    for name in (f"test_{slug}.py", f"{slug}.plan.json", f"BRIEF_{slug}.md"):
        src = source_run_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
    plan = TestPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    return BuildResult(spec, plan, test_path, plan_path, brief_path, reused=True)


def _spec_mode(spec: TomlSpec, toml_mode: str = "auto") -> str:
    """Resolve the build mode for a TOML spec.

    An explicit ``toml_mode`` of ``plan`` or ``request`` (from ``--toml-mode``)
    overrides everything. Otherwise fall back to the spec's own auto-detection
    (``[meta].mode`` if set, else ``plan`` iff it carries ``[[cases]]``).
    """
    if toml_mode in ("plan", "request"):
        return toml_mode
    return spec.mode
