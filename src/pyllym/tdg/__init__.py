"""TDG (test-driven generator): turn a TOML spec into a red pytest suite that
serves as the building instruction for an implementing agent.

TOML is the single spec front-end. It carries the API surface, coverable rules
and edge cases, optional mermaid ``[[sequences]]`` blocks (parsed by
:mod:`~pyllym.tdg.sequence`), and — in *plan mode* — hand-written ``[[cases]]``.

Pipeline: parse (:mod:`toml_spec`) -> plan (deterministic ``to_test_plan`` in
plan mode, or a structured-output call via :mod:`planner` in request mode) ->
deterministic rendering (:mod:`renderer`) plus an implementation brief
(:mod:`brief`)::

    from pyllym.tdg import build

    results = await build("features/", "tests/generated", model="gpt-5.4")

Or from the shell::

    python -m pyllym.tdg features/ --out tests/generated --model gpt-5.4

The LLM only ever fills the :class:`~pyllym.tdg.schema.TestPlan` schema; test
code is rendered from templates, so plans are reviewable and re-runs diff
cleanly. Review the plan and generated tests *before* implementation — the
suite is treated as the specification from then on.
"""

from __future__ import annotations

from . import toml_spec
from .builder import BuildResult, PlanSafetyError, PlanValidationError, build
from .checks import validate_plan
from .deps import Dependencies, check_dependencies, inspect_dependencies
from .planner import plan_from_spec
from .renderer import render_conftest, render_tests
from .runs import LatestPointer, read_latest
from .safety import scan_plan
from .schema import ApiSurface, FixtureDef, TestCase, TestPlan
from .sequence import SequenceDiagram, SequenceMessage
from .toml_spec import TomlSpec, parse, parse_file

__all__ = [
    "ApiSurface",
    "BuildResult",
    "Dependencies",
    "FixtureDef",
    "LatestPointer",
    "PlanSafetyError",
    "PlanValidationError",
    "SequenceDiagram",
    "SequenceMessage",
    "TestCase",
    "TestPlan",
    "TomlSpec",
    "build",
    "check_dependencies",
    "inspect_dependencies",
    "parse",
    "parse_file",
    "plan_from_spec",
    "read_latest",
    "render_conftest",
    "render_tests",
    "scan_plan",
    "toml_spec",
    "validate_plan",
]
