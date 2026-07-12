"""Render the implementation brief — the building instruction an agent
(or human) follows to turn the red suite green.

The first argument is duck-typed against a :class:`~pyllym.tdg.toml_spec.
TomlSpec`. Only these attributes are read:

- ``description`` (str; read off ``spec.meta`` or the spec directly)
- ``sequences`` — a sequence of :class:`~pyllym.tdg.sequence.SequenceDiagram`
  (each with ``.source`` and ``.messages`` of ``.id``/``.sender``/
  ``.receiver``/``.text``); may be empty/absent
- ``rules`` / ``edge_cases`` — sequences of items exposing ``.id`` and
  ``.text``; may be empty/absent
"""

from __future__ import annotations

from typing import Any

from .schema import TestPlan


def _attr(spec: Any, name: str) -> list[Any]:
    """Tolerantly read an optional list-valued attribute (Feature lacks some)."""
    return list(getattr(spec, name, None) or [])


def render_brief(
    spec: Any,
    plan: TestPlan,
    *,
    tests_path: str,
    problems: list[str] | None = None,
) -> str:
    sequences = _attr(spec, "sequences")
    rules = _attr(spec, "rules")
    edge_cases = _attr(spec, "edge_cases")

    lines: list[str] = [f"# BUILD BRIEF: {plan.feature}", ""]
    meta = getattr(spec, "meta", None)
    description = getattr(meta, "description", "") or getattr(spec, "description", "") or ""
    if description:
        lines.append(description)
        lines.append("")

    if problems:
        lines.append("## ⚠ Plan validation warnings")
        lines.append("")
        lines.extend(f"- {p}" for p in problems)
        lines.append("")

    lines.append("## Target API (implement exactly these signatures)")
    lines.append("")
    lines.append(f"Module: `{plan.api.module}`")
    lines.append("")
    lines.append("```python")
    lines.extend(plan.api.signatures)
    lines.append("```")
    lines.append("")

    if sequences:
        lines.append("## Collaboration contract")
        lines.append("")
        lines.append("The implementation must produce these interactions, in order —")
        lines.append("the suite asserts recorded call sequences against them.")
        lines.append("")
        for diagram in sequences:
            lines.append("```mermaid")
            lines.append(diagram.source)
            lines.append("```")
            lines.append("")

    lines.append(f"## Test manifest ({len(plan.cases)} cases, all currently red)")
    lines.append("")
    lines.append("| Test | Scenario | Kind | Covers |")
    lines.append("|------|----------|------|--------|")
    for case in plan.cases:
        kind = "edge case" if case.edge_case else "scenario"
        covers = ", ".join(case.covers) if case.covers else "—"
        lines.append(f"| `{case.test_name}` | {case.scenario} | {kind} | {covers} |")
    lines.append("")

    if sequences:
        lines.append("### Message traceability")
        lines.append("")
        lines.append("| Message | Interaction | Tests |")
        lines.append("|---------|-------------|-------|")
        for diagram in sequences:
            for message in diagram.messages:
                tests = [c.test_name for c in plan.cases if message.id in c.covers]
                shown = ", ".join(f"`{t}`" for t in tests) if tests else "**uncovered**"
                lines.append(
                    f"| {message.id} | {message.sender} → {message.receiver}: "
                    f"{message.text} | {shown} |"
                )
        lines.append("")

    if rules or edge_cases:
        lines.append("### Rule/edge traceability")
        lines.append("")
        lines.append("| Id | Rule/Edge | Tests |")
        lines.append("|----|-----------|-------|")
        for item in (*rules, *edge_cases):
            tests = [c.test_name for c in plan.cases if item.id in c.covers]
            shown = ", ".join(f"`{t}`" for t in tests) if tests else "**uncovered**"
            lines.append(f"| {item.id} | {item.text} | {shown} |")
        lines.append("")

    lines.append("## Rules")
    lines.append("")
    lines.append(f"- The tests in `{tests_path}` are the specification. They are")
    lines.append("  read-only during implementation — if a test looks wrong, stop and")
    lines.append("  flag it; do not adapt the test to the implementation.")
    lines.append("- Implement only the API surface listed above; keep everything else")
    lines.append("  private.")
    lines.append("")

    lines.append("## Definition of done")
    lines.append("")
    lines.append(f"`pytest {tests_path}` passes with zero edits to the test file.")
    return "\n".join(lines) + "\n"
