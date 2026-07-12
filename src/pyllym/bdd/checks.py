"""Deterministic cross-checks between the spec and the LLM-produced plan.

This is the airtight half of the pipeline: coverage accounting happens in
plain code, never by asking the model whether it covered everything. A plan
that drops a scenario, ignores a sequence-diagram message, skips a required
rule/edge id, or claims an id that doesn't exist fails the build.
"""

from __future__ import annotations

from .schema import TestPlan


def _message_sort_key(mid: str) -> tuple[int, str]:
    """Sort 'M1', 'M2', ... numerically; fall back to lexical for other ids."""
    if mid.startswith("M") and mid[1:].isdigit():
        return (0, f"{int(mid[1:]):020d}")
    return (1, mid)


def validate_plan(
    plan: TestPlan,
    *,
    scenario_names: set[str],
    message_ids: set[str],
    required_ids: set[str],
) -> list[str]:
    """Return human-readable problems; an empty list means the plan is sound.

    ``scenario_names`` are the scenarios every plan must cover (matched against
    ``TestCase.scenario``). ``message_ids`` (sequence-diagram message ids) and
    ``required_ids`` (rule/edge ids) are BOTH valid targets for
    ``TestCase.covers``; every id in either set must be claimed by some case,
    and any claimed id outside their union is rejected as unknown.
    """
    problems: list[str] = []

    covered_scenarios = {case.scenario for case in plan.cases}
    for scenario in sorted(scenario_names):
        if scenario not in covered_scenarios:
            problems.append(f"scenario not covered by any test case: {scenario!r}")

    valid_ids = message_ids | required_ids
    claimed = {mid for case in plan.cases for mid in case.covers}

    for cid in sorted(claimed - valid_ids):
        problems.append(f"test case claims unknown id: {cid!r}")

    for mid in sorted(message_ids - claimed, key=_message_sort_key):
        problems.append(f"sequence message not covered by any test case: {mid}")

    for rid in sorted(required_ids - claimed):
        problems.append(f"rule/edge case not covered by any test case: {rid}")

    seen_names: set[str] = set()
    for case in plan.cases:
        if case.test_name in seen_names:
            problems.append(f"duplicate test name in plan: {case.test_name!r}")
        seen_names.add(case.test_name)

    return problems
