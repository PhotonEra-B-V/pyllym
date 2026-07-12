"""Expand a request-mode TOML spec into a :class:`TestPlan` via structured
output.

:func:`plan_from_spec` uses the structured-output pattern
(``with_instructions`` / ``with_schema`` / ``ask``): the model only ever fills
:class:`TestPlan`; it never writes pytest text. Plan-mode specs skip this
module entirely (``TomlSpec.to_test_plan`` is deterministic, no LLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..chat import Chat
from .schema import TestPlan

if TYPE_CHECKING:
    from .toml_spec import TomlSpec

SPEC_PLANNER_INSTRUCTIONS = """\
You are a test architect. You receive one behavior specification (an API
surface plus rules, edge cases, and optional sequence diagrams) and produce a
test plan that will be rendered into a failing pytest suite. The suite is
written BEFORE any implementation exists — it is the specification an
implementer must satisfy without editing the tests.

Rules:
- Commit to the public API surface exactly as given: module path, imports, and
  the exact signatures. Every test must exercise only that surface.
- Write one or more test cases covering the rules, plus derived edge cases
  (boundaries, error paths) marked edge_case=true. Do not invent behavior the
  spec does not imply.
- 'given' entries, fixture bodies, 'when', and 'then' must be executable
  Python against the declared API. 'when' is a single expression whose value
  is assigned to `result`; include `await` when the call is async and set
  is_async accordingly. 'then' entries are bare boolean expressions about
  `result` (no `assert` keyword).
- For error paths, make the whole act-and-assert a 'given' statement using
  the canonical form:
    with pytest.raises(SomeError): call(...)
  then set when = "None" and leave 'then' empty. ('import pytest' is already
  present in the rendered file.)
- Prefer plain values over mocks. Use fixtures only for setup shared by
  several cases. Fixture bodies end with 'return' or 'yield'.
- test_name values must be unique, snake_case, and start with 'test_'.

COVERAGE IS MECHANICALLY ENFORCED — the build fails if any id is uncovered:
- Set 'covers' on each test case to the ids it exercises. Every rule id, every
  edge-case id, and every sequence-diagram message id listed below MUST appear
  in some case's 'covers'. Never invent ids; only claim ids that were given.

When sequence diagrams are present, they are binding collaboration contracts:
- Design the API so the collaborators named in a diagram are injectable
  (constructor or function parameters) so tests can substitute recording fakes.
- Add at least one interaction-order case: build a plain `calls = []` list in
  'given', pass fakes that append to it, and assert in 'then' that the recorded
  order matches the diagram (e.g. calls == ['query_users']).
"""


def _spec_prompt(spec: TomlSpec, *, api_hint: str | None) -> str:
    """Build the request-mode prompt from a :class:`TomlSpec`.

    Prefers ``spec.to_prompt()`` when available; otherwise assembles the prompt
    defensively from the spec's parts so a minor method-name drift in the
    toml_spec module doesn't break the planner.
    """
    to_prompt = getattr(spec, "to_prompt", None)
    if callable(to_prompt):
        prompt = to_prompt()
    else:  # defensive fallback — assemble from parts
        prompt = _assemble_spec_prompt(spec)
    if api_hint:
        prompt += f"\nTarget API constraints (honor these exactly):\n{api_hint}\n"
    return prompt


def _assemble_spec_prompt(spec: TomlSpec) -> str:
    parts: list[str] = []
    api = getattr(spec, "api", None)
    if api is not None:
        parts.append(f"Module under test: {getattr(api, 'module', '')}")
        signatures = getattr(api, "signatures", None) or []
        if signatures:
            parts.append("Signatures:\n" + "\n".join(f"  {s}" for s in signatures))
    rules = getattr(spec, "rules", None) or []
    if rules:
        parts.append(
            "Rules (cover every id via 'covers'):\n"
            + "\n".join(f"  {_id_of(r)}: {_text_of(r)}" for r in rules)
        )
    edge_cases = getattr(spec, "edge_cases", None) or []
    if edge_cases:
        parts.append(
            "Edge cases (cover every id via 'covers'):\n"
            + "\n".join(f"  {_id_of(e)}: {_text_of(e)}" for e in edge_cases)
        )
    sequences = getattr(spec, "sequences", None) or []
    for diagram in sequences:
        annotate = getattr(diagram, "to_annotated", None)
        if callable(annotate):
            parts.append("Sequence-diagram messages (cover every id):\n" + annotate())
    return "\n\n".join(parts) + "\n"


def _id_of(item: Any) -> str:
    return getattr(item, "id", None) or (item.get("id", "") if isinstance(item, dict) else "")


def _text_of(item: Any) -> str:
    return getattr(item, "text", None) or (item.get("text", "") if isinstance(item, dict) else "")


async def plan_from_spec(
    spec: TomlSpec,
    *,
    api_hint: str | None = None,
    chat: Chat | None = None,
    **chat_kwargs: Any,
) -> TestPlan:
    """Ask a model to expand a loose (request-mode) :class:`TomlSpec` into a
    :class:`TestPlan`.

    Only used when the TOML carries no hand-written ``[[cases]]`` (plan mode
    skips the LLM entirely via ``spec.to_test_plan()``). The prompt comes from
    ``spec.to_prompt()``; the model is instructed to claim EVERY rule, edge, and
    sequence-message id in some case's ``covers`` — uncovered ids fail the build
    mechanically in :func:`~pyllym.tdg.checks.validate_plan`.
    """
    chat = (
        (chat or Chat(**chat_kwargs))
        .with_instructions(SPEC_PLANNER_INSTRUCTIONS)
        .with_schema(TestPlan)
    )
    prompt = _spec_prompt(spec, api_hint=api_hint)
    message = await chat.ask(prompt)
    if not isinstance(message.content, dict):
        feature_name = getattr(getattr(spec, "meta", spec), "feature_name", "<spec>")
        raise ValueError(f"planner model returned non-JSON content for spec {feature_name!r}")
    return TestPlan.model_validate(message.content)
