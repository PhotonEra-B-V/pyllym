"""The structured contract the planner LLM fills in.

The model never writes freeform test code — it fills :class:`TestPlan`, and
the renderer turns that into pytest source deterministically. Field
descriptions reach the provider as part of the JSON schema, so they double
as prompt instructions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApiSurface(BaseModel):
    """The public API the tests exercise — this is the design commitment."""

    module: str = Field(
        description="Dotted import path of the module under test, e.g. 'myapp.retry'"
    )
    imports: list[str] = Field(
        default_factory=list,
        description="Exact import statements the test file needs, "
        "e.g. 'from myapp.retry import RetryPolicy'",
    )
    signatures: list[str] = Field(
        description="Public def/class signatures to implement, with PEP 604 type hints, "
        "e.g. 'def backoff(attempt: int, *, base: float = 0.5) -> float'"
    )


class FixtureDef(BaseModel):
    name: str = Field(description="Valid Python identifier for the pytest fixture")
    body: list[str] = Field(
        description="Executable statements for the fixture body; "
        "the last must be a 'return' or 'yield'"
    )


class TestCase(BaseModel):
    __test__ = False  # keep pytest from collecting the Test*-named schema

    scenario: str = Field(description="Name of the scenario this case comes from")
    test_name: str = Field(description="snake_case pytest function name, starting with 'test_'")
    given: list[str] = Field(
        default_factory=list,
        description="Arrange statements — executable Python, one statement per entry",
    )
    when: str = Field(
        description="The single act expression; its value is assigned to a variable "
        "named 'result'. Include 'await' if the call is async."
    )
    then: list[str] = Field(
        description="Boolean assert expressions (without the 'assert' keyword), "
        "typically about 'result'"
    )
    fixtures: list[str] = Field(
        default_factory=list, description="Names of fixtures this test requests as parameters"
    )
    covers: list[str] = Field(
        default_factory=list,
        description="Ids this test exercises: sequence-diagram message ids (e.g. 'M1') "
        "and/or rule/edge-case ids (e.g. 'rule_1', 'edge_1'); "
        "required when the feature has sequence diagrams, rules, or edge cases",
    )
    is_async: bool = Field(
        default=False, description="True when the act or arrange steps need await"
    )
    edge_case: bool = Field(
        default=False,
        description="True when this case was derived beyond the literal scenarios "
        "(boundary, error path)",
    )
    notes: str = Field(default="", description="One line of implementer-facing context, if useful")


class TestPlan(BaseModel):
    __test__ = False  # keep pytest from collecting the Test*-named schema

    feature: str = Field(description="The feature name, verbatim from the spec")
    api: ApiSurface
    fixtures: list[FixtureDef] = Field(default_factory=list)
    cases: list[TestCase]
    required_ids: list[str] = Field(
        default_factory=list,
        description="Rule and edge-case ids (e.g. 'rule_1', 'edge_1') that every "
        "sound plan MUST cover via some TestCase.covers entry",
    )
