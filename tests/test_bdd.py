from __future__ import annotations

import json

import pytest

from pyllm.bdd import (
    ApiSurface,
    FixtureDef,
    PlanSafetyError,
    PlanValidationError,
    TestCase,
    TestPlan,
    build,
    parse,
    parse_file,
    plan_from_spec,
    render_tests,
    scan_plan,
    sequence,
    toml_spec,
    validate_plan,
)

from .conftest import sent_requests


def sample_plan() -> TestPlan:
    return TestPlan(
        feature="Retry policy",
        api=ApiSurface(
            module="myapp.retry",
            imports=["from myapp.retry import backoff"],
            signatures=["def backoff(attempt: int, *, base: float = 0.5) -> float"],
        ),
        fixtures=[FixtureDef(name="base", body=["return 0.5"])],
        cases=[
            TestCase(
                scenario="First retry uses the base delay",
                test_name="test_first_retry_uses_base_delay",
                given=[],
                when="backoff(1, base=base)",
                then=["result == 0.5"],
                fixtures=["base"],
            ),
            TestCase(
                scenario="Delays are capped",
                test_name="test_delay_is_capped",
                when="backoff(50)",
                then=["result <= 30"],
            ),
            TestCase(
                scenario="Delays are capped",
                test_name="test_negative_attempt_rejected",
                given=["with pytest.raises(ValueError):", "    backoff(-1)"],
                when="None",
                then=[],
                edge_case=True,
            ),
        ],
    )


MERMAID = """\
sequenceDiagram
    participant Client
    Client->>+Server: GET /users
    Server->>+Database: Query Users
    Database-->>-Server: Return Data
    alt service unavailable
        Server-->>Client: 503 Unavailable
    end
    Server-->>-Client: 200 OK
"""


# --- sequence diagrams -----------------------------------------------------------
def test_parse_sequence_diagram():
    diagram = sequence.parse(MERMAID)
    assert diagram.participants == ("Client", "Server", "Database")
    assert [m.id for m in diagram.messages] == ["M1", "M2", "M3", "M4", "M5"]
    assert diagram.messages[0].sender == "Client"
    assert diagram.messages[0].receiver == "Server"
    assert diagram.messages[0].text == "GET /users"
    assert diagram.messages[0].kind == "call"
    assert diagram.messages[2].kind == "reply"
    assert diagram.messages[3].context == ("alt service unavailable",)
    assert diagram.messages[4].context == ()
    assert "M2: Server -[call]-> Database: Query Users" in diagram.to_annotated()


def test_parse_sequence_requires_header():
    with pytest.raises(ValueError):
        sequence.parse("Client->>Server: hi\n")


# --- renderer ------------------------------------------------------------------
def test_render_tests_is_valid_python():
    source = render_tests(sample_plan())
    compile(source, "<generated>", "exec")
    assert "pytestmark = [pytest.mark.bdd_pending]" in source
    assert "from myapp.retry import backoff" in source
    assert "def test_first_retry_uses_base_delay(base):" in source
    assert "result = backoff(1, base=base)" in source
    assert "assert result == 0.5" in source
    # canonical error-path form: act lives in `given`, no result assignment
    assert "with pytest.raises(ValueError):" in source
    assert "# Edge case derived from: Delays are capped" in source


def test_render_sanitizes_and_dedupes_names():
    plan_ = sample_plan()
    plan_.cases[0].test_name = "first retry!"
    plan_.cases[1].test_name = "first retry!"
    source = render_tests(plan_)
    compile(source, "<generated>", "exec")
    assert "def test_first_retry(base):" in source
    assert "def test_first_retry_2():" in source


def test_render_async_case():
    plan_ = sample_plan()
    plan_.cases[1].is_async = True
    plan_.cases[1].when = "await backoff(50)"
    source = render_tests(plan_)
    compile(source, "<generated>", "exec")
    assert "@pytest.mark.asyncio" in source
    assert "async def test_delay_is_capped():" in source


def test_render_includes_covers_comment():
    plan_ = sample_plan()
    plan_.cases[0].covers = ["M1", "M4"]
    source = render_tests(plan_)
    compile(source, "<generated>", "exec")
    assert "# Covers: M1, M4" in source


# --- planner mock ---------------------------------------------------------------
def _mock_planner_response(mock_http) -> None:
    payload = {
        "id": "cmpl-test",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(sample_plan().model_dump()),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_http.post("https://api.openai.com/v1/chat/completions", payload=payload, repeat=True)


# --- toml front-end ------------------------------------------------------------
# A complete plan-mode spec: hand-written [[cases]], so NO LLM is consulted.
PLAN_TOML = """\
[meta]
feature_name = "Retry policy"
description = "Exponential backoff with a cap, so transient failures self-heal."
async_required = false

[api]
module = "myapp.retry"
imports = ["from myapp.retry import backoff"]
signatures = ["def backoff(attempt: int, *, base: float = 0.5) -> float"]

[[rules]]
id = "rule_1"
text = "The first retry uses the base delay."

[[edge_cases]]
id = "edge_1"
text = "A negative attempt is rejected."

[[fixtures]]
name = "base"
body = ["return 0.5"]

[[cases]]
test_name = "test_first_retry_uses_base_delay"
scenario = "First retry uses the base delay"
when = "backoff(1, base=base)"
then = ["result == 0.5"]
fixtures = ["base"]
covers = ["rule_1"]

[[cases]]
test_name = "test_negative_attempt_rejected"
scenario = "Negative attempt is rejected"
given = ["with pytest.raises(ValueError):", "    backoff(-1)"]
when = "None"
then = []
edge_case = true
covers = ["edge_1"]
"""

# A loose request-mode spec: rules/edge_cases only, NO [[cases]] -> planner LLM.
REQUEST_TOML = """\
[meta]
feature_name = "Retry policy"
description = "Exponential backoff with a cap."

[api]
module = "myapp.retry"
signatures = ["def backoff(attempt: int, *, base: float = 0.5) -> float"]

[[rules]]
id = "rule_1"
text = "The first retry uses the base delay."

[[edge_cases]]
id = "edge_1"
text = "A negative attempt is rejected."
"""

# A spec that ABSORBS mermaid via [[sequences]] (parsed by the existing sequence.py).
SEQUENCE_TOML = (
    "[meta]\n"
    'feature_name = "User listing"\n'
    "\n"
    "[api]\n"
    'module = "myapp.users"\n'
    'signatures = ["def list_users(client, db) -> list"]\n'
    "\n"
    "[[cases]]\n"
    'test_name = "test_listing_users"\n'
    'scenario = "Listing users"\n'
    'when = "list_users(client, db)"\n'
    'then = ["result == []"]\n'
    'covers = ["M1"]\n'
    "\n"
    "[[sequences]]\n"
    'mermaid = """\n' + MERMAID + '"""\n'
)


def test_toml_parse_basics():
    """The TOML front-end parses meta/api and derives slug + mode."""
    spec = parse(PLAN_TOML)
    assert spec.meta.feature_name == "Retry policy"
    assert "self-heal" in spec.meta.description
    assert spec.slug == "retry_policy"
    assert spec.api.module == "myapp.retry"
    assert spec.api.signatures == ("def backoff(attempt: int, *, base: float = 0.5) -> float",)


def test_toml_parse_requires_feature_name():
    with pytest.raises(ValueError):
        parse('[api]\nmodule = "x"\n')


def test_toml_plan_mode_round_trip_no_llm(mock_http):
    """Plan-mode: hand-written cases render straight to valid pytest, no HTTP."""
    spec = parse(PLAN_TOML)
    assert spec.mode == "plan"
    assert spec.slug == "retry_policy"

    plan_ = spec.to_test_plan()
    assert isinstance(plan_, TestPlan)
    assert plan_.feature == "Retry policy"
    assert plan_.api.module == "myapp.retry"
    assert {c.test_name for c in plan_.cases} == {
        "test_first_retry_uses_base_delay",
        "test_negative_attempt_rejected",
    }

    source = render_tests(plan_)
    compile(source, "<generated>", "exec")
    assert "from myapp.retry import backoff" in source
    assert "def test_first_retry_uses_base_delay(base):" in source
    assert "result = backoff(1, base=base)" in source
    assert "assert result == 0.5" in source
    assert "with pytest.raises(ValueError):" in source

    # The whole plan path is deterministic: the model is never consulted.
    assert sent_requests(mock_http) == []


@pytest.mark.asyncio
async def test_toml_request_mode_dispatches_to_planner(mock_http):
    """Request-mode: no cases -> the planner LLM fills a TestPlan."""
    _mock_planner_response(mock_http)
    spec = parse(REQUEST_TOML)
    assert spec.mode == "request"
    # a request-mode spec has no hand-written cases
    assert spec.to_prompt()

    result = await plan_from_spec(spec, model="gpt-4o")
    assert isinstance(result, TestPlan)
    assert result.api.module == "myapp.retry"
    assert sent_requests(mock_http)  # the planner was actually called


def test_toml_watertight_uncovered_rule_flagged():
    """THE guarantee: a declared rule left uncovered is flagged by validate_plan."""
    spec = parse(PLAN_TOML)
    plan_ = spec.to_test_plan()
    # drop the case that covers rule_1 -> rule_1 is now uncovered
    plan_.cases = [c for c in plan_.cases if "rule_1" not in c.covers]

    problems = validate_plan(
        plan_,
        scenario_names=spec.scenario_names(),
        message_ids=spec.message_ids(),
        required_ids=spec.required_ids(),
    )
    assert any("rule_1" in p for p in problems)


def test_validate_plan_flags_all_gap_kinds():
    """validate_plan flags uncovered scenarios, unknown ids, and uncovered messages."""
    spec = parse(SEQUENCE_TOML)
    plan_ = spec.to_test_plan()
    # claim an id that doesn't exist and a scenario that's declared but uncovered
    plan_.cases[0].covers = ["M1", "M99"]
    problems = validate_plan(
        plan_,
        scenario_names=spec.scenario_names() | {"Listing users elsewhere"},
        message_ids=spec.message_ids(),
        required_ids=spec.required_ids(),
    )
    assert any("Listing users elsewhere" in p for p in problems)  # scenario not covered
    assert any("'M99'" in p for p in problems)  # unknown id
    assert any("M2" in p for p in problems)  # uncovered message
    # a sound plan for the plan-mode spec has no problems
    good = parse(PLAN_TOML)
    assert (
        validate_plan(
            good.to_test_plan(),
            scenario_names=good.scenario_names(),
            message_ids=good.message_ids(),
            required_ids=good.required_ids(),
        )
        == []
    )


@pytest.mark.asyncio
async def test_build_writes_suite_plan_and_brief(mock_http, tmp_path):
    """Request-mode build: the planner fills a plan; suite/plan/brief are written."""
    _mock_planner_response(mock_http)
    spec_file = tmp_path / "retry.toml"
    spec_file.write_text(REQUEST_TOML, encoding="utf-8")
    out = tmp_path / "generated"

    # sample_plan does not cover rule_1/edge_1, so run lenient to still write files.
    results = await build(spec_file, out, model="gpt-4o", strict=False)

    assert len(results) == 1
    result = results[0]
    assert result.test_path == out / "test_retry_policy.py"
    compile(result.test_path.read_text(encoding="utf-8"), str(result.test_path), "exec")
    saved = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assert TestPlan.model_validate(saved) == result.plan
    brief = result.brief_path.read_text(encoding="utf-8")
    assert "def backoff(attempt: int, *, base: float = 0.5) -> float" in brief
    assert "test_negative_attempt_rejected" in brief
    assert "read-only" in brief
    conftest = (out / "conftest.py").read_text(encoding="utf-8")
    assert "bdd_pending" in conftest


@pytest.mark.asyncio
async def test_toml_build_strict_rejects_uncovered_rule(mock_http, tmp_path):
    """build(strict=True) raises, writes plan.json, and withholds the test file."""
    # A request-mode TOML whose declared rule the mocked plan won't cover
    # (sample_plan() covers no rule/edge ids).
    _mock_planner_response(mock_http)
    spec_file = tmp_path / "retry.spec.toml"
    spec_file.write_text(REQUEST_TOML, encoding="utf-8")
    out = tmp_path / "generated"

    with pytest.raises(PlanValidationError) as excinfo:
        await build(spec_file, out, model="gpt-4o")

    assert "rule_1" in str(excinfo.value) or "not covered" in str(excinfo.value)
    assert (out / "retry_policy.plan.json").exists()
    assert not (out / "test_retry_policy.py").exists()


@pytest.mark.asyncio
async def test_toml_build_lenient_records_warnings_in_brief(mock_http, tmp_path):
    """Lenient build records validation warnings and the collaboration contract."""
    _mock_planner_response(mock_http)  # sample_plan claims no message coverage
    # Request-mode spec with a sequence but no [[cases]] -> planner (mocked) plan
    # covers none of the M-ids, so the build warns and marks them uncovered.
    request_sequence = (
        "[meta]\n"
        'feature_name = "User listing"\n'
        "\n"
        "[api]\n"
        'module = "myapp.users"\n'
        'signatures = ["def list_users(client, db) -> list"]\n'
        "\n"
        "[[sequences]]\n"
        'mermaid = """\n' + MERMAID + '"""\n'
    )
    spec_file = tmp_path / "users.toml"
    spec_file.write_text(request_sequence, encoding="utf-8")
    out = tmp_path / "generated"

    results = await build(spec_file, out, model="gpt-4o", strict=False)

    assert len(results[0].feature.sequences) == 1
    brief = results[0].brief_path.read_text(encoding="utf-8")
    assert "Plan validation warnings" in brief
    assert "```mermaid" in brief  # collaboration contract section
    assert "Message traceability" in brief
    assert "**uncovered**" in brief


def test_toml_absorbs_mermaid_sequences():
    """[[sequences]] mermaid is parsed by the existing sequence.py into M-ids."""
    spec = parse(SEQUENCE_TOML)
    assert len(spec.sequences) == 1
    assert [m.id for m in spec.sequences[0].messages] == ["M1", "M2", "M3", "M4", "M5"]
    assert spec.message_ids() == {"M1", "M2", "M3", "M4", "M5"}

    plan_ = spec.to_test_plan()  # covers only M1
    problems = validate_plan(
        plan_,
        scenario_names=spec.scenario_names(),
        message_ids=spec.message_ids(),
        required_ids=spec.required_ids(),
    )
    # M2..M5 are declared but uncovered -> flagged
    assert any("M2" in p for p in problems)


def test_toml_mode_auto_detected_from_cases():
    """Presence of [[cases]] means plan mode; absence means request mode."""
    assert parse(PLAN_TOML).mode == "plan"
    assert parse(REQUEST_TOML).mode == "request"


def test_toml_explicit_meta_mode_overrides_autodetect():
    """[meta].mode wins even when [[cases]] would auto-detect otherwise."""
    # PLAN_TOML carries [[cases]] (would auto-detect 'plan'); force 'request'.
    forced = PLAN_TOML.replace("[meta]\n", '[meta]\nmode = "request"\n', 1)
    assert parse(forced).mode == "request"


def test_toml_parse_file_round_trip(tmp_path):
    """parse_file reads the same spec off disk."""
    path = tmp_path / "retry.spec.toml"
    path.write_text(PLAN_TOML, encoding="utf-8")
    spec = parse_file(path)
    assert spec.mode == "plan"
    assert spec.slug == "retry_policy"


def test_toml_module_is_exported():
    """The toml_spec module is exported for direct access."""
    assert toml_spec.parse is parse
    assert toml_spec.parse_file is parse_file


# --- safety gate (AST screen over executable strings) --------------------------
def _plan_with(**case_overrides) -> TestPlan:
    """A minimal single-case plan, with the one case's fields overridable."""
    base = {"scenario": "s", "test_name": "test_x", "when": "f()", "then": ["result"]}
    base.update(case_overrides)
    return TestPlan(
        feature="F",
        api=ApiSurface(module="m", signatures=["def f() -> int"]),
        cases=[TestCase(**base)],
    )


def test_scan_plan_passes_benign_plan():
    """The clean plan-mode sample raises no safety problems."""
    assert scan_plan(parse(PLAN_TOML).to_test_plan()) == []


def test_scan_plan_flags_import_in_when():
    problems = scan_plan(_plan_with(when="__import__('os').system('rm -rf /')"))
    assert any("__import__" in p for p in problems)


def test_scan_plan_flags_import_statement_in_given():
    problems = scan_plan(_plan_with(given=["import os"], when="None", then=[]))
    assert any("import statement" in p for p in problems)


def test_scan_plan_flags_dunder_reflection():
    """The classic sandbox-escape via ``().__class__.__bases__`` is caught."""
    problems = scan_plan(_plan_with(when="().__class__.__bases__[0].__subclasses__()"))
    assert any("dunder" in p for p in problems)


def test_scan_plan_flags_banned_name_and_call():
    assert any("subprocess" in p for p in scan_plan(_plan_with(when="subprocess.run(['ls'])")))
    assert any("open" in p for p in scan_plan(_plan_with(when="open('/etc/passwd')")))


def test_scan_plan_flags_non_python():
    """A syntactically invalid string fails here, not at pytest import time."""
    problems = scan_plan(_plan_with(when="this is not python !!!"))
    assert any("not valid Python" in p for p in problems)


def test_scan_plan_flags_fixture_body():
    plan_ = TestPlan(
        feature="F",
        api=ApiSurface(module="m", signatures=["def f() -> int"]),
        fixtures=[FixtureDef(name="evil", body=["import socket", "return 1"])],
        cases=[TestCase(scenario="s", test_name="test_x", when="f()", then=["result"])],
    )
    assert any("fixture 'evil'" in p for p in scan_plan(plan_))


def test_scan_plan_allows_canonical_error_path():
    """The multi-line ``with pytest.raises(...):`` given form parses cleanly."""
    plan_ = _plan_with(
        given=["with pytest.raises(ValueError):", "    f(-1)"], when="None", then=[]
    )
    assert scan_plan(plan_) == []


@pytest.mark.asyncio
async def test_build_safety_gate_blocks_and_writes_nothing_runnable(mock_http, tmp_path):
    """A hostile plan-mode spec raises PlanSafetyError; plan.json kept, no test file.

    Uses plan mode so no planner call is needed — proving the gate guards the
    LLM-free path too (a malicious hand-authored / PR-supplied .toml).
    """
    hostile = (
        "[meta]\n"
        'feature_name = "Evil"\n'
        "[api]\n"
        'module = "m"\n'
        'signatures = ["def f() -> int"]\n'
        "[[cases]]\n"
        'test_name = "test_pwn"\n'
        'scenario = "pwn"\n'
        "when = \"__import__('os').system('id')\"\n"
        'then = ["result"]\n'
    )
    spec_file = tmp_path / "evil.toml"
    spec_file.write_text(hostile, encoding="utf-8")
    out = tmp_path / "generated"

    with pytest.raises(PlanSafetyError) as excinfo:
        await build(spec_file, out)

    assert "__import__" in str(excinfo.value)
    assert (out / "evil.plan.json").exists()  # kept for inspection
    assert not (out / "test_evil.py").exists()  # nothing runnable written
    assert sent_requests(mock_http) == []  # plan mode: no model call
