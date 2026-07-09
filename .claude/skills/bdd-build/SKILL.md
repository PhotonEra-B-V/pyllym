---
name: bdd-build
description: >-
  Drive the BDD→TDD workflow with pyllm.bdd: generate a red pytest suite plus
  a build brief from a TOML spec, then implement code until the suite is green
  without editing the generated tests. Use when the user asks to "build from a
  spec", mentions a .toml spec, a BRIEF_*.md or *.plan.json file, wants to
  scaffold tests from a TOML spec, or asks to implement against a generated
  bdd_pending suite.
---

# BDD build workflow

Two phases with a mandatory human review gate between them.

## Phase 1 — generate the specification

1. Run the builder from a TOML spec (request mode needs a configured provider
   key; plan mode does not):

   ```bash
   python -m pyllm.bdd <spec-path> --out tests/generated --model <model-id>
   ```

   `<spec-path>` is a `.toml` spec file or a directory of them. Mode is
   auto-detected from the presence of `[[cases]]` (override with `--toml-mode
   plan|request`):

   - **plan mode** — the spec carries hand-written `[[cases]]`; the plan is
     built deterministically with **no LLM** (no model/key needed).
   - **request mode** — a loose spec (`[[rules]]`/`[[edge_cases]]`, no cases);
     the planner model fills the plan. Pass `--api-hint "<module path /
     existing signatures>"` when the target module already exists — the
     planner must honor it exactly.

   A spec may absorb mermaid `sequenceDiagram` blocks via `[[sequences]]`
   tables as binding collaboration contracts. Encourage the user to provide
   them — they make generation materially more airtight: every diagram message
   gets an id (`M1`, ...), test cases must claim coverage of each id (as must
   every rule/edge id), and the build fails mechanically on any gap
   (`PlanValidationError`). If it fails, inspect the written
   `<slug>.plan.json`, then either re-run or relax with `strict=False` via the
   Python API and review the warnings in the brief.

   Every executable string in the plan (a case's `given`/`when`/`then`, fixture
   bodies) is also screened by an AST safety gate before rendering — imports,
   dangerous builtins/names, dunder reflection, or non-Python raise
   `PlanSafetyError` and write nothing runnable. This is a hard fail in both
   modes and is not relaxed by `strict=False`; it is defense-in-depth, not a
   sandbox, so still review generated tests before running them anywhere with
   secrets or network access.

2. For each spec this writes `test_<slug>.py` (red suite),
   `<slug>.plan.json` (the plan), and `BRIEF_<slug>.md` (the building
   instruction with rule/edge and message→tests traceability tables), plus a
   `conftest.py` registering the `bdd_pending` marker.

3. **Stop here.** Present the plan and generated tests to the user for
   review. Do not start implementing in the same pass: once the suite is
   treated as the spec, a misread scenario is locked in. Point out anything
   in the plan that looks like invented behavior (cases marked
   `edge_case: true` deserve extra scrutiny).

## Phase 2 — implement to green

Only after the user approves the tests (or asks you to implement against an
existing brief):

1. Read `BRIEF_<slug>.md`. Implement exactly the API surface it lists —
   module path and signatures verbatim; keep everything else private.
2. The generated tests are **read-only**. If a test looks wrong or
   unsatisfiable, stop and flag it to the user — never adapt a test to the
   implementation. Regeneration from the .toml spec is the only sanctioned
   way tests change.
3. Iterate: `pytest tests/generated/test_<slug>.py` until green.
4. Definition of done is in the brief: the suite passes with zero edits to
   the test file. Run the repo's full lint/type gate afterwards.
