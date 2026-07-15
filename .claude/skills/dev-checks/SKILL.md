---
name: dev-checks
description: Run the pyllym quality gate — ruff lint, ruff format check, mypy type-check, and pytest. Use when the user asks to "run checks", "lint", "typecheck", "run the tests", "is it green?", verify a change is ready to commit, or before opening a PR.
---

# dev-checks

Run the full local quality gate for the **pyllym** package and report results.
Prefer the project virtualenv (`.venv/bin/*`) so the pinned tool versions are used.

## The gate

Run these in order. Stop and report on the first failure unless the user asked
for the whole sweep regardless.

```bash
# 1. Lint
.venv/bin/ruff check src tests

# 2. Format (check only — do not auto-format unless asked)
.venv/bin/ruff format --check src tests

# 3. Types
.venv/bin/mypy src

# 4. Tests
.venv/bin/pytest
```

If `.venv/bin/*` is missing, fall back to bare `ruff` / `mypy` / `pytest`, and
tell the user the venv wasn't found (they may need `pip install -e ".[dev,db]"`).

## Conventions to respect

- `ruff` config lives in `pyproject.toml` (`line-length = 100`, `target-version = py313`).
- Do **not** run `ruff format` (mutating) as part of a check — only `--check`.
  Offer to apply formatting as a separate, explicit step.
- `mypy` runs against `src` only, not `tests`.
- Generated BDD suites carry the `bdd_pending` marker and may be expected to
  fail (red) during a build — mention this if pytest reports `bdd_pending`
  failures so they aren't mistaken for regressions.

## Reporting

Summarize pass/fail per stage. On failure, show the offending output and point
at the file:line. Don't claim green unless every stage actually passed.