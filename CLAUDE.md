# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**pyllym** is one delightful, async-first framework for every major AI provider
(OpenAI, Anthropic, Google Gemini, AWS Bedrock, DeepSeek, Mistral, Ollama,
OpenRouter, Perplexity, VertexAI, xAI, GPUStack, Azure, Qwen, Zhipu GLM,
Moonshot, Doubao, ERNIE, MiniMax, NVIDIA, Cerebras, Hugging Face, Databricks,
fal.ai, and any OpenAI-compatible API).

It provides chat (text, images, audio, PDFs), streaming, tools (function
calling), structured output, embeddings, image generation, video generation,
speech, transcription, moderation, and an optional SQLAlchemy persistence
layer.

The Python package lives under `src/pyllym/`. A read-only reference source may
exist under `llm_ignore/` (git-ignored — never ship or import from it).

| Item | Value |
|------|-------|
| Package | `pyllym` (import root) |
| Language | Python 3.13+ |
| HTTP | `aiohttp` (async) |
| Validation / value objects | `dataclasses` + `pydantic` (where schema-shaped) |
| Persistence (optional) | SQLAlchemy 2.x (async) |
| Tests | `pytest` + `pytest-asyncio` |
| Lint / format / types | `ruff`, `mypy` |

## Layout

```
src/pyllym/                 # the package
  __init__.py              # façade: chat(), embed(), paint(), configure(), config, models
  configuration.py
  errors.py
  chat.py  message.py  content.py  tool.py  tool_call.py
  connection.py  protocol.py  streaming.py  stream_accumulator.py
  models.py  model/...      # registry + Model.Info value objects + models.json
  protocols/                # wire formats (chat_completions, anthropic, gemini, ...)
  providers/                # where/who (openai, anthropic, gemini, ...)
  persistence/              # optional async SQLAlchemy model factory
  celery/                   # optional Celery task factory (celery extra)
  bdd/                      # BDD→TDD builder: TOML spec → red pytest suite + brief
tests/
llm_ignore/                 # read-only reference source, if present (git-ignored)
```

## Architecture

- **Provider (where/who) → Protocol (wire format) → Connection (aiohttp)** is the
  central architecture; preserve it when adding providers or protocols.
- A protocol composes per-concern methods (chat, tools, streaming, media,
  embeddings, ...). OpenAI-compatible providers reuse `ChatCompletions`; others
  (Anthropic, Gemini, Converse, fal) implement their own protocol.
- I/O is `async def`; pure transforms are sync. Streaming yields chunks via an
  async generator consumed by `StreamAccumulator`.
- Value objects (`Tokens`, `Thinking`, `Citation`) are frozen dataclasses with
  `to_dict()` and `build()` classmethods. `ToolCall` is a mutable dataclass
  (streaming accumulates argument fragments in place) and `Cost` is a derived
  calculator over `Tokens` + pricing, not a frozen record.
- Persistence is a Python-native async SQLAlchemy model factory under
  `persistence/`.
- Celery integration (`celery` extra) is a task factory under `celery/`:
  `create_tasks(app)` registers sync tasks that drive coroutines via
  `run_async`, which closes the per-loop HTTP pools before `asyncio.run`
  returns.
- The BDD builder under `bdd/` (no extra deps) takes a single **TOML spec**
  (`toml_spec.py`) as its front-end — the spec carries the API surface,
  coverable `[[rules]]`/`[[edge_cases]]`, optional mermaid `[[sequences]]`
  blocks (parsed by the surviving `sequence.py`, never reimplemented), and, in
  plan mode, hand-written `[[cases]]`. Two modes, auto-detected from the
  presence of `[[cases]]` unless `[meta].mode` (or `--toml-mode`) says
  otherwise: **plan mode** builds a `TestPlan` deterministically via
  `TomlSpec.to_test_plan()` with NO LLM; **request mode** (loose spec, no
  cases) sends `spec.to_prompt()` to the planner (`planner.plan_from_spec`,
  a structured-output call filling the `TestPlan` pydantic schema). The LLM
  only ever fills the schema; test code is always rendered from templates
  (`renderer.py`, `brief.py`). Every rule/edge id and every sequence message
  id (`M1`, ...) must be claimed by some case's `covers`; `checks.validate_plan`
  (decoupled from any spec type — it takes `scenario_names`, `message_ids`,
  `required_ids` sets) fails the build mechanically on uncovered
  scenarios/messages/ids or unknown claimed ids. This is the watertight
  guarantee. CLI: `python -m pyllym.bdd specs/ --out tests/generated`.
  Generated tests carry the `bdd_pending` marker and are treated as the
  read-only specification during implementation.
- **Security model.** A spec's `given`/`when`/`then` and fixture bodies are
  rendered verbatim into a pytest module that then gets run — every such string
  is executable Python (true in plan mode too, so a hostile `.toml` from a PR is
  arbitrary code execution). Before rendering, `safety.scan_plan` AST-screens
  every executable string (both modes, unconditionally — never downgraded to a
  warning) and raises `PlanSafetyError` on imports, dangerous builtins/names
  (`os`, `subprocess`, `eval`, `__import__`, `open`, ...), dunder reflection, or
  non-Python; nothing runnable is written. **This gate is defense-in-depth, not
  a sandbox** — a determined attacker can evade a static denylist. The real
  boundary is procedural: (1) never run a freshly generated suite where secrets
  or network are reachable before a human reviews the diff; (2) sandbox
  generation + execution of untrusted specs; (3) treat `.toml` specs from
  untrusted contributors as needing review *before* generation (the spec text
  reaches the planner prompt via `to_prompt`, a prompt-injection vector) and
  *before* execution.

Adding an OpenAI-compatible provider is a ~30-line file under `providers/`
(set `api_base`, `headers`, config options, and `assumes_models_exist`), then
register it in `pyllym/__init__.py`.

## Commands

```bash
# install (editable, with dev + persistence extras)
pip install -e ".[dev,db]"

# lint / format / types
ruff check src tests && ruff format --check src tests && mypy src

# tests
pytest
```

## Conventions

- `from __future__ import annotations` at the top of every module.
- PEP 604 unions, builtin generics (`list[str]`, `dict[str, Any]`).
- Public façade functions live in `pyllym/__init__.py` and delegate to classes.
- Global config is a module singleton: `pyllym.config` / `pyllym.configure()`.
  Unset provider options fall back to the uppercase env var of the same name
  (`openai_api_key` → `OPENAI_API_KEY`); values set in code win.
- The model registry data file `models.json` is generated data; prefer
  `pyllym.models.refresh()` over hand-editing it.
