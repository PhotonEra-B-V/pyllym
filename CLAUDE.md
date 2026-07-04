# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**pyllm** is one delightful, async-first framework for every major AI provider
(OpenAI, Anthropic, Google Gemini, AWS Bedrock, DeepSeek, Mistral, Ollama,
OpenRouter, Perplexity, VertexAI, xAI, GPUStack, Azure, Qwen, Zhipu GLM,
Moonshot, Doubao, ERNIE, MiniMax, NVIDIA, Cerebras, Hugging Face, Databricks,
fal.ai, and any OpenAI-compatible API).

It provides chat (text, images, audio, PDFs), streaming, tools (function
calling), structured output, embeddings, image generation, video generation,
speech, transcription, moderation, and an optional SQLAlchemy persistence
layer.

The Python package lives under `src/pyllm/`. A read-only reference source may
exist under `llm_ignore/` (git-ignored — never ship or import from it).

| Item | Value |
|------|-------|
| Package | `pyllm` (import root) |
| Language | Python 3.13+ |
| HTTP | `httpx` (async) |
| Validation / value objects | `dataclasses` + `pydantic` (where schema-shaped) |
| Persistence (optional) | SQLAlchemy 2.x (async) |
| Tests | `pytest` + `pytest-asyncio` |
| Lint / format / types | `ruff`, `mypy` |

## Layout

```
src/pyllm/                 # the package
  __init__.py              # façade: chat(), embed(), paint(), configure(), config, models
  configuration.py
  errors.py
  chat.py  message.py  content.py  tool.py  tool_call.py
  connection.py  protocol.py  streaming.py  stream_accumulator.py
  models.py  model/...      # registry + Model.Info value objects + models.json
  protocols/                # wire formats (chat_completions, anthropic, gemini, ...)
  providers/                # where/who (openai, anthropic, gemini, ...)
  persistence/              # optional async SQLAlchemy model factory
tests/
llm_ignore/                 # read-only reference source, if present (git-ignored)
```

## Architecture

- **Provider (where/who) → Protocol (wire format) → Connection (httpx)** is the
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

Adding an OpenAI-compatible provider is a ~30-line file under `providers/`
(set `api_base`, `headers`, config options, and `assumes_models_exist`), then
register it in `pyllm/__init__.py`.

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
- Public façade functions live in `pyllm/__init__.py` and delegate to classes.
- Global config is a module singleton: `pyllm.config` / `pyllm.configure()`.
- The model registry data file `models.json` is generated data; prefer
  `pyllm.models.refresh()` over hand-editing it.
