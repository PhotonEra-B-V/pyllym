---
name: pyllym-structured-prompting
description: >-
  Build structured prompting with the async-first pyllym library (Python 3.14):
  structured output via Pydantic schemas, tools/function-calling, system
  instructions, reasoning/thinking controls, streaming, and multimodal input —
  plus how to register new models and add new providers. Use when writing or
  reviewing code that calls pyllym (pyllym.create_chat / embed / paint /
  animate), when
  designing a JSON-schema/Pydantic response contract, when a model id isn't
  found, or when wiring up a new AI provider.
---

# pyllym: structured prompting & new models

`pyllym` is one async-first Python framework for every major AI provider. This
skill covers the two things people most often need: **structured prompting**
(reliable, typed outputs and tool use) and **extending the model/provider set**.

Target runtime: **Python 3.14** (3.13+). Everything I/O is `async`.

## Setup

```python
import pyllym

pyllym.configure(lambda c: setattr(c, "openai_api_key", "sk-..."))
# multiple keys:
pyllym.configure(lambda c: (
    setattr(c, "anthropic_api_key", "sk-ant-..."),
    setattr(c, "gemini_api_key", "..."),
))
```

Per-call overrides use a `Context` (isolated config copy):

```python
ctx = pyllym.context(lambda c: setattr(c, "request_timeout", 30))
chat = ctx.create_chat(model="claude-sonnet-4-6")
```

## Structured output (typed responses)

Prefer a **Pydantic model** — pass the *class* (not an instance) to
`with_schema`. The provider is asked for JSON matching the schema, and
`message.content` comes back as a parsed `dict`; validate it into your model.

```python
from pydantic import BaseModel
import pyllym

class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    minutes: int

chat = pyllym.create_chat(model="gpt-5.4").with_schema(Recipe)
msg = await chat.ask("Give me a quick pancake recipe")
recipe = Recipe.model_validate(msg.content)   # msg.content is a dict
```

- `with_schema` also accepts a raw JSON-Schema `dict` or any object exposing
  `to_json_schema()`.
- Schemas are sent as strict `json_schema` where the provider supports it. If a
  model can't guarantee JSON, `msg.content` may stay a string — guard with
  `isinstance(msg.content, dict)`.
- Keep schemas flat and name fields descriptively; add field descriptions via
  Pydantic `Field(description=...)` — they reach the model.

## Tools (function calling)

Subclass `Tool`; the parameter schema is **inferred from `execute`'s
keyword-only signature** (or declare it explicitly). `execute` may be sync or
async. `chat.ask` runs the full agentic loop (model → tool → model → answer).

```python
from pyllym import Tool

class Weather(Tool):
    description = "Look up the current weather for a city"

    async def execute(self, *, city: str, units: str = "celsius") -> str:
        return f"18°{units[0].upper()} and clear in {city}"

chat = pyllym.create_chat(model="gpt-5.4").with_tools(Weather)
answer = await chat.ask("What's the weather in Oslo?")
```

- Force/limit tool use: `with_tools(Weather, choice="required", calls="one")`
  (`choice` ∈ `auto|none|required|<tool-name>`; `calls` ∈ `many|one`).
- Run independent tool calls concurrently: `with_tools(A, B, concurrency=True)`
  (uses `asyncio.gather`).
- Explicit params instead of signature inference — set a raw JSON-schema class
  attribute, or declare `Parameter`s via the `param` classmethod:
  ```python
  class Search(Tool):
      description = "Search the KB"
      _params_schema = {                     # raw JSON schema wins
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"],
      }
      def execute(self, *, query: str): ...

  class Lookup(Tool):
      description = "Look up a record"
  Lookup.param("record_id", type="string", description="the id", required=True)
  ```
- Return `pyllym.SearchResults(title=..., url=..., text=...)` from a tool to give
  citation-capable providers citable results.

## System instructions, reasoning, params

```python
chat = (
    pyllym.create_chat(model="claude-sonnet-4-6")
    .with_instructions("You are a terse assistant. Answer in one sentence.")
    .with_thinking(effort="high")          # or budget=2048
    .with_temperature(0.2)
    .with_citations()
    .with_params(top_p=0.9)                 # provider-passthrough
)
```

Compose a reusable configuration as an `Agent`:

```python
from pyllym import Agent

class Analyst(Agent):
    chat_model = "gpt-5.4"
    instructions = "You are a rigorous financial analyst."
    tools = [Weather]
    temperature = 0.1

chat = Analyst.create_chat()
```

## Streaming & multimodal

```python
async for chunk in chat.stream("Write a haiku about async"):
    print(chunk.content or "", end="", flush=True)

await chat.ask("What's in this image?", with_="diagram.png")   # url, path, bytes, or file
```

## Other capabilities

```python
emb   = await pyllym.embed("hello")                                  # Embedding(vectors=...)
img   = await pyllym.paint("a red panda", provider="fal", model="fal-ai/flux/dev")
vid   = await pyllym.animate("a timelapse sunrise", model="fal-ai/ltx-video-13b-distilled")
audio = await pyllym.speak("hello", model="gpt-4o-mini-tts")
text  = await pyllym.transcribe("meeting.wav")
flags = await pyllym.moderate("some text")
```

## Including new models

The bundled registry (`models.json`) powers `pyllym.models.find(...)`. Three ways
to use a model that isn't in it yet:

1. **assume-exists** — pass the id + provider and skip the registry:
   ```python
   await pyllym.create_chat(model="deepseek-r2", provider="deepseek",
                    assume_model_exists=True).ask("hi")
   ```
   Providers whose catalogs live off-registry set `assumes_models_exist=True`,
   so `assume_model_exists` isn't even needed (e.g. `qwen`, `zhipu`, `moonshot`,
   `nvidia`, `fal`).
2. **refresh the registry** from providers + models.dev, then persist:
   ```python
   await pyllym.models.refresh()
   pyllym.models.save_to_json()
   ```
3. Inspect what's available: `pyllym.models.find("gpt-5.4")`,
   `pyllym.models.chat_models()`, `[m.id for m in pyllym.models.all() if m.provider == "gemini"]`.

Never hand-edit `models.json`; it's generated data.

## Adding a new provider

An OpenAI-compatible provider is ~30 lines. Create
`src/pyllym/providers/<slug>.py`:

```python
from __future__ import annotations

from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider


class Acme(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    @property
    def api_base(self) -> str:
        return self.config.acme_api_base or "https://api.acme.ai/v1"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.acme_api_key}"}

    @classmethod
    def assumes_models_exist(cls) -> bool:   # catalog not in models.json
        return True

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["acme_api_key", "acme_api_base"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["acme_api_key"]
```

Then register it in `src/pyllym/__init__.py` (add `("acme", "Acme")` to the
`registrations` list). Local providers override `is_local`; providers needing a
different wire format implement their own protocol under `protocols/` (see
`anthropic.py`, `gemini.py`, `converse.py`, `fal.py`) instead of reusing
`ChatCompletions`.

Add a test with a seeded factory response (`tests/factories.py`) and mount it
with `respx` — see `tests/test_provider_matrix.py`.

## Gotchas

- Everything I/O is `async`; `await` `ask`/`embed`/`paint`/`animate`, and
  `async for` over `stream`.
- The message-building kwarg is `with_=` (trailing underscore — `with` is a
  Python keyword).
- `msg.content` is a `dict` only when `with_schema` was set and the model
  returned valid JSON; otherwise it's a `str`.
- Predicates read as `message.is_tool_call()`, `is_tool_result()`,
  `is_stopped()`.
- Token usage: `msg.input_tokens` / `msg.output_tokens`; cost via `chat.cost` /
  `msg.cost()`.
```
