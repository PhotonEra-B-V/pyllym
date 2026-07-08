# pyllm

**One delightful, async-first Python framework for every major AI provider.**
Build AI agents, chatbots, RAG apps, and multimodal workflows in clean,
expressive Python.

Works with OpenAI, Anthropic, Google Gemini, AWS Bedrock, DeepSeek, Mistral,
Ollama (local), OpenRouter, Perplexity, Vertex AI, xAI, GPUStack, Azure, and any
OpenAI-compatible API — behind one consistent interface.

```python
import pyllm

pyllm.configure(lambda c: setattr(c, "openai_api_key", "sk-..."))

chat = pyllm.create_chat(model="gpt-5.4")
message = await chat.ask("What's the best way to learn Python?")
print(message.content)
```

> Every provider ships its own bloated client with different APIs, response
> formats, and conventions. pyllm gives you **one** interface for all of them —
> the same code whether you're using GPT, Claude, Gemini, or your local Ollama.

---

## Features

- 💬 **Chat** with text, images, audio, PDFs, and documents
- 🌊 **Streaming** responses via async generators
- 🔧 **Tools** (function calling) — sync *or* async, run sequentially or concurrently
- 📋 **Structured output** with JSON Schema / Pydantic models
- 🧠 **Thinking / reasoning** controls (effort & budget)
- 🔢 **Embeddings**, 🎨 **image generation**, 🗣️ **speech**, 📝 **transcription**, 🛡️ **moderation**
- 📚 **Citations** normalized across providers
- 🗄️ **Optional SQLAlchemy persistence** (async) for chats, messages, and tool calls
- ⚙️ **Optional Celery integration** — run chat, embeddings & more as background tasks
- 📇 A bundled **model registry** (`models.json`) with pricing & capabilities
- 🐍 Modern, fully type-annotated **Python 3.13+**, async-first on `aiohttp`

## Installation

```bash
pip install pyllm                # core
pip install "pyllm[db]"          # + SQLAlchemy persistence
pip install "pyllm[celery]"      # + Celery background tasks
pip install "pyllm[mime]"        # + content-based MIME sniffing
pip install "pyllm[dev]"         # + test/lint tooling
```

> This repository is a source tree; install it editable with
> `pip install -e ".[dev,db]"`.

## Configuration

```python
import pyllm

pyllm.configure(lambda c: (
    setattr(c, "openai_api_key", "sk-..."),
    setattr(c, "anthropic_api_key", "sk-ant-..."),
    setattr(c, "gemini_api_key", "..."),
))
```

Any provider option that isn't set in code falls back to the environment
variable of the same name in uppercase — `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_API_BASE`, and so on. Values
set via `pyllm.configure` always take precedence, so exporting keys in your
shell is enough to get started (including for CLI entry points like
`python -m pyllm.bdd`).

Per-call overrides use a `Context`:

```python
ctx = pyllm.context(lambda c: setattr(c, "request_timeout", 30))
chat = ctx.create_chat(model="claude-sonnet-4-6")
```

## Show me the code

### Ask anything

```python
chat = pyllm.create_chat(model="gpt-5.4")
await chat.ask("What's the capital of France?")
```

### Analyze files

```python
await chat.ask("What's in this image?", with_="diagram.png")
await chat.ask("Summarize this paper", with_="paper.pdf")
await chat.ask("Describe this meeting", with_="meeting.wav")
```

### Stream responses

```python
async for chunk in chat.stream("Write a haiku about Python"):
    print(chunk.content or "", end="", flush=True)
```

### Tools (function calling)

```python
from pyllm import Tool

class Weather(Tool):
    description = "Look up the weather for a city"

    async def execute(self, *, city: str) -> str:
        return f"It's sunny in {city}"

chat = pyllm.create_chat(model="gpt-5.4").with_tool(Weather)
answer = await chat.ask("What's the weather in Paris?")
# pyllm runs the agentic loop: model -> tool -> model -> final answer
```

Tools may be sync or async. Run them concurrently with
`chat.with_tools(A, B, concurrency=True)`.

### Structured output

```python
from pydantic import BaseModel

class Recipe(BaseModel):
    title: str
    ingredients: list[str]

chat = pyllm.create_chat(model="gpt-5.4").with_schema(Recipe)
msg = await chat.ask("Give me a recipe for pancakes")
msg.content  # -> a dict matching the schema
```

### Thinking / reasoning

```python
chat = pyllm.create_chat(model="claude-sonnet-4-6").with_thinking(effort="high")
```

### Embeddings, images, speech, transcription, moderation

```python
emb   = await pyllm.embed("Hello world")                       # Embedding(vectors=...)
image = await pyllm.paint("a red panda coding", model="gpt-image-1.5")
image.save("panda.png")
speech = await pyllm.speak("Hello there", model="gpt-4o-mini-tts")
speech.save("hello.mp3")
text  = await pyllm.transcribe("meeting.wav")                  # Transcription(text=...)
mod   = await pyllm.moderate("some text")                      # Moderation(...)
```

### Image & video generation via fal.ai

`pyllm.paint` reaches fal-hosted image models (FLUX.2, HunyuanImage, Qwen-Image);
`pyllm.animate` is a video-generation capability (LTX, Wan, HunyuanVideo) that
submits to fal's queue and polls until the render completes.

```python
pyllm.configure(lambda c: setattr(c, "fal_api_key", "..."))

image = await pyllm.paint("a red panda", provider="fal", model="fal-ai/flux/dev")
image.save("panda.png")

video = await pyllm.animate("a timelapse sunrise over mountains",
                            model="fal-ai/ltx-video-13b-distilled")  # provider="fal" default
video.save("sunrise.mp4")
```

### Agents

```python
from pyllm import Agent

class Researcher(Agent):
    chat_model = "claude-sonnet-4-6"
    instructions = "You are a meticulous research assistant."
    tools = [Weather]
    temperature = 0.2

chat = Researcher.create_chat()
await chat.ask("What should I pack for Paris this weekend?")
```

## Persistence (SQLAlchemy)

Optionally persist chats, messages, and tool calls with an async SQLAlchemy
model factory:

```python
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from pyllm.persistence import create_models

class Base(DeclarativeBase): ...

Chat, Message, ToolCall = create_models(Base)   # bound to your Base

# ... create tables, open an AsyncSession ...
record = Chat(model_id="gpt-5.4", provider="openai")
session.add(record); await session.commit()

chat = record.to_chat(session)        # a pyllm.Chat backed by the DB
await chat.ask("Hello!")             # user + assistant rows are persisted
await session.commit()
```

## Background tasks (Celery)

Optionally run pyllm operations on Celery workers. `create_tasks` registers
ready-made tasks (`ask`, `embed`, `paint`, `speak`, `transcribe`, `moderate`)
on your app — broker-friendly JSON in, plain dicts out:

```python
from celery import Celery
from pyllm.celery import create_tasks

app = Celery("worker", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")
tasks = create_tasks(app)                 # or create_tasks(app, queue="llm")

result = tasks.ask.delay("What's the capital of France?", model="gpt-5.4")
result.get()                              # -> the assistant Message as a dict

tasks.embed.delay("Hello world")
tasks.paint.delay("a red panda coding", save_path="/tmp/panda.png")
```

Workers are synchronous; each task drives the underlying coroutine with
`pyllm.celery.run_async`, which also closes pyllm's HTTP pools before the
event loop shuts down. Use `run_async` directly in your own tasks for
richer features (tools, agents, callbacks) that don't serialize through a
broker:

```python
@app.task
def research(question: str) -> dict:
    chat = pyllm.create_chat(model="gpt-5.4").with_tool(Weather)
    return run_async(chat.ask(question)).to_dict()
```

## BDD test builder (pre-TDD scaffolding)

`pyllm.bdd` turns a **TOML spec** into a *red* pytest suite plus an
implementation brief — a building instruction for a coding agent (or a human)
to implement against. pyllm dogfoods itself: in request mode the planning step
is a structured-output call filling a `TestPlan` schema, and the actual test
code is *always* rendered from deterministic templates, never written freeform
by the model.

```python
from pyllm.bdd import build

results = await build("specs/", "tests/generated", model="gpt-5.4")
```

On Mac it might make sense to have an alias for Python `alias python="python3"`.

or from the shell:

```bash
python -m pyllm.bdd specs/retry.toml --out tests/generated --model gpt-5.4
```

For each spec this writes `test_<slug>.py` (the failing suite),
`<slug>.plan.json` (the reviewable plan, including the API surface committed
to), and `BRIEF_<slug>.md` (target signatures, test manifest, and the
definition of done: the suite passes with zero edits to the tests).

### Two modes

A spec declares its API surface, coverable `[[rules]]` / `[[edge_cases]]`, and
optionally hand-written `[[cases]]`. Mode is auto-detected from the presence of
`[[cases]]` (override with `[meta].mode` or `--toml-mode`):

- **plan mode** — the spec carries `[[cases]]`; the `TestPlan` is built
  deterministically with **no LLM in the loop**.
- **request mode** — a loose spec (rules/edge cases, no cases); the planner
  model fills the `TestPlan`, told to cover every id.

```toml
[meta]
feature_name = "Retry policy"

[api]
module = "myapp.retry"
imports = ["from myapp.retry import backoff"]
signatures = ["def backoff(attempt: int, *, base: float = 0.5) -> float"]

[[rules]]
id = "rule_1"
text = "The first retry uses the base delay."

[[cases]]
test_name = "test_first_retry_uses_base_delay"
scenario = "First retry uses the base delay"
when = "backoff(1, base=0.5)"
then = ["result == 0.5"]
covers = ["rule_1"]
```

### Sequence diagrams (collaboration contracts)

A spec can absorb Mermaid **sequence diagrams** via `[[sequences]]` tables as
binding collaboration contracts:

```toml
[[sequences]]
mermaid = """
sequenceDiagram
    Client->>+Server: GET /users
    Server->>+Database: Query Users
    Database-->>-Server: Return Data
    Server-->>-Client: 200 OK
"""
```

Every diagram message gets a stable id (`M1`, `M2`, ...) that a test case must
claim via `covers`. Coverage is then verified **mechanically**, not by the
model: a plan that drops a scenario, ignores a diagram message, or claims a
nonexistent id fails the build (`strict=False` downgrades this to warnings in
the brief). The brief renders the diagram plus a message-to-tests traceability
table.

### Safety

A case's `given` / `when` / `then` and fixture bodies are rendered verbatim
into a pytest module that then gets run — every such string is executable
Python (true in plan mode too). Before rendering, an AST safety gate screens
every executable string (both modes, unconditionally) and fails the build on
imports, dangerous builtins/names (`os`, `subprocess`, `eval`, `__import__`,
`open`, ...), dunder reflection, or non-Python — nothing runnable is written.

> **This gate is defense-in-depth, not a sandbox.** A determined attacker can
> evade any static denylist. Never run a freshly generated suite where secrets
> or network are reachable before a human reviews the diff; sandbox generation
> and execution of untrusted specs; and treat specs from untrusted contributors
> as needing review *before* generation (the spec text reaches the planner
> prompt) and *before* execution.

Review the plan and tests **before** starting implementation — from that
point on the suite is the specification, and generated tests are read-only.
No extra dependencies are required.

## Supported providers

| Provider | Status |
|----------|--------|
| OpenAI (Chat Completions) | ✅ |
| Anthropic (Messages) | ✅ |
| Google Gemini (generateContent) | ✅ |
| DeepSeek, Mistral, xAI, Perplexity, OpenRouter | ✅ (OpenAI-compatible) |
| NVIDIA NIM, Cerebras, Hugging Face, Databricks | ✅ (OpenAI-compatible) |
| Qwen (DashScope), Zhipu GLM, Moonshot Kimi, Doubao, ERNIE, MiniMax | ✅ (OpenAI-compatible) |
| Ollama, GPUStack (local) | ✅ |
| fal.ai (image via `paint`, video via `animate`) | ✅ FLUX.2, HunyuanImage, Qwen-Image, LTX, Wan, HunyuanVideo |
| Azure OpenAI | ✅ (v1-compatible endpoint) |
| AWS Bedrock (Converse) | ✅ non-streaming (SigV4 signed); streaming WIP |
| Vertex AI (Gemini) | ✅ with a supplied access token; OAuth minting WIP |

## Design

- **Async/await everywhere** for I/O. `await chat.ask(...)` and
  `async for chunk in chat.stream(...)`.
- **Provider → Protocol → Connection** architecture: a provider knows *where*
  and *who*; a protocol knows the wire format; the connection is `aiohttp`.
- **`aiohttp`** for HTTP, with connection pools shared across chats per event
  loop; call `await pyllm.aclose()` once at application shutdown.
- **SQLAlchemy** for optional persistence.
- Fully type-annotated, `ruff`/`mypy`-friendly, Python 3.13+.
- **Planned:** OpenAI's *Responses* protocol, Bedrock event-stream decoding,
  and Vertex AI OAuth token minting.

## Development

```bash
pip install -e ".[dev,db]"
ruff check src tests && ruff format --check src tests
pytest
```

## Something is missing

If you feel like the library is missing something like a new LLM or existing one, that you think shold be there then can you open and issue for that, and I'll see what I can do about that.

## License

MIT — see [LICENSE](LICENSE).

Inspired by RubyLLM, it is not a direct port and diverges on quite a few ways. If you are using Ruby then checkout out, it is great.
https://rubyllm.com
