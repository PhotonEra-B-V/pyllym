# pyllym

<img width="720" height="405" alt="wdmbP" src="https://github.com/user-attachments/assets/184527c0-e550-40dd-b2d0-d9517b0ee542" />

<br>

**One delightful, async-first Python framework for every major AI provider.**
Build AI agents, chatbots, RAG apps, and multimodal workflows in clean,
expressive Python.

Works with OpenAI, Anthropic, Google Gemini, AWS Bedrock, DeepSeek, Mistral,
Ollama (local), vLLM (local), OpenRouter, Perplexity, Vertex AI, xAI, GPUStack,
Azure, and any OpenAI-compatible API — behind one consistent interface.

> [!NOTE]
> **Alpha release.** This is an early, experimental alpha — we are still
> testing it. APIs may change without notice and things may break. Not
> recommended for production use.

```python
import pyllym

pyllym.configure(lambda c: setattr(c, "openai_api_key", "sk-..."))

chat = pyllym.create_chat(model="gpt-5.4")
message = await chat.ask("What's the best way to learn Python?")
print(message.content)
```

> Every provider ships its own bloated client with different APIs, response
> formats, and conventions. pyllym gives you **one** interface for all of them —
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
pip install pyllym                # core
pip install "pyllym[db]"          # + SQLAlchemy persistence
pip install "pyllym[celery]"      # + Celery background tasks
pip install "pyllym[mime]"        # + content-based MIME sniffing
pip install "pyllym[sci]"         # + numerical stack for the data-analysis examples
pip install "pyllym[dev]"         # + test/lint tooling
```

The `sci` extra pulls in the scientific Python stack — **numpy**, **scipy**,
**pandas**, **matplotlib**, **seaborn**, **scikit-learn**, and **sympy** — used
only by the data-analysis examples (e.g. [`examples/stats.py`](examples/stats.py)).
The core library depends on nothing beyond `aiohttp` and `pydantic`; the
numerical packages are never imported by `pyllym` itself.

> This repository is a source tree; install it editable with
> `pip install -e ".[dev,db]"`.

## Configuration

```python
import pyllym

pyllym.configure(lambda c: (
    setattr(c, "openai_api_key", "sk-..."),
    setattr(c, "anthropic_api_key", "sk-ant-..."),
    setattr(c, "gemini_api_key", "..."),
))
```

Any provider option that isn't set in code falls back to the environment
variable of the same name in uppercase — `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_API_BASE`, and so on. Values
set via `pyllym.configure` always take precedence, so exporting keys in your
shell is enough to get started (including for CLI entry points like
`python -m pyllym.bdd`).

Per-call overrides use a `Context`:

```python
ctx = pyllym.context(lambda c: setattr(c, "request_timeout", 30))
chat = ctx.create_chat(model="claude-sonnet-4-6")
```

## Show me the code

### Ask anything

```python
chat = pyllym.create_chat(model="gpt-5.4")
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
from pyllym import Tool

class Weather(Tool):
    description = "Look up the weather for a city"

    async def execute(self, *, city: str) -> str:
        return f"It's sunny in {city}"

chat = pyllym.create_chat(model="gpt-5.4").with_tool(Weather)
answer = await chat.ask("What's the weather in Paris?")
# pyllym runs the agentic loop: model -> tool -> model -> final answer
```

Tools may be sync or async. Run them concurrently with
`chat.with_tools(A, B, concurrency=True)`.

### Structured output

```python
from pydantic import BaseModel

class Recipe(BaseModel):
    title: str
    ingredients: list[str]

chat = pyllym.create_chat(model="gpt-5.4").with_schema(Recipe)
msg = await chat.ask("Give me a recipe for pancakes")
msg.content  # -> a dict matching the schema
```

### Thinking / reasoning

```python
chat = pyllym.create_chat(model="claude-sonnet-4-6").with_thinking(effort="high")
```

### Embeddings, images, speech, transcription, moderation

```python
emb   = await pyllym.embed("Hello world")                       # Embedding(vectors=...)
image = await pyllym.paint("a red panda coding", model="gpt-image-1.5")
image.save("panda.png")
speech = await pyllym.speak("Hello there", model="gpt-4o-mini-tts")
speech.save("hello.mp3")
text  = await pyllym.transcribe("meeting.wav")                  # Transcription(text=...)
mod   = await pyllym.moderate("some text")                      # Moderation(...)
```

### Image & video generation via fal.ai

`pyllym.paint` reaches fal-hosted image models (FLUX.2, HunyuanImage, Qwen-Image);
`pyllym.animate` is a video-generation capability (LTX, Wan, HunyuanVideo) that
submits to fal's queue and polls until the render completes.

```python
pyllym.configure(lambda c: setattr(c, "fal_api_key", "..."))

image = await pyllym.paint("a red panda", provider="fal", model="fal-ai/flux/dev")
image.save("panda.png")

video = await pyllym.animate("a timelapse sunrise over mountains",
                            model="fal-ai/ltx-video-13b-distilled")  # provider="fal" default
video.save("sunrise.mp4")
```

### Agents

```python
from pyllym import Agent

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
from pyllym.persistence import create_models

class Base(DeclarativeBase): ...

Chat, Message, ToolCall = create_models(Base)   # bound to your Base

# ... create tables, open an AsyncSession ...
record = Chat(model_id="gpt-5.4", provider="openai")
session.add(record); await session.commit()

chat = record.to_chat(session)        # a pyllym.Chat backed by the DB
await chat.ask("Hello!")             # user + assistant rows are persisted
await session.commit()
```

## Background tasks (Celery)

Optionally run pyllym operations on Celery workers. `create_tasks` registers
ready-made tasks (`ask`, `embed`, `paint`, `speak`, `transcribe`, `moderate`)
on your app — broker-friendly JSON in, plain dicts out:

```python
from celery import Celery
from pyllym.celery import create_tasks

app = Celery("worker", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")
tasks = create_tasks(app)                 # or create_tasks(app, queue="llm")

result = tasks.ask.delay("What's the capital of France?", model="gpt-5.4")
result.get()                              # -> the assistant Message as a dict

tasks.embed.delay("Hello world")
tasks.paint.delay("a red panda coding", save_path="/tmp/panda.png")
```

Workers are synchronous; each task drives the underlying coroutine with
`pyllym.celery.run_async`, which also closes pyllym's HTTP pools before the
event loop shuts down. Use `run_async` directly in your own tasks for
richer features (tools, agents, callbacks) that don't serialize through a
broker:

```python
@app.task
def research(question: str) -> dict:
    chat = pyllym.create_chat(model="gpt-5.4").with_tool(Weather)
    return run_async(chat.ask(question)).to_dict()
```

## BDD test builder (pre-TDD scaffolding)

`pyllym.bdd` turns a **TOML spec** into a *red* pytest suite plus an
implementation brief — a building instruction for a coding agent (or a human)
to implement against. pyllym dogfoods itself: in request mode the planning step
is a structured-output call filling a `TestPlan` schema, and the actual test
code is *always* rendered from deterministic templates, never written freeform
by the model.

```python
from pyllym.bdd import build

results = await build("specs/", "tests/generated", model="gpt-5.4")
```

On Mac it might make sense to have an alias for Python `alias python="python3"`.

or from the shell:

```bash
python -m pyllym.bdd specs/retry.toml --out tests/generated --model gpt-5.4
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

### TDG — dependency-aware, incremental variant

> [!WARNING]
> **Use at your own responsibility.** The generated output is **not**
> guaranteed to be correct, complete, or safe. You are solely responsible for
> reviewing, testing, and validating anything it produces before relying on it.
> The maintainers accept no liability for any outcomes resulting from its use.

`pyllym.tdg` is a newer sibling of `pyllym.bdd` with the same TOML front-end,
two-mode planning, template rendering, and AST safety gate — plus two additions:

- **Dependency introspection** — it inspects the module under test for its real
  dependencies (static AST read, with a runtime-import fallback) and
  cross-checks them against the spec's declared imports, so the scaffolding
  reflects the code's actual collaborators rather than a stale hand-written
  surface.
- **Incremental runs** — each run writes into a timestamped output directory
  with a `_DONE.json` completion marker and a `latest.json` pointer, so a
  subsequent run skips specs that haven't changed and only regenerates what did.

The interface mirrors `pyllym.bdd`:

```python
from pyllym.tdg import build

results = await build("specs/", "tests/generated", model="gpt-5.4")
```

```bash
python -m pyllym.tdg specs/ --out tests/generated --model gpt-5.4
```

## ReactuLLM bridge (request-mode planner backend)

pyllym can serve as the **request-mode planner backend** for
[reactuLLM-sdd](https://github.com/PhotonEra-B-V/ReactuLLM) — a sibling TypeScript framework that
compiles a TOML spec into a red React Testing Library suite. In request mode
reactuLLM needs a model to fill a typed `TestPlan` via structured output;
instead of coupling the two over HTTP, they agree on a **shared JSON contract**
(`reactullm-pyllum.contract.json`). pyllym reads that file, sends its
`planner_instructions` as the system prompt, constrains a structured-output
`Chat` to its `test_plan_schema`, and returns one `TestPlan` object with
camelCase keys. Neither repo imports the other; the contract is the only
interface.

The whole connection is gated by one env var, `REACTULLM_PYLLUM_CONTRACT` (an
absolute path to the contract) — **set** turns the bridge on, **unset** leaves
it off with no error. `REACTULLM_PYLLUM_MODEL` picks the model (defaults to a
sensible pyllym model); provider keys follow pyllym's usual config precedence.

```python
from pyllym.reactullm_bridge import plan_from_spec, is_enabled

if is_enabled():
    plan = await plan_from_spec(spec_prompt)  # dict honoring test_plan_schema
```

```bash
# fills the TestPlan from a spec prompt; honors REACTULLM_PYLLUM_CONTRACT/_MODEL
python -m pyllym.reactullm_bridge spec.prompt.txt --out plan.json
```

### Cross-stack handoff (bidirectional)

Full-stack generation also needs a **handoff contract** that flows both ways:
sometimes the FastAPI backend is generated first and React matches it
(*backend-first*), sometimes React is generated first and the backend matches it
(*frontend-first*). `pyllym.reactullm_handoff` mirrors reactuLLM's
`handoff.ts` / `handoffStore.ts` — same file names, `runId` format
(`YYYYMMDDThhmmssZ`), and latest-wins de-collision — through a directory shared
by both repos, located by `REACTULLM_PYLLUM_HANDOFF_DIR` (unset = off).

```python
from pyllym.reactullm_handoff import commit_handoff, latest_handoff, is_newer_than_implemented

# producing (backend-first): write the API surface reactuLLM must conform to
commit_handoff(dir, {"version": 1, "direction": "backend_first",
                     "producer": "pyllum", "consumer": "reactullm",
                     "feature": "Job search", "apiSurface": {...}})

# consuming (frontend-first): implement only a strictly-newer handoff
if is_newer_than_implemented(dir, last_run_id):
    surface = latest_handoff(dir)["apiSurface"]
```

```bash
python -m pyllym.reactullm_handoff produce handoff.json   # commit a backend-first handoff
python -m pyllym.reactullm_handoff consume --implemented 20260713T142233Z
```

### Runtime handler (serving the live LLM envelope)

Where the bridge is a *build-time* planner, `pyllym.reactullm_runtime` is its
**runtime twin**: a framework-agnostic `handle()` that serves the live
end-user LLM request/response envelope agreed with reactuLLM. The server
registers the tasks it is willing to serve on a `TaskRegistry` — the **task
name is the authorization key**, so a request can only pick the model and system
prompt the server registered for that task, never inject its own. Requests and
responses are validated against the contract's JSON schemas (`LLMRequest` /
`LLMResponse`, with `LLMError` for failures).

Like the bridge, it is off unless a single env var is set —
`REACTULLM_PYLLUM_RUNTIME_CONTRACT` (an absolute path to the runtime contract);
**unset** leaves `is_enabled()` false and the handler dormant.

```python
from pyllym.reactullm_runtime import handle, is_enabled, LLMRequest, TaskConfig, TaskRegistry

registry = TaskRegistry().register("summarize", TaskConfig(default_prompt="Summarize concisely."))

if is_enabled():
    request = LLMRequest(task="summarize", input="... text to summarize ...")
    response = await handle(request, registry=registry)  # LLMResponse honoring the contract
```

## Supported providers

| Provider | Status |
|----------|--------|
| OpenAI (Chat Completions) | ✅ |
| Anthropic (Messages) | ✅ |
| Google Gemini (generateContent) | ✅ |
| DeepSeek, Mistral, xAI, Perplexity, OpenRouter | ✅ (OpenAI-compatible) |
| NVIDIA NIM, Cerebras, Hugging Face, Databricks | ✅ (OpenAI-compatible) |
| Qwen (DashScope), Zhipu GLM, Moonshot Kimi, Doubao, ERNIE, MiniMax | ✅ (OpenAI-compatible) |
| Ollama, GPUStack, vLLM (local) | ✅ |
| fal.ai (image via `paint`, video via `animate`) | ✅ FLUX.2, HunyuanImage, Qwen-Image, LTX, Wan, HunyuanVideo |
| Azure OpenAI | ✅ (v1-compatible endpoint) |
| AWS Bedrock (Converse) | ✅ non-streaming (SigV4 signed); streaming WIP |
| Vertex AI (Gemini) | ✅ with a supplied access token; OAuth minting WIP |

### GPU / CUDA acceleration

`pyllym` is a pure async **HTTP client** — it sends requests to provider APIs
and never loads or runs a model in-process, so it has no `torch`/CUDA
dependency and no in-library GPU code path. **GPU acceleration is a property of
the server you point pyllym at, not of pyllym.**

Numeric work — statistics, fitting, plotting — belongs on *your* side of that
line: expose the library functions you trust as **tools** and let the model call
them through pyllym's function-calling, so the numbers come from `numpy`/`scipy`,
not the model's guesses. See [Config-driven tools](#config-driven-tools-math--stats--plotting)
below; those libraries stay an opt-in extra, so the core install remains
dependency-light.

For the local providers (**Ollama**, **vLLM**, **GPUStack**), run the server on
a CUDA-capable host and it will use the GPU automatically; pyllym just talks to
its HTTP endpoint:

```python
import pyllym

# Point at a local, GPU-backed server (CUDA handled entirely by the server).
pyllym.configure(lambda c: setattr(c, "ollama_api_base", "http://localhost:11434"))
chat = pyllym.create_chat(model="llama3.1:70b")
```

- **Ollama** — uses an available NVIDIA GPU out of the box (see its CUDA docs);
  set `OLLAMA_API_BASE` and pyllym connects to it.
- **vLLM** — start `vllm serve <model>` on a CUDA host (tune `--tensor-parallel-size`,
  `--gpu-memory-utilization` there); point `VLLM_API_BASE` at it.
- **GPUStack** — manages GPU workers itself; pyllym targets its OpenAI-compatible
  endpoint via `GPUSTACK_API_BASE`.

For hosted providers (OpenAI, Anthropic, Gemini, …) the GPUs live in the
provider's infrastructure and are not something pyllym configures.

### Config-driven tools (math / stats / plotting)

A chat model is unreliable at arithmetic and worse at statistics. Rather than
hand-write a `Tool` subclass per function, declare the library callables the
model may use in a **TOML toolset** and pyllym generates one tool each —
introspecting the signature so `list[float]` params advertise as JSON arrays:

```toml
# analysis_tools.toml
[[tools]]
path = "statistics.mean"
description = "Arithmetic mean of a list of numbers."
[tools.params.data]
type = "array"
items = "number"
description = "The numbers to average."

[[tools]]
path = "numpy.corrcoef"
name = "correlation_matrix"
description = "Pearson correlation of one or more equal-length number series."
```

```python
import pyllym

tools = pyllym.load_toolset("analysis_tools.toml")
chat = pyllym.create_chat(model="llama3.1").with_tools(*tools)
answer = await chat.ask(
    "Given hours [1,2,3,4] and scores [52,55,61,60], are they correlated? "
    "Use the tools for every number."
)
```

The model then computes on your data (feed a stats result straight into a
plotting callable to produce a chart) with real library code, while pyllym stays
a pure HTTP client. **Security — explicit allowlist only:** just the exact
dotted paths in the file are importable and callable; there is no wildcard or
module expansion, and no model-authored-code path. Those callables run
**in-process**, so treat a toolset file like a list of imports you are choosing
to run and review it the same way. Heavy libraries (`numpy`, `matplotlib`, …)
are imported lazily, only when a toolset names them, and live behind the opt-in
`sci` extra (`pip install -e ".[sci]"`) — which bundles `numpy`, `scipy`,
`pandas`, `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `sympy`, and the
plotting stack. Any of these (and any other installed library) is reachable by
dotted path: `sklearn.metrics.r2_score`, `statsmodels.robust.mad`, etc. Model
objects whose method you want (`xgboost`/`lightgbm` `.predict`, `sklearn`
estimators) aren't bare callables — wrap your trained model's method in a
module-level function and point a `[[tools]]` entry at that.

A toolset can freely mix always-present and opt-in libraries: an entry whose
top-level package isn't installed is **skipped with a warning** (not a crash) by
default, so the rest of the file still loads. Pass
`load_toolset(path, skip_missing=False)` to require every named package, and
catch `pyllym.MissingToolPackageError` to detect the skipped ones.

**Optional libraries you already have — e.g. PyTorch.** pyllym never depends on
`torch`, but a toolset can name torch callables (`torch.mean`, or your own local
model's `predict`) by dotted path: if torch is installed the client imports and
calls it; if it isn't, the toolset fails with an actionable *"package 'torch' is
not installed — pip install torch"* rather than a crash. So torch is *usable
when present and a clean no-op when absent* — the client talks to whatever the
user has installed, and pyllym itself stays a pure HTTP client with no torch/CUDA
code path. See
[`examples/analysis_tools.toml`](examples/analysis_tools.toml) and
[`examples/stats.py`](examples/stats.py).

## Design

- **Async/await everywhere** for I/O. `await chat.ask(...)` and
  `async for chunk in chat.stream(...)`.
- **Provider → Protocol → Connection** architecture: a provider knows *where*
  and *who*; a protocol knows the wire format; the connection is `aiohttp`.
- **`aiohttp`** for HTTP, with connection pools shared across chats per event
  loop; call `await pyllym.aclose()` once at application shutdown.
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
