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
- 📇 A bundled **model registry** (`models.json`) with pricing & capabilities
- 🐍 Modern, fully type-annotated **Python 3.13+**, async-first on `httpx`

## Installation

```bash
pip install pyllm                # core
pip install "pyllm[db]"          # + SQLAlchemy persistence
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
  and *who*; a protocol knows the wire format; the connection is `httpx`.
- **`httpx`** for HTTP, with connection pools shared across chats per event
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

## License

MIT — see [LICENSE](LICENSE).
