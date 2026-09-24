# Changelog

## Unreleased

### Fixed

- **Rate limits are actually waited out.** A 429 used to be retried with the
  generic backoff (0.1s, 0.2s, 0.4s — under a second in total), so hitting a
  provider's throttle (seen with Zhipu GLM's 1302/1303/1305 limits) surfaced
  as `RateLimitError` almost immediately. Rate limits now back off from a new
  `rate_limit_retry_interval` (default 1s, so 1s/2s/4s across the default
  three retries), and a server-supplied `Retry-After` / `retry-after-ms` /
  OpenAI `x-ratelimit-reset-*` hint is honored over the computed delay. A new
  `retry_max_interval` (default 60s) caps any single wait; a hint above it
  gives up at once rather than retrying into a certain refusal. The parsed
  hint is exposed as `Error.retry_after` for callers that schedule their own
  retry.
- **Streaming requests retry too.** `Connection.stream` raised on the first
  429/5xx or connection error without a single retry, so `chat.stream(...)`
  was far more fragile than `chat.ask(...)`. Retryable failures that happen
  before the first body byte now follow the same policy; a stream that drops
  after it has started still surfaces as `ConnectionFailedError`, since it
  cannot be replayed.
- **Exhausted quota is `PaymentRequiredError`, not a retried 429.** OpenAI
  `insufficient_quota` and Zhipu/z.ai code 1113 ("insufficient balance ...
  please recharge") arrive as HTTP 429 but cannot succeed on retry; they now
  map to `PaymentRequiredError` (both HTTP and in-stream) instead of being
  backed off and re-sent. Code catching `RateLimitError` for those should
  catch `PaymentRequiredError` as well.
- Each retry is logged at INFO with the delay and the error class, so
  throttling is visible instead of silent.

## 1.16.0b4 (2026-09-22)

### Changed

- **Package management moved to [uv](https://docs.astral.sh/uv/).** The repo
  now ships a committed `uv.lock`; `uv sync` creates `.venv` with the package
  installed editable plus the dev tooling. Dev tooling (`pytest`, `ruff`,
  `mypy`, ...) lives in a PEP 735 `[dependency-groups] dev` group instead of
  the published `dev` extra — `pip install "pyllym[dev]"` no longer exists.
  The dev group also pulls in the `db`/`celery`/`mcp`/`mime` extras the test
  suite exercises, and `aiosqlite`, which the persistence tests always needed
  but nothing declared. The PyPI publish workflow builds with `uv build`, and
  Dependabot tracks the `uv` ecosystem.

### Fixed

- **`db` extra now declares `sqlalchemy[asyncio]`.** The persistence layer is
  async SQLAlchemy, which requires `greenlet`; a clean install of
  `pyllym[db]` previously failed at first use with *"the greenlet library is
  required to use this function"* unless greenlet happened to be present.
- **`pyllym[mcp]` works on the mcp 2.x SDK.** mcp 2.0 removed
  `streamablehttp_client` (its `headers=` keyword moved onto a caller-supplied
  httpx client) and changed the streams yielded by the transport from
  `(read, write, get_session_id)` to `(read, write)`, so a fresh install of the
  extra — which now resolves to 2.x — failed at connect time for HTTP servers.
  `MCPServer.http` now probes the installed SDK for the newer
  `streamable_http_client` entry point (present since late 1.x), passes
  headers through an httpx client it owns and closes, and accepts either
  stream shape. Verified against real streamable-HTTP servers on mcp 1.20,
  1.30 and 2.2; stdio transport is unchanged.

## 1.16.0b3 (2026-09-21)

### Added

- **DeepSeek V4.1 Flash** (`deepseek-flash`) registered for the direct
  `deepseek` provider (released 2026-09-10). 1,000,000-token context window,
  384,000 max output tokens, native image input (standard `image_url` content
  blocks — no protocol change needed), tools, structured output, and thinking
  with a `low`/`high`/`max` effort ladder. A `deepseek-v4.1-flash` convenience
  key resolves to the canonical id, so
  `pyllym.chat("deepseek-flash")` works without `provider=`.

### Changed

- **`deepseek-v4-flash` and `deepseek-v4-pro` now carry V4.1 Flash rates and
  capabilities.** DeepSeek retired V4-Flash and routes the legacy id to
  V4.1-Flash; since 2026-09-14 04:00 UTC `deepseek-v4-pro` requests are also
  served and billed as V4.1-Flash until V4.1-Pro launches. Both entries record
  this as `metadata.served_by = "deepseek-flash"`. Previously costs for these
  ids were computed from the stale V4 rate card.

  DeepSeek bills by time of day, which the registry's pricing tiers cannot
  express. `pricing` holds the **off-peak** rate ($0.15 input / $0.60 output /
  $0.003 cache read per MTok), matching `models.dev`. Peak rates are exactly
  double (weekdays 01:00–04:00 and 06:00–10:00 UTC, excluding Chinese public
  holidays) and are recorded under `metadata.peak_cost`; `Cost` will
  under-report requests made in those windows.

## 1.16.0b2 (2026-09-08)

### Added

- **Claude Fable 5.1** (`claude-fable-5-1`) registered for the `anthropic`
  provider. 1,000,000-token context window, 128,000 max output tokens,
  $10/$50 per MTok. Cache reads are **$0.25/MTok** — a real reduction from
  Claude Fable 5's $1.00, not inherited from it. Exposes the five-level effort
  ladder (`low`, `medium`, `high`, `xhigh`, `max`). Aliases map the model
  across `anthropic`, `openrouter`, `bedrock` and `vertexai`; Bedrock serves it
  as `anthropic.claude-fable-5-1`, without the `eu.` prefix Fable 5 uses. A
  `claude-fable-5.1` convenience key resolves the dotted form to the canonical
  id.
- **GPT-6 Astra** (`gpt-6-astra`) and **GPT-6 Astra Pro**
  (`gpt-6-astra-pro`) registered for the `openai` provider. 1,050,000-token
  context window, 128,000 max output tokens, $10/$50 per MTok with $1.00 cache
  reads. Both carry the long-context pricing tier that applies past 272K input
  tokens (2x input and cache, 1.5x output), using the same metadata shape as
  `gpt-5.5`, and the five-level effort ladder. Aliases cover `openai` and
  `openrouter`.

  Note that OpenAI's sixth-generation flagship ships as `gpt-6-astra`; there is
  no bare `gpt-6` model id.

  Neither of these models was reachable via `pyllym.models.refresh()` — the
  upstream `models.dev` dataset has not indexed them yet, so both were added
  with `metadata.source = "manual"`.

## 1.16.0b1 (2026-09-08)

### Added

- **GLM-OCR document layout parsing** via z.ai. New `pyllym.parse_layout(file)`
  façade extracts Markdown text and positional layout elements from images and
  PDFs. z.ai serves `glm-ocr` from a dedicated `layout_parsing` endpoint rather
  than `chat/completions`, so this adds a `parse_layout` concern to the protocol
  layer (`Protocol.parse_layout` raises `NotImplementedError` by default) and a
  `ZhipuLayout` protocol implementing it. Returns a `LayoutParsing` with
  `.markdown`, `.elements` (each a `LayoutElement` carrying `label`, `bbox`,
  `content`), `.page_count` and token usage. Accepts URLs, `data:` URIs, and
  local paths (read and base64-encoded automatically). Configurable via
  `config.default_layout_parsing_model` (default `glm-ocr`).
- **GLM-5.3 and GLM-5.3-Flash** registered for the `zhipu` provider with real
  context/pricing metadata, so cost tracking and context-limit checks work
  rather than falling back to the assumed-model defaults. Both carry a
  1,048,576-token context window and 131,072 max output tokens. Aliases map
  `glm-5.3` / `glm-5.3-flash` across the `zhipu` and `openrouter` providers.

### Fixed

- **`models_schema.json` was not valid JSON Schema.** Four properties
  (`created_at`, `context_window`, `max_output_tokens`, `knowledge_cutoff`)
  declared `"type": ["null", {…}]`, but a `type` array may only contain type-name
  strings. Any attempt to validate the registry failed on the schema itself
  before reaching the data. Rewritten as `anyOf`, preserving the intent; the
  packaged registry now validates cleanly.

## 1.16.0a2 (2026-07-18)

### Added

- **MCP client support** (`mcp` extra). `pyllym.MCPServer.stdio(...)` /
  `.http(...)` connect to Model Context Protocol servers and adapt their tools
  into ordinary pyllym `Tool` objects (`MCPTool`) usable in the same agentic
  loop; `tools_from_session` adapts any live session. See the README's
  "MCP tools" section.

### Fixed

- **Transport errors are now always pyllym errors.** `Connection` no longer
  re-raises raw `aiohttp` / `TimeoutError` exceptions after retries are
  exhausted (or on non-retryable transport failures). They are wrapped in the
  new `pyllym.ConnectionFailedError` (a subclass of `pyllym.Error`), with the
  original exception preserved as `__cause__`. The same guarantee now covers
  streaming (`Connection.stream`), multipart uploads, and image/video URL
  downloads (`Image.ato_blob` / `Video.ato_blob`, which also map HTTP error
  statuses through the standard error hierarchy instead of raising
  `aiohttp.ClientResponseError`). Callers only ever need
  `except pyllym.Error`.

### Changed (wire payloads)

- **Gemini: system messages are sent as `systemInstruction`.** Previously
  system prompts were folded into `contents` as a `user` turn. They are now
  emitted via the API's first-class `systemInstruction` field (multiple system
  messages are concatenated); `contents` carries only user/assistant/tool
  turns.
- **OpenAI-compatible providers send the classic `system` role again.** The
  `developer` role is now only used for the OpenAI API itself (overridable
  back to `system` via `config.openai_use_system_role`). All other providers
  speaking the Chat Completions protocol — DeepSeek, Mistral, Ollama,
  OpenRouter, vLLM, GPUStack, `openai_compatible`, etc. — send `system`, which
  local/self-hosted servers actually accept. Custom providers can opt in via
  the `Provider.uses_developer_role()` classmethod.