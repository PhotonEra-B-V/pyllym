# Changelog

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