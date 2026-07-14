---
name: add-provider
description: Add a new OpenAI-compatible AI provider to pyllym — create the provider class under src/pyllym/providers/ and register it in the package façade. Use when the user wants to "add a provider", "support <vendor>'s API", or wire up a new OpenAI-compatible endpoint.
---

# add-provider

Add a provider following pyllym's **Provider (where/who) → Protocol (wire format)
→ Connection (aiohttp)** architecture. For any endpoint that speaks the
OpenAI chat-completions wire format this is a ~10-line file plus one line of
registration. Non-compatible vendors (Anthropic, Gemini, Bedrock Converse, fal)
implement their own protocol instead — that is a larger task; flag it rather
than forcing `OpenAICompatible`.

## Steps for an OpenAI-compatible provider

1. **Create `src/pyllym/providers/<slug>.py`.** Subclass `OpenAICompatible`.
   Use an existing minimal provider such as `providers/deepseek.py` as the
   template:

   ```python
   """<Vendor> API integration."""

   from __future__ import annotations

   from .openai_compatible import OpenAICompatible


   class <Vendor>(OpenAICompatible):
       default_api_base = "https://api.<vendor>.com/v1"
       assume_models = True  # skip a models-exist preflight when the vendor
                             # has no /models listing
   ```

   - `default_api_base` — the chat-completions base URL.
   - `assume_models` — set `True` when the vendor doesn't expose a model list.
   - Only override headers / config options if the vendor needs a non-standard
     auth header; otherwise `OpenAICompatible` handles `Authorization: Bearer`.

2. **Register it** in `src/pyllym/__init__.py`, inside
   `_register_builtin_providers()`, by appending a `(slug, ClassName)` tuple to
   the `registrations` list (keep it grouped with its peers). Registration is
   defensive — a module that fails to import is silently skipped.

3. **Config / env fallback.** Provider options fall back to the uppercase env
   var of the same name (e.g. `<vendor>_api_key` → `<VENDOR>_API_KEY`); values
   set via `pyllym.configure()` win. No extra wiring needed for the standard
   API-key case.

4. **Verify** with the `dev-checks` skill (ruff, mypy, pytest). Confirm the
   provider is discoverable:

   ```bash
   .venv/bin/python -c "import pyllym; print('<slug>' in [p.slug for p in pyllym.list_providers()])"
   ```

## Conventions

- `from __future__ import annotations` at the top; PEP 604 unions; builtin
  generics.
- Don't hand-edit `models.json` — it's generated. Prefer `pyllym.models.refresh()`.
- Keep provider modules thin: they declare *where/who*, not wire format.