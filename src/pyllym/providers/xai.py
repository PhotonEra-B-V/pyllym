"""xAI (Grok) API integration."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class XAI(OpenAICompatible):
    default_api_base = "https://api.x.ai/v1"
    extra_headers = {"Content-Type": "application/json"}
