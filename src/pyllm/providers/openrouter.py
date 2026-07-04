"""OpenRouter API integration."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class OpenRouter(OpenAICompatible):
    default_api_base = "https://openrouter.ai/api/v1"
