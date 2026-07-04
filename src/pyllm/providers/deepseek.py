"""DeepSeek API integration."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class DeepSeek(OpenAICompatible):
    default_api_base = "https://api.deepseek.com"
    assume_models = True
