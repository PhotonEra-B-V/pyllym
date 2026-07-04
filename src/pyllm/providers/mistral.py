"""Mistral API integration."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Mistral(OpenAICompatible):
    default_api_base = "https://api.mistral.ai/v1"
