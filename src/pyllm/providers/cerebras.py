"""Cerebras inference API integration (OpenAI-compatible)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Cerebras(OpenAICompatible):
    default_api_base = "https://api.cerebras.ai/v1"
    assume_models = True
