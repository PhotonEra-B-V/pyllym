"""Hugging Face Inference Providers router (OpenAI-compatible)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class HuggingFace(OpenAICompatible):
    default_api_base = "https://router.huggingface.co/v1"
    assume_models = True
