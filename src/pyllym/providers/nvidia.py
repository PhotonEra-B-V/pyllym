"""NVIDIA NIM API integration (OpenAI-compatible)."""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class NVIDIA(OpenAICompatible):
    default_api_base = "https://integrate.api.nvidia.com/v1"
    assume_models = True
