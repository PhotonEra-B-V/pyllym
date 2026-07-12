"""Alibaba Qwen via DashScope (OpenAI-compatible).

Default base is the international DashScope endpoint; set ``qwen_api_base`` to
``https://dashscope.aliyuncs.com/compatible-mode/v1`` for the China region.
Serves the Qwen family (qwen-max, qwen3-*, qwen-plus, qwen-turbo, ...).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Qwen(OpenAICompatible):
    default_api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assume_models = True
