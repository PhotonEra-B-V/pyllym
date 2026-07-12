"""Moonshot AI (Kimi) API integration (OpenAI-compatible).

Default base is the international endpoint; set ``moonshot_api_base`` to
``https://api.moonshot.cn/v1`` for the China region. Serves the Kimi family
(kimi-k2, moonshot-v1-*, ...).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Moonshot(OpenAICompatible):
    default_api_base = "https://api.moonshot.ai/v1"
    assume_models = True
