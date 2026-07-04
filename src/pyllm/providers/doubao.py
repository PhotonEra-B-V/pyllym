"""ByteDance Doubao via Volcengine Ark (OpenAI-compatible).

Serves the Doubao family (doubao-pro-*, doubao-seed-*, ...). Models are referred
to by their Ark endpoint id or model name.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Doubao(OpenAICompatible):
    default_api_base = "https://ark.cn-beijing.volces.com/api/v3"
    assume_models = True
