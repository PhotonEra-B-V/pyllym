"""Zhipu AI GLM via the z.ai / BigModel API (OpenAI-compatible).

Default base is the international z.ai endpoint; set ``zhipu_api_base`` to
``https://open.bigmodel.cn/api/paas/v4`` for the China region. Serves the GLM
family (glm-5, glm-5.1, glm-4.6, glm-4.5, ...).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Zhipu(OpenAICompatible):
    default_api_base = "https://api.z.ai/api/paas/v4"
    assume_models = True
