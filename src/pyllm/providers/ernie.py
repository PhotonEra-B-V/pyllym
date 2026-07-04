"""Baidu ERNIE via the Qianfan v2 API (OpenAI-compatible).

Serves the ERNIE family (ernie-4.5-turbo, ernie-speed, ernie-x1, ...).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class ERNIE(OpenAICompatible):
    default_api_base = "https://qianfan.baidubce.com/v2"
    assume_models = True
