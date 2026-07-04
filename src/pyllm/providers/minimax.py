"""MiniMax API integration (OpenAI-compatible).

Default base is the international endpoint; set ``minimax_api_base`` to
``https://api.minimaxi.com/v1`` for the China region. Serves the MiniMax family
(minimax-m2, minimax-m1, abab-*, ...).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class MiniMax(OpenAICompatible):
    default_api_base = "https://api.minimax.io/v1"
    assume_models = True
