"""TypeSafe AI (Jev, the "System One" decision model).

Jev answers typed questions with calibrated distributions via
``pyllym.decide()``; it has no chat endpoint. Auth is a bearer token
(``typesafe_api_key`` / ``TYPESAFE_API_KEY``). The default base is TypeSafe's
own API; set ``typesafe_api_base`` for a gateway such as a LiteLLM proxy.
"""

from __future__ import annotations

from ..protocols.systemone import SystemOne
from .openai_compatible import OpenAICompatible


class TypeSafe(OpenAICompatible):
    protocols = {"systemone": SystemOne}
    default_protocol_name = "systemone"
    default_api_base = "https://api.typesafe.ai/v1"
    # Jev versions ship faster than the registry; unknown ids pass through.
    assume_models = True

    @classmethod
    def display_name(cls) -> str:
        return "TypeSafe"
