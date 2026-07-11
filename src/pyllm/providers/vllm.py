"""vLLM (self-hosted) API integration.

vLLM's OpenAI-compatible server speaks the Chat Completions dialect at a
user-supplied endpoint (e.g. ``http://localhost:8000/v1``). Auth is optional:
a bearer token is sent only when ``vllm_api_key`` is configured (matching the
server's ``--api-key`` flag).
"""

from __future__ import annotations

from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider


class VLLM(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    @property
    def api_base(self) -> str:
        return self.config.vllm_api_base

    @property
    def headers(self) -> dict[str, str]:
        if not self.config.vllm_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.vllm_api_key}"}

    @classmethod
    def is_local(cls) -> bool:
        return True

    @classmethod
    def assumes_models_exist(cls) -> bool:
        # Served models are whatever the operator loaded; not in models.json.
        return True

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["vllm_api_base", "vllm_api_key"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["vllm_api_base"]
