"""Ollama (local) API integration."""

from __future__ import annotations

from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider


class Ollama(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    @property
    def api_base(self) -> str:
        return self.config.ollama_api_base

    @property
    def headers(self) -> dict[str, str]:
        if not self.config.ollama_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.ollama_api_key}"}

    @classmethod
    def is_local(cls) -> bool:
        return True

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["ollama_api_base", "ollama_api_key"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["ollama_api_base"]
