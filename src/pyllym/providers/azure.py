"""Azure OpenAI-compatible API integration.

Simplified: assumes ``azure_api_base`` already points at an OpenAI v1-compatible
endpoint. Multi-mode deployment URL routing is not yet supported.
"""

from __future__ import annotations

from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider


class Azure(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    @property
    def api_base(self) -> str:
        return self.config.azure_api_base

    @property
    def headers(self) -> dict[str, str]:
        if self.config.azure_api_key:
            return {"api-key": self.config.azure_api_key}
        return {"Authorization": f"Bearer {self.config.azure_ai_auth_token}"}

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["azure_api_base", "azure_api_key", "azure_ai_auth_token", "azure_api_version"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["azure_api_base"]
