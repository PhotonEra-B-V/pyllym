"""OpenAI API integration.

Note: this port currently speaks Chat Completions (the widely compatible
default); the OpenAI *Responses* protocol is a planned addition.
"""

from __future__ import annotations

from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider
from . import capabilities as _caps


class OpenAICapabilities(_caps.Capabilities):
    pass


class OpenAI(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    @property
    def api_base(self) -> str:
        return self.config.openai_api_base or "https://api.openai.com/v1"

    @property
    def headers(self) -> dict[str, str]:
        raw = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "OpenAI-Organization": self.config.openai_organization_id,
            "OpenAI-Project": self.config.openai_project_id,
        }
        return {k: v for k, v in raw.items() if v}

    @classmethod
    def uses_developer_role(cls) -> bool:
        # `cls is OpenAI` so the OpenAI-compatible providers subclassing this
        # class (DeepSeek, OpenRouter, ...) keep the classic "system" role.
        return cls is OpenAI

    @classmethod
    def capabilities_cls(cls):
        return OpenAICapabilities

    @classmethod
    def configuration_options(cls) -> list[str]:
        return [
            "openai_api_key",
            "openai_api_base",
            "openai_organization_id",
            "openai_project_id",
            "openai_use_system_role",
        ]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["openai_api_key"]
