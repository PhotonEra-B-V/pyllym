"""Shared base for providers that speak the Chat Completions dialect with
Bearer-token auth.

A concrete provider only declares its default endpoint and catalog behavior;
the config option names are derived from the registered slug::

    class Acme(OpenAICompatible):
        default_api_base = "https://api.acme.ai/v1"
        assume_models = True          # catalog not in models.json

which reads ``acme_api_key`` / ``acme_api_base`` from the configuration.
"""

from __future__ import annotations

from typing import ClassVar

from ..errors import ConfigurationError
from ..protocols.chat_completions import ChatCompletions
from ..provider import Provider


class OpenAICompatible(Provider):
    protocols = {"chat_completions": ChatCompletions}
    default_protocol_name = "chat_completions"

    #: Default endpoint; ``None`` means ``<slug>_api_base`` is required config.
    default_api_base: ClassVar[str | None] = None
    #: Whether model ids resolve without a registry entry.
    assume_models: ClassVar[bool] = False
    #: Extra static headers merged after Authorization.
    extra_headers: ClassVar[dict[str, str]] = {}

    @property
    def api_base(self) -> str:
        configured = getattr(self.config, f"{self.slug}_api_base")
        base = configured or self.default_api_base
        if base is None:
            raise ConfigurationError(f"{self.slug}_api_base is required")
        return str(base)

    @property
    def headers(self) -> dict[str, str]:
        api_key = getattr(self.config, f"{self.slug}_api_key")
        return {"Authorization": f"Bearer {api_key}", **self.extra_headers}

    @classmethod
    def assumes_models_exist(cls) -> bool:
        return cls.assume_models

    @classmethod
    def configuration_options(cls) -> list[str]:
        slug = cls.slug_name()
        return [f"{slug}_api_key", f"{slug}_api_base"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        slug = cls.slug_name()
        if cls.default_api_base is None:
            return [f"{slug}_api_base", f"{slug}_api_key"]
        return [f"{slug}_api_key"]
