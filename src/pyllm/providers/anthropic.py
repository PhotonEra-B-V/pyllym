"""Anthropic Claude API integration."""

from __future__ import annotations

from ..protocols.anthropic import Anthropic as AnthropicProtocol
from ..provider import Provider
from . import capabilities as _caps


class AnthropicCapabilities(_caps.Capabilities):
    pass


class Anthropic(Provider):
    protocols = {"anthropic": AnthropicProtocol}
    default_protocol_name = "anthropic"

    @property
    def api_base(self) -> str:
        return self.config.anthropic_api_base or "https://api.anthropic.com"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config.anthropic_api_key or "",
            "anthropic-version": "2023-06-01",
        }

    @classmethod
    def capabilities_cls(cls):
        return AnthropicCapabilities

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["anthropic_api_key", "anthropic_api_base"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["anthropic_api_key"]
