"""Native Gemini API integration."""

from __future__ import annotations

import re
from typing import Any

from ..protocols.gemini import Gemini as GeminiProtocol
from ..provider import Provider
from . import capabilities as _caps

_PRICES: dict[str, dict[str, float]] = {
    "flash_2": {"input": 0.10, "output": 0.40},
    "flash_lite_2": {"input": 0.075, "output": 0.30},
    "flash": {"input": 0.075, "output": 0.30},
    "flash_8b": {"input": 0.0375, "output": 0.15},
    "pro": {"input": 1.25, "output": 5.0},
    "pro_2_5": {"input": 0.12, "output": 0.50},
    "gemini_embedding": {"input": 0.002, "output": 0.004},
    "embedding": {"input": 0.00, "output": 0.00},
    "imagen": {"price": 0.03},
    "aqa": {"input": 0.00, "output": 0.00},
}


class GeminiCapabilities(_caps.Capabilities):
    @classmethod
    def context_window_for(cls, model_id: str) -> int | None:
        if re.search(
            r"gemini-2\.5-pro-exp-03-25|gemini-2\.0-flash|gemini-2\.0-flash-lite|"
            r"gemini-1\.5-flash|gemini-1\.5-flash-8b",
            model_id,
        ):
            return 1_048_576
        if re.search(r"gemini-1\.5-pro", model_id):
            return 2_097_152
        if re.search(r"gemini-embedding-exp", model_id):
            return 8_192
        if re.search(r"text-embedding-004|embedding-001", model_id):
            return 2_048
        if re.search(r"aqa", model_id):
            return 7_168
        if re.search(r"imagen-3", model_id):
            return None
        return 32_768

    @classmethod
    def max_tokens_for(cls, model_id: str) -> int | None:
        if re.search(r"gemini-2\.5-pro-exp-03-25", model_id):
            return 64_000
        if re.search(
            r"gemini-2\.0-flash|gemini-2\.0-flash-lite|gemini-1\.5-flash|"
            r"gemini-1\.5-flash-8b|gemini-1\.5-pro",
            model_id,
        ):
            return 8_192
        if re.search(r"gemini-embedding-exp", model_id):
            return None
        if re.search(r"text-embedding-004|embedding-001", model_id):
            return 768
        if re.search(r"imagen-3", model_id):
            return 4
        return 4_096

    @classmethod
    def critical_capabilities_for(cls, model_id: str) -> list[str]:
        out: list[str] = []
        if cls.supports_functions(model_id):
            out.append("function_calling")
        if cls.supports_structured_output(model_id):
            out.append("structured_output")
        if cls.supports_vision(model_id):
            out.append("vision")
        return out

    @classmethod
    def pricing_for(cls, model_id: str) -> dict[str, Any]:
        prices = _PRICES.get(cls.pricing_family(model_id), {"input": 0.075, "output": 0.30})
        return {
            "text_tokens": {
                "standard": {
                    "input_per_million": prices.get("input") or prices.get("price") or 0.075,
                    "output_per_million": prices.get("output") or prices.get("price") or 0.30,
                }
            }
        }

    @classmethod
    def supports_vision(cls, model_id: str) -> bool:
        if re.search(r"text-embedding|embedding-001|aqa", model_id):
            return False
        return bool(re.search(r"gemini|flash|pro|imagen", model_id))

    @classmethod
    def supports_functions(cls, model_id: str) -> bool:
        if re.search(
            r"text-embedding|embedding-001|aqa|flash-lite|imagen|gemini-2\.0-flash-lite", model_id
        ):
            return False
        return bool(re.search(r"gemini|pro|flash", model_id))

    @classmethod
    def supports_structured_output(cls, model_id: str) -> bool:
        if re.search(
            r"text-embedding|embedding-001|aqa|imagen|gemini-2\.0-flash-lite|"
            r"gemini-2\.5-pro-exp-03-25",
            model_id,
        ):
            return False
        return bool(re.search(r"gemini|pro|flash", model_id))

    @classmethod
    def pricing_family(cls, model_id: str) -> str:
        patterns = [
            (r"gemini-2\.5-pro-exp-03-25", "pro_2_5"),
            (r"gemini-2\.0-flash-lite", "flash_lite_2"),
            (r"gemini-2\.0-flash", "flash_2"),
            (r"gemini-1\.5-flash-8b", "flash_8b"),
            (r"gemini-1\.5-flash", "flash"),
            (r"gemini-1\.5-pro", "pro"),
            (r"gemini-embedding-exp", "gemini_embedding"),
            (r"text-embedding|embedding", "embedding"),
            (r"imagen", "imagen"),
            (r"aqa", "aqa"),
        ]
        for pattern, family in patterns:
            if re.search(pattern, model_id):
                return family
        return "base"


class Gemini(Provider):
    protocols = {"gemini": GeminiProtocol}
    default_protocol_name = "gemini"

    @property
    def api_base(self) -> str:
        return self.config.gemini_api_base or "https://generativelanguage.googleapis.com/v1beta"

    @property
    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.config.gemini_api_key}

    @classmethod
    def capabilities_cls(cls):
        return GeminiCapabilities

    @classmethod
    def configuration_options(cls) -> list[str]:
        return ["gemini_api_key", "gemini_api_base"]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["gemini_api_key"]
