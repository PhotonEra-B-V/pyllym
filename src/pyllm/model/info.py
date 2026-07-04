"""Model capability/pricing/metadata record."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import utils
from .modalities import Modalities
from .pricing import Pricing

if TYPE_CHECKING:
    from ..cost import Cost

_CAPABILITY_PREDICATES = (
    "function_calling",
    "structured_output",
    "batch",
    "reasoning",
    "citations",
    "streaming",
)


class Info:
    """Information about an AI model's capabilities, pricing, and metadata."""

    @classmethod
    def default(cls, model_id: str, provider: str) -> Info:
        return cls(
            {
                "id": model_id,
                "name": model_id.replace("-", " ").capitalize(),
                "provider": provider,
                "capabilities": [
                    "function_calling",
                    "streaming",
                    "vision",
                    "structured_output",
                ],
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "metadata": {"warning": "Assuming model exists, capabilities may not be accurate"},
            }
        )

    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data.get("id")  # type: ignore[assignment]
        self.name: str | None = data.get("name")
        self.provider: str = data.get("provider")  # type: ignore[assignment]
        self.family: str | None = data.get("family")
        created = utils.to_time(data.get("created_at"))
        self.created_at = created
        self.context_window: int | None = data.get("context_window")
        self.max_output_tokens: int | None = data.get("max_output_tokens")
        self.knowledge_cutoff = utils.to_date(data.get("knowledge_cutoff"))
        self.modalities = Modalities(data.get("modalities") or {})
        self.capabilities: list[str] = list(data.get("capabilities") or [])
        self.pricing = Pricing(data.get("pricing") or {})
        self.metadata: dict[str, Any] = dict(data.get("metadata") or {})
        self.reasoning_options = self._normalize_reasoning_options(
            data.get("reasoning_options") or self.metadata.get("reasoning_options")
        )
        if self.reasoning_options:
            self.metadata["reasoning_options"] = self.reasoning_options

    def supports(self, capability: str) -> bool:
        return str(capability) in self.capabilities

    def __getattr__(self, name: str) -> Any:
        # capability predicates: function_calling(), reasoning(), ...
        if name in _CAPABILITY_PREDICATES:
            return lambda: self.supports(name)
        raise AttributeError(name)

    @property
    def display_name(self) -> str | None:
        return self.name

    @property
    def label(self) -> str:
        return f"{self.provider} - {self.display_name}"

    @property
    def max_tokens(self) -> int | None:
        return self.max_output_tokens

    def supports_vision(self) -> bool:
        return "image" in self.modalities.input

    def supports_video(self) -> bool:
        return "video" in self.modalities.input

    def supports_functions(self) -> bool:
        return self.supports("function_calling")

    def is_function_calling(self) -> bool:
        return self.supports("function_calling")

    def is_structured_output(self) -> bool:
        return self.supports("structured_output")

    def is_reasoning(self) -> bool:
        return self.supports("reasoning")

    def is_citations(self) -> bool:
        return self.supports("citations")

    def is_streaming(self) -> bool:
        return self.supports("streaming")

    def is_batch(self) -> bool:
        return self.supports("batch")

    def reasoning_option(self, type: str) -> dict[str, Any] | None:
        for option in self.reasoning_options:
            if option.get("type") == str(type):
                return option
        return None

    def reasoning_option_values(self, type: str) -> list[Any]:
        option = self.reasoning_option(type)
        return list(option.get("values") or []) if option else []

    @property
    def input_price_per_million(self) -> float | None:
        return self.pricing.text_tokens.input

    @property
    def output_price_per_million(self) -> float | None:
        return self.pricing.text_tokens.output

    def cost_for(self, tokens: Any) -> Cost:
        from ..cost import Cost

        if hasattr(tokens, "tokens"):
            tokens = tokens.tokens
        return Cost(tokens=tokens, model=self)

    @property
    def type(self) -> str:
        output = self.modalities.output
        if "embeddings" in output:
            return "embedding"
        if "moderation" in output:
            return "moderation"
        if "image" in output:
            return "image"
        if "audio" in output:
            return "audio"
        if "video" in output:
            return "video"
        return "chat"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "family": self.family,
            "created_at": self.created_at,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "knowledge_cutoff": self.knowledge_cutoff,
            "modalities": self.modalities.to_dict(),
            "capabilities": self.capabilities,
            "pricing": self.pricing.to_dict(),
            "metadata": self.metadata,
        }

    @staticmethod
    def _normalize_reasoning_options(options: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for option in utils.to_safe_array(options):
            if not isinstance(option, dict):
                continue
            entry = {str(k): v for k, v in option.items()}
            if "type" in entry:
                entry["type"] = str(entry["type"])
            if "values" in entry:
                entry["values"] = [str(v) for v in utils.to_safe_array(entry["values"])]
            normalized.append(entry)
        return normalized
