"""Provider thinking/reasoning output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Thinking:
    text: str | None = None
    signature: str | None = None

    @classmethod
    def build(cls, *, text: str | None = None, signature: str | None = None) -> Thinking | None:
        if isinstance(text, str) and not text:
            text = None
        if isinstance(signature, str) and not signature:
            signature = None
        if text is None and signature is None:
            return None
        return cls(text=text, signature=signature)

    def __repr__(self) -> str:  # keep signatures out of logs
        sig = "[REDACTED]" if self.signature else None
        return f"Thinking(text={self.text!r}, signature={sig!r})"


@dataclass(frozen=True, slots=True)
class ThinkingConfig:
    """Normalized config for requesting thinking across providers."""

    effort: str | None = None
    budget: int | None = None

    def __post_init__(self) -> None:
        # effort may arrive as an enum-like; normalize to str
        if self.effort is not None and not isinstance(self.effort, str):
            object.__setattr__(self, "effort", str(self.effort))

    @property
    def enabled(self) -> bool:
        return self.effort is not None or self.budget is not None
