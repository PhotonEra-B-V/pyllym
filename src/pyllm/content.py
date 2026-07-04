"""Content sent to / received from an LLM."""

from __future__ import annotations

from typing import Any

from . import utils
from .attachment import Attachment


class Content:
    """Text plus zero or more attachments."""

    def __init__(self, text: str | None = None, attachments: Any = None) -> None:
        self.text = text
        self.attachments: list[Attachment] = []
        self._process_attachments(attachments)
        if self.text is None and not self.attachments:
            raise ValueError("Text and attachments cannot be both nil")

    def add_attachment(self, source: Any, *, filename: str | None = None) -> Content:
        self.attachments.append(self._build_attachment(source, filename=filename))
        return self

    def format(self) -> Any:
        """Return the bare text when there are no attachments, else ``self``."""
        if self.text is not None and not self.attachments:
            return self.text
        return self

    def is_empty(self) -> bool:
        return not self.attachments and (self.text is None or self.text == "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "attachments": [a.to_dict() for a in self.attachments],
        }

    def _build_attachment(self, source: Any, *, filename: str | None = None) -> Attachment:
        if isinstance(source, Attachment):
            if not filename:
                return source
            return Attachment(source.source, filename=filename)
        return Attachment(source, filename=filename)

    def _process_attachments(self, attachments: Any) -> None:
        if isinstance(attachments, dict):
            for value in attachments.values():
                self._process_array_or_string(value)
        else:
            self._process_array_or_string(attachments)

    def _process_array_or_string(self, attachments: Any) -> None:
        for file in utils.to_safe_array(attachments):
            if self._is_blank_entry(file):
                continue
            self.add_attachment(file)

    @staticmethod
    def _is_blank_entry(file: Any) -> bool:
        return file is None or (isinstance(file, str) and not file.strip())


class RawContent:
    """Provider-specific payload that bypasses pyllm formatting."""

    def __init__(self, value: Any) -> None:
        if value is None:
            raise ValueError("Raw content payload cannot be nil")
        self.value = value

    def format(self) -> Any:
        return self.value

    def to_dict(self) -> Any:
        return self.value
