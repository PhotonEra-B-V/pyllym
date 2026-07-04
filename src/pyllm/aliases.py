"""Model alias resolution."""

from __future__ import annotations

import json
from pathlib import Path

_ALIASES_FILE = Path(__file__).with_name("aliases.json")
_aliases: dict[str, dict[str, str]] | None = None


def _load() -> dict[str, dict[str, str]]:
    global _aliases
    if _aliases is None:
        if _ALIASES_FILE.exists():
            _aliases = json.loads(_ALIASES_FILE.read_text())
        else:
            _aliases = {}
    return _aliases


def resolve(model_id: str, provider: str | None = None) -> str:
    aliases = _load()
    entry = aliases.get(model_id)
    if not entry:
        return model_id
    if provider:
        return entry.get(str(provider)) or model_id
    return next(iter(entry.values()), model_id)


def reload() -> None:
    global _aliases
    _aliases = None
    _load()
