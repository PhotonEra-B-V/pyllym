"""Small data-manipulation helpers."""

from __future__ import annotations

import copy
import datetime as _dt
import re
from typing import Any, TypeVar

T = TypeVar("T")

_ACRONYM_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_ACRONYM_RE2 = re.compile(r"([a-z\d])([A-Z])")


def underscore(name: str) -> str:
    """Acronym-aware underscoring: ``HTTPProxyTool`` -> ``http_proxy_tool``."""
    name = _ACRONYM_RE1.sub(r"\1_\2", name)
    name = _ACRONYM_RE2.sub(r"\1_\2", name)
    return name.lower()


def to_safe_array(item: Any) -> list[Any]:
    """Wrap a value in a list unless it already is one. ``None`` -> ``[]``."""
    if isinstance(item, list):
        return item
    if item is None:
        return []
    if isinstance(item, tuple):
        return list(item)
    return [item]


def to_time(value: Any) -> _dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day, tzinfo=_dt.UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    parsed = parse_iso_date_prefix(value)
    return parsed


def parse_iso_date_prefix(value: Any) -> _dt.date | None:
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    date = str(value).strip()
    if not date:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return _dt.date.fromisoformat(date)
        if re.fullmatch(r"\d{4}-\d{2}", date):
            return _dt.date.fromisoformat(f"{date}-01")
        if re.fullmatch(r"\d{4}", date):
            return _dt.date.fromisoformat(f"{date}-01-01")
    except ValueError:
        return None
    return None


def iso_date_prefix_to_utc_midnight_string(value: Any) -> str | None:
    date = parse_iso_date_prefix(value)
    if date:
        return f"{date.strftime('%Y-%m-%d')} 00:00:00 UTC"
    return None


def deep_merge(original: dict[Any, Any], overrides: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively merge ``overrides`` into ``original`` (non-mutating)."""
    result = dict(original)
    for key, override_value in overrides.items():
        original_value = result.get(key)
        if isinstance(original_value, dict) and isinstance(override_value, dict):
            result[key] = deep_merge(original_value, override_value)
        else:
            result[key] = override_value
    return result


def deep_dup(value: T) -> T:
    return copy.deepcopy(value)


def compact(mapping: dict[Any, Any]) -> dict[Any, Any]:
    """Drop ``None`` values from a dict."""
    return {k: v for k, v in mapping.items() if v is not None}


def dig(data: Any, *keys: Any) -> Any:
    """Traverse nested dicts/lists, returning ``None`` on any miss.

    Integer keys index lists (with bounds checking); other keys index dicts.
    """
    for key in keys:
        if data is None:
            return None
        if isinstance(key, int) and isinstance(data, list):
            if -len(data) <= key < len(data):
                data = data[key]
            else:
                return None
        elif isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data
