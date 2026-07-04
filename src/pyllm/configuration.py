"""Global configuration.

A plain object with known system options plus a permissive bag for
provider-specific keys (``openai_api_key``, ``anthropic_api_base``, ...),
registered when providers register. Unknown provider option access returns
``None``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# System-level defaults (mirrors the `option ... , default` declarations).
_SYSTEM_DEFAULTS: dict[str, Any] = {
    "default_model": "gpt-5.4",
    "default_embedding_model": "text-embedding-3-small",
    "default_moderation_model": "omni-moderation-latest",
    "default_image_model": "gpt-image-1.5",
    "default_speech_model": "gpt-4o-mini-tts",
    "default_transcription_model": "whisper-1",
    "default_video_model": None,
    "model_registry_file": None,  # resolved lazily to packaged models.json
    "model_registry_class": "Model",
    "model_registry_source": None,
    "request_timeout": 300,
    "max_retries": 3,
    "retry_interval": 0.1,
    "retry_backoff_factor": 2,
    "retry_interval_randomness": 0.5,
    "http_proxy": None,
    "tool_concurrency": False,
    "auto_upload_large_files": True,
    "logger": None,
    "instrumenter": None,
    "deprecation_behavior": "warn",
    "log_file": None,
    "log_level": None,
    "log_stream_debug": False,
}

_SECRET_SUFFIXES = ("_id", "_key", "_secret", "_token", "_credential_provider")


class Configuration:
    """Holds global and provider-specific options.

    Provider option keys (e.g. ``openai_api_key``) are registered via
    :meth:`register_provider_options` when providers register themselves, so
    reading an un-set provider option yields ``None`` rather than raising.
    """

    # Provider option names are global knowledge (registered once per provider
    # class), shared by every Configuration instance.
    _provider_keys: set[str] = set()

    def __init__(self) -> None:
        for key, default in _SYSTEM_DEFAULTS.items():
            object.__setattr__(self, key, default)
        if self.log_level is None:
            object.__setattr__(
                self,
                "log_level",
                logging.DEBUG if os.environ.get("PYLLM_DEBUG") else logging.INFO,
            )
        if self.log_stream_debug is False:
            object.__setattr__(
                self,
                "log_stream_debug",
                os.environ.get("PYLLM_STREAM_DEBUG") == "true",
            )

    def register_provider_options(self, keys: list[str]) -> None:
        for key in keys:
            type(self)._provider_keys.add(key)
            if not hasattr(self, key):
                object.__setattr__(self, key, None)

    def __getattr__(self, name: str) -> Any:
        # Only reached when the attribute is not set normally. Registered
        # provider options read as None when unset; anything else is a typo
        # and should fail loudly instead of silently reading as None.
        if not name.startswith("_") and name in self._provider_keys:
            return None
        raise AttributeError(f"Unknown configuration option: {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        # Empty/whitespace strings normalize to None.
        if isinstance(value, str) and not value.strip():
            value = None
        object.__setattr__(self, name, value)

    def to_dict(self, *, redact_secrets: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in {*_SYSTEM_DEFAULTS, *self._provider_keys}:
            if redact_secrets and key.endswith(_SECRET_SUFFIXES):
                continue
            out[key] = getattr(self, key, None)
        return out

    def copy(self) -> Configuration:
        import copy as _copy

        return _copy.copy(self)
