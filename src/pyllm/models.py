"""Registry of available models.

The heavy ``models.dev`` merge logic behind ``refresh!`` is provided
in an async-friendly form; the common runtime paths (``find``/``resolve`` and
the type filters) are the focus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from . import aliases
from .errors import ModelNotFoundError
from .model.info import Info

if TYPE_CHECKING:
    from .configuration import Configuration
    from .provider import Provider

_MODELS_FILE = Path(__file__).with_name("models.json")

# First-party providers outrank the aggregators that resell their models.
PROVIDER_PREFERENCE = [
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "mistral",
    "perplexity",
    "xai",
    "vertexai",
    "bedrock",
    "openrouter",
    "azure",
    "ollama",
    "gpustack",
]


def _read_from_json(file: Path | str = _MODELS_FILE) -> list[Info]:
    path = Path(file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return [Info(model) for model in data]


class Models:
    """Holds a list of :class:`~pyllm.model.info.Info` records."""

    def __init__(self, models: list[Info] | None = None) -> None:
        self._models = models if models is not None else _read_from_json()

    def __iter__(self):
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def all(self) -> list[Info]:
        return self._models

    def load_from_json(self, file: Path | str = _MODELS_FILE) -> Models:
        self._models = _read_from_json(file)
        return self

    def save_to_json(self, file: Path | str = _MODELS_FILE) -> None:
        Path(file).write_text(
            json.dumps([m.to_dict() for m in self._models], indent=2, default=str)
        )

    # --- lookup ----------------------------------------------------------------
    def find(self, model_id: str, provider: str | None = None) -> Info:
        if provider:
            return self._find_with_provider(model_id, provider)
        return self._find_without_provider(model_id)

    def chat_models(self) -> Models:
        return Models([m for m in self._models if m.type == "chat"])

    def embedding_models(self) -> Models:
        return Models(
            [
                m
                for m in self._models
                if m.type == "embedding" or "embeddings" in m.modalities.output
            ]
        )

    def audio_models(self) -> Models:
        return Models(
            [m for m in self._models if m.type == "audio" or "audio" in m.modalities.output]
        )

    def image_models(self) -> Models:
        return Models(
            [m for m in self._models if m.type == "image" or "image" in m.modalities.output]
        )

    def by_family(self, family: str) -> Models:
        return Models([m for m in self._models if m.family == str(family)])

    def by_provider(self, provider: str) -> Models:
        return Models([m for m in self._models if m.provider == str(provider)])

    def resolve(
        self,
        model_id: str,
        *,
        provider: str | None = None,
        assume_exists: bool = False,
        config: Configuration | None = None,
    ) -> tuple[Info, Provider]:
        return resolve(model_id, provider=provider, assume_exists=assume_exists, config=config)

    async def refresh(self, *, remote_only: bool = False) -> Models:
        self._models = await fetch_merged_models(remote_only=remote_only)
        return self

    # --- internals -------------------------------------------------------------
    def _find_with_provider(self, model_id: str, provider: str) -> Info:
        resolved_id = aliases.resolve(model_id, provider)
        match = next(
            (m for m in self._models if m.id == resolved_id and m.provider == str(provider)),
            None,
        ) or next(
            (m for m in self._models if m.id == model_id and m.provider == str(provider)),
            None,
        )
        if match:
            return match
        raise ModelNotFoundError(
            f"Unknown model: {model_id!r} for provider: {provider!r}. {_REFRESH_GUIDANCE}"
        )

    def _find_without_provider(self, model_id: str) -> Info:
        resolved_id = aliases.resolve(model_id)
        candidates = [m for m in self._models if m.id in (model_id, resolved_id)]
        candidates.sort(key=lambda m: 0 if m.id == model_id else 1)
        match = self._preferred_match(candidates)
        if match:
            return match
        raise ModelNotFoundError(f"Unknown model: {model_id!r}. {_REFRESH_GUIDANCE}")

    @staticmethod
    def _preferred_match(candidates: list[Info]) -> Info | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return min(
            candidates,
            key=lambda m: (
                PROVIDER_PREFERENCE.index(m.provider)
                if m.provider in PROVIDER_PREFERENCE
                else len(PROVIDER_PREFERENCE)
            ),
        )


_REFRESH_GUIDANCE = (
    "If the model exists at the provider, refresh the registry with "
    "`await pyllm.models.refresh()` and persist it with `pyllm.models.save_to_json()`."
)

# --- module-level singleton ---------------------------
_instance: Models | None = None


def instance() -> Models:
    global _instance
    if _instance is None:
        _instance = Models()
    return _instance


def all() -> list[Info]:
    return instance().all()


def find(model_id: str, provider: str | None = None) -> Info:
    return instance().find(model_id, provider)


def chat_models() -> Models:
    return instance().chat_models()


def embedding_models() -> Models:
    return instance().embedding_models()


def image_models() -> Models:
    return instance().image_models()


def audio_models() -> Models:
    return instance().audio_models()


def by_provider(provider: str) -> Models:
    return instance().by_provider(provider)


def by_family(family: str) -> Models:
    return instance().by_family(family)


def save_to_json(file: Path | str = _MODELS_FILE) -> None:
    instance().save_to_json(file)


async def refresh(*, remote_only: bool = False) -> Models:
    return await instance().refresh(remote_only=remote_only)


def resolve(
    model_id: str,
    *,
    provider: str | None = None,
    assume_exists: bool = False,
    config: Configuration | None = None,
) -> tuple[Info, Provider]:
    from . import config as _config
    from .provider import Provider

    cfg = config or _config()
    provider_class = Provider.providers().get(provider) if provider else None
    if provider_class and (provider_class.is_local() or provider_class.assumes_models_exist()):
        assume_exists = True

    if assume_exists:
        if not provider:
            raise ValueError("Provider must be specified if assume_exists is true")
        provider_class = provider_class or Provider.resolve_bang(provider)
        # Prefer the registry entry when one exists (real pricing/capabilities/
        # modalities); fall back to an assumed default so unknown ids still work.
        try:
            model: Info | None = instance().find(model_id, provider)
        except ModelNotFoundError:
            model = None
        if model is None:
            model = Info.default(model_id, provider_class.slug_name())
    else:
        model = instance().find(model_id, provider)
        provider_class = Provider.resolve_bang(model.provider)

    return model, provider_class(cfg)


async def fetch_merged_models(*, remote_only: bool = False) -> list[Info]:
    """Fetch the latest models from configured providers and merge by key.

    A simplified async merge: provider lists win for their own
    providers, existing entries are preserved for providers not fetched.
    """
    from . import config as _config
    from .provider import Provider

    cfg = _config()
    existing = list(instance().all()) or _read_from_json()
    providers = (
        Provider.configured_remote_providers(cfg)
        if remote_only
        else Provider.configured_providers(cfg)
    )
    fetched: list[Info] = []
    fetched_slugs: set[str] = set()
    for provider_class in providers:
        try:
            fetched.extend(await provider_class(cfg).list_models())
            fetched_slugs.add(provider_class.slug_name())
        except Exception:
            continue
    preserved = [m for m in existing if m.provider not in fetched_slugs]
    merged: dict[str, Info] = {f"{m.provider}:{m.id}": m for m in preserved}
    for m in fetched:
        merged[f"{m.provider}:{m.id}"] = m
    return sorted(merged.values(), key=lambda m: (m.provider or "", m.id or ""))
