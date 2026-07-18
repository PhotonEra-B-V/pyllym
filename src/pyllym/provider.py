"""Base class for LLM providers.

A provider knows *where* to talk (host, auth, configuration) and *which*
protocol to speak for a given model/request. Subclasses declare protocols via
the ``protocols`` / ``default_protocol_name`` / ``file_protocol`` class
attributes and the ``batches`` mixin map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .connection import Connection, parse_error_body
from .errors import ConfigurationError, Error

if TYPE_CHECKING:
    from .configuration import Configuration
    from .message import Message
    from .model.info import Info
    from .protocol import Protocol

# global provider registry (slug -> Provider subclass)
_REGISTRY: dict[str, type[Provider]] = {}


class Provider:
    # Declared by subclasses:
    protocols: ClassVar[dict[str, type[Protocol]]] = {}
    default_protocol_name: ClassVar[str | None] = None
    file_protocol: ClassVar[type[Protocol] | None] = None
    batches: ClassVar[dict[str, type]] = {}  # protocol name -> batch mixin

    _slug: ClassVar[str | None] = None
    _batch_cache: ClassVar[dict[str, type[Protocol]]]

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self._ensure_configured()
        self.connection = Connection(self, config)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._batch_cache = {}
        if cls.default_protocol_name is None and cls.protocols:
            cls.default_protocol_name = next(iter(cls.protocols))

    # --- identity --------------------------------------------------------------
    @property
    def api_base(self) -> str:
        raise NotImplementedError

    @property
    def headers(self) -> dict[str, str]:
        return {}

    @property
    def slug(self) -> str:
        return type(self).slug_name()

    @property
    def name(self) -> str:
        return type(self).display_name()

    @property
    def capabilities(self) -> Any:
        return type(self).capabilities_cls()

    @property
    def configuration_requirements(self) -> list[str]:
        return type(self).configuration_requirements_list()

    # --- routing ---------------------------------------------------------------
    def protocol_for(self, model: Info | None, **request: Any) -> type[Protocol]:
        return self._default_protocol()

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Any],
        temperature: float | None,
        model: Info,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
        protocol: str | None = None,
        on_chunk: Any = None,
    ) -> Message:
        protocol_class = self._resolve_protocol(protocol, model)
        return await protocol_class(self, model).complete(
            messages,
            tools=tools,
            tool_prefs=tool_prefs,
            temperature=temperature,
            params=params or {},
            headers=headers,
            schema=schema,
            thinking=thinking,
            citations=citations,
            on_chunk=on_chunk,
        )

    def render(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Any],
        temperature: float | None,
        model: Info,
        params: dict[str, Any] | None = None,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
        protocol: str | None = None,
    ) -> dict[str, Any]:
        protocol_class = self._resolve_protocol(protocol, model)
        return protocol_class(self, model).render(
            messages,
            tools=tools,
            tool_prefs=tool_prefs,
            temperature=temperature,
            params=params or {},
            schema=schema,
            thinking=thinking,
            citations=citations,
        )

    def preprocess_message(
        self, message: Message, *, model: Info, protocol: str | None = None
    ) -> Message:
        protocol_class = self._resolve_protocol(protocol, model)
        return protocol_class(self, model).preprocess_message(message)

    async def list_models(self) -> list[Info]:
        return await self._default_protocol()(self).list_models()

    async def embed(self, text: Any, *, model: str, dimensions: int | None) -> Any:
        return await self._default_protocol()(self).embed(text, model=model, dimensions=dimensions)

    async def moderate(self, input: Any, *, model: str) -> Any:
        return await self._default_protocol()(self).moderate(input, model=model)

    async def paint(
        self,
        prompt: str,
        *,
        model: str,
        size: str,
        with_: Any = None,
        mask: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        return await self._default_protocol()(self).paint(
            prompt, model=model, size=size, with_=with_, mask=mask, params=params or {}
        )

    async def animate(
        self, prompt: str, *, model: str, params: dict[str, Any] | None = None
    ) -> Any:
        return await self._default_protocol()(self).animate(
            prompt, model=model, params=params or {}
        )

    async def speak(
        self,
        input: str,
        *,
        model: str,
        voice: str | None,
        format: str | None,
        params: dict[str, Any] | None = None,
        **options: Any,
    ) -> Any:
        return await self._default_protocol()(self).speak(
            input, model=model, voice=voice, format=format, params=params or {}, **options
        )

    async def transcribe(
        self, audio_file: str, *, model: str, language: str | None, **options: Any
    ) -> Any:
        return await self._default_protocol()(self).transcribe(
            audio_file, model=model, language=language, **options
        )

    def files_supported(self) -> bool:
        return self.file_protocol is not None

    async def upload_file(self, file: Any, **options: Any) -> Any:
        raise NotImplementedError

    async def find_file(self, id: str) -> Any:
        raise NotImplementedError

    async def download_file(self, id: str) -> bytes:
        raise NotImplementedError

    def sign_headers(self, method: str, url: str, body: str) -> dict[str, str]:
        raise NotImplementedError

    def model_path(self, model: str, *, publisher: str = "google") -> str:
        raise NotImplementedError

    # --- config / errors -------------------------------------------------------
    def configured(self) -> bool:
        return type(self).is_configured(self.config)

    def local(self) -> bool:
        return type(self).is_local()

    def assume_models_exist(self) -> bool:
        return type(self).assumes_models_exist()

    def parse_error(self, response: Any) -> Any:
        body = getattr(response, "body", None)
        if body is None or body == "":
            return None
        return parse_error_body(body)

    # --- class-level configuration hooks ---------------------------------------
    @classmethod
    def slug_name(cls) -> str:
        return cls._slug or cls.__name__.lower()

    @classmethod
    def display_name(cls) -> str:
        return cls.__name__

    @classmethod
    def capabilities_cls(cls) -> Any:
        return None

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return []

    @classmethod
    def configuration_options(cls) -> list[str]:
        return []

    @classmethod
    def is_local(cls) -> bool:
        return False

    @classmethod
    def is_remote(cls) -> bool:
        return not cls.is_local()

    @classmethod
    def assumes_models_exist(cls) -> bool:
        return False

    @classmethod
    def uses_developer_role(cls) -> bool:
        """Whether system messages should be sent as OpenAI's ``developer`` role.

        Only the OpenAI API itself understands ``developer``; OpenAI-compatible
        servers mostly reject or ignore it, so everyone else keeps ``system``.
        """
        return False

    @classmethod
    def is_configured(cls, config: Configuration) -> bool:
        return all(getattr(config, req, None) for req in cls.configuration_requirements_list())

    # --- global registry -------------------------------------------------------
    @classmethod
    def register(cls, name: str, provider_class: type[Provider]) -> None:
        provider_class._slug = name
        _REGISTRY[name] = provider_class
        from . import config as _config

        keys = list(provider_class.configuration_options()) + [f"{name}_protocol"]
        _config().register_provider_options(keys)

    @staticmethod
    def resolve(name: str) -> type[Provider] | None:
        return _REGISTRY.get(name)

    @staticmethod
    def resolve_bang(name: str) -> type[Provider]:
        provider = _REGISTRY.get(name)
        if provider is None:
            available = ", ".join(_REGISTRY)
            raise Error(f"Unknown provider: {name!r}. Available providers: {available}")
        return provider

    @staticmethod
    def providers() -> dict[str, type[Provider]]:
        return dict(_REGISTRY)

    @classmethod
    def configured_providers(cls, config: Configuration) -> list[type[Provider]]:
        return [p for p in _REGISTRY.values() if p.is_configured(config)]

    @classmethod
    def configured_remote_providers(cls, config: Configuration) -> list[type[Provider]]:
        return [p for p in _REGISTRY.values() if p.is_remote() and p.is_configured(config)]

    # --- internals -------------------------------------------------------------
    def _resolve_protocol(
        self, name: str | None, model: Info | None, **request: Any
    ) -> type[Protocol]:
        explicit = name or self._configured_protocol()
        if explicit:
            return self._fetch_protocol(explicit)
        return self.protocol_for(model, **request)

    def _default_protocol(self) -> type[Protocol]:
        return self._fetch_protocol(self._configured_protocol() or self.default_protocol_name)

    def _configured_protocol(self) -> str | None:
        return getattr(self.config, f"{self.slug}_protocol", None)

    def _fetch_protocol(self, name: str | None) -> type[Protocol]:
        if name not in self.protocols:
            available = ", ".join(self.protocols)
            raise Error(
                f"{name} is not a protocol of {type(self).display_name()}. Available: {available}"
            )
        return self.protocols[name]

    def batch_protocol_for_name(self, name: str) -> type[Protocol] | None:
        if name in self._batch_cache:
            return self._batch_cache[name]
        mixin = self.batches.get(name)
        if mixin is None:
            return None
        base = self.protocols[name]
        built = type(f"{base.__name__}Batch", (mixin, base), {})
        self._batch_cache[name] = built
        return built

    def _ensure_configured(self) -> None:
        if self.configured():
            return
        missing = [
            req for req in self.configuration_requirements if not getattr(self.config, req, None)
        ]
        lines = "\n  ".join(f"config.{key} = os.environ['{key.upper()}']" for key in missing)
        raise ConfigurationError(
            f"{self.name} provider is not configured. Add this to your initialization:\n\n"
            f"import pyllym\npyllym.configure(lambda config: (\n  {lines}\n))"
        )
