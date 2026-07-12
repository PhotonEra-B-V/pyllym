"""pyllym — one delightful, async-first Python framework for every major AI
provider.

Public façade::

    import pyllym

    pyllym.configure(lambda c: setattr(c, "openai_api_key", "sk-..."))

    chat = pyllym.create_chat(model="gpt-5.4")
    message = await chat.ask("What's the best way to learn Python?")
    print(message.content)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .agent import Agent
from .chat import Chat
from .citation import Citation
from .configuration import Configuration
from .connection import aclose
from .content import Content, RawContent
from .context import Context
from .embedding import Embedding
from .errors import (
    BadRequestError,
    ConfigurationError,
    ContextLengthExceededError,
    Error,
    ForbiddenError,
    InvalidRoleError,
    InvalidToolChoiceError,
    ModelNotFoundError,
    OverloadedError,
    PaymentRequiredError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    UnauthorizedError,
    UnsupportedAttachmentError,
)
from .image import Image
from .message import Message, Role
from .moderation import Moderation
from .provider import Provider
from .search_results import SearchResults
from .speech import Speech
from .thinking import Thinking
from .tokens import Tokens
from .tool import Parameter, Tool
from .tool_call import ToolCall
from .transcription import Transcription
from .uploaded_file import UploadedFile
from .video import Video

__version__ = "1.16.0"

_config: Configuration | None = None


def config() -> Configuration:
    """The global configuration singleton."""
    global _config
    if _config is None:
        _config = Configuration()
    return _config


def configure(block: Callable[[Configuration], Any]) -> Configuration:
    """Configure pyllym: ``pyllym.configure(lambda c: setattr(c, 'openai_api_key', ...))``."""
    cfg = config()
    block(cfg)
    return cfg


def context(block: Callable[[Configuration], Any] | None = None) -> Context:
    """Create a per-call :class:`Context` with an isolated config copy."""
    cfg = config().copy()
    if block is not None:
        block(cfg)
    return Context(cfg)


def create_chat(**kwargs: Any) -> Chat:
    """Create a new conversation with an AI model."""
    return Chat(**kwargs)


async def embed(text: Any, **kwargs: Any) -> Embedding:
    from .embedding import embed as _embed

    return await _embed(text, **kwargs)


async def moderate(input: Any, **kwargs: Any) -> Moderation:
    from .moderation import moderate as _moderate

    return await _moderate(input, **kwargs)


async def paint(prompt: str, **kwargs: Any) -> Image:
    from .image import paint as _paint

    return await _paint(prompt, **kwargs)


async def animate(prompt: str, **kwargs: Any) -> Video:
    from .video import animate as _animate

    return await _animate(prompt, **kwargs)


async def speak(input: str, **kwargs: Any) -> Speech:
    from .speech import speak as _speak

    return await _speak(input, **kwargs)


async def transcribe(audio_file: str, **kwargs: Any) -> Transcription:
    from .transcription import transcribe as _transcribe

    return await _transcribe(audio_file, **kwargs)


async def upload(file: Any, **kwargs: Any) -> UploadedFile:
    from .uploaded_file import upload as _upload

    return await _upload(file, **kwargs)


async def download(id: str, **kwargs: Any) -> bytes:
    from .uploaded_file import download as _download

    return await _download(id, **kwargs)


def list_providers() -> list[type[Provider]]:
    """All registered provider classes.

    Note: ``pyllym.providers`` is the provider *subpackage*; use this function
    or :meth:`Provider.providers` to enumerate registered providers.
    """
    return list(Provider.providers().values())


class _ModelsProxy:
    """Attribute proxy so ``pyllym.models.find(...)`` reaches the registry module
    even though the ``models`` attribute on the package shadows the submodule."""

    def __getattr__(self, name: str) -> Any:
        import importlib

        module = importlib.import_module("pyllym.models")
        return getattr(module, name)


models = _ModelsProxy()


def _register_builtin_providers() -> None:
    """Register provider classes if their modules import cleanly.

    Providers depend on protocol modules that may be filled in incrementally;
    importing defensively keeps the core usable as coverage grows.
    """
    registrations = [
        ("anthropic", "Anthropic"),
        ("openai", "OpenAI"),
        ("gemini", "Gemini"),
        ("deepseek", "DeepSeek"),
        ("mistral", "Mistral"),
        ("ollama", "Ollama"),
        ("openrouter", "OpenRouter"),
        ("perplexity", "Perplexity"),
        ("xai", "XAI"),
        ("gpustack", "GPUStack"),
        ("vllm", "VLLM"),
        ("azure", "Azure"),
        ("bedrock", "Bedrock"),
        ("vertexai", "VertexAI"),
        ("nvidia", "NVIDIA"),
        ("cerebras", "Cerebras"),
        ("huggingface", "HuggingFace"),
        ("databricks", "Databricks"),
        ("qwen", "Qwen"),
        ("zhipu", "Zhipu"),
        ("moonshot", "Moonshot"),
        ("doubao", "Doubao"),
        ("ernie", "ERNIE"),
        ("minimax", "MiniMax"),
        ("fal", "Fal"),
    ]
    import importlib

    for slug, class_name in registrations:
        try:
            module = importlib.import_module(f".providers.{slug}", __name__)
            provider_class = getattr(module, class_name)
        except Exception:
            continue
        Provider.register(slug, provider_class)


_register_builtin_providers()

__all__ = [
    "Agent",
    "BadRequestError",
    "Chat",
    "Citation",
    "Configuration",
    "ConfigurationError",
    "Content",
    "Context",
    "ContextLengthExceededError",
    "Embedding",
    "Error",
    "ForbiddenError",
    "Image",
    "InvalidRoleError",
    "InvalidToolChoiceError",
    "Message",
    "ModelNotFoundError",
    "Moderation",
    "OverloadedError",
    "Parameter",
    "PaymentRequiredError",
    "Provider",
    "RateLimitError",
    "RawContent",
    "Role",
    "SearchResults",
    "ServerError",
    "ServiceUnavailableError",
    "Speech",
    "Thinking",
    "Tokens",
    "Tool",
    "ToolCall",
    "Transcription",
    "UnauthorizedError",
    "UnsupportedAttachmentError",
    "UploadedFile",
    "Video",
    "__version__",
    "aclose",
    "animate",
    "config",
    "configure",
    "context",
    "create_chat",
    "download",
    "embed",
    "list_providers",
    "models",
    "moderate",
    "paint",
    "speak",
    "transcribe",
    "upload",
]
