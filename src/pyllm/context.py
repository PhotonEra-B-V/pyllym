"""Per-call configuration scope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat import Chat
    from .configuration import Configuration


class Context:
    def __init__(self, config: Configuration) -> None:
        self.config = config

    def create_chat(self, **kwargs: Any) -> Chat:
        """Create a new conversation scoped to this context's config."""
        from .chat import Chat

        return Chat(context=self, **kwargs)

    async def embed(self, text: Any, **kwargs: Any) -> Any:
        from .embedding import embed

        return await embed(text, context=self, **kwargs)

    async def paint(self, prompt: str, **kwargs: Any) -> Any:
        from .image import paint

        return await paint(prompt, context=self, **kwargs)

    async def animate(self, prompt: str, **kwargs: Any) -> Any:
        from .video import animate

        return await animate(prompt, context=self, **kwargs)

    async def moderate(self, input: Any, **kwargs: Any) -> Any:
        from .moderation import moderate

        return await moderate(input, context=self, **kwargs)

    async def speak(self, input: str, **kwargs: Any) -> Any:
        from .speech import speak

        return await speak(input, context=self, **kwargs)

    async def transcribe(self, audio_file: str, **kwargs: Any) -> Any:
        from .transcription import transcribe

        return await transcribe(audio_file, context=self, **kwargs)

    async def upload(self, file: Any, **kwargs: Any) -> Any:
        from .uploaded_file import upload

        return await upload(file, context=self, **kwargs)

    async def download(self, id: str, **kwargs: Any) -> Any:
        from .uploaded_file import download

        return await download(id, context=self, **kwargs)
