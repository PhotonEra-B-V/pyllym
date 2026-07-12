"""Class-configured agents.

Subclasses declare their configuration (model, instructions, tools,
temperature, …) as class attributes and produce a ready-to-use
:class:`~pyllym.chat.Chat`. Persistence is handled separately by
:mod:`pyllym.persistence`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .chat import Chat
from .tool import Tool


class Agent:
    """Subclass and set class attributes to configure a reusable chat.

    Example::

        class Researcher(Agent):
            chat_model = "claude-sonnet-4-6"
            instructions = "You are a meticulous research assistant."
            tools = [WebSearch]
            temperature = 0.2

        chat = Researcher.create_chat()
        await chat.ask("Summarize the latest on ...")
    """

    chat_model: ClassVar[str | None] = None
    chat_kwargs: ClassVar[dict[str, Any]] = {}
    instructions: ClassVar[str | None] = None
    tools: ClassVar[list[type[Tool] | Tool]] = []
    temperature: ClassVar[float | None] = None
    thinking: ClassVar[dict[str, Any] | None] = None
    citations: ClassVar[bool | None] = None
    params: ClassVar[dict[str, Any]] = {}
    headers: ClassVar[dict[str, str]] = {}
    schema: ClassVar[Any] = None
    context: ClassVar[Any] = None

    @classmethod
    def create_chat(cls, **chat_options: Any) -> Chat:
        """Create a conversation pre-configured by this agent's class attributes."""
        kwargs = dict(cls.chat_kwargs)
        if cls.chat_model and "model" not in kwargs:
            kwargs["model"] = cls.chat_model
        kwargs.update(chat_options)
        if cls.context is not None and "context" not in kwargs:
            kwargs["context"] = cls.context
        chat = Chat(**kwargs)
        cls._configure(chat)
        return chat

    @classmethod
    def _configure(cls, chat: Chat) -> None:
        if cls.instructions:
            chat.with_instructions(cls.instructions)
        if cls.tools:
            chat.with_tools(*cls.tools)
        if cls.temperature is not None:
            chat.with_temperature(cls.temperature)
        if cls.thinking:
            chat.with_thinking(effort=cls.thinking.get("effort"), budget=cls.thinking.get("budget"))
        if cls.citations is not None:
            chat.with_citations(cls.citations)
        if cls.params:
            chat.with_params(**cls.params)
        if cls.headers:
            chat.with_headers(**cls.headers)
        if cls.schema is not None:
            chat.with_schema(cls.schema)
