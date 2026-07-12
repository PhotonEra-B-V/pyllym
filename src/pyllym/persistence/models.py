"""Async SQLAlchemy model factory for chat/message/tool-call persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..chat import Chat as LLMChat
from ..citation import Citation
from ..message import Message as LLMMessage
from ..thinking import Thinking
from ..tokens import Tokens
from ..tool_call import ToolCall as LLMToolCall

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


def create_models(Base: type, *, table_prefix: str = "llm_") -> tuple[type, type, type]:
    """Build ``(Chat, Message, ToolCall)`` ORM classes bound to ``Base``.

    Tables are named ``{table_prefix}chats``, ``..._messages``, ``..._tool_calls``.
    """

    class ToolCall(Base):
        __tablename__ = f"{table_prefix}tool_calls"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        message_id: Mapped[int | None] = mapped_column(
            ForeignKey(f"{table_prefix}messages.id", ondelete="CASCADE"), nullable=True
        )
        tool_call_id: Mapped[str] = mapped_column(String(255))
        name: Mapped[str] = mapped_column(String(255))
        arguments: Mapped[Any] = mapped_column(JSON, default=dict)

        def to_llm(self) -> LLMToolCall:
            return LLMToolCall(id=self.tool_call_id, name=self.name, arguments=self.arguments or {})

    class Message(Base):
        __tablename__ = f"{table_prefix}messages"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        chat_id: Mapped[int] = mapped_column(
            ForeignKey(f"{table_prefix}chats.id", ondelete="CASCADE")
        )
        role: Mapped[str] = mapped_column(String(32))
        content: Mapped[str | None] = mapped_column(Text, nullable=True)
        model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
        tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
        input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
        output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
        cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
        cache_creation_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
        thinking_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
        thinking_text: Mapped[str | None] = mapped_column(Text, nullable=True)
        thinking_signature: Mapped[str | None] = mapped_column(Text, nullable=True)
        citations: Mapped[Any | None] = mapped_column(JSON, nullable=True)
        finish_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

        tool_calls: Mapped[list[ToolCall]] = relationship(
            ToolCall, cascade="all, delete-orphan", lazy="selectin"
        )

        def to_llm(self) -> LLMMessage:
            content = self.content
            if isinstance(content, str) and content[:1] in "{[":
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass
            tool_calls = {tc.tool_call_id: tc.to_llm() for tc in self.tool_calls} or None
            return LLMMessage(
                role=self.role,
                content=content,
                model_id=self.model_id,
                tool_call_id=self.tool_call_id,
                tool_calls=tool_calls,
                tokens=Tokens.build(
                    input=self.input_tokens,
                    output=self.output_tokens,
                    cached=self.cached_tokens,
                    cache_creation=self.cache_creation_tokens,
                    thinking=self.thinking_tokens,
                ),
                thinking=Thinking.build(text=self.thinking_text, signature=self.thinking_signature),
                citations=[Citation.from_dict(c) for c in (self.citations or [])],
                finish_reason=self.finish_reason,
            )

        @classmethod
        def from_llm(cls, message: LLMMessage, *, chat_id: int) -> Message:
            content = message.content
            if not isinstance(content, str) and content is not None:
                content = json.dumps(content)
            record = cls(
                chat_id=chat_id,
                role=str(message.role),
                content=content,
                model_id=message.model_id,
                tool_call_id=message.tool_call_id,
                input_tokens=message.input_tokens,
                output_tokens=message.output_tokens,
                cached_tokens=message.cached_tokens,
                cache_creation_tokens=message.cache_creation_tokens,
                thinking_tokens=message.thinking_tokens,
                thinking_text=message.thinking.text if message.thinking else None,
                thinking_signature=message.thinking.signature if message.thinking else None,
                citations=[c.to_dict() for c in message.citations] or None,
                finish_reason=message.finish_reason,
            )
            for tc in (message.tool_calls or {}).values():
                record.tool_calls.append(
                    ToolCall(tool_call_id=tc.id, name=tc.name, arguments=tc.arguments)
                )
            return record

    class Chat(Base):
        __tablename__ = f"{table_prefix}chats"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        model_id: Mapped[str] = mapped_column(String(255))
        provider: Mapped[str | None] = mapped_column(String(64), nullable=True)

        messages: Mapped[list[Message]] = relationship(
            Message, cascade="all, delete-orphan", lazy="selectin", order_by=Message.id
        )

        def to_chat(
            self,
            session: AsyncSession,
            *,
            assume_model_exists: bool = False,
            context: Any = None,
        ) -> LLMChat:
            """Return a :class:`pyllym.Chat` backed by this record.

            New messages produced by ``ask``/``run_until_done`` are persisted
            to ``session`` automatically (you still ``await session.commit()``).
            """
            from sqlalchemy import inspect as sa_inspect

            chat = LLMChat(
                model=self.model_id,
                provider=self.provider,
                assume_model_exists=assume_model_exists,
                context=context,
            )
            # Only read the relationship if it was eagerly loaded; a freshly
            # created record has no history and must not trigger sync lazy IO.
            loaded = "messages" not in sa_inspect(self).unloaded
            history = list(self.messages) if loaded else []
            chat.messages = [m.to_llm() for m in history]
            persisted = {id(m) for m in chat.messages}

            async def _persist() -> None:
                for message in chat.messages:
                    if id(message) in persisted:
                        continue
                    persisted.add(id(message))
                    session.add(Message.from_llm(message, chat_id=self.id))
                await session.flush()

            # _persist iterates all not-yet-saved messages, so the staged user
            # message is written alongside the first assistant reply.
            chat.after_message(lambda *_: _persist())
            return chat

    Chat.MessageModel = Message
    Chat.ToolCallModel = ToolCall
    return Chat, Message, ToolCall
