"""Optional async SQLAlchemy persistence for chats, messages, and tool calls.

Requires the ``db`` extra (``pip install pyllym[db]``).

:func:`create_models` is a small factory that builds async SQLAlchemy ORM
classes bound to your declarative ``Base``::

    from sqlalchemy.orm import DeclarativeBase
    from pyllym.persistence import create_models

    class Base(DeclarativeBase): ...

    Chat, Message, ToolCall = create_models(Base)

    # after creating tables and a row:
    record = Chat(model_id="gpt-4o", provider="openai")
    session.add(record); await session.commit()

    chat = record.to_chat(session)       # a pyllym.Chat backed by the DB
    await chat.ask("Hello!")             # user + assistant rows persisted
"""

from __future__ import annotations

from .models import create_models

__all__ = ["create_models"]
