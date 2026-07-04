from __future__ import annotations

import httpx
import pytest
import respx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from pyllm.persistence import create_models


class Base(DeclarativeBase):
    pass


Chat, Message, ToolCall = create_models(Base)


@pytest.mark.asyncio
@respx.mock
async def test_chat_persistence_roundtrip():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Persisted reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        record = Chat(model_id="gpt-4o", provider="openai")
        session.add(record)
        await session.commit()

        chat = record.to_chat(session, assume_model_exists=True)
        await chat.ask("Hello there")
        await session.commit()

        rows = (await session.execute(select(Message).order_by(Message.id))).scalars().all()
        assert [(r.role, r.content) for r in rows] == [
            ("user", "Hello there"),
            ("assistant", "Persisted reply"),
        ]
        assert rows[1].output_tokens == 4
    await engine.dispose()
