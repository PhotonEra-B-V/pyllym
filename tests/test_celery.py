from __future__ import annotations

import pytest

celery = pytest.importorskip("celery")

from pyllm.celery import create_tasks, run_async  # noqa: E402

from .conftest import sent_requests  # noqa: E402


@pytest.fixture
def tasks():
    app = celery.Celery("pyllm-tests")
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    return create_tasks(app)


def test_task_names_and_options():
    app = celery.Celery("pyllm-tests")
    tasks = create_tasks(app, name_prefix="myapp.llm", queue="llm")
    assert tasks.ask.name == "myapp.llm.ask"
    assert tasks.embed.name == "myapp.llm.embed"
    assert tasks.ask.queue == "llm"
    assert set(app.tasks) >= {
        f"myapp.llm.{name}" for name in ("ask", "embed", "paint", "speak", "transcribe", "moderate")
    }


def test_ask_task(tasks, mock_http):
    mock_http.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "id": "x",
            "model": "gpt-4o",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    result = tasks.ask.delay("Hello", model="gpt-4o", temperature=0.2).get()
    assert result["role"] == "assistant"
    assert result["content"] == "Hi!"
    assert result["input_tokens"] == 10


def test_ask_task_with_prior_messages(tasks, mock_http):
    mock_http.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "model": "gpt-4o",
            "choices": [
                {"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}
            ],
        },
    )
    result = tasks.ask.delay(
        "And its capital?",
        model="gpt-4o",
        instructions="Answer tersely.",
        messages=[
            {"role": "user", "content": "Pick a country"},
            {"role": "assistant", "content": "France"},
        ],
    ).get()
    assert result["content"] == "Paris"

    sent = sent_requests(mock_http)[0].kwargs["json"]
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["developer", "user", "assistant", "user"]  # system -> developer on the wire


def test_embed_task(tasks, mock_http):
    mock_http.post(
        "https://api.openai.com/v1/embeddings",
        payload={
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 3},
        },
    )
    result = tasks.embed.delay("hello", model="text-embedding-3-small").get()
    assert result["vectors"] == [0.1, 0.2]
    assert result["input_tokens"] == 3


def test_moderate_task(tasks, mock_http):
    mock_http.post(
        "https://api.openai.com/v1/moderations",
        payload={
            "id": "modr-1",
            "model": "omni-moderation-latest",
            "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
        },
    )
    result = tasks.moderate.delay("some text", model="omni-moderation-latest").get()
    assert result["flagged"] is False
    assert result["model"] == "omni-moderation-latest"


def test_run_async_returns_value():
    async def coro():
        return 42

    assert run_async(coro()) == 42
