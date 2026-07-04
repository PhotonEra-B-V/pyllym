from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel

import pyllm
from pyllm import Agent, Message, Tool, utils
from pyllm.chunk import Chunk
from pyllm.cost import Cost
from pyllm.model.info import Info
from pyllm.stream_accumulator import StreamAccumulator
from pyllm.tokens import Tokens
from pyllm.tool_call import ToolCall


# --- structured output ---------------------------------------------------------
@pytest.mark.asyncio
@respx.mock
async def test_structured_output_parses_json_and_sends_schema():
    class Recipe(BaseModel):
        title: str
        steps: list[str]

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"title": "Toast", "steps": ["toast bread"]}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    chat = pyllm.create_chat(model="gpt-4o").with_schema(Recipe)
    msg = await chat.ask("recipe?")
    assert msg.content == {"title": "Toast", "steps": ["toast bread"]}
    assert b'"json_schema"' in route.calls.last.request.content


# --- cost ----------------------------------------------------------------------
def _priced_model() -> Info:
    return Info(
        {
            "id": "demo",
            "provider": "openai",
            "pricing": {
                "text_tokens": {"standard": {"input_per_million": 1.0, "output_per_million": 2.0}}
            },
        }
    )


def test_cost_aggregate_sums_messages():
    model = _priced_model()
    one = Cost(tokens=Tokens.build(input=1_000_000, output=1_000_000), model=model)
    agg = Cost.aggregate([one, one])
    assert agg.input == 2.0
    assert agg.total == 6.0


def test_cost_total_none_without_pricing():
    model = Info({"id": "unpriced", "provider": "openai"})
    cost = Cost(tokens=Tokens.build(input=100, output=100), model=model)
    assert cost.total is None


# --- message -------------------------------------------------------------------
def test_message_predicates_and_to_dict():
    call = Message(role="assistant", tool_calls={"c1": ToolCall(id="c1", name="x", arguments={})})
    assert call.is_tool_call()
    result = Message(role="tool", content="done", tool_call_id="c1")
    assert result.is_tool_result()
    plain = Message(role="assistant", content="hi", finish_reason="stop")
    assert plain.is_stopped()
    d = plain.to_dict()
    assert d["role"] == "assistant" and d["content"] == "hi"


# --- stream accumulator --------------------------------------------------------
def test_stream_accumulator_splits_think_tags():
    acc = StreamAccumulator()
    acc.add(Chunk(role="assistant", content="Hello "))
    acc.add(Chunk(role="assistant", content="<think>because</think>world"))
    msg = acc.to_message(None)
    assert msg.content == "Hello world"
    assert msg.thinking.text == "because"


def test_stream_accumulator_assembles_tool_call_fragments():
    acc = StreamAccumulator()
    acc.add(
        Chunk(role="assistant", tool_calls={0: ToolCall(id="c1", name="weather", arguments="")})
    )
    acc.add(
        Chunk(role="assistant", tool_calls={0: ToolCall(id=None, name=None, arguments='{"city":')})
    )
    acc.add(
        Chunk(role="assistant", tool_calls={0: ToolCall(id=None, name=None, arguments='"Paris"}')})
    )
    msg = acc.to_message(None)
    assert msg.is_tool_call()
    tc = next(iter(msg.tool_calls.values()))
    assert tc.name == "weather" and tc.arguments == {"city": "Paris"}


# --- utils ---------------------------------------------------------------------
def test_utils_to_safe_array_and_dates():
    assert utils.to_safe_array(None) == []
    assert utils.to_safe_array("x") == ["x"]
    assert utils.to_safe_array(["a", "b"]) == ["a", "b"]
    assert utils.parse_iso_date_prefix("2025").isoformat() == "2025-01-01"
    assert utils.parse_iso_date_prefix("2025-06").isoformat() == "2025-06-01"


# --- agent ---------------------------------------------------------------------
def test_agent_applies_configuration():
    class Adder(Tool):
        description = "add"

        def execute(self, *, a: int):
            return a

    class Helper(Agent):
        chat_model = "gpt-4o"
        instructions = "be concise"
        tools = [Adder]
        temperature = 0.3

    chat = Helper.create_chat()
    assert chat.model.id == "gpt-4o"
    assert any(m.role == "system" for m in chat.messages)
    assert "adder" in chat.tools
    assert chat._temperature == 0.3


# --- render payloads (no network) ----------------------------------------------
def test_render_openai_includes_media_parts():
    chat = pyllm.create_chat(model="gpt-4o")
    chat.add_user_message("look", with_="https://example.com/cat.png")
    payload = chat.render()
    content = payload["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_render_anthropic_shape():
    chat = pyllm.create_chat(model="claude-sonnet-4-6")
    chat.add_user_message("hi")
    payload = chat.render()
    assert payload["model"] and payload["max_tokens"]
    assert payload["messages"][0]["role"] == "user"


def test_render_gemini_shape():
    chat = pyllm.create_chat(model="gemini-2.5-flash")
    chat.add_user_message("hi")
    payload = chat.render()
    assert "contents" in payload
