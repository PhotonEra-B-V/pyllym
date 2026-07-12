from __future__ import annotations

from pyllym.citation import Citation
from pyllym.cost import Cost
from pyllym.model.info import Info
from pyllym.thinking import Thinking
from pyllym.tokens import Tokens
from pyllym.tool_call import ToolCall


def test_tokens_build_returns_none_when_all_empty():
    assert Tokens.build() is None
    t = Tokens.build(input=10, output=5)
    assert t.input == 10
    assert t.cache_read is None
    assert t.to_dict() == {"input_tokens": 10, "output_tokens": 5}


def test_thinking_build_and_redaction():
    assert Thinking.build(text="", signature="") is None
    t = Thinking.build(text="reasoning", signature="secret-abc123")
    assert "[REDACTED]" in repr(t)
    assert "secret-abc123" not in repr(t)


def test_citation_equality_and_roundtrip():
    a = Citation(url="https://x", title="t", start_index=0, end_index=3)
    b = Citation.from_dict(a.to_dict())
    assert a == b
    assert "url" in a.to_dict()


def test_tool_call_to_dict_compacts():
    tc = ToolCall(id="1", name="weather", arguments={"city": "Paris"})
    assert tc.to_dict() == {"id": "1", "name": "weather", "arguments": {"city": "Paris"}}


def test_cost_computation():
    model = Info(
        {
            "id": "demo",
            "provider": "openai",
            "pricing": {
                "text_tokens": {"standard": {"input_per_million": 1.0, "output_per_million": 2.0}}
            },
        }
    )
    cost = Cost(tokens=Tokens.build(input=1_000_000, output=1_000_000), model=model)
    assert cost.input == 1.0
    assert cost.output == 2.0
    assert cost.total == 3.0
