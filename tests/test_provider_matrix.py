from __future__ import annotations

import pytest
from aioresponses import CallbackResult, aioresponses

import pyllym

from . import factories as f
from .conftest import sent_requests

# provider slug -> chat/completions base URL
OPENAI_COMPATIBLE = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "perplexity": "https://api.perplexity.ai",
    "openrouter": "https://openrouter.ai/api/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://api.z.ai/api/paas/v4",
    "moonshot": "https://api.moonshot.ai/v1",
    "minimax": "https://api.minimax.io/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "ernie": "https://qianfan.baidubce.com/v2",
    "databricks": "https://ws.cloud.databricks.com/serving-endpoints",
}


@pytest.mark.parametrize("provider,base", sorted(OPENAI_COMPATIBLE.items()))
@pytest.mark.asyncio
async def test_openai_compatible_chat(provider: str, base: str):
    with aioresponses() as m:
        m.post(f"{base}/chat/completions", payload=f.openai_chat(f"reply from {provider}"))
        chat = pyllym.create_chat(model="some-model", provider=provider, assume_model_exists=True)
        msg = await chat.ask("hi")
        assert sent_requests(m)
        assert msg.content == f"reply from {provider}"
        assert msg.input_tokens == 10


@pytest.mark.parametrize("provider,base", sorted(OPENAI_COMPATIBLE.items()))
@pytest.mark.asyncio
async def test_openai_compatible_tool_loop(provider: str, base: str):
    class Weather(pyllym.Tool):
        description = "weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    with aioresponses() as m:
        calls = {"n": 0}

        def responder(url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return CallbackResult(
                    payload=f.openai_chat(
                        tool_calls=f.openai_tool_call("weather", {"city": "Rome"})
                    )
                )
            return CallbackResult(payload=f.openai_chat("Rome is sunny."))

        m.post(f"{base}/chat/completions", callback=responder, repeat=True)
        chat = pyllym.create_chat(
            model="some-model", provider=provider, assume_model_exists=True
        ).with_tool(Weather)
        msg = await chat.ask("weather in Rome?")
        assert msg.content == "Rome is sunny."
        assert [m2.role for m2 in chat.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_vllm_local_chat():
    """vLLM is self-hosted: base is required config, auth optional."""
    base = "http://localhost:8000/v1"
    pyllym.config().vllm_api_base = base
    with aioresponses() as m:
        m.post(f"{base}/chat/completions", payload=f.openai_chat("reply from vllm"))
        chat = pyllym.create_chat(model="meta-llama/Llama-3.1-8B-Instruct", provider="vllm")
        msg = await chat.ask("hi")
        requests = sent_requests(m)
        assert requests
        assert msg.content == "reply from vllm"
        # No api key configured -> no Authorization header sent.
        headers = requests[-1].kwargs.get("headers") or {}
        assert "authorization" not in {k.lower() for k in headers}
