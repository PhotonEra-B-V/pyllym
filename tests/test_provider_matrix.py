from __future__ import annotations

import httpx
import pytest
import respx

import pyllm

from . import factories as f

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
    with respx.mock:
        route = respx.post(f"{base}/chat/completions").mock(
            return_value=httpx.Response(200, json=f.openai_chat(f"reply from {provider}"))
        )
        chat = pyllm.create_chat(model="some-model", provider=provider, assume_model_exists=True)
        msg = await chat.ask("hi")
        assert route.called
        assert msg.content == f"reply from {provider}"
        assert msg.input_tokens == 10


@pytest.mark.parametrize("provider,base", sorted(OPENAI_COMPATIBLE.items()))
@pytest.mark.asyncio
async def test_openai_compatible_tool_loop(provider: str, base: str):
    class Weather(pyllm.Tool):
        description = "weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    with respx.mock:
        calls = {"n": 0}

        def responder(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    200,
                    json=f.openai_chat(tool_calls=f.openai_tool_call("weather", {"city": "Rome"})),
                )
            return httpx.Response(200, json=f.openai_chat("Rome is sunny."))

        respx.post(f"{base}/chat/completions").mock(side_effect=responder)
        chat = pyllm.create_chat(
            model="some-model", provider=provider, assume_model_exists=True
        ).with_tool(Weather)
        msg = await chat.ask("weather in Rome?")
        assert msg.content == "Rome is sunny."
        assert [m.role for m in chat.messages] == ["user", "assistant", "tool", "assistant"]
