"""Tool / function-calling loop against local Gemma 4.

Gemma 4 in Ollama advertises the ``tools`` capability, so the model can call
this ``Weather`` tool and pyllym runs the full model -> tool -> model loop.

    python examples/tools.py
"""

from __future__ import annotations

import asyncio

import _bootstrap as boot

from pyllym import Tool


class Weather(Tool):
    description = "Look up the current weather for a city"

    async def execute(self, *, city: str, units: str = "celsius") -> str:
        # Canned data — swap in a real API if you like.
        return f"18°{units[0].upper()} and clear in {city}"


async def main() -> None:
    boot.setup()
    chat = boot.chat().with_tools(Weather)
    answer = await chat.ask("What's the weather in Oslo?")
    print(answer.content)
    print("\nconversation roles:", [m.role for m in chat.messages])


if __name__ == "__main__":
    asyncio.run(main())
