"""Shared setup for the example scripts.

Points pyllym at a local Ollama server and exposes the Gemma 4 model id. Import
``MODEL`` and call ``setup()`` at the top of each example.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running straight from the repo without installing (``python examples/x.py``).
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pyllym  # noqa: E402

MODEL = os.environ.get("GEMMA_MODEL", "gemma4:latest")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434/v1")


def setup() -> None:
    """Configure pyllym to talk to the local Ollama server."""
    pyllym.configure(lambda c: setattr(c, "ollama_api_base", OLLAMA_BASE))


def chat(model: str | None = None):
    """A fresh chat bound to a local Ollama model (defaults to Gemma)."""
    return pyllym.create_chat(model=model or MODEL, provider="ollama")
