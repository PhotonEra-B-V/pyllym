# examples

Small, runnable scripts for poking at `pyllm` from the terminal against a
**local Gemma 4 model served by Ollama**.

## Prerequisites

- Ollama running with Gemma 4 pulled:
  ```bash
  ollama pull gemma4          # or gemma4:31b
  ollama serve                # usually already running
  ```
- `pyllm` importable. From the repo root, either install it editable:
  ```bash
  pip install -e ".[dev]"
  ```
  or just run the scripts from the repo root (they add `src/` to the path).

## Run

```bash
python examples/chat.py "Explain async/await in one sentence"
python examples/stream.py "Write a haiku about the terminal"
python examples/structured.py
python examples/tools.py
```

Override the model or endpoint with env vars:

```bash
GEMMA_MODEL=gemma4:31b python examples/chat.py "hi"
OLLAMA_BASE=http://localhost:11434/v1 python examples/chat.py "hi"
```

`_bootstrap.py` centralizes the Ollama config so each script stays tiny.
