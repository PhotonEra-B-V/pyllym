from __future__ import annotations

import pytest

import pyllm

from .seed_fixtures import DATA_DIR, seed_all

_KEYS = {
    "openai_api_key": "sk-test",
    "anthropic_api_key": "sk-ant-test",
    "gemini_api_key": "gm-test",
    "deepseek_api_key": "ds-test",
    "mistral_api_key": "ms-test",
    "xai_api_key": "xai-test",
    "perplexity_api_key": "px-test",
    "openrouter_api_key": "or-test",
    "qwen_api_key": "qw-test",
    "zhipu_api_key": "zp-test",
    "moonshot_api_key": "mn-test",
    "minimax_api_key": "mm-test",
    "nvidia_api_key": "nv-test",
    "cerebras_api_key": "cb-test",
    "huggingface_api_key": "hf-test",
    "doubao_api_key": "db-test",
    "ernie_api_key": "er-test",
    "databricks_api_key": "dbx-test",
    "databricks_api_base": "https://ws.cloud.databricks.com/serving-endpoints",
    "bedrock_api_key": "AKIA-test",
    "bedrock_secret_key": "secret-test",
    "bedrock_region": "us-east-1",
}


@pytest.fixture(autouse=True)
def _configure_keys():
    """Give every provider dummy credentials so construction succeeds."""
    cfg = pyllm.config()
    for key, value in _KEYS.items():
        setattr(cfg, key, value)
    yield


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures():
    """Regenerate the on-disk fixtures once per session so they stay in sync
    with the factories."""
    seed_all()
    assert DATA_DIR.exists()
    yield
