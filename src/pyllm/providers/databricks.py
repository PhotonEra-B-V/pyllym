"""Databricks Foundation Model / serving-endpoint API (OpenAI-compatible).

Requires a per-workspace base URL, e.g.
``https://<workspace>.cloud.databricks.com/serving-endpoints``.
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatible


class Databricks(OpenAICompatible):
    default_api_base = None  # databricks_api_base is required config
    assume_models = True
