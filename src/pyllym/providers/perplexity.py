"""Perplexity API integration."""

from __future__ import annotations

import re
from typing import Any

from ..connection import parse_error_body
from .openai_compatible import OpenAICompatible


class Perplexity(OpenAICompatible):
    default_api_base = "https://api.perplexity.ai"
    extra_headers = {"Content-Type": "application/json"}

    def parse_error(self, response: Any) -> Any:
        body = getattr(response, "body", None)
        if not body:
            return None
        # Perplexity returns HTML for auth errors.
        if isinstance(body, str) and "<html>" in body and "<title>" in body:
            match = re.search(r"<title>(.+?)</title>", body)
            if match:
                return re.sub(r"^\d+\s+", "", match.group(1))
        return parse_error_body(body)
