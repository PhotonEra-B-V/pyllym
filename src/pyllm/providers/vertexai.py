"""Google Vertex AI integration.

Vertex AI hosts the Gemini family behind project/location-scoped endpoints. This
port speaks the Gemini protocol over those endpoints (a :class:`VertexAIGemini`
subclass that rewrites the completion/stream/embedding URLs to the
``publishers/google/models`` path).

Auth requires a Google OAuth access token (service-account or
application-default credentials). Token minting (JWT signing, OAuth token
exchange) is not implemented here; instead this provider expects a
pre-supplied ``config.vertexai_access_token``; if it is missing a
:class:`ConfigurationError` is raised with guidance on how to obtain one (e.g.
``gcloud auth print-access-token``).
"""

from __future__ import annotations

from typing import Any

from ..errors import ConfigurationError
from ..model.info import Info
from ..protocols.gemini import Gemini as GeminiProtocol
from ..provider import Provider
from . import capabilities as _caps


class VertexAIGemini(GeminiProtocol):
    """The Gemini protocol over Vertex AI's project-scoped endpoints."""

    def completion_url(self) -> str:
        assert self.model is not None
        return f"{self.provider.model_path(self.model.id)}:generateContent"

    def stream_url(self) -> str:
        assert self.model is not None
        return f"{self.provider.model_path(self.model.id)}:streamGenerateContent?alt=sse"

    def embedding_url(self, *, model: str | None = None) -> str:
        assert model is not None
        return f"{self.provider.model_path(model)}:predict"


class VertexAICapabilities(_caps.Capabilities):
    pass


class VertexAI(Provider):
    protocols = {"gemini": VertexAIGemini}
    default_protocol_name = "gemini"

    SCOPES = (
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language.retriever",
    )

    def protocol_for(self, model: Info | None, **request: Any) -> type:
        # Only the Gemini protocol is ported; publisher-prefixed MaaS models
        # (anthropic/mistral/openai-compatible) are not yet supported here.
        return self._default_protocol()

    def location_path(self) -> str:
        return (
            f"projects/{self.config.vertexai_project_id}/locations/{self.config.vertexai_location}"
        )

    def model_path(self, model: str, *, publisher: str = "google") -> str:
        return f"{self.location_path()}/publishers/{publisher}/models/{model}"

    @property
    def api_base(self) -> str:
        if self.config.vertexai_api_base:
            return self.config.vertexai_api_base
        location = str(self.config.vertexai_location or "")
        if location == "global":
            return "https://aiplatform.googleapis.com/v1beta1"
        return f"https://{location}-aiplatform.googleapis.com/v1beta1"

    @property
    def headers(self) -> dict[str, str]:
        token = self.config.vertexai_access_token
        if not token:
            raise ConfigurationError(
                "Vertex AI requires an OAuth access token. pyllm does not mint Google "
                "credentials; supply a pre-obtained token via "
                "config.vertexai_access_token (e.g. from "
                "`gcloud auth print-access-token`)."
            )
        return {"Authorization": f"Bearer {token}"}

    @classmethod
    def capabilities_cls(cls):
        return VertexAICapabilities

    @classmethod
    def configuration_options(cls) -> list[str]:
        return [
            "vertexai_project_id",
            "vertexai_location",
            "vertexai_access_token",
            "vertexai_service_account_key",
            "vertexai_api_base",
        ]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["vertexai_project_id", "vertexai_location"]
