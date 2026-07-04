"""AWS Bedrock integration.

Implements AWS Signature Version 4 (SigV4) request signing in pure stdlib
(``hashlib`` + ``hmac``). The Converse protocol signs each non-streaming
request via :meth:`Bedrock.sign_headers`.

Streaming over Bedrock uses the binary AWS *event-stream* framing
(``application/vnd.amazon.eventstream``). That binary frame decoder is not
implemented here:
:meth:`Bedrock.stream_headers` raises :class:`NotImplementedError` with a clear
message. Non-streaming completions, embeddings parsing, and model listing work.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from dataclasses import dataclass
from urllib.parse import quote, urlparse

from ..connection import parse_error_body
from ..protocols.converse import Converse
from ..provider import Provider
from . import capabilities as _caps

_SERVICE = "bedrock"
_ALGORITHM = "AWS4-HMAC-SHA256"
_UNRESERVED_PATH = "-._~/"


@dataclass(slots=True)
class _Credentials:
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None = None


class BedrockCapabilities(_caps.Capabilities):
    pass


class Bedrock(Provider):
    protocols = {"converse": Converse}
    default_protocol_name = "converse"

    @property
    def api_base(self) -> str:
        return (
            self.config.bedrock_api_base
            or f"https://bedrock-runtime.{self._bedrock_region}.amazonaws.com"
        )

    @property
    def control_api_base(self) -> str:
        return (
            self.config.bedrock_api_base or f"https://bedrock.{self._bedrock_region}.amazonaws.com"
        )

    @property
    def headers(self) -> dict[str, str]:
        # Bedrock requests are signed per-request; no static auth headers.
        return {}

    def parse_error(self, response):  # type: ignore[override]
        body = getattr(response, "body", None)
        if body is None or body == "":
            return None
        if isinstance(body, str):
            return parse_error_body(body)
        if isinstance(body, dict):
            return (
                body.get("message")
                or body.get("Message")
                or body.get("error")
                or body.get("__type")
                or parse_error_body(body)
            )
        return parse_error_body(body)

    # --- SigV4 signing ---------------------------------------------------------
    def sign_headers(
        self, method: str, path: str, body: str, *, base_url: str | None = None
    ) -> dict[str, str]:
        """Return SigV4 ``Authorization`` and ``x-amz-*`` headers for a request.

        ``path`` may include a query string; ``body`` is the request body string.
        """
        base_url = base_url or self.api_base
        credentials = self._bedrock_credentials()
        now = _dt.datetime.now(_dt.UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        parsed_path = urlparse(path)
        canonical_uri = _canonical_uri(parsed_path.path)
        canonical_query = _canonical_query_string(parsed_path.query)
        payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        headers: dict[str, str] = {
            "host": urlparse(base_url).netloc,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if credentials.session_token:
            headers["x-amz-security-token"] = credentials.session_token

        sorted_keys = sorted(headers)
        signed_headers = ";".join(sorted_keys)
        canonical_headers = "".join(f"{key}:{headers[key].strip()}\n" for key in sorted_keys)

        canonical_request = "\n".join(
            [
                method,
                canonical_uri,
                canonical_query,
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )

        credential_scope = f"{date_stamp}/{self._bedrock_region}/{_SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            [
                _ALGORITHM,
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )

        signing_key = self._signing_key(date_stamp, credentials.secret_access_key or "")
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        result = {
            "X-Amz-Date": amz_date,
            "X-Amz-Content-Sha256": payload_hash,
            "Authorization": (
                f"{_ALGORITHM} Credential={credentials.access_key_id}/{credential_scope}, "
                f"SignedHeaders={signed_headers}, Signature={signature}"
            ),
            "Content-Type": "application/json",
        }
        if credentials.session_token:
            result["X-Amz-Security-Token"] = credentials.session_token
        return result

    def stream_headers(self, method: str, path: str, body: str) -> dict[str, str]:
        """Signed headers for a streaming request.

        Bedrock streams binary AWS event-stream frames; the frame decoder is not
        ported. Callers attempting streaming get a clear error rather than a
        silent failure.
        """
        raise NotImplementedError(
            "Bedrock streaming uses the binary AWS event-stream framing "
            "(application/vnd.amazon.eventstream), which pyllm does not yet decode. "
            "Use a non-streaming completion (omit on_chunk / use chat.ask) instead."
        )

    @classmethod
    def capabilities_cls(cls):
        return BedrockCapabilities

    @classmethod
    def configuration_options(cls) -> list[str]:
        return [
            "bedrock_api_key",
            "bedrock_secret_key",
            "bedrock_region",
            "bedrock_session_token",
            "bedrock_api_base",
        ]

    @classmethod
    def configuration_requirements_list(cls) -> list[str]:
        return ["bedrock_api_key", "bedrock_secret_key", "bedrock_region"]

    # --- internals -------------------------------------------------------------
    @property
    def _bedrock_region(self) -> str:
        return self.config.bedrock_region

    def _bedrock_credentials(self) -> _Credentials:
        return _Credentials(
            access_key_id=self.config.bedrock_api_key,
            secret_access_key=self.config.bedrock_secret_key,
            session_token=self.config.bedrock_session_token,
        )

    def _signing_key(self, date_stamp: str, secret_access_key: str) -> bytes:
        k_date = _hmac(f"AWS4{secret_access_key}".encode(), date_stamp)
        k_region = _hmac(k_date, self._bedrock_region)
        k_service = _hmac(k_region, _SERVICE)
        return _hmac(k_service, "aws4_request")


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _uri_encode(text: str, *, safe: str = "") -> str:
    return quote(text, safe=safe)


def _canonical_uri(path: str) -> str:
    if not path:
        return "/"
    segments = path.split("/")
    canonical = "/".join(_uri_encode(segment) for segment in segments)
    return canonical if canonical.startswith("/") else f"/{canonical}"


def _canonical_query_string(raw_query: str) -> str:
    if not raw_query:
        return ""
    pairs: list[tuple[str, str]] = []
    for part in raw_query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        pairs.append((key, value))
    pairs.sort(key=lambda kv: kv[0])
    return "&".join(f"{_uri_encode(k)}={_uri_encode(v)}" for k, v in pairs)
