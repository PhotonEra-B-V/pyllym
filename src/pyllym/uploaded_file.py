"""Provider-managed file metadata.

The concrete wire formats live under each provider's ``files`` protocol; this
module holds the public :class:`UploadedFile` value object and the module-level
``upload``/``find``/``download`` helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import Context
    from .provider import Provider


class UploadedFile:
    def __init__(self, *, id: str, **attributes: Any) -> None:
        self.id = id
        self.provider = attributes.get("provider")
        self.filename = attributes.get("filename")
        self.byte_size = attributes.get("byte_size")
        self.created_at = attributes.get("created_at")
        self.expires_at = attributes.get("expires_at")
        self.status = attributes.get("status")
        self.mime_type = attributes.get("mime_type")
        self.purpose = attributes.get("purpose")
        self.uri = attributes.get("uri")
        self.downloadable = attributes.get("downloadable")
        self.metadata = attributes.get("metadata") or {}


def _provider_for(provider: str | None, context: Context | None) -> Provider:
    from . import config as _config
    from . import models as _models
    from .provider import Provider

    cfg = context.config if context else _config()
    if provider:
        return Provider.resolve_bang(provider)(cfg)
    return _models.resolve(cfg.default_model, config=cfg)[1]


async def upload(
    file: Any, *, provider: str | None = None, context: Context | None = None, **options: Any
) -> UploadedFile:
    return await _provider_for(provider, context).upload_file(file, **options)


async def find(
    id: str, *, provider: str | None = None, context: Context | None = None
) -> UploadedFile:
    return await _provider_for(provider, context).find_file(id)


async def download(
    id: str, *, provider: str | None = None, context: Context | None = None
) -> bytes:
    return await _provider_for(provider, context).download_file(id)
