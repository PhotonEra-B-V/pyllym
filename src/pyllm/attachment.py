"""File attachment handling.

Supported sources are URLs, filesystem paths (:class:`str`/:class:`pathlib.Path`),
raw ``bytes``, binary file objects, and provider-managed
:class:`~pyllm.uploaded_file.UploadedFile` references.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import mime_type as mime
from .errors import Error

DOCUMENT_EXTENSIONS = frozenset(
    [
        "doc",
        "docx",
        "dot",
        "key",
        "numbers",
        "odp",
        "ods",
        "odt",
        "pages",
        "pot",
        "pps",
        "ppt",
        "pptx",
        "rtf",
        "xls",
        "xlsx",
    ]
)


class Attachment:
    def __init__(self, source: Any, *, filename: str | None = None) -> None:
        self.source = source
        self.filename = filename or self._source_filename()
        self._content: bytes | str | None = None
        self.mime_type = self._determine_mime_type()

    # --- source classification -------------------------------------------------
    def is_url(self) -> bool:
        if isinstance(self.source, str):
            return self.source.startswith(("http://", "https://"))
        return False

    def is_provider_file(self) -> bool:
        from .uploaded_file import UploadedFile

        return isinstance(self.source, UploadedFile)

    @property
    def provider_file_id(self) -> str | None:
        return self.source.id if self.is_provider_file() else None

    @property
    def provider_file_uri(self) -> str | None:
        return getattr(self.source, "uri", None) if self.is_provider_file() else None

    def is_path(self) -> bool:
        if self.is_provider_file():
            return False
        return isinstance(self.source, Path) or (isinstance(self.source, str) and not self.is_url())

    def is_io_like(self) -> bool:
        return hasattr(self.source, "read") and not self.is_path() and not self.is_provider_file()

    # --- content ---------------------------------------------------------------
    @property
    def content(self) -> bytes | str | None:
        if self.is_provider_file():
            raise Error(
                f"Provider-managed file {self.provider_file_id} cannot be read as "
                "inline attachment content"
            )
        if self._content is None:
            self._content = self._load_content()
        return self._content

    @property
    def encoded(self) -> str:
        data = self.content
        if isinstance(data, str):
            data = data.encode("utf-8")
        return base64.b64encode(data or b"").decode("ascii")

    def for_llm(self) -> str:
        if self.type == "text":
            return (
                f"<file name='{self.filename}' mime_type='{self.mime_type}'>{self.content}</file>"
            )
        return f"data:{self.mime_type};base64,{self.encoded}"

    # --- type predicates -------------------------------------------------------
    @property
    def type(self) -> str:
        if self.is_image():
            return "image"
        if self.is_video():
            return "video"
        if self.is_audio():
            return "audio"
        if self.is_pdf():
            return "pdf"
        if self.is_text():
            return "text"
        if self.is_document():
            return "document"
        return "unknown"

    def is_image(self) -> bool:
        return mime.is_image(self.mime_type)

    def is_video(self) -> bool:
        return mime.is_video(self.mime_type)

    def is_audio(self) -> bool:
        return mime.is_audio(self.mime_type)

    def is_pdf(self) -> bool:
        return mime.is_pdf(self.mime_type)

    def is_text(self) -> bool:
        return mime.is_text(self.mime_type)

    def is_document(self) -> bool:
        if self.is_pdf() or self.is_text():
            return False
        return mime.is_document(self.mime_type) or (self.extension in DOCUMENT_EXTENSIONS)

    @property
    def format(self) -> str:
        if self.mime_type == "audio/mpeg":
            return "mp3"
        if self.mime_type in ("audio/wav", "audio/wave", "audio/x-wav"):
            return "wav"
        return self.mime_type.split("/")[-1]

    @property
    def extension(self) -> str | None:
        ext = Path(str(self.filename)).suffix.lstrip(".").lower()
        return ext or None

    @property
    def byte_size(self) -> int | None:
        if self.is_provider_file():
            return getattr(self.source, "byte_size", None)
        if self.is_path():
            try:
                return os.path.getsize(self.source)
            except OSError:
                return None
        if isinstance(self.source, (bytes, bytearray)):
            return len(self.source)
        data = self._content
        if isinstance(data, str):
            return len(data.encode("utf-8"))
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "source": self.source}

    # --- internals -------------------------------------------------------------
    def _load_content(self) -> bytes | str | None:
        if self.is_url():
            return self._fetch_content()
        if self.is_path():
            return Path(self.source).read_bytes()
        if isinstance(self.source, (bytes, bytearray)):
            return bytes(self.source)
        if self.is_io_like():
            if hasattr(self.source, "seek"):
                try:
                    self.source.seek(0)
                except Exception:
                    pass
            return self.source.read()
        return None

    def _fetch_content(self) -> bytes:
        from urllib.request import urlopen

        with urlopen(str(self.source), timeout=30) as resp:
            return resp.read()

    def _determine_mime_type(self) -> str:
        if self.is_provider_file():
            provider_mime = getattr(self.source, "mime_type", None)
            if provider_mime:
                return provider_mime
            return mime.for_source(None, name=getattr(self.source, "filename", None))
        source = None if self.is_url() else self.source
        mime_type = mime.for_source(source, name=self.filename)
        if mime_type == mime.DEFAULT and not self.is_url():
            data = self.content
            if isinstance(data, (bytes, bytearray)):
                mime_type = mime.for_source(bytes(data))
        if mime_type == "audio/x-wav":
            mime_type = "audio/wav"
        return mime_type

    def _source_filename(self) -> str | None:
        if self.is_url():
            return os.path.basename(urlparse(self.source).path) or "attachment"
        if self.is_provider_file():
            return str(getattr(self.source, "filename", "attachment"))
        if self.is_path():
            return Path(self.source).name
        if self.is_io_like():
            name = getattr(self.source, "name", None)
            return os.path.basename(name) if name else "attachment"
        return None
