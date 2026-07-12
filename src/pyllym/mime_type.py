"""MIME-type helpers.

Uses the stdlib :mod:`mimetypes` for filename-based detection and falls
back to ``puremagic`` (optional ``mime`` extra) for content sniffing.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

TEXT_SUFFIXES = (
    "+json",
    "+xml",
    "+html",
    "+yaml",
    "+csv",
    "+plain",
    "+javascript",
    "+svg",
)

NON_TEXT_PREFIX_TEXT_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/ecmascript",
        "application/rtf",
        "application/sql",
        "application/x-sh",
        "application/x-csh",
        "application/x-httpd-php",
        "application/sdp",
        "application/sparql-query",
        "application/graphql",
        "application/yang",
        "application/mbox",
        "application/x-tex",
        "application/x-latex",
        "application/x-perl",
        "application/x-python",
        "application/x-tcl",
        "application/pgp-signature",
        "application/pgp-keys",
        "application/vnd.coffeescript",
        "application/vnd.dart",
        "application/vnd.oai.openapi",
        "application/vnd.zul",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
    }
)

DOCUMENT_MIME_TYPES = frozenset(
    {
        "application/msword",
        "application/rtf",
        "application/vnd.apple.keynote",
        "application/vnd.apple.numbers",
        "application/vnd.apple.pages",
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }
)

DOCUMENT_MIME_PREFIXES = (
    "application/vnd.openxmlformats-officedocument.",
    "application/vnd.oasis.opendocument.",
)

DEFAULT = "application/octet-stream"


def for_source(source: object = None, *, name: str | None = None) -> str:
    """Best-effort MIME type for a path/bytes source and/or filename."""
    if name:
        guessed, _ = mimetypes.guess_type(name)
        if guessed:
            return guessed
    if isinstance(source, (str, Path)):
        guessed, _ = mimetypes.guess_type(str(source))
        if guessed:
            return guessed
    if isinstance(source, (bytes, bytearray)):
        sniffed = _sniff_bytes(bytes(source))
        if sniffed:
            return sniffed
    return DEFAULT


def _sniff_bytes(data: bytes) -> str | None:
    try:
        import puremagic
    except ImportError:
        return None
    try:
        return puremagic.from_string(data, mime=True) or None
    except Exception:
        return None


def is_image(type: str) -> bool:
    return type.startswith("image/")


def is_video(type: str) -> bool:
    return type.startswith("video/")


def is_audio(type: str) -> bool:
    return type.startswith("audio/")


def is_pdf(type: str) -> bool:
    return type == "application/pdf"


def is_text(type: str) -> bool:
    return (
        type.startswith("text/")
        or any(type.endswith(suffix) for suffix in TEXT_SUFFIXES)
        or type in NON_TEXT_PREFIX_TEXT_MIME_TYPES
    )


def is_document(type: str) -> bool:
    if is_pdf(type) or is_text(type):
        return False
    return type in DOCUMENT_MIME_TYPES or any(
        type.startswith(prefix) for prefix in DOCUMENT_MIME_PREFIXES
    )
