"""Exception hierarchy that wraps provider API errors into a consistent shape."""

from __future__ import annotations

from typing import Any


class Error(Exception):
    """Base error wrapping an HTTP response from a provider.

    Can be constructed from a response object, an explicit message, or both.
    """

    default_message: str | None = None

    def __init__(self, response: Any = None, message: str | None = None) -> None:
        if isinstance(response, str):
            message = response
            response = None

        self.response = response
        body = getattr(response, "body", None) if response is not None else None
        super().__init__(message or body or self.default_message)


# Non-HTTP errors
class ConfigurationError(Exception):
    """Raised when a provider is used without the required configuration."""


class PromptNotFoundError(Exception):
    pass


class InvalidRoleError(Exception):
    pass


class InvalidToolChoiceError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


class UnsupportedAttachmentError(Exception):
    """Raised when an attachment cannot be formatted for the selected provider."""

    GUIDANCE = "Consider using a model that supports this attachment type."

    def __init__(self, type: str | None = None) -> None:
        message = "Unsupported attachment type"
        if type:
            message = f"{message}: {type}"
        super().__init__(f"{message}. {self.GUIDANCE}")


class ConnectionFailedError(Error):
    """A request failed at the transport layer (DNS, refused connection,
    timeout, dropped stream) without a usable HTTP response.

    The original transport exception is preserved as ``__cause__``.
    """

    default_message = "Connection failed - unable to reach the provider"

    @classmethod
    def wrap(cls, exc: BaseException) -> ConnectionFailedError:
        text = str(exc)
        return cls(None, f"{type(exc).__name__}: {text}" if text else type(exc).__name__)


# HTTP status-code errors
class BadRequestError(Error):
    default_message = "Invalid request - please check your input"


class ForbiddenError(Error):
    default_message = "Forbidden - you do not have permission to access this resource"


class ContextLengthExceededError(Error):
    default_message = "Context length exceeded"


class OverloadedError(Error):
    default_message = "Service overloaded - please try again later"


class PaymentRequiredError(Error):
    default_message = "Payment required - please top up your account"


class RateLimitError(Error):
    default_message = "Rate limit exceeded - please wait a moment"


class ServerError(Error):
    default_message = "API server error - please try again"


class ServiceUnavailableError(Error):
    default_message = "API server unavailable - please try again later"


class UnauthorizedError(Error):
    default_message = "Invalid API key - check your credentials"


# status code -> error class (ported from error_middleware.rb)
STATUS_ERRORS: dict[int, type[Error]] = {
    400: BadRequestError,
    401: UnauthorizedError,
    402: PaymentRequiredError,
    403: ForbiddenError,
    429: RateLimitError,
    500: ServerError,
    502: ServiceUnavailableError,
    503: ServiceUnavailableError,
    504: ServiceUnavailableError,
    529: OverloadedError,
}

# Provider messages that signal a context/token-limit overflow on 400/429
# responses; these map to ContextLengthExceededError so callers can truncate
# history and retry.
CONTEXT_LENGTH_PATTERNS = (
    "context length",
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "exceeds the maximum number of tokens",
)


def is_context_length_message(message: str | None) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(pattern in lowered for pattern in CONTEXT_LENGTH_PATTERNS)


def error_for_status(status: int, message: str | None = None) -> type[Error]:
    """Return the most appropriate :class:`Error` subclass for an HTTP status."""
    if status in (400, 413, 429) and is_context_length_message(message):
        return ContextLengthExceededError
    if status in STATUS_ERRORS:
        return STATUS_ERRORS[status]
    if 400 <= status < 500:
        return BadRequestError
    return ServerError
