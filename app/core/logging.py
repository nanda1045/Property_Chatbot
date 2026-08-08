"""Structured application logging without external logging dependencies."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)

_STRUCTURED_FIELDS = (
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "run_id",
    "conversation_id",
    "property_code",
    "decision",
    "error_type",
)


class JsonLogFormatter(logging.Formatter):
    """Render stable JSON records suitable for local output or log collection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context_request_id = request_id_context.get()
        if context_request_id:
            payload["request_id"] = context_request_id
        for field in _STRUCTURED_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(level: str) -> logging.Logger:
    """Configure the project logger once and return it."""
    logger = logging.getLogger("aker")
    logger.setLevel(level.upper())
    logger.propagate = False
    handler = next(
        (
            candidate
            for candidate in logger.handlers
            if getattr(candidate, "_aker_structured_handler", False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        handler._aker_structured_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    handler.setLevel(level.upper())
    return logger
