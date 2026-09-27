from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_LOGGER_NAME = "rag"
_CONFIGURED = False
_RESERVED_FIELDS = set(logging.LogRecord(None, 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    """Format records as structured JSON for local and production log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Configure application logging once, using environment defaults."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    configured_level = level or os.getenv("LOG_LEVEL", "INFO")
    use_json = json_logs if json_logs is not None else os.getenv("LOG_JSON", "false").lower() == "true"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(configured_level.upper())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured child logger for an application module."""
    configure_logging()
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def log_exception(
    logger: logging.Logger,
    message: str,
    error: BaseException,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Log an exception with safe structured context and traceback."""
    logger.error(
        message,
        exc_info=(type(error), error, error.__traceback__),
        extra=dict(context or {}),
    )
