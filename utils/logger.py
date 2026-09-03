"""Structured Logger Utility.

Enforces structured logging standards defined in rules.md:
Every log statement must start with the name of the enclosing function,
followed by a dash separator: logger.info("[functionName] - message").
"""

import sys
import logging
from typing import Any

DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class StandardLoggerWrapper:
    """Wrapper around python standard logging module ensuring required formatting."""

    def __init__(self, name: str = "app") -> None:
        """Initialize logger instance.

        @param name: Logger channel name.
        """
        self._logger: logging.Logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stdout)
            formatter: logging.Formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _format_msg(self, func_name: str, message: str) -> str:
        """Format message string as required by rules.md.

        @param func_name: Enclosing function name.
        @param message: Core message payload.
        @returns: Formatted log line string.
        """
        return f"[{func_name}] - {message}"

    def info(self, func_name: str, message: str) -> None:
        """Log info level message.

        @param func_name: Enclosing function name.
        @param message: Message text.
        """
        self._logger.info(self._format_msg(func_name, message))

    def error(self, func_name: str, message: str, exc: Exception | None = None) -> None:
        """Log error level message with optional exception detail.

        @param func_name: Enclosing function name.
        @param message: Message text.
        @param exc: Optional exception object.
        """
        self._logger.error(self._format_msg(func_name, message), exc_info=exc)

    def warning(self, func_name: str, message: str) -> None:
        """Log warning level message.

        @param func_name: Enclosing function name.
        @param message: Message text.
        """
        self._logger.warning(self._format_msg(func_name, message))

    def debug(self, func_name: str, message: str) -> None:
        """Log debug level message.

        @param func_name: Enclosing function name.
        @param message: Message text.
        """
        self._logger.debug(self._format_msg(func_name, message))


# Global Singleton Logger Instance
logger: StandardLoggerWrapper = StandardLoggerWrapper("HybridRAG")
