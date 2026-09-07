"""Structured JSON Logger Utility for Hybrid RAG Platform.

Enforces production-grade structured JSON logging standards:
- JSON output with standardized fields:
  timestamp, level, correlation_id, operation, message, document_id, query_id, error_type, duration_ms, status.
- Correlation IDs and tracing context propagation via contextvars.
- Strict sanitization: Redacts API keys, credentials, tokens, passwords, raw prompts, and chunk contents.
- Strictly preserves: document_id, query_id, correlation_id, error codes, page numbers, and operation names.
- Thread-safe and async-safe context propagation across FastAPI requests and background tasks.
"""

import sys
import json
import logging
import re
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Any, Optional

# Context variables for request-scoped correlation and tracing identifiers
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
current_document_id_ctx: ContextVar[Optional[str]] = ContextVar("current_document_id", default=None)
current_query_id_ctx: ContextVar[Optional[str]] = ContextVar("current_query_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Return active correlation ID for the current context."""
    return correlation_id_ctx.get()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set correlation ID for current context."""
    correlation_id_ctx.set(correlation_id)


def get_document_id() -> Optional[str]:
    """Return active document ID for the current context."""
    return current_document_id_ctx.get()


def set_document_id(document_id: Optional[str]) -> None:
    """Set active document ID for current context."""
    current_document_id_ctx.set(document_id)


def get_query_id() -> Optional[str]:
    """Return active query ID for the current context."""
    return current_query_id_ctx.get()


def set_query_id(query_id: Optional[str]) -> None:
    """Set active query ID for current context."""
    current_query_id_ctx.set(query_id)


# In-memory structured log history for verification and tests
_LOG_RECORDS: list[dict[str, Any]] = []


def get_log_records() -> list[dict[str, Any]]:
    """Return in-memory log history."""
    return list(_LOG_RECORDS)


def clear_log_records() -> None:
    """Clear in-memory log history."""
    global _LOG_RECORDS
    _LOG_RECORDS.clear()


# Sensitive key patterns to redact
SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "auth",
    "credential",
    "database_url",
    "db_url",
    "connection_string",
    "private_key",
)

# Raw content keys to redact/summarize (never log prompts or raw text bodies)
RAW_CONTENT_KEYS: tuple[str, ...] = (
    "prompt",
    "prompts",
    "raw_text",
    "chunk_text",
    "chunk_content",
    "file_bytes",
    "pdf_bytes",
    "content_bytes",
)

# Sensitive value prefixes
SENSITIVE_VALUE_PREFIXES: tuple[str, ...] = (
    "sk-",
    "pcsk_",
    "lsv2_",
    "npg_",
    "bearer ",
    "Bearer ",
)

# Regex patterns to scrub from log messages and exception text
SECRET_REGEXES: tuple[re.Pattern, ...] = (
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    re.compile(r"pcsk_[a-zA-Z0-9_-]{20,}"),
    re.compile(r"lsv2_pt_[a-zA-Z0-9_-]{20,}"),
    re.compile(r"npg_[a-zA-Z0-9_-]{12,}"),
    re.compile(r"postgres(?:ql)?://[^:]+:([^@]+)@"),  # password in postgres url
)

# Allowed identifiers and system properties that MUST NOT be redacted
PRESERVED_KEYS: set[str] = {
    "document_id",
    "query_id",
    "correlation_id",
    "error_type",
    "error_code",
    "page",
    "page_number",
    "operation",
    "status",
    "duration_ms",
    "timestamp",
    "level",
    "model_name",
    "tenant_id",
    "source_filename",
}


def sanitize_message_text(text: str) -> str:
    """Scrub sensitive credentials and passwords from freeform string messages.

    @param text: Raw message string.
    @returns: Sanitized message string.
    """
    if not isinstance(text, str):
        return str(text)

    sanitized = text
    for regex in SECRET_REGEXES:
        sanitized = regex.sub("[REDACTED]", sanitized)
    return sanitized


def sanitize_log_field(key: str, value: Any) -> Any:
    """Sanitize an individual key-value pair for structured logging.

    Preserves document_id, query_id, correlation_id, error codes, page numbers, and operation names.
    Redacts secrets, passwords, tokens, raw prompts, and chunk contents.
    """
    key_lower = str(key).lower()

    # Always preserve explicit domain identifiers
    if key in PRESERVED_KEYS:
        return value

    # Redact sensitive keys
    if any(pat in key_lower for pat in SENSITIVE_KEY_PATTERNS):
        return "[REDACTED]"

    # Redact raw content or prompt text
    if any(pat == key_lower or pat in key_lower for pat in RAW_CONTENT_KEYS):
        if isinstance(value, (str, bytes)):
            return f"[CONTENT_REDACTED ({len(value)} bytes/chars)]"
        return "[CONTENT_REDACTED]"

    # Check string values for sensitive prefixes
    if isinstance(value, str):
        if any(value.startswith(pfx) for pfx in SENSITIVE_VALUE_PREFIXES):
            return "[REDACTED]"
        return sanitize_message_text(value)

    if isinstance(value, dict):
        return {k: sanitize_log_field(k, v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [sanitize_log_field(key, item) for item in value]

    return value


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs serialized JSON strings per log line."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a structured JSON object.

        @param record: LogRecord to format.
        @returns: Serialized JSON line.
        """
        payload: dict[str, Any] = getattr(record, "structured_payload", None)
        if payload is not None:
            return json.dumps(payload, ensure_ascii=False)

        # Fallback for logs emitted through external packages
        msg_str = sanitize_message_text(record.getMessage())
        fallback_payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "correlation_id": get_correlation_id(),
            "operation": record.funcName or record.name,
            "message": msg_str,
            "document_id": get_document_id(),
            "query_id": get_query_id(),
            "error_type": None,
            "duration_ms": None,
            "status": None,
        }
        return json.dumps(fallback_payload, ensure_ascii=False)


class StandardLoggerWrapper:
    """Wrapper around standard python logging providing structured JSON logs and context."""

    def __init__(self, name: str = "HybridRAG") -> None:
        """Initialize structured logger wrapper.

        @param name: Logger hierarchy name.
        """
        self._logger: logging.Logger = logging.getLogger(name)
        self._logger.handlers.clear()
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def _build_entry(
        self,
        level: str,
        func_name: str,
        message: str,
        correlation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Construct, sanitize, and record structured log dictionary."""
        effective_corr_id = correlation_id or get_correlation_id()
        effective_doc_id = document_id or get_document_id()
        effective_query_id = query_id or get_query_id()

        clean_msg = sanitize_message_text(message)
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "correlation_id": effective_corr_id,
            "operation": func_name,
            "message": clean_msg,
            "document_id": effective_doc_id,
            "query_id": effective_query_id,
            "error_type": error_type,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            "status": status,
        }

        if extra:
            sanitized_extra = {k: sanitize_log_field(k, v) for k, v in extra.items()}
            entry["details"] = sanitized_extra

        # Keep in-memory record for testing and audits
        _LOG_RECORDS.append(entry)
        return entry

    def info(
        self,
        func_name: str,
        message: str,
        correlation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log structured info event."""
        entry = self._build_entry(
            level="INFO",
            func_name=func_name,
            message=message,
            correlation_id=correlation_id,
            document_id=document_id,
            query_id=query_id,
            duration_ms=duration_ms,
            status=status or "success",
            error_type=error_type,
            extra=kwargs,
        )
        self._logger.info(message, extra={"structured_payload": entry})

    def warning(
        self,
        func_name: str,
        message: str,
        correlation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log structured warning event."""
        entry = self._build_entry(
            level="WARNING",
            func_name=func_name,
            message=message,
            correlation_id=correlation_id,
            document_id=document_id,
            query_id=query_id,
            duration_ms=duration_ms,
            status=status or "warning",
            error_type=error_type,
            extra=kwargs,
        )
        self._logger.warning(message, extra={"structured_payload": entry})

    def error(
        self,
        func_name: str,
        message: str,
        exc: Optional[Exception] = None,
        correlation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log structured error event with exception taxonomy."""
        resolved_error_type = error_type
        if not resolved_error_type and exc is not None:
            resolved_error_type = getattr(exc, "error_code", type(exc).__name__)

        entry = self._build_entry(
            level="ERROR",
            func_name=func_name,
            message=message,
            correlation_id=correlation_id,
            document_id=document_id,
            query_id=query_id,
            duration_ms=duration_ms,
            status=status or "failure",
            error_type=resolved_error_type,
            extra=kwargs,
        )
        self._logger.error(message, exc_info=exc, extra={"structured_payload": entry})

    def debug(
        self,
        func_name: str,
        message: str,
        correlation_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        status: Optional[str] = None,
        error_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log structured debug event."""
        entry = self._build_entry(
            level="DEBUG",
            func_name=func_name,
            message=message,
            correlation_id=correlation_id,
            document_id=document_id,
            query_id=query_id,
            duration_ms=duration_ms,
            status=status,
            error_type=error_type,
            extra=kwargs,
        )
        self._logger.debug(message, extra={"structured_payload": entry})


# Global Singleton Logger Instance
logger: StandardLoggerWrapper = StandardLoggerWrapper("HybridRAG")
