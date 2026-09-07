"""Utilities package for pure helper functions, logging, configuration, and errors."""

from utils.logger import (
    logger,
    get_correlation_id,
    set_correlation_id,
    get_document_id,
    set_document_id,
    get_query_id,
    set_query_id,
    get_log_records,
    clear_log_records,
    sanitize_message_text,
    sanitize_log_field,
)
from utils.errors import (
    RAGError,
    ValidationError,
    DocumentNotFoundError,
    SecurityValidationError,
    IngestionError,
    RetrievalError,
    ExternalProviderError,
    GenerationError,
)

__all__ = [
    "logger",
    "get_correlation_id",
    "set_correlation_id",
    "get_document_id",
    "set_document_id",
    "get_query_id",
    "set_query_id",
    "get_log_records",
    "clear_log_records",
    "sanitize_message_text",
    "sanitize_log_field",
    "RAGError",
    "ValidationError",
    "DocumentNotFoundError",
    "SecurityValidationError",
    "IngestionError",
    "RetrievalError",
    "ExternalProviderError",
    "GenerationError",
]
