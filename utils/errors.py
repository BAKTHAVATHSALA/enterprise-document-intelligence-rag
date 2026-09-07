"""Centralized Error Taxonomy for Hybrid RAG Platform.

Defines standardized error classifications for:
- Validation failures (e.g. empty files, prompt injection, invalid input)
- Resource not found (DocumentNotFoundError)
- Ingestion failures (parsing, chunking, PII redaction)
- Multi-channel retrieval failures (vector, BM25, graph)
- External provider failures (OpenAI, Pinecone, Neo4j, PostgreSQL)
- Answer generation and citation failures

Ensures consistent HTTP status codes, structured error metadata, and safe client messages.
"""

from typing import Any, Optional


class RAGError(Exception):
    """Base application exception for all Hybrid RAG pipeline errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Initialize base RAGError.

        @param message: Human-readable error description safe for client/logging.
        @param error_code: Machine-readable uppercase taxonomy classification code.
        @param status_code: Associated HTTP status code.
        @param details: Optional non-sensitive structured contextual metadata.
        @param correlation_id: Optional correlation / request ID tracking the operation.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.correlation_id = correlation_id

    def to_dict(self) -> dict[str, Any]:
        """Convert exception attributes to a serializable dictionary."""
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.correlation_id:
            payload["correlation_id"] = self.correlation_id
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(RAGError):
    """Input validation failure (e.g., unsupported format, empty input, invalid fields)."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details,
            correlation_id=correlation_id,
        )


class DocumentNotFoundError(RAGError):
    """Target document or metadata record does not exist in store."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        det = details or {}
        if document_id:
            det["document_id"] = document_id
        super().__init__(
            message=message,
            error_code="DOCUMENT_NOT_FOUND",
            status_code=404,
            details=det,
            correlation_id=correlation_id,
        )


class SecurityValidationError(RAGError):
    """Security policy violation (e.g., prompt injection, jailbreak attempt)."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="SECURITY_VIOLATION",
            status_code=400,
            details=details,
            correlation_id=correlation_id,
        )


class IngestionError(RAGError):
    """Failure during document parsing, OCR, chunking, or ingestion orchestration."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="INGESTION_ERROR",
            status_code=500,
            details=details,
            correlation_id=correlation_id,
        )


class RetrievalError(RAGError):
    """Failure during multi-source candidate retrieval (vector, BM25, graph, or fusion)."""

    def __init__(
        self,
        message: str,
        channel: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        det = details or {}
        if channel:
            det["channel"] = channel
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=502,
            details=det,
            correlation_id=correlation_id,
        )


class ExternalProviderError(RAGError):
    """External API or third-party service failure (OpenAI, Pinecone, Neo4j, PostgreSQL)."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        status_code: int = 502,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        det = details or {}
        if provider:
            det["provider"] = provider
        super().__init__(
            message=message,
            error_code="EXTERNAL_PROVIDER_ERROR",
            status_code=status_code,
            details=det,
            correlation_id=correlation_id,
        )


class GenerationError(RAGError):
    """Failure during LLM synthesis, neural reranking, or source citation validation."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        det = details or {}
        if stage:
            det["stage"] = stage
        super().__init__(
            message=message,
            error_code="GENERATION_ERROR",
            status_code=500,
            details=det,
            correlation_id=correlation_id,
        )
