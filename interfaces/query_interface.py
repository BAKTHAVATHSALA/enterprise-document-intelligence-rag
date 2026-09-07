"""Query and citation interfaces.

Defines payloads for incoming query requests, answer output responses,
and verifiable citation models.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CitationValidationStatusEnum(str, Enum):
    """Validation outcome categories for source citations."""
    VALID = "VALID"
    INVALID = "INVALID"
    FABRICATED = "FABRICATED"
    MISSING = "MISSING"


class QueryRequestInterface(BaseModel):
    """Payload for natural language query execution."""
    question: str = Field(..., min_length=1, description="Natural language question")
    document_ids: Optional[list[str]] = Field(default=None, description="Optional document ID filter")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of evidence chunks to retrieve")


class CitationInterface(BaseModel):
    """Source reference citation object attached to generated claims."""
    document_id: str = Field(..., description="Referenced document ID")
    page: int = Field(..., description="Referenced page number")
    section: str = Field(..., description="Referenced section title")
    chunk_id: str = Field(default="", description="Referenced chunk ID")
    source: str = Field(default="", description="Source filename")
    snippet: str = Field(default="", description="Supporting excerpt snippet from chunk")
    is_valid: bool = Field(default=True, description="Citation validation state")
    validation_status: CitationValidationStatusEnum = Field(
        default=CitationValidationStatusEnum.VALID,
        description="Detailed validation status (VALID, INVALID, FABRICATED, MISSING)",
    )
    validation_message: str = Field(
        default="Citation verified against retrieved evidence.",
        description="Explanatory validation message",
    )


class CitationValidationSummaryInterface(BaseModel):
    """Overall summary of citation validation for a query response."""
    total_citations: int = Field(default=0, description="Total citations evaluated")
    valid_count: int = Field(default=0, description="Count of valid citations")
    invalid_count: int = Field(default=0, description="Count of invalid citations")
    fabricated_count: int = Field(default=0, description="Count of fabricated citations")
    missing_count: int = Field(default=0, description="Count of missing citations")
    is_fully_validated: bool = Field(
        default=True, description="True if all citations are valid and no fabricated or invalid citations exist"
    )


class QueryResponseInterface(BaseModel):
    """Response payload containing generated answer and citations."""
    question: str = Field(..., description="Original query question")
    answer: str = Field(..., description="Grounded LLM generated answer")
    citations: list[CitationInterface] = Field(default_factory=list, description="Validated citations")
    validation_summary: Optional[CitationValidationSummaryInterface] = Field(
        default=None, description="Detailed citation validation summary"
    )
    confidence_score: float = Field(default=1.0, description="Overall answer grounding score")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")

