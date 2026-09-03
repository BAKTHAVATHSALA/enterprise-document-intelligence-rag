"""Query and citation interfaces.

Defines payloads for incoming query requests, answer output responses,
and verifiable citation models.
"""

from typing import Optional
from pydantic import BaseModel, Field


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
    chunk_id: str = Field(..., description="Referenced chunk ID")
    source: str = Field(..., description="Source filename")
    snippet: str = Field(..., description="Supporting excerpt snippet from chunk")
    is_valid: bool = Field(default=True, description="Citation validation state")


class QueryResponseInterface(BaseModel):
    """Response payload containing generated answer and citations."""
    question: str = Field(..., description="Original query question")
    answer: str = Field(..., description="Grounded LLM generated answer")
    citations: list[CitationInterface] = Field(default_factory=list, description="Validated citations")
    confidence_score: float = Field(default=1.0, description="Overall answer grounding score")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
