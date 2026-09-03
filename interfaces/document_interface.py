"""Document processing interfaces and domain models.

Defines Pydantic data models and schemas for documents, chunks,
metadata, PII redactions, and named entities.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# Domain Constants
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
UNKNOWN_SECTION_NAME: str = "General"


class DocumentStatusEnum(str, Enum):
    """Status options for document processing jobs."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EntityInterface(BaseModel):
    """Named entity model extracted from document text."""
    text: str = Field(..., description="Extracted entity string")
    label: str = Field(..., description="Entity category (ORGANIZATION, PERSON, CONTRACT, etc.)")
    start_char: int = Field(..., description="Start character offset")
    end_char: int = Field(..., description="End character offset")
    confidence: float = Field(default=1.0, description="Extraction confidence score")


class PIIMatchInterface(BaseModel):
    """Sensitive PII match detected in text."""
    pii_type: str = Field(..., description="PII category (SSN, EMAIL, PHONE, API_KEY)")
    original_text: str = Field(..., description="Original sensitive text")
    redacted_text: str = Field(..., description="Redacted replacement token")
    start_char: int = Field(..., description="Start character index")
    end_char: int = Field(..., description="End character index")


class ChunkMetadataInterface(BaseModel):
    """Metadata retained for document chunks to enable traceability."""
    document_id: str = Field(..., description="Unique parent document identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    page: int = Field(..., description="Page number origin")
    section: str = Field(default=UNKNOWN_SECTION_NAME, description="Section heading or title")
    entities: list[str] = Field(default_factory=list, description="Extracted entity names in chunk")
    source: str = Field(..., description="Source filename or origin path")


class ChunkInterface(BaseModel):
    """Semantic document chunk with text and lineage metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Extracted or processed chunk text")
    metadata: ChunkMetadataInterface = Field(..., description="Traceability metadata")
    embedding: Optional[list[float]] = Field(default=None, description="Semantic vector embedding")


class DocumentMetadataInterface(BaseModel):
    """High-level metadata for ingested document."""
    document_id: str = Field(..., description="Unique document ID")
    title: str = Field(..., description="Document title")
    source_filename: str = Field(..., description="Source file name")
    total_pages: int = Field(..., description="Total page count")
    total_chunks: int = Field(..., description="Total generated chunk count")
    created_at: str = Field(..., description="Ingestion ISO timestamp")


class DocumentStatusInterface(BaseModel):
    """Tracking status for async document ingestion job."""
    document_id: str = Field(..., description="Unique document ID")
    status: DocumentStatusEnum = Field(..., description="Current ingestion state")
    error_message: Optional[str] = Field(default=None, description="Failure reason if FAILED")
    processed_chunks: int = Field(default=0, description="Count of processed chunks so far")
