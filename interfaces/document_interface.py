"""Document processing interfaces and domain models.

Defines Pydantic data models and schemas for documents, chunks,
metadata, PII redactions, named entities, ingestion stages, and async jobs.
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
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class IngestionStageEnum(str, Enum):
    """Granular stages for asynchronous document ingestion workflow."""
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    PII_REDACTION = "PII_REDACTION"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


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
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    page: int = Field(..., description="Page number origin")
    section: str = Field(default=UNKNOWN_SECTION_NAME, description="Section heading or title")
    entities: list[str] = Field(default_factory=list, description="Extracted entity names in chunk")
    source: str = Field(..., description="Source filename or origin path")


class ChunkInterface(BaseModel):
    """Semantic document chunk with text and lineage metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    text: str = Field(..., description="Extracted or processed chunk text")
    metadata: ChunkMetadataInterface = Field(..., description="Traceability metadata")
    embedding: Optional[list[float]] = Field(default=None, description="Semantic vector embedding")


class DocumentMetadataInterface(BaseModel):
    """High-level metadata for ingested document."""
    document_id: str = Field(..., description="Unique document ID")
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    title: str = Field(..., description="Document title")
    source_filename: str = Field(..., description="Source file name")
    total_pages: int = Field(..., description="Total page count")
    total_chunks: int = Field(..., description="Total generated chunk count")
    created_at: str = Field(..., description="Ingestion ISO timestamp")


class IngestionJobInterface(BaseModel):
    """Tracking entity for an individual document ingestion job run."""
    job_id: str = Field(..., description="Unique ingestion job identifier")
    document_id: str = Field(..., description="Target document identifier")
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    status: DocumentStatusEnum = Field(default=DocumentStatusEnum.PENDING, description="Job status")
    stage: IngestionStageEnum = Field(default=IngestionStageEnum.QUEUED, description="Current workflow stage")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="Job progress percentage")
    retry_count: int = Field(default=0, ge=0, description="Current retry attempt count")
    max_retries: int = Field(default=3, ge=0, description="Configured maximum retry attempts")
    error_message: Optional[str] = Field(default=None, description="Failure reason if failed")
    correlation_id: Optional[str] = Field(default=None, description="Correlation / request identifier")
    created_at: str = Field(..., description="Job creation ISO timestamp")
    started_at: Optional[str] = Field(default=None, description="Job start ISO timestamp")
    completed_at: Optional[str] = Field(default=None, description="Job completion ISO timestamp")


class DocumentStatusInterface(BaseModel):
    """Tracking status for async document ingestion job."""
    job_id: Optional[str] = Field(default=None, description="Ingestion job ID")
    document_id: str = Field(..., description="Unique document ID")
    tenant_id: str = Field(default="default_tenant", description="Tenant identifier for isolation")
    status: DocumentStatusEnum = Field(..., description="Current ingestion state")
    stage: Optional[IngestionStageEnum] = Field(default=None, description="Current workflow stage")
    progress_percent: Optional[float] = Field(default=0.0, description="Ingestion progress percentage")
    error_message: Optional[str] = Field(default=None, description="Failure reason if FAILED")
    processed_chunks: int = Field(default=0, description="Count of processed chunks so far")
    correlation_id: Optional[str] = Field(default=None, description="Trace correlation ID")
