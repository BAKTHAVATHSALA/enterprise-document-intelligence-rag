"""Document Service Layer.

Manages real PDF document ingestion workflows, storage indexing side effects,
and document job status tracking.
"""

from typing import Optional
from interfaces.document_interface import (
    DocumentMetadataInterface,
    DocumentStatusInterface,
    DocumentStatusEnum,
)
from helpers.document_helper import (
    process_document_pipeline,
    process_pdf_pipeline,
    ProcessedDocumentResult,
)
from data_access import (
    upsert_vector_chunks,
    index_bm25_chunks,
    upsert_graph_nodes,
)
from utils.logger import logger

FUNC_CREATE_JOB: str = "create_ingestion_job"
FUNC_CREATE_PDF_JOB: str = "create_pdf_ingestion_job"
FUNC_GET_METADATA: str = "get_document_metadata"
FUNC_GET_STATUS: str = "get_document_status"

# In-memory document status and metadata registries
_DOC_METADATA_STORE: dict[str, DocumentMetadataInterface] = {}
_DOC_STATUS_STORE: dict[str, DocumentStatusInterface] = {}


def create_pdf_ingestion_job(
    pdf_bytes: bytes,
    filename: str,
    title: str | None = None,
) -> DocumentStatusInterface:
    """Execute real PDF document ingestion workflow via Docling.

    @param pdf_bytes: Raw binary content of uploaded PDF file.
    @param filename: Name of uploaded file.
    @param title: Document title string.
    @returns: DocumentStatusInterface job tracking object.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid file extension. Document must be a .pdf file.")

    # 1. Process PDF layout using Docling, PII redaction, entity extraction, and chunking
    pipeline_res: ProcessedDocumentResult = process_pdf_pipeline(
        pdf_bytes=pdf_bytes,
        filename=filename,
        title=title,
    )

    doc_id: str = pipeline_res.metadata.document_id

    # Initialize status tracking
    status_obj: DocumentStatusInterface = DocumentStatusInterface(
        document_id=doc_id,
        status=DocumentStatusEnum.PROCESSING,
        processed_chunks=0,
    )
    _DOC_STATUS_STORE[doc_id] = status_obj
    _DOC_METADATA_STORE[doc_id] = pipeline_res.metadata

    # 2. Side-effecting operations (Storage indexing) wrapped in try/catch at service layer
    try:
        upsert_vector_chunks(pipeline_res.chunks)
        index_bm25_chunks(pipeline_res.chunks)
        upsert_graph_nodes(pipeline_res.chunks, pipeline_res.entities)

        status_obj.status = DocumentStatusEnum.COMPLETED
        status_obj.processed_chunks = len(pipeline_res.chunks)
        _DOC_STATUS_STORE[doc_id] = status_obj

        logger.info(FUNC_CREATE_PDF_JOB, f"PDF ingestion job completed for document: {doc_id} ('{filename}')")
        return status_obj

    except Exception as exc:
        status_obj.status = DocumentStatusEnum.FAILED
        status_obj.error_message = str(exc)
        _DOC_STATUS_STORE[doc_id] = status_obj
        logger.error(FUNC_CREATE_PDF_JOB, f"PDF ingestion job failed for document: {doc_id}", exc=exc)
        raise exc


def create_ingestion_job(
    raw_text: str,
    title: str,
    source_filename: str,
) -> DocumentStatusInterface:
    """Execute text document ingestion workflow and index across multi-modal stores."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Document payload text cannot be empty.")

    pipeline_res: ProcessedDocumentResult = process_document_pipeline(
        raw_text=raw_text,
        title=title,
        source_filename=source_filename,
    )

    doc_id: str = pipeline_res.metadata.document_id

    status_obj: DocumentStatusInterface = DocumentStatusInterface(
        document_id=doc_id,
        status=DocumentStatusEnum.PROCESSING,
        processed_chunks=0,
    )
    _DOC_STATUS_STORE[doc_id] = status_obj
    _DOC_METADATA_STORE[doc_id] = pipeline_res.metadata

    try:
        upsert_vector_chunks(pipeline_res.chunks)
        index_bm25_chunks(pipeline_res.chunks)
        upsert_graph_nodes(pipeline_res.chunks, pipeline_res.entities)

        status_obj.status = DocumentStatusEnum.COMPLETED
        status_obj.processed_chunks = len(pipeline_res.chunks)
        _DOC_STATUS_STORE[doc_id] = status_obj

        logger.info(FUNC_CREATE_JOB, f"Ingestion job completed successfully for document: {doc_id}")
        return status_obj

    except Exception as exc:
        status_obj.status = DocumentStatusEnum.FAILED
        status_obj.error_message = str(exc)
        _DOC_STATUS_STORE[doc_id] = status_obj
        logger.error(FUNC_CREATE_JOB, f"Ingestion job failed for document: {doc_id}", exc=exc)
        raise exc


def get_document_metadata(document_id: str) -> Optional[DocumentMetadataInterface]:
    """Retrieve document metadata by document ID."""
    if not document_id:
        return None

    meta: DocumentMetadataInterface | None = _DOC_METADATA_STORE.get(document_id)
    if meta:
        logger.info(FUNC_GET_METADATA, f"Retrieved metadata for document: {document_id}")
    return meta


def get_document_status(document_id: str) -> Optional[DocumentStatusInterface]:
    """Retrieve document ingestion job status."""
    if not document_id:
        return None

    status: DocumentStatusInterface | None = _DOC_STATUS_STORE.get(document_id)
    if status:
        logger.info(FUNC_GET_STATUS, f"Retrieved status for document: {document_id}")
    return status
