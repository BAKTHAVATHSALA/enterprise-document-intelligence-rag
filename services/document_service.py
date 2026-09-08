"""Document Service Layer.

Coordinates asynchronous PDF document ingestion workflows, queue dispatching,
cancellation handling, storage indexing side effects, and status tracking.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from interfaces.document_interface import (
    DocumentMetadataInterface,
    DocumentStatusInterface,
    DocumentStatusEnum,
    IngestionStageEnum,
    IngestionJobInterface,
)
from helpers.document_helper import (
    process_document_pipeline,
    process_pdf_pipeline,
    ProcessedDocumentResult,
)
from services.queue_service import (
    IngestionJobPayload,
    get_job_queue,
)
from data_access import (
    upsert_vector_chunks,
    delete_vector_chunks_by_document_id,
    index_bm25_chunks,
    evict_document_from_bm25,
    upsert_graph_nodes,
    delete_graph_nodes_by_document_id,
    save_document_to_db,
    save_chunks_to_db,
    update_document_status_in_db,
    get_document_metadata_from_db,
    get_document_status_from_db,
    save_ingestion_job_to_db,
    update_job_progress_in_db,
    get_ingestion_job_from_db,
    get_latest_job_for_document_from_db,
    delete_document_from_db,
)
from utils.logger import logger, get_correlation_id

FUNC_CREATE_JOB: str = "create_ingestion_job"
FUNC_ENQUEUE_JOB: str = "enqueue_pdf_ingestion_job"
FUNC_CREATE_PDF_JOB: str = "create_pdf_ingestion_job"
FUNC_GET_METADATA: str = "get_document_metadata"
FUNC_GET_STATUS: str = "get_document_status"
FUNC_CANCEL_JOB: str = "cancel_ingestion_job"
FUNC_DELETE_DOC: str = "delete_document"


# Fallback in-memory registries for offline testing
_DOC_METADATA_STORE: dict[str, DocumentMetadataInterface] = {}
_DOC_STATUS_STORE: dict[str, DocumentStatusInterface] = {}


async def enqueue_pdf_ingestion_job(
    pdf_bytes: bytes,
    filename: str,
    title: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> DocumentStatusInterface:
    """Validate, record, and asynchronously enqueue PDF document ingestion job.

    Returns HTTP 202 Accepted payload with job_id and document_id immediately.

    @param pdf_bytes: Raw binary content of uploaded PDF file.
    @param filename: Uploaded file name.
    @param title: Optional document title.
    @param correlation_id: Optional request correlation ID.
    @returns: DocumentStatusInterface with status=PENDING and stage=QUEUED.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid file extension. Document must be a .pdf file.")

    doc_id: str = f"doc_{uuid.uuid4().hex[:12]}"
    job_id: str = f"job_{uuid.uuid4().hex[:12]}"
    tenant_id: str = "default_tenant"
    doc_title: str = title if title else filename
    now_iso: str = datetime.now(timezone.utc).isoformat()
    active_corr: Optional[str] = correlation_id or get_correlation_id()

    # 1. Initialize persistent Document record in PostgreSQL
    initial_meta: DocumentMetadataInterface = DocumentMetadataInterface(
        document_id=doc_id,
        tenant_id=tenant_id,
        title=doc_title,
        source_filename=filename,
        total_pages=1,
        total_chunks=0,
        created_at=now_iso,
    )
    await asyncio.to_thread(save_document_to_db, initial_meta)
    _DOC_METADATA_STORE[doc_id] = initial_meta

    # 2. Initialize persistent IngestionJob record in PostgreSQL
    job_record: IngestionJobInterface = IngestionJobInterface(
        job_id=job_id,
        document_id=doc_id,
        tenant_id=tenant_id,
        status=DocumentStatusEnum.PENDING,
        stage=IngestionStageEnum.QUEUED,
        progress_percent=0.0,
        retry_count=0,
        max_retries=3,
        correlation_id=active_corr,
        created_at=now_iso,
    )
    await asyncio.to_thread(save_ingestion_job_to_db, job_record)

    # 3. Build status object
    status_obj: DocumentStatusInterface = DocumentStatusInterface(
        job_id=job_id,
        document_id=doc_id,
        tenant_id=tenant_id,
        status=DocumentStatusEnum.PENDING,
        stage=IngestionStageEnum.QUEUED,
        progress_percent=0.0,
        processed_chunks=0,
        correlation_id=active_corr,
    )
    _DOC_STATUS_STORE[doc_id] = status_obj

    # 4. Push payload to Queue
    payload = IngestionJobPayload(
        job_id=job_id,
        document_id=doc_id,
        tenant_id=tenant_id,
        filename=filename,
        title=doc_title,
        pdf_bytes=pdf_bytes,
        correlation_id=active_corr,
    )

    from workers.ingestion_worker import get_or_start_worker
    get_or_start_worker()

    await get_job_queue().enqueue(payload)

    logger.info(
        FUNC_ENQUEUE_JOB,
        f"Enqueued async ingestion job {job_id} for doc {doc_id} ('{filename}')",
        correlation_id=active_corr,
        document_id=doc_id,
    )
    return status_obj


async def cancel_ingestion_job(
    job_id: Optional[str] = None,
    document_id: Optional[str] = None,
) -> bool:
    """Request cooperative cancellation for an enqueued or in-progress ingestion job.

    @param job_id: Target job ID to cancel.
    @param document_id: Target document ID whose latest job will be cancelled.
    @returns: True if cancellation was registered, False otherwise.
    """
    queue = get_job_queue()
    target_job_id: Optional[str] = job_id
    target_doc_id: Optional[str] = document_id

    if not target_job_id and target_doc_id:
        latest_job = await asyncio.to_thread(get_latest_job_for_document_from_db, target_doc_id)
        if latest_job:
            target_job_id = latest_job.job_id
        else:
            status_entry = _DOC_STATUS_STORE.get(target_doc_id)
            if status_entry:
                target_job_id = status_entry.job_id

    if not target_job_id:
        return False

    # Mark cancelled in queue
    await queue.cancel_job(target_job_id)

    # Update DB status
    now_iso = datetime.now(timezone.utc).isoformat()
    await asyncio.to_thread(
        update_job_progress_in_db,
        job_id=target_job_id,
        status=DocumentStatusEnum.CANCELLED,
        stage=IngestionStageEnum.CANCELLED,
        progress_percent=0.0,
        error_message="Job cancelled by user request.",
        completed_at=now_iso,
    )

    if target_doc_id:
        await asyncio.to_thread(
            update_document_status_in_db,
            target_doc_id,
            DocumentStatusEnum.CANCELLED,
            0,
            "Job cancelled by user request.",
        )
        if target_doc_id in _DOC_STATUS_STORE:
            _DOC_STATUS_STORE[target_doc_id].status = DocumentStatusEnum.CANCELLED

    logger.info(FUNC_CANCEL_JOB, f"Successfully requested cancellation for job {target_job_id}")
    return True



def get_ingestion_job(job_id: str) -> Optional[IngestionJobInterface]:
    """Retrieve ingestion job details by job_id."""
    return get_ingestion_job_from_db(job_id)


def create_pdf_ingestion_job(
    pdf_bytes: bytes,
    filename: str,
    title: str | None = None,
) -> DocumentStatusInterface:
    """Execute synchronous PDF document ingestion workflow (legacy fallback)."""
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise ValueError("Uploaded PDF file is empty (0 bytes).")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Invalid file extension. Document must be a .pdf file.")

    pipeline_res: ProcessedDocumentResult = process_pdf_pipeline(
        pdf_bytes=pdf_bytes,
        filename=filename,
        title=title,
    )

    doc_id: str = pipeline_res.metadata.document_id
    status_obj: DocumentStatusInterface = DocumentStatusInterface(
        document_id=doc_id,
        tenant_id=pipeline_res.metadata.tenant_id,
        status=DocumentStatusEnum.PROCESSING,
        processed_chunks=0,
    )
    _DOC_STATUS_STORE[doc_id] = status_obj
    _DOC_METADATA_STORE[doc_id] = pipeline_res.metadata

    save_document_to_db(pipeline_res.metadata)
    save_chunks_to_db(pipeline_res.chunks)

    try:
        upsert_vector_chunks(pipeline_res.chunks)
        index_bm25_chunks(pipeline_res.chunks)
        upsert_graph_nodes(pipeline_res.chunks, pipeline_res.entities)

        status_obj.status = DocumentStatusEnum.COMPLETED
        status_obj.processed_chunks = len(pipeline_res.chunks)
        _DOC_STATUS_STORE[doc_id] = status_obj
        update_document_status_in_db(doc_id, DocumentStatusEnum.COMPLETED, len(pipeline_res.chunks))

        logger.info(FUNC_CREATE_PDF_JOB, f"PDF ingestion job completed for document: {doc_id} ('{filename}')")
        return status_obj

    except Exception as exc:
        status_obj.status = DocumentStatusEnum.FAILED
        status_obj.error_message = str(exc)
        _DOC_STATUS_STORE[doc_id] = status_obj
        update_document_status_in_db(doc_id, DocumentStatusEnum.FAILED, 0, str(exc))
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
        tenant_id=pipeline_res.metadata.tenant_id,
        status=DocumentStatusEnum.PROCESSING,
        processed_chunks=0,
    )
    _DOC_STATUS_STORE[doc_id] = status_obj
    _DOC_METADATA_STORE[doc_id] = pipeline_res.metadata

    save_document_to_db(pipeline_res.metadata)
    save_chunks_to_db(pipeline_res.chunks)

    try:
        upsert_vector_chunks(pipeline_res.chunks)
        index_bm25_chunks(pipeline_res.chunks)
        upsert_graph_nodes(pipeline_res.chunks, pipeline_res.entities)

        status_obj.status = DocumentStatusEnum.COMPLETED
        status_obj.processed_chunks = len(pipeline_res.chunks)
        _DOC_STATUS_STORE[doc_id] = status_obj
        update_document_status_in_db(doc_id, DocumentStatusEnum.COMPLETED, len(pipeline_res.chunks))

        logger.info(FUNC_CREATE_JOB, f"Ingestion job completed successfully for document: {doc_id}")
        return status_obj

    except Exception as exc:
        status_obj.status = DocumentStatusEnum.FAILED
        status_obj.error_message = str(exc)
        _DOC_STATUS_STORE[doc_id] = status_obj
        update_document_status_in_db(doc_id, DocumentStatusEnum.FAILED, 0, str(exc))
        logger.error(FUNC_CREATE_JOB, f"Ingestion job failed for document: {doc_id}", exc=exc)
        raise exc


def get_document_metadata(document_id: str) -> Optional[DocumentMetadataInterface]:
    """Retrieve document metadata by document ID from PostgreSQL or local fallback."""
    if not document_id:
        return None

    db_meta = get_document_metadata_from_db(document_id)
    if db_meta:
        return db_meta

    return _DOC_METADATA_STORE.get(document_id)


def get_document_status(document_id: str) -> Optional[DocumentStatusInterface]:
    """Retrieve document ingestion job status from PostgreSQL or local fallback."""
    if not document_id:
        return None

    db_status = get_document_status_from_db(document_id)
    if db_status:
        return db_status

    return _DOC_STATUS_STORE.get(document_id)


def list_all_documents(tenant_id: Optional[str] = None) -> list[dict]:
    """Retrieve list of all documents with status, stage, progress, and metadata."""
    from data_access import list_documents_from_db
    db_docs = list_documents_from_db(tenant_id)
    if db_docs:
        return db_docs

    # Fallback to local memory store
    results = []
    for doc_id, meta in _DOC_METADATA_STORE.items():
        if tenant_id and meta.tenant_id != tenant_id:
            continue
        status_entry = _DOC_STATUS_STORE.get(doc_id)
        results.append({
            "document_id": doc_id,
            "tenant_id": meta.tenant_id,
            "title": meta.title,
            "source_filename": meta.source_filename,
            "total_pages": meta.total_pages,
            "total_chunks": meta.total_chunks,
            "status": status_entry.status.value if status_entry else "COMPLETED",
            "stage": status_entry.stage.value if status_entry and status_entry.stage else ("COMPLETED" if (status_entry and status_entry.status == DocumentStatusEnum.COMPLETED) else "QUEUED"),
            "progress_percent": status_entry.progress_percent if status_entry and status_entry.progress_percent is not None else 100.0,
            "error_message": status_entry.error_message if status_entry else None,
            "created_at": meta.created_at,
            "job_id": status_entry.job_id if status_entry else None,
        })
    return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)


async def delete_document(document_id: str) -> bool:
    """Safely and atomically delete a document and all of its associated data across all stores.

    Document-scoped deletion:
    1. First requests cooperative cancellation of any active/queued ingestion job.
    2. Removes vector embeddings from Pinecone and mock vector store.
    3. Removes Document/Chunk/Entity nodes from Neo4j graph store.
    4. Evicts chunks from in-memory BM25 index.
    5. Removes document, chunks, and ingestion jobs from PostgreSQL database.
    6. Removes local fallback memory store entries.

    @param document_id: Unique document identifier string.
    @returns: True if document was found and deleted, False otherwise.
    """
    if not document_id:
        return False

    meta = get_document_metadata(document_id)
    status_entry = get_document_status(document_id)
    if not meta and not status_entry:
        logger.warning(FUNC_DELETE_DOC, f"Cannot delete document '{document_id}': Document not found.")
        return False

    # 1. Cooperative cancellation if job is active/queued
    try:
        await cancel_ingestion_job(document_id=document_id)
    except Exception as exc:
        logger.warning(FUNC_DELETE_DOC, f"Cancellation attempt during delete for '{document_id}' issued warning: {exc}")

    # 2. Delete document vector chunks from Pinecone
    delete_vector_chunks_by_document_id(document_id)

    # 3. Delete document nodes & relationships from Neo4j
    delete_graph_nodes_by_document_id(document_id)

    # 4. Evict document chunks from BM25 index
    evict_document_from_bm25(document_id)

    # 5. Delete document record from PostgreSQL (cascades to chunks & ingestion_jobs)
    db_deleted = delete_document_from_db(document_id)

    # 6. Evict from local fallback memory store
    _DOC_METADATA_STORE.pop(document_id, None)
    _DOC_STATUS_STORE.pop(document_id, None)

    logger.info(FUNC_DELETE_DOC, f"Successfully executed document-scoped delete for '{document_id}' (db_deleted={db_deleted}).")
    return True

