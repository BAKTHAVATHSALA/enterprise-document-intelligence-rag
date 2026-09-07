"""FastAPI Application Main Entry Point.

Exposes REST API endpoints for multipart PDF document ingestion, status tracking,
metadata retrieval, cancellation, and natural language Hybrid RAG query processing.

Integrates:
- Asynchronous document ingestion with 202 Accepted + job_id response.
- Background IngestionWorker outside the HTTP request lifecycle.
- Structured JSON logging with request correlation IDs (X-Correlation-ID).
- Centralized RAGError taxonomy exception handlers.
- Cooperative cancellation endpoints (/documents/{id}/cancel and /jobs/{id}/cancel).
"""

import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from interfaces import (
    DocumentStatusInterface,
    DocumentMetadataInterface,
    IngestionJobInterface,
    QueryRequestInterface,
    QueryResponseInterface,
)
from services import (
    create_ingestion_job,
    create_pdf_ingestion_job,
    enqueue_pdf_ingestion_job,
    cancel_ingestion_job,
    get_ingestion_job,
    get_document_metadata,
    get_document_status,
    execute_query_workflow,
    get_job_queue,
)
from workers.ingestion_worker import IngestionWorker
from data_access import ensure_db_schema, init_bm25_from_db
from utils.logger import (
    logger,
    set_correlation_id,
    get_correlation_id,
    set_document_id,
    get_document_id,
    set_query_id,
    get_query_id,
    sanitize_message_text,
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

FUNC_API_INGEST: str = "api_create_document"
FUNC_API_GET_META: str = "api_get_document_metadata"
FUNC_API_GET_STATUS: str = "api_get_document_status"
FUNC_API_GET_JOB: str = "api_get_job"
FUNC_API_CANCEL: str = "api_cancel_job"
FUNC_API_QUERY: str = "api_execute_query"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan context manager.

    Verifies PostgreSQL database schema, hydates BM25 in-memory index on application startup,
    and manages background IngestionWorker task.
    """
    logger.info("lifespan_startup", "Verifying PostgreSQL schema and initializing tables...")
    schema_ok: bool = ensure_db_schema()
    if not schema_ok:
        logger.warning("lifespan_startup", "PostgreSQL database connection or schema verification returned false.")

    count: int = init_bm25_from_db()
    logger.info("lifespan_startup", f"BM25 keyword search index hydrated with {count} chunks from PostgreSQL.")

    # Start asynchronous ingestion worker
    worker = IngestionWorker(queue=get_job_queue())
    worker.start()

    yield

    # Cleanly stop worker on application shutdown
    await worker.stop()


app = FastAPI(
    title="Enterprise Document Intelligence & Hybrid RAG Platform",
    description="Production-grade AI platform for PDF layout document understanding, hybrid retrieval, and grounded Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Assign or propagate request-scoped correlation ID and record API latency."""
    incoming_corr: Optional[str] = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID")
    corr_id: str = incoming_corr.strip() if (incoming_corr and incoming_corr.strip()) else f"corr_{uuid.uuid4().hex[:12]}"

    set_correlation_id(corr_id)
    set_document_id(None)
    set_query_id(None)

    start_time: float = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms: float = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Correlation-ID"] = corr_id
        response.headers["X-Request-ID"] = corr_id

        op_name: str = f"api_{request.method.lower()}_{request.url.path.strip('/').replace('/', '_') or 'root'}"
        logger.info(
            func_name=op_name,
            message=f"{request.method} {request.url.path} returned {response.status_code} in {duration_ms}ms",
            correlation_id=corr_id,
            duration_ms=duration_ms,
            status="success" if response.status_code < 400 else "failure",
        )
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        op_name = f"api_{request.method.lower()}_{request.url.path.strip('/').replace('/', '_') or 'root'}"
        logger.error(
            func_name=op_name,
            message=f"Unhandled failure handling {request.method} {request.url.path}: {exc}",
            exc=exc,
            correlation_id=corr_id,
            duration_ms=duration_ms,
            status="failure",
        )
        raise exc


# Centralized Exception Handlers
@app.exception_handler(RAGError)
async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    """Handle structured RAGError taxonomy exceptions with safe sanitized responses."""
    corr_id: str = exc.correlation_id or get_correlation_id() or f"corr_{uuid.uuid4().hex[:12]}"
    clean_msg: str = sanitize_message_text(exc.message)

    content = {
        "detail": clean_msg,
        "error_type": exc.error_code,
        "correlation_id": corr_id,
        "status_code": exc.status_code,
    }
    if exc.details:
        content["details"] = exc.details

    headers = {"X-Correlation-ID": corr_id}
    return JSONResponse(status_code=exc.status_code, content=content, headers=headers)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Format HTTPException with correlation ID while preserving 'detail' string contract."""
    corr_id: str = get_correlation_id() or f"corr_{uuid.uuid4().hex[:12]}"
    clean_detail = sanitize_message_text(str(exc.detail)) if isinstance(exc.detail, str) else exc.detail

    headers = {"X-Correlation-ID": corr_id}
    if exc.headers:
        headers.update(exc.headers)

    content = {
        "detail": clean_detail,
        "error_type": "HTTP_EXCEPTION",
        "correlation_id": corr_id,
        "status_code": exc.status_code,
    }
    return JSONResponse(status_code=exc.status_code, content=content, headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all unhandled exception handler ensuring zero secret leaks in 500 responses."""
    corr_id: str = get_correlation_id() or f"corr_{uuid.uuid4().hex[:12]}"
    clean_err = sanitize_message_text(str(exc))

    logger.error(
        func_name="unhandled_exception_handler",
        message=f"Unhandled internal server error: {clean_err}",
        exc=exc,
        correlation_id=corr_id,
        status="failure",
        error_type=type(exc).__name__,
    )

    headers = {"X-Correlation-ID": corr_id}
    content = {
        "detail": f"Internal server error: {clean_err}" if clean_err else "Internal server error.",
        "error_type": "INTERNAL_SERVER_ERROR",
        "correlation_id": corr_id,
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=content, headers=headers)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "service": "Hybrid RAG Platform API"}


@app.post("/documents", response_model=DocumentStatusInterface, status_code=status.HTTP_202_ACCEPTED)
async def create_document_job(
    file: UploadFile = File(..., description="Uploaded PDF file document"),
    title: Optional[str] = Form(default=None, description="Optional document title override"),
) -> DocumentStatusInterface:
    """Accept and asynchronously enqueue a PDF document ingestion job.

    Returns HTTP 202 Accepted with job_id and document_id immediately.
    """
    filename: str = file.filename or "uploaded_document.pdf"

    # 1. Extension Validation
    if not filename.lower().endswith(".pdf"):
        logger.warning(FUNC_API_INGEST, f"Rejected file '{filename}': Not a .pdf extension")
        raise ValidationError(f"Invalid file type for '{filename}'. Only .pdf files are supported.")

    # 2. Content Type Validation
    if file.content_type and "pdf" not in file.content_type.lower() and file.content_type != "application/octet-stream":
        logger.warning(FUNC_API_INGEST, f"Rejected file '{filename}': Invalid content-type '{file.content_type}'")
        raise ValidationError(f"Invalid content-type '{file.content_type}'. Must be 'application/pdf'.")

    try:
        pdf_bytes: bytes = await file.read()

        # 3. Empty File Validation
        if not pdf_bytes or len(pdf_bytes) == 0:
            logger.warning(FUNC_API_INGEST, f"Rejected empty PDF file '{filename}' (0 bytes)")
            raise ValidationError(f"Uploaded file '{filename}' is empty (0 bytes).")

        # 4. PDF Magic Header Validation (%PDF-)
        if not pdf_bytes.startswith(b"%PDF-"):
            logger.warning(FUNC_API_INGEST, f"Rejected file '{filename}': Missing or invalid PDF magic header")
            raise ValidationError(f"Invalid file content for '{filename}'. File does not have a valid PDF header (%PDF-).")

        # 5. Enqueue Asynchronous Ingestion Job
        status_res: DocumentStatusInterface = await enqueue_pdf_ingestion_job(
            pdf_bytes=pdf_bytes,
            filename=filename,
            title=title or filename,
            correlation_id=get_correlation_id(),
        )

        set_document_id(status_res.document_id)
        logger.info(
            FUNC_API_INGEST,
            f"Accepted and enqueued PDF upload '{filename}': doc_id={status_res.document_id}, job_id={status_res.job_id}",
            document_id=status_res.document_id,
        )
        return status_res

    except RAGError:
        raise
    except HTTPException:
        raise
    except ValueError as val_err:
        logger.warning(FUNC_API_INGEST, f"Validation error processing PDF '{filename}': {val_err}")
        raise ValidationError(str(val_err))
    except Exception as exc:
        logger.error(FUNC_API_INGEST, f"Unhandled failure ingesting PDF '{filename}'", exc=exc)
        raise IngestionError(f"Failed to process PDF: {exc}")


@app.get("/documents/{document_id}", response_model=DocumentMetadataInterface)
def read_document_metadata(document_id: str) -> DocumentMetadataInterface:
    """Return document metadata for given document_id."""
    set_document_id(document_id)
    meta: Optional[DocumentMetadataInterface] = get_document_metadata(document_id)
    if not meta:
        logger.warning(FUNC_API_GET_META, f"Document not found: {document_id}", document_id=document_id)
        raise DocumentNotFoundError(f"Document '{document_id}' not found", document_id=document_id)

    return meta


@app.get("/documents/{document_id}/status", response_model=DocumentStatusInterface)
def read_document_status(document_id: str) -> DocumentStatusInterface:
    """Return processing job status for given document_id."""
    set_document_id(document_id)
    job_status: Optional[DocumentStatusInterface] = get_document_status(document_id)
    if not job_status:
        logger.warning(FUNC_API_GET_STATUS, f"Document status not found: {document_id}", document_id=document_id)
        raise DocumentNotFoundError(f"Document '{document_id}' not found", document_id=document_id)

    return job_status


@app.get("/jobs/{job_id}", response_model=IngestionJobInterface)
def read_job_details(job_id: str) -> IngestionJobInterface:
    """Return ingestion job details for given job_id."""
    job: Optional[IngestionJobInterface] = get_ingestion_job(job_id)
    if not job:
        logger.warning(FUNC_API_GET_JOB, f"Job not found: {job_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found")

    return job


@app.post("/documents/{document_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_document_job(document_id: str) -> dict[str, str]:
    """Cancel ongoing or queued document ingestion job by document_id."""
    success: bool = await cancel_ingestion_job(document_id=document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active job found to cancel for document '{document_id}'",
        )
    return {"status": "cancelled", "document_id": document_id}


@app.post("/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job_by_id(job_id: str) -> dict[str, str]:
    """Cancel ongoing or queued ingestion job by job_id."""
    success: bool = await cancel_ingestion_job(job_id=job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or cannot be cancelled",
        )
    return {"status": "cancelled", "job_id": job_id}


@app.post("/query", response_model=QueryResponseInterface)
def query_rag_pipeline(payload: QueryRequestInterface) -> QueryResponseInterface:
    """Answer natural-language question using Hybrid RAG pipeline and validated citations."""
    query_id = getattr(payload, "query_id", None) or f"query_{uuid.uuid4().hex[:8]}"
    set_query_id(query_id)
    if payload.document_ids and len(payload.document_ids) == 1:
        set_document_id(payload.document_ids[0])

    try:
        response: QueryResponseInterface = execute_query_workflow(payload)
        logger.info(
            FUNC_API_QUERY,
            f"Successfully processed query in {response.processing_time_ms}ms",
            query_id=query_id,
            duration_ms=response.processing_time_ms,
        )
        return response
    except RAGError:
        raise
    except ValueError as val_err:
        logger.warning(FUNC_API_QUERY, f"Validation error during query execution: {val_err}", query_id=query_id)
        if "security policy" in str(val_err).lower():
            raise SecurityValidationError(str(val_err))
        raise ValidationError(str(val_err))
    except Exception as exc:
        logger.error(FUNC_API_QUERY, "Unhandled error during query pipeline execution", exc=exc, query_id=query_id)
        raise GenerationError(f"Query pipeline failure: {exc}")
