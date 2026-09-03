"""FastAPI Application Main Entry Point.

Exposes REST API endpoints for multipart PDF document ingestion, status tracking,
metadata retrieval, and natural language Hybrid RAG query processing.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from interfaces import (
    DocumentStatusInterface,
    DocumentMetadataInterface,
    QueryRequestInterface,
    QueryResponseInterface,
)
from services import (
    create_ingestion_job,
    create_pdf_ingestion_job,
    get_document_metadata,
    get_document_status,
    execute_query_workflow,
)
from utils.logger import logger

FUNC_API_INGEST: str = "api_create_document"
FUNC_API_GET_META: str = "api_get_document_metadata"
FUNC_API_GET_STATUS: str = "api_get_document_status"
FUNC_API_QUERY: str = "api_execute_query"

app = FastAPI(
    title="Enterprise Document Intelligence & Hybrid RAG Platform",
    description="Production-grade AI platform for PDF layout document understanding, hybrid retrieval, and grounded Q&A.",
    version="1.0.0",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "service": "Hybrid RAG Platform API"}


@app.post("/documents", response_model=DocumentStatusInterface, status_code=status.HTTP_201_CREATED)
async def create_document_job(
    file: UploadFile = File(..., description="Uploaded PDF file document"),
    title: Optional[str] = Form(default=None, description="Optional document title override"),
) -> DocumentStatusInterface:
    """Create a PDF document ingestion job using Docling layout parsing."""
    filename: str = file.filename or "uploaded_document.pdf"

    # 1. Extension Validation
    if not filename.lower().endswith(".pdf"):
        logger.warning(FUNC_API_INGEST, f"Rejected file '{filename}': Not a .pdf extension")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only .pdf files are supported.",
        )

    # 2. Content Type Validation
    if file.content_type and "pdf" not in file.content_type.lower() and file.content_type != "application/octet-stream":
        logger.warning(FUNC_API_INGEST, f"Rejected file '{filename}': Invalid content-type '{file.content_type}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content-type '{file.content_type}'. Must be 'application/pdf'.",
        )

    try:
        pdf_bytes: bytes = await file.read()

        # 3. Empty File Validation
        if not pdf_bytes or len(pdf_bytes) == 0:
            logger.warning(FUNC_API_INGEST, f"Rejected empty PDF file '{filename}' (0 bytes)")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is empty (0 bytes).",
            )

        status_res: DocumentStatusInterface = create_pdf_ingestion_job(
            pdf_bytes=pdf_bytes,
            filename=filename,
            title=title or filename,
        )

        logger.info(FUNC_API_INGEST, f"Successfully processed PDF upload '{filename}': doc_id={status_res.document_id}")
        return status_res

    except HTTPException:
        raise
    except ValueError as val_err:
        logger.warning(FUNC_API_INGEST, f"Validation error processing PDF '{filename}': {val_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error(FUNC_API_INGEST, f"Unhandled failure ingesting PDF '{filename}'", exc=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process PDF: {exc}")


@app.get("/documents/{document_id}", response_model=DocumentMetadataInterface)
def read_document_metadata(document_id: str) -> DocumentMetadataInterface:
    """Return document metadata for given document_id."""
    meta: Optional[DocumentMetadataInterface] = get_document_metadata(document_id)
    if not meta:
        logger.warning(FUNC_API_GET_META, f"Document not found: {document_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found")

    return meta


@app.get("/documents/{document_id}/status", response_model=DocumentStatusInterface)
def read_document_status(document_id: str) -> DocumentStatusInterface:
    """Return processing job status for given document_id."""
    job_status: Optional[DocumentStatusInterface] = get_document_status(document_id)
    if not job_status:
        logger.warning(FUNC_API_GET_STATUS, f"Document status not found: {document_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found")

    return job_status


@app.post("/query", response_model=QueryResponseInterface)
def query_rag_pipeline(payload: QueryRequestInterface) -> QueryResponseInterface:
    """Answer natural-language question using Hybrid RAG pipeline and validated citations."""
    try:
        response: QueryResponseInterface = execute_query_workflow(payload)
        logger.info(FUNC_API_QUERY, f"Successfully processed query in {response.processing_time_ms}ms")
        return response
    except ValueError as val_err:
        logger.warning(FUNC_API_QUERY, f"Validation error during query execution: {val_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        logger.error(FUNC_API_QUERY, "Unhandled error during query pipeline execution", exc=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
