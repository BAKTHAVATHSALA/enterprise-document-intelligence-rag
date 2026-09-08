"""Services Layer Package.

Exports services for document ingestion management, queue coordination, and RAG query execution.
"""

from services.document_service import (
    create_ingestion_job,
    create_pdf_ingestion_job,
    enqueue_pdf_ingestion_job,
    cancel_ingestion_job,
    delete_document,
    get_ingestion_job,
    get_document_metadata,
    get_document_status,
    list_all_documents,
)
from services.query_service import (
    execute_query_workflow,
    validate_query_security,
)
from services.queue_service import (
    JobQueueInterface,
    AsyncInMemoryJobQueue,
    IngestionJobPayload,
    get_job_queue,
    set_job_queue,
)

__all__ = [
    "create_ingestion_job",
    "create_pdf_ingestion_job",
    "enqueue_pdf_ingestion_job",
    "cancel_ingestion_job",
    "delete_document",
    "get_ingestion_job",
    "get_document_metadata",
    "get_document_status",
    "list_all_documents",
    "execute_query_workflow",
    "validate_query_security",
    "JobQueueInterface",
    "AsyncInMemoryJobQueue",
    "IngestionJobPayload",
    "get_job_queue",
    "set_job_queue",
]

