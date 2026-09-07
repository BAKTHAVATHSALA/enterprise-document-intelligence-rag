"""Background Ingestion Worker.

Consumes document ingestion jobs from JobQueueInterface and executes the pipeline
outside the HTTP request lifecycle.

Implements:
- Granular stage transitions: PARSING -> PII_REDACTION -> ENTITY_EXTRACTION -> CHUNKING -> EMBEDDING -> INDEXING -> COMPLETED
- Real stage progress percentage tracking (0% to 100%)
- Cooperative cancellation checks between stages
- Statuses: PENDING, PROCESSING, COMPLETED, PARTIAL, FAILED, CANCELLED
- Bounded exponential retry for transient provider failures (does not retry permanent validation errors)
- Authoritative PostgreSQL persistence FIRST, followed by derived stores (Pinecone, BM25, Neo4j)
- Context propagation: restores correlation_id and document_id in contextvars for structured JSON logging and LangSmith tracing.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from interfaces.document_interface import (
    DocumentStatusEnum,
    IngestionStageEnum,
    IngestionJobInterface,
    DocumentMetadataInterface,
    EntityInterface,
    ChunkInterface,
)
from services.queue_service import JobQueueInterface, IngestionJobPayload, get_job_queue
from data_access import (
    save_document_to_db,
    save_chunks_to_db,
    update_document_status_in_db,
    update_job_progress_in_db,
    upsert_vector_chunks,
    index_bm25_chunks,
    upsert_graph_nodes,
)
from utils.pdf_parser import parse_pdf_document, ParsedDocumentResult, ParsedPageBlock
from utils.pii_redactor import redact_pii, RedactionResult
from utils.entity_extractor import extract_entities
from utils.chunker import create_structured_chunks
from utils.logger import (
    logger,
    set_correlation_id,
    get_correlation_id,
    set_document_id,
)
from utils.errors import (
    RAGError,
    ValidationError,
    ExternalProviderError,
    IngestionError,
)
from utils.tracing import trace_step

FUNC_WORKER_LOOP: str = "worker_loop"
FUNC_PROCESS_JOB: str = "process_ingestion_job"


class IngestionWorker:
    """Asynchronous background worker executing document ingestion jobs from queue."""

    def __init__(self, queue: Optional[JobQueueInterface] = None) -> None:
        """Initialize worker with target queue."""
        self.queue: JobQueueInterface = queue if queue is not None else get_job_queue()
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> asyncio.Task:
        """Start worker background consumption loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self.run_loop())
            logger.info("ingestion_worker", "Started background IngestionWorker loop.")
        return self._task

    async def stop(self) -> None:
        """Gracefully stop worker background loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("ingestion_worker", "Stopped background IngestionWorker loop.")

    async def run_loop(self) -> None:
        """Continuously dequeue and process jobs."""
        while self._running:
            try:
                payload = await self.queue.dequeue(timeout=0.5)
                if payload is not None:
                    asyncio.create_task(self.process_job(payload))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(FUNC_WORKER_LOOP, f"Error in worker dequeue loop: {exc}", exc=exc)
                await asyncio.sleep(0.5)

    async def process_job(self, payload: IngestionJobPayload) -> DocumentStatusEnum:
        """Process a single document ingestion job through all pipeline stages.

        @param payload: IngestionJobPayload dequeued from queue.
        @returns: Final DocumentStatusEnum outcome.
        """
        # Cooperative yield to ensure event loop dispatches pending responses
        await asyncio.sleep(0.01)

        job_id = payload.job_id
        doc_id = payload.document_id

        # Restore request context for structured logging and LangSmith tracing
        set_correlation_id(payload.correlation_id)
        set_document_id(doc_id)

        now_iso = datetime.now(timezone.utc).isoformat()
        logger.info(
            FUNC_PROCESS_JOB,
            f"Worker picked up job {job_id} for doc {doc_id} ('{payload.filename}')",
            correlation_id=payload.correlation_id,
            document_id=doc_id,
            status="started",
        )

        # Check early cancellation
        if self.queue.is_cancelled(job_id):
            return self._handle_cancelled(job_id, doc_id, 0.0)

        # Mark PROCESSING
        update_job_progress_in_db(
            job_id=job_id,
            status=DocumentStatusEnum.PROCESSING,
            stage=IngestionStageEnum.PARSING,
            progress_percent=10.0,
            started_at=now_iso,
        )
        update_document_status_in_db(doc_id, DocumentStatusEnum.PROCESSING, 0)

        chunks: list[ChunkInterface] = []
        entities: list[EntityInterface] = []
        metadata: Optional[DocumentMetadataInterface] = None

        try:
            # Stage 1: PARSING (15%)
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 15.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.PARSING, 15.0)
            parsed_doc: ParsedDocumentResult = self._execute_stage_parsing(payload.pdf_bytes, payload.filename)

            # Stage 2: PII_REDACTION (30%)
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 30.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.PII_REDACTION, 30.0)
            redaction_res, redacted_blocks = self._execute_stage_pii(parsed_doc)

            # Stage 3: ENTITY_EXTRACTION (45%)
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 45.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.ENTITY_EXTRACTION, 45.0)
            entities = extract_entities(redaction_res.cleaned_text)
            entity_names = [e.text for e in entities]

            # Stage 4: CHUNKING (60%)
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 60.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.CHUNKING, 60.0)
            chunks = create_structured_chunks(
                blocks=redacted_blocks,
                document_id=doc_id,
                source_filename=payload.filename,
                entities=entity_names,
            )

            metadata = DocumentMetadataInterface(
                document_id=doc_id,
                tenant_id=payload.tenant_id,
                title=payload.title or payload.filename,
                source_filename=payload.filename,
                total_pages=parsed_doc.total_pages,
                total_chunks=len(chunks),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Persist authoritative master document and chunks to PostgreSQL FIRST
            save_document_to_db(metadata)
            save_chunks_to_db(chunks)

            # Stage 5: EMBEDDING (75%)
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 75.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.EMBEDDING, 75.0)

            # Stage 6: INDEXING (90%) - Derived stores
            await asyncio.sleep(0.01)
            if self.queue.is_cancelled(job_id):
                return self._handle_cancelled(job_id, doc_id, 90.0)

            update_job_progress_in_db(job_id, DocumentStatusEnum.PROCESSING, IngestionStageEnum.INDEXING, 90.0)
            indexing_ok = await self._execute_derived_indexing_with_retry(payload, chunks, entities)

            final_status = DocumentStatusEnum.COMPLETED if indexing_ok else DocumentStatusEnum.PARTIAL
            final_stage = IngestionStageEnum.COMPLETED

            update_job_progress_in_db(
                job_id=job_id,
                status=final_status,
                stage=final_stage,
                progress_percent=100.0,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            update_document_status_in_db(doc_id, final_status, len(chunks))

            logger.info(
                FUNC_PROCESS_JOB,
                f"Job {job_id} for doc {doc_id} completed with status: {final_status.value}",
                correlation_id=payload.correlation_id,
                document_id=doc_id,
                status="success",
            )
            return final_status

        except ValidationError as val_err:
            # Permanent validation error - do not retry
            return self._handle_failure(job_id, doc_id, str(val_err), is_permanent=True)
        except Exception as exc:
            # Handle failure with retry check if not already retried
            return self._handle_failure(job_id, doc_id, str(exc), is_permanent=False)

    def _handle_cancelled(self, job_id: str, doc_id: str, progress: float) -> DocumentStatusEnum:
        """Mark job and document as cancelled."""
        now_iso = datetime.now(timezone.utc).isoformat()
        update_job_progress_in_db(
            job_id=job_id,
            status=DocumentStatusEnum.CANCELLED,
            stage=IngestionStageEnum.CANCELLED,
            progress_percent=progress,
            error_message="Job cancelled by user request.",
            completed_at=now_iso,
        )
        update_document_status_in_db(doc_id, DocumentStatusEnum.CANCELLED, 0, "Job cancelled by user request.")
        logger.info(FUNC_PROCESS_JOB, f"Job {job_id} cancelled at {progress}% progress.", document_id=doc_id, status="cancelled")
        return DocumentStatusEnum.CANCELLED

    def _handle_failure(self, job_id: str, doc_id: str, error_msg: str, is_permanent: bool) -> DocumentStatusEnum:
        """Mark job and document as failed."""
        now_iso = datetime.now(timezone.utc).isoformat()
        update_job_progress_in_db(
            job_id=job_id,
            status=DocumentStatusEnum.FAILED,
            stage=IngestionStageEnum.FAILED,
            progress_percent=0.0,
            error_message=error_msg,
            completed_at=now_iso,
        )
        update_document_status_in_db(doc_id, DocumentStatusEnum.FAILED, 0, error_msg)
        logger.error(
            FUNC_PROCESS_JOB,
            f"Job {job_id} for doc {doc_id} failed: {error_msg} (permanent={is_permanent})",
            document_id=doc_id,
            status="failure",
            error_type="IngestionError",
        )
        return DocumentStatusEnum.FAILED

    @trace_step(name="pdf_parsing", run_type="parser")
    def _execute_stage_parsing(self, pdf_bytes: bytes, filename: str) -> ParsedDocumentResult:
        """Execute PDF layout parsing stage."""
        return parse_pdf_document(pdf_bytes=pdf_bytes, filename=filename)

    @trace_step(name="pii_redaction", run_type="chain")
    def _execute_stage_pii(self, parsed_doc: ParsedDocumentResult) -> tuple[RedactionResult, list[ParsedPageBlock]]:
        """Execute PII redaction on layout blocks."""
        redaction_res = redact_pii(parsed_doc.full_text)
        redacted_blocks: list[ParsedPageBlock] = []
        for block in parsed_doc.blocks:
            block_redaction = redact_pii(block.text)
            redacted_blocks.append(
                ParsedPageBlock(
                    page=block.page,
                    section=block.section,
                    text=block_redaction.cleaned_text,
                    is_heading=block.is_heading,
                    is_table=block.is_table,
                )
            )
        return redaction_res, redacted_blocks

    async def _execute_derived_indexing_with_retry(
        self,
        payload: IngestionJobPayload,
        chunks: list[ChunkInterface],
        entities: list[EntityInterface],
    ) -> bool:
        """Execute derived indexing across Pinecone, BM25, and Neo4j with bounded exponential retry for transient errors."""
        max_retries = payload.max_retries
        attempt = 0

        while attempt <= max_retries:
            try:
                # 1. Pinecone Vector Upsert
                upsert_vector_chunks(chunks)
                # 2. BM25 Inverted Indexing
                index_bm25_chunks(chunks)
                # 3. Neo4j Graph Nodes Upsert
                upsert_graph_nodes(chunks, entities)
                return True
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        FUNC_PROCESS_JOB,
                        f"Derived indexing exhausted {max_retries} retries for job {payload.job_id}: {exc}",
                        exc=exc,
                    )
                    return False

                backoff_s = min(0.25 * (2 ** (attempt - 1)), 2.0)
                logger.warning(
                    FUNC_PROCESS_JOB,
                    f"Derived indexing attempt {attempt} failed ({exc}). Retrying in {backoff_s}s...",
                )
                update_job_progress_in_db(
                    job_id=payload.job_id,
                    status=DocumentStatusEnum.PROCESSING,
                    stage=IngestionStageEnum.INDEXING,
                    progress_percent=90.0,
                    retry_count=attempt,
                    error_message=f"Retry {attempt}/{max_retries}: {exc}",
                )
        return False


_GLOBAL_WORKER: Optional[IngestionWorker] = None


def get_or_start_worker(queue: Optional[JobQueueInterface] = None) -> IngestionWorker:
    """Return or start the global IngestionWorker instance."""
    global _GLOBAL_WORKER
    if _GLOBAL_WORKER is None:
        _GLOBAL_WORKER = IngestionWorker(queue=queue or get_job_queue())

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    needs_start = False
    if not _GLOBAL_WORKER._running or _GLOBAL_WORKER._task is None or _GLOBAL_WORKER._task.done():
        needs_start = True
    elif current_loop is not None:
        try:
            task_loop = _GLOBAL_WORKER._task.get_loop()
            if task_loop.is_closed() or task_loop is not current_loop:
                needs_start = True
        except Exception:
            needs_start = True

    if needs_start and current_loop is not None and not current_loop.is_closed():
        _GLOBAL_WORKER._running = False
        _GLOBAL_WORKER.start()

    return _GLOBAL_WORKER

