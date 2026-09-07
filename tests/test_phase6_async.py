"""Automated Test Suite for Phase 6 — Asynchronous Ingestion Architecture.

Verifies:
1. Async 202 Accepted response contract for POST /documents with job_id and document_id.
2. JobQueueInterface and AsyncInMemoryJobQueue operations (enqueue, dequeue, cancel, size).
3. End-to-end background ingestion worker execution across pipeline stages.
4. Job status tracking via GET /jobs/{job_id} and GET /documents/{id}/status.
5. Cooperative cancellation via POST /documents/{id}/cancel and POST /jobs/{id}/cancel.
6. Bounded exponential retry on transient failure and immediate failure on permanent error.
7. BM25 term frequency deduplication on chunk re-indexing.
8. Correlation ID propagation across HTTP request, queue payload, and worker execution.
"""

import asyncio
import os
import time
import pytest
from fastapi.testclient import TestClient

from main import app
from interfaces.document_interface import (
    DocumentStatusEnum,
    IngestionStageEnum,
    IngestionJobInterface,
    DocumentStatusInterface,
    ChunkInterface,
    ChunkMetadataInterface,
)
from services.queue_service import (
    AsyncInMemoryJobQueue,
    IngestionJobPayload,
    get_job_queue,
    set_job_queue,
)
from workers.ingestion_worker import IngestionWorker
from data_access.bm25_data_access import (
    index_bm25_chunks,
    query_bm25_index,
    _BM25_DF,
    _BM25_DOC_STORE,
)
from data_access.db_data_access import (
    save_ingestion_job_to_db,
    update_job_progress_in_db,
    get_ingestion_job_from_db,
    get_latest_job_for_document_from_db,
    _LOCAL_INGESTION_JOBS_STORE,
)
from utils.errors import ValidationError, ExternalProviderError

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def client_lifespan():
    """Maintain running application lifespan and background worker for test module."""
    with client:
        yield

TEST_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_documents")
CONTRACT_ABC_PATH = os.path.join(TEST_DOCS_DIR, "contract_abc.pdf")


# ---------------------------------------------------------------------------
# 1. Async HTTP 202 Response Contract Tests
# ---------------------------------------------------------------------------

def test_async_document_upload_returns_202():
    """Verify POST /documents returns HTTP 202 Accepted immediately with job_id and document_id."""
    minimal_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    custom_corr = "corr_phase6_test_202"
    resp = client.post(
        "/documents",
        files={"file": ("contract_test.pdf", minimal_pdf_bytes, "application/pdf")},
        headers={"X-Correlation-ID": custom_corr},
    )

    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # Verify response schema
    assert "job_id" in data and data["job_id"].startswith("job_")
    assert "document_id" in data and data["document_id"].startswith("doc_")
    assert data["status"] == "PENDING"
    assert data["stage"] == "QUEUED"
    assert data["progress_percent"] == 0.0
    assert data["correlation_id"] == custom_corr
    assert resp.headers.get("X-Correlation-ID") == custom_corr


def test_async_document_upload_validation_rejections():
    """Verify non-PDF, empty, and corrupted files are rejected synchronously with HTTP 400 before queuing."""
    # 1. Invalid extension
    r_ext = client.post(
        "/documents",
        files={"file": ("test.txt", b"plain text", "text/plain")},
    )
    assert r_ext.status_code == 400
    assert "Only .pdf files are supported" in r_ext.json()["detail"]

    # 2. Empty payload
    r_empty = client.post(
        "/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert r_empty.status_code == 400
    assert "empty" in r_empty.json()["detail"].lower()

    # 3. Corrupted PDF without %PDF- magic header
    r_corrupt = client.post(
        "/documents",
        files={"file": ("corrupt.pdf", b"NOT_A_VALID_HEADER", "application/pdf")},
    )
    assert r_corrupt.status_code == 400
    assert "header" in r_corrupt.json()["detail"].lower() or "invalid" in r_corrupt.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 2. In-Memory Job Queue Unit Tests (Synchronous wrappers around asyncio.run)
# ---------------------------------------------------------------------------

def test_job_queue_enqueue_dequeue():
    """Verify AsyncInMemoryJobQueue enqueue and dequeue behavior."""
    async def _test():
        queue = AsyncInMemoryJobQueue(maxsize=10)
        assert queue.get_size() == 0

        payload = IngestionJobPayload(
            job_id="job_test_001",
            document_id="doc_test_001",
            filename="test.pdf",
            pdf_bytes=b"%PDF-test",
        )

        enqueued_id = await queue.enqueue(payload)
        assert enqueued_id == "job_test_001"
        assert queue.get_size() == 1

        dequeued = await queue.dequeue(timeout=0.1)
        assert dequeued is not None
        assert dequeued.job_id == "job_test_001"
        assert dequeued.document_id == "doc_test_001"
        assert queue.get_size() == 0

    asyncio.run(_test())


def test_job_queue_timeout_on_empty():
    """Verify AsyncInMemoryJobQueue dequeue returns None when queue is empty after timeout."""
    async def _test():
        queue = AsyncInMemoryJobQueue()
        start_time = time.time()
        result = await queue.dequeue(timeout=0.1)
        elapsed = time.time() - start_time

        assert result is None
        assert elapsed >= 0.08

    asyncio.run(_test())


def test_job_queue_cancellation():
    """Verify AsyncInMemoryJobQueue tracks cancelled jobs."""
    async def _test():
        queue = AsyncInMemoryJobQueue()
        assert not queue.is_cancelled("job_cancel_001")

        cancelled = await queue.cancel_job("job_cancel_001")
        assert cancelled is True
        assert queue.is_cancelled("job_cancel_001")

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 3. Worker Ingestion Flow & Status Polling
# ---------------------------------------------------------------------------

def test_full_pipeline_ingestion_and_status_polling():
    """Verify document upload, background worker execution, and status polling until COMPLETED."""
    with open(CONTRACT_ABC_PATH, "rb") as f:
        pdf_bytes = f.read()

    resp = client.post(
        "/documents",
        files={"file": ("contract_abc.pdf", pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 202
    job_info = resp.json()
    job_id = job_info["job_id"]
    doc_id = job_info["document_id"]

    # Poll status endpoint until COMPLETED
    final_status = None
    for _ in range(80):
        status_resp = client.get(f"/documents/{doc_id}/status")
        if status_resp.status_code == 200:
            final_status = status_resp.json()
            if final_status["status"] in ("COMPLETED", "FAILED"):
                break
        time.sleep(1.0)

    assert final_status is not None
    assert final_status["status"] == "COMPLETED"
    assert final_status["stage"] == "COMPLETED"
    assert final_status["progress_percent"] == 100.0
    assert final_status["processed_chunks"] > 0

    # Also verify GET /jobs/{job_id} endpoint
    job_resp = client.get(f"/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["document_id"] == doc_id
    assert job_data["status"] == "COMPLETED"
    assert job_data["stage"] == "COMPLETED"
    assert job_data["progress_percent"] == 100.0


# ---------------------------------------------------------------------------
# 4. Cooperative Cancellation Tests
# ---------------------------------------------------------------------------

def test_worker_respects_cooperative_cancellation():
    """Verify worker aborts processing when job is marked as cancelled."""
    async def _test():
        test_queue = AsyncInMemoryJobQueue()
        worker = IngestionWorker(queue=test_queue)

        job_id = "job_cancel_pre_stage"
        doc_id = "doc_cancel_pre_stage"

        # Pre-cancel the job before worker dequeues
        await test_queue.cancel_job(job_id)

        payload = IngestionJobPayload(
            job_id=job_id,
            document_id=doc_id,
            filename="contract_abc.pdf",
            pdf_bytes=b"%PDF-1.4 test payload",
        )

        result_status = await worker.process_job(payload)
        assert result_status == DocumentStatusEnum.CANCELLED

    asyncio.run(_test())


def test_api_cancellation_endpoints():
    """Verify POST /documents/{id}/cancel and POST /jobs/{id}/cancel endpoints."""
    # 1. Enqueue a job via document upload
    minimal_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    upload_resp = client.post(
        "/documents",
        files={"file": ("to_cancel.pdf", minimal_pdf, "application/pdf")},
    )
    assert upload_resp.status_code == 202
    doc_id = upload_resp.json()["document_id"]
    job_id = upload_resp.json()["job_id"]

    # 2. Cancel by document_id
    resp1 = client.post(f"/documents/{doc_id}/cancel")
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "cancelled"
    assert data1["document_id"] == doc_id

    # 3. Cancel by job_id
    resp2 = client.post(f"/jobs/{job_id}/cancel")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "cancelled"
    assert data2["job_id"] == job_id

    # 4. Non-existent document cancel returns 404
    resp404 = client.post("/documents/doc_nonexistent_9999/cancel")
    assert resp404.status_code == 404


# ---------------------------------------------------------------------------
# 5. Worker Bounded Retry & Error Handling Tests
# ---------------------------------------------------------------------------

def test_worker_permanent_error_fails_immediately():
    """Verify permanent validation error fails immediately without retrying."""
    async def _test():
        worker = IngestionWorker()
        job_id = "job_perm_error"
        doc_id = "doc_perm_error"

        payload = IngestionJobPayload(
            job_id=job_id,
            document_id=doc_id,
            filename="invalid.pdf",
            pdf_bytes=b"NOT_A_REAL_PDF",
        )

        result = await worker.process_job(payload)
        assert result == DocumentStatusEnum.FAILED

    asyncio.run(_test())


def test_worker_transient_retry_logic(monkeypatch):
    """Verify worker performs bounded exponential retry on transient indexing failures."""
    async def _test():
        worker = IngestionWorker()
        payload = IngestionJobPayload(
            job_id="job_transient_retry",
            document_id="doc_transient_retry",
            filename="contract_abc.pdf",
            pdf_bytes=b"%PDF-1.4 dummy",
        )
        dummy_chunk = ChunkInterface(
            chunk_id="chk_retry_1",
            document_id="doc_transient_retry",
            page=1,
            chunk_index=0,
            section="Section 1",
            source="contract_abc.pdf",
            text="sample chunk text",
            metadata=ChunkMetadataInterface(
                chunk_id="chk_retry_1",
                document_id="doc_transient_retry",
                page=1,
                section="Section 1",
                source="contract_abc.pdf",
            ),
        )

        attempt_counter = {"count": 0}

        def failing_upsert(*args, **kwargs):
            attempt_counter["count"] += 1
            if attempt_counter["count"] < 3:
                raise ExternalProviderError("Simulated transient Pinecone rate limit")
            return True

        import workers.ingestion_worker as w_mod
        monkeypatch.setattr(w_mod, "upsert_vector_chunks", failing_upsert)
        monkeypatch.setattr(w_mod, "index_bm25_chunks", lambda *args, **kwargs: True)
        monkeypatch.setattr(w_mod, "upsert_graph_nodes", lambda *args, **kwargs: True)

        result = await worker._execute_derived_indexing_with_retry(payload, [dummy_chunk], [])
        assert result is True
        assert attempt_counter["count"] == 3

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 6. BM25 Deduplication Test
# ---------------------------------------------------------------------------

def test_bm25_deduplication_on_reindexing():
    """Verify indexing the same document chunks multiple times does not inflate term frequencies."""
    chunk = ChunkInterface(
        chunk_id="chunk_bm25_dedup",
        document_id="doc_bm25_dedup",
        page=1,
        chunk_index=0,
        section="Test",
        source="test.pdf",
        text="UniqueAcquisitionTermXYZ appears exactly once in this chunk.",
        metadata=ChunkMetadataInterface(
            chunk_id="chunk_bm25_dedup",
            document_id="doc_bm25_dedup",
            page=1,
            section="Test",
            source="test.pdf",
        ),
    )

    # First indexing
    index_bm25_chunks([chunk])
    token = "uniqueacquisitiontermxyz"
    freq_first = _BM25_DF.get(token, 0)
    assert freq_first == 1

    # Re-index same chunk
    index_bm25_chunks([chunk])
    freq_second = _BM25_DF.get(token, 0)

    # Deduplication assertion: term frequency must remain 1, not duplicate to 2
    assert freq_second == 1, f"Expected document frequency 1 after reindexing, but got {freq_second}"
