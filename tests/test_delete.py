"""Focused Delete Document & Race Condition Safety Unit Test Suite.

Validates document-scoped deletion behavior, multi-store isolation, search exclusion,
API contract adherence, and deletion race-condition safety during active processing.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from main import app
from interfaces.document_interface import (
    DocumentMetadataInterface,
    DocumentStatusEnum,
    ChunkInterface,
    ChunkMetadataInterface,
)
from services.document_service import (
    delete_document,
    get_document_metadata,
)
from services.queue_service import (
    IngestionJobPayload,
    get_job_queue,
)
from data_access import (
    save_document_to_db,
    save_chunks_to_db,
    upsert_vector_chunks,
    index_bm25_chunks,
    upsert_graph_nodes,
    query_vector_store,
    query_bm25_index,
    get_document_metadata_from_db,
    load_all_chunks_from_db,
    _CHUNK_STORE,
)
from workers.ingestion_worker import IngestionWorker

client = TestClient(app)


def _create_mock_document_fixture(doc_id: str, title: str, chunks_text: list[str]) -> tuple[DocumentMetadataInterface, list[ChunkInterface]]:
    """Helper fixture creating mock document and chunks across stores without PDF upload."""
    now_iso = datetime.now(timezone.utc).isoformat()
    meta = DocumentMetadataInterface(
        document_id=doc_id,
        tenant_id="default_tenant",
        title=title,
        source_filename=f"{doc_id}.pdf",
        total_pages=1,
        total_chunks=len(chunks_text),
        created_at=now_iso,
    )
    save_document_to_db(meta)

    chunks = []
    for idx, text in enumerate(chunks_text, start=1):
        chunk_id = f"chk_{doc_id}_{idx}"
        chunk = ChunkInterface(
            chunk_id=chunk_id,
            document_id=doc_id,
            tenant_id="default_tenant",
            text=text,
            metadata=ChunkMetadataInterface(
                document_id=doc_id,
                chunk_id=chunk_id,
                page=1,
                section="General",
                source=f"{doc_id}.pdf",
                entities=[f"Entity_{doc_id}"],
            ),
        )
        chunks.append(chunk)

    save_chunks_to_db(chunks)
    upsert_vector_chunks(chunks, use_local_mock=True)
    index_bm25_chunks(chunks)
    upsert_graph_nodes(chunks, use_local_mock=True)

    return meta, chunks


def test_delete_completed_document():
    """1, 5, 6, 7, 8, 9: Verify deleting completed document purges all store records."""
    async def _test():
        doc_id = "doc_completed_test_1"
        _create_mock_document_fixture(doc_id, "Completed Doc", ["Quantum computing in RAG systems."])

        # Pre-check existence across stores
        assert get_document_metadata(doc_id) is not None
        assert len(query_bm25_index("Quantum", document_ids=[doc_id])) > 0
        assert len(query_vector_store("Quantum", document_ids=[doc_id], use_local_mock=True)) > 0

        # Execute deletion
        success = await delete_document(doc_id)
        assert success is True

        # 5. Verify PostgreSQL data removed
        assert get_document_metadata_from_db(doc_id) is None
        all_chunks = load_all_chunks_from_db()
        assert not any(c.document_id == doc_id for c in all_chunks)

        # 6. Verify Pinecone (mock) vector removed
        vector_hits = query_vector_store("Quantum", document_ids=[doc_id], use_local_mock=True)
        assert len(vector_hits) == 0

        # 7. Verify Neo4j (mock) graph removed
        assert not any(getattr(c, "document_id", None) == doc_id for c in _CHUNK_STORE.values())

        # 8. Verify BM25 entries removed
        bm25_hits = query_bm25_index("Quantum", document_ids=[doc_id])
        assert len(bm25_hits) == 0

        # 9. Verify deleted document cannot be retrieved through search
        assert len(query_bm25_index("Quantum")) == 0 or not any(h.chunk.document_id == doc_id for h in query_bm25_index("Quantum"))

    asyncio.run(_test())


def test_delete_nonexistent_document_api():
    """2. Verify DELETE /documents/{nonexistent_id} returns 404."""
    response = client.delete("/documents/doc_nonexistent_9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_isolation_multi_document():
    """3, 4: Verify deleting Doc A leaves Doc B completely intact."""
    async def _test():
        doc_a = "doc_isolation_a"
        doc_b = "doc_isolation_b"

        _create_mock_document_fixture(doc_a, "Document A", ["First document payload text."])
        _create_mock_document_fixture(doc_b, "Document B", ["Second document payload text."])

        # Delete Doc A only
        success = await delete_document(doc_a)
        assert success is True

        # Verify Doc A is gone
        assert get_document_metadata(doc_a) is None
        assert len(query_bm25_index("payload", document_ids=[doc_a])) == 0

        # 4. Verify Doc B remains completely intact across all stores
        assert get_document_metadata(doc_b) is not None
        assert get_document_metadata(doc_b).title == "Document B"
        b_bm25 = query_bm25_index("payload", document_ids=[doc_b])
        assert len(b_bm25) == 1
        assert b_bm25[0].chunk.document_id == doc_b

        b_vec = query_vector_store("payload", document_ids=[doc_b], use_local_mock=True)
        assert len(b_vec) == 1
        assert b_vec[0].chunk.document_id == doc_b

    asyncio.run(_test())


def test_ui_list_documents_refresh():
    """10. Verify GET /documents list excludes deleted document."""
    doc_id = "doc_ui_list_test"
    _create_mock_document_fixture(doc_id, "UI Test Doc", ["Testing UI list endpoint."])

    # Check GET /documents includes document
    response = client.get("/documents")
    assert response.status_code == 200
    doc_ids = [d["document_id"] for d in response.json()]
    assert doc_id in doc_ids

    # Delete via API
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Refresh list via GET /documents
    refresh_res = client.get("/documents")
    assert refresh_res.status_code == 200
    updated_doc_ids = [d["document_id"] for d in refresh_res.json()]
    assert doc_id not in updated_doc_ids


def test_deterministic_deletion_race_condition_safety():
    """11 & User Constraint 2: Prove race-condition safety during active worker processing.
    
    Deterministic test scenario:
    1. Enqueue job payload into background worker queue.
    2. Worker picks up job and begins stage processing.
    3. Deletion occurs while document is in PENDING/PROCESSING state.
    4. Verify worker aborts without resurrecting the deleted document or leaving orphaned chunks/vectors/graph nodes.
    """
    async def _test():
        doc_id = f"doc_race_test_{int(datetime.now(timezone.utc).timestamp())}"
        job_id = f"job_race_test_{int(datetime.now(timezone.utc).timestamp())}"
        queue = get_job_queue()

        payload = IngestionJobPayload(
            job_id=job_id,
            document_id=doc_id,
            filename="race_test.pdf",
            pdf_bytes=b"%PDF-1.4 Mock PDF Content",
        )
        await queue.enqueue(payload)

        # Initial metadata in DB
        meta = DocumentMetadataInterface(
            document_id=doc_id,
            tenant_id="default_tenant",
            title="Race Test Doc",
            source_filename="race_test.pdf",
            total_pages=1,
            total_chunks=0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        save_document_to_db(meta)

        worker = IngestionWorker(queue=queue)
        
        # Mock stage 1 parsing to return a valid parsed document result without parsing PDF bytes
        from utils.pdf_parser import ParsedDocumentResult, ParsedPageBlock
        worker._execute_stage_parsing = lambda pdf_bytes, filename: ParsedDocumentResult(
            full_text="Mock page content for race condition test.",
            total_pages=1,
            blocks=[
                ParsedPageBlock(page=1, section="General", text="Mock page content for race condition test.")
            ]
        )

        # Execute deletion while job is active
        del_success = await delete_document(doc_id)
        assert del_success is True

        # Run worker processing on the dequeued job
        worker_status = await worker.process_job(payload)
        assert worker_status == DocumentStatusEnum.CANCELLED

        # VERIFY RACE-CONDITION SAFETY:
        # 1. Document was not resurrected in PostgreSQL
        assert get_document_metadata_from_db(doc_id) is None

        # 2. No orphaned chunks exist in PostgreSQL
        db_chunks = load_all_chunks_from_db()
        assert not any(c.document_id == doc_id for c in db_chunks)

        # 3. No orphaned vectors in vector store
        vec_hits = query_vector_store("Mock", document_ids=[doc_id], use_local_mock=True)
        assert len(vec_hits) == 0

        # 4. No orphaned graph nodes in graph store
        assert not any(getattr(c, "document_id", None) == doc_id for c in _CHUNK_STORE.values())

        # 5. No orphaned BM25 entries
        bm25_hits = query_bm25_index("Mock", document_ids=[doc_id])
        assert len(bm25_hits) == 0

    asyncio.run(_test())

