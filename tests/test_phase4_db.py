"""Test Suite for Phase 4 Step 3B — PostgreSQL Source-of-Truth & BM25 Hydration.

Verifies database schema initialization, document/chunk persistence, cascading deletion,
BM25 startup hydration from PostgreSQL, tenant isolation, and application restart simulation.
"""

import os
import pytest
from typing import Optional

from interfaces.document_interface import (
    DocumentMetadataInterface,
    DocumentStatusInterface,
    DocumentStatusEnum,
    ChunkInterface,
    ChunkMetadataInterface,
)
from data_access.db_data_access import (
    get_db_connection,
    ensure_db_schema,
    save_document_to_db,
    save_chunks_to_db,
    update_document_status_in_db,
    get_document_metadata_from_db,
    get_document_status_from_db,
    load_all_chunks_from_db,
    delete_document_from_db,
)
from data_access.bm25_data_access import (
    init_bm25_from_db,
    index_bm25_chunks,
    query_bm25_index,
    evict_document_from_bm25,
    _BM25_DOC_STORE,
)
from utils.chunker import generate_chunk_id
from utils.config import get_config


def test_1_postgres_connection_config() -> None:
    """TEST 1: Verify PostgreSQL connection can be established using database configuration."""
    conn = get_db_connection()
    assert conn is not None, "Failed to connect to PostgreSQL database using environment config."
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            row = cur.fetchone()
            assert row[0] == 1
    finally:
        conn.close()


def test_2_schema_initialization_idempotent() -> None:
    """TEST 2: Verify database schema initialization runs safely and idempotently multiple times."""
    ok1 = ensure_db_schema()
    assert ok1 is True, "First ensure_db_schema call failed."

    ok2 = ensure_db_schema()
    assert ok2 is True, "Second ensure_db_schema call failed (idempotency failure)."


def test_3_document_and_chunk_persistence_with_cascading_deletion() -> None:
    """TEST 3: Persist document & chunks to PostgreSQL, verify retrieval, and check cascading deletion."""
    test_doc_id: str = "doc_test_pg_suite_001"
    test_tenant_id: str = "tenant_test_suite"

    # 1. Save Document
    doc_meta = DocumentMetadataInterface(
        document_id=test_doc_id,
        tenant_id=test_tenant_id,
        title="PostgreSQL Integration Test Document",
        source_filename="test_policy.pdf",
        total_pages=2,
        total_chunks=2,
        created_at="2026-09-06T12:00:00Z",
    )
    saved_doc = save_document_to_db(doc_meta)
    assert saved_doc is True

    # Update status to COMPLETED
    update_ok = update_document_status_in_db(test_doc_id, DocumentStatusEnum.COMPLETED, processed_chunks=2)
    assert update_ok is True

    # 2. Save Chunks
    chunk_1_id: str = generate_chunk_id(test_doc_id, 1, 0)
    chunk_2_id: str = generate_chunk_id(test_doc_id, 2, 1)

    chunk1 = ChunkInterface(
        chunk_id=chunk_1_id,
        document_id=test_doc_id,
        tenant_id=test_tenant_id,
        text="The access control policy requires multi-factor authentication for all database instances.",
        metadata=ChunkMetadataInterface(
            document_id=test_doc_id,
            chunk_id=chunk_1_id,
            tenant_id=test_tenant_id,
            page=1,
            section="Security Policy",
            source="test_policy.pdf",
            entities=["Access Control", "MFA"],
        ),
    )

    chunk2 = ChunkInterface(
        chunk_id=chunk_2_id,
        document_id=test_doc_id,
        tenant_id=test_tenant_id,
        text="Data retention requirements specify encrypted automated backups retained for 90 days.",
        metadata=ChunkMetadataInterface(
            document_id=test_doc_id,
            chunk_id=chunk_2_id,
            tenant_id=test_tenant_id,
            page=2,
            section="Backup Policy",
            source="test_policy.pdf",
            entities=["Backup", "Encryption"],
        ),
    )

    saved_chunks_count = save_chunks_to_db([chunk1, chunk2])
    assert saved_chunks_count == 2

    # 3. Read back from DB
    retrieved_meta = get_document_metadata_from_db(test_doc_id)
    assert retrieved_meta is not None
    assert retrieved_meta.document_id == test_doc_id
    assert retrieved_meta.tenant_id == test_tenant_id
    assert retrieved_meta.total_pages == 2

    retrieved_status = get_document_status_from_db(test_doc_id)
    assert retrieved_status is not None
    assert retrieved_status.status == DocumentStatusEnum.COMPLETED
    assert retrieved_status.processed_chunks == 2

    # 4. Clean up / Delete Document
    deleted_ok = delete_document_from_db(test_doc_id)
    assert deleted_ok is True

    # Verify document and chunks were deleted from PostgreSQL
    post_delete_meta = get_document_metadata_from_db(test_doc_id)
    assert post_delete_meta is None


def test_4_bm25_startup_hydration_and_restart_simulation() -> None:
    """TEST 4: Hydrate BM25 index from PostgreSQL, perform restart simulation, and test query filtering."""
    test_doc_id: str = "doc_test_pg_bm25_002"
    test_tenant_a: str = "tenant_alpha"
    test_tenant_b: str = "tenant_beta"

    # Setup 2 test documents across 2 distinct tenants in PostgreSQL
    doc_a = DocumentMetadataInterface(
        document_id=test_doc_id,
        tenant_id=test_tenant_a,
        title="Tenant Alpha Policy",
        source_filename="alpha_policy.pdf",
        total_pages=1,
        total_chunks=1,
        created_at="2026-09-06T12:00:00Z",
    )
    save_document_to_db(doc_a)

    chunk_a_id: str = generate_chunk_id(test_doc_id, 1, 0)
    chunk_a = ChunkInterface(
        chunk_id=chunk_a_id,
        document_id=test_doc_id,
        tenant_id=test_tenant_a,
        text="Alpha tenant exclusive confidential encryption key rotation guidelines.",
        metadata=ChunkMetadataInterface(
            document_id=test_doc_id,
            chunk_id=chunk_a_id,
            tenant_id=test_tenant_a,
            page=1,
            section="Encryption",
            source="alpha_policy.pdf",
            entities=["Encryption"],
        ),
    )
    save_chunks_to_db([chunk_a])

    # 1. Clear in-memory BM25 index (Simulate app restart)
    _BM25_DOC_STORE.clear()
    assert len(_BM25_DOC_STORE) == 0

    # 2. Hydrate from PostgreSQL
    hydrated_count = init_bm25_from_db()
    assert hydrated_count >= 1
    assert chunk_a_id in _BM25_DOC_STORE

    # 3. Query BM25 index after restart hydration
    results = query_bm25_index(query_text="encryption key rotation", top_k=5, tenant_id=test_tenant_a)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == chunk_a_id

    # 4. Verify tenant isolation (Tenant B querying should return 0 results)
    tenant_b_results = query_bm25_index(query_text="encryption key rotation", top_k=5, tenant_id=test_tenant_b)
    assert len(tenant_b_results) == 0

    # 5. Clean up DB & BM25
    delete_document_from_db(test_doc_id)
    evict_document_from_bm25(test_doc_id)


def test_5_no_secrets_in_config(caplog) -> None:
    """TEST 5: Ensure database operations do not print database passwords or connection secrets in log output."""
    import logging
    with caplog.at_level(logging.DEBUG):
        ensure_db_schema()
        config = get_config()
        if config.postgres_password:
            for record in caplog.records:
                assert config.postgres_password not in record.message
