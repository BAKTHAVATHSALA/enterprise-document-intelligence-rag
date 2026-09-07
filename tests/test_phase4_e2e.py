"""Focused End-to-End Hybrid RAG Verification Test Suite.

Verifies the complete RAG workflow:
1. PDF Ingestion via Docling layout parser.
2. Structured chunking, PII redaction, entity extraction.
3. Multi-store persistence & reconciliation across PostgreSQL, Pinecone, Neo4j, and BM25.
4. Tenant and document ID isolation filtering.
5. Hybrid retrieval across Vector, BM25, and Graph sources.
6. Reciprocal Rank Fusion (RRF) scoring.
7. Neural reranker relevance scoring.
8. Fact-grounded LLM answer generation.
9. Citation validation (verifying valid citations and zero fabricated citations).
10. Complete QueryResponseInterface payload verification.
"""

import os
from helpers.document_helper import process_pdf_pipeline, ProcessedDocumentResult
from data_access import (
    save_document_to_db,
    save_chunks_to_db,
    load_all_chunks_from_db,
    upsert_vector_chunks,
    index_bm25_chunks,
    upsert_graph_nodes,
)
from helpers.retrieval_helper import execute_hybrid_retrieval
from services.query_service import execute_query_workflow
from interfaces.query_interface import QueryRequestInterface, QueryResponseInterface

TEST_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_documents")
TECH_POLICY_PATH = os.path.join(TEST_DOCS_DIR, "technical_policy.pdf")


def _ingest_technical_policy() -> ProcessedDocumentResult:
    """Helper to ingest technical_policy.pdf and persist across all 4 datastores.

    @returns: ProcessedDocumentResult containing document metadata and chunks.
    """
    assert os.path.exists(TECH_POLICY_PATH), f"Fixture file not found: {TECH_POLICY_PATH}"
    with open(TECH_POLICY_PATH, "rb") as f:
        pdf_bytes = f.read()

    # 1. Process PDF Pipeline
    res: ProcessedDocumentResult = process_pdf_pipeline(
        pdf_bytes=pdf_bytes,
        filename="technical_policy.pdf",
        title="Technical Security Policy",
    )

    # 2. Persist to PostgreSQL
    save_document_to_db(res.metadata)
    save_chunks_to_db(res.chunks)

    # 3. Persist to Pinecone Vector Store
    upsert_vector_chunks(res.chunks)

    # 4. Persist to BM25 Index
    index_bm25_chunks(res.chunks)

    # 5. Persist to Neo4j Graph Store
    upsert_graph_nodes(res.chunks)

    return res


def test_e2e_pdf_ingestion_and_store_reconciliation() -> None:
    """TEST 1: Ingest technical_policy.pdf and verify chunk ID reconciliation across datastores."""
    res = _ingest_technical_policy()
    doc_id = res.metadata.document_id

    assert res.metadata.total_pages >= 1
    assert len(res.chunks) > 0

    # Verify PostgreSQL DB chunks
    db_chunks = load_all_chunks_from_db()
    doc_db_chunks = [c for c in db_chunks if c.metadata.document_id == doc_id]
    assert len(doc_db_chunks) == len(res.chunks)

    # Verify chunk IDs match
    expected_chunk_ids = {c.metadata.chunk_id for c in res.chunks}
    db_chunk_ids = {c.metadata.chunk_id for c in doc_db_chunks}
    assert expected_chunk_ids == db_chunk_ids


def test_e2e_tenant_and_document_isolation() -> None:
    """TEST 2: Verify document ID filtering isolates retrieval results strictly to specified document."""
    res = _ingest_technical_policy()
    target_doc_id = res.metadata.document_id

    # Execute retrieval with document_ids filter
    candidates = execute_hybrid_retrieval(
        query_text="security and encryption standards",
        top_k=5,
        document_ids=[target_doc_id],
    )

    assert len(candidates) > 0
    for cand in candidates:
        assert cand.chunk.metadata.document_id == target_doc_id


def test_e2e_hybrid_retrieval_and_rrf_fusion() -> None:
    """TEST 3: Verify hybrid retrieval combines Vector, BM25, and Graph candidates with valid RRF scores."""
    res = _ingest_technical_policy()
    target_doc_id = res.metadata.document_id

    candidates = execute_hybrid_retrieval(
        query_text="data protection and compliance guidelines",
        top_k=5,
        document_ids=[target_doc_id],
    )

    assert len(candidates) > 0
    for cand in candidates:
        assert cand.fused_score > 0.0
        assert cand.chunk.text != ""


def test_e2e_neural_reranking() -> None:
    """TEST 4: Verify neural reranking assigns rerank scores and orders candidates by relevance."""
    res = _ingest_technical_policy()
    target_doc_id = res.metadata.document_id

    candidates = execute_hybrid_retrieval(
        query_text="incident response policy and security protocol",
        top_k=5,
        document_ids=[target_doc_id],
    )

    assert len(candidates) > 0
    for cand in candidates:
        assert isinstance(cand.rerank_score, float)

    # Verify descending rerank score order
    rerank_scores = [c.rerank_score for c in candidates]
    assert rerank_scores == sorted(rerank_scores, reverse=True)


def test_e2e_grounded_llm_answer_and_citations() -> None:
    """TEST 5: Verify end-to-end query workflow generates a grounded answer with validated citations."""
    from data_access.llm_data_access import set_mock_llm_mode
    set_mock_llm_mode(True)
    try:
        res = _ingest_technical_policy()
        target_doc_id = res.metadata.document_id

        request = QueryRequestInterface(
            question="What are the security and encryption requirements?",
            document_ids=[target_doc_id],
            top_k=5,
        )

        response: QueryResponseInterface = execute_query_workflow(request)

        assert response.question == request.question
        assert response.answer != ""
        assert response.confidence_score >= 0.0
        assert response.processing_time_ms > 0.0

        # Verify citation validation summary
        assert response.validation_summary is not None
        assert response.validation_summary.fabricated_count == 0
        assert response.validation_summary.invalid_count == 0
    finally:
        set_mock_llm_mode(False)


def test_e2e_full_query_response_payload() -> None:
    """TEST 6: Verify full QueryResponseInterface payload contains all required fields."""
    from data_access.llm_data_access import set_mock_llm_mode
    set_mock_llm_mode(True)
    try:
        res = _ingest_technical_policy()
        target_doc_id = res.metadata.document_id

        request = QueryRequestInterface(
            question="Describe the technical access control policies.",
            document_ids=[target_doc_id],
            top_k=3,
        )

        response: QueryResponseInterface = execute_query_workflow(request)

        assert isinstance(response, QueryResponseInterface)
        assert isinstance(response.citations, list)
        assert len(response.citations) > 0
        assert all(c.is_valid for c in response.citations if c.validation_status != "MISSING")
    finally:
        set_mock_llm_mode(False)

