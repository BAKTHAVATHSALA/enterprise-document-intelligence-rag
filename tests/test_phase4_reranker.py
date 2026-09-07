"""Test Suite for Phase 4 Step 3C — Neural Cross-Encoder Reranker.

Verifies neural cross-encoder candidate scoring (BAAI/bge-reranker-base),
interface compliance, rank ordering, document/tenant lineage preservation,
empty candidate handling, and mock fallback execution.
"""

import pytest
from interfaces.document_interface import ChunkInterface, ChunkMetadataInterface
from interfaces.retrieval_interface import (
    FusedCandidateInterface,
    RerankedCandidateInterface,
    RetrievalSourceEnum,
)
from data_access.reranker_data_access import (
    rerank_fused_candidates,
    set_mock_reranker_mode,
    get_reranker_model,
)
from helpers.retrieval_helper import rerank_candidates


def _create_sample_fused_candidates() -> list[FusedCandidateInterface]:
    """Helper to create dummy fused candidate chunks for testing."""
    c1 = ChunkInterface(
        chunk_id="chunk_sec_001",
        document_id="doc_pol_100",
        tenant_id="tenant_finance",
        text="All database instances require multi-factor authentication (MFA) and encrypted backups.",
        metadata=ChunkMetadataInterface(
            document_id="doc_pol_100",
            chunk_id="chunk_sec_001",
            tenant_id="tenant_finance",
            page=1,
            section="Authentication",
            source="security_policy.pdf",
            entities=["MFA", "Encryption"],
        ),
    )
    c2 = ChunkInterface(
        chunk_id="chunk_cat_002",
        document_id="doc_pol_100",
        tenant_id="tenant_finance",
        text="The company cafeteria provides vegetarian lunch options on Tuesdays.",
        metadata=ChunkMetadataInterface(
            document_id="doc_pol_100",
            chunk_id="chunk_cat_002",
            tenant_id="tenant_finance",
            page=4,
            section="Cafeteria",
            source="office_guide.pdf",
            entities=["Cafeteria"],
        ),
    )

    fused1 = FusedCandidateInterface(
        chunk=c1,
        rrf_score=0.032,
        vector_score=0.88,
        bm25_score=12.4,
        graph_score=1.0,
        sources=[RetrievalSourceEnum.VECTOR, RetrievalSourceEnum.BM25],
    )
    fused2 = FusedCandidateInterface(
        chunk=c2,
        rrf_score=0.031,
        vector_score=0.45,
        bm25_score=1.1,
        graph_score=None,
        sources=[RetrievalSourceEnum.VECTOR],
    )
    return [fused1, fused2]


def test_1_neural_reranker_scoring_and_ranking() -> None:
    """TEST 1: Verify Neural Reranker (BAAI/bge-reranker-base) scores relevant chunk higher."""
    set_mock_reranker_mode(False)
    candidates = _create_sample_fused_candidates()
    query = "What authentication security policy is required for database instances?"

    reranked = rerank_candidates(fused_candidates=candidates, query_text=query, top_k=2)

    assert len(reranked) == 2
    assert isinstance(reranked[0], RerankedCandidateInterface)
    # Security chunk (c1) should rank #1 over cafeteria chunk (c2)
    assert reranked[0].chunk.chunk_id == "chunk_sec_001"
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_2_reranker_top_k_and_interface_compliance() -> None:
    """TEST 2: Verify candidate output list obeys top_k and maintains interface attributes."""
    candidates = _create_sample_fused_candidates()
    query = "database authentication policy"

    reranked = rerank_candidates(fused_candidates=candidates, query_text=query, top_k=1)

    assert len(reranked) == 1
    hit = reranked[0]
    assert hit.chunk.document_id == "doc_pol_100"
    assert hit.chunk.tenant_id == "tenant_finance"
    assert hit.fused_score == 0.032
    assert isinstance(hit.rerank_score, float)


def test_3_mock_fallback_reranker_mode() -> None:
    """TEST 3: Verify mock reranker mode forces heuristic fallback correctly."""
    set_mock_reranker_mode(True)
    candidates = _create_sample_fused_candidates()
    query = "cafeteria lunch tuesday"

    reranked = rerank_candidates(fused_candidates=candidates, query_text=query, top_k=2)

    assert len(reranked) == 2
    # In mock mode with cafeteria query, chunk 2 should rank higher
    assert reranked[0].chunk.chunk_id == "chunk_cat_002"

    # Reset mock mode
    set_mock_reranker_mode(False)


def test_4_empty_candidates_handling() -> None:
    """TEST 4: Verify passing empty candidates list returns empty list."""
    res = rerank_candidates(fused_candidates=[], query_text="any query")
    assert res == []
