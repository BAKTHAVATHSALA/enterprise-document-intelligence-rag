"""Focused Unit and Integration Tests for Phase 5 Step 5.4 LangSmith Tracing.

Verifies:
- Tracing enablement/disablement behavior (zero overhead and bypass when disabled).
- Secret and credential sanitization (strict redaction of API keys and prefixes).
- Span recording across all 7 Hybrid RAG pipeline stages:
  1. query_pipeline
  2. embedding
  3. vector_retrieval, bm25_retrieval, graph_retrieval
  4. rrf_fusion
  5. reranking
  6. llm_generation
  7. citation_validation
- Structured metadata extraction (query_id, document_id, retrieval_methods, latency_ms, model_name).
- Error handling resilience (exceptions captured in span metadata, original exception cleanly raised).
- Tracing gracefully disabled when credentials or flags are absent.
- Non-intrusive behavior (identical RAG results with tracing on vs off).
"""

import os
import re
import pytest
from unittest.mock import patch, MagicMock

from interfaces.document_interface import ChunkInterface, ChunkMetadataInterface
from interfaces.retrieval_interface import (
    CandidateChunkInterface,
    FusedCandidateInterface,
    RerankedCandidateInterface,
    RetrievalSourceEnum,
)
from interfaces.query_interface import (
    QueryRequestInterface,
    QueryResponseInterface,
    CitationInterface,
)
from utils.tracing import (
    is_tracing_enabled,
    set_tracing_override,
    configure_tracing_environment,
    sanitize_metadata,
    trace_step,
    get_trace_records,
    clear_trace_records,
)
from services.query_service import execute_query_workflow
from data_access.vector_data_access import (
    generate_embedding,
    query_vector_store,
    upsert_vector_chunks,
    get_pinecone_index,
    clear_pinecone_index_cache,
)
from data_access.bm25_data_access import query_bm25_index, index_bm25_chunks
from data_access.graph_data_access import query_graph_store
from helpers.retrieval_helper import apply_rrf_fusion, rerank_candidates
from helpers.generation_helper import generate_grounded_answer
from helpers.citation_helper import validate_citations, build_citations


@pytest.fixture(autouse=True)
def reset_tracing_state():
    """Reset tracing override, record history, and pinecone cache before and after each test."""
    clear_trace_records()
    clear_pinecone_index_cache()
    set_tracing_override(False)
    yield
    clear_trace_records()
    clear_pinecone_index_cache()
    set_tracing_override(False)


def _make_mock_chunk(chunk_id: str = "chk_001", doc_id: str = "doc_001", text: str = "Test chunk text.") -> ChunkInterface:
    return ChunkInterface(
        chunk_id=chunk_id,
        document_id=doc_id,
        tenant_id="default_tenant",
        text=text,
        metadata=ChunkMetadataInterface(
            document_id=doc_id,
            chunk_id=chunk_id,
            page=1,
            section="Section A",
            source="test_doc.pdf",
        ),
    )


def test_tracing_enabled_and_disabled_modes():
    """Verify programmatic override toggles is_tracing_enabled correctly."""
    set_tracing_override(False)
    assert not is_tracing_enabled()

    set_tracing_override(True)
    assert is_tracing_enabled()

    # Clear override to fall back to environment config
    set_tracing_override(None)


def test_metadata_sanitization():
    """Verify sanitize_metadata scrubs sensitive keys and prefixes."""
    dirty_meta = {
        "query_id": "query_12345",
        "latency_ms": 12.5,
        "openai_api_key": "sk-proj-supersecret1234567890",
        "pinecone_api_key": "pcsk_verysecrettoken",
        "langsmith_api_key": "lsv2_pt_secrettoken",
        "db_password": "npg_supersecretpassword",
        "token": "bearer xyz",
        "safe_model": "gpt-4o-mini",
        "sample_identifiers": ["sk-12345", "safe_value", "lsv2_abc"],
    }

    clean = sanitize_metadata(dirty_meta)

    # Sensitive keys must be redacted
    assert clean["openai_api_key"] == "[REDACTED]"
    assert clean["pinecone_api_key"] == "[REDACTED]"
    assert clean["langsmith_api_key"] == "[REDACTED]"
    assert clean["db_password"] == "[REDACTED]"
    assert clean["token"] == "[REDACTED]"

    # Non-sensitive keys must be preserved
    assert clean["query_id"] == "query_12345"
    assert clean["latency_ms"] == 12.5
    assert clean["safe_model"] == "gpt-4o-mini"

    # Nested secret values must be redacted
    assert clean["sample_identifiers"] == ["[REDACTED]", "safe_value", "[REDACTED]"]


def test_tracing_disabled_bypasses_records():
    """Verify that when tracing is disabled, no trace records are accumulated."""
    set_tracing_override(False)
    clear_trace_records()

    emb = generate_embedding("Sample text for disabled tracing test")
    assert isinstance(emb, list)
    assert len(get_trace_records()) == 0


def test_embedding_span_recorded():
    """Verify embedding step produces a trace record with expected metadata."""
    set_tracing_override(True)
    clear_trace_records()

    emb = generate_embedding("Semantic embedding test string")
    assert isinstance(emb, list)

    records = [r for r in get_trace_records() if r["name"] == "embedding"]
    assert len(records) >= 1
    rec = records[0]
    assert rec["run_type"] == "embedding"
    assert "latency_ms" in rec["metadata"]
    assert "model_name" in rec["metadata"]
    assert rec["metadata"]["text_length"] > 0
    assert rec["metadata"]["dimension"] == len(emb)


def test_retrieval_spans_recorded():
    """Verify vector, bm25, and graph retrieval steps record spans and metadata."""
    set_tracing_override(True)
    clear_trace_records()

    chunk = _make_mock_chunk(chunk_id="chk_bm_1", doc_id="doc_bm_1", text="Cloud architecture and security")
    index_bm25_chunks([chunk])

    # 1. BM25 Retrieval
    bm25_res = query_bm25_index(query_text="cloud security", top_k=3)
    records = [r for r in get_trace_records() if r["name"] == "bm25_retrieval"]
    assert len(records) == 1
    assert records[0]["run_type"] == "retriever"
    assert records[0]["metadata"]["retrieval_method"] == "bm25"
    assert records[0]["metadata"]["top_k"] == 3

    # 2. Vector Retrieval (mock mode)
    vec_res = query_vector_store(query_text="vector similarity", top_k=2, use_local_mock=True)
    v_records = [r for r in get_trace_records() if r["name"] == "vector_retrieval"]
    assert len(v_records) == 1
    assert v_records[0]["run_type"] == "retriever"
    assert v_records[0]["metadata"]["retrieval_method"] == "vector"

    # 3. Graph Retrieval (mock mode)
    graph_res = query_graph_store(query_text="graph entity", top_k=2, use_local_mock=True)
    g_records = [r for r in get_trace_records() if r["name"] == "graph_retrieval"]
    assert len(g_records) == 1
    assert g_records[0]["run_type"] == "retriever"
    assert g_records[0]["metadata"]["retrieval_method"] == "graph"


def test_rrf_fusion_and_reranking_spans():
    """Verify RRF fusion and neural reranking steps record spans with proper metadata."""
    set_tracing_override(True)
    clear_trace_records()

    chunk1 = _make_mock_chunk(chunk_id="c1", doc_id="d1", text="Chunk one")
    chunk2 = _make_mock_chunk(chunk_id="c2", doc_id="d2", text="Chunk two")

    cand1 = CandidateChunkInterface(chunk=chunk1, score=0.9, source=RetrievalSourceEnum.VECTOR, raw_rank=1)
    cand2 = CandidateChunkInterface(chunk=chunk2, score=0.8, source=RetrievalSourceEnum.BM25, raw_rank=1)

    fused = apply_rrf_fusion([[cand1], [cand2]])
    rrf_records = [r for r in get_trace_records() if r["name"] == "rrf_fusion"]
    assert len(rrf_records) == 1
    assert rrf_records[0]["run_type"] == "chain"
    assert rrf_records[0]["metadata"]["channels_count"] == 2
    assert rrf_records[0]["metadata"]["fused_count"] == 2

    # Reranking with mock
    with patch("data_access.rerank_fused_candidates") as mock_rerank:
        mock_rerank.return_value = [
            RerankedCandidateInterface(
                chunk=chunk1,
                fused_score=0.03,
                rerank_score=0.95,
            )
        ]
        reranked = rerank_candidates(fused_candidates=fused, query_text="test query", top_k=1)

    rerank_records = [r for r in get_trace_records() if r["name"] == "reranking"]
    assert len(rerank_records) == 1
    assert rerank_records[0]["run_type"] == "chain"
    assert rerank_records[0]["metadata"]["candidates_in"] == 2
    assert rerank_records[0]["metadata"]["candidates_out"] == 1
    assert rerank_records[0]["metadata"]["model_name"] == "BAAI/bge-reranker-base"


def test_llm_generation_and_citation_validation_spans():
    """Verify LLM generation and citation validation record spans with metadata."""
    set_tracing_override(True)
    clear_trace_records()

    chunk = _make_mock_chunk(chunk_id="c1", doc_id="doc_xyz", text="Contract duration is 36 months.")
    reranked = [
        RerankedCandidateInterface(
            chunk=chunk,
            fused_score=0.03,
            rerank_score=0.95,
        )
    ]

    with patch("helpers.generation_helper.generate_llm_completion") as mock_llm:
        mock_llm.return_value = "Contract duration is 36 months. [Doc: doc_xyz, Page: 1, Section: Section A]"
        gen_res = generate_grounded_answer(question="What is the contract duration?", evidence_list=reranked)

    gen_records = [r for r in get_trace_records() if r["name"] == "llm_generation"]
    assert len(gen_records) == 1
    assert gen_records[0]["run_type"] == "llm"
    assert gen_records[0]["metadata"]["model_name"] == "gpt-4o-mini"
    assert gen_records[0]["metadata"]["evidence_count"] == 1

    # Citation Validation
    citations = build_citations(reranked)
    val_cits, val_sum = validate_citations(citations=citations, evidence_list=reranked)

    cit_records = [r for r in get_trace_records() if r["name"] == "citation_validation"]
    assert len(cit_records) == 1
    assert cit_records[0]["run_type"] == "chain"
    assert cit_records[0]["metadata"]["is_fully_validated"] is True
    assert cit_records[0]["metadata"]["valid_count"] >= 1


def test_error_handling_in_trace_step():
    """Verify decorated function catches exceptions, logs error metadata, and re-raises original exception."""
    set_tracing_override(True)
    clear_trace_records()

    @trace_step(name="failing_operation", run_type="chain")
    def faulty_action(x: int) -> int:
        if x < 0:
            raise ValueError("Negative value not allowed")
        return x * 2

    # Normal call
    res = faulty_action(5)
    assert res == 10
    assert len(get_trace_records()) == 1
    assert get_trace_records()[0]["error"] is None

    # Failing call
    with pytest.raises(ValueError, match="Negative value not allowed"):
        faulty_action(-1)

    records = [r for r in get_trace_records() if r["name"] == "failing_operation"]
    assert len(records) == 2
    failing_record = records[1]
    assert failing_record["error"] == "Negative value not allowed"
    assert failing_record["metadata"]["error_type"] == "ValueError"


def test_query_pipeline_end_to_end_tracing():
    """Verify end-to-end execute_query_workflow traces parent query_pipeline span."""
    set_tracing_override(True)
    clear_trace_records()

    chunk = _make_mock_chunk(chunk_id="chk_e2e_1", doc_id="doc_e2e_1", text="Compliance is audited annually.")
    reranked = [
        RerankedCandidateInterface(
            chunk=chunk,
            fused_score=0.03,
            rerank_score=0.9,
        )
    ]

    with patch("services.query_service.execute_hybrid_retrieval", return_value=reranked), \
         patch("services.query_service.generate_grounded_answer") as mock_gen:
        from helpers.generation_helper import GenerationResult
        mock_gen.return_value = GenerationResult(
            answer="Compliance is audited annually. [Doc: doc_e2e_1, Page: 1, Section: Section A]",
            confidence_score=0.95,
        )

        req = QueryRequestInterface(question="How often is compliance audited?", top_k=3)
        resp = execute_query_workflow(req)

    assert isinstance(resp, QueryResponseInterface)
    assert resp.confidence_score == 0.95

    # Check that query_pipeline span was captured
    pipe_records = [r for r in get_trace_records() if r["name"] == "query_pipeline"]
    assert len(pipe_records) == 1
    pipe = pipe_records[0]
    assert pipe["run_type"] == "chain"
    assert "query_id" in pipe["metadata"]
    assert pipe["metadata"]["top_k"] == 3
    assert pipe["metadata"]["confidence_score"] == 0.95
    assert pipe["metadata"]["valid_citations"] >= 1


def test_non_intrusive_observability():
    """Verify that results from execute_query_workflow are identical whether tracing is enabled or disabled."""
    chunk = _make_mock_chunk(chunk_id="c_same", doc_id="doc_same", text="Policy rules.")
    reranked = [
        RerankedCandidateInterface(
            chunk=chunk,
            fused_score=0.03,
            rerank_score=0.9,
        )
    ]

    req = QueryRequestInterface(question="What are policy rules?", top_k=2)

    with patch("services.query_service.execute_hybrid_retrieval", return_value=reranked), \
         patch("services.query_service.generate_grounded_answer") as mock_gen:
        from helpers.generation_helper import GenerationResult
        mock_gen.return_value = GenerationResult(
            answer="Policy rules. [Doc: doc_same, Page: 1, Section: Section A]",
            confidence_score=0.95,
        )

        # 1. Tracing disabled
        set_tracing_override(False)
        resp_off = execute_query_workflow(req)

        # 2. Tracing enabled
        set_tracing_override(True)
        resp_on = execute_query_workflow(req)

    assert resp_off.answer == resp_on.answer
    assert resp_off.confidence_score == resp_on.confidence_score
    assert len(resp_off.citations) == len(resp_on.citations)


def test_tracing_disabled_when_credentials_absent():
    """Verify is_tracing_enabled evaluates to False when API key is missing."""
    with patch("utils.tracing.get_config") as mock_cfg, \
         patch.dict(os.environ, {"LANGSMITH_API_KEY": "", "LANGSMITH_TRACING": "true"}, clear=False):
        cfg_obj = MagicMock()
        cfg_obj.langsmith_api_key = ""
        cfg_obj.langsmith_tracing = True
        mock_cfg.return_value = cfg_obj

        set_tracing_override(None)
        assert is_tracing_enabled() is False


def test_no_secret_leaks_in_source_code():
    """Scan all tracked source files to ensure no API keys or secrets are hardcoded."""
    secret_patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"pcsk_[a-zA-Z0-9_]{20,}"),
        re.compile(r"lsv2_pt_[a-zA-Z0-9_]{20,}"),
    ]

    source_dirs = ["services", "helpers", "data_access", "eval", "utils", "interfaces"]
    for sdir in source_dirs:
        for root, _, files in os.walk(sdir):
            for file in files:
                if file.endswith(".py"):
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                        for pat in secret_patterns:
                            match = pat.search(content)
                            assert match is None, f"Found hardcoded secret in {fpath}: {match.group() if match else ''}"


def test_query_vector_store_triggers_embedding_span():
    """Verify query_vector_store invokes generate_embedding producing the embedding trace span."""
    set_tracing_override(True)
    clear_trace_records()

    chunk = _make_mock_chunk(chunk_id="chk_emb_test", doc_id="doc_emb_test", text="Sample embedded content")
    upsert_vector_chunks([chunk], use_local_mock=True)
    clear_trace_records()
    
    # Run mock query
    hits = query_vector_store(query_text="Find embedded content", top_k=2, use_local_mock=True)

    names = [r["name"] for r in get_trace_records()]
    assert "vector_retrieval" in names
    assert "embedding" in names

    emb_record = next(r for r in get_trace_records() if r["name"] == "embedding")
    assert emb_record["run_type"] == "embedding"
    assert "model_name" in emb_record["metadata"]
    assert emb_record["metadata"]["text_length"] == len("Find embedded content")
    assert emb_record["metadata"]["dimension"] > 0


def test_pinecone_index_caching_and_hot_path():
    """Verify get_pinecone_index caches validated index and skips list_indexes / describe_index_stats on subsequent queries."""
    clear_pinecone_index_cache()

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="enterprise-rag")]
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value.dimension = 1536
    mock_pc.Index.return_value = mock_index

    with patch("data_access.vector_data_access.Pinecone", return_value=mock_pc):
        # 1. Cold path: First call must validate existence and stats
        idx1 = get_pinecone_index(api_key="test-cache-key", index_name="enterprise-rag", dimension=1536)
        assert idx1 is not None
        assert mock_pc.list_indexes.call_count == 1
        assert mock_index.describe_index_stats.call_count == 1

        # 2. Hot path: Second call must reuse validated index and NOT call list_indexes or describe_index_stats
        idx2 = get_pinecone_index(api_key="test-cache-key", index_name="enterprise-rag", dimension=1536)
        assert idx2 is not None
        assert mock_pc.list_indexes.call_count == 1
        assert mock_index.describe_index_stats.call_count == 1

        # 3. Third call: Still 1
        idx3 = get_pinecone_index(api_key="test-cache-key", index_name="enterprise-rag", dimension=1536)
        assert idx3 is not None
        assert mock_pc.list_indexes.call_count == 1
        assert mock_index.describe_index_stats.call_count == 1


def test_clear_pinecone_index_cache():
    """Verify clear_pinecone_index_cache evicts cached index forcing re-validation."""
    clear_pinecone_index_cache()

    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="enterprise-rag")]
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value.dimension = 1536
    mock_pc.Index.return_value = mock_index

    with patch("data_access.vector_data_access.Pinecone", return_value=mock_pc):
        # Call 1: cold
        get_pinecone_index(api_key="test-evict-key", index_name="enterprise-rag", dimension=1536)
        assert mock_pc.list_indexes.call_count == 1

        # Call 2: hot
        get_pinecone_index(api_key="test-evict-key", index_name="enterprise-rag", dimension=1536)
        assert mock_pc.list_indexes.call_count == 1

        # Evict cache
        clear_pinecone_index_cache()

        # Call 3: must re-validate
        get_pinecone_index(api_key="test-evict-key", index_name="enterprise-rag", dimension=1536)
        assert mock_pc.list_indexes.call_count == 2
        assert mock_index.describe_index_stats.call_count == 2


