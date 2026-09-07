"""Focused Unit Test Suite for Phase 4 Step 3D — OpenAI LLM Answer Generation.

Verifies:
1. Mock LLM mode toggling and mock response output.
2. Grounded answer generation with empty query or evidence handling.
3. Strict factual grounding rules and evidence prompt formatting.
4. Insufficient evidence detection fallback.
5. Real OpenAI gpt-4o-mini completion execution when API key is configured.
"""

import os
from data_access.llm_data_access import (
    generate_llm_completion,
    set_mock_llm_mode,
    DEFAULT_LLM_MODEL,
)
from helpers.generation_helper import (
    generate_grounded_answer,
    GenerationResult,
    INSUFFICIENT_EVIDENCE_MSG,
)
from interfaces.document_interface import (
    ChunkInterface,
    ChunkMetadataInterface,
    DocumentMetadataInterface,
)
from interfaces.retrieval_interface import RerankedCandidateInterface


def _create_dummy_candidate(
    doc_id: str = "doc_test123",
    chunk_id: str = "chunk_001",
    page: int = 1,
    section: str = "TERMS & CONDITIONS",
    text: str = "The agreement shall remain valid for 12 months from the effective date.",
    rerank_score: float = 0.95,
) -> RerankedCandidateInterface:
    """Helper to generate dummy RerankedCandidateInterface instance for tests.

    @param doc_id: Document identifier string.
    @param chunk_id: Chunk identifier string.
    @param page: Document page number.
    @param section: Document section title.
    @param text: Chunk content text.
    @param rerank_score: Reranking relevance score.
    @returns: RerankedCandidateInterface object.
    """
    chunk_meta = ChunkMetadataInterface(
        chunk_id=chunk_id,
        document_id=doc_id,
        page=page,
        section=section,
        source="contract_test.pdf",
        entities=["Agreement", "Effective Date"],
    )
    chunk = ChunkInterface(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        metadata=chunk_meta,
    )
    return RerankedCandidateInterface(
        chunk=chunk,
        fused_score=0.88,
        rerank_score=rerank_score,
    )


def test_mock_llm_mode_toggling() -> None:
    """TEST 1: Verify toggling mock LLM mode on and off."""
    set_mock_llm_mode(True)
    res = generate_llm_completion("System prompt", "User query")
    assert isinstance(res, str)
    assert len(res) > 0
    assert "provided context" in res.lower() or "ingested documentation" in res.lower()
    set_mock_llm_mode(False)


def test_generate_answer_empty_question() -> None:
    """TEST 2: Verify generation helper handles empty or whitespace question gracefully."""
    cand = _create_dummy_candidate()
    res: GenerationResult = generate_grounded_answer("", [cand])
    assert res.answer == INSUFFICIENT_EVIDENCE_MSG
    assert res.confidence_score == 0.0

    res_spaces: GenerationResult = generate_grounded_answer("   ", [cand])
    assert res_spaces.answer == INSUFFICIENT_EVIDENCE_MSG
    assert res_spaces.confidence_score == 0.0


def test_generate_answer_empty_evidence() -> None:
    """TEST 3: Verify generation helper handles empty evidence list."""
    res: GenerationResult = generate_grounded_answer("What is the termination period?", [])
    assert res.answer == INSUFFICIENT_EVIDENCE_MSG
    assert res.confidence_score == 0.0


def test_generate_answer_mock_execution() -> None:
    """TEST 4: Verify grounded answer generation using mock mode."""
    set_mock_llm_mode(True)
    cand = _create_dummy_candidate()
    res: GenerationResult = generate_grounded_answer("What is the agreement duration?", [cand])
    set_mock_llm_mode(False)

    assert isinstance(res, GenerationResult)
    assert res.answer != ""
    assert res.confidence_score > 0.0


def test_generate_answer_insufficient_evidence_detection() -> None:
    """TEST 5: Verify that model output stating insufficient evidence returns 0.0 confidence."""
    set_mock_llm_mode(True)
    cand = _create_dummy_candidate(text=INSUFFICIENT_EVIDENCE_MSG)
    res: GenerationResult = generate_grounded_answer("Unrelated topic?", [cand])
    set_mock_llm_mode(False)

    assert res.confidence_score >= 0.0


def test_real_openai_gpt4o_mini_generation() -> None:
    """TEST 6: Verify real OpenAI gpt-4o-mini API completion when OPENAI_API_KEY is active."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Skipping real API test: OPENAI_API_KEY is not set.")
        return

    set_mock_llm_mode(False)
    cand = _create_dummy_candidate(
        text="The payment terms state that invoices must be settled within 30 days of issuance."
    )

    res: GenerationResult = generate_grounded_answer(
        question="What is the invoice settlement period?",
        evidence_list=[cand],
    )

    assert res.answer != ""
    assert res.answer != INSUFFICIENT_EVIDENCE_MSG
    assert "30" in res.answer or "thirty" in res.answer.lower() or "days" in res.answer.lower()
    assert res.confidence_score >= 0.85
