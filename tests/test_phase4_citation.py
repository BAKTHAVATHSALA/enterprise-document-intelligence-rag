"""Focused Unit Test Suite for Phase 4 Step 3E — Citation Validation.

Verifies:
1. Valid citation matching document_id, page, and section in evidence.
2. Invalid citation detection (valid document_id, but invalid page/section).
3. Fabricated citation detection (document_id absent from evidence).
4. Missing citation detection (evidence retrieved but omitted from citations).
5. Parsing inline citations from LLM answer text.
6. Verification summary metrics calculations.
"""

from helpers.citation_helper import (
    build_citations,
    extract_inline_citations,
    validate_citations,
)
from interfaces.document_interface import (
    ChunkInterface,
    ChunkMetadataInterface,
)
from interfaces.query_interface import (
    CitationInterface,
    CitationValidationStatusEnum,
    CitationValidationSummaryInterface,
)
from interfaces.retrieval_interface import RerankedCandidateInterface


def _create_mock_candidate(
    doc_id: str = "doc_abc123",
    chunk_id: str = "chunk_001",
    page: int = 1,
    section: str = "PAYMENT TERMS",
    text: str = "Invoices must be settled within 30 days of receipt.",
) -> RerankedCandidateInterface:
    """Helper to construct dummy RerankedCandidateInterface for citation tests.

    @param doc_id: Document identifier string.
    @param chunk_id: Chunk identifier string.
    @param page: Page number integer.
    @param section: Section heading title.
    @param text: Chunk content text.
    @returns: RerankedCandidateInterface instance.
    """
    chunk_meta = ChunkMetadataInterface(
        chunk_id=chunk_id,
        document_id=doc_id,
        page=page,
        section=section,
        source="contract_abc.pdf",
        entities=["Invoice", "30 Days"],
    )
    chunk = ChunkInterface(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        metadata=chunk_meta,
    )
    return RerankedCandidateInterface(
        chunk=chunk,
        fused_score=0.90,
        rerank_score=0.95,
    )


def test_extract_inline_citations() -> None:
    """TEST 1: Verify parsing inline citation markers from LLM response text."""
    answer_text = (
        "The agreement lasts for 12 months [Doc: doc_abc123, Page: 1, Section: PAYMENT TERMS]. "
        "Termination requires 30 days notice [Doc: doc_xyz789, Page: 4]."
    )
    extracted = extract_inline_citations(answer_text)

    assert len(extracted) == 2
    assert extracted[0].document_id == "doc_abc123"
    assert extracted[0].page == 1
    assert extracted[0].section == "PAYMENT TERMS"

    assert extracted[1].document_id == "doc_xyz789"
    assert extracted[1].page == 4
    assert extracted[1].section == "General"


def test_valid_citation_verification() -> None:
    """TEST 2: Verify that valid citations matching document_id, page, and section evaluate to VALID."""
    cand = _create_mock_candidate(doc_id="doc_abc123", page=1, section="PAYMENT TERMS")

    citation = CitationInterface(
        document_id="doc_abc123",
        page=1,
        section="PAYMENT TERMS",
        chunk_id="",
        source="",
        snippet="",
    )

    validated_list, summary = validate_citations([citation], [cand])

    assert len(validated_list) == 1
    val_cit = validated_list[0]
    assert val_cit.is_valid is True
    assert val_cit.validation_status == CitationValidationStatusEnum.VALID
    assert val_cit.chunk_id == "chunk_001"
    assert summary.valid_count == 1
    assert summary.is_fully_validated is True


def test_invalid_citation_detection() -> None:
    """TEST 3: Verify detection of invalid citation (matching document_id, but invalid page/section)."""
    cand = _create_mock_candidate(doc_id="doc_abc123", page=1, section="PAYMENT TERMS")

    invalid_citation = CitationInterface(
        document_id="doc_abc123",
        page=99,  # Non-existent page number
        section="NON_EXISTENT_SECTION",
        chunk_id="",
        source="",
        snippet="",
    )

    validated_list, summary = validate_citations([invalid_citation], [cand])

    assert summary.invalid_count == 1
    val_cit = validated_list[0]
    assert val_cit.is_valid is False
    assert val_cit.validation_status == CitationValidationStatusEnum.INVALID
    assert "Invalid citation" in val_cit.validation_message
    assert summary.is_fully_validated is False


def test_fabricated_citation_detection() -> None:
    """TEST 4: Verify detection of fabricated citation (document_id absent from evidence)."""
    cand = _create_mock_candidate(doc_id="doc_abc123", page=1, section="PAYMENT TERMS")

    fabricated_citation = CitationInterface(
        document_id="doc_FAKE_HALLUCINATED_ID",
        page=1,
        section="PAYMENT TERMS",
        chunk_id="",
        source="",
        snippet="",
    )

    validated_list, summary = validate_citations([fabricated_citation], [cand])

    assert summary.fabricated_count == 1
    val_cit = validated_list[0]
    assert val_cit.is_valid is False
    assert val_cit.validation_status == CitationValidationStatusEnum.FABRICATED
    assert "Fabricated citation" in val_cit.validation_message
    assert summary.is_fully_validated is False


def test_missing_citation_detection() -> None:
    """TEST 5: Verify detection of missing citations when evidence is retrieved but omitted from citations."""
    cand1 = _create_mock_candidate(doc_id="doc_abc123", chunk_id="chunk_001", page=1, section="PAYMENT TERMS")
    cand2 = _create_mock_candidate(doc_id="doc_xyz789", chunk_id="chunk_002", page=2, section="GOVERNING LAW")

    # Only cite cand1
    citation1 = CitationInterface(
        document_id="doc_abc123",
        page=1,
        section="PAYMENT TERMS",
    )

    validated_list, summary = validate_citations([citation1], [cand1, cand2])

    assert summary.valid_count == 1
    assert summary.missing_count == 1
    assert summary.is_fully_validated is False

    missing_cit = next(c for c in validated_list if c.validation_status == CitationValidationStatusEnum.MISSING)
    assert missing_cit.document_id == "doc_xyz789"
    assert missing_cit.is_valid is False
    assert "Missing citation" in missing_cit.validation_message


def test_validation_summary_metrics() -> None:
    """TEST 6: Verify validation summary metric computation with mixed citation types."""
    cand1 = _create_mock_candidate(doc_id="doc_1", chunk_id="c1", page=1, section="SEC1")
    cand2 = _create_mock_candidate(doc_id="doc_2", chunk_id="c2", page=2, section="SEC2")

    citations = [
        CitationInterface(document_id="doc_1", page=1, section="SEC1"),  # VALID
        CitationInterface(document_id="doc_1", page=99, section="WRONG"),  # INVALID
        CitationInterface(document_id="doc_FABRICATED", page=1, section="SEC1"),  # FABRICATED
    ]

    validated_list, summary = validate_citations(citations, [cand1, cand2])

    assert summary.valid_count == 1
    assert summary.invalid_count == 1
    assert summary.fabricated_count == 1
    assert summary.missing_count == 1
    assert summary.total_citations == 4  # 3 input + 1 missing
    assert summary.is_fully_validated is False
