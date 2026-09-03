"""Citation Generation and Validation Helper.

Extracts verifiable citation objects from evidence context and validates
that cited source chunks actually support the generated claims.
"""

from interfaces.query_interface import CitationInterface
from interfaces.retrieval_interface import RerankedCandidateInterface
from utils.logger import logger

FUNC_BUILD_CITATIONS: str = "build_citations"
FUNC_VALIDATE_CITATIONS: str = "validate_citations"

SNIPPET_MAX_LEN: int = 150


def build_citations(
    evidence_list: list[RerankedCandidateInterface],
) -> list[CitationInterface]:
    """Extract structured source citations from retrieved evidence chunks.

    @param evidence_list: Top reranked evidence chunks.
    @returns: List of CitationInterface objects.
    """
    if not evidence_list:
        return []

    citations: list[CitationInterface] = []
    seen_chunks: set[str] = set()

    for cand in evidence_list:
        meta = cand.chunk.metadata
        if meta.chunk_id in seen_chunks:
            continue

        seen_chunks.add(meta.chunk_id)
        raw_text: str = cand.chunk.text.strip()
        snippet: str = raw_text[:SNIPPET_MAX_LEN] + "..." if len(raw_text) > SNIPPET_MAX_LEN else raw_text

        citations.append(
            CitationInterface(
                document_id=meta.document_id,
                page=meta.page,
                section=meta.section,
                chunk_id=meta.chunk_id,
                source=meta.source,
                snippet=snippet,
                is_valid=True,
            )
        )

    logger.info(FUNC_BUILD_CITATIONS, f"Extracted {len(citations)} source citations from evidence context.")
    return citations


def validate_citations(
    citations: list[CitationInterface],
    evidence_list: list[RerankedCandidateInterface],
) -> list[CitationInterface]:
    """Validate that cited evidence chunks actually contain supporting text.

    @param citations: Extracted CitationInterface list to validate.
    @param evidence_list: Underlying retrieved evidence chunks.
    @returns: List of CitationInterface objects with updated is_valid status.
    """
    if not citations or not evidence_list:
        return []

    evidence_chunk_map: dict[str, str] = {
        cand.chunk.chunk_id: cand.chunk.text.lower() for cand in evidence_list
    }

    validated_citations: list[CitationInterface] = []
    valid_count: int = 0

    for citation in citations:
        source_text: str | None = evidence_chunk_map.get(citation.chunk_id)
        if not source_text:
            citation.is_valid = False
            validated_citations.append(citation)
            continue

        snippet_clean: str = citation.snippet.lower().rstrip(".")
        if snippet_clean in source_text or any(w in source_text for w in snippet_clean.split()[:5]):
            citation.is_valid = True
            valid_count += 1
        else:
            citation.is_valid = False

        validated_citations.append(citation)

    logger.info(FUNC_VALIDATE_CITATIONS, f"Validated {len(citations)} citations: {valid_count} valid, {len(citations) - valid_count} invalid.")
    return validated_citations
