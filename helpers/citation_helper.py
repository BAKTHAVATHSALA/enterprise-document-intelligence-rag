"""Citation Generation and Validation Helper.

Extracts verifiable citation objects from evidence context and LLM answer text,
and validates that cited sources accurately match retrieved evidence attributes.
"""

import re
from interfaces.query_interface import (
    CitationInterface,
    CitationValidationStatusEnum,
    CitationValidationSummaryInterface,
)
from interfaces.retrieval_interface import RerankedCandidateInterface
from utils.logger import logger
from utils.tracing import trace_step

FUNC_BUILD_CITATIONS: str = "build_citations"
FUNC_VALIDATE_CITATIONS: str = "validate_citations"
FUNC_EXTRACT_INLINE: str = "extract_inline_citations"

SNIPPET_MAX_LEN: int = 150

INLINE_CITATION_PATTERN: str = r"\[Doc:\s*([^,\]]+),\s*Page:\s*(\d+)(?:,\s*Section:\s*([^\]]+))?\]"


def extract_inline_citations(answer_text: str) -> list[CitationInterface]:
    """Parse and extract inline citation markers from LLM generated answer text.

    @param answer_text: Generated response text containing inline citations.
    @returns: List of extracted CitationInterface objects.
    """
    if not answer_text or not answer_text.strip():
        return []

    citations: list[CitationInterface] = []
    seen: set[tuple[str, int, str]] = set()

    matches = re.finditer(INLINE_CITATION_PATTERN, answer_text, re.IGNORECASE)
    for match in matches:
        doc_id: str = match.group(1).strip()
        page_num: int = int(match.group(2))
        section_name: str = match.group(3).strip() if match.group(3) else "General"

        key = (doc_id, page_num, section_name.lower())
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            CitationInterface(
                document_id=doc_id,
                page=page_num,
                section=section_name,
                chunk_id="",
                source="",
                snippet="",
                is_valid=True,
                validation_status=CitationValidationStatusEnum.VALID,
                validation_message="Extracted from answer text",
            )
        )

    logger.info(FUNC_EXTRACT_INLINE, f"Extracted {len(citations)} inline citations from answer text.")
    return citations


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
                validation_status=CitationValidationStatusEnum.VALID,
                validation_message="Citation constructed from evidence chunk",
            )
        )

    logger.info(FUNC_BUILD_CITATIONS, f"Extracted {len(citations)} source citations from evidence context.")
    return citations


def _extract_citation_validation_metadata(args, kwargs, result, error):
    cits = args[0] if args else kwargs.get("citations", [])
    ev = args[1] if len(args) > 1 else kwargs.get("evidence_list", [])
    meta = {
        "citations_in": len(cits) if isinstance(cits, list) else 0,
        "evidence_count": len(ev) if isinstance(ev, list) else 0,
    }
    if result and isinstance(result, tuple) and len(result) >= 2:
        val_list, summary = result[0], result[1]
        if summary:
            meta["total_citations"] = summary.total_citations
            meta["valid_count"] = summary.valid_count
            meta["invalid_count"] = summary.invalid_count
            meta["fabricated_count"] = summary.fabricated_count
            meta["missing_count"] = summary.missing_count
            meta["is_fully_validated"] = summary.is_fully_validated
    return meta


@trace_step(name="citation_validation", run_type="chain", extract_metadata=_extract_citation_validation_metadata)
def validate_citations(
    citations: list[CitationInterface],
    evidence_list: list[RerankedCandidateInterface],
) -> tuple[list[CitationInterface], CitationValidationSummaryInterface]:
    """Validate citations against retrieved evidence chunks to verify document_id, page, section, and text.

    @param citations: CitationInterface list to validate.
    @param evidence_list: Underlying retrieved evidence chunks.
    @returns: Tuple containing list of validated CitationInterface objects and CitationValidationSummaryInterface.
    """
    if not evidence_list:
        # If no evidence was retrieved, mark all citations as FABRICATED
        validated: list[CitationInterface] = []
        for cit in citations:
            cit.is_valid = False
            cit.validation_status = CitationValidationStatusEnum.FABRICATED
            cit.validation_message = f"Fabricated citation: Document '{cit.document_id}' does not exist in retrieved evidence."
            validated.append(cit)

        summary = CitationValidationSummaryInterface(
            total_citations=len(citations),
            valid_count=0,
            invalid_count=0,
            fabricated_count=len(citations),
            missing_count=0,
            is_fully_validated=(len(citations) == 0),
        )
        logger.info(FUNC_VALIDATE_CITATIONS, f"No evidence available. Marked {len(citations)} citations as FABRICATED.")
        return validated, summary

    # Index evidence context
    evidence_doc_ids: set[str] = {cand.chunk.metadata.document_id for cand in evidence_list}
    evidence_chunks_by_doc: dict[str, list[RerankedCandidateInterface]] = {}
    for cand in evidence_list:
        doc_id = cand.chunk.metadata.document_id
        if doc_id not in evidence_chunks_by_doc:
            evidence_chunks_by_doc[doc_id] = []
        evidence_chunks_by_doc[doc_id].append(cand)

    validated_citations: list[CitationInterface] = []
    valid_count: int = 0
    invalid_count: int = 0
    fabricated_count: int = 0
    missing_count: int = 0

    cited_evidence_keys: set[tuple[str, int, str]] = set()

    for citation in citations:
        doc_id = citation.document_id
        page = citation.page
        section = citation.section

        # 1. Check FABRICATED: Document ID not present in retrieved evidence
        if doc_id not in evidence_doc_ids:
            citation.is_valid = False
            citation.validation_status = CitationValidationStatusEnum.FABRICATED
            citation.validation_message = f"Fabricated citation: Document '{doc_id}' does not exist in retrieved evidence."
            fabricated_count += 1
            validated_citations.append(citation)
            continue

        # 2. Check INVALID: Document ID exists, but page or section does not match any chunk for that document
        doc_cands = evidence_chunks_by_doc[doc_id]
        matching_cand: RerankedCandidateInterface | None = None

        for cand in doc_cands:
            c_meta = cand.chunk.metadata
            page_matches = (c_meta.page == page)
            section_matches = (
                c_meta.section.lower() == section.lower()
                or section.lower() in c_meta.section.lower()
                or c_meta.section.lower() in section.lower()
                or section == "General"
            )
            if page_matches and section_matches:
                matching_cand = cand
                break

        if matching_cand is None:
            # Check if page matches at least
            page_matched_cand = next((c for c in doc_cands if c.chunk.metadata.page == page), None)
            if page_matched_cand:
                matching_cand = page_matched_cand

        if matching_cand is None:
            citation.is_valid = False
            citation.validation_status = CitationValidationStatusEnum.INVALID
            citation.validation_message = f"Invalid citation: Document '{doc_id}' exists, but Page {page} / Section '{section}' is invalid."
            invalid_count += 1
            validated_citations.append(citation)
            continue

        # 3. VALID Citation
        meta = matching_cand.chunk.metadata
        citation.is_valid = True
        citation.validation_status = CitationValidationStatusEnum.VALID
        citation.chunk_id = meta.chunk_id
        citation.source = meta.source
        if not citation.snippet:
            raw_text = matching_cand.chunk.text.strip()
            citation.snippet = raw_text[:SNIPPET_MAX_LEN] + "..." if len(raw_text) > SNIPPET_MAX_LEN else raw_text
        citation.validation_message = f"Citation verified against document '{meta.document_id}' page {meta.page}."
        valid_count += 1
        cited_evidence_keys.add((meta.document_id, meta.page, meta.section.lower()))
        validated_citations.append(citation)

    # 4. Detect MISSING Citations: Evidence retrieved but omitted from citations
    seen_missing: set[tuple[str, int, str]] = set()
    for cand in evidence_list:
        meta = cand.chunk.metadata
        key = (meta.document_id, meta.page, meta.section.lower())
        if key not in cited_evidence_keys and key not in seen_missing and len(citations) > 0:
            seen_missing.add(key)
            missing_count += 1
            raw_text = cand.chunk.text.strip()
            snippet = raw_text[:SNIPPET_MAX_LEN] + "..." if len(raw_text) > SNIPPET_MAX_LEN else raw_text
            validated_citations.append(
                CitationInterface(
                    document_id=meta.document_id,
                    page=meta.page,
                    section=meta.section,
                    chunk_id=meta.chunk_id,
                    source=meta.source,
                    snippet=snippet,
                    is_valid=False,
                    validation_status=CitationValidationStatusEnum.MISSING,
                    validation_message=f"Missing citation: Evidence chunk from Document '{meta.document_id}' Page {meta.page} was omitted.",
                )
            )

    is_fully_validated = (invalid_count == 0 and fabricated_count == 0 and missing_count == 0)
    summary = CitationValidationSummaryInterface(
        total_citations=len(validated_citations),
        valid_count=valid_count,
        invalid_count=invalid_count,
        fabricated_count=fabricated_count,
        missing_count=missing_count,
        is_fully_validated=is_fully_validated,
    )

    logger.info(
        FUNC_VALIDATE_CITATIONS,
        f"Validated citations: {valid_count} valid, {invalid_count} invalid, {fabricated_count} fabricated, {missing_count} missing.",
    )
    return validated_citations, summary
