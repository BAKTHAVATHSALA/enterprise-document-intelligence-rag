"""Query Service Layer.

Coordinates end-to-end RAG query execution, prompt injection security validation,
retrieval, generation, citation verification, and latency measurement.
"""

import time
import re
from typing import Optional
from interfaces.query_interface import (
    QueryRequestInterface,
    QueryResponseInterface,
    CitationInterface,
)
from interfaces.retrieval_interface import RerankedCandidateInterface
from helpers.retrieval_helper import execute_hybrid_retrieval
from helpers.generation_helper import generate_grounded_answer, GenerationResult
from helpers.citation_helper import build_citations, validate_citations
from utils.logger import logger

FUNC_EXECUTE_QUERY: str = "execute_query_workflow"
FUNC_VALIDATE_QUERY: str = "validate_query_security"

# Prompt Injection Defense Patterns
PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+)?previous\s+instructions",
    r"bypass\s+safety",
    r"reveal\s+system\s+prompt",
    r"jailbreak",
    r"you\s+are\s+now\s+a",
]

MAX_QUERY_CHAR_LENGTH: int = 1000


def validate_query_security(question: str) -> None:
    """Validate query size and defend against prompt injection attacks.

    @param question: User input question string.
    @raises ValueError: If query violates security policies or size limits.
    """
    if not question or not question.strip():
        raise ValueError("Query question cannot be empty.")

    if len(question) > MAX_QUERY_CHAR_LENGTH:
        raise ValueError(f"Query length exceeds maximum limit of {MAX_QUERY_CHAR_LENGTH} characters.")

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning(FUNC_VALIDATE_QUERY, f"Security alert: Prompt injection attempt blocked in query.")
            raise ValueError("Query rejected due to security policy violation.")


def execute_query_workflow(
    request: QueryRequestInterface,
) -> QueryResponseInterface:
    """Execute end-to-end Hybrid RAG pipeline and return grounded response.

    @param request: Input QueryRequestInterface payload.
    @returns: Grounded QueryResponseInterface output.
    """
    start_time: float = time.perf_counter()
    
    # 1. Security & Validation
    validate_query_security(request.question)
    
    logger.info(FUNC_EXECUTE_QUERY, f"Processing query: '{request.question[:50]}...'")

    try:
        # 2. Hybrid Retrieval & Reranking
        evidence_list: list[RerankedCandidateInterface] = execute_hybrid_retrieval(
            query_text=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )

        # 3. Grounded LLM Generation
        gen_result: GenerationResult = generate_grounded_answer(
            question=request.question,
            evidence_list=evidence_list,
        )

        # 4. Citation Extraction & Validation
        raw_citations: list[CitationInterface] = build_citations(evidence_list)
        validated_citations: list[CitationInterface] = validate_citations(
            citations=raw_citations,
            evidence_list=evidence_list,
        )

        end_time: float = time.perf_counter()
        latency_ms: float = round((end_time - start_time) * 1000.0, 2)

        logger.info(FUNC_EXECUTE_QUERY, f"Completed query in {latency_ms}ms with {len(validated_citations)} citations.")

        return QueryResponseInterface(
            question=request.question,
            answer=gen_result.answer,
            citations=validated_citations,
            confidence_score=gen_result.confidence_score,
            processing_time_ms=latency_ms,
        )

    except Exception as exc:
        logger.error(FUNC_EXECUTE_QUERY, "Failure during query workflow execution", exc=exc)
        raise exc
