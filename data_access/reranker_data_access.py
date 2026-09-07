"""Neural Cross-Encoder Reranker Data Access Layer.

Handles neural cross-encoder model initialization, scoring, and candidate reranking
using BAAI/bge-reranker-base with graceful heuristic fallback.
"""

from typing import Optional, Any
from interfaces.retrieval_interface import (
    FusedCandidateInterface,
    RerankedCandidateInterface,
    DEFAULT_TOP_K,
)
from utils.logger import logger

FUNC_GET_MODEL: str = "get_reranker_model"
FUNC_RERANK: str = "rerank_fused_candidates"

DEFAULT_RERANKER_MODEL: str = "BAAI/bge-reranker-base"
_RERANKER_MODEL_INSTANCE: Optional[Any] = None
_USE_MOCK_RERANKER: bool = False


def set_mock_reranker_mode(enabled: bool) -> None:
    """Toggle mock reranker mode for unit testing or offline environments.

    @param enabled: True to force heuristic mock scoring, False for neural inference.
    """
    global _USE_MOCK_RERANKER
    _USE_MOCK_RERANKER = enabled


def get_reranker_model() -> Optional[Any]:
    """Retrieve or initialize singleton CrossEncoder model instance.

    @returns: CrossEncoder instance or None if loading failed or mock enabled.
    """
    global _RERANKER_MODEL_INSTANCE
    if _RERANKER_MODEL_INSTANCE is not None:
        return _RERANKER_MODEL_INSTANCE

    if _USE_MOCK_RERANKER:
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info(FUNC_GET_MODEL, f"Loading neural reranker model: '{DEFAULT_RERANKER_MODEL}'")
        _RERANKER_MODEL_INSTANCE = CrossEncoder(DEFAULT_RERANKER_MODEL)
        logger.info(FUNC_GET_MODEL, f"Successfully loaded neural reranker: '{DEFAULT_RERANKER_MODEL}'")
        return _RERANKER_MODEL_INSTANCE
    except Exception as exc:
        logger.warning(FUNC_GET_MODEL, f"Failed to load neural reranker '{DEFAULT_RERANKER_MODEL}': {exc}")
        return None


def rerank_fused_candidates(
    fused_candidates: list[FusedCandidateInterface],
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RerankedCandidateInterface]:
    """Re-score fused candidates using CrossEncoder neural model or fallback heuristic.

    @param fused_candidates: Input fused RRF candidates list.
    @param query_text: User question string.
    @param top_k: Count of top candidates to return.
    @returns: Ordered list of RerankedCandidateInterface objects sorted by relevance score.
    """
    if not fused_candidates:
        return []

    model = get_reranker_model()

    if model is not None and not _USE_MOCK_RERANKER:
        try:
            pairs: list[tuple[str, str]] = [
                (query_text, candidate.chunk.text) for candidate in fused_candidates
            ]
            raw_scores = model.predict(pairs)

            reranked: list[RerankedCandidateInterface] = []
            for candidate, raw_score in zip(fused_candidates, raw_scores):
                score_val: float = float(raw_score)
                reranked.append(
                    RerankedCandidateInterface(
                        chunk=candidate.chunk,
                        rerank_score=round(score_val, 4),
                        fused_score=candidate.rrf_score,
                    )
                )

            reranked.sort(key=lambda x: x.rerank_score, reverse=True)
            selected_neural = reranked[:top_k]
            logger.info(FUNC_RERANK, f"Neural reranker scored {len(fused_candidates)} candidates -> top {len(selected_neural)}")
            return selected_neural

        except Exception as exc:
            logger.warning(FUNC_RERANK, f"Neural reranking failed, falling back to heuristic scoring: {exc}")

    # Fallback heuristic scoring (word overlap + multi-source boost)
    query_words: set[str] = {w.lower() for w in query_text.split()}
    fallback_reranked: list[RerankedCandidateInterface] = []

    for candidate in fused_candidates:
        text_words: set[str] = {w.lower() for w in candidate.chunk.text.split()}
        overlap_count: int = len(query_words.intersection(text_words))
        overlap_boost: float = (overlap_count / float(len(query_words) or 1)) * 0.2
        multi_source_boost: float = (len(candidate.sources) - 1) * 0.1
        final_score: float = candidate.rrf_score + overlap_boost + multi_source_boost

        fallback_reranked.append(
            RerankedCandidateInterface(
                chunk=candidate.chunk,
                rerank_score=round(final_score, 4),
                fused_score=candidate.rrf_score,
            )
        )

    fallback_reranked.sort(key=lambda x: x.rerank_score, reverse=True)
    selected_fallback = fallback_reranked[:top_k]
    logger.info(FUNC_RERANK, f"Heuristic fallback reranked {len(fused_candidates)} candidates -> top {len(selected_fallback)}")
    return selected_fallback
