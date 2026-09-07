"""Hybrid Retrieval, Reciprocal Rank Fusion (RRF), and Reranking Helper.

Orchestrates multi-source retrieval (Vector + BM25 + Graph), fuses candidate results
using Reciprocal Rank Fusion (RRF), and re-scores candidates for optimal evidence selection.
"""

from typing import Optional
from interfaces.retrieval_interface import (
    CandidateChunkInterface,
    FusedCandidateInterface,
    RerankedCandidateInterface,
    RetrievalSourceEnum,
    RRF_K_CONSTANT,
    DEFAULT_TOP_K,
)
from data_access import (
    query_vector_store,
    query_bm25_index,
    query_graph_store,
)
from utils.logger import logger
from utils.tracing import trace_step

FUNC_HYBRID_RETRIEVAL: str = "execute_hybrid_retrieval"
FUNC_FUSE_RRF: str = "apply_rrf_fusion"
FUNC_RERANK: str = "rerank_candidates"


def _extract_rrf_metadata(args, kwargs, result, error):
    c_lists = args[0] if args else kwargs.get("candidate_lists", [])
    k_const = args[1] if len(args) > 1 else kwargs.get("k_constant", RRF_K_CONSTANT)
    channels_count = len(c_lists) if isinstance(c_lists, list) else 0
    total_in = sum(len(cl) for cl in c_lists if isinstance(cl, list)) if channels_count else 0
    meta = {
        "k_constant": k_const,
        "channels_count": channels_count,
        "total_input_candidates": total_in,
    }
    if result is not None:
        meta["fused_count"] = len(result)
    return meta


def _extract_rerank_metadata(args, kwargs, result, error):
    fused = args[0] if args else kwargs.get("fused_candidates", [])
    query = args[1] if len(args) > 1 else kwargs.get("query_text", "")
    top_k = args[2] if len(args) > 2 else kwargs.get("top_k", DEFAULT_TOP_K)
    meta = {
        "model_name": "BAAI/bge-reranker-base",
        "top_k": top_k,
        "candidates_in": len(fused) if isinstance(fused, list) else 0,
        "query_length": len(query) if isinstance(query, str) else 0,
    }
    if result is not None:
        meta["candidates_out"] = len(result)
        meta["top_score"] = result[0].relevance_score if result and hasattr(result[0], "relevance_score") else None
    return meta


@trace_step(name="rrf_fusion", run_type="chain", extract_metadata=_extract_rrf_metadata)
def apply_rrf_fusion(
    candidate_lists: list[list[CandidateChunkInterface]],
    k_constant: int = RRF_K_CONSTANT,
) -> list[FusedCandidateInterface]:
    """Fuse candidate rankings from multiple retrieval sources using Reciprocal Rank Fusion.

    @param candidate_lists: List of candidate hit lists from different retrievers.
    @param k_constant: Smoothing constant for RRF calculation (default 60).
    @returns: List of FusedCandidateInterface objects sorted by fused RRF score.
    """
    if not candidate_lists:
        return []

    fused_map: dict[str, dict] = {}

    for cand_list in candidate_lists:
        for cand in cand_list:
            chunk_id: str = cand.chunk.chunk_id
            rank: int = cand.raw_rank
            rrf_delta: float = 1.0 / (k_constant + rank)

            if chunk_id not in fused_map:
                fused_map[chunk_id] = {
                    "chunk": cand.chunk,
                    "rrf_score": 0.0,
                    "vector_score": None,
                    "bm25_score": None,
                    "graph_score": None,
                    "sources": set(),
                }

            entry = fused_map[chunk_id]
            entry["rrf_score"] += rrf_delta
            entry["sources"].add(cand.source)

            if cand.source == RetrievalSourceEnum.VECTOR:
                entry["vector_score"] = cand.score
            elif cand.source == RetrievalSourceEnum.BM25:
                entry["bm25_score"] = cand.score
            elif cand.source == RetrievalSourceEnum.GRAPH:
                entry["graph_score"] = cand.score

    fused_results: list[FusedCandidateInterface] = [
        FusedCandidateInterface(
            chunk=v["chunk"],
            rrf_score=round(v["rrf_score"], 6),
            vector_score=v["vector_score"],
            bm25_score=v["bm25_score"],
            graph_score=v["graph_score"],
            sources=list(v["sources"]),
        )
        for v in fused_map.values()
    ]

    # Sort descending by RRF score
    fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
    return fused_results


@trace_step(name="reranking", run_type="chain", extract_metadata=_extract_rerank_metadata)
def rerank_candidates(
    fused_candidates: list[FusedCandidateInterface],
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RerankedCandidateInterface]:
    """Re-score fused candidate evidence using BAAI/bge-reranker-base CrossEncoder neural model.

    @param fused_candidates: Input fused candidate list.
    @param query_text: User question string.
    @param top_k: Final top_k candidates count.
    @returns: Ordered list of RerankedCandidateInterface objects sorted by relevance score.
    """
    from data_access import rerank_fused_candidates
    return rerank_fused_candidates(fused_candidates=fused_candidates, query_text=query_text, top_k=top_k)


def execute_hybrid_retrieval(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    document_ids: Optional[list[str]] = None,
) -> list[RerankedCandidateInterface]:
    """Orchestrate hybrid multi-retriever search, RRF fusion, and candidate reranking.

    @param query_text: User question string.
    @param top_k: Maximum final evidence candidates count.
    @param document_ids: Optional list of document ID filters.
    @returns: Ordered list of RerankedCandidateInterface items.
    """
    if not query_text or not query_text.strip():
        return []

    logger.info(FUNC_HYBRID_RETRIEVAL, f"Executing hybrid retrieval for query: '{query_text}'")

    # 1. Parallel / Multi-source retrieval step
    vector_hits: list[CandidateChunkInterface] = query_vector_store(
        query_text=query_text, top_k=top_k * 2, document_ids=document_ids
    )
    bm25_hits: list[CandidateChunkInterface] = query_bm25_index(
        query_text=query_text, top_k=top_k * 2, document_ids=document_ids
    )
    graph_hits: list[CandidateChunkInterface] = query_graph_store(
        query_text=query_text, top_k=top_k * 2, document_ids=document_ids
    )

    # 2. Reciprocal Rank Fusion (RRF)
    all_candidate_lists: list[list[CandidateChunkInterface]] = [
        vector_hits,
        bm25_hits,
        graph_hits,
    ]
    fused_candidates: list[FusedCandidateInterface] = apply_rrf_fusion(all_candidate_lists)
    logger.info(FUNC_FUSE_RRF, f"Fused {len(fused_candidates)} unique candidates across 3 retrieval channels.")

    # 3. Reranking Step
    final_evidence: list[RerankedCandidateInterface] = rerank_candidates(
        fused_candidates=fused_candidates,
        query_text=query_text,
        top_k=top_k,
    )

    return final_evidence
