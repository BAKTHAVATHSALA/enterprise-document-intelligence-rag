"""BM25 Keyword Search Data Access Layer.

Handles exact-term BM25 inverted indexing and retrieval for names, IDs, and contract numbers.
"""

import math
import re
from typing import Optional
from interfaces.document_interface import ChunkInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from utils.logger import logger

FUNC_INDEX_BM25: str = "index_bm25_chunks"
FUNC_QUERY_BM25: str = "query_bm25_index"

# BM25 Parameters
BM25_K1: float = 1.5
BM25_B: float = 0.75

_BM25_DOC_STORE: dict[str, ChunkInterface] = {}
_BM25_DOC_TOKENS: dict[str, list[str]] = {}
_BM25_DOC_FREQS: dict[str, dict[str, int]] = {}
_BM25_DOC_LENGTHS: dict[str, int] = {}
_BM25_DF: dict[str, int] = {}
_AVG_DOC_LENGTH: float = 0.0


def _tokenize(text: str) -> list[str]:
    """Tokenize and normalize text into lowercase word tokens."""
    if not text:
        return []
    return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_-]+\b", text)]


def index_bm25_chunks(chunks: list[ChunkInterface]) -> int:
    """Index list of document chunks into BM25 inverted index.

    @param chunks: List of ChunkInterface objects to index.
    @returns: Count of indexed chunks.
    """
    global _AVG_DOC_LENGTH
    if not chunks:
        return 0

    indexed_count: int = 0

    for chunk in chunks:
        tokens: list[str] = _tokenize(chunk.text)
        token_freqs: dict[str, int] = {}
        for t in tokens:
            token_freqs[t] = token_freqs.get(t, 0) + 1

        chunk_id: str = chunk.chunk_id
        _BM25_DOC_STORE[chunk_id] = chunk
        _BM25_DOC_TOKENS[chunk_id] = tokens
        _BM25_DOC_FREQS[chunk_id] = token_freqs
        _BM25_DOC_LENGTHS[chunk_id] = len(tokens)

        # Update document frequency
        for term in token_freqs:
            _BM25_DF[term] = _BM25_DF.get(term, 0) + 1

        indexed_count += 1

    total_docs: int = len(_BM25_DOC_STORE)
    if total_docs > 0:
        _AVG_DOC_LENGTH = sum(_BM25_DOC_LENGTHS.values()) / float(total_docs)

    logger.info(FUNC_INDEX_BM25, f"Indexed {indexed_count} chunks in BM25 keyword index.")
    return indexed_count


def query_bm25_index(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
) -> list[CandidateChunkInterface]:
    """Query BM25 index for exact-term evidence matches.

    @param query_text: User question string.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional document ID filter list.
    @returns: Ordered list of CandidateChunkInterface objects.
    """
    if not query_text or not _BM25_DOC_STORE:
        return []

    query_tokens: list[str] = _tokenize(query_text)
    if not query_tokens:
        return []

    total_docs: int = len(_BM25_DOC_STORE)
    doc_filter_set: set[str] | None = set(document_ids) if document_ids else None
    scores: dict[str, float] = {}

    for term in set(query_tokens):
        df: int = _BM25_DF.get(term, 0)
        if df == 0:
            continue

        # Inverse Document Frequency (IDF)
        idf: float = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        for chunk_id, freqs in _BM25_DOC_FREQS.items():
            if term not in freqs:
                continue

            chunk: ChunkInterface = _BM25_DOC_STORE[chunk_id]
            if doc_filter_set and chunk.document_id not in doc_filter_set:
                continue

            tf: int = freqs[term]
            doc_len: int = _BM25_DOC_LENGTHS[chunk_id]
            numerator: float = tf * (BM25_K1 + 1.0)
            denominator: float = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / (_AVG_DOC_LENGTH or 1.0)))
            
            term_score: float = idf * (numerator / denominator)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

    # Sort candidates descending by BM25 score
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    candidates: list[CandidateChunkInterface] = [
        CandidateChunkInterface(
            chunk=_BM25_DOC_STORE[cid],
            score=round(score, 4),
            source=RetrievalSourceEnum.BM25,
            raw_rank=rank,
        )
        for rank, (cid, score) in enumerate(sorted_chunks, start=1)
    ]

    logger.info(FUNC_QUERY_BM25, f"Retrieved {len(candidates)} candidates from BM25 keyword index.")
    return candidates
