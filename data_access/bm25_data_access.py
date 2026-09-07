"""BM25 Keyword Search Data Access Layer.

Handles exact-term BM25 inverted indexing and retrieval for names, IDs, and contract numbers.
"""

import math
import re
from typing import Optional
from interfaces.document_interface import ChunkInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from utils.logger import logger
from utils.tracing import trace_step

FUNC_INIT_BM25: str = "init_bm25_from_db"
FUNC_EVICT_BM25: str = "evict_document_from_bm25"
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


def init_bm25_from_db(tenant_id: Optional[str] = None) -> int:
    """Hydrate in-memory BM25 index from PostgreSQL authoritative chunks table at startup.

    @param tenant_id: Optional tenant identifier filter.
    @returns: Total count of hydrated and indexed chunks.
    """
    from data_access.db_data_access import load_all_chunks_from_db

    _BM25_DOC_STORE.clear()
    _BM25_DOC_TOKENS.clear()
    _BM25_DOC_FREQS.clear()
    _BM25_DOC_LENGTHS.clear()
    _BM25_DF.clear()
    global _AVG_DOC_LENGTH
    _AVG_DOC_LENGTH = 0.0

    chunks: list[ChunkInterface] = load_all_chunks_from_db(tenant_id=tenant_id)
    if not chunks:
        logger.info(FUNC_INIT_BM25, "No chunks found in PostgreSQL database for BM25 hydration.")
        return 0

    count = index_bm25_chunks(chunks)
    logger.info(FUNC_INIT_BM25, f"Successfully hydrated BM25 index with {count} chunks from PostgreSQL.")
    return count


def evict_document_from_bm25(document_id: str) -> int:
    """Evict all chunks belonging to a document from in-memory BM25 index and update statistics.

    @param document_id: Unique document identifier string.
    @returns: Total count of evicted chunks.
    """
    if not document_id or not _BM25_DOC_STORE:
        return 0

    global _AVG_DOC_LENGTH
    evicted_ids = [cid for cid, chunk in _BM25_DOC_STORE.items() if chunk.document_id == document_id]
    if not evicted_ids:
        return 0

    for cid in evicted_ids:
        freqs = _BM25_DOC_FREQS.get(cid, {})
        for term in freqs:
            if term in _BM25_DF:
                _BM25_DF[term] = max(0, _BM25_DF[term] - 1)
                if _BM25_DF[term] == 0:
                    del _BM25_DF[term]

        _BM25_DOC_STORE.pop(cid, None)
        _BM25_DOC_TOKENS.pop(cid, None)
        _BM25_DOC_FREQS.pop(cid, None)
        _BM25_DOC_LENGTHS.pop(cid, None)

    total_docs = len(_BM25_DOC_STORE)
    _AVG_DOC_LENGTH = (sum(_BM25_DOC_LENGTHS.values()) / float(total_docs)) if total_docs > 0 else 0.0

    logger.info(FUNC_EVICT_BM25, f"Evicted {len(evicted_ids)} chunks for document {document_id} from BM25 index.")
    return len(evicted_ids)


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
        # If chunk already indexed, deduct prior document frequencies to ensure idempotency
        if chunk_id in _BM25_DOC_STORE:
            old_freqs = _BM25_DOC_FREQS.get(chunk_id, {})
            for term in old_freqs:
                if term in _BM25_DF:
                    _BM25_DF[term] = max(0, _BM25_DF[term] - 1)
                    if _BM25_DF[term] == 0:
                        del _BM25_DF[term]

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


def _extract_bm25_metadata(args, kwargs, result, error):
    query_text = args[0] if args else kwargs.get("query_text", "")
    top_k = args[1] if len(args) > 1 else kwargs.get("top_k", 5)
    doc_ids = args[2] if len(args) > 2 else kwargs.get("document_ids")
    meta = {
        "retrieval_method": "bm25",
        "top_k": top_k,
        "query_length": len(query_text) if isinstance(query_text, str) else 0,
        "document_id_filters": list(doc_ids) if doc_ids else None,
    }
    if result is not None:
        meta["hits_count"] = len(result)
        meta["document_ids"] = [
            d for d in {
                getattr(c.chunk, "document_id", getattr(getattr(c.chunk, "metadata", None), "document_id", None))
                for c in result if hasattr(c, "chunk")
            } if d
        ]
    return meta


@trace_step(name="bm25_retrieval", run_type="retriever", extract_metadata=_extract_bm25_metadata)
def query_bm25_index(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
    tenant_id: Optional[str] = None,
) -> list[CandidateChunkInterface]:
    """Query BM25 index for exact-term evidence matches with tenant isolation.

    @param query_text: User question string.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional document ID filter list.
    @param tenant_id: Optional tenant ID filter for multi-tenant isolation.
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

            if tenant_id and chunk.tenant_id != tenant_id:
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

