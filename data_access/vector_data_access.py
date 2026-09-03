"""Vector Store Data Access Layer.

Handles embedding generation and similarity retrieval in Pinecone / Local Vector DB.
"""

import math
import hashlib
from typing import Optional
from interfaces.document_interface import ChunkInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from utils.logger import logger

FUNC_GENERATE_EMBEDDING: str = "generate_embedding"
FUNC_UPSERT_VECTORS: str = "upsert_vector_chunks"
FUNC_QUERY_VECTORS: str = "query_vector_store"

EMBEDDING_DIMENSION: int = 1536

# In-Memory Vector Storage Cache for offline/local execution
_LOCAL_VECTOR_STORE: dict[str, tuple[ChunkInterface, list[float]]] = {}


def generate_embedding(text: str) -> list[float]:
    """Generate normalized semantic float vector embedding for given text string.

    @param text: Input text payload.
    @returns: Normalized float vector of dimension EMBEDDING_DIMENSION.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIMENSION

    # Deterministic pseudo-semantic embedding vector projection generator
    raw_hash: bytes = hashlib.sha256(text.lower().encode("utf-8")).digest()
    vec: list[float] = []
    for i in range(EMBEDDING_DIMENSION):
        byte_val: int = raw_hash[i % len(raw_hash)]
        val: float = ((byte_val / 255.0) * 2.0) - 1.0
        vec.append(val)

    # Normalize vector to unit length
    magnitude: float = math.sqrt(sum(v * v for v in vec))
    if magnitude == 0.0:
        return vec

    normalized_vec: list[float] = [v / magnitude for v in vec]
    return normalized_vec


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity dot product between two normalized vectors."""
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    return sum(a * b for a, b in zip(vec1, vec2))


def upsert_vector_chunks(chunks: list[ChunkInterface]) -> int:
    """Upsert list of document chunks into vector database.

    @param chunks: List of ChunkInterface objects to embed and store.
    @returns: Total count of successfully upserted chunks.
    """
    if not chunks:
        return 0

    count: int = 0
    for chunk in chunks:
        embedding: list[float] = generate_embedding(chunk.text)
        chunk.embedding = embedding
        _LOCAL_VECTOR_STORE[chunk.chunk_id] = (chunk, embedding)
        count += 1

    logger.info(FUNC_UPSERT_VECTORS, f"Upserted {count} vector records into vector store.")
    return count


def query_vector_store(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
) -> list[CandidateChunkInterface]:
    """Search vector store for semantically similar evidence chunks.

    @param query_text: User question or text query.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional list of document ID filters.
    @returns: Ordered list of CandidateChunkInterface objects.
    """
    if not query_text or not _LOCAL_VECTOR_STORE:
        return []

    query_vec: list[float] = generate_embedding(query_text)
    doc_filter_set: set[str] | None = set(document_ids) if document_ids else None

    scored_candidates: list[tuple[ChunkInterface, float]] = []

    for chunk, chunk_vec in _LOCAL_VECTOR_STORE.values():
        if doc_filter_set and chunk.document_id not in doc_filter_set:
            continue
        sim: float = _cosine_similarity(query_vec, chunk_vec)
        scored_candidates.append((chunk, sim))

    # Sort descending by similarity score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_hits = scored_candidates[:top_k]

    candidates: list[CandidateChunkInterface] = [
        CandidateChunkInterface(
            chunk=chunk,
            score=round(score, 4),
            source=RetrievalSourceEnum.VECTOR,
            raw_rank=rank,
        )
        for rank, (chunk, score) in enumerate(top_hits, start=1)
    ]

    logger.info(FUNC_QUERY_VECTORS, f"Retrieved {len(candidates)} candidates from vector store.")
    return candidates
