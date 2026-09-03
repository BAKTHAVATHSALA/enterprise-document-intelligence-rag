"""Retrieval interfaces and domain models.

Defines schemas for vector search candidates, BM25 keyword hits,
Neo4j graph traversal hits, Reciprocal Rank Fusion (RRF) results, and reranked candidates.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from interfaces.document_interface import ChunkInterface


# Retrieval Constants
DEFAULT_TOP_K: int = 5
RRF_K_CONSTANT: int = 60


class RetrievalSourceEnum(str, Enum):
    """Source retriever category."""
    VECTOR = "VECTOR"
    BM25 = "BM25"
    GRAPH = "GRAPH"


class CandidateChunkInterface(BaseModel):
    """Retrieved candidate chunk from a single retrieval source."""
    chunk: ChunkInterface = Field(..., description="Retrieved chunk payload")
    score: float = Field(..., description="Raw retriever score")
    source: RetrievalSourceEnum = Field(..., description="Retrieval mechanism source")
    raw_rank: int = Field(..., description="1-indexed rank order within source")


class FusedCandidateInterface(BaseModel):
    """Candidate result resulting from Reciprocal Rank Fusion (RRF)."""
    chunk: ChunkInterface = Field(..., description="Retrieved chunk payload")
    rrf_score: float = Field(..., description="Calculated RRF score")
    vector_score: Optional[float] = Field(default=None, description="Vector score if present")
    bm25_score: Optional[float] = Field(default=None, description="BM25 score if present")
    graph_score: Optional[float] = Field(default=None, description="Graph score if present")
    sources: list[RetrievalSourceEnum] = Field(..., description="Sources contributing this candidate")


class RerankedCandidateInterface(BaseModel):
    """Candidate after post-fusion reranking."""
    chunk: ChunkInterface = Field(..., description="Retrieved chunk payload")
    rerank_score: float = Field(..., description="Reranker output confidence score")
    fused_score: float = Field(..., description="Pre-rerank fused score")
