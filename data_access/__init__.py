"""Data Access Layer Package.

Exports data access functions for Pinecone vector storage, OpenAI embedding provider,
BM25 keyword search, and Neo4j graph storage.
"""

from data_access.embedding_provider import (
    OpenAIEmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)
from data_access.vector_data_access import (
    generate_embedding,
    get_pinecone_index,
    upsert_vector_chunks,
    query_vector_store,
)
from data_access.bm25_data_access import (
    index_bm25_chunks,
    query_bm25_index,
)
from data_access.graph_data_access import (
    upsert_graph_nodes,
    query_graph_store,
)

__all__ = [
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "get_embedding_provider",
    "generate_embedding",
    "get_pinecone_index",
    "upsert_vector_chunks",
    "query_vector_store",
    "index_bm25_chunks",
    "query_bm25_index",
    "upsert_graph_nodes",
    "query_graph_store",
]
