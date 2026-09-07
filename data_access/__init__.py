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
    clear_pinecone_index_cache,
    upsert_vector_chunks,
    query_vector_store,
)
from data_access.bm25_data_access import (
    index_bm25_chunks,
    query_bm25_index,
    init_bm25_from_db,
    evict_document_from_bm25,
)
from data_access.graph_data_access import (
    upsert_graph_nodes,
    query_graph_store,
)
from data_access.db_data_access import (
    ensure_db_schema,
    save_document_to_db,
    update_document_status_in_db,
    save_chunks_to_db,
    get_document_metadata_from_db,
    get_document_status_from_db,
    load_all_chunks_from_db,
    delete_document_from_db,
    save_ingestion_job_to_db,
    update_job_progress_in_db,
    get_ingestion_job_from_db,
    get_latest_job_for_document_from_db,
    clear_local_job_store,
)
from data_access.reranker_data_access import (
    get_reranker_model,
    rerank_fused_candidates,
    set_mock_reranker_mode,
)
from data_access.llm_data_access import (
    generate_llm_completion,
    set_mock_llm_mode,
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
    "init_bm25_from_db",
    "evict_document_from_bm25",
    "upsert_graph_nodes",
    "query_graph_store",
    "ensure_db_schema",
    "save_document_to_db",
    "update_document_status_in_db",
    "save_chunks_to_db",
    "get_document_metadata_from_db",
    "get_document_status_from_db",
    "load_all_chunks_from_db",
    "delete_document_from_db",
    "get_reranker_model",
    "rerank_fused_candidates",
    "set_mock_reranker_mode",
    "generate_llm_completion",
    "set_mock_llm_mode",
]

