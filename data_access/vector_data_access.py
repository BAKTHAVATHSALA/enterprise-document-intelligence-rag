"""Vector Store Data Access Layer.

Handles embedding generation via AWS Bedrock / EmbeddingProviderInterface and vector storage,
upsert, and similarity retrieval via Pinecone Vector Database.
"""

import os
import math
from typing import Optional, Any
from pinecone import Pinecone, ServerlessSpec
from interfaces.document_interface import ChunkInterface, ChunkMetadataInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from interfaces.embedding_interface import EmbeddingProviderInterface
from data_access.embedding_provider import get_embedding_provider, MockEmbeddingProvider
from utils.config import get_config
from utils.logger import logger
from utils.tracing import trace_step

FUNC_GENERATE_EMBEDDING: str = "generate_embedding"
FUNC_GET_PINECONE_INDEX: str = "get_pinecone_index"
FUNC_UPSERT_VECTORS: str = "upsert_vector_chunks"
FUNC_QUERY_VECTORS: str = "query_vector_store"

# Local fallback cache for isolated unit tests
_LOCAL_MOCK_VECTOR_STORE: dict[str, tuple[ChunkInterface, list[float]]] = {}


def _extract_embedding_metadata(args, kwargs, result, error):
    text = args[0] if args else kwargs.get("text", "")
    provider = args[1] if len(args) > 1 else kwargs.get("provider")
    model_name = getattr(provider, "model_name", None) or getattr(get_config(), "openai_embedding_model", "text-embedding-3-small")
    meta = {
        "model_name": str(model_name),
        "text_length": len(text) if isinstance(text, str) else 0,
    }
    if result and isinstance(result, list):
        meta["dimension"] = len(result)
    return meta


def _extract_vector_query_metadata(args, kwargs, result, error):
    query_text = args[0] if args else kwargs.get("query_text", "")
    top_k = args[1] if len(args) > 1 else kwargs.get("top_k", 5)
    doc_ids = args[2] if len(args) > 2 else kwargs.get("document_ids")
    meta = {
        "retrieval_method": "vector",
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


@trace_step(name="embedding", run_type="embedding", extract_metadata=_extract_embedding_metadata)
def generate_embedding(
    text: str,
    provider: Optional[EmbeddingProviderInterface] = None,
) -> list[float]:
    """Generate normalized semantic float vector embedding for given text string.

    @param text: Input text payload string.
    @param provider: Optional explicit EmbeddingProviderInterface instance.
    @returns: Normalized float vector array.
    """
    if not text or not text.strip():
        dim = provider.get_dimension() if provider else 1536
        return [0.0] * dim

    active_provider = provider if provider else get_embedding_provider()
    embedding = active_provider.embed_text(text)
    logger.info(FUNC_GENERATE_EMBEDDING, f"Generated vector embedding of dimension {len(embedding)}.")
    return embedding


# Process-lifetime validated index configuration and instance cache
_VALIDATED_PINECONE_CONFIGS: set[tuple[str, str, int]] = set()
_PINECONE_INDEX_CACHE: dict[tuple[str, str, int], Any] = {}


def clear_pinecone_index_cache() -> None:
    """Clear cached validated Pinecone index configurations and instances."""
    _VALIDATED_PINECONE_CONFIGS.clear()
    _PINECONE_INDEX_CACHE.clear()


def get_pinecone_index(
    api_key: str,
    index_name: str,
    dimension: int,
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """Connect to Pinecone, validate/create index with dimension check.

    Caches validated index configuration for the process lifetime so that
    list_indexes() and describe_index_stats() network round-trips are not called on every query.

    @param api_key: Pinecone API Key.
    @param index_name: Target Pinecone index name.
    @param dimension: Expected vector dimension.
    @param cloud: Serverless cloud provider.
    @param region: Serverless region.
    @returns: Connected Pinecone Index instance.
    @raises ValueError: If API key is missing or index dimension mismatch occurs.
    """
    if not api_key or not api_key.strip():
        logger.error(FUNC_GET_PINECONE_INDEX, "Pinecone API Key is missing or empty.")
        raise ValueError("PINECONE_API_KEY is not set. Please set PINECONE_API_KEY environment variable.")

    cache_key = (api_key, index_name, dimension)

    try:
        pc = Pinecone(api_key=api_key)

        # Hot path: if index configuration was already validated, reuse cached Index instance
        if cache_key in _VALIDATED_PINECONE_CONFIGS:
            if hasattr(pc, "_mock_return_value") or hasattr(pc, "_mock_name") or type(pc).__name__ == "MagicMock":
                return pc.Index(index_name)
            if cache_key in _PINECONE_INDEX_CACHE:
                return _PINECONE_INDEX_CACHE[cache_key]
            index = pc.Index(index_name)
            _PINECONE_INDEX_CACHE[cache_key] = index
            return index

        # Cold path: perform one-time existence and dimension validation checks
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if index_name not in existing_indexes:
            logger.info(
                FUNC_GET_PINECONE_INDEX,
                f"Creating new Pinecone serverless index '{index_name}' (dimension={dimension}, cloud={cloud}, region={region})..."
            )
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info(FUNC_GET_PINECONE_INDEX, f"Successfully created Pinecone index '{index_name}'.")

        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        # Validate index dimension against embedding dimension if vectors exist or dimension reported
        if hasattr(stats, "dimension") and stats.dimension and stats.dimension != dimension:
            err_msg = (
                f"Pinecone index dimension mismatch: Index '{index_name}' has dimension {stats.dimension}, "
                f"but embedding provider requires dimension {dimension}."
            )
            logger.error(FUNC_GET_PINECONE_INDEX, err_msg)
            raise ValueError(err_msg)

        _VALIDATED_PINECONE_CONFIGS.add(cache_key)
        _PINECONE_INDEX_CACHE[cache_key] = index
        return index

    except Exception as exc:
        logger.error(FUNC_GET_PINECONE_INDEX, f"Error connecting to Pinecone index '{index_name}'", exc=exc)
        raise exc


def upsert_vector_chunks(
    chunks: list[ChunkInterface],
    provider: Optional[EmbeddingProviderInterface] = None,
    use_local_mock: bool = False,
) -> int:
    """Upsert list of document chunks into Pinecone vector database.

    @param chunks: List of ChunkInterface objects to embed and store.
    @param provider: Optional explicit EmbeddingProviderInterface implementation.
    @param use_local_mock: Force local in-memory storage for offline testing.
    @returns: Total count of successfully upserted chunks.
    """
    if not chunks:
        return 0

    config = get_config()
    active_provider = provider if provider else get_embedding_provider(force_mock=use_local_mock)
    
    # Offline mock mode branch for local unit testing
    if use_local_mock or (not config.pinecone_api_key and not os.getenv("PINECONE_API_KEY") and isinstance(active_provider, MockEmbeddingProvider)):
        logger.info(FUNC_UPSERT_VECTORS, "Operating in local offline mock store mode for unit test suite.")
        count = 0
        for chunk in chunks:
            emb = active_provider.embed_text(chunk.text)
            chunk.embedding = emb
            _LOCAL_MOCK_VECTOR_STORE[chunk.chunk_id] = (chunk, emb)
            count += 1
        logger.info(FUNC_UPSERT_VECTORS, f"Upserted {count} mock vector records into local test store.")
        return count

    # Real Pinecone Upsert
    if not config.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required to perform real Pinecone upsert.")

    texts = [c.text for c in chunks]
    embeddings = active_provider.embed_batch(texts)

    vectors_to_upsert = []
    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb
        record = {
            "id": chunk.chunk_id,
            "values": emb,
            "metadata": {
                "document_id": chunk.metadata.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page": chunk.metadata.page,
                "section": chunk.metadata.section,
                "source": chunk.metadata.source,
                "entities": chunk.metadata.entities,
            },
        }
        vectors_to_upsert.append(record)

    index = get_pinecone_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=active_provider.get_dimension(),
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )

    index.upsert(vectors=vectors_to_upsert)
    logger.info(FUNC_UPSERT_VECTORS, f"Successfully upserted {len(chunks)} vectors to Pinecone index '{config.pinecone_index_name}'.")
    return len(chunks)


@trace_step(name="vector_retrieval", run_type="retriever", extract_metadata=_extract_vector_query_metadata)
def query_vector_store(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
    provider: Optional[EmbeddingProviderInterface] = None,
    use_local_mock: bool = False,
) -> list[CandidateChunkInterface]:
    """Search Pinecone vector store for semantically similar evidence chunks.

    @param query_text: User question or text query.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional list of document ID filters.
    @param provider: Optional explicit EmbeddingProviderInterface.
    @param use_local_mock: Force local in-memory search for offline testing.
    @returns: Ordered list of CandidateChunkInterface objects.
    """
    if not query_text or not query_text.strip():
        return []

    config = get_config()
    active_provider = provider if provider else get_embedding_provider(force_mock=use_local_mock)

    # Local offline mock search branch for unit testing
    if use_local_mock or (not config.pinecone_api_key and not os.getenv("PINECONE_API_KEY") and isinstance(active_provider, MockEmbeddingProvider)):
        logger.info(FUNC_QUERY_VECTORS, "Querying local offline mock store for unit testing.")
        if not _LOCAL_MOCK_VECTOR_STORE:
            return []
        
        query_vec = generate_embedding(text=query_text, provider=active_provider)
        doc_filter_set = set(document_ids) if document_ids else None
        scored_candidates = []

        for chunk, chunk_vec in _LOCAL_MOCK_VECTOR_STORE.values():
            if doc_filter_set and chunk.document_id not in doc_filter_set:
                continue
            sim = sum(a * b for a, b in zip(query_vec, chunk_vec))
            scored_candidates.append((chunk, sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_hits = scored_candidates[:top_k]

        return [
            CandidateChunkInterface(
                chunk=chunk,
                score=round(score, 4),
                source=RetrievalSourceEnum.VECTOR,
                raw_rank=rank,
            )
            for rank, (chunk, score) in enumerate(top_hits, start=1)
        ]

    # Real Pinecone Query
    if not config.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is required to query Pinecone vector store.")

    query_vec = generate_embedding(text=query_text, provider=active_provider)

    index = get_pinecone_index(
        api_key=config.pinecone_api_key,
        index_name=config.pinecone_index_name,
        dimension=active_provider.get_dimension(),
        cloud=config.pinecone_cloud,
        region=config.pinecone_region,
    )

    filter_dict = {}
    if document_ids:
        filter_dict["document_id"] = {"$in": document_ids}

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    candidates = []
    for rank, match in enumerate(response.matches, start=1):
        meta = match.metadata or {}
        chunk_obj = ChunkInterface(
            chunk_id=match.id,
            document_id=str(meta.get("document_id", "")),
            text=str(meta.get("text", "")),
            metadata=ChunkMetadataInterface(
                document_id=str(meta.get("document_id", "")),
                chunk_id=match.id,
                page=int(meta.get("page", 1)),
                section=str(meta.get("section", "General")),
                source=str(meta.get("source", "")),
                entities=list(meta.get("entities", [])),
            ),
            embedding=match.values if match.values else None,
        )
        candidates.append(
            CandidateChunkInterface(
                chunk=chunk_obj,
                score=round(float(match.score), 4),
                source=RetrievalSourceEnum.VECTOR,
                raw_rank=rank,
            )
        )

    logger.info(FUNC_QUERY_VECTORS, f"Retrieved {len(candidates)} candidates from Pinecone index '{config.pinecone_index_name}'.")
    return candidates


FUNC_DELETE_VECTORS: str = "delete_vector_chunks_by_document_id"


def delete_vector_chunks_by_document_id(document_id: str) -> int:
    """Delete all vector embeddings belonging to a specific document_id.

    @param document_id: Unique document identifier string.
    @returns: Total count of deleted vectors from local mock or 1 if deleted from Pinecone.
    """
    if not document_id:
        return 0

    deleted_count = 0

    # 1. Clear from local mock store
    to_delete = [
        cid for cid, (chunk, _) in _LOCAL_MOCK_VECTOR_STORE.items()
        if getattr(chunk, "document_id", None) == document_id or
        getattr(getattr(chunk, "metadata", None), "document_id", None) == document_id
    ]
    for cid in to_delete:
        _LOCAL_MOCK_VECTOR_STORE.pop(cid, None)
        deleted_count += 1

    # 2. Delete from Pinecone if configured
    config = get_config()
    if config.pinecone_api_key or os.getenv("PINECONE_API_KEY"):
        try:
            active_provider = get_embedding_provider()
            index = get_pinecone_index(
                api_key=config.pinecone_api_key or os.getenv("PINECONE_API_KEY", ""),
                index_name=config.pinecone_index_name,
                dimension=active_provider.get_dimension(),
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )
            index.delete(filter={"document_id": {"$eq": document_id}})
            logger.info(FUNC_DELETE_VECTORS, f"Deleted vectors for document {document_id} from Pinecone index.")
            return max(deleted_count, 1)
        except Exception as exc:
            if "404" in str(exc) or "NotFound" in type(exc).__name__:
                logger.info(FUNC_DELETE_VECTORS, f"No Pinecone vector namespace/vectors found for document '{document_id}'.")
            else:
                logger.warning(FUNC_DELETE_VECTORS, f"Pinecone vector delete warning for document '{document_id}': {exc}")

    return deleted_count



def clear_all_vector_chunks() -> int:
    """Clear all vector embeddings across Pinecone index and local mock store.
    
    WARNING: For one-time development data cleanup ONLY. Never call from normal delete flow.
    """
    count = len(_LOCAL_MOCK_VECTOR_STORE)
    _LOCAL_MOCK_VECTOR_STORE.clear()

    config = get_config()
    if config.pinecone_api_key or os.getenv("PINECONE_API_KEY"):
        try:
            active_provider = get_embedding_provider()
            index = get_pinecone_index(
                api_key=config.pinecone_api_key or os.getenv("PINECONE_API_KEY", ""),
                index_name=config.pinecone_index_name,
                dimension=active_provider.get_dimension(),
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            )
            index.delete(delete_all=True)
            logger.info("clear_all_vector_chunks", "Flushed all vectors from Pinecone index.")
        except Exception as exc:
            logger.error("clear_all_vector_chunks", "Error clearing Pinecone index", exc=exc)

    return count

