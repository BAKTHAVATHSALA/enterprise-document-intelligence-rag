"""Vector Store Data Access Layer.

Handles embedding generation via AWS Bedrock / EmbeddingProviderInterface and vector storage,
upsert, and similarity retrieval via Pinecone Vector Database.
"""

import os
import math
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from interfaces.document_interface import ChunkInterface, ChunkMetadataInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from interfaces.embedding_interface import EmbeddingProviderInterface
from data_access.embedding_provider import get_embedding_provider, MockEmbeddingProvider
from utils.config import get_config
from utils.logger import logger

FUNC_GENERATE_EMBEDDING: str = "generate_embedding"
FUNC_GET_PINECONE_INDEX: str = "get_pinecone_index"
FUNC_UPSERT_VECTORS: str = "upsert_vector_chunks"
FUNC_QUERY_VECTORS: str = "query_vector_store"

# Local fallback cache for isolated unit tests
_LOCAL_MOCK_VECTOR_STORE: dict[str, tuple[ChunkInterface, list[float]]] = {}


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


def get_pinecone_index(
    api_key: str,
    index_name: str,
    dimension: int,
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """Connect to Pinecone, validate/create index with dimension check.

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

    try:
        pc = Pinecone(api_key=api_key)
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
        
        query_vec = active_provider.embed_text(query_text)
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

    query_vec = active_provider.embed_text(query_text)

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
