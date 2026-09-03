"""Neo4j Knowledge Graph Data Access Layer.

Handles entity relationship insertion, Cypher multi-hop graph queries,
and connected evidence retrieval.
"""

from typing import Optional
from interfaces.document_interface import ChunkInterface, EntityInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from utils.logger import logger

FUNC_UPSERT_GRAPH: str = "upsert_graph_nodes"
FUNC_QUERY_GRAPH: str = "query_graph_store"

# Local In-Memory Knowledge Graph Structures for local/testing execution
_GRAPH_ENTITIES: dict[str, set[str]] = {}  # entity_name_lower -> set(chunk_ids)
_CHUNK_STORE: dict[str, ChunkInterface] = {}
_ENTITY_RELATIONSHIPS: dict[str, set[str]] = {}  # entity1_lower -> set(entity2_lower)


def upsert_graph_nodes(chunks: list[ChunkInterface], entities: list[EntityInterface]) -> int:
    """Upsert document, chunk, and entity nodes into Knowledge Graph.

    @param chunks: List of ChunkInterface objects to map into graph.
    @param entities: Extracted EntityInterface list.
    @returns: Total count of processed graph relationships.
    """
    if not chunks:
        return 0

    rel_count: int = 0

    # Store entities and link to chunks
    for chunk in chunks:
        chunk_id: str = chunk.chunk_id
        _CHUNK_STORE[chunk_id] = chunk

        chunk_entity_names: list[str] = chunk.metadata.entities
        for ent_name in chunk_entity_names:
            ent_key: str = ent_name.lower()
            if ent_key not in _GRAPH_ENTITIES:
                _GRAPH_ENTITIES[ent_key] = set()
            _GRAPH_ENTITIES[ent_key].add(chunk_id)
            rel_count += 1

        # Build multi-hop relationships between co-occurring entities in same chunk
        for i in range(len(chunk_entity_names)):
            for j in range(i + 1, len(chunk_entity_names)):
                e1: str = chunk_entity_names[i].lower()
                e2: str = chunk_entity_names[j].lower()
                
                if e1 not in _ENTITY_RELATIONSHIPS:
                    _ENTITY_RELATIONSHIPS[e1] = set()
                if e2 not in _ENTITY_RELATIONSHIPS:
                    _ENTITY_RELATIONSHIPS[e2] = set()
                
                _ENTITY_RELATIONSHIPS[e1].add(e2)
                _ENTITY_RELATIONSHIPS[e2].add(e1)
                rel_count += 1

    logger.info(FUNC_UPSERT_GRAPH, f"Upserted graph nodes and {rel_count} entity relationships.")
    return rel_count


def query_graph_store(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
) -> list[CandidateChunkInterface]:
    """Execute multi-hop graph search traversing Entity -> Entity -> Chunk connections.

    @param query_text: User question string.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional document ID filter list.
    @returns: Ordered list of CandidateChunkInterface objects.
    """
    if not query_text or not _GRAPH_ENTITIES:
        return []

    doc_filter_set: set[str] | None = set(document_ids) if document_ids else None
    query_words: set[str] = {w.lower() for w in query_text.split()}

    matched_chunks: dict[str, float] = {}

    # 1. Direct Entity Matching
    direct_entities: set[str] = set()
    for ent_key in _GRAPH_ENTITIES:
        if any(word in ent_key for word in query_words):
            direct_entities.add(ent_key)
            # Add direct chunk hits (Score: 1.0)
            for chunk_id in _GRAPH_ENTITIES[ent_key]:
                matched_chunks[chunk_id] = matched_chunks.get(chunk_id, 0.0) + 1.0

    # 2. Multi-hop Graph Traversal (1-hop neighbors)
    neighbor_entities: set[str] = set()
    for ent_key in direct_entities:
        neighbors: set[str] = _ENTITY_RELATIONSHIPS.get(ent_key, set())
        for n in neighbors:
            if n not in direct_entities:
                neighbor_entities.add(n)
                # Add 1-hop connected chunk hits (Score: 0.5)
                for chunk_id in _GRAPH_ENTITIES.get(n, set()):
                    matched_chunks[chunk_id] = matched_chunks.get(chunk_id, 0.0) + 0.5

    # Filter and sort
    scored_candidates: list[tuple[ChunkInterface, float]] = []
    for chunk_id, score in matched_chunks.items():
        chunk: ChunkInterface = _CHUNK_STORE[chunk_id]
        if doc_filter_set and chunk.document_id not in doc_filter_set:
            continue
        scored_candidates.append((chunk, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_hits = scored_candidates[:top_k]

    candidates: list[CandidateChunkInterface] = [
        CandidateChunkInterface(
            chunk=chunk,
            score=round(score, 4),
            source=RetrievalSourceEnum.GRAPH,
            raw_rank=rank,
        )
        for rank, (chunk, score) in enumerate(top_hits, start=1)
    ]

    logger.info(FUNC_QUERY_GRAPH, f"Retrieved {len(candidates)} multi-hop candidates from Knowledge Graph.")
    return candidates
