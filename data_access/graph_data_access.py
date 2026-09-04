"""Neo4j Knowledge Graph Data Access Layer.

Handles entity relationship insertion, Cypher multi-hop graph queries,
and connected evidence retrieval with support for live Neo4j database and local fallback mode.
Document-scoped entity identity strategy: entity_key = document_id + "_" + normalized_entity_name.
"""

import os
from typing import Optional
from interfaces.document_interface import ChunkInterface, ChunkMetadataInterface, EntityInterface
from interfaces.retrieval_interface import CandidateChunkInterface, RetrievalSourceEnum
from utils.config import get_config
from utils.logger import logger

FUNC_GET_DRIVER: str = "get_neo4j_driver"
FUNC_CLOSE_DRIVER: str = "close_neo4j_driver"
FUNC_UPSERT_GRAPH: str = "upsert_graph_nodes"
FUNC_QUERY_GRAPH: str = "query_graph_store"

# Global Driver Singleton
_NEO4J_DRIVER = None

# Local In-Memory Knowledge Graph Structures for local/testing execution
_GRAPH_ENTITIES: dict[str, set[str]] = {}  # entity_key -> set(chunk_ids)
_CHUNK_STORE: dict[str, ChunkInterface] = {}
_ENTITY_RELATIONSHIPS: dict[str, set[str]] = {}  # entity1_key -> set(entity2_key)
_ENTITY_NAMES_MAP: dict[str, str] = {}  # entity_key -> norm_name
_ENTITY_DOC_MAP: dict[str, str] = {}    # entity_key -> doc_id


def get_neo4j_driver(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
):
    """Get or instantiate singleton Neo4j database driver connection.

    @param uri: Optional explicit Neo4j URI override.
    @param user: Optional explicit Neo4j user override.
    @param password: Optional explicit Neo4j password override.
    @returns: Connected Neo4j Driver instance or None if unconfigured.
    """
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER

    config = get_config()
    target_uri = uri if uri else (config.neo4j_uri or os.getenv("NEO4J_URI", ""))
    target_user = user if user else (config.neo4j_user or os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j")))
    target_password = password if password else (config.neo4j_password or os.getenv("NEO4J_PASSWORD", ""))

    if not target_uri or not target_user or "localhost" in target_uri:
        if "localhost" in target_uri and not os.getenv("NEO4J_TEST_LOCAL"):
            return None

    try:
        import neo4j
        _NEO4J_DRIVER = neo4j.GraphDatabase.driver(target_uri, auth=(target_user, target_password))
        logger.info(FUNC_GET_DRIVER, f"Initialized Neo4j driver for URI '{target_uri}'.")
        return _NEO4J_DRIVER
    except Exception as exc:
        logger.warning(FUNC_GET_DRIVER, f"Could not initialize Neo4j driver: {exc}")
        return None


def close_neo4j_driver() -> None:
    """Close Neo4j driver connection cleanly."""
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        try:
            _NEO4J_DRIVER.close()
            logger.info(FUNC_CLOSE_DRIVER, "Closed Neo4j driver connection.")
        except Exception as exc:
            logger.warning(FUNC_CLOSE_DRIVER, f"Error closing Neo4j driver: {exc}")
        finally:
            _NEO4J_DRIVER = None


def ensure_schema_constraints(driver) -> None:
    """Create uniqueness constraints for Document, Chunk, and document-scoped Entity keys."""
    drop_stmts = [
        "DROP CONSTRAINT entity_name_unique IF EXISTS",
    ]
    constraints = [
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
    ]
    try:
        with driver.session() as session:
            for stmt in drop_stmts:
                try:
                    session.run(stmt)
                except Exception:
                    pass
            for stmt in constraints:
                session.run(stmt)
    except Exception as exc:
        logger.warning(FUNC_UPSERT_GRAPH, f"Schema constraint initialization warning: {exc}")



def upsert_graph_nodes(
    chunks: list[ChunkInterface],
    entities: Optional[list[EntityInterface]] = None,
    use_local_mock: bool = False,
) -> int:
    """Upsert document, chunk, entity nodes and relationships into Neo4j graph database using document-scoped entity identity.

    @param chunks: List of ChunkInterface objects to map into graph.
    @param entities: Optional extracted EntityInterface list.
    @param use_local_mock: Force local in-memory graph for offline unit testing.
    @returns: Total count of processed graph relationships.
    """
    if not chunks:
        return 0

    driver = None if use_local_mock else get_neo4j_driver()

    # Local Offline Mock Storage Branch (Document-scoped Entity Keys)
    if use_local_mock or driver is None:
        rel_count: int = 0
        for chunk in chunks:
            chunk_id: str = chunk.chunk_id
            doc_id: str = chunk.document_id
            _CHUNK_STORE[chunk_id] = chunk

            chunk_entity_names: list[str] = chunk.metadata.entities
            for ent_name in chunk_entity_names:
                norm_name = ent_name.lower().strip()
                if not norm_name:
                    continue
                ent_key: str = f"{doc_id}_{norm_name}"
                
                _ENTITY_NAMES_MAP[ent_key] = norm_name
                _ENTITY_DOC_MAP[ent_key] = doc_id
                if ent_key not in _GRAPH_ENTITIES:
                    _GRAPH_ENTITIES[ent_key] = set()
                _GRAPH_ENTITIES[ent_key].add(chunk_id)
                rel_count += 1

            for i in range(len(chunk_entity_names)):
                for j in range(i + 1, len(chunk_entity_names)):
                    e1_norm: str = chunk_entity_names[i].lower().strip()
                    e2_norm: str = chunk_entity_names[j].lower().strip()
                    if not e1_norm or not e2_norm or e1_norm == e2_norm:
                        continue
                    k1: str = f"{doc_id}_{e1_norm}"
                    k2: str = f"{doc_id}_{e2_norm}"

                    if k1 not in _ENTITY_RELATIONSHIPS:
                        _ENTITY_RELATIONSHIPS[k1] = set()
                    if k2 not in _ENTITY_RELATIONSHIPS:
                        _ENTITY_RELATIONSHIPS[k2] = set()
                    _ENTITY_RELATIONSHIPS[k1].add(k2)
                    _ENTITY_RELATIONSHIPS[k2].add(k1)
                    rel_count += 1

        logger.info(FUNC_UPSERT_GRAPH, f"Upserted {rel_count} graph relationships into local mock store (document-scoped entity identity).")
        return rel_count

    # Real Neo4j Parameterized Cypher Ingestion with Document-Scoped Entity Keys
    ensure_schema_constraints(driver)

    chunks_payload = []
    mentions_payload = []
    co_occurrences_payload = []
    seen_co_occurrences = set()

    for chunk in chunks:
        chunks_payload.append({
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "page": chunk.metadata.page,
            "section": chunk.metadata.section,
            "source": chunk.metadata.source,
            "text": chunk.text,
        })

        chunk_entities = chunk.metadata.entities
        for ent_name in chunk_entities:
            clean_name = ent_name.strip()
            if not clean_name:
                continue
            norm_name = clean_name.lower()
            ent_key = f"{chunk.document_id}_{norm_name}"

            mentions_payload.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "entity_key": ent_key,
                "name": norm_name,
                "display_name": clean_name,
            })

        for i in range(len(chunk_entities)):
            for j in range(i + 1, len(chunk_entities)):
                e1_raw = chunk_entities[i].strip()
                e2_raw = chunk_entities[j].strip()
                if not e1_raw or not e2_raw or e1_raw.lower() == e2_raw.lower():
                    continue

                e1_norm, e2_norm = sorted([e1_raw.lower(), e2_raw.lower()])
                e1_key = f"{chunk.document_id}_{e1_norm}"
                e2_key = f"{chunk.document_id}_{e2_norm}"

                pair_key = (chunk.chunk_id, e1_key, e2_key)
                if pair_key in seen_co_occurrences:
                    continue
                seen_co_occurrences.add(pair_key)

                co_occurrences_payload.append({
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "e1_key": e1_key,
                    "e2_key": e2_key,
                    "e1_name": e1_norm,
                    "e2_name": e2_norm,
                })

    cypher_chunks = """
    UNWIND $chunks AS c
    MERGE (d:Document {document_id: c.document_id})
    MERGE (chk:Chunk {chunk_id: c.chunk_id})
    SET chk.document_id = c.document_id,
        chk.page = c.page,
        chk.section = c.section,
        chk.source = c.source,
        chk.text = c.text
    MERGE (d)-[:CONTAINS_CHUNK]->(chk)
    """

    cypher_mentions = """
    UNWIND $mentions AS m
    MERGE (e:Entity {entity_key: m.entity_key})
    ON CREATE SET e.document_id = m.document_id,
                  e.name = m.name,
                  e.display_name = m.display_name
    WITH e, m
    MATCH (chk:Chunk {chunk_id: m.chunk_id})
    MERGE (chk)-[:MENTIONS_ENTITY]->(e)
    """

    cypher_co_occurrences = """
    UNWIND $co_occurrences AS co
    MERGE (e1:Entity {entity_key: co.e1_key})
    ON CREATE SET e1.document_id = co.document_id, e1.name = co.e1_name
    MERGE (e2:Entity {entity_key: co.e2_key})
    ON CREATE SET e2.document_id = co.document_id, e2.name = co.e2_name
    MERGE (e1)-[r:CO_OCCURS_WITH {chunk_id: co.chunk_id, document_id: co.document_id}]->(e2)
    """

    total_rels = len(chunks_payload) + len(mentions_payload) + len(co_occurrences_payload)

    with driver.session() as session:
        if chunks_payload:
            session.run(cypher_chunks, chunks=chunks_payload)
        if mentions_payload:
            session.run(cypher_mentions, mentions=mentions_payload)
        if co_occurrences_payload:
            session.run(cypher_co_occurrences, co_occurrences=co_occurrences_payload)

    logger.info(FUNC_UPSERT_GRAPH, f"Successfully upserted {total_rels} graph nodes and relationships into Neo4j.")
    return total_rels


def query_graph_store(
    query_text: str,
    top_k: int = 5,
    document_ids: Optional[list[str]] = None,
    use_local_mock: bool = False,
) -> list[CandidateChunkInterface]:
    """Execute Cypher multi-hop graph search traversing Document -> Chunk -> Entity -> CO_OCCURS_WITH connections.

    @param query_text: User question string.
    @param top_k: Maximum candidate hits to return.
    @param document_ids: Optional document ID filter list.
    @param use_local_mock: Force local in-memory graph search for unit testing.
    @returns: Ordered list of CandidateChunkInterface objects.
    """
    if not query_text or not query_text.strip():
        return []

    driver = None if use_local_mock else get_neo4j_driver()

    # Local Offline Mock Retrieval Branch
    if use_local_mock or driver is None:
        if not _GRAPH_ENTITIES:
            return []

        doc_filter_set: set[str] | None = set(document_ids) if document_ids else None
        query_words: set[str] = {w.lower() for w in query_text.split() if len(w) > 2}

        matched_chunks: dict[str, float] = {}
        direct_entity_keys: set[str] = set()

        for ent_key, chunk_ids in _GRAPH_ENTITIES.items():
            ent_norm_name = _ENTITY_NAMES_MAP.get(ent_key, "")
            ent_doc_id = _ENTITY_DOC_MAP.get(ent_key, "")

            if doc_filter_set and ent_doc_id not in doc_filter_set:
                continue

            if any(word in ent_norm_name for word in query_words):
                direct_entity_keys.add(ent_key)
                for chunk_id in chunk_ids:
                    matched_chunks[chunk_id] = matched_chunks.get(chunk_id, 0.0) + 1.0

        for ent_key in direct_entity_keys:
            neighbors: set[str] = _ENTITY_RELATIONSHIPS.get(ent_key, set())
            for n_key in neighbors:
                if n_key not in direct_entity_keys:
                    n_doc_id = _ENTITY_DOC_MAP.get(n_key, "")
                    if doc_filter_set and n_doc_id not in doc_filter_set:
                        continue
                    for chunk_id in _GRAPH_ENTITIES.get(n_key, set()):
                        matched_chunks[chunk_id] = matched_chunks.get(chunk_id, 0.0) + 0.5

        scored_candidates: list[tuple[ChunkInterface, float]] = []
        for chunk_id, score in matched_chunks.items():
            if chunk_id not in _CHUNK_STORE:
                continue
            chunk: ChunkInterface = _CHUNK_STORE[chunk_id]
            if doc_filter_set and chunk.document_id not in doc_filter_set:
                continue
            scored_candidates.append((chunk, score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_hits = scored_candidates[:top_k]

        return [
            CandidateChunkInterface(
                chunk=chunk,
                score=round(score, 4),
                source=RetrievalSourceEnum.GRAPH,
                raw_rank=rank,
            )
            for rank, (chunk, score) in enumerate(top_hits, start=1)
        ]

    # Real Neo4j Parameterized Cypher Query Execution
    query_tokens = [w.lower().strip() for w in query_text.split() if len(w.strip()) > 2]
    if not query_tokens:
        return []

    cypher_query = """
    MATCH (e:Entity)
    WHERE any(term IN $query_tokens WHERE e.name CONTAINS term OR term CONTAINS e.name)
      AND ($doc_ids IS NULL OR e.document_id IN $doc_ids)
    OPTIONAL MATCH (e)-[:CO_OCCURS_WITH]-(neighbor:Entity)
    WHERE ($doc_ids IS NULL OR neighbor.document_id IN $doc_ids)
    WITH collect(distinct e) + collect(distinct neighbor) AS target_entities
    UNWIND target_entities AS target
    MATCH (chk:Chunk)-[:MENTIONS_ENTITY]->(target)
    WHERE ($doc_ids IS NULL OR chk.document_id IN $doc_ids)
    WITH chk, count(distinct target) AS match_score
    RETURN chk.chunk_id AS chunk_id,
           chk.document_id AS document_id,
           chk.text AS text,
           chk.page AS page,
           chk.section AS section,
           chk.source AS source,
           match_score
    ORDER BY match_score DESC
    LIMIT $top_k
    """

    candidates: list[CandidateChunkInterface] = []

    try:
        with driver.session() as session:
            result = session.run(
                cypher_query,
                query_tokens=query_tokens,
                doc_ids=document_ids if document_ids else None,
                top_k=top_k,
            )
            records = list(result)

            for rank, rec in enumerate(records, start=1):
                chunk_id = rec["chunk_id"]
                document_id = rec["document_id"]
                text = rec["text"]
                page = int(rec["page"]) if rec["page"] is not None else 1
                section = rec["section"] or "General"
                source = rec["source"] or ""
                score = float(rec["match_score"])

                chunk_obj = ChunkInterface(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=text,
                    metadata=ChunkMetadataInterface(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        page=page,
                        section=section,
                        source=source,
                        entities=[],
                    ),
                )

                candidates.append(
                    CandidateChunkInterface(
                        chunk=chunk_obj,
                        score=round(score, 4),
                        source=RetrievalSourceEnum.GRAPH,
                        raw_rank=rank,
                    )
                )

        logger.info(FUNC_QUERY_GRAPH, f"Retrieved {len(candidates)} candidates from Neo4j Cypher query.")
        return candidates

    except Exception as exc:
        logger.error(FUNC_QUERY_GRAPH, f"Error executing Cypher graph retrieval query", exc=exc)
        return []
