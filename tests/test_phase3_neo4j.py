"""Unit tests for Phase 3 Neo4j Knowledge Graph persistence layer with Document-Scoped Entity Identity."""

import pytest
from interfaces import ChunkInterface, ChunkMetadataInterface, RetrievalSourceEnum
from data_access.graph_data_access import (
    upsert_graph_nodes,
    query_graph_store,
    _GRAPH_ENTITIES,
    _CHUNK_STORE,
    _ENTITY_RELATIONSHIPS,
    _ENTITY_DOC_MAP,
)


@pytest.fixture
def doc_a_chunks():
    return [
        ChunkInterface(
            chunk_id="chunk_docA_0_1",
            document_id="doc_A",
            text="Apple released a new technical policy for Contract #100.",
            metadata=ChunkMetadataInterface(
                document_id="doc_A",
                chunk_id="chunk_docA_0_1",
                page=1,
                section="Overview",
                source="docA.pdf",
                entities=["Apple", "Contract #100"],
            ),
        )
    ]


@pytest.fixture
def doc_b_chunks():
    return [
        ChunkInterface(
            chunk_id="chunk_docB_0_1",
            document_id="doc_B",
            text="Apple signed Agreement #200 with Director Jane Doe.",
            metadata=ChunkMetadataInterface(
                document_id="doc_B",
                chunk_id="chunk_docB_0_1",
                page=1,
                section="Overview",
                source="docB.pdf",
                entities=["Apple", "Agreement #200", "Director Jane Doe"],
            ),
        )
    ]


def test_document_scoped_entity_isolation(doc_a_chunks, doc_b_chunks):
    """Verify Document A + 'Apple' produces Entity A and Document B + 'Apple' produces Entity B (Entity A != Entity B)."""
    _GRAPH_ENTITIES.clear()
    _CHUNK_STORE.clear()
    _ENTITY_RELATIONSHIPS.clear()
    _ENTITY_DOC_MAP.clear()

    upsert_graph_nodes(doc_a_chunks, use_local_mock=True)
    upsert_graph_nodes(doc_b_chunks, use_local_mock=True)

    key_a = "doc_A_apple"
    key_b = "doc_B_apple"

    assert key_a in _GRAPH_ENTITIES, "Entity key for Document A must be document-scoped: doc_A_apple"
    assert key_b in _GRAPH_ENTITIES, "Entity key for Document B must be document-scoped: doc_B_apple"

    assert key_a != key_b, "Entity A and Entity B must be separate distinct nodes"
    assert _ENTITY_DOC_MAP[key_a] == "doc_A"
    assert _ENTITY_DOC_MAP[key_b] == "doc_B"


def test_same_document_same_entity_deduplication(doc_a_chunks):
    """Verify same document + same entity maps to one single Entity node."""
    _GRAPH_ENTITIES.clear()
    _CHUNK_STORE.clear()
    _ENTITY_RELATIONSHIPS.clear()

    # Add a second chunk in doc_A also referencing "Apple"
    second_chunk = ChunkInterface(
        chunk_id="chunk_docA_1_2",
        document_id="doc_A",
        text="Apple specifies security guidelines in Section 2.",
        metadata=ChunkMetadataInterface(
            document_id="doc_A",
            chunk_id="chunk_docA_1_2",
            page=2,
            section="Security",
            source="docA.pdf",
            entities=["Apple"],
        ),
    )

    combined = doc_a_chunks + [second_chunk]
    upsert_graph_nodes(combined, use_local_mock=True)

    key_a = "doc_A_apple"
    assert key_a in _GRAPH_ENTITIES
    # Both chunks should link to the single entity node for doc_A
    assert len(_GRAPH_ENTITIES[key_a]) == 2
    assert "chunk_docA_0_1" in _GRAPH_ENTITIES[key_a]
    assert "chunk_docA_1_2" in _GRAPH_ENTITIES[key_a]


def test_idempotent_reingestion(doc_a_chunks):
    """Verify re-ingesting identical chunks creates no duplicate entity nodes or relationships."""
    _GRAPH_ENTITIES.clear()
    _CHUNK_STORE.clear()

    count1 = upsert_graph_nodes(doc_a_chunks, use_local_mock=True)
    count2 = upsert_graph_nodes(doc_a_chunks, use_local_mock=True)

    assert count1 == count2
    assert len(_GRAPH_ENTITIES) == 2  # "doc_A_apple" and "doc_A_contract #100"


def test_co_occurrence_document_scoping(doc_a_chunks, doc_b_chunks):
    """Verify co-occurrence relationships remain document/chunk scoped."""
    _GRAPH_ENTITIES.clear()
    _CHUNK_STORE.clear()
    _ENTITY_RELATIONSHIPS.clear()

    upsert_graph_nodes(doc_a_chunks, use_local_mock=True)
    upsert_graph_nodes(doc_b_chunks, use_local_mock=True)

    # doc_A has (doc_A_apple <-> doc_A_contract #100)
    # doc_B has (doc_B_apple <-> doc_B_agreement #200)
    key_a_apple = "doc_A_apple"
    key_b_apple = "doc_B_apple"

    assert "doc_A_contract #100" in _ENTITY_RELATIONSHIPS[key_a_apple]
    assert "doc_B_agreement #200" not in _ENTITY_RELATIONSHIPS[key_a_apple], "doc_A entity must not link to doc_B entity"

    assert "doc_B_agreement #200" in _ENTITY_RELATIONSHIPS[key_b_apple]
    assert "doc_A_contract #100" not in _ENTITY_RELATIONSHIPS[key_b_apple], "doc_B entity must not link to doc_A entity"


def test_document_id_graph_filtering(doc_a_chunks, doc_b_chunks):
    """Verify document_id graph filtering strictly isolates results."""
    _GRAPH_ENTITIES.clear()
    _CHUNK_STORE.clear()
    _ENTITY_RELATIONSHIPS.clear()

    upsert_graph_nodes(doc_a_chunks, use_local_mock=True)
    upsert_graph_nodes(doc_b_chunks, use_local_mock=True)

    # Query for "Apple" filtered to doc_A
    results_a = query_graph_store("Apple", top_k=5, document_ids=["doc_A"], use_local_mock=True)
    assert len(results_a) == 1
    assert results_a[0].chunk.document_id == "doc_A"

    # Query for "Apple" filtered to doc_B
    results_b = query_graph_store("Apple", top_k=5, document_ids=["doc_B"], use_local_mock=True)
    assert len(results_b) == 1
    assert results_b[0].chunk.document_id == "doc_B"
