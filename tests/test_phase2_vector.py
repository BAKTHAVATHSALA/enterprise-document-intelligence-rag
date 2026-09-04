"""Automated Unit and Integration Test Suite for Phase 2 — Real Embeddings + Pinecone Storage."""

import os
from unittest.mock import MagicMock, patch
import pytest

from interfaces import ChunkInterface, ChunkMetadataInterface, RetrievalSourceEnum
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


def test_1_embedding_provider_mock() -> None:
    """TEST 1: MockEmbeddingProvider generates correct vector dimensions and handles empty input."""
    provider = MockEmbeddingProvider(dimension=1536)
    assert provider.get_dimension() == 1536

    # Test valid text
    vec = provider.embed_text("Enterprise document intelligence RAG system")
    assert len(vec) == 1536
    assert isinstance(vec[0], float)

    # Test empty input
    empty_vec = provider.embed_text("")
    assert len(empty_vec) == 1536
    assert all(v == 0.0 for v in empty_vec)


def test_2_openai_embedding_provider_mocked() -> None:
    """TEST 2: OpenAIEmbeddingProvider invokes openai client correctly."""
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1] * 1536
    mock_response.data = [mock_item]
    mock_openai_client.embeddings.create.return_value = mock_response

    with patch("data_access.embedding_provider.OpenAI", return_value=mock_openai_client):
        provider = OpenAIEmbeddingProvider(api_key="fake-openai-key", model_id="text-embedding-3-small")
        vec = provider.embed_text("Sample query string")

        assert len(vec) == 1536
        assert vec[0] == 0.1
        mock_openai_client.embeddings.create.assert_called_once()


def test_3_pinecone_index_connection_and_creation_mocked() -> None:
    """TEST 3: Pinecone index connection creates index if not present and validates dimension."""
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = []
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value.dimension = 1536
    mock_pc.Index.return_value = mock_index

    with patch("data_access.vector_data_access.Pinecone", return_value=mock_pc):
        idx = get_pinecone_index(
            api_key="fake-test-key",
            index_name="enterprise-rag",
            dimension=1536,
        )
        assert idx == mock_index
        mock_pc.create_index.assert_called_once()


def test_4_pinecone_upsert_and_metadata_payload() -> None:
    """TEST 4: Pinecone vector upsert structures records with complete metadata lineage."""
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="enterprise-rag")]
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value.dimension = 1536
    mock_pc.Index.return_value = mock_index

    chunk = ChunkInterface(
        chunk_id="chunk_test_123_0_1",
        document_id="doc_test_123",
        text="PII redacted contract body chunk",
        metadata=ChunkMetadataInterface(
            document_id="doc_test_123",
            chunk_id="chunk_test_123_0_1",
            page=1,
            section="Payment Terms",
            source="contract_abc.pdf",
            entities=["ABC Corp"],
        ),
    )

    mock_provider = MockEmbeddingProvider(dimension=1536)

    with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-key"}), \
         patch("data_access.vector_data_access.Pinecone", return_value=mock_pc):
        count = upsert_vector_chunks([chunk], provider=mock_provider)

        assert count == 1
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args[1]
        vectors = call_args["vectors"]
        assert len(vectors) == 1
        record = vectors[0]
        assert record["id"] == "chunk_test_123_0_1"
        assert record["metadata"]["document_id"] == "doc_test_123"
        assert record["metadata"]["source"] == "contract_abc.pdf"
        assert record["metadata"]["entities"] == ["ABC Corp"]


def test_5_pinecone_query_and_candidate_conversion() -> None:
    """TEST 5: Pinecone vector query filters by document_ids and converts matches to CandidateChunkInterface."""
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="enterprise-rag")]
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value.dimension = 1536

    mock_match = MagicMock()
    mock_match.id = "chunk_test_123_0_1"
    mock_match.score = 0.95
    mock_match.metadata = {
        "document_id": "doc_test_123",
        "text": "PII redacted contract body chunk",
        "page": 1,
        "section": "Payment Terms",
        "source": "contract_abc.pdf",
        "entities": ["ABC Corp"],
    }
    mock_match.values = [0.1] * 1536

    mock_response = MagicMock()
    mock_response.matches = [mock_match]
    mock_index.query.return_value = mock_response
    mock_pc.Index.return_value = mock_index

    mock_provider = MockEmbeddingProvider(dimension=1536)

    with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-key"}), \
         patch("data_access.vector_data_access.Pinecone", return_value=mock_pc):
        hits = query_vector_store(
            query_text="What are the payment terms?",
            top_k=5,
            document_ids=["doc_test_123"],
            provider=mock_provider,
        )

        assert len(hits) == 1
        cand = hits[0]
        assert cand.source == RetrievalSourceEnum.VECTOR
        assert cand.score == 0.95
        assert cand.chunk.chunk_id == "chunk_test_123_0_1"
        assert cand.chunk.metadata.source == "contract_abc.pdf"
        mock_index.query.assert_called_once()
        filter_arg = mock_index.query.call_args[1]["filter"]
        assert filter_arg == {"document_id": {"$in": ["doc_test_123"]}}


def test_6_failure_when_pinecone_key_missing_in_real_mode() -> None:
    """TEST 6: Upsert and query raise clear ValueError if PINECONE_API_KEY is missing in real mode."""
    mock_provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
    mock_provider.get_dimension = lambda: 1536
    mock_provider.embed_batch = lambda texts: [[0.1] * 1536]
    mock_provider.embed_text = lambda text: [0.1] * 1536

    chunk = ChunkInterface(
        chunk_id="chunk_1",
        document_id="doc_1",
        text="Sample text",
        metadata=ChunkMetadataInterface(
            document_id="doc_1",
            chunk_id="chunk_1",
            page=1,
            section="General",
            source="sample.pdf",
            entities=[],
        ),
    )

    with patch.dict(os.environ, {"PINECONE_API_KEY": ""}, clear=True):
        with pytest.raises(ValueError) as exc_upsert:
            upsert_vector_chunks([chunk], provider=mock_provider, use_local_mock=False)
        assert "PINECONE_API_KEY" in str(exc_upsert.value)

        with pytest.raises(ValueError) as exc_query:
            query_vector_store("Sample question", provider=mock_provider, use_local_mock=False)
        assert "PINECONE_API_KEY" in str(exc_query.value)
