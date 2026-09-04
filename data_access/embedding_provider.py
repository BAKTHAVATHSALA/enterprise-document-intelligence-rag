"""Embedding Provider Implementation Layer.

Implements OpenAI Embedding Provider and Mock Embedding Provider for offline testing,
adhering to the EmbeddingProviderInterface domain abstraction.
"""

import os
import math
import hashlib
from typing import Optional
from openai import OpenAI
from interfaces.embedding_interface import EmbeddingProviderInterface
from utils.config import get_config
from utils.logger import logger

FUNC_OPENAI_EMBED: str = "OpenAIEmbeddingProvider.embed_text"
FUNC_MOCK_EMBED: str = "MockEmbeddingProvider.embed_text"
FUNC_GET_PROVIDER: str = "get_embedding_provider"

OPENAI_SMALL_EMBEDDING_DIMENSION: int = 1536


class OpenAIEmbeddingProvider(EmbeddingProviderInterface):
    """Official OpenAI real embedding vector provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
    ) -> None:
        """Initialize OpenAI API client.

        @param api_key: Optional explicit OpenAI API key override.
        @param model_id: Optional explicit embedding model ID override.
        """
        config = get_config()
        self.api_key: str = api_key if api_key else (config.openai_api_key or os.getenv("OPENAI_API_KEY", ""))
        self.model_id: str = model_id if model_id else (config.openai_embedding_model or "text-embedding-3-small")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set OPENAI_API_KEY environment variable.")

        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(
                FUNC_OPENAI_EMBED,
                f"Initialized OpenAI embedding client with model '{self.model_id}'."
            )
        except Exception as exc:
            logger.error(
                FUNC_OPENAI_EMBED,
                f"Failed to initialize OpenAI client for model '{self.model_id}'",
                exc=exc,
            )
            raise exc

    def embed_text(self, text: str) -> list[float]:
        """Generate normalized float embedding vector via OpenAI API.

        @param text: Input text payload string.
        @returns: Float list representing dense vector.
        """
        if not text or not text.strip():
            return [0.0] * self.get_dimension()

        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_id,
            )
            embedding: list[float] = response.data[0].embedding

            logger.info(
                FUNC_OPENAI_EMBED,
                f"Generated embedding vector of dimension {len(embedding)} via OpenAI model '{self.model_id}'."
            )
            return embedding

        except Exception as exc:
            logger.error(
                FUNC_OPENAI_EMBED,
                f"Error invoking OpenAI embedding model '{self.model_id}'",
                exc=exc,
            )
            raise exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings in batch via OpenAI API for list of texts."""
        if not texts:
            return []

        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty_indices:
            return [[0.0] * self.get_dimension() for _ in texts]

        try:
            valid_texts = [texts[i] for i in non_empty_indices]
            response = self.client.embeddings.create(
                input=valid_texts,
                model=self.model_id,
            )

            result: list[list[float]] = [[0.0] * self.get_dimension() for _ in texts]
            for idx, item in zip(non_empty_indices, response.data):
                result[idx] = item.embedding

            logger.info(
                FUNC_OPENAI_EMBED,
                f"Generated batch embeddings for {len(valid_texts)} items via OpenAI model '{self.model_id}'."
            )
            return result

        except Exception as exc:
            logger.error(
                FUNC_OPENAI_EMBED,
                f"Error invoking batch OpenAI embeddings for model '{self.model_id}'",
                exc=exc,
            )
            raise exc

    def get_dimension(self) -> int:
        """Return default expected embedding dimension for configured OpenAI model."""
        if "large" in self.model_id.lower():
            return 3072
        return OPENAI_SMALL_EMBEDDING_DIMENSION


class MockEmbeddingProvider(EmbeddingProviderInterface):
    """Mock embedding generator for deterministic, isolated unit testing."""

    def __init__(self, dimension: int = OPENAI_SMALL_EMBEDDING_DIMENSION) -> None:
        """Initialize mock embedding provider with given vector dimension."""
        self.dimension: int = dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic pseudo-semantic float vector embedding for unit testing."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        raw_hash: bytes = hashlib.sha256(text.lower().encode("utf-8")).digest()
        vec: list[float] = []
        for i in range(self.dimension):
            byte_val: int = raw_hash[i % len(raw_hash)]
            val: float = ((byte_val / 255.0) * 2.0) - 1.0
            vec.append(val)

        magnitude: float = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0.0:
            return vec

        normalized: list[float] = [v / magnitude for v in vec]
        logger.info(FUNC_MOCK_EMBED, f"Generated mock vector embedding of dimension {len(normalized)}.")
        return normalized

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch list of texts."""
        if not texts:
            return []
        return [self.embed_text(t) for t in texts]

    def get_dimension(self) -> int:
        """Retrieve output dimension of mock vectors."""
        return self.dimension


def get_embedding_provider(
    force_mock: bool = False,
    api_key: str | None = None,
    model_id: str | None = None,
) -> EmbeddingProviderInterface:
    """Factory function to instantiate configured EmbeddingProviderInterface.

    @param force_mock: If True, forces MockEmbeddingProvider for unit tests.
    @param api_key: Optional explicit OpenAI API Key override.
    @param model_id: Optional explicit OpenAI model ID override.
    @returns: Instantiated EmbeddingProviderInterface implementation.
    """
    if force_mock:
        logger.info(FUNC_GET_PROVIDER, "Instantiating MockEmbeddingProvider (force_mock=True).")
        return MockEmbeddingProvider()

    config = get_config()
    has_openai_key = bool(api_key or config.openai_api_key or os.getenv("OPENAI_API_KEY"))

    if not has_openai_key:
        logger.info(FUNC_GET_PROVIDER, "No OPENAI_API_KEY detected. Defaulting to MockEmbeddingProvider for offline mode.")
        return MockEmbeddingProvider()

    try:
        provider = OpenAIEmbeddingProvider(api_key=api_key, model_id=model_id)
        logger.info(FUNC_GET_PROVIDER, "Successfully initialized real OpenAIEmbeddingProvider.")
        return provider
    except Exception as exc:
        if has_openai_key:
            logger.error(FUNC_GET_PROVIDER, "Failed initializing OpenAI client despite configured API key.", exc=exc)
            raise exc
        logger.warning(
            FUNC_GET_PROVIDER,
            f"Could not initialize OpenAI client: {exc}. Falling back to MockEmbeddingProvider."
        )
        return MockEmbeddingProvider()
