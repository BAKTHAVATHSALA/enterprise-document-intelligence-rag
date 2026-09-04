"""Embedding Provider Interfaces and Domain Abstractions.

Defines the abstract interface contract for dense vector embedding generators,
enabling dependency inversion across AWS Bedrock, local, and mock embedding providers.
"""

from abc import ABC, abstractmethod


class EmbeddingProviderInterface(ABC):
    """Abstract interface contract for embedding vector providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate normalized float embedding vector for a single text string.

        @param text: Input text payload string.
        @returns: Float list representing normalized dense vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized float embedding vectors for a list of text strings.

        @param texts: List of text payload strings.
        @returns: List of float lists representing dense vectors.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Retrieve output vector dimension of the embedding provider.

        @returns: Integer vector dimension (e.g., 1536).
        """
        pass
