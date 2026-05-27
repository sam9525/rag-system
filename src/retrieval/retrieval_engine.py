"""Interfaces for retrieval engines and fusion strategies."""

from typing import Protocol
from src.retrieval.search_result import SearchResult


class RetrievalEngine(Protocol):
    """Interface for a single retrieval branch.

    Implementations: SemanticSearch (FAISS), KeywordSearch (BM25).
    """

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search and return top-k results."""
        ...

    def load(self, path: str) -> None:
        """Load index from disk."""
        ...

    def save(self, path: str) -> None:
        """Save index to disk."""
        ...


class FusionStrategy(Protocol):
    """Interface for combining results from multiple retrieval engines.

    Implementations: RRFFusion.
    """

    def fuse(self, results: list[list[SearchResult]], k: int) -> list[SearchResult]:
        """Fuse ranked results from multiple engines.

        Args:
            results: List of ranked result lists, one per engine
            k: RRF smoothing parameter

        Returns:
            Combined ranked results
        """
        ...
