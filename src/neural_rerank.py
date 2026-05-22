"""Neural reranking using cross-encoder models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RerankResult:
    """Result from reranking with score."""

    text: str
    rerank_score: float
    original_index: int


class BaseRerank(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def __call__(self, query: str, chunks: List[str], top_k: int) -> List[RerankResult]:
        """Rerank chunks based on query relevance."""
        pass


class NoOpRerank(BaseRerank):
    """Pass-through reranker that returns chunks in original order.

    Used when neural reranking is disabled or for testing.
    """

    def __call__(self, query: str, chunks: List[str], top_k: int) -> List[RerankResult]:
        results = []
        for i, text in enumerate(chunks[:top_k]):
            results.append(
                RerankResult(
                    text=text,
                    rerank_score=1.0 / (i + 1),  # Decreasing score
                    original_index=i,
                )
            )
        return results


class NeuralRerank(BaseRerank):
    """Neural reranker using sentence-transformers CrossEncoder.

    Uses Microsoft's MiniLM model optimized for MS MARCO passage ranking.
    Downloads model on first use (~100MB).
    """

    default_model: str = "cross-encoder/ms-marco-MiniLM-L-12v2"

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None):
        """Initialize reranker with cross-encoder model.

        Args:
            model: HuggingFace model name. Defaults to ms-marco-MiniLM-L-12v2.
            device: Device for model ('cpu', 'cuda', 'mps'). Auto-detects if None.
        """
        self.model_name = model or self.default_model
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load model on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self._device, max_length=512
            )
        return self._model

    def __call__(self, query: str, chunks: List[str], top_k: int) -> List[RerankResult]:
        """Rerank chunks using cross-encoder relevance scoring.

        Args:
            query: Search query
            chunks: List of text chunks to rerank
            top_k: Number of top results to return

        Returns:
            List of RerankResult sorted by relevance score (descending)
        """
        if not chunks:
            return []

        # Create query-document pairs
        pairs = [[query, chunk] for chunk in chunks]

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs)

        # Build results with scores
        results = [
            RerankResult(
                text=chunks[i],
                rerank_score=(
                    float(scores[i])
                    if hasattr(scores[i], "__float__")
                    else float(scores[i])
                ),
                original_index=i,
            )
            for i in range(len(chunks))
        ]

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        return results[:top_k]
