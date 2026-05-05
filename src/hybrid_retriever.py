"""Hybrid retriever combining semantic and keyword search with RRF fusion."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore, SearchResult
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result
from src.config import config


@dataclass
class RRFResult:
    """Result from RRF fusion."""
    text: str
    score: float
    metadata: dict
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


class HybridRetriever:
    """Combines semantic (FAISS) and keyword (BM25) retrieval with RRF fusion."""

    def __init__(self, embedding_dim: int = None, config_override=None):
        """Initialize hybrid retriever."""
        self.config = config_override or config.retrieval
        self._embedding_dim = embedding_dim or config.embedding.dimension

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(self._embedding_dim)
        self.bm25_retriever = BM25RetrieverWrapper()

    def index_documents(self, chunks: List[Dict]):
        """Index documents for hybrid retrieval.

        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        metadata_list = [chunk.get("metadata", {}) for chunk in chunks]

        # Semantic indexing - embed and store in FAISS
        embeddings = self.embedding_manager.embed_batch(texts)
        self.vector_store.add_vectors(embeddings, texts, metadata_list)

        # Keyword indexing - BM25
        self.bm25_retriever.index_documents(texts)

    def _rrf_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """Apply Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            semantic_results: List of (text, score) from FAISS
            keyword_results: List of (text, score) from BM25
            k: RRF smoothing parameter (default 60)

        Returns:
            Combined ranking as list of (text, rrf_score)
        """
        # Build score maps
        semantic_scores: Dict[str, float] = {}
        keyword_scores: Dict[str, float] = {}

        for text, score in semantic_results:
            semantic_scores[text] = score

        for text, score in keyword_results:
            keyword_scores[text] = score

        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}

        for rank, (text, _) in enumerate(semantic_results, 1):
            if text not in rrf_scores:
                rrf_scores[text] = 0
            rrf_scores[text] += 1 / (k + rank)

        for rank, (text, _) in enumerate(keyword_results, 1):
            if text not in rrf_scores:
                rrf_scores[text] = 0
            rrf_scores[text] += 1 / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def search(
        self,
        query: str,
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        final_top_k: int = None,
        top_k: int = None  # Alias for final_top_k
    ) -> List[RRFResult]:
        """Search using hybrid retrieval and return top chunks.

        Args:
            query: Search query
            semantic_top_k: Top k from semantic search (default from config)
            keyword_top_k: Top k from keyword search (default from config)
            final_top_k: Final number of results (default from config)
            top_k: Alias for final_top_k

        Returns:
            List of RRFResult objects ranked by fusion score
        """
        # Handle top_k alias for final_top_k
        if top_k is not None:
            final_top_k = top_k
        else:
            final_top_k = final_top_k or self.config.final_top_k

        semantic_top_k = semantic_top_k or self.config.semantic_top_k
        keyword_top_k = keyword_top_k or self.config.keyword_top_k

        # Empty corpus handling
        if self.vector_store.count() == 0:
            return []

        # Semantic search
        query_embedding = self.embedding_manager.embed_text(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=semantic_top_k)

        # Keyword search
        keyword_results = self.bm25_retriever.search(query, top_k=keyword_top_k)

        # RRF fusion
        fused_ranking = self._rrf_fusion(
            [(text, score) for text, score, _ in semantic_results],
            [(r.text, r.score) for r in keyword_results],
            k=self.config.rrf_k
        )

        # Build lookup dicts for O(1) access
        sem_lookup = {t: (s, m) for t, s, m in semantic_results}
        kw_lookup = {r.text: r.score for r in keyword_results}

        # Build final results with metadata
        results = []
        for text, rrf_score in fused_ranking[:final_top_k]:
            sem_score, metadata = sem_lookup.get(text, (None, {}))
            kw_score = kw_lookup.get(text)

            results.append(RRFResult(
                text=text,
                score=rrf_score,
                metadata=metadata,
                semantic_score=sem_score,
                keyword_score=kw_score
            ))

        return results

    def count(self) -> int:
        """Get number of indexed documents."""
        return self.vector_store.count()