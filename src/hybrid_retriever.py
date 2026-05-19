"""Hybrid retriever combining semantic and keyword search with RRF fusion."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result
from src.config import config


@dataclass
class RRFResult:
    """Result from RRF fusion with full chunk data."""
    text: str
    score: float
    metadata: dict
    chunk_index: int = 0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


class HybridRetriever:
    """Combines semantic (FAISS) and keyword (BM25) retrieval with RRF fusion.

    Uses chunks.json as single source of truth. FAISS and BM25 only store
    embeddings/inverted-index, not text/metadata.
    """

    def __init__(self, embedding_manager: EmbeddingManager = None, embedding_dim: int = None, config_override=None):
        """Initialize hybrid retriever.

        Args:
            embedding_manager: Optional embedding manager. Creates default if None.
            embedding_dim: Dimension for vector store (required if no manager provided).
            config_override: Optional config replacement.
        """
        self.config = config_override or config.retrieval

        if embedding_manager is not None:
            self.embedding_manager = embedding_manager
            self._embedding_dim = embedding_manager.dimension
        else:
            self.embedding_manager = EmbeddingManager()
            self._embedding_dim = embedding_dim or config.embedding.dimension

        self.vector_store = VectorStore(self._embedding_dim)
        self.bm25_retriever = BM25RetrieverWrapper()

        # Single source of truth for all chunks
        self.chunks: List[Dict] = []

    def index_documents(self, chunks: List[Dict]):
        """Index documents for hybrid retrieval.

        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        if not chunks:
            return

        # Store chunks as single source of truth
        self.chunks = chunks

        # Get texts for embedding
        texts = [chunk["text"] for chunk in chunks]

        # Semantic indexing - embed and store in FAISS
        embeddings = self.embedding_manager.embed_batch(texts)
        self.vector_store.add_vectors(embeddings)

        # Keyword indexing - BM25 (BM25 stores indices, not full text)
        self.bm25_retriever.index_documents_from_chunks(chunks)

    def _rrf_fusion(
        self,
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """Apply Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1/(k + rank_i(d))

        Args:
            semantic_results: List of (chunk_index, score) from FAISS
            keyword_results: List of (chunk_index, score) from BM25
            k: RRF smoothing parameter (default 60)

        Returns:
            Combined ranking as list of (chunk_index, rrf_score)
        """
        # Build score maps keyed by chunk index
        rrf_scores: Dict[int, float] = {}

        for rank, (chunk_idx, _) in enumerate(semantic_results, 1):
            if chunk_idx not in rrf_scores:
                rrf_scores[chunk_idx] = 0
            rrf_scores[chunk_idx] += 1 / (k + rank)

        for rank, (chunk_idx, _) in enumerate(keyword_results, 1):
            if chunk_idx not in rrf_scores:
                rrf_scores[chunk_idx] = 0
            rrf_scores[chunk_idx] += 1 / (k + rank)

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
            List of RRFResult objects with full chunk data from chunks list
        """
        # Handle top_k alias for final_top_k
        if top_k is not None:
            final_top_k = top_k
        else:
            final_top_k = final_top_k or self.config.final_top_k

        semantic_top_k = semantic_top_k or self.config.semantic_top_k
        keyword_top_k = keyword_top_k or self.config.keyword_top_k

        # Empty corpus handling
        if not self.chunks:
            return []

        # Semantic search - returns (chunk_index, distance)
        query_embedding = self.embedding_manager.embed_text(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=semantic_top_k)

        # Keyword search - returns (chunk_index, score)
        keyword_results = self.bm25_retriever.search(query, top_k=keyword_top_k)

        # RRF fusion on indices
        fused_ranking = self._rrf_fusion(
            semantic_results,
            [(r.chunk_index, r.score) for r in keyword_results],
            k=self.config.rrf_k
        )

        # Build lookup for scores
        sem_lookup = {idx: score for idx, score in semantic_results}
        kw_lookup = {r.chunk_index: r.score for r in keyword_results}

        # Build final results with full chunk data from self.chunks
        results = []
        for chunk_idx, rrf_score in fused_ranking[:final_top_k]:
            if chunk_idx < len(self.chunks):
                chunk = self.chunks[chunk_idx]
                sem_score = sem_lookup.get(chunk_idx)
                kw_score = kw_lookup.get(chunk_idx)

                results.append(RRFResult(
                    text=chunk["text"],
                    score=rrf_score,
                    metadata=chunk.get("metadata", {}),
                    chunk_index=chunk_idx,
                    semantic_score=sem_score,
                    keyword_score=kw_score
                ))

        return results

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self.chunks)

    def load_chunks(self, chunks: List[Dict]):
        """Load chunks from external source (e.g., chunks.json)."""
        self.chunks = chunks

    def save(self, vector_path: str, bm25_path: str):
        """Save both vector and BM25 indexes to disk.

        Args:
            vector_path: Path to save FAISS index
            bm25_path: Path to save BM25 index
        """
        self.vector_store.save(vector_path)
        self.bm25_retriever.save(bm25_path)

    def load(self, vector_path: str, bm25_path: str):
        """Load both vector and BM25 indexes from disk.

        Args:
            vector_path: Path to FAISS index
            bm25_path: Path to BM25 index
        """
        print(f"Loading vector index from {vector_path}...")
        self.vector_store.load(vector_path)
        print(f"Loading BM25 index from {bm25_path}...")
        self.bm25_retriever.load(bm25_path)