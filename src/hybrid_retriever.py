"""Hybrid retriever combining semantic and keyword search with RRF fusion."""

from typing import List, Dict, Optional, Tuple, Union

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result
from src.neural_rerank import NeuralRerank, RerankResult
from src.rrf_fusion import RRFResult, rrf_fusion
from src.config import config
from src.search_result import SearchResult


class HybridRetriever:
    """Combines semantic (FAISS) and keyword (BM25) retrieval with RRF fusion.

    Uses chunks.json as single source of truth. FAISS and BM25 only store
    embeddings/inverted-index, not text/metadata.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager = None,
        embedding_dim: int = None,
        config_override=None,
    ):
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

        # Reranker for second-stage ranking (None = disabled)
        self.rerank: Optional[NeuralRerank] = None

    def set_rerank(self, rerank: NeuralRerank):
        """Set the reranker to use after RRF fusion.

        Args:
            rerank: NeuralRerank instance for reranking
        """
        self.rerank = rerank

    def _to_search_result(self, chunk_idx: int, score: float) -> SearchResult:
        """Convert a chunk index + score to SearchResult using chunks list."""
        if 0 <= chunk_idx < len(self.chunks):
            chunk = self.chunks[chunk_idx]
            return SearchResult(
                chunk_index=chunk_idx,
                score=score,
                text=chunk["text"],
                metadata=chunk.get("metadata", {}),
            )
        return SearchResult(chunk_index=-1, score=0.0, text="", metadata={})

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

    def search(
        self,
        query: str,
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        final_top_k: int = None,
        rerank_mode: str = "hybrid",
    ) -> List[SearchResult]:
        """Search using hybrid retrieval.

        Args:
            query: Search query
            semantic_top_k: Top k from semantic search (default from config)
            keyword_top_k: Top k from keyword search (default from config)
            final_top_k: Final number of results (default from config)
            rerank_mode: One of "rrf", "neural", or "hybrid".
                - "rrf": Only RRF fusion, returns List[RRFResult]
                - "neural": Semantic + keyword + neural reranking, returns List[RerankResult]
                - "hybrid": RRF fusion + neural reranking, returns List[RRFResult]

        Returns:
            For "rrf"/"hybrid": List of RRFResult objects
            For "neural": List of RerankResult objects
        """
        final_top_k = final_top_k or self.config.final_top_k
        semantic_top_k = semantic_top_k or self.config.semantic_top_k
        keyword_top_k = keyword_top_k or self.config.keyword_top_k

        # Empty corpus handling
        if not self.chunks:
            return []

        # Semantic and bm25 search always runs
        query_embedding = self.embedding_manager.embed_text(query)
        semantic_results = self.vector_store.search(
            query_embedding, top_k=semantic_top_k
        )
        keyword_results = self.bm25_retriever.search(query, top_k=keyword_top_k)

        if rerank_mode == "rrf":
            # Pure RRF fusion - no neural reranking
            fused_ranking = rrf_fusion(
                semantic_results,
                [(r.chunk_index, r.score) for r in keyword_results],
                k=self.config.rrf_k,
            )
            results = self._build_results_from_fusion(
                fused_ranking, semantic_results, keyword_results, final_top_k
            )
        elif rerank_mode == "neural":
            # Combine semantic + keyword results, then neural reranking
            all_texts = []
            seen = set()
            for idx, score in semantic_results:
                chunk = self.chunks[idx]
                if chunk["text"] not in seen:
                    all_texts.append(chunk["text"])
                    seen.add(chunk["text"])
            for r in keyword_results:
                kw_text = (
                    self.chunks[r.chunk_index]["text"]
                    if r.chunk_index < len(self.chunks)
                    else None
                )
                if kw_text and kw_text not in seen:
                    all_texts.append(kw_text)
                    seen.add(kw_text)
            reranked = self.rerank(query, all_texts, top_k=final_top_k)
            results = [self._rerank_to_search_result(r) for r in reranked]
        else:
            # hybrid mode: RRF fusion + neural reranking
            fused_ranking = rrf_fusion(
                semantic_results,
                [(r.chunk_index, r.score) for r in keyword_results],
                k=self.config.rrf_k,
            )
            candidates_for_rerank = self._build_results_from_fusion(
                fused_ranking,
                semantic_results,
                keyword_results,
                self.config.rerank_top_k,
            )
            reranked = self.rerank(
                query, [c.text for c in candidates_for_rerank], top_k=final_top_k
            )
            results = [self._rerank_to_search_result(r) for r in reranked]

        return results

    def _build_results_from_fusion(
        self,
        fused_ranking: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
        keyword_results: List,
        final_top_k: int,
    ) -> List[SearchResult]:
        """Build SearchResult list from RRF fusion."""
        results = []
        for chunk_idx, rrf_score in fused_ranking[:final_top_k]:
            if chunk_idx < len(self.chunks):
                results.append(self._to_search_result(chunk_idx, rrf_score))
        return results

    def _rerank_to_search_result(self, rerank_result: RerankResult) -> SearchResult:
        """Convert RerankResult to SearchResult, preserving metadata."""
        chunk_idx = -1
        metadata = {}
        for idx, chunk in enumerate(self.chunks):
            if chunk["text"] == rerank_result.text:
                chunk_idx = idx
                metadata = chunk.get("metadata", {})
                break
        return SearchResult(
            chunk_index=chunk_idx,
            score=rerank_result.rerank_score,
            text=rerank_result.text,
            metadata=metadata,
        )

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self.chunks)

    def load_chunks(self, chunks: List[Dict]):
        """Load chunks from external source (e.g., chunks.json)."""
        self.chunks = chunks

    def save(self, vector_path: str, bm25_path: str, chunks_path: str = None):
        """Save vector index, BM25 index, and chunks to disk.

        Args:
            vector_path: Path to save FAISS index
            bm25_path: Path to save BM25 index
            chunks_path: Path to save chunks (defaults to vector_path.replace('.index', '.chunks.json'))
        """
        import json

        self.vector_store.save(vector_path)
        self.bm25_retriever.save(bm25_path)
        if chunks_path is None:
            chunks_path = vector_path.replace(".index", ".chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        print(f"Saved {len(self.chunks)} chunks to {chunks_path}")

    def load(self, vector_path: str, bm25_path: str, chunks_path: str = None):
        """Load vector index, BM25 index, and chunks from disk.

        Args:
            vector_path: Path to FAISS index
            bm25_path: Path to BM25 index
            chunks_path: Path to chunks (defaults to vector_path.replace('.index', '.chunks.json'))
        """
        import json

        print(f"Loading vector index from {vector_path}...")
        self.vector_store.load(vector_path)
        print(f"Loading BM25 index from {bm25_path}...")
        self.bm25_retriever.load(bm25_path)
        if chunks_path is None:
            chunks_path = vector_path.replace(".index", ".chunks.json")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
