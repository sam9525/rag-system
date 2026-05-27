"""Thin orchestrator for hybrid retrieval using pluggable engines."""

from typing import Optional
from pathlib import Path

from src.retrieval.search_result import SearchResult
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.fusion import RRFFusion
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore
from src.retrieval.neural_rerank import NeuralRerank, RerankResult
from src.retrieval.rrf_fusion import RRFResult
from src.storage.chunk_store import ChunkStore
from src.system.config import RetrievalConfig


class HybridRetriever:
    """Orchestrates semantic, keyword, and optional neural retrieval.

    Delegates to pluggable RetrievalEngine and FusionStrategy instances.
    ChunkStore provides the single source of truth for chunk data.
    """

    def __init__(
        self,
        config_override: RetrievalConfig | None = None,
        embedding_manager: EmbeddingManager = None,
        embedding_dim: int = None,
        chunk_store: ChunkStore | None = None,
    ):
        self.config = config_override or RetrievalConfig()
        self._chunk_store = chunk_store or ChunkStore(Path(".rag_index"))

        self.embedding_manager = (
            embedding_manager if embedding_manager is not None else EmbeddingManager()
        )
        # Infer dimension from embedding manager if available
        self._embedding_dim = (
            embedding_dim
            or getattr(self.embedding_manager, "dimension", None)
            or EmbeddingConfig().dimension
        )

        self._semantic = SemanticSearch(
            embedding_manager=self.embedding_manager,
            chunk_store=self._chunk_store,
            embedding_dim=self._embedding_dim,
        )
        self._keyword = KeywordSearch(chunk_store=self._chunk_store)
        self._fusion = RRFFusion(k=self.config.rrf_k)

        # Expose vector_store for backwards compatibility
        self.vector_store = self._semantic._vector_store

        self.rerank: Optional[NeuralRerank] = None

    def set_rerank(self, rerank: NeuralRerank):
        self.rerank = rerank

    def index_documents(self, chunks: list[dict]):
        """Index documents for retrieval."""
        if not chunks:
            return
        self._chunk_store.set_chunks(chunks)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_manager.embed_batch(texts)
        self._semantic.add_vectors(embeddings)
        self._keyword.index_documents(chunks)

    def search(
        self,
        query: str,
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        final_top_k: int = None,
        rerank_mode: str = "hybrid",
    ) -> list[RRFResult]:
        """Search using hybrid retrieval."""
        final_top_k = final_top_k or self.config.final_top_k
        semantic_top_k = semantic_top_k or self.config.semantic_top_k
        keyword_top_k = keyword_top_k or self.config.keyword_top_k

        if self._chunk_store.chunk_count() == 0:
            return []

        semantic_results = self._semantic.search(query, top_k=semantic_top_k)
        keyword_results = self._keyword.search(query, top_k=keyword_top_k)

        if rerank_mode == "rrf":
            fused = self._fusion.fuse(
                [semantic_results, keyword_results], k=self.config.rrf_k
            )
            return self._search_results_to_rrf(fused)[:final_top_k]
        elif rerank_mode == "neural":
            all_results = self._deduplicate(semantic_results, keyword_results)
            if self.rerank is None:
                return self._search_results_to_rrf(all_results)[:final_top_k]
            reranked = self.rerank(
                query,
                [r.text for r in all_results],
                top_k=final_top_k,
            )
            return [self._rerank_to_rrf(r) for r in reranked]
        else:
            fused = self._fusion.fuse(
                [semantic_results, keyword_results], k=self.config.rrf_k
            )
            candidates = fused[: self.config.rerank_top_k]
            if self.rerank is None:
                return self._search_results_to_rrf(candidates)[:final_top_k]
            reranked = self.rerank(
                query,
                [c.text for c in candidates],
                top_k=final_top_k,
            )
            return [self._rerank_to_rrf(r) for r in reranked]

    def _search_results_to_rrf(self, results: list[SearchResult]) -> list[RRFResult]:
        """Convert SearchResult list to RRFResult list."""
        sem_lookup = {}
        return [
            RRFResult(
                text=r.text,
                score=r.score,
                metadata=r.metadata,
                chunk_index=r.chunk_index,
                semantic_score=sem_lookup.get(r.chunk_index),
                keyword_score=sem_lookup.get(r.chunk_index),
            )
            for r in results
        ]

    def _deduplicate(
        self, semantic: list[SearchResult], keyword: list[SearchResult]
    ) -> list[SearchResult]:
        seen = set()
        results = []
        for r in semantic + keyword:
            if r.text not in seen:
                seen.add(r.text)
                results.append(r)
        return results

    def _rerank_to_rrf(self, rerank_result: RerankResult) -> RRFResult:
        """Convert RerankResult to RRFResult, preserving metadata from chunks."""
        chunk_idx, chunk = self._chunk_store.lookup_by_text(rerank_result.text)
        return RRFResult(
            text=rerank_result.text,
            score=rerank_result.rerank_score,
            metadata=chunk.get("metadata", {}) if chunk else {},
            chunk_index=chunk_idx,
        )

    def count(self) -> int:
        """Get number of indexed documents."""
        return self._chunk_store.chunk_count()

    def load_chunks(self, chunks: list[dict]):
        """Load chunks from external source (e.g., chunks.json)."""
        self._chunk_store.set_chunks(chunks)

    def load(self, vector_path: str, bm25_path: str, chunks_path: str = None):
        self._semantic.load(vector_path)
        self._keyword.load(bm25_path)
        self._chunk_store.load()

    def save(self, vector_path: str, bm25_path: str, chunks_path: str = None):
        self._semantic.save(vector_path)
        self._keyword.save(bm25_path)
        self._chunk_store.save()
