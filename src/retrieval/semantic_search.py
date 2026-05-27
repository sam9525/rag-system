"""Semantic search using FAISS and embeddings."""

from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore
from src.retrieval.search_result import SearchResult
from src.storage.chunk_store import ChunkStore
from src.retrieval.retrieval_engine import RetrievalEngine


class SemanticSearch(RetrievalEngine):
    """FAISS-based semantic search backed by BGE embeddings."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        chunk_store: ChunkStore,
        embedding_dim: int = 1024,
    ):
        self._embedding_manager = embedding_manager
        self._chunk_store = chunk_store
        self._vector_store = VectorStore(embedding_dim)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search semantic index for similar chunks."""
        query_embedding = self._embedding_manager.embed_text(query)
        results = self._vector_store.search(query_embedding, top_k=top_k)
        return [
            SearchResult(
                chunk_index=idx,
                score=score,
                text=self._chunk_store.get_chunk(idx)["text"],
                metadata=self._chunk_store.get_chunk(idx).get("metadata", {}),
            )
            for idx, score in results
        ]

    def load(self, path: str) -> None:
        self._vector_store.load(path)

    def save(self, path: str) -> None:
        self._vector_store.save(path)

    def add_vectors(self, embeddings) -> None:
        """Add vectors to the index. Called during indexing."""
        self._vector_store.add_vectors(embeddings)
