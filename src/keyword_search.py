"""Keyword search using BM25."""

from src.bm25_retriever import BM25RetrieverWrapper
from src.search_result import SearchResult
from src.chunk_store import ChunkStore
from src.retrieval_engine import RetrievalEngine


class KeywordSearch(RetrievalEngine):
    """BM25-based keyword search."""

    def __init__(self, chunk_store: ChunkStore):
        self._chunk_store = chunk_store
        self._bm25 = BM25RetrieverWrapper()

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search using BM25 keyword matching."""
        results = self._bm25.search(query, top_k=top_k)
        return [
            SearchResult(
                chunk_index=r.chunk_index,
                score=r.score,
                text=self._chunk_store.get_chunk(r.chunk_index)["text"],
                metadata=self._chunk_store.get_chunk(r.chunk_index).get("metadata", {}),
            )
            for r in results
        ]

    def load(self, path: str) -> None:
        self._bm25.load(path)

    def save(self, path: str) -> None:
        self._bm25.save(path)

    def index_documents(self, chunks: list[dict]) -> None:
        """Index documents. Called during indexing."""
        self._bm25.index_documents_from_chunks(chunks)
