"""Tests for BM25 retriever with index-based storage."""

import pytest
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result


class TestBM25RetrieverWrapper:
    """Test BM25RetrieverWrapper with index-based storage."""

    def test_index_documents_preserves_metadata(self):
        """Test that indexing works with chunks (metadata from chunks.json)."""
        retriever = BM25RetrieverWrapper()

        chunks = [
            {
                "text": "Python programming guide",
                "metadata": {"source": "python.pdf", "page": 1},
            },
            {
                "text": "JavaScript tutorial",
                "metadata": {"source": "js.pdf", "page": 5},
            },
            {
                "text": "Machine learning basics",
                "metadata": {"source": "ml.pdf", "page": 10},
            },
        ]

        retriever.index_documents_from_chunks(chunks)

        assert retriever.count() == 3

        results = retriever.search("python programming", top_k=2)

        assert len(results) <= 2
        # Results now return (chunk_index, score), not full chunks
        assert all(hasattr(r, "chunk_index") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_search_returns_ranked_results(self):
        """Test search returns ranked results with chunk indices."""
        retriever = BM25RetrieverWrapper()

        chunks = [
            {
                "text": "Python programming guide",
                "metadata": {"source": "python.pdf", "page": 1},
            },
            {
                "text": "JavaScript tutorial",
                "metadata": {"source": "js.pdf", "page": 5},
            },
            {
                "text": "Machine learning basics",
                "metadata": {"source": "ml.pdf", "page": 10},
            },
        ]

        retriever.index_documents_from_chunks(chunks)
        results = retriever.search("python", top_k=2)

        assert len(results) >= 1
        # Result is (chunk_index, score)
        assert results[0].chunk_index == 0  # First chunk contains "python"

    def test_search_empty_corpus(self):
        """Test search on empty corpus raises ValueError."""
        retriever = BM25RetrieverWrapper()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("query", top_k=3)

    def test_save_and_load_index(self, tmp_path):
        """Test saving and loading BM25 index."""
        retriever = BM25RetrieverWrapper()

        chunks = [
            {
                "text": "Python programming guide",
                "metadata": {"source": "python.pdf", "page": 1},
            },
            {
                "text": "JavaScript tutorial",
                "metadata": {"source": "js.pdf", "page": 5},
            },
        ]

        retriever.index_documents_from_chunks(chunks)

        # Save
        save_path = tmp_path / "bm25.json"
        retriever.save(str(save_path))

        # Load into new retriever
        new_retriever = BM25RetrieverWrapper()
        new_retriever.load(str(save_path))

        assert new_retriever.count() == 2

        # Verify search works after load
        results = new_retriever.search("python", top_k=2)
        assert len(results) >= 1
