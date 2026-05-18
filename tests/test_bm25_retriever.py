"""Tests for BM25 retriever with metadata support."""

import pytest
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result


class TestBM25RetrieverWrapper:
    """Test BM25RetrieverWrapper with metadata."""

    def test_index_documents_preserves_metadata(self):
        """Test that indexing preserves metadata alongside text."""
        retriever = BM25RetrieverWrapper()

        chunks = [
            {"text": "Python programming guide", "metadata": {"source": "python.pdf", "page": 1}},
            {"text": "JavaScript tutorial", "metadata": {"source": "js.pdf", "page": 5}},
            {"text": "Machine learning basics", "metadata": {"source": "ml.pdf", "page": 10}},
        ]

        retriever.index_documents_from_chunks(chunks)

        assert retriever.count() == 3

        results = retriever.search("python programming", top_k=2)

        assert len(results) <= 2
        assert all(hasattr(r, 'text') for r in results)
        assert all(hasattr(r, 'metadata') for r in results)
        # Metadata should contain source and page
        assert results[0].metadata.get("source") == "python.pdf"
        assert results[0].metadata.get("page") == 1

    def test_search_returns_ranked_results(self):
        """Test search returns ranked results with scores."""
        retriever = BM25RetrieverWrapper()

        chunks = [
            {"text": "Python programming guide", "metadata": {"source": "python.pdf", "page": 1}},
            {"text": "JavaScript tutorial", "metadata": {"source": "js.pdf", "page": 5}},
            {"text": "Machine learning basics", "metadata": {"source": "ml.pdf", "page": 10}},
        ]

        retriever.index_documents_from_chunks(chunks)
        results = retriever.search("python", top_k=2)

        assert len(results) >= 1
        assert results[0].text == "Python programming guide"
        assert results[0].metadata["source"] == "python.pdf"

    def test_search_empty_corpus(self):
        """Test search on empty corpus raises ValueError."""
        retriever = BM25RetrieverWrapper()

        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("query", top_k=3)