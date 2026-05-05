"""Tests for BM25 retriever."""

import pytest
from src.bm25_retriever import BM25RetrieverWrapper, BM25Result


class TestBM25Retriever:
    """Test BM25Retriever class."""

    def test_init(self):
        """Test initialization."""
        retriever = BM25RetrieverWrapper()
        assert retriever.corpus is not None

    def test_index_documents(self):
        """Test indexing documents."""
        retriever = BM25RetrieverWrapper()
        docs = ["Document one text", "Document two content", "Third document here"]
        retriever.index_documents(docs)
        assert len(retriever.corpus) == 3

    def test_search_returns_results(self):
        """Test search returns ranked results."""
        retriever = BM25RetrieverWrapper()
        docs = ["Python programming guide", "JavaScript tutorial", "Machine learning basics"]
        retriever.index_documents(docs)

        results = retriever.search("python programming", top_k=2)
        assert len(results) <= 2
        assert all(hasattr(r, 'text') for r in results)
