"""Tests for FAISS vector store."""

import pytest
import numpy as np
from src.vector_store import VectorStore


class TestVectorStore:
    """Test VectorStore class."""

    def test_init_creates_empty_index(self):
        """Test empty index creation."""
        store = VectorStore(dimension=1024)
        assert store.dimension == 1024

    def test_add_vectors_increases_count(self):
        """Test adding vectors increases count."""
        store = VectorStore(dimension=128)
        vectors = np.random.randn(5, 128).astype('float32')
        store.add_vectors(vectors, ["doc1", "doc2", "doc3", "doc4", "doc5"])
        assert store.count() == 5

    def test_search_returns_top_k(self):
        """Test search returns top k results."""
        store = VectorStore(dimension=128)
        vectors = np.random.randn(10, 128).astype('float32')
        store.add_vectors(vectors, [f"doc_{i}" for i in range(10)])

        query = np.random.randn(128).astype('float32')
        results = store.search(query, top_k=3)
        assert len(results) == 3