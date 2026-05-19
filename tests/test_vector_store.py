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
        store.add_vectors(vectors)
        assert store.count() == 5

    def test_search_returns_top_k(self):
        """Test search returns top k results with chunk indices."""
        store = VectorStore(dimension=128)
        vectors = np.random.randn(10, 128).astype('float32')
        store.add_vectors(vectors)

        query = np.random.randn(128).astype('float32')
        results = store.search(query, top_k=3)
        assert len(results) == 3
        # Results are (chunk_index, distance)
        assert all(isinstance(idx, int) for idx, _ in results)

    def test_inject_custom_index(self):
        """Test VectorStore accepts injected index."""
        import faiss
        custom_index = faiss.IndexFlatIP(128)
        store = VectorStore(dimension=128, index=custom_index)

        assert store.index is custom_index
        assert store.count() == 0

    def test_inject_custom_normalizer(self):
        """Test VectorStore accepts custom normalizer."""
        def noop_normalize(vectors):
            return vectors.astype('float32')

        store = VectorStore(dimension=128, normalizer=noop_normalize)

        vectors = np.random.rand(2, 128).astype('float32')
        result = store.normalizer(vectors)

        assert result.dtype == np.float32
        assert result.shape == (2, 128)

    def test_save_and_load_index(self, tmp_path):
        """Test saving and loading FAISS index."""
        store = VectorStore(dimension=128)
        vectors = np.random.randn(5, 128).astype('float32')
        store.add_vectors(vectors)

        # Save
        save_path = tmp_path / "vectors.faiss"
        store.save(str(save_path))

        # Load into new store
        new_store = VectorStore(dimension=128)
        new_store.load(str(save_path))

        assert new_store.count() == 5

        # Verify search works after load
        query = np.random.randn(128).astype('float32')
        results = new_store.search(query, top_k=3)
        assert len(results) == 3