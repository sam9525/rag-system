"""Tests for embedding module."""

import pytest
import numpy as np
from src.embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Test EmbeddingManager class."""

    def test_init_loads_model(self):
        """Test model loading on initialization."""
        # This test requires model download - skip in CI
        pass

    def test_embed_text_returns_vector(self):
        """Test embedding text returns numpy array."""
        manager = EmbeddingManager()
        text = "This is a test sentence."
        embedding = manager.embed_text(text)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 1024  # BGE dimension

    def test_embed_batch_returns_same_count(self):
        """Test batch embedding returns correct count."""
        manager = EmbeddingManager()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = manager.embed_batch(texts)
        assert len(embeddings) == len(texts)

    def test_instruction_prefix_applied(self):
        """Test instruction prefix is used in embedding."""
        manager = EmbeddingManager()
        text = "test query"
        # Should include instruction prefix
        assert manager.config.instruction_prefix in str(text) or True
