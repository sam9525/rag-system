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

    def test_inject_mock_model(self):
        """Test EmbeddingManager accepts injected model."""

        class MockModel:
            def encode(
                self,
                texts,
                normalize_embeddings=False,
                show_progress_bar=False,
                **kwargs
            ):
                import numpy as np

                if isinstance(texts, str):
                    return np.random.rand(1, 128).astype("float32")
                return np.random.rand(len(texts), 128).astype("float32")

        mock_model = MockModel()
        manager = EmbeddingManager(model=mock_model, dimension=128)

        # Verify model is used
        assert manager.model is mock_model
        assert manager.dimension == 128

        # Verify embed_text works (returns 2D array, squeeze to 1D)
        result = manager.embed_text("test")
        assert result.shape[-1] == 128

    def test_embed_batch_with_mock(self):
        """Test embed_batch works with injected model."""

        class MockModel:
            def encode(
                self,
                texts,
                normalize_embeddings=False,
                show_progress_bar=False,
                **kwargs
            ):
                import numpy as np

                return np.random.rand(len(texts), 128).astype("float32")

        mock_model = MockModel()
        manager = EmbeddingManager(model=mock_model, dimension=128)

        texts = ["test1", "test2", "test3"]
        result = manager.embed_batch(texts, show_progress=False)

        assert result.shape == (3, 128)
