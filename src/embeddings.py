"""Embedding manager for BGE model."""

from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import config


class EmbeddingManager:
    """Manages text embeddings using BGE model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model=None,
        dimension=None,
    ):
        """Initialize embedding manager.

        Args:
            model_name: Name of the sentence transformer model.
            device: Device to use ('cpu', 'cuda').
            model: Pre-loaded model instance. If provided, model_name and device are ignored.
            dimension: Embedding dimension. Required when injecting a model.
        """
        self.config = config.embedding

        if model is not None:
            self.model = model
            self._dimension = dimension
        elif model_name:
            self.config.model_name = model_name
            self._dimension = None
        else:
            self._dimension = None

        if model is None:
            self.model = self._load_model(device)

        self._device = device or ("cuda" if self._has_cuda() else "cpu")

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self, device=None):
        """Load the sentence transformer model."""
        device = device or ("cuda" if self._has_cuda() else "cpu")
        try:
            return SentenceTransformer(self.config.model_name, device=device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.config.model_name}'. "
                f"Ensure the model is available. Error: {e}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text with instruction prefix."""
        prefixed_text = self.config.instruction_prefix + text
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,  # L2 normalize for FAISS IPS
            show_progress_bar=False,
        )
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts in batch."""
        prefixed_texts = [self.config.instruction_prefix + text for text in texts]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is not None:
            return self._dimension
        return self.config.dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name
