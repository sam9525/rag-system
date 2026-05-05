"""Embedding manager for BGE model."""

from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from src.config import config


class EmbeddingManager:
    """Manages text embeddings using BGE model."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize embedding manager with model."""
        self.config = config.embedding

        if model_name:
            self.config.model_name = model_name

        # Load model (default to CPU, can specify 'cuda' if available)
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=device or ("cuda" if self._has_cuda() else "cpu")
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.config.model_name}'. "
                f"Ensure the model is available. Error: {e}"
            )

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text with instruction prefix."""
        prefixed_text = self.config.instruction_prefix + text
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=True,  # L2 normalize for FAISS IPS
            show_progress_bar=False
        )
        return embedding

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed multiple texts in batch."""
        prefixed_texts = [
            self.config.instruction_prefix + text
            for text in texts
        ]
        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.model_name
