"""FAISS vector store for semantic search."""

from typing import List, Tuple
import numpy as np

import faiss


class VectorStore:
    """FAISS-based vector store for semantic search.

    Stores ONLY embeddings. Text and metadata come from chunks.json.
    """

    def __init__(self, dimension: int, index=None, normalizer=None):
        """Initialize vector store.

        Args:
            dimension: Embedding dimension.
            index: Optional FAISS index. Creates IndexFlatIP if None.
            normalizer: Optional normalizer function. Defaults to L2 normalizer.
        """
        self.dimension = dimension
        self.index = index or faiss.IndexFlatIP(dimension)
        self.normalizer = normalizer or self._l2_normalize

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        vectors = vectors.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        return vectors

    def add_vectors(self, embeddings: np.ndarray):
        """Add vectors to the index.

        Args:
            embeddings: Numpy array of shape (n, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {embeddings.shape[1]} != expected {self.dimension}"
            )

        embeddings = self.normalizer(embeddings)
        self.index.add(embeddings)

    def search(
        self, query_vector: np.ndarray, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Search for top k similar vectors.

        Returns:
            List of (chunk_index, distance) tuples - indices only, no text/metadata
        """
        query_vector = query_vector.astype("float32")

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # L2 normalize query vector for cosine similarity
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        distances, indices = self.index.search(query_vector, min(top_k, self.count()))

        return [
            (int(idx), float(dist))
            for dist, idx in zip(distances[0], indices[0])
            if idx >= 0
        ]

    def count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def save(self, path: str):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        loaded_dim = self.index.d
        if loaded_dim != self.dimension:
            raise ValueError(
                f"Loaded index dimension {loaded_dim} != expected {self.dimension}"
            )

    def clear(self):
        """Clear all vectors from the store."""
        self.index = faiss.IndexFlatIP(self.dimension)
