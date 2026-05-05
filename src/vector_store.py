"""FAISS vector store for semantic search."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

import faiss


@dataclass
class SearchResult:
    """Represents a search result."""
    text: str
    score: float
    metadata: dict


class VectorStore:
    """FAISS-based vector store for semantic search."""

    def __init__(self, dimension: int):
        """Initialize empty vector store."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product Search (cosine sim with normalized)
        self.texts: List[str] = []
        self.metadata_list: List[dict] = []

    def add_vectors(self, vectors: np.ndarray, texts: List[str], metadata_list: Optional[List[dict]] = None):
        """Add vectors to the index."""
        if len(vectors) != len(texts):
            raise ValueError("Vectors and texts must have same length")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != expected {self.dimension}")

        # FAISS requires float32
        vectors = vectors.astype('float32')

        self.index.add(vectors)
        self.texts.extend(texts)

        if metadata_list:
            self.metadata_list.extend(metadata_list)
        else:
            self.metadata_list.extend([{} for _ in texts])

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, dict]]:
        """Search for top k similar vectors."""
        query_vector = query_vector.astype('float32')

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, min(top_k, self.count()))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(dist), self.metadata_list[idx]))

        return results

    def count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def save(self, path: str):
        """Save index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)

    def clear(self):
        """Clear all vectors from the store."""
        self.dimension = self.dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.metadata_list = []