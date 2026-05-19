"""BM25 keyword-based retriever with metadata support."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math


@dataclass
class BM25Result:
    """Represents a BM25 search result."""
    chunk_index: int
    score: float


class BM25RetrieverWrapper:
    """BM25 keyword retriever that works with indices, not full text storage.

    Stores inverted index and document lengths. Text/metadata come from chunks.json.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 parameters.

        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.k1 = k1
        self.b = b

        # Inverted index: term -> list of (chunk_index, term_count)
        self._inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0

    def index_documents_from_chunks(self, chunks: List[Dict]):
        """Build inverted index from chunks.

        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        self._inverted_index = {}
        self._doc_lengths = []

        for chunk_idx, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk["text"])
            self._doc_lengths.append(len(tokens))

            # Count term frequencies for this chunk
            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            # Add to inverted index
            for term, count in term_counts.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                self._inverted_index[term].append((chunk_idx, count))

        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text: lowercase, simple word split."""
        return text.lower().split()

    def _calculate_score(self, chunk_idx: int, query_terms: List[str]) -> float:
        """Calculate BM25 score for a chunk given query terms."""
        doc_length = self._doc_lengths[chunk_idx]
        score = 0.0

        for term in query_terms:
            if term not in self._inverted_index:
                continue

            # Find this chunk in the inverted index
            df = 0
            tf = 0
            for idx, count in self._inverted_index[term]:
                if idx == chunk_idx:
                    tf = count
                    df += 1
                    break

            if tf == 0:
                continue

            # IDF calculation
            n = len(self._doc_lengths)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # TF saturation
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self._avg_doc_length, 1))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """Search for query using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of BM25Result with chunk_index and score
        """
        if not self._doc_lengths:
            raise ValueError("No documents indexed. Call index_documents_from_chunks first.")

        query_terms = self._tokenize(query)

        scores = []
        for chunk_idx in range(len(self._doc_lengths)):
            score = self._calculate_score(chunk_idx, query_terms)
            if score > 0:
                scores.append((chunk_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [BM25Result(chunk_index=idx, score=score) for idx, score in scores[:top_k]]

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self._doc_lengths)

    def save(self, path: str):
        """Save BM25 index to disk (inverted index + parameters).

        Args:
            path: File path to save JSON data
        """
        import json

        data = {
            "k1": self.k1,
            "b": self.b,
            "_doc_lengths": self._doc_lengths,
            "_avg_doc_length": self._avg_doc_length,
            "_inverted_index": self._inverted_index
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        """Load BM25 index from disk.

        Args:
            path: File path to load JSON data from
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self._doc_lengths = data["_doc_lengths"]
        self._avg_doc_length = data["_avg_doc_length"]
        self._inverted_index = data["_inverted_index"]