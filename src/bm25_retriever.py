"""BM25 keyword-based retriever using rank_bm25 library."""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    """Represents a BM25 search result."""

    chunk_index: int
    score: float


class BM25RetrieverWrapper:
    """BM25 keyword retriever using rank_bm25 library.

    Wraps BM25Okapi to provide index-based storage while maintaining
    compatibility with chunks.json source of truth.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 retriever.

        Note: k1 and b parameters are accepted for API compatibility
        but rank_bm25 uses its own defaults (k1=1.5, b=0.75).
        """
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi = None
        self._tokenized_corpus: List[List[str]] = []
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text: lowercase, simple word split."""
        return text.lower().split()

    def index_documents_from_chunks(self, chunks: List[Dict]):
        """Build BM25 index from chunks.

        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        if not chunks:
            self._bm25 = None
            self._tokenized_corpus = []
            self._doc_count = 0
            return

        # Tokenize all chunks
        self._tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self._doc_count = len(chunks)

        # Add a dummy empty document to ensure IDF values are positive.
        # rank_bm25 uses formula: log((N - df + 0.5) / (df + 0.5))
        # With small corpora where a term appears in N/2 documents,
        # IDF can become 0, making all scores 0.
        # Adding a dummy document shifts N, making IDF positive.
        dummy_corpus = self._tokenized_corpus + [["__dummy__"]]
        self._bm25 = BM25Okapi(dummy_corpus)

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """Search for query using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of BM25Result with chunk_index and score
        """
        if self._bm25 is None:
            raise ValueError(
                "No documents indexed. Call index_documents_from_chunks first."
            )

        query_tokens = self._tokenize(query)

        # Get scores for all documents
        all_scores = self._bm25.get_scores(query_tokens)

        # Pair indices with scores, filter non-zero
        scored_docs = [
            (idx, score) for idx, score in enumerate(all_scores) if score > 0
        ]

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [
            BM25Result(chunk_index=idx, score=score)
            for idx, score in scored_docs[:top_k]
        ]

    def count(self) -> int:
        """Get number of indexed documents."""
        return self._doc_count

    def save(self, path: str):
        """Save BM25 index state to disk.

        Note: rank_bm25 doesn't have built-in serialization.
        We store the tokenized corpus to rebuild the index on load.

        Args:
            path: File path to save JSON data
        """
        if self._bm25 is None:
            raise ValueError(
                "No index to save. Call index_documents_from_chunks first."
            )

        data = {
            "k1": self.k1,
            "b": self.b,
            "_tokenized_corpus": self._tokenized_corpus,
            "_doc_count": self._doc_count,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        """Load BM25 index from disk.

        Args:
            path: File path to load JSON data from
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self._tokenized_corpus = data["_tokenized_corpus"]
        self._doc_count = data["_doc_count"]

        # Rebuild BM25 index from tokenized corpus (with dummy doc for IDF)
        dummy_corpus = self._tokenized_corpus + [["__dummy__"]]
        self._bm25 = BM25Okapi(dummy_corpus)
