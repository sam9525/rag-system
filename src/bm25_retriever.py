"""BM25 keyword-based retriever using rank_bm25 library."""

from dataclasses import dataclass
from typing import List, Dict
import json

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    """Represents a BM25 search result."""

    chunk_index: int
    score: float


class BM25RetrieverWrapper:
    """BM25 keyword retriever using rank_bm25 library."""

    def __init__(self):
        self._bm25: BM25Okapi = None
        self._corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text: lowercase, simple word split."""
        return text.lower().split()

    def index_documents_from_chunks(self, chunks: List[Dict]):
        """Build BM25 index from chunks."""
        self._corpus = [self._tokenize(c["text"]) for c in chunks]
        # Add dummy doc to fix IDF edge case where term appears in ~50% of docs
        self._bm25 = BM25Okapi(self._corpus + [["__dummy__"]])

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """Search for query using BM25."""
        if self._bm25 is None:
            raise ValueError("No documents indexed.")

        scores = self._bm25.get_scores(self._tokenize(query))
        results = [(i, s) for i, s in enumerate(scores) if s > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return [BM25Result(chunk_index=i, score=s) for i, s in results[:top_k]]

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self._corpus)

    def save(self, path: str):
        """Save BM25 index state to disk."""
        if self._bm25 is None:
            raise ValueError("No index to save.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._corpus, f)

    def load(self, path: str):
        """Load BM25 index from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both old format (with _tokenized_corpus key) and new format (list directly)
        self._corpus = data if isinstance(data, list) else data.get("_corpus", data.get("_tokenized_corpus", []))
        self._bm25 = BM25Okapi(self._corpus + [["__dummy__"]])