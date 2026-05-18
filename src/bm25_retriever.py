"""BM25 keyword-based retriever with metadata support."""

from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class BM25Result:
    """Represents a BM25 search result with metadata."""
    text: str
    score: float
    metadata: dict
    index: int


class BM25RetrieverWrapper:
    """BM25 keyword retriever that preserves document metadata."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 parameters.

        Args:
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.metadata_list: List[dict] = []
        self._doc_lengths: List[int] = []
        self._avg_doc_length: float = 0.0
        self._inverted_index: Dict[str, List[tuple]] = {}

    def index_documents_from_chunks(self, chunks: List[Dict]):
        """Index documents from chunks dict with text and metadata.

        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        self.corpus = [chunk["text"] for chunk in chunks]
        self.metadata_list = [chunk.get("metadata", {}) for chunk in chunks]

        self._build_index()

    def index_documents(self, documents: List[str]):
        """Index plain text documents (legacy compatibility).

        Args:
            documents: List of text strings
        """
        self.corpus = documents
        self.metadata_list = [{} for _ in documents]

        self._build_index()

    def _build_index(self):
        """Build inverted index for BM25."""
        if not self.corpus:
            return

        self._doc_lengths = [len(self._tokenize(doc)) for doc in self.corpus]
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)

        self._inverted_index = {}
        for idx, doc in enumerate(self.corpus):
            tokens = self._tokenize(doc)
            for token in set(tokens):
                if token not in self._inverted_index:
                    self._inverted_index[token] = []
                self._inverted_index[token].append(idx)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text: lowercase, simple word split."""
        return text.lower().split()

    def _calculate_score(self, doc_idx: int, query_terms: List[str]) -> float:
        """Calculate BM25 score for a document given query terms."""
        doc_length = self._doc_lengths[doc_idx]
        score = 0.0
        doc_tokens = set(self._tokenize(self.corpus[doc_idx]))

        for term in query_terms:
            if term not in doc_tokens:
                continue

            df = len(self._inverted_index.get(term, []))
            idf = math.log((len(self.corpus) - df + 0.5) / (df + 0.5) + 1)

            tf = self._tokenize(self.corpus[doc_idx]).count(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / max(self._avg_doc_length, 1))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """Search for query using BM25 with metadata.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of BM25Result with text, score, and metadata
        """
        if not self.corpus:
            raise ValueError("No documents indexed. Call index_documents or index_documents_from_chunks first.")

        query_terms = self._tokenize(query)

        scores = []
        for idx in range(len(self.corpus)):
            score = self._calculate_score(idx, query_terms)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append(BM25Result(
                text=self.corpus[idx],
                score=score,
                metadata=self.metadata_list[idx],
                index=idx
            ))

        return results

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self.corpus)
