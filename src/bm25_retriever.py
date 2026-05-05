"""BM25 keyword-based retriever."""

from dataclasses import dataclass
from typing import List

try:
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "BM25Retriever requires langchain-core and langchain-community. "
        "Install with: pip install langchain-core langchain-community"
    ) from e


@dataclass
class BM25Result:
    """Represents a BM25 search result."""
    text: str
    score: float
    index: int


class BM25RetrieverWrapper:
    """BM25 keyword retriever wrapper."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 parameters."""
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self._retriever = None

    def index_documents(self, documents: List[str]):
        """Index documents for BM25 search."""
        self.corpus = documents

        # Create LangChain documents
        langchain_docs = [
            Document(page_content=doc, metadata={"index": i})
            for i, doc in enumerate(documents)
        ]

        # Create BM25 retriever
        self._retriever = BM25Retriever.from_documents(
            langchain_docs,
            k1=self.k1,
            b=self.b
        )

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """Search for query using BM25."""
        if self._retriever is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        # Get more results than needed for ranking
        retrieved_docs = self._retriever.invoke(query, k=top_k)

        results = []
        for doc in retrieved_docs:
            results.append(BM25Result(
                text=doc.page_content,
                score=1.0,  # BM25 scores are not normalized
                index=doc.metadata.get("index", 0)
            ))

        return results[:top_k]

    def count(self) -> int:
        """Get number of indexed documents."""
        return len(self.corpus)
