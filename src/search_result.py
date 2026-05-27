"""Unified search result type for all retrieval branches."""

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Single search result from any retrieval branch.

    This is the universal output type for all retrieval engines (semantic,
    keyword, neural). It unifies the different internal types (FAISS tuples,
    BM25Result, RerankResult) into a single interface.
    """

    chunk_index: int
    score: float
    text: str
    metadata: dict[str, Any]
