"""Shared Chunk type for RAG pipeline."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Chunk:
    """Represents a document chunk in the RAG pipeline.

    Attributes:
        text: The chunk content.
        metadata: Source metadata (source, page, section, etc.).
        chunk_id: Index of this chunk in the document.
    """
    text: str
    metadata: Dict[str, Any]
    chunk_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for retriever indexing."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }