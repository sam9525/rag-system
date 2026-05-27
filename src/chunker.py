"""Chunker using langchain RecursiveCharacterTextSplitter."""

from dataclasses import dataclass
from typing import Dict, Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import ChunkingConfig


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
        return {"text": self.text, "metadata": self.metadata}


_langchain_splitter_cache = {}


def _get_langchain_splitter(chunk_config) -> RecursiveCharacterTextSplitter:
    """Get or create a langchain splitter for the given config."""
    cache_key = (
        chunk_config.max_chunk_size,
        chunk_config.overlap_size,
        tuple(chunk_config.separators),
    )
    if cache_key not in _langchain_splitter_cache:
        _langchain_splitter_cache[cache_key] = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config.max_chunk_size,
            chunk_overlap=chunk_config.overlap_size,
            separators=chunk_config.separators,
            length_function=len,
        )
    return _langchain_splitter_cache[cache_key]


def create_chunks(text: str, base_metadata: Dict, chunk_config=None) -> List[Chunk]:
    """Create chunks using langchain RecursiveCharacterTextSplitter.

    Args:
        text: Document text to chunk.
        base_metadata: Metadata to attach to each chunk.
        chunk_config: Optional configuration override.

    Returns:
        List of Chunk objects.
    """
    cfg = chunk_config or ChunkingConfig()
    splitter = _get_langchain_splitter(cfg)

    doc = Document(page_content=text)
    split_docs = splitter.split_documents([doc])

    result = []
    for idx, split_doc in enumerate(split_docs):
        result.append(
            Chunk(
                text=split_doc.page_content,
                metadata={**base_metadata, "chunk_id": idx},
                chunk_id=idx,
            )
        )

    return result
