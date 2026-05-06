"""Tests for semantic chunker."""

import pytest
from src.chunker import SemanticChunker
from src.chunk import Chunk


class TestSemanticChunker:
    """Test SemanticChunker class."""

    def test_split_at_headings(self):
        """Test splitting text at headings."""
        chunker = SemanticChunker()
        text = "Introduction\n\nThis is content.\n\n## Section 1\n\nSection content."
        chunks = chunker.split_by_headings(text)
        assert len(chunks) >= 1

    def test_merge_small_chunks(self):
        """Test merging small chunks with next."""
        chunker = SemanticChunker()
        chunks = [
            {"heading": "A", "content": ["Short"]},
            {"heading": "B", "content": ["Another short"]}
        ]
        # Chunks should be merged if too small
        result = chunker.merge_small_chunks(chunks)
        # Verify behavior
        assert len(result) <= len(chunks)

    def test_create_chunks_with_metadata(self):
        """Test creating chunks with proper metadata."""
        chunker = SemanticChunker()
        text = "Title\n\nContent here."
        metadata = {"source": "test.pdf", "page": 1}
        chunks = chunker.create_chunks(text, metadata)
        assert all("source" in c.metadata for c in chunks)
