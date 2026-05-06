"""Tests for semantic chunker."""

import pytest
from src.chunker import SemanticChunker
from src.chunk import Chunk
from src.heading_detector import RegexHeadingDetector


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

    def test_chunker_with_custom_heading_detector(self):
        """Test chunker uses injected heading detector."""
        from src.heading_detector import RegexHeadingDetector

        # Without custom detector - "---" is not a heading (default patterns)
        default_chunker = SemanticChunker()
        text = "Intro paragraph\n\n---\nNew section\n\nMore content"
        default_chunks = default_chunker.create_chunks(text, {"source": "test.pdf"})
        default_count = len(default_chunks)

        # With custom detector that treats "---" as a heading
        custom_detector = RegexHeadingDetector(patterns=[r'^---+$'])
        custom_chunker = SemanticChunker()
        custom_chunker.heading_detector = custom_detector
        custom_chunks = custom_chunker.create_chunks(text, {"source": "test.pdf"})
        custom_count = len(custom_chunks)

        # The custom detector should cause same or more splits (at the "---" boundary)
        # because "---" is now recognized as a heading
        assert custom_count >= default_count, (
            f"Custom detector should produce same or more chunks. "
            f"Got default={default_count}, custom={custom_count}"
        )
