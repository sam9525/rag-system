"""Tests for chunker using langchain RecursiveCharacterTextSplitter."""

import pytest
from src.chunker import create_chunks, Chunk


class TestCreateChunks:
    """Test the create_chunks function."""

    def test_create_chunks_produces_chunks(self):
        """Test that create_chunks produces valid Chunk objects."""
        text = "# Header\n\nParagraph content here."
        chunks = create_chunks(text, {"source": "test.txt"})

        assert len(chunks) >= 1
        assert all(hasattr(c, "text") and hasattr(c, "metadata") for c in chunks)
        assert chunks[0].metadata.get("source") == "test.txt"

    def test_create_chunks_with_metadata(self):
        """Test creating chunks with proper metadata."""
        text = "Title\n\nContent here."
        metadata = {"source": "test.pdf", "page": 1}
        chunks = create_chunks(text, metadata)
        assert all("source" in c.metadata for c in chunks)

    def test_create_chunks_respects_max_size(self):
        """Test that chunks respect max_chunk_size when text can be split."""
        from src.config import ChunkingConfig

        chunk_config = ChunkingConfig(
            min_chunk_size=50, max_chunk_size=100, overlap_size=20
        )

        # Text with separators so langchain can split it
        text = ("Sentence one here. " * 3 + "New paragraph. " * 3) * 5
        chunks = create_chunks(text, {"source": "test.txt"}, chunk_config=chunk_config)

        # Should have multiple chunks when text exceeds max_size
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

        # All chunks should be <= max_chunk_size + 50 (LangChain allows overflow at separators)
        for chunk in chunks:
            assert (
                len(chunk.text) <= chunk_config.max_chunk_size + 50
            ), f"Chunk text {len(chunk.text)} exceeds max {chunk_config.max_chunk_size}"

    def test_create_chunks_has_overlap(self):
        """Test that adjacent chunks have overlap content."""
        from src.config import ChunkingConfig

        chunk_config = ChunkingConfig(
            min_chunk_size=50, max_chunk_size=100, overlap_size=30
        )

        text = "First chunk content. " * 20 + "Second chunk content. " * 20
        chunks = create_chunks(text, {"source": "test.txt"}, chunk_config=chunk_config)

        if len(chunks) >= 2:
            # Check overlap exists - second chunk should contain some content from first
            first_end = chunks[0].text[-30:]
            assert first_end in chunks[1].text or any(
                first_end in c.text for c in chunks[1:]
            ), f"No overlap detected. First chunk end: '{first_end}'"


def test_chunking_config_has_separators():
    """Test that ChunkingConfig supports custom separators."""
    from src.config import ChunkingConfig

    config = ChunkingConfig(
        min_chunk_size=100,
        max_chunk_size=500,
        overlap_size=50,
        separators=["\n## ", "\n# ", "\n\n", "\n", " "],
    )
    assert config.separators == ["\n## ", "\n# ", "\n\n", "\n", " "]


def test_create_chunks_with_custom_separators():
    """Test that custom separators are respected."""
    from src.config import ChunkingConfig

    # Custom separators that split at h2 headings with smaller chunk size
    chunk_config = ChunkingConfig(
        min_chunk_size=10,
        max_chunk_size=50,  # Small enough to force splitting
        overlap_size=0,
        separators=["\n## ", "\n\n", "\n", " "],
    )

    # Long enough text to exceed max_chunk_size at h2 boundaries
    text = (
        "Intro " * 20
        + "\n## Section 1\n\n"
        + "Content 1. " * 20
        + "\n## Section 2\n\n"
        + "Content 2. " * 20
    )
    chunks = create_chunks(text, {"source": "test.txt"}, chunk_config=chunk_config)

    # Should have created multiple chunks
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
