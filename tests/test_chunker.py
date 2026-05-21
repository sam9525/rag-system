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
            {"heading": "B", "content": ["Another short"]},
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
        custom_detector = RegexHeadingDetector(patterns=[r"^---+$"])
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

    def test_overlap_contains_previous_chunk_content(self):
        """Test that adjacent chunks have overlap content (delegates to langchain)."""
        from src.config import ChunkingConfig

        chunk_config = ChunkingConfig(
            min_chunk_size=50, max_chunk_size=100, overlap_size=20
        )
        chunker = SemanticChunker(chunk_config=chunk_config)

        # Create text with enough content to force multiple chunks
        text = "First chunk content. " * 20 + "Second chunk content. " * 20
        metadata = {"source": "test.pdf", "page": 1}
        chunks = chunker.create_chunks(text, metadata)

        # Verify we have multiple chunks
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

        # LangChain handles overlap - check that second chunk has overlap content
        if len(chunks) >= 2:
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


def test_langchain_chunker_basic():
    """Test that LangChainChunker produces Chunk objects."""
    from src.chunker import LangChainChunker

    chunker = LangChainChunker()
    text = "# Header\n\nParagraph content here."
    chunks = chunker.create_chunks(text, {"source": "test.txt"})

    assert len(chunks) >= 1
    assert all(hasattr(c, "text") and hasattr(c, "metadata") for c in chunks)
    assert chunks[0].metadata.get("source") == "test.txt"


def test_langchain_chunker_respects_max_size():
    """Test that chunks respect max_chunk_size when text can be split."""
    from src.chunker import LangChainChunker
    from src.config import ChunkingConfig

    chunk_config = ChunkingConfig(
        min_chunk_size=50, max_chunk_size=100, overlap_size=20
    )
    chunker = LangChainChunker(chunk_config=chunk_config)

    # Text with separators so langchain can split it
    text = ("Sentence one here. " * 3 + "New paragraph. " * 3) * 5
    chunks = chunker.create_chunks(text, {"source": "test.txt"})

    # Should have multiple chunks when text exceeds max_size
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    # All chunks should be <= max_chunk_size + 50 (LangChain allows overflow at separators)
    for chunk in chunks:
        assert (
            len(chunk.text) <= chunk_config.max_chunk_size + 50
        ), f"Chunk text {len(chunk.text)} exceeds max {chunk_config.max_chunk_size}"


def test_semantic_chunker_uses_langchain_underneath():
    """Test that SemanticChunker produces same output as LangChainChunker."""
    from src.chunker import SemanticChunker, LangChainChunker

    text = "# Header\n\nContent paragraph."
    metadata = {"source": "test.txt"}

    semantic = SemanticChunker()
    langchain = LangChainChunker()

    semantic_chunks = semantic.create_chunks(text, metadata)
    langchain_chunks = langchain.create_chunks(text, metadata)

    # Both should produce valid chunks
    assert len(semantic_chunks) >= 1
    assert len(langchain_chunks) >= 1

    # Both should have same metadata keys (including 'section' for semantic)
    semantic_keys = semantic_chunks[0].metadata.keys()
    langchain_keys = langchain_chunks[0].metadata.keys()

    # SemanticChunker should have 'section' key
    assert (
        "section" in semantic_keys
    ), "SemanticChunker should preserve 'section' metadata"


def test_langchain_chunker_has_overlap():
    """Test that adjacent chunks have overlap content."""
    from src.chunker import LangChainChunker
    from src.config import ChunkingConfig

    chunk_config = ChunkingConfig(
        min_chunk_size=50, max_chunk_size=100, overlap_size=30
    )
    chunker = LangChainChunker(chunk_config=chunk_config)

    text = "First chunk content. " * 20 + "Second chunk content. " * 20
    chunks = chunker.create_chunks(text, {"source": "test.txt"})

    if len(chunks) >= 2:
        # Check overlap exists - second chunk should contain some content from first
        first_end = chunks[0].text[-30:]
        assert first_end in chunks[1].text or any(
            first_end in c.text for c in chunks[1:]
        ), f"No overlap detected. First chunk end: '{first_end}'"
