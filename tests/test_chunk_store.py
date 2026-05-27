"""Tests for ChunkStore."""

import pytest
import tempfile
from pathlib import Path
from src.chunk_store import ChunkStore


class TestChunkStore:
    """Test ChunkStore."""

    def test_set_and_get_chunks(self):
        """Test setting and retrieving chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir))
            chunks = [
                {"text": "chunk1", "metadata": {"source": "a.pdf"}},
                {"text": "chunk2", "metadata": {"source": "b.pdf"}},
            ]
            store.set_chunks(chunks)
            assert store.chunk_count() == 2
            assert store.get_chunk(0)["text"] == "chunk1"
            assert store.get_chunk(1)["text"] == "chunk2"

    def test_save_and_load(self):
        """Test persistence round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir))
            chunks = [{"text": "test", "metadata": {"page": 1}}]
            store.set_chunks(chunks)
            store.save()

            new_store = ChunkStore(Path(tmpdir))
            assert new_store.load() is True
            assert new_store.chunk_count() == 1
            assert new_store.get_chunk(0)["text"] == "test"

    def test_load_returns_false_when_no_file(self):
        """Test load returns False if chunks.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir))
            assert store.load() is False
            assert store.chunk_count() == 0

    def test_lookup_by_text(self):
        """Test finding chunk by exact text match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir))
            chunks = [
                {"text": "find me", "metadata": {"source": "x.pdf"}},
                {"text": "other", "metadata": {"source": "y.pdf"}},
            ]
            store.set_chunks(chunks)
            idx, chunk = store.lookup_by_text("find me")
            assert idx == 0
            assert chunk["metadata"]["source"] == "x.pdf"

            idx, chunk = store.lookup_by_text("not found")
            assert idx == -1
