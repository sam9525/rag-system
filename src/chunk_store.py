"""ChunkStore — single source of truth for chunk data."""

import json
from pathlib import Path
from typing import Any


class ChunkStore:
    """Owns chunk data as single source of truth — both in-memory and persisted.

    All retrieval and generation code accesses chunks through this interface.
    This decouples data storage from retrieval logic.
    """

    def __init__(self, index_dir: Path):
        self._chunks: list[dict[str, Any]] = []
        self._index_dir = Path(index_dir)

    def set_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Set chunks from the processing pipeline."""
        self._chunks = chunks

    def get_chunk(self, index: int) -> dict[str, Any]:
        """Get a single chunk by its index."""
        return self._chunks[index]

    def get_all(self) -> list[dict[str, Any]]:
        """Get all chunks."""
        return self._chunks

    def save(self) -> None:
        """Persist chunks to chunks.json in the index directory."""
        path = self._index_dir / "chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False)

    def load(self) -> bool:
        """Load chunks from chunks.json.

        Returns:
            True if chunks.json was found and loaded, False otherwise.
        """
        path = self._index_dir / "chunks.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        return True

    def chunk_count(self) -> int:
        """Number of chunks currently stored."""
        return len(self._chunks)

    def lookup_by_text(self, text: str) -> tuple[int, dict[str, Any]]:
        """Find chunk index and data by exact text match.

        Used by the reranker to recover metadata after neural reranking.

        Returns:
            Tuple of (index, chunk_dict) or (-1, {}) if not found.
        """
        for idx, chunk in enumerate(self._chunks):
            if chunk["text"] == text:
                return idx, chunk
        return -1, {}
