"""Tests for index persistence and dirty tracking."""

import pytest
import tempfile
import hashlib
from pathlib import Path
from src.index_manager import IndexManager, IndexManifest


class TestIndexManifest:
    """Test IndexManifest structure."""

    def test_manifest_stores_file_info(self, tmp_path):
        """Test manifest can store file paths and checksums."""
        manifest = IndexManifest(index_dir=tmp_path)
        manifest.add_file("doc.pdf", "abc123", 10)
        manifest.add_file("doc2.pdf", "def456", 5)

        assert "doc.pdf" in manifest.files
        assert manifest.files["doc.pdf"]["checksum"] == "abc123"
        assert manifest.files["doc.pdf"]["chunk_count"] == 10

    def test_manifest_to_dict_roundtrip(self, tmp_path):
        """Test manifest serializes and deserializes correctly."""
        manifest = IndexManifest(index_dir=tmp_path)
        manifest.add_file("doc.pdf", "abc123", 10)
        manifest.add_file("doc2.pdf", "def456", 5)

        data = manifest.to_dict()
        restored = IndexManifest.from_dict(data, index_dir=tmp_path)

        assert restored.files == manifest.files


class TestIndexManager:
    """Test IndexManager for persistence and dirty tracking."""

    def test_save_and_load_manifest(self, tmp_path):
        """Test manifest saves and loads from disk."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        manifest = IndexManifest(index_dir=index_dir)
        manifest.add_file("doc.pdf", "abc123", 10)

        manager = IndexManager(index_dir=index_dir)
        manager.save_manifest(manifest)

        loaded = manager.load_manifest()
        assert loaded is not None
        assert "doc.pdf" in loaded.files
        assert loaded.files["doc.pdf"]["checksum"] == "abc123"

    def test_dirty_detection_new_file(self, tmp_path):
        """Test that new files are detected as dirty."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create initial manifest
        (pdf_dir / "doc1.pdf").write_text("content1")

        manifest = IndexManifest(index_dir=index_dir)
        manifest.add_file("doc1.pdf", self._checksum("content1"), 5)

        manager = IndexManager(index_dir=index_dir)
        manager.save_manifest(manifest)

        # Add a new file
        (pdf_dir / "doc2.pdf").write_text("content2")

        # Should detect doc2.pdf as new
        dirty = manager.detect_dirty_files(pdf_dir, ["doc1.pdf", "doc2.pdf"])
        assert "doc2.pdf" in dirty["new"]
        assert "doc1.pdf" not in dirty["new"]

    def test_dirty_detection_modified_file(self, tmp_path):
        """Test that modified files are detected as dirty."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        file_path = pdf_dir / "doc.pdf"
        file_path.write_text("original content")

        manifest = IndexManifest(index_dir=index_dir)
        manifest.add_file("doc.pdf", self._checksum("original content"), 5)

        manager = IndexManager(index_dir=index_dir)
        manager.save_manifest(manifest)

        # Modify the file
        file_path.write_text("modified content")

        dirty = manager.detect_dirty_files(pdf_dir, ["doc.pdf"])
        assert "doc.pdf" in dirty["modified"]

    def test_dirty_detection_deleted_file(self, tmp_path):
        """Test that deleted files are detected as dirty."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        manifest = IndexManifest(index_dir=index_dir)
        manifest.add_file("doc.pdf", "abc123", 5)
        manifest.add_file("doc2.pdf", "def456", 3)

        manager = IndexManager(index_dir=index_dir)
        manager.save_manifest(manifest)

        dirty = manager.detect_dirty_files(tmp_path / "pdfs", ["doc.pdf"])
        assert "doc2.pdf" in dirty["deleted"]

    @staticmethod
    def _checksum(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
