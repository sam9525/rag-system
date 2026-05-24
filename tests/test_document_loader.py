"""Tests for document loader."""

import pytest
from pathlib import Path
import tempfile
from src.document_loader import DocumentLoader, Document


class TestDocumentLoader:
    """Test DocumentLoader class."""

    def test_init_with_valid_path(self, tmp_path):
        """Test initialization with valid path."""
        loader = DocumentLoader(tmp_path)
        assert loader.source_dir == tmp_path

    def test_load_pdf_extracts_text(self, tmp_path):
        """Test PDF text extraction."""
        # This test requires a real PDF - use fixtures for real test
        pass

    def test_document_dataclass(self):
        """Test Document dataclass."""
        doc = Document(
            page_content="Sample text",
            metadata={"source": "test.pdf", "page": 1, "section": "Test"},
        )
        assert doc.page_content == "Sample text"
        assert doc.metadata["source"] == "test.pdf"
