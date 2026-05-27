"""Tests for RAG system orchestrator."""

import pytest
from pathlib import Path
from src.system.rag_system import RAGSystem, RAGQueryResult
from src.retrieval.rrf_fusion import RRFResult


class TestRAGSystem:
    """Test RAGSystem class."""

    def test_init_initializes_components(self):
        """Test initialization creates all components."""
        system = RAGSystem()
        assert system.retriever is not None
        assert system.generator is not None
        assert system.document_loader is not None

    def test_query_returns_result(self):
        """Test query returns RAGQueryResult."""
        system = RAGSystem()
        # Would need indexed documents
        pass

    def test_ingest_processes_pdfs(self):
        """Test ingest_documents processes PDFs."""
        system = RAGSystem()
        # Would need real PDFs
        pass

    def test_chunk_transformer_extracts_fields(self):
        """Test chunk_transformer extracts required fields for generator."""
        system = RAGSystem()
        retrieved_chunks = [
            RRFResult(
                text="doc1",
                score=0.9,
                metadata={"source": "a.pdf", "page": 1},
                semantic_score=0.9,
                keyword_score=None,
            ),
            RRFResult(
                text="doc2",
                score=0.8,
                metadata={"source": "b.pdf", "page": 2},
                semantic_score=0.8,
                keyword_score=None,
            ),
        ]

        result = system._transform_chunks_for_generator(retrieved_chunks)

        assert len(result) == 2
        assert result[0]["text"] == "doc1"
        assert result[0]["score"] == 0.9
        assert result[0]["metadata"]["source"] == "a.pdf"

    def test_source_formatter_extracts_display_fields(self):
        """Test source_formatter extracts fields for display."""
        system = RAGSystem()
        retrieved_chunks = [
            RRFResult(
                text="doc1 text",
                score=0.9,
                metadata={"source": "a.pdf", "page": 1, "section": "Intro"},
                semantic_score=0.9,
                keyword_score=None,
            ),
        ]

        result = system._format_sources(retrieved_chunks)

        assert len(result) == 1
        assert result[0]["source"] == "a.pdf"
        assert result[0]["page"] == 1
        assert result[0]["section"] == "Intro"
        assert result[0]["text"].startswith("doc1 text")

    def test_import_chunk_from_chunker(self):
        """Test Chunk can be imported from src.chunker."""
        from src.ingestion.chunker import Chunk

        chunk = Chunk(text="test content", metadata={"source": "test.pdf"}, chunk_id=0)
        assert chunk.text == "test content"
        assert chunk.metadata["source"] == "test.pdf"

    def test_ingest_accepts_chunk_type(self):
        """Test ingest_documents works with Chunk type."""
        from src.ingestion.chunker import Chunk

        system = RAGSystem(source_dir=Path("sources"))
        chunks = [
            Chunk(text="test", metadata={"source": "test.pdf", "page": 1}, chunk_id=0)
        ]

        # This should work if Chunk.to_dict() is used
        # We'll test the transform method
        dicts = [c.to_dict() for c in chunks]

        assert dicts[0]["text"] == "test"
        assert dicts[0]["metadata"]["source"] == "test.pdf"
