"""Tests for RAG system orchestrator."""

import pytest
from pathlib import Path
from src.rag_system import RAGSystem, RAGQueryResult
from src.hybrid_retriever import RRFResult


class TestRAGSystem:
    """Test RAGSystem class."""

    def test_init_initializes_components(self):
        """Test initialization creates all components."""
        system = RAGSystem()
        assert system.retriever is not None
        assert system.generator is not None
        assert system.document_loader is not None
        assert system.chunker is not None

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
            RRFResult(text="doc1", score=0.9, metadata={"source": "a.pdf", "page": 1}, semantic_score=0.9, keyword_score=None),
            RRFResult(text="doc2", score=0.8, metadata={"source": "b.pdf", "page": 2}, semantic_score=0.8, keyword_score=None),
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
            RRFResult(text="doc1 text", score=0.9, metadata={"source": "a.pdf", "page": 1, "section": "Intro"}, semantic_score=0.9, keyword_score=None),
        ]

        result = system._format_sources(retrieved_chunks)

        assert len(result) == 1
        assert result[0]["source"] == "a.pdf"
        assert result[0]["page"] == 1
        assert result[0]["section"] == "Intro"
        assert result[0]["text"].startswith("doc1 text")
