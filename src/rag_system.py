"""Main RAG system orchestrator."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from src.config import config
from src.document_loader import DocumentLoader
from src.chunker import SemanticChunker, Chunk
from src.hybrid_retriever import HybridRetriever, RRFResult
from src.generator import OllamaGenerator, OllamaConnectionError


@dataclass
class RAGQueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[Dict]
    query: str
    retrieved_chunks: List[RRFResult]


class RAGSystem:
    """Main RAG system orchestrating retrieval and generation."""

    def __init__(self, source_dir: Optional[Path] = None):
        """Initialize RAG system with all components."""
        self.source_dir = source_dir or Path("sources")

        # Initialize components
        self.document_loader = DocumentLoader(self.source_dir)
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever()
        self.generator = OllamaGenerator()

        self._indexed = False

    def _transform_chunks_for_generator(self, chunks: List[RRFResult]) -> List[Dict]:
        """Transform RRFResult chunks into format expected by generator."""
        return [
            {
                "text": chunk.text,
                "score": chunk.score,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]

    def _format_sources(self, chunks: List[RRFResult]) -> List[Dict]:
        """Format retrieved chunks into sources list for display."""
        return [
            {
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "?"),
                "section": chunk.metadata.get("section", ""),
                "score": chunk.score
            }
            for chunk in chunks
        ]

    def ingest_documents(self, source_dir: Path = None) -> Dict:
        """Ingest and index all documents.

        Returns:
            Dictionary with ingestion statistics
        """
        source_dir = source_dir or self.source_dir

        if not Path(source_dir).exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # Load documents
        documents = self.document_loader.load_all_pdfs()

        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.create_chunks(
                doc.page_content,
                doc.metadata
            )
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata
                })

        # Index chunks
        self.retriever.index_documents(all_chunks)
        self._indexed = True

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(all_chunks),
            "source_dir": str(source_dir)
        }

    def query(self, question: str, top_k: int = 3) -> RAGQueryResult:
        """Query the RAG system.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve (default 3)

        Returns:
            RAGQueryResult with answer and sources
        """
        if not self._indexed:
            raise ValueError(
                "No documents indexed. Call ingest_documents() first."
            )

        # Retrieve chunks
        retrieved_chunks = self.retriever.search(
            question,
            final_top_k=top_k
        )

        # Transform chunks for generator
        chunks_for_gen = self._transform_chunks_for_generator(retrieved_chunks)

        # Generate answer
        try:
            answer = self.generator.generate(question, chunks_for_gen)
        except OllamaConnectionError as e:
            return RAGQueryResult(
                answer=f"Error: {e}",
                sources=[],
                query=question,
                retrieved_chunks=retrieved_chunks
            )

        # Format sources for display
        sources = self._format_sources(retrieved_chunks)

        return RAGQueryResult(
            answer=answer,
            sources=sources,
            query=question,
            retrieved_chunks=retrieved_chunks
        )

    def is_indexed(self) -> bool:
        """Check if documents are indexed."""
        return self._indexed

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            "indexed": self._indexed,
            "document_count": self.retriever.count(),
            "model": self.generator.config.model,
            "embedding_model": self.retriever.embedding_manager.model_name
        }
