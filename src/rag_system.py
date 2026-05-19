"""Main RAG system orchestrator."""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from src.config import config
from src.document_loader import DocumentLoader
from src.chunker import SemanticChunker, Chunk
from src.hybrid_retriever import HybridRetriever, RRFResult
from src.generator import OllamaGenerator, OllamaConnectionError, OllamaAPIError
from src.index_manager import IndexManager, IndexManifest


@dataclass
class RAGQueryResult:
    """Result from a RAG query."""
    answer: str
    sources: List[Dict]
    query: str
    retrieved_chunks: List[RRFResult]


class RAGSystem:
    """Main RAG system orchestrating retrieval and generation.

    Uses chunks.json as single source of truth for all chunk text and metadata.
    """

    def __init__(self, source_dir: Optional[Path] = None, index_dir: Optional[Path] = None):
        """Initialize RAG system with all components."""
        self.source_dir = source_dir or Path("sources")
        self.index_dir = index_dir or Path(".rag_index")

        # Initialize components
        self.document_loader = DocumentLoader(self.source_dir)
        self.chunker = SemanticChunker()
        self.retriever = HybridRetriever()
        self.generator = OllamaGenerator()
        self.index_manager = IndexManager(self.index_dir)

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

    def ingest_documents(self, source_dir: Path = None, force_rebuild: bool = False) -> Dict:
        """Ingest and index all documents.

        Args:
            source_dir: Source directory (default: self.source_dir)
            force_rebuild: If True, rebuild index even if cached version exists

        Returns:
            Dictionary with ingestion statistics and warnings
        """
        source_dir = source_dir or self.source_dir

        if not Path(source_dir).exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # Check if we can use cached index
        if not force_rebuild and self.index_manager.is_index_valid(source_dir):
            manifest = self.index_manager.load_manifest()
            print(f"Loading cached index for {len(manifest.files)} files...")

            # Load cached index
            chunks_path = self.index_dir / "chunks.json"
            vector_path = self.index_dir / "vectors.faiss"
            bm25_path = self.index_dir / "bm25.json"
            if vector_path.exists() and bm25_path.exists() and chunks_path.exists():
                self.retriever.load(str(vector_path), str(bm25_path), str(chunks_path))

            self._indexed = True
            total_chunks = sum(f["chunk_count"] for f in manifest.files.values())
            return {
                "documents_loaded": len(manifest.files),
                "chunks_created": total_chunks,
                "source_dir": str(source_dir),
                "cached": True,
                "empty_pages": {}
            }

        # Dirty check - show what changed
        pdf_files = self.document_loader.get_pdf_files()
        current_files = [f.name for f in pdf_files]
        dirty = self.index_manager.detect_dirty_files(source_dir, current_files)

        if dirty["new"]:
            print(f"New files to index: {dirty['new']}")
        if dirty["modified"]:
            print(f"Modified files to re-index: {dirty['modified']}")
        if dirty["deleted"]:
            print(f"Removed files from index: {dirty['deleted']}")

        # Load documents
        loader = DocumentLoader(source_dir)
        documents, empty_pages_by_file = loader.load_all_pdfs()

        # Warn about empty pages (scanned/image PDFs)
        if empty_pages_by_file:
            for filename, pages in empty_pages_by_file.items():
                print(f"WARNING: {filename} has {len(pages)} empty page(s) - likely a scanned PDF: pages {pages}")

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

        # Index chunks (stores in retriever's chunks list)
        self.retriever.index_documents(all_chunks)

        # Save all indexes and chunks
        self.retriever.save(
            str(self.index_dir / "vectors.faiss"),
            str(self.index_dir / "bm25.json")
        )

        # Save manifest
        manifest = IndexManifest(index_dir=self.index_dir)
        files_to_track = set(doc.metadata["source"] for doc in documents)
        chunk_count_per_file = len(all_chunks) // max(len(files_to_track), 1)
        for filename in files_to_track:
            file_path = source_dir / filename
            if file_path.exists():
                checksum = self.index_manager.file_checksum(file_path)
                manifest.add_file(filename, checksum, chunk_count_per_file)

        self.index_manager.save_manifest(manifest)
        self._indexed = True

        return {
            "documents_loaded": len(files_to_track),
            "chunks_created": len(all_chunks),
            "source_dir": str(source_dir),
            "cached": False,
            "empty_pages": empty_pages_by_file,
            "dirty_files": dirty
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
                answer=f"Connection error: {e}",
                sources=self._format_sources(retrieved_chunks),
                query=question,
                retrieved_chunks=retrieved_chunks
            )
        except OllamaAPIError as e:
            return RAGQueryResult(
                answer=f"API error: {e}",
                sources=self._format_sources(retrieved_chunks),
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
            "embedding_model": self.retriever.embedding_manager.model_name,
            "index_dir": str(self.index_dir)
        }