"""Semantic chunker that splits by document structure."""

import re
from typing import List, Dict, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.chunk import Chunk
from src.config import config
from src.heading_detector import RegexHeadingDetector, HeadingDetector


class SemanticChunker:
    """Chunks documents by semantic structure (headings, sections)."""

    def __init__(
        self, chunk_config=None, heading_detector: Optional[HeadingDetector] = None
    ):
        """Initialize with optional custom config and detector.

        Args:
            chunk_config: Configuration for chunking behavior.
            heading_detector: Heading detection strategy. Uses RegexHeadingDetector if None.
        """
        self.config = chunk_config or config.chunking
        self.heading_detector = heading_detector or RegexHeadingDetector()

    def is_heading(self, line: str) -> bool:
        """Check if a line is a heading."""
        return self.heading_detector.is_heading(line)

    def split_by_headings(self, text: str) -> List[Dict]:
        """Split text into sections at heading boundaries."""
        lines = text.split("\n")
        sections = []
        current_section = {"heading": "", "content": []}

        for line in lines:
            if self.is_heading(line):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {"heading": line.strip(), "content": [line]}
            else:
                current_section["content"].append(line)

        if current_section["content"]:
            sections.append(current_section)

        return sections

    def merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge chunks smaller than min_size with next chunk."""
        min_size = self.config.min_chunk_size
        merged = []
        buffer = {"heading": "", "content": []}

        for chunk in chunks:
            combined = "\n".join(buffer["content"]) + "\n" + "\n".join(chunk["content"])

            if len(combined) >= min_size:
                merged.append(
                    {"heading": buffer.get("heading", ""), "content": buffer["content"]}
                )
                buffer = {
                    "heading": chunk.get("heading", ""),
                    "content": chunk["content"],
                }
            else:
                buffer["content"].extend(chunk["content"])
                buffer["heading"] = chunk.get("heading", "") or buffer["heading"]

        if buffer["content"]:
            merged.append(buffer)

        return merged

    def split_large_chunks(self, chunk: Dict) -> List[Dict]:
        """Split large chunks into smaller pieces at sentence boundaries."""
        max_size = self.config.max_chunk_size
        text = "\n".join(chunk["content"])

        if len(text) <= max_size:
            return [chunk]

        # Split at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > max_size and current_chunk:
                chunks.append(
                    {
                        "heading": chunk.get("heading", ""),
                        "content": [" ".join(current_chunk)],
                    }
                )
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)

        if current_chunk:
            chunks.append(
                {
                    "heading": chunk.get("heading", ""),
                    "content": [" ".join(current_chunk)],
                }
            )

        return chunks

    def create_chunks(self, text: str, base_metadata: Dict) -> List[Chunk]:
        """Create chunks from text with metadata."""
        sections = self.split_by_headings(text)

        # Build initial chunks
        chunks = []
        for section in sections:
            chunks.append(
                {"heading": section["heading"], "content": section["content"]}
            )

        # Merge small chunks
        chunks = self.merge_small_chunks(chunks)

        # Split large chunks
        final_chunks = []
        for chunk in chunks:
            sub_chunks = self.split_large_chunks(chunk)
            final_chunks.extend(sub_chunks)

        # Create Chunk objects with metadata
        result = []
        for idx, chunk in enumerate(final_chunks):
            text_content = "\n".join(chunk["content"])

            # Add overlap with previous chunk
            if idx > 0:
                prev_content_list = final_chunks[idx - 1]["content"]
                prev_text = "\n".join(prev_content_list)
                overlap_text = prev_text[-self.config.overlap_size :]
                text_content = overlap_text + "\n" + text_content

            result.append(
                Chunk(
                    text=text_content.strip(),
                    metadata={
                        **base_metadata,
                        "section": chunk.get("heading", ""),
                        "chunk_id": idx,
                    },
                    chunk_id=idx,
                )
            )

        return result


class LangChainChunker:
    """Adapter wrapping langchain RecursiveTextSplitter to produce Chunk objects."""

    def __init__(
        self,
        chunk_config=None,
        heading_detector=None,  # Kept for API compatibility, not used
    ):
        """Initialize with optional config.

        Args:
            chunk_config: Configuration for chunking behavior.
            heading_detector: Not used (kept for SemanticChunker API compatibility).
        """
        self.config = chunk_config or config.chunking

    def create_chunks(self, text: str, base_metadata: Dict) -> List[Chunk]:
        """Create chunks using langchain RecursiveTextSplitter.

        Args:
            text: Document text to chunk.
            base_metadata: Metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=self.config.overlap_size,
            separators=self.config.separators,
            length_function=len,
        )

        # Convert text to langchain Document for splitting
        doc = Document(page_content=text)
        split_docs = splitter.split_documents([doc])

        result = []
        for idx, split_doc in enumerate(split_docs):
            result.append(
                Chunk(
                    text=split_doc.page_content,
                    metadata={
                        **base_metadata,
                        "chunk_id": idx,
                    },
                    chunk_id=idx,
                )
            )

        return result
