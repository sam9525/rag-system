"""Semantic chunker that splits by document structure."""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.config import config


@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    metadata: Dict
    chunk_id: int = 0


class SemanticChunker:
    """Chunks documents by semantic structure (headings, sections)."""

    HEADING_PATTERNS = [
        r'^#{1,3}\s+(.+)$',  # Markdown headings
        r'^([A-Z][A-Z\s]{3,})\s*$',  # ALL CAPS headings
        r'^(\d+\.\s+[A-Z].+)$',  # Numbered sections
    ]

    def __init__(self, chunk_config=None):
        """Initialize with optional custom config."""
        self.config = chunk_config or config.chunking

    def is_heading(self, line: str) -> bool:
        """Check if a line is a heading."""
        line = line.strip()
        for pattern in self.HEADING_PATTERNS:
            if re.match(pattern, line, re.MULTILINE):
                return True
        return False

    def split_by_headings(self, text: str) -> List[Dict]:
        """Split text into sections at heading boundaries."""
        lines = text.split('\n')
        sections = []
        current_section = {"heading": "", "content": []}

        for line in lines:
            if self.is_heading(line):
                if current_section["content"]:
                    sections.append(current_section)
                current_section = {
                    "heading": line.strip(),
                    "content": [line]
                }
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
                if buffer["content"]:
                    merged.append(buffer)
                buffer = {"heading": chunk.get("heading", ""), "content": chunk["content"]}
            else:
                buffer["content"].extend(chunk["content"])

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
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            if current_size + len(sentence) > max_size and current_chunk:
                chunks.append({
                    "heading": chunk.get("heading", ""),
                    "content": [" ".join(current_chunk)]
                })
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)

        if current_chunk:
            chunks.append({
                "heading": chunk.get("heading", ""),
                "content": [" ".join(current_chunk)]
            })

        return chunks

    def create_chunks(self, text: str, base_metadata: Dict) -> List[Chunk]:
        """Create chunks from text with metadata."""
        sections = self.split_by_headings(text)

        # Build initial chunks
        chunks = []
        for section in sections:
            chunks.append({
                "heading": section["heading"],
                "content": section["content"]
            })

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
                prev_text = "\n".join(final_chunks[idx - 1]["content"])
                overlap_text = prev_text[-self.config.overlap_size:]
                text_content = overlap_text + "\n" + text_content

            result.append(Chunk(
                text=text_content.strip(),
                metadata={
                    **base_metadata,
                    "section": chunk.get("heading", ""),
                    "chunk_id": idx
                },
                chunk_id=idx
            ))

        return result
