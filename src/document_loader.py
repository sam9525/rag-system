"""Document loader using PyMuPDF."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Document:
    """Represents a loaded document chunk."""
    page_content: str
    metadata: Dict[str, any]


class DocumentLoader:
    """Loads and processes PDF documents."""

    def __init__(self, source_dir: Path):
        """Initialize with source directory."""
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

    def load_pdf(self, pdf_path: Path) -> List[Document]:
        """Load a single PDF and return documents by page."""
        documents = []

        try:
            with fitz.open(str(pdf_path)) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num + 1,
                                "total_pages": len(doc),
                                "section": ""
                            }
                        ))
        except Exception as e:
            raise ValueError(f"Failed to load PDF {pdf_path}: {e}")

        return documents

    def load_all_pdfs(self) -> List[Document]:
        """Load all PDFs from source directory."""
        all_documents = []

        for pdf_file in self.source_dir.glob("*.pdf"):
            documents = self.load_pdf(pdf_file)
            all_documents.extend(documents)

        return all_documents

    def get_pdf_files(self) -> List[Path]:
        """Get list of PDF files in source directory."""
        return list(self.source_dir.glob("*.pdf"))