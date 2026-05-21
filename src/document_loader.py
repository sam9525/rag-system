"""Document loader using PyMuPDF."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Document:
    """Represents a loaded document chunk."""

    page_content: str
    metadata: Dict[str, Any]


@dataclass
class LoadResult:
    """Result from loading a PDF with warning tracking."""

    documents: List[Document]
    empty_pages: List[int]
    total_pages: int


class DocumentLoader:
    """Loads and processes PDF documents."""

    def __init__(self, source_dir: Path):
        """Initialize with source directory."""
        self.source_dir = Path(source_dir)
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

    def load_pdf(self, pdf_path: Path) -> LoadResult:
        """Load a single PDF and return documents by page.

        Returns:
            LoadResult with documents, list of empty page numbers, and total pages.
        """
        documents = []
        empty_pages = []

        try:
            with fitz.open(str(pdf_path)) as doc:
                total_pages = len(doc)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": pdf_path.name,
                                    "page": page_num + 1,
                                    "total_pages": total_pages,
                                    "section": "",
                                },
                            )
                        )
                    else:
                        empty_pages.append(page_num + 1)

                return LoadResult(
                    documents=documents,
                    empty_pages=empty_pages,
                    total_pages=total_pages,
                )
        except Exception as e:
            raise ValueError(f"Failed to load PDF {pdf_path}: {e}")

    def load_all_pdfs(self) -> Tuple[List[Document], Dict[str, List[int]]]:
        """Load all PDFs from source directory.

        Returns:
            Tuple of (all_documents, empty_pages_by_file)
        """
        all_documents = []
        empty_pages_by_file = {}

        for pdf_file in self.source_dir.glob("*.pdf"):
            result = self.load_pdf(pdf_file)
            all_documents.extend(result.documents)
            if result.empty_pages:
                empty_pages_by_file[pdf_file.name] = result.empty_pages

        return all_documents, empty_pages_by_file

    def get_pdf_files(self) -> List[Path]:
        """Get list of PDF files in source directory."""
        return list(self.source_dir.glob("*.pdf"))
