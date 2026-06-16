"""
pdf_processor.py
----------------
Handles PDF ingestion and text extraction using PyMuPDF (fitz).
Extracts text page-by-page and attaches rich metadata to each page.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


@dataclass
class PageDocument:
    """Represents a single page extracted from a PDF."""
    page_content: str
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """
    Processes PDF files and extracts text with metadata.

    Attributes:
        min_page_chars: Minimum characters required to consider a page non-empty.
    """

    def __init__(self, min_page_chars: int = 50):
        self.min_page_chars = min_page_chars

    def extract_title(self, doc: fitz.Document, filename: str) -> str:
        """
        Attempt to extract a meaningful title from the PDF metadata
        or fall back to the first non-empty line of page 1.
        """
        # Try PDF metadata title first
        meta_title = doc.metadata.get("title", "").strip()
        if meta_title and len(meta_title) > 3:
            return meta_title

        # Fall back: grab the first substantial line from page 0
        if doc.page_count > 0:
            first_page_text = doc[0].get_text("text").strip()
            lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
            for line in lines[:10]:
                # Heuristic: title lines are usually short-ish and capitalised
                if 10 < len(line) < 200:
                    return line

        # Last resort: use filename without extension
        return Path(filename).stem.replace("_", " ").replace("-", " ").title()

    def clean_text(self, text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        # Replace multiple newlines with two newlines max
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        # Remove null bytes and other control chars (keep \n and \t)
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
        return text.strip()

    def process_pdf(self, file_path: str) -> List[PageDocument]:
        """
        Extract text from all pages of a PDF.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of PageDocument objects, one per non-empty page.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If PyMuPDF cannot open the file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        filename = path.name

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{filename}': {e}") from e

        title = self.extract_title(doc, filename)
        pages: List[PageDocument] = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            raw_text = page.get_text("text")
            cleaned = self.clean_text(raw_text)

            # Skip essentially empty pages (headers/footers only)
            if len(cleaned) < self.min_page_chars:
                continue

            page_doc = PageDocument(
                page_content=cleaned,
                metadata={
                    "title": title,
                    "filename": filename,
                    "page_number": page_num + 1,   # 1-indexed for readability
                    "total_pages": doc.page_count,
                    "source": str(path.resolve()),
                },
            )
            pages.append(page_doc)

        doc.close()

        if not pages:
            raise ValueError(
                f"No extractable text found in '{filename}'. "
                "The PDF may be scanned/image-based."
            )

        return pages

    def process_multiple_pdfs(
        self, file_paths: List[str]
    ) -> List[PageDocument]:
        """
        Process multiple PDFs and return a combined list of PageDocuments.

        Args:
            file_paths: List of paths to PDF files.

        Returns:
            Combined list of PageDocuments from all PDFs.
        """
        all_pages: List[PageDocument] = []
        errors: List[str] = []

        for fp in file_paths:
            try:
                pages = self.process_pdf(fp)
                all_pages.extend(pages)
            except Exception as e:
                errors.append(f"  • {Path(fp).name}: {e}")

        if errors:
            error_msg = "Errors during PDF processing:\n" + "\n".join(errors)
            if not all_pages:
                raise RuntimeError(error_msg)
            # Partial success – log but continue
            print(f"[WARNING] {error_msg}")

        return all_pages
