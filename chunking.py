"""
chunking.py
-----------
Splits page-level documents into smaller, overlapping chunks suitable
for embedding and retrieval, while preserving all source metadata.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config
from pdf_processor import PageDocument


class DocumentChunker:
    """
    Converts PageDocument objects into LangChain Document chunks.

    Uses RecursiveCharacterTextSplitter so that splits respect
    paragraph → sentence → word boundaries in that priority order.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_pages(self, pages: List[PageDocument]) -> List[Document]:
        """
        Split a list of PageDocuments into LangChain Document chunks.

        Each chunk inherits the parent page's metadata and also receives
        a unique chunk_id composed of filename + page + chunk index.

        Args:
            pages: List of PageDocument objects from PDFProcessor.

        Returns:
            List of LangChain Document objects ready for embedding.
        """
        all_chunks: List[Document] = []

        for page in pages:
            # Convert PageDocument → LangChain Document
            lc_doc = Document(
                page_content=page.page_content,
                metadata=page.metadata.copy(),
            )

            # Split the page text into sub-chunks
            sub_chunks = self.splitter.split_documents([lc_doc])

            # Attach chunk-level metadata
            filename = page.metadata.get("filename", "unknown")
            page_num = page.metadata.get("page_number", 0)

            for idx, chunk in enumerate(sub_chunks):
                chunk.metadata["chunk_id"] = (
                    f"{filename}__page{page_num}__chunk{idx}"
                )
                chunk.metadata["chunk_index"] = idx
                all_chunks.append(chunk)

        return all_chunks

    def chunk_single_paper(self, pages: List[PageDocument]) -> List[Document]:
        """Convenience wrapper: chunk pages for a single paper."""
        return self.chunk_pages(pages)
