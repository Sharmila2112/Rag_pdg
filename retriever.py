"""
retriever.py
------------
High-level retrieval interface that wraps the vector store and returns
citation-enriched context bundles for use by the RAG chain and other modules.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from config import config
from vector_store import get_vector_store


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with full citation metadata."""
    content: str
    title: str
    filename: str
    page_number: int
    chunk_id: str
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    """Bundle returned by the retriever: context text + citation list."""
    chunks: List[RetrievedChunk]
    context_text: str          # Formatted context ready to inject into prompts
    citations: List[str]       # Human-readable citation strings


def _doc_to_chunk(doc: Document) -> RetrievedChunk:
    """Convert a LangChain Document into a RetrievedChunk."""
    meta = doc.metadata
    return RetrievedChunk(
        content=doc.page_content,
        title=meta.get("title", meta.get("filename", "Unknown")),
        filename=meta.get("filename", "unknown"),
        page_number=meta.get("page_number", 0),
        chunk_id=meta.get("chunk_id", ""),
        chunk_index=meta.get("chunk_index", 0),
    )


def _build_context(chunks: List[RetrievedChunk]) -> str:
    """
    Format chunks into a numbered context block for LLM prompts.
    Each chunk is labelled with its paper title and page number.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Context {i}] (Paper: \"{chunk.title}\", Page {chunk.page_number})\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


def _build_citations(chunks: List[RetrievedChunk]) -> List[str]:
    """Return a deduplicated list of citation strings."""
    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk.filename, chunk.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(
                f'"{chunk.title}" — Page {chunk.page_number} [{chunk.filename}]'
            )
    return citations


class PaperRetriever:
    """
    Retrieves relevant chunks from the vector store for a given query,
    optionally scoped to specific papers.
    """

    def __init__(self, k: int = config.TOP_K_RESULTS):
        self.k = k
        self.vs = get_vector_store()

    def retrieve(
        self,
        query: str,
        filenames: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Run a similarity search and return a structured RetrievalResult.

        Args:
            query: The user's question or search phrase.
            filenames: Optional list of paper filenames to restrict search.
            k: Override the default top-k if provided.

        Returns:
            RetrievalResult with chunks, formatted context, and citations.
        """
        top_k = k or self.k
        docs = self.vs.similarity_search(query, k=top_k, filenames=filenames)

        chunks = [_doc_to_chunk(doc) for doc in docs]
        context = _build_context(chunks)
        citations = _build_citations(chunks)

        return RetrievalResult(
            chunks=chunks,
            context_text=context,
            citations=citations,
        )

    def retrieve_for_section(
        self,
        section_keywords: List[str],
        filenames: Optional[List[str]] = None,
        k_per_keyword: int = 3,
    ) -> RetrievalResult:
        """
        Retrieve chunks for multiple keyword queries (useful for structured
        summaries where we want results for 'methodology', 'results', etc.).

        Args:
            section_keywords: List of queries/keywords to search.
            filenames: Optional paper filter.
            k_per_keyword: Chunks to fetch per keyword.

        Returns:
            Combined RetrievalResult (deduplicated by chunk_id).
        """
        seen_ids: set = set()
        all_chunks: List[RetrievedChunk] = []

        for keyword in section_keywords:
            result = self.retrieve(keyword, filenames=filenames, k=k_per_keyword)
            for chunk in result.chunks:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        context = _build_context(all_chunks)
        citations = _build_citations(all_chunks)

        return RetrievalResult(
            chunks=all_chunks,
            context_text=context,
            citations=citations,
        )
