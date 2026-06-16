"""
vector_store.py
---------------
Manages the ChromaDB vector store: adding documents, listing papers,
deleting papers, and retrieving the Chroma retriever object.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import config
from embeddings import get_embeddings


class VectorStoreManager:
    """
    Handles all interactions with the persistent ChromaDB vector store.

    The store is keyed by a single collection (config.CHROMA_COLLECTION_NAME).
    Papers are identified by their 'filename' metadata field, making it
    easy to delete or filter by individual paper.
    """

    def __init__(self):
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        self._embeddings = get_embeddings()
        self._store: Optional[Chroma] = None
        self._init_store()

    def _init_store(self) -> None:
        """Initialise or load the Chroma collection from disk."""
        self._store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._init_store()
        return self._store

    # ------------------------------------------------------------------
    # Write Operations
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add a list of chunked documents to the vector store.

        Duplicate chunk_ids are handled by Chroma's upsert behaviour.

        Args:
            documents: LangChain Document objects with metadata.

        Returns:
            Number of documents added.
        """
        if not documents:
            return 0

        # Use chunk_id as the Chroma document ID for idempotent upserts
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]

        self.store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def delete_paper(self, filename: str) -> int:
        """
        Remove all chunks belonging to a specific paper from the store.

        Args:
            filename: The PDF filename as stored in metadata.

        Returns:
            Number of chunks deleted.
        """
        collection = self.store._collection
        results = collection.get(where={"filename": filename})
        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # Read / Query Operations
    # ------------------------------------------------------------------

    def list_papers(self) -> List[str]:
        """
        Return a deduplicated list of filenames currently in the store.

        Returns:
            Sorted list of paper filenames.
        """
        collection = self.store._collection
        results = collection.get(include=["metadatas"])
        filenames = {
            meta.get("filename", "unknown")
            for meta in results.get("metadatas", [])
            if meta
        }
        return sorted(filenames)

    def get_paper_titles(self) -> dict:
        """
        Return a mapping of filename -> title for all stored papers.

        Returns:
            Dict[filename, title]
        """
        collection = self.store._collection
        results = collection.get(include=["metadatas"])
        mapping = {}
        for meta in results.get("metadatas", []):
            if meta:
                fn = meta.get("filename", "unknown")
                title = meta.get("title", fn)
                mapping[fn] = title
        return mapping

    def get_retriever(self, filenames: Optional[List[str]] = None, k: int = config.TOP_K_RESULTS):
        """
        Return a LangChain retriever, optionally filtered to specific papers.

        Args:
            filenames: If provided, restrict retrieval to these papers.
            k: Number of top chunks to retrieve per query.

        Returns:
            LangChain BaseRetriever
        """
        search_kwargs = {"k": k}

        if filenames:
            if len(filenames) == 1:
                search_kwargs["filter"] = {"filename": filenames[0]}
            else:
                # Chroma supports $in for multi-value filter
                search_kwargs["filter"] = {"filename": {"$in": filenames}}

        return self.store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = config.TOP_K_RESULTS,
        filenames: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Direct similarity search returning Document objects.

        Args:
            query: Natural language query.
            k: Number of results.
            filenames: Optional filter to specific papers.

        Returns:
            List of relevant Document chunks.
        """
        filter_dict = None
        if filenames:
            if len(filenames) == 1:
                filter_dict = {"filename": filenames[0]}
            else:
                filter_dict = {"filename": {"$in": filenames}}

        return self.store.similarity_search(query, k=k, filter=filter_dict)

    def total_chunks(self) -> int:
        """Return total number of chunks stored across all papers."""
        return self.store._collection.count()


# Module-level singleton
_vs_manager: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """Return a shared VectorStoreManager instance (lazy singleton)."""
    global _vs_manager
    if _vs_manager is None:
        _vs_manager = VectorStoreManager()
    return _vs_manager
