"""
embeddings.py
-------------
Wraps the BAAI/bge-small-en-v1.5 Sentence Transformer model as a
LangChain-compatible Embeddings class so it can be plugged directly
into ChromaDB or any other LangChain vector store.
"""

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import config


class BGEEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper around SentenceTransformer.

    BGE models benefit from a short instruction prefix for queries
    (not for documents). This is handled automatically.
    """

    # BGE-specific instruction prefix for queries (improves retrieval quality)
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        print(f"[Embeddings] Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        print(f"[Embeddings] Model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.
        Documents are embedded WITHOUT the query instruction prefix.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (List[float]).
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # Cosine similarity works on L2-normalised vecs
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string WITH the BGE instruction prefix.

        Args:
            text: Query string.

        Returns:
            Embedding vector as List[float].
        """
        prefixed = self.QUERY_INSTRUCTION + text
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# Module-level singleton – instantiated once and reused
_embeddings_instance: BGEEmbeddings | None = None


def get_embeddings() -> BGEEmbeddings:
    """Return a shared BGEEmbeddings instance (lazy singleton)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = BGEEmbeddings()
    return _embeddings_instance
