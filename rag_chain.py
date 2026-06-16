"""
rag_chain.py
------------
Core RAG (Retrieval-Augmented Generation) question-answering chain.
Retrieves relevant chunks from the vector store and passes them
along with the user question to Gemini 2.0 Flash.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from retriever import PaperRetriever, RetrievalResult


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """You are an expert AI research assistant specializing in \
analyzing and explaining academic papers.

You are given retrieved context passages from one or more research papers. \
Use ONLY the provided context to answer the question. \
If the answer cannot be found in the context, say so clearly — do not hallucinate.

Always be precise, technical when needed, and cite the paper name and page \
number for every claim you make.

Context:
{context}
"""

QA_USER_PROMPT = """Question: {question}

Please provide a thorough answer based on the context above. \
After your answer, list the sources you used in this format:
Sources:
- "<Paper Title>" — Page <N>
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """Structured output from the RAG QA chain."""
    question: str
    answer: str
    sources: List[str]         # Citation strings
    retrieved_chunks: int      # How many chunks were used


# ---------------------------------------------------------------------------
# RAG Chain
# ---------------------------------------------------------------------------

class RAGChain:
    """
    Orchestrates retrieval + generation for question answering.

    Uses PaperRetriever to fetch relevant context, then calls the
    Gemini 2.0 Flash model to generate a grounded answer.
    """

    def __init__(self):
        config.validate()
        self.retriever = PaperRetriever(k=config.TOP_K_RESULTS)
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def _build_messages(
        self, question: str, filenames: Optional[List[str]] = None
    ):
        """Retrieve context and build the message list. Returns (messages, retrieval)."""
        retrieval: RetrievalResult = self.retriever.retrieve(
            query=question,
            filenames=filenames,
        )
        if not retrieval.chunks:
            return None, retrieval

        system_msg = QA_SYSTEM_PROMPT.format(context=retrieval.context_text)
        user_msg = QA_USER_PROMPT.format(question=question)
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
        return messages, retrieval

    def ask(
        self,
        question: str,
        filenames: Optional[List[str]] = None,
    ) -> QAResult:
        """
        Answer a question using RAG over the stored papers.

        Args:
            question: The user's natural language question.
            filenames: Optional list of paper filenames to restrict retrieval.

        Returns:
            QAResult with the answer, sources, and retrieval stats.

        Raises:
            ValueError: If no papers are in the vector store.
            RuntimeError: If the LLM call fails.
        """
        messages, retrieval = self._build_messages(question, filenames)

        if messages is None:
            return QAResult(
                question=question,
                answer=(
                    "No relevant content was found for your question in the "
                    "selected paper(s). Please try rephrasing or uploading "
                    "additional documents."
                ),
                sources=[],
                retrieved_chunks=0,
            )

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        return QAResult(
            question=question,
            answer=answer_text,
            sources=retrieval.citations,
            retrieved_chunks=len(retrieval.chunks),
        )

    def stream(
        self,
        question: str,
        filenames: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the answer token-by-token for fast perceived latency.

        Yields:
            String chunks as they arrive from the LLM.
        """
        messages, retrieval = self._build_messages(question, filenames)

        if messages is None:
            yield (
                "No relevant content was found for your question in the "
                "selected paper(s). Please try rephrasing or uploading "
                "additional documents."
            )
            return

        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n\n⚠️ Error: {e}"

    def stream_with_sources(
        self,
        question: str,
        filenames: Optional[List[str]] = None,
    ):
        """
        Stream the answer and also return retrieval metadata.

        Returns:
            (generator, retrieval) tuple — iterate the generator for tokens.
        """
        messages, retrieval = self._build_messages(question, filenames)

        if messages is None:
            def _empty():
                yield (
                    "No relevant content was found for your question in the "
                    "selected paper(s). Please try rephrasing or uploading "
                    "additional documents."
                )
            return _empty(), retrieval

        def _stream():
            try:
                for chunk in self.llm.stream(messages):
                    if chunk.content:
                        yield chunk.content
            except Exception as e:
                yield f"\n\n⚠️ Error: {e}"

        return _stream(), retrieval


# ---------------------------------------------------------------------------
# Module-level singleton (avoids re-creating LLM client per request)
# ---------------------------------------------------------------------------

_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Return a shared RAGChain instance (lazy singleton)."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
