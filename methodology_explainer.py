"""
methodology_explainer.py
------------------------
Retrieves methodology-related sections from a paper and explains
the technical content in beginner-friendly, step-by-step language.
"""

from dataclasses import dataclass, field
from typing import Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from retriever import PaperRetriever


# ---------------------------------------------------------------------------
# Keywords used to pull methodology-relevant chunks
# ---------------------------------------------------------------------------

METHODOLOGY_KEYWORDS = [
    "methodology method approach framework architecture model",
    "algorithm training procedure steps pipeline workflow",
    "proposed method technical approach system design",
    "implementation details model architecture layers",
    "training fine-tuning optimization loss function",
]

EXPLAINER_SYSTEM_PROMPT = """You are an expert AI teacher who specializes in \
explaining complex machine learning and research concepts to beginners. \
You never use jargon without explaining it first. You use analogies, \
plain language, and numbered steps.

You are given context passages from a research paper titled: "{title}"

Context:
{context}
"""

EXPLAINER_USER_PROMPT = """Please explain the methodology of this research paper \
in a beginner-friendly way.

Structure your explanation as follows:

## What is this paper trying to do?
[1-2 sentence high-level overview]

## The Core Idea (Simple Analogy)
[Explain the key idea using a real-world analogy anyone can understand]

## Step-by-Step: How It Works
[Number each step. Use plain English. Explain any technical term the first time you use it]

## Why This Approach?
[Why did the authors choose this method over simpler alternatives?]

## Key Takeaway
[One sentence: what is the clever/novel part of this methodology?]
"""


@dataclass
class MethodologyResult:
    """Output from the methodology explainer."""
    filename: str
    title: str
    explanation: str
    sources: List[str] = field(default_factory=list)


class MethodologyExplainer:
    """
    Fetches methodology sections and converts them into accessible explanations.
    """

    def __init__(self):
        config.validate()
        self.retriever = PaperRetriever(k=5)
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.3,   # Slightly higher for readable, engaging prose
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def _build_messages(self, filename: str, title: str, custom_question: Optional[str] = None):
        """Retrieve context and build messages. Returns (messages, retrieval)."""
        retrieval = self.retriever.retrieve_for_section(
            section_keywords=METHODOLOGY_KEYWORDS,
            filenames=[filename],
            k_per_keyword=2,
        )

        if not retrieval.chunks:
            raise ValueError(
                f"No methodology content found for '{filename}'. "
                "The paper may not contain detailed method sections."
            )

        system_msg = EXPLAINER_SYSTEM_PROMPT.format(
            title=title,
            context=retrieval.context_text,
        )
        user_msg = custom_question if custom_question else EXPLAINER_USER_PROMPT
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
        return messages, retrieval

    def explain(
        self,
        filename: str,
        title: Optional[str] = None,
        custom_question: Optional[str] = None,
    ) -> MethodologyResult:
        """
        Explain the methodology of a paper in simple terms.

        Args:
            filename: PDF filename in the vector store.
            title: Optional display title.
            custom_question: If provided, answer this specific methodology
                             question instead of generating the standard explanation.

        Returns:
            MethodologyResult with the explanation and citations.
        """
        display_title = title or filename
        messages, retrieval = self._build_messages(filename, display_title, custom_question)

        try:
            response = self.llm.invoke(messages)
            explanation = response.content
        except Exception as e:
            raise RuntimeError(f"Methodology explanation failed: {e}") from e

        return MethodologyResult(
            filename=filename,
            title=display_title,
            explanation=explanation,
            sources=retrieval.citations,
        )

    def stream(
        self,
        filename: str,
        title: Optional[str] = None,
        custom_question: Optional[str] = None,
    ):
        """
        Stream the explanation token-by-token. Returns (generator, sources).
        """
        display_title = title or filename
        messages, retrieval = self._build_messages(filename, display_title, custom_question)

        def _stream() -> Generator[str, None, None]:
            try:
                for chunk in self.llm.stream(messages):
                    if chunk.content:
                        yield chunk.content
            except Exception as e:
                yield f"\n\n⚠️ Error: {e}"

        return _stream(), retrieval.citations


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_explainer: Optional[MethodologyExplainer] = None


def get_methodology_explainer() -> MethodologyExplainer:
    """Return a shared MethodologyExplainer instance (lazy singleton)."""
    global _explainer
    if _explainer is None:
        _explainer = MethodologyExplainer()
    return _explainer
