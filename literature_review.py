"""
literature_review.py
--------------------
Generates a structured literature review from multiple research papers.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from retriever import PaperRetriever


LIT_REVIEW_KEYWORDS = [
    "introduction background overview problem area",
    "existing methods prior work baseline related work survey",
    "research gap open problem unsolved challenge limitation",
    "challenge difficulty obstacle constraint real-world issue",
    "future direction next steps open research potential",
    "conclusion summary contribution impact significance",
]

LIT_REVIEW_SYSTEM_PROMPT = """You are an expert academic writer with deep knowledge \
of machine learning and AI research. You are writing a formal literature review \
based on the following research papers:

{paper_list}

You have been provided with retrieved excerpts from these papers as context. \
Write in a formal academic style. Cite papers by name when making claims. \
Synthesize ideas across papers rather than describing each paper in isolation.

Context:
{context}
"""

LIT_REVIEW_USER_PROMPT = """Write a comprehensive literature review covering the \
research area represented by the provided papers. \
Structure the review with these exact sections:

## Introduction
[Introduce the research area, its importance, and scope of this review. \
Mention the papers reviewed.]

## Existing Methods and Approaches
[Synthesize the methods proposed across the papers. Group similar approaches together. \
Highlight key ideas and compare approaches.]

## Research Gaps
[What problems remain unsolved? What do the authors themselves acknowledge as gaps?]

## Challenges
[What are the practical and theoretical challenges in this research area \
based on what the papers report?]

## Future Directions
[Synthesize the future work sections. What promising directions emerge?]

## Conclusion
[Summarize the state of the field based on these papers. \
What is the overall trajectory of research?]
"""


@dataclass
class LiteratureReviewResult:
    paper_titles: List[str]
    review_text: str
    sources: List[str] = field(default_factory=list)


class LiteratureReviewGenerator:
    def __init__(self):
        config.validate()
        self.retriever = PaperRetriever(k=5)
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.3,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def _build_messages(self, filenames, titles=None):
        if len(filenames) < 2:
            raise ValueError(f"Literature review requires at least 2 papers. Only {len(filenames)} provided.")
        titles = titles or {}
        display_titles = [titles.get(fn, fn) for fn in filenames]
        retrieval = self.retriever.retrieve_for_section(
            section_keywords=LIT_REVIEW_KEYWORDS,
            filenames=filenames, k_per_keyword=3,
        )
        if not retrieval.chunks:
            raise ValueError("No content found for the selected papers.")
        paper_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(display_titles))
        system_msg = LIT_REVIEW_SYSTEM_PROMPT.format(paper_list=paper_list, context=retrieval.context_text)
        messages = [SystemMessage(content=system_msg), HumanMessage(content=LIT_REVIEW_USER_PROMPT)]
        return messages, retrieval, display_titles

    def generate(self, filenames, titles=None):
        messages, retrieval, display_titles = self._build_messages(filenames, titles)
        try:
            response = self.llm.invoke(messages)
            review_text = response.content
        except Exception as e:
            raise RuntimeError(f"Literature review generation failed: {e}") from e
        return LiteratureReviewResult(paper_titles=display_titles, review_text=review_text, sources=retrieval.citations)

    def stream(self, filenames, titles=None):
        messages, retrieval, display_titles = self._build_messages(filenames, titles)
        def _stream():
            try:
                for chunk in self.llm.stream(messages):
                    if chunk.content:
                        yield chunk.content
            except Exception as e:
                yield f"\n\n⚠️ Error: {e}"
        return _stream(), retrieval.citations, display_titles


_lit_review_gen: Optional[LiteratureReviewGenerator] = None

def get_literature_review_generator() -> LiteratureReviewGenerator:
    global _lit_review_gen
    if _lit_review_gen is None:
        _lit_review_gen = LiteratureReviewGenerator()
    return _lit_review_gen
