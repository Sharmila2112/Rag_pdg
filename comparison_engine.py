"""
comparison_engine.py
--------------------
Generates a structured comparison table across multiple research papers.
Extracts: Problem, Model, Dataset, Accuracy/Metrics, Advantages, Limitations.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from retriever import PaperRetriever


COMPARISON_KEYWORDS = [
    "research problem objective contribution",
    "model architecture proposed method approach",
    "dataset benchmark evaluation data",
    "accuracy performance results metrics BLEU F1 ROUGE",
    "advantages benefits strengths over baseline",
    "limitations weaknesses shortcomings future",
]

COMPARISON_SYSTEM_PROMPT = """You are a senior research analyst. \
You are given context passages retrieved from multiple research papers. \
Each context block is labelled with its paper title and page number.

Your task is to compare the papers across key dimensions. \
Be factual, use numbers where available, and keep each cell concise (1-3 sentences max).

Papers to compare:
{paper_list}

Context from all papers:
{context}
"""

COMPARISON_USER_PROMPT = """Generate a structured comparison of all the papers above.

First, produce a markdown comparison table with these columns:
| Paper | Problem | Proposed Model/Method | Dataset | Key Results/Accuracy | Advantages | Limitations |

Then, below the table, write a short paragraph (3-5 sentences) highlighting:
- The most significant differences between the approaches
- Which paper appears most novel or impactful and why
- Suggested reading order for someone new to this area
"""


@dataclass
class ComparisonResult:
    """Output from the comparison engine."""
    paper_titles: List[str]
    comparison_text: str
    sources: List[str] = field(default_factory=list)


class ComparisonEngine:
    def __init__(self):
        config.validate()
        self.retriever = PaperRetriever(k=4)
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def _build_messages(self, filenames, titles=None):
        if len(filenames) < 2:
            raise ValueError("Comparison requires at least 2 papers.")
        titles = titles or {}
        display_titles = [titles.get(fn, fn) for fn in filenames]
        retrieval = self.retriever.retrieve_for_section(
            section_keywords=COMPARISON_KEYWORDS,
            filenames=filenames, k_per_keyword=3,
        )
        if not retrieval.chunks:
            raise ValueError("No content found for the selected papers.")
        paper_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(display_titles))
        system_msg = COMPARISON_SYSTEM_PROMPT.format(paper_list=paper_list, context=retrieval.context_text)
        messages = [SystemMessage(content=system_msg), HumanMessage(content=COMPARISON_USER_PROMPT)]
        return messages, retrieval, display_titles

    def compare(self, filenames, titles=None):
        messages, retrieval, display_titles = self._build_messages(filenames, titles)
        try:
            response = self.llm.invoke(messages)
            comparison_text = response.content
        except Exception as e:
            raise RuntimeError(f"Comparison generation failed: {e}") from e
        return ComparisonResult(paper_titles=display_titles, comparison_text=comparison_text, sources=retrieval.citations)

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


_comparison_engine: Optional[ComparisonEngine] = None

def get_comparison_engine() -> ComparisonEngine:
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = ComparisonEngine()
    return _comparison_engine
