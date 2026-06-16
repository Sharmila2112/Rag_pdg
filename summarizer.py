"""
summarizer.py
-------------
Generates structured 8-section research summaries from a paper's content.
Sections: Research Problem, Objective, Methodology, Dataset, Experiments,
Results, Limitations, Future Work.
"""

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import config
from retriever import PaperRetriever


# ---------------------------------------------------------------------------
# Section definitions – used for both retrieval and structured output
# ---------------------------------------------------------------------------

SUMMARY_SECTIONS = [
    ("research_problem",  "research problem statement motivation challenge"),
    ("objective",         "research objectives goals aims contributions"),
    ("methodology",       "methodology methods approach framework architecture"),
    ("dataset",           "dataset data corpus benchmark evaluation data"),
    ("experiments",       "experiments experimental setup implementation details"),
    ("results",           "results performance accuracy metrics findings"),
    ("limitations",       "limitations shortcomings weaknesses drawbacks"),
    ("future_work",       "future work directions open problems next steps"),
]

SECTION_LABELS = {
    "research_problem": "🔍 Research Problem",
    "objective":        "🎯 Objective",
    "methodology":      "⚙️ Methodology",
    "dataset":          "📊 Dataset Used",
    "experiments":      "🧪 Experiments",
    "results":          "📈 Results",
    "limitations":      "⚠️ Limitations",
    "future_work":      "🔭 Future Work",
}

SUMMARY_SYSTEM_PROMPT = """You are an expert AI research analyst. \
You will be given retrieved passages from a research paper. \
Your task is to generate a structured summary of the paper based ONLY on the \
provided context. Be concise, precise, and factual. \
If information for a section is not available in the context, write "Not explicitly mentioned."

The paper title is: {title}

Context from the paper:
{context}
"""

SUMMARY_USER_PROMPT = """Generate a structured research summary with exactly these sections. \
Use markdown headers for each section:

## Research Problem
[What problem does the paper address? What is the motivation?]

## Objective
[What are the specific research goals and contributions?]

## Methodology
[What methods, models, or frameworks are proposed or used?]

## Dataset Used
[What datasets or data sources are used?]

## Experiments
[How were the experiments set up and conducted?]

## Results
[What were the main quantitative/qualitative findings?]

## Limitations
[What limitations or shortcomings are acknowledged?]

## Future Work
[What future research directions are suggested?]
"""


@dataclass
class SummaryResult:
    """Structured output from the summarizer."""
    filename: str
    title: str
    full_summary: str          # Raw markdown from Gemini
    sources: List[str] = field(default_factory=list)


class Summarizer:
    """
    Generates structured 8-section summaries of research papers.
    Retrieves sections individually to maximise coverage of the paper.
    """

    def __init__(self):
        config.validate()
        self.retriever = PaperRetriever(k=4)
        self.llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.1,   # Low temperature for factual summaries
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
        )

    def _build_messages(self, filename: str, title: str):
        """Retrieve context and build messages. Returns (messages, retrieval)."""
        keywords = [kw for _, kw in SUMMARY_SECTIONS]
        retrieval = self.retriever.retrieve_for_section(
            section_keywords=keywords,
            filenames=[filename],
            k_per_keyword=2,   # Reduced from 3 → 2 for faster context processing
        )

        if not retrieval.chunks:
            raise ValueError(
                f"No content found for paper '{filename}'. "
                "Ensure it has been processed and indexed."
            )

        system_msg = SUMMARY_SYSTEM_PROMPT.format(
            title=title,
            context=retrieval.context_text,
        )
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=SUMMARY_USER_PROMPT),
        ]
        return messages, retrieval

    def summarize(
        self,
        filename: str,
        title: Optional[str] = None,
    ) -> SummaryResult:
        """
        Generate a structured summary for a single paper.

        Args:
            filename: The PDF filename as stored in the vector store.
            title: Optional human-readable title (falls back to filename).

        Returns:
            SummaryResult containing the full markdown summary and citations.
        """
        display_title = title or filename
        messages, retrieval = self._build_messages(filename, display_title)

        try:
            response = self.llm.invoke(messages)
            summary_text = response.content
        except Exception as e:
            raise RuntimeError(f"Summary generation failed: {e}") from e

        return SummaryResult(
            filename=filename,
            title=display_title,
            full_summary=summary_text,
            sources=retrieval.citations,
        )

    def stream(
        self,
        filename: str,
        title: Optional[str] = None,
    ):
        """
        Stream the summary token-by-token. Returns (generator, sources).
        """
        display_title = title or filename
        messages, retrieval = self._build_messages(filename, display_title)

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

_summarizer: Optional[Summarizer] = None


def get_summarizer() -> Summarizer:
    """Return a shared Summarizer instance (lazy singleton)."""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer
