"""
app.py
------
Main Streamlit application for the Research Paper Assistant.
Provides a minimal, functional UI over the RAG backend.
"""

import os
import sys
import time
import traceback

import streamlit as st


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Environment / config guard
# ---------------------------------------------------------------------------
from config import config

try:
    config.validate()
    CONFIG_OK = True
except ValueError as e:
    CONFIG_OK = False
    CONFIG_ERROR = str(e)

# ---------------------------------------------------------------------------
# Lazy imports (avoids loading heavy models before API key is confirmed)
# ---------------------------------------------------------------------------
if CONFIG_OK:
    from pdf_processor import PDFProcessor
    from chunking import DocumentChunker
    from vector_store import get_vector_store
    from rag_chain import get_rag_chain
    from summarizer import get_summarizer
    from methodology_explainer import get_methodology_explainer
    from comparison_engine import get_comparison_engine
    from literature_review import get_literature_review_generator
    from utils import (
        save_uploaded_files,
        cleanup_temp_files,
        display_sources,
        paper_selector_widget,
    )

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "uploaded_papers" not in st.session_state:
    st.session_state["uploaded_papers"] = {}

if "processing_log" not in st.session_state:
    st.session_state["processing_log"] = []

# ===========================================================================
# Sidebar: PDF Upload & Paper Management
# ===========================================================================
def render_sidebar():
    """Render the upload panel and paper list in the sidebar."""
    st.sidebar.title("📄 Research Paper Assistant")
    st.sidebar.markdown("---")

    if not CONFIG_OK:
        st.sidebar.error(f"⚠️ Config Error:\n{CONFIG_ERROR}")
        st.sidebar.info("Add your `GOOGLE_API_KEY` to the `.env` file and restart.")
        return

    st.sidebar.header("Upload Papers")
    uploaded = st.sidebar.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports any text-based PDF. Scanned PDFs may not work.",
    )

    if st.sidebar.button("📥 Process & Index PDFs", disabled=(not uploaded)):
        process_pdfs(uploaded)

    # Paper list
    st.sidebar.markdown("---")
    st.sidebar.header("Indexed Papers")

    vs = get_vector_store()
    paper_titles = vs.get_paper_titles()   # {filename: title}

    if not paper_titles:
        st.sidebar.info("No papers indexed yet.")
    else:
        st.session_state["uploaded_papers"] = paper_titles
        for fn, title in paper_titles.items():
            col1, col2 = st.sidebar.columns([4, 1])
            col1.markdown(f"📌 **{title[:35]}{'…' if len(title) > 35 else ''}**")
            if col2.button("🗑️", key=f"del_{fn}", help=f"Remove {fn}"):
                deleted = vs.delete_paper(fn)
                st.sidebar.success(f"Removed {deleted} chunks for '{fn}'")
                st.rerun()

        st.sidebar.markdown(f"_Total chunks: {vs.total_chunks()}_")


def process_pdfs(uploaded_files):
    """Save, extract, chunk, and index uploaded PDFs."""
    processor = PDFProcessor()
    chunker = DocumentChunker()
    vs = get_vector_store()

    saved = save_uploaded_files(uploaded_files)
    temp_paths = list(saved.values())

    progress = st.sidebar.progress(0, text="Starting…")
    log = []

    for i, (original_name, temp_path) in enumerate(saved.items()):
        progress.progress((i) / len(saved), text=f"Processing: {original_name}")
        try:
            # Extract
            pages = processor.process_pdf(temp_path)
            # Override filename in metadata to use original name
            for p in pages:
                p.metadata["filename"] = original_name

            # Chunk
            chunks = chunker.chunk_pages(pages)

            # Index
            added = vs.add_documents(chunks)

            title = pages[0].metadata.get("title", original_name) if pages else original_name
            log.append(f"✅ **{original_name}** — {len(pages)} pages, {added} chunks indexed")

        except Exception as e:
            log.append(f"❌ **{original_name}** — Error: {e}")

    cleanup_temp_files(temp_paths)
    progress.progress(1.0, text="Done!")

    for msg in log:
        st.sidebar.markdown(msg)

    st.session_state["processing_log"] = log
    st.rerun()


# ===========================================================================
# Main Tabs
# ===========================================================================
def render_main():
    """Render the main content area with feature tabs."""
    if not CONFIG_OK:
        st.error(f"⚠️ Configuration Error: {CONFIG_ERROR}")
        st.info("Please add your `GOOGLE_API_KEY` to the `.env` file and restart the app.")
        st.code("GOOGLE_API_KEY=your_key_here", language="bash")
        return

    st.title("📚 Research Paper Assistant")
    st.caption(f"RAG-powered analysis using {config.GEMINI_MODEL} + ChromaDB + BGE Embeddings")

    paper_titles = st.session_state.get("uploaded_papers", {})

    if not paper_titles:
        st.info("👈 Upload one or more PDF papers in the sidebar to get started.")
        return

    tabs = st.tabs([
        "❓ Q&A",
        "📋 Summary",
        "⚙️ Methodology",
        "📊 Compare Papers",
        "📖 Literature Review",
    ])

    with tabs[0]:
        render_qa_tab(paper_titles)
    with tabs[1]:
        render_summary_tab(paper_titles)
    with tabs[2]:
        render_methodology_tab(paper_titles)
    with tabs[3]:
        render_comparison_tab(paper_titles)
    with tabs[4]:
        render_literature_review_tab(paper_titles)


# ---------------------------------------------------------------------------
# Tab: Q&A
# ---------------------------------------------------------------------------
def render_qa_tab(paper_titles: dict):
    st.header("❓ Ask a Question")
    st.markdown("Ask anything about the uploaded papers. Answers include page citations.")

    selected = paper_selector_widget(
        paper_titles,
        label="Scope to specific paper(s) (or leave all selected for cross-paper search)",
        multi=True,
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g. What loss function is used in training?",
        height=100,
    )

    if st.button("🔍 Get Answer", disabled=(not question.strip())):
        start_time = time.time()
        try:
            chain = get_rag_chain()
            token_stream, retrieval = chain.stream_with_sources(
                question, filenames=selected if selected else None
            )

            st.markdown("### Answer")
            response_text = st.write_stream(token_stream)
            elapsed = time.time() - start_time

            if retrieval and retrieval.chunks:
                st.caption(
                    f"_Retrieved {len(retrieval.chunks)} chunk(s) • "
                    f"Answered in {elapsed:.1f}s_"
                )
                display_sources(retrieval.citations, "📎 Citations")
            else:
                st.caption(f"_Completed in {elapsed:.1f}s_")

        except Exception as e:
            st.error(f"Error: {e}")
            with st.expander("Debug"):
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab: Summary
# ---------------------------------------------------------------------------
def render_summary_tab(paper_titles: dict):
    st.header("📋 Structured Summary")
    st.markdown(
        "Generate an 8-section structured summary of any indexed paper."
    )

    selected = paper_selector_widget(paper_titles, label="Select paper to summarise", multi=False)

    if selected and st.button("📝 Generate Summary"):
        filename = selected[0]
        title = paper_titles.get(filename, filename)
        start_time = time.time()

        try:
            summarizer = get_summarizer()
            token_stream, sources = summarizer.stream(
                filename=filename, title=title
            )

            st.markdown(f"## Summary: {title}")
            st.write_stream(token_stream)
            elapsed = time.time() - start_time

            st.caption(f"_Generated in {elapsed:.1f}s_")
            display_sources(sources, "📎 Sources")

        except Exception as e:
            st.error(f"Summary failed: {e}")


# ---------------------------------------------------------------------------
# Tab: Methodology
# ---------------------------------------------------------------------------
def render_methodology_tab(paper_titles: dict):
    st.header("⚙️ Methodology Explainer")
    st.markdown(
        "Get a beginner-friendly, step-by-step explanation of the paper's methodology."
    )

    selected = paper_selector_widget(paper_titles, label="Select paper", multi=False)

    custom_q = st.text_input(
        "Custom methodology question (optional)",
        placeholder="e.g. How does the attention mechanism work in this model?",
    )

    if selected and st.button("💡 Explain Methodology"):
        filename = selected[0]
        title = paper_titles.get(filename, filename)
        start_time = time.time()

        try:
            explainer = get_methodology_explainer()
            token_stream, sources = explainer.stream(
                filename=filename,
                title=title,
                custom_question=custom_q or None,
            )

            st.markdown(f"## Methodology: {title}")
            st.write_stream(token_stream)
            elapsed = time.time() - start_time

            st.caption(f"_Generated in {elapsed:.1f}s_")
            display_sources(sources, "📎 Sources")

        except Exception as e:
            st.error(f"Explanation failed: {e}")


# ---------------------------------------------------------------------------
# Tab: Compare Papers
# ---------------------------------------------------------------------------
def render_comparison_tab(paper_titles: dict):
    st.header("📊 Compare Papers")
    st.markdown(
        "Select two or more papers to generate a structured comparison table."
    )

    if len(paper_titles) < 2:
        st.warning("Please upload and index at least 2 papers to use this feature.")
        return

    selected = paper_selector_widget(
        paper_titles, label="Select papers to compare (min 2)", multi=True
    )

    if len(selected) < 2:
        st.info("Select at least 2 papers above.")
        return

    if st.button("🔀 Generate Comparison"):
        start_time = time.time()
        try:
            engine = get_comparison_engine()
            token_stream, sources, display_titles = engine.stream(
                filenames=selected, titles=paper_titles
            )

            st.markdown("## Paper Comparison")
            st.write_stream(token_stream)
            elapsed = time.time() - start_time

            st.caption(f"_Generated in {elapsed:.1f}s_")
            display_sources(sources, "📎 Sources")

        except Exception as e:
            st.error(f"Comparison failed: {e}")


# ---------------------------------------------------------------------------
# Tab: Literature Review
# ---------------------------------------------------------------------------
def render_literature_review_tab(paper_titles: dict):
    st.header("📖 Literature Review Generator")
    st.markdown(
        "Generate a formal, academic-style literature review synthesizing multiple papers."
    )

    if len(paper_titles) < 2:
        st.warning("Please upload and index at least 2 papers to use this feature.")
        return

    selected = paper_selector_widget(
        paper_titles, label="Select papers to include in review", multi=True
    )

    if len(selected) < 2:
        st.info("Select at least 2 papers above.")
        return

    if st.button("📖 Generate Literature Review"):
        start_time = time.time()
        try:
            generator = get_literature_review_generator()
            token_stream, sources, display_titles = generator.stream(
                filenames=selected, titles=paper_titles
            )

            st.markdown("## Literature Review")
            for t in display_titles:
                st.markdown(f"- _{t}_")
            st.markdown("---")
            st.write_stream(token_stream)
            elapsed = time.time() - start_time

            st.caption(f"_Generated in {elapsed:.1f}s_")
            display_sources(sources, "📎 Sources")

        except Exception as e:
            st.error(f"Literature review failed: {e}")


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
