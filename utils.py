"""
utils.py
--------
Shared utility functions used across the application:
  - Saving uploaded PDFs to disk
  - Formatting citation lists for display
  - Building paper title mappings
  - Streamlit session state helpers
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to a temporary file on disk.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Absolute path to the saved temp file.
    """
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def save_uploaded_files(uploaded_files) -> Dict[str, str]:
    """
    Save multiple uploaded files and return a mapping of original
    filename -> temp file path.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Dict[original_filename, temp_path]
    """
    saved: Dict[str, str] = {}
    for uf in uploaded_files:
        tmp_path = save_uploaded_file(uf)
        saved[uf.name] = tmp_path
    return saved


def cleanup_temp_files(paths: List[str]) -> None:
    """Remove temporary files from disk."""
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def format_citations(citations: List[str]) -> str:
    """
    Format a list of citation strings into a numbered markdown list.

    Args:
        citations: List of citation strings.

    Returns:
        Markdown-formatted numbered list as a string.
    """
    if not citations:
        return "_No citations available._"
    return "\n".join(f"{i+1}. {c}" for i, c in enumerate(citations))


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display in UI previews."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def get_session_state(key: str, default=None):
    """Safely get a value from st.session_state."""
    return st.session_state.get(key, default)


def set_session_state(key: str, value) -> None:
    """Safely set a value in st.session_state."""
    st.session_state[key] = value


def init_session_defaults(defaults: dict) -> None:
    """
    Initialise multiple session state keys if they don't already exist.

    Args:
        defaults: Dict of key -> default_value
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_sources(citations: List[str], expander_label: str = "📎 Sources") -> None:
    """
    Display citations inside a Streamlit expander.

    Args:
        citations: List of citation strings.
        expander_label: Label for the expander widget.
    """
    if not citations:
        return
    with st.expander(expander_label):
        for i, c in enumerate(citations, start=1):
            st.markdown(f"**{i}.** {c}")


def paper_selector_widget(
    paper_options: Dict[str, str],
    label: str = "Select paper(s)",
    multi: bool = False,
) -> List[str]:
    """
    Render a Streamlit selectbox or multiselect for choosing papers.

    Args:
        paper_options: Dict[filename, display_title]
        label: Widget label.
        multi: If True, use multiselect; otherwise selectbox.

    Returns:
        List of selected filenames.
    """
    if not paper_options:
        st.warning("No papers uploaded yet. Upload PDFs in the sidebar.")
        return []

    display_names = list(paper_options.values())
    filenames = list(paper_options.keys())

    if multi:
        selected_titles = st.multiselect(label, options=display_names, default=display_names)
        return [fn for fn, title in paper_options.items() if title in selected_titles]
    else:
        selected_title = st.selectbox(label, options=display_names)
        for fn, title in paper_options.items():
            if title == selected_title:
                return [fn]
        return []
