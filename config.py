import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class Config:
    GOOGLE_API_KEY = GOOGLE_API_KEY

    CHROMA_PERSIST_DIR = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./data/chroma_db"
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.0-flash"
    )

    CHUNK_SIZE = int(
        os.getenv("CHUNK_SIZE", "1000")
    )

    CHUNK_OVERLAP = int(
        os.getenv("CHUNK_OVERLAP", "200")
    )

    TOP_K_RESULTS = int(
        os.getenv("TOP_K_RESULTS", "5")
    )

    def validate(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add your Gemini API key."
            )


config = Config()