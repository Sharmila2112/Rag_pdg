"""
config.py
---------
Centralized configuration management for the Research Paper Assistant.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application-wide configuration loaded from environment variables."""

    # --- API Keys ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # --- Model Settings ---
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # --- Storage ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = "research_papers"

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Retrieval ---
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # --- Gemini Generation Settings ---
    TEMPERATURE: float = 0.2
    MAX_OUTPUT_TOKENS: int = 4096
    STREAMING: bool = True

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration values are present."""
        if not cls.GOOGLE_API_KEY or cls.GOOGLE_API_KEY == "your_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Please add your Gemini API key to the .env file."
            )
        # Ensure data directory exists
        os.makedirs(cls.CHROMA_PERSIST_DIR, exist_ok=True)


# Singleton config instance
config = Config()
