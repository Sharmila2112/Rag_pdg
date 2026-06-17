import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Read from Streamlit Secrets first, then .env
GOOGLE_API_KEY = ""

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class Config:
    GOOGLE_API_KEY = GOOGLE_API_KEY
    GEMINI_MODEL = "gemini-2.0-flash"

    def validate(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please add your Gemini API key."
            )


config = Config()