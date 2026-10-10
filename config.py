"""Central config: loads credentials and shared paths from .env / environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
VALIDATION_DIR = DATA_DIR / "validation"
TOPICS_DIR = DATA_DIR / "topics"
DISCOVERY_DIR = DATA_DIR / "discovery"

for d in (DATA_DIR, RAW_DIR, EXTRACTED_DIR, VALIDATION_DIR, TOPICS_DIR, DISCOVERY_DIR):
    d.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def require_gemini_credentials() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env. Get a free key (no card required) at "
            "https://aistudio.google.com/apikey, then copy .env.example to .env and fill it in. "
            "(You can reuse the same key from fibro-nlp-pipeline's .env if you already have one.)"
        )
