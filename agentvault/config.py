"""Configuration settings for the RAG application."""

import os
from pathlib import Path

# Data directories
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Partitioned data directories
GOOGLE_DRIVE_INDEX_DIR = DATA_DIR / "google_drive_index"
PDF_EXTRACTIONS_DIR = DATA_DIR / "pdf_extractions"
CONTENT_PHRASES_DIR = DATA_DIR / "content_phrases"
CONTENT_SUMMARIES_DIR = DATA_DIR / "content_summaries"
TITLE_CARDS_DIR = DATA_DIR / "title_cards"

# Gutenberg texts
GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/32154/pg32154.txt",
    "https://www.gutenberg.org/cache/epub/31547/pg31547.txt",
]

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Google Drive settings
GOOGLE_DRIVE_INDEX_FILE = "google_drive_index.parquet"
SUPPORTED_MIME_TYPES = [
    "text/plain",
    "text/csv",
    "application/json",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
]

# BookWyrm API settings
BOOKWYRM_API_URL = "https://api.bookwyrm.ai/classify"  # Placeholder

# LanceDB configuration
LANCEDB_URI = os.getenv("LANCEDB_URI", str(DATA_DIR / "lancedb"))
TITLES_TABLE = "title_cards"
EMBEDDING_DIMENSION = 1536  # For OpenAI text-embedding-3-small

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

# Partitioned directories will be created as needed
