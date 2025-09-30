"""Configuration settings for the RAG application."""

import os
from pathlib import Path

# Data directories
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

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

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "europe-west4-gcp")
TITLES_DENSE_INDEX = "titles-dense"
TITLES_SPARSE_INDEX = "titles-sparse"
EMBEDDING_DIMENSION = 384  # For sentence-transformers/all-MiniLM-L6-v2

# Create directories
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)
