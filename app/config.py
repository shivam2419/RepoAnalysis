"""Load model, provider, and repository settings from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load secrets and local overrides from the project root without hard-coding them.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Model and client settings shared by the backend and frontend.
APP_TITLE = "RepoAnalysis — AI Codebase Assistant"
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-oss-120b")
AI_ENDPOINT = os.getenv("AI_ENDPOINT", "https://api.groq.com/openai/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Ingestion and retrieval limits help keep indexing within practical resource bounds.
MAX_REPO_FILE_COUNT = int(os.getenv("MAX_REPO_FILE_COUNT", "2000"))
MAX_REPO_BYTES = int(os.getenv("MAX_REPO_BYTES", "157286400"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", "262144"))
CHUNK_SIZE_LINES = int(os.getenv("CHUNK_SIZE_LINES", "200"))
CHUNK_OVERLAP_LINES = int(os.getenv("CHUNK_OVERLAP_LINES", "40"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1400"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "160"))

# Keep this equal to the selected local embedding model's output dimension.
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
