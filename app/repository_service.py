"""Validate GitHub URLs, extract safe source files, and build repository indexes."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from app.chunker import chunk_document
from app.config import CHUNK_OVERLAP_LINES, CHUNK_SIZE_LINES, MAX_FILE_BYTES, MAX_REPO_BYTES, MAX_REPO_FILE_COUNT, VECTOR_DIMENSION
from app.embeddings import EmbeddingService
from app.faiss_store import FAISSStore

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".nox",
    ".idea",
    ".vscode",
    ".next",
    "coverage",
    "tmp",
    "cache",
    ".cache",
    ".gradle",
    "bin",
    "obj",
}

# These names are excluded to avoid reading credentials, generated data, and lockfiles.
IGNORED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.example",
    ".gitignore",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.crt",
    "*.csr",
    "*.sqlite",
    "*.db",
}

TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".html",
    ".css",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".ini",
    ".cfg",
    ".config",
    ".xml",
    ".csv",
}


class RepositoryValidationError(ValueError):
    """Raised when a repository URL is malformed or unsupported."""


class RepositoryFetchError(RuntimeError):
    """Raised when a repository cannot be cloned."""


def validate_repository_url(url: str) -> str:
    """Accept only HTTPS URLs pointing to a GitHub owner/repository path."""
    if not url or not url.strip():
        raise RepositoryValidationError("Repository URL is required.")

    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise RepositoryValidationError("Only public GitHub HTTPS URLs are supported.")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise RepositoryValidationError("Use the format https://github.com/<owner>/<repo>.")

    owner = path_parts[0]
    repo_name = path_parts[1].removesuffix(".git")
    if not owner or not repo_name:
        raise RepositoryValidationError("Repository owner and name are required.")

    # Normalize the URL so an optional .git suffix or extra path cannot affect clone input.
    return f"https://github.com/{owner}/{repo_name}"


class RepositoryService:
    """Clone repositories, read source files, and index them for retrieval."""

    def __init__(self, embedding_service: EmbeddingService | None = None):
        """Use the supplied embedding client or configure one from environment settings."""
        self.embedding_service = embedding_service or EmbeddingService()
        self._repositories: dict[str, dict] = {}

    def index_repository(self, repo_url: str) -> dict:
        """Clone a repository, filter and chunk text files, then build its FAISS index."""
        # Validate before using the URL in a subprocess argument list.
        normalized_url = validate_repository_url(repo_url)
        repo_name = normalized_url.rstrip("/").split("/")[-1]
        # A stable ID lets the UI refer to the in-memory index without exposing a filesystem path.
        repo_id = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:12]

        if repo_id in self._repositories:
            self.clear_repository(repo_id)

        temp_root = Path(tempfile.mkdtemp(prefix="repoanalysis_"))
        repo_path = temp_root / repo_name

        try:
            # Pass Git arguments separately; repository code is cloned but never executed.
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", normalized_url, str(repo_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise RepositoryFetchError(f"Unable to clone repository: {exc.stderr.strip() or str(exc)}") from exc
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise RepositoryFetchError("Repository cloning timed out.") from exc

        discovered_files = 0
        indexed_files = 0
        skipped_files = 0
        total_repo_size = 0
        all_chunks = []
        # Scan only allowlisted text extensions and enforce the configured ingestion limits.
        for file_path in sorted(repo_path.rglob("*")):
            if file_path.is_dir():
                continue

            relative_path = file_path.relative_to(repo_path).as_posix()
            path_parts = file_path.parts
            if any(part in IGNORED_DIRECTORIES for part in path_parts):
                skipped_files += 1
                continue

            file_name = file_path.name.lower()
            if file_name in {".env", ".env.local", ".env.production", ".env.example"}:
                skipped_files += 1
                continue

            extension = file_path.suffix.lower()
            if extension not in TEXT_EXTENSIONS and file_path.name.lower() not in {"readme", "license", "makefile", "dockerfile"}:
                skipped_files += 1
                continue

            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_BYTES:
                skipped_files += 1
                continue

            total_repo_size += file_size
            if total_repo_size > MAX_REPO_BYTES:
                shutil.rmtree(repo_path, ignore_errors=True)
                raise RepositoryFetchError("Repository exceeds the configured size limit.")

            discovered_files += 1
            if discovered_files > MAX_REPO_FILE_COUNT:
                shutil.rmtree(repo_path, ignore_errors=True)
                raise RepositoryFetchError("Repository exceeds the configured file-count limit.")

            try:
                # Strict UTF-8 decoding skips binary or malformed text instead of indexing it.
                text = file_path.read_bytes().decode("utf-8")
            except (UnicodeDecodeError, OSError):
                skipped_files += 1
                continue

            if not text.strip():
                skipped_files += 1
                continue

            chunks = chunk_document(
                repo_id=repo_id,
                repo_name=repo_name,
                file_path=relative_path,
                content=text,
                chunk_size_lines=CHUNK_SIZE_LINES,
                overlap_lines=CHUNK_OVERLAP_LINES,
            )

            if chunks:
                indexed_files += 1
            all_chunks.extend(chunks)

        chunk_embeddings = self.embedding_service.embed_batch(
            [chunk.content for chunk in all_chunks],
        )
        # Ensure each source chunk receives exactly one vector before building the index.
        if len(chunk_embeddings) != len(all_chunks):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(all_chunks)}, received {len(chunk_embeddings)}."
            )
        for chunk, embedding in zip(all_chunks, chunk_embeddings):
            chunk.embedding = embedding

        vector_store = FAISSStore(dimension=VECTOR_DIMENSION)
        vector_store.add_chunks(all_chunks)

        # Keep repository files, vectors, and statistics together for later questions.
        repo_entry = {
            "repo_id": repo_id,
            "repo_url": normalized_url,
            "repo_name": repo_name,
            "repo_path": repo_path,
            "chunks": all_chunks,
            "vector_store": vector_store,
            "stats": {
                "repo_name": repo_name,
                "files_discovered": discovered_files,
                "files_indexed": indexed_files,
                "files_skipped": skipped_files,
                "chunks_created": len(all_chunks),
            },
        }
        self._repositories[repo_id] = repo_entry

        return {
            "repo_id": repo_id,
            "repo_url": normalized_url,
            "repo_name": repo_name,
            "status": "indexed",
            "stats": repo_entry["stats"],
        }

    def get_repository(self, repo_id: str) -> dict | None:
        """Return an indexed repository entry, or None when the ID is unknown."""
        return self._repositories.get(repo_id)

    def clear_repository(self, repo_id: str) -> None:
        """Remove the in-memory index and its temporary clone."""
        repo_entry = self._repositories.get(repo_id)
        if repo_entry:
            repo_path = repo_entry.get("repo_path")
            if repo_path and repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            self._repositories.pop(repo_id, None)
