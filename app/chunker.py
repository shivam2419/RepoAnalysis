"""Split source files into overlapping chunks while retaining line-number metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodeChunk:
    """Represent one source excerpt and its retrieval metadata."""

    repo_id: str
    repo_name: str
    file_path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    embedding: List[float] = field(default_factory=list)


def chunk_document(
    repo_id: str,
    repo_name: str,
    file_path: str,
    content: str,
    chunk_size_lines: int,
    overlap_lines: int,
) -> list[CodeChunk]:
    """Split file text into line-based chunks while preserving file metadata."""
    lines = content.splitlines()
    if not lines:
        return []

    # Keep a positive step so overlap never causes an infinite chunking loop.
    chunk_size_lines = max(1, chunk_size_lines)
    overlap_lines = max(0, overlap_lines)
    step = max(1, chunk_size_lines - overlap_lines)

    chunks: list[CodeChunk] = []
    for index, start in enumerate(range(0, len(lines), step)):
        end = min(start + chunk_size_lines, len(lines))
        snippet = "\n".join(lines[start:end])
        if not snippet.strip():
            continue
        # Store 1-based inclusive line ranges so citations match editor line numbers.
        chunks.append(
            CodeChunk(
                repo_id=repo_id,
                repo_name=repo_name,
                file_path=file_path,
                chunk_index=index,
                start_line=start + 1,
                end_line=end,
                content=snippet,
            )
        )
    return chunks
