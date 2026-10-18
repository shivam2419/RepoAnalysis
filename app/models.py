"""Pydantic request and response models used by the FastAPI endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RepositoryIndexRequest(BaseModel):
    """Validate the GitHub URL submitted for indexing."""

    url: str = Field(..., min_length=1)


class RepositoryIndexStats(BaseModel):
    """Summarize source files and code chunks produced during indexing."""

    repo_name: str
    files_discovered: int
    files_indexed: int
    files_skipped: int
    chunks_created: int


class RepositoryIndexResponse(BaseModel):
    """Return the indexed repository identity and ingestion statistics."""

    repo_id: str
    repo_url: str
    repo_name: str
    status: str
    stats: RepositoryIndexStats


class QuestionRequest(BaseModel):
    """Carry a user question and the repository it refers to."""

    question: str = Field(..., min_length=1)
    repo_id: Optional[str] = None


class Reference(BaseModel):
    """Identify a retrieved code excerpt and its original source location."""

    file_path: str
    start_line: int
    end_line: int
    snippet: str
    score: Optional[float] = None


class AnswerResponse(BaseModel):
    """Return a generated answer together with its supporting code references."""

    answer: str
    references: List[Reference] = Field(default_factory=list)
    repo_id: str
    repository_name: str
    used_context: bool = True

    @field_validator("references")
    @classmethod
    def ensure_references_are_list(cls, value):
        # Normalize explicit null references to the same shape as an empty result.
        if value is None:
            return []
        return value
