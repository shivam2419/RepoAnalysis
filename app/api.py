"""Expose health, repository indexing, question-answering, and reset HTTP endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import APP_TITLE
from app.embeddings import EmbeddingService
from app.models import AnswerResponse, QuestionRequest, RepositoryIndexRequest, RepositoryIndexResponse
from app.rag_service import RAGService
from app.repository_service import RepositoryFetchError, RepositoryService, RepositoryValidationError

app = FastAPI(title=APP_TITLE)

# Share in-memory repository state between the index and question endpoints.
repository_service = RepositoryService(embedding_service=EmbeddingService())
rag_service = RAGService(repository_service=repository_service, embedding_service=repository_service.embedding_service)


@app.get("/health")
def health() -> dict[str, str]:
    """Provide a simple endpoint for checking that the backend is running."""
    return {"status": "ok"}


@app.post("/index", response_model=RepositoryIndexResponse)
def index_repository(payload: RepositoryIndexRequest) -> RepositoryIndexResponse:
    """Index a submitted GitHub URL and map expected failures to HTTP responses."""
    try:
        result = repository_service.index_repository(payload.url)
        return RepositoryIndexResponse(**result)
    except RepositoryValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RepositoryFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest) -> AnswerResponse:
    """Answer a question using the repository identified by the request."""
    if not payload.repo_id:
        raise HTTPException(status_code=400, detail="Repository must be indexed before asking a question.")

    try:
        result = rag_service.answer_question(payload.repo_id, payload.question)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/clear")
def clear_repository(repo_id: str) -> dict[str, str]:
    """Delete the selected in-memory index and its temporary repository clone."""
    repository_service.clear_repository(repo_id)
    return {"status": "cleared", "repo_id": repo_id}
