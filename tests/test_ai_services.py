"""Test local embeddings and Groq-backed RAG answer generation without network calls."""

from types import SimpleNamespace

import numpy as np

from app.config import MAX_COMPLETION_TOKENS, MAX_CONTEXT_CHARS, VECTOR_DIMENSION
from app.embeddings import EmbeddingService
from app.rag_service import RAGService


class FakeEncoder:
    """Mimic the sentence-transformers methods used by the embedding service."""

    def __init__(self, dimension=VECTOR_DIMENSION):
        self.dimension = dimension
        self.encoded_texts = None
        self.encode_options = None

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.encoded_texts = texts
        self.encode_options = kwargs
        return np.ones((len(texts), self.dimension), dtype=np.float32)


class FakeChatCompletions:
    """Capture chat requests and return a deterministic answer."""

    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The login route is defined in the retrieved context."
                    )
                )
            ]
        )


class FakeClient:
    """Provide a fake Groq-compatible chat endpoint for offline tests."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class FakeEmbeddingService:
    """Record query text without loading a local model."""

    def __init__(self):
        self.question = None

    def embed(self, text):
        self.question = text
        return [0.25] * VECTOR_DIMENSION


def test_embedding_service_encodes_text_locally_in_batches():
    """Check local model selection, output dimensions, and batch options."""
    service = EmbeddingService(model="test-local-model")
    encoder = FakeEncoder()
    service._encoder = encoder

    embeddings = service.embed_batch(["source code", "more source code"])

    assert len(embeddings) == 2
    assert all(len(embedding) == VECTOR_DIMENSION for embedding in embeddings)
    assert encoder.encoded_texts == ["source code", "more source code"]
    assert encoder.encode_options["batch_size"] == service.BATCH_SIZE
    assert encoder.encode_options["normalize_embeddings"] is False


def test_embedding_service_rejects_model_dimension_mismatch():
    """Prevent incompatible vectors from being inserted into the FAISS index."""
    service = EmbeddingService(model="wrong-dimension-model")
    service._encoder = FakeEncoder(dimension=VECTOR_DIMENSION - 1)

    try:
        service.embed_batch(["source code"])
        assert False, "Expected dimension mismatch to fail"
    except RuntimeError as exc:
        assert "expected" in str(exc)
        assert str(VECTOR_DIMENSION) in str(exc)


def test_rag_service_generates_answer_with_groq_compatible_chat_api():
    """Check retrieved source citations and Groq-compatible chat request settings."""
    chunk = SimpleNamespace(
        file_path="app/routes.py",
        start_line=10,
        end_line=12,
        content=("def login():\n    return authenticate()\n" * 1000),
    )

    search_requests = []

    def search(query_embedding, top_k):
        search_requests.append((query_embedding, top_k))
        return [{"chunk": chunk, "score": 0.9}]

    repository_ids = []
    vector_store = SimpleNamespace(
        index=SimpleNamespace(ntotal=1),
        search=search,
    )

    def get_repository(repo_id):
        repository_ids.append(repo_id)
        return {"repo_name": "example", "vector_store": vector_store}

    repository_service = SimpleNamespace(get_repository=get_repository)
    embedding_service = FakeEmbeddingService()
    service = RAGService(repository_service, embedding_service, model="openai/gpt-oss-120b")
    client = FakeClient()
    service.client = client

    answer = service.answer_question("repo-id", "How does login work?")

    assert answer.answer == "The login route is defined in the retrieved context."
    assert answer.references[0].file_path == "app/routes.py"
    assert len(answer.references[0].snippet) <= min(600, MAX_CONTEXT_CHARS)
    assert repository_ids == ["repo-id"]
    assert search_requests[0][1] == 5
    assert embedding_service.question == "How does login work?"
    request = client.chat.completions.request
    assert request["model"] == "openai/gpt-oss-120b"
    assert request["messages"][0]["role"] == "system"
    assert request["max_completion_tokens"] == MAX_COMPLETION_TOKENS
    assert len(request["messages"][1]["content"].split("Relevant repository context:\n", 1)[1]) <= MAX_CONTEXT_CHARS
