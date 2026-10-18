"""Create local semantic embeddings for repository code and user questions."""

from __future__ import annotations

from app.config import EMBEDDING_MODEL, VECTOR_DIMENSION


class EmbeddingService:
    """Load a sentence-transformers model locally and encode text in batches."""

    BATCH_SIZE = 32

    def __init__(self, model: str | None = None):
        """Remember the local model name; load model weights on the first request."""
        self.model = model or EMBEDDING_MODEL
        self._encoder = None

    def _get_encoder(self):
        """Load the local model once and confirm its vector size matches FAISS."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Local embeddings require sentence-transformers. Install project dependencies with "
                    "`python -m pip install -r requirements.txt`."
                ) from exc

            try:
                encoder = SentenceTransformer(self.model, device="cpu")
            except OSError as exc:
                raise RuntimeError(
                    f"Could not load local embedding model '{self.model}'. Check the model name and network access "
                    "for its first download."
                ) from exc

            actual_dimension = encoder.get_embedding_dimension()
            if actual_dimension != VECTOR_DIMENSION:
                raise RuntimeError(
                    f"Embedding model '{self.model}' returns {actual_dimension} dimensions, but "
                    f"VECTOR_DIMENSION is {VECTOR_DIMENSION}. Update .env to match the model."
                )
            self._encoder = encoder

        return self._encoder

    def embed(self, text: str) -> list[float]:
        """Embed a single text through the same validated batch path."""
        if not text or not text.strip():
            return [0.0] * VECTOR_DIMENSION

        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode texts locally in bounded batches and preserve their input ordering."""
        if not texts:
            return []
        encoder = self._get_encoder()
        vectors = encoder.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        if vectors.ndim != 2 or vectors.shape != (len(texts), VECTOR_DIMENSION):
            raise RuntimeError(
                f"Local embedding model returned shape {vectors.shape}; expected "
                f"({len(texts)}, {VECTOR_DIMENSION})."
            )
        return vectors.tolist()
