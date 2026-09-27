"""Embedding-provider boundary for policy retrieval."""

from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI


class EmbeddingError(RuntimeError):
    """Base exception for embedding failures."""


class EmbeddingRequestError(EmbeddingError):
    """Raised when the embedding provider cannot complete the request."""


class EmbeddingResponseError(EmbeddingError):
    """Raised when the embedding provider returns malformed or incomplete data."""


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text while preserving input order."""


class OpenAIEmbeddingProvider:
    """OpenAI-backed embedding provider for deterministic policy retrieval."""

    def __init__(self, client: AsyncOpenAI, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise EmbeddingRequestError("No texts were supplied for embedding.")

        cleaned: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingRequestError(f"Text at index {index} is not a string.")
            if not text.strip():
                raise EmbeddingRequestError(f"Text at index {index} is blank.")
            cleaned.append(text)

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=cleaned,
                encoding_format="float",
            )
        except Exception as exc:  # pragma: no cover - exercised via mocked client in tests
            raise EmbeddingRequestError("Embedding request failed.") from exc

        embeddings = [None] * len(cleaned)
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) == 0:
            raise EmbeddingResponseError("Embedding response was empty.")

        for item in data:
            if item is None:
                raise EmbeddingResponseError("Embedding response item was null.")

            index = getattr(item, "index", None)
            if index is None:
                raise EmbeddingResponseError("Embedding response item is missing its index.")

            if not isinstance(index, int) or index < 0 or index >= len(cleaned):
                raise EmbeddingResponseError("Embedding response item index is out of range.")

            embedding = getattr(item, "embedding", None)
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingResponseError(f"Embedding for input {index} was empty or missing.")
            if any(not isinstance(value, (int, float)) for value in embedding):
                raise EmbeddingResponseError(f"Embedding for input {index} contains a non-numeric value.")

            if embeddings[index] is not None:
                raise EmbeddingResponseError(f"Duplicate embedding returned for input {index}.")
            embeddings[index] = list(float(value) for value in embedding)

        if any(item is None for item in embeddings):
            missing_indices = [index for index, value in enumerate(embeddings) if value is None]
            raise EmbeddingResponseError(f"Missing embedding responses for index(es): {missing_indices}")

        return embeddings
