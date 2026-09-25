"""Pure cosine-similarity ranking for policy chunks."""

from __future__ import annotations

import math

from app.integrations.embeddings import EmbeddingProvider
from app.models.policy import PolicyChunk
from app.models.retrieval import EmbeddedPolicyChunk, PolicySearchResult


class PolicyRetrievalError(RuntimeError):
    """Base error for policy retrieval validation and search failures."""


class InvalidVectorError(PolicyRetrievalError):
    """Raised when a retrieval vector is empty, mismatched, or otherwise invalid."""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity between two non-zero vectors."""
    if not left or not right:
        raise InvalidVectorError("Cosine similarity requires non-empty vectors.")
    if len(left) != len(right):
        raise InvalidVectorError("Cosine similarity requires vectors with matching dimensions.")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_magnitude = math.sqrt(sum(value * value for value in left))
    right_magnitude = math.sqrt(sum(value * value for value in right))

    if left_magnitude == 0 or right_magnitude == 0:
        raise InvalidVectorError("Cosine similarity requires non-zero magnitude vectors.")

    return dot_product / (left_magnitude * right_magnitude)


def _chunk_embedding_text(chunk: PolicyChunk) -> str:
    """Build the stable text used to create an embedding for a chunk."""
    return f"{chunk.document_title}\n{chunk.section_title}\n{chunk.text}"


class PolicyRetriever:
    """In-memory policy retriever that ranks chunks by cosine similarity."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedding_provider = embedding_provider
        self._index: list[EmbeddedPolicyChunk] = []

    @property
    def indexed_chunks(self) -> list[EmbeddedPolicyChunk]:
        return list(self._index)

    async def index(self, chunks: list[PolicyChunk]) -> None:
        if not chunks:
            raise PolicyRetrievalError("Cannot index an empty chunk list.")

        seen_ids: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                raise PolicyRetrievalError(f"Duplicate chunk ID detected: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)

        candidate_texts = [_chunk_embedding_text(chunk) for chunk in chunks]
        vectors = await self._embedding_provider.embed_texts(candidate_texts)
        if len(vectors) != len(chunks):
            raise PolicyRetrievalError("Embedding provider returned the wrong number of vectors.")

        dims: int | None = None
        next_index: list[EmbeddedPolicyChunk] = []
        for chunk, vector in zip(chunks, vectors):
            if not vector:
                raise PolicyRetrievalError(f"Embedding for chunk {chunk.chunk_id} was empty.")
            if dims is None:
                dims = len(vector)
            elif len(vector) != dims:
                raise PolicyRetrievalError(f"Embedding size mismatch for chunk {chunk.chunk_id}.")
            next_index.append(EmbeddedPolicyChunk(chunk=chunk, embedding=vector))

        self._index = next_index

    async def search(self, query: str, limit: int = 3) -> list[PolicySearchResult]:
        if not query or not query.strip():
            raise PolicyRetrievalError("Search query cannot be blank.")
        if limit < 1:
            raise PolicyRetrievalError("Search limit must be at least 1.")
        if not self._index:
            raise PolicyRetrievalError("No chunks have been indexed yet.")

        query_vector = (await self._embedding_provider.embed_texts([query.strip()]))[0]
        if not query_vector:
            raise PolicyRetrievalError("Embedded query vector was empty.")

        scored: list[PolicySearchResult] = []
        for embedded in self._index:
            score = cosine_similarity(query_vector, embedded.embedding)
            scored.append(PolicySearchResult(chunk=embedded.chunk, similarity_score=score))

        scored.sort(key=lambda result: (-result.similarity_score, result.chunk.chunk_id))
        return scored[:limit]
