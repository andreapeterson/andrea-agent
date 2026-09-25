"""Models for embedded policy vectors and ranked retrieval results."""

from pydantic import BaseModel, ConfigDict, Field

from .policy import PolicyChunk


class EmbeddedPolicyChunk(BaseModel):
    """A policy chunk paired with its embedding vector."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk: PolicyChunk
    embedding: list[float] = Field(..., min_length=1, description="Embedding vector for this policy chunk.")


class PolicySearchResult(BaseModel):
    """A ranked policy search result."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk: PolicyChunk
    similarity_score: float
