"""Models for clinic-policy documents and chunked policy text."""

from pydantic import BaseModel, ConfigDict, Field


class PolicyDocument(BaseModel):
    """A policy document loaded from a Markdown source file."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(..., min_length=1, description="Stable identifier derived from the source filename.")
    title: str = Field(..., min_length=1, description="The H1 title of the policy document.")
    source_name: str = Field(..., min_length=1, description="User-presentable source label for citations.")
    content: str = Field(..., description="Full Markdown content for the source document.")


class PolicyChunk(BaseModel):
    """A section-level chunk extracted from a clinic policy document."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Stable identifier for this chunk.")
    document_id: str = Field(..., min_length=1, description="Identifier of the source policy document.")
    document_title: str = Field(..., min_length=1, description="The document's H1 title.")
    section_title: str = Field(..., min_length=1, description="The H2 section heading for this chunk.")
    source_name: str = Field(..., min_length=1, description="Filename or citation-friendly source name.")
    text: str = Field(..., description="Body text for the chunk without the H1 title.")
