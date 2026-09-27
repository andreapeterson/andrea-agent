"""Deterministic loading and section chunking for clinic policy documents."""

import re
from pathlib import Path

from app.models.policy import PolicyChunk, PolicyDocument


class PolicyLoadError(RuntimeError):
    """Raised when a policy document corpus is missing or malformed."""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def load_policy_documents(policy_directory: Path) -> list[PolicyDocument]:
    """Load and validate all Markdown policy documents in a directory."""
    if not policy_directory.exists() or not policy_directory.is_dir():
        raise PolicyLoadError(f"Policy directory does not exist: {policy_directory}")

    markdown_files = sorted(policy_directory.glob("*.md"), key=lambda path: path.name)
    if not markdown_files:
        raise PolicyLoadError(f"No Markdown policy documents found in: {policy_directory}")

    documents: list[PolicyDocument] = []
    seen_ids: set[str] = set()

    for path in markdown_files:
        document_id = path.stem
        if document_id in seen_ids:
            raise PolicyLoadError(f"Duplicate document ID detected: {document_id}")
        seen_ids.add(document_id)

        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise PolicyLoadError(f"Empty document: {path.name}")

        title_match = re.search(r"^#\s+(.*?)\s*$", content, flags=re.MULTILINE)
        if title_match is None:
            raise PolicyLoadError(f"Missing H1 title in document: {path.name}")

        title = title_match.group(1).strip()
        if not title:
            raise PolicyLoadError(f"Missing H1 title in document: {path.name}")

        documents.append(
            PolicyDocument(
                document_id=document_id,
                title=title,
                source_name=path.name,
                content=content,
            )
        )

    return documents


def chunk_policy_document(document: PolicyDocument) -> list[PolicyChunk]:
    """Chunk a single policy document by H2 section, preserving a short overview when present."""
    if not document.content.strip():
        raise PolicyLoadError(f"Empty document: {document.source_name}")

    after_h1 = re.sub(r"^#\s+.*?\n+", "", document.content, count=1, flags=re.MULTILINE)
    body = after_h1.strip()
    if not body:
        return []

    matches = list(re.finditer(r"^##\s+(.*)$", body, flags=re.MULTILINE))
    chunks: list[PolicyChunk] = []
    seen_chunk_ids: set[str] = set()

    if matches:
        first_heading_start = matches[0].start()
        overview_text = body[:first_heading_start].strip()
        if overview_text:
            overview_id = f"{document.document_id}::overview"
            seen_chunk_ids.add(overview_id)
            chunks.append(
                PolicyChunk(
                    chunk_id=overview_id,
                    document_id=document.document_id,
                    document_title=document.title,
                    section_title="Overview",
                    source_name=document.source_name,
                    text=overview_text,
                )
            )

        for index, match in enumerate(matches):
            section_title = match.group(1).strip()
            if not section_title:
                continue

            section_start = match.end()
            section_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            section_text = body[section_start:section_end].strip()
            if not section_text:
                continue

            chunk_id = f"{document.document_id}::{_slugify(section_title)}"
            if chunk_id in seen_chunk_ids:
                raise PolicyLoadError(f"Duplicate chunk ID detected: {chunk_id}")
            seen_chunk_ids.add(chunk_id)

            chunks.append(
                PolicyChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    document_title=document.title,
                    section_title=section_title,
                    source_name=document.source_name,
                    text=section_text,
                )
            )
        return chunks

    if not body.strip():
        return []

    chunk_id = f"{document.document_id}::overview"
    if chunk_id in seen_chunk_ids:
        raise PolicyLoadError(f"Duplicate chunk ID detected: {chunk_id}")
    seen_chunk_ids.add(chunk_id)
    chunks.append(
        PolicyChunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            document_title=document.title,
            section_title="Overview",
            source_name=document.source_name,
            text=body.strip(),
        )
    )
    return chunks


def load_policy_chunks(policy_directory: Path) -> list[PolicyChunk]:
    """Load and chunk all policy documents in a directory."""
    documents = load_policy_documents(policy_directory)
    chunks: list[PolicyChunk] = []
    seen_chunk_ids: set[str] = set()

    for document in documents:
        for chunk in chunk_policy_document(document):
            if chunk.chunk_id in seen_chunk_ids:
                raise PolicyLoadError(f"Duplicate chunk ID detected: {chunk.chunk_id}")
            seen_chunk_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    return chunks
