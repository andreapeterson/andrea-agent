from pathlib import Path

import pytest

from app.models.policy import PolicyChunk, PolicyDocument
from app.services.policy_loader import PolicyLoadError, chunk_policy_document, load_policy_chunks, load_policy_documents


def test_real_policy_directory_loads_three_documents() -> None:
    documents = load_policy_documents(Path("knowledge/clinic_policies"))

    assert len(documents) == 3
    assert [document.document_id for document in documents] == [
        "appointments_and_cancellations",
        "hours_and_emergencies",
        "payments_and_prescriptions",
    ]
    assert all(isinstance(document, PolicyDocument) for document in documents)


def test_files_are_loaded_in_sorted_order_and_h1_is_used_as_title(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()

    (docs_dir / "b.md").write_text("# Beta\n\n## Second\n\nText B\n", encoding="utf-8")
    (docs_dir / "a.md").write_text("# Alpha\n\n## First\n\nText A\n", encoding="utf-8")

    documents = load_policy_documents(docs_dir)

    assert [document.document_id for document in documents] == ["a", "b"]
    assert documents[0].title == "Alpha"
    assert documents[1].title == "Beta"


def test_every_h2_section_becomes_a_policy_chunk() -> None:
    documents = load_policy_documents(Path("knowledge/clinic_policies"))
    chunks = []
    for document in documents:
        chunks.extend(chunk_policy_document(document))

    assert len(chunks) == 13
    assert all(isinstance(chunk, PolicyChunk) for chunk in chunks)
    assert sum(1 for chunk in chunks if chunk.section_title == "Overview") == 3
    assert {chunk.document_id for chunk in chunks} == {
        "appointments_and_cancellations",
        "hours_and_emergencies",
        "payments_and_prescriptions",
    }


def test_chunk_metadata_points_back_to_the_correct_document() -> None:
    documents = load_policy_documents(Path("knowledge/clinic_policies"))
    document_map = {document.document_id: document for document in documents}
    chunks = load_policy_chunks(Path("knowledge/clinic_policies"))

    for chunk in chunks:
        document = document_map[chunk.document_id]
        assert chunk.document_title == document.title
        assert chunk.source_name == document.source_name


def test_chunk_ids_remain_stable_across_repeated_loads() -> None:
    first = [chunk.chunk_id for chunk in load_policy_chunks(Path("knowledge/clinic_policies"))]
    second = [chunk.chunk_id for chunk in load_policy_chunks(Path("knowledge/clinic_policies"))]

    assert first == second


def test_introductory_text_becomes_an_overview_chunk_when_present(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "# Example Policy\n\nIntro paragraph before the first section.\n\n## First Section\n\nAlpha text.\n",
        encoding="utf-8",
    )

    chunks = chunk_policy_document(load_policy_documents(docs_dir)[0])

    assert chunks[0].section_title == "Overview"
    assert chunks[0].text.startswith("Intro paragraph before the first section")
    assert chunks[1].section_title == "First Section"


def test_empty_sections_do_not_produce_chunks(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text("# Example\n\n## Filled\n\nSome text\n\n## Empty\n\n", encoding="utf-8")

    chunks = chunk_policy_document(load_policy_documents(docs_dir)[0])

    assert [chunk.section_title for chunk in chunks] == ["Filled"]


def test_non_markdown_files_are_ignored(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "keep.md").write_text("# Keep\n\n## Policy\n\nText\n", encoding="utf-8")
    (docs_dir / "ignore.txt").write_text("not markdown", encoding="utf-8")

    documents = load_policy_documents(docs_dir)

    assert [document.source_name for document in documents] == ["keep.md"]


def test_missing_directory_raises_policy_load_error(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(PolicyLoadError):
        load_policy_documents(missing_dir)


def test_missing_h1_raises_policy_load_error(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "bad.md").write_text("## Section\n\nSome text\n", encoding="utf-8")

    with pytest.raises(PolicyLoadError):
        load_policy_documents(docs_dir)


def test_empty_corpus_raises_policy_load_error(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()

    with pytest.raises(PolicyLoadError):
        load_policy_documents(docs_dir)


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "doc.md").write_text(
        "# Doc\n\n## Shared\n\nAlpha\n\n## Shared\n\nBeta\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyLoadError):
        chunk_policy_document(load_policy_documents(docs_dir)[0])
