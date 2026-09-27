from pathlib import Path
from types import SimpleNamespace

import pytest

from app.integrations.embeddings import (
    EmbeddingRequestError,
    EmbeddingResponseError,
    OpenAIEmbeddingProvider,
)
from app.models.policy import PolicyChunk
from app.services.policy_loader import load_policy_chunks
from app.services.policy_retriever import (
    InvalidVectorError,
    PolicyRetrievalError,
    PolicyRetriever,
    cosine_similarity,
)


class FakeEmbeddingClient:
    def __init__(self, *, payloads: list[list[float]] | None = None, error: Exception | None = None) -> None:
        self.embeddings = self
        self.payloads = payloads or []
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        data = []
        for index, values in enumerate(self.payloads):
            data.append(SimpleNamespace(index=index, embedding=values))
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_provider_submits_all_texts_in_one_request_and_preserves_order() -> None:
    client = FakeEmbeddingClient(payloads=[[1.0, 0.0], [0.0, 1.0]])
    provider = OpenAIEmbeddingProvider(client=client, model="text-embedding-3-small")

    result = await provider.embed_texts(["first", "second"])

    assert client.calls[0]["input"] == ["first", "second"]
    assert client.calls[0]["model"] == "text-embedding-3-small"
    assert result == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.asyncio
async def test_provider_rejects_blank_inputs() -> None:
    provider = OpenAIEmbeddingProvider(client=FakeEmbeddingClient())

    with pytest.raises(EmbeddingRequestError):
        await provider.embed_texts([" "])


@pytest.mark.asyncio
async def test_provider_converts_request_failures_to_embedding_request_error() -> None:
    provider = OpenAIEmbeddingProvider(client=FakeEmbeddingClient(error=RuntimeError("network issue")))

    with pytest.raises(EmbeddingRequestError):
        await provider.embed_texts(["hello"])


@pytest.mark.asyncio
async def test_provider_rejects_missing_response_items() -> None:
    class MissingItemClient:
        def __init__(self) -> None:
            self.embeddings = self

        async def create(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])

    provider = OpenAIEmbeddingProvider(client=MissingItemClient())

    with pytest.raises(EmbeddingResponseError):
        await provider.embed_texts(["one", "two"])


@pytest.mark.asyncio
async def test_provider_rejects_empty_vectors() -> None:
    provider = OpenAIEmbeddingProvider(client=FakeEmbeddingClient(payloads=[[]]))

    with pytest.raises(EmbeddingResponseError):
        await provider.embed_texts(["hello"])


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(InvalidVectorError):
        cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_similarity_rejects_empty_vectors() -> None:
    with pytest.raises(InvalidVectorError):
        cosine_similarity([], [1.0])


def test_cosine_similarity_rejects_zero_magnitude_vectors() -> None:
    with pytest.raises(InvalidVectorError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


class FakeEmbeddingProvider:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.mapping[text] for text in texts]


@pytest.mark.asyncio
async def test_policy_retriever_indexes_policy_chunks_and_ranks_by_similarity() -> None:
    chunks = load_policy_chunks(Path("knowledge/clinic_policies"))
    provider = FakeEmbeddingProvider({})
    for chunk in chunks:
        text = f"{chunk.document_title}\n{chunk.section_title}\n{chunk.text}"
        if chunk.section_title == "Cancellation and Rescheduling":
            provider.mapping[text] = [1.0, 0.0]
        elif chunk.section_title in {"Accepted Payment Methods", "Payment Plans and Limits", "Prescription Refill Requests", "Veterinarian-Client-Patient Relationship"}:
            provider.mapping[text] = [0.0, 1.0]
        else:
            provider.mapping[text] = [0.5, 0.5]
    provider.mapping["cancellation question"] = [0.9, 0.1]

    retriever = PolicyRetriever(provider)
    await retriever.index(chunks)

    assert len(provider.calls) == 1
    assert all("\n" in text for text in provider.calls[0])
    assert any("Appointment Policies" in provider.calls[0][0] for _ in [0])

    results = await retriever.search("cancellation question", limit=2)

    assert results[0].chunk.section_title == "Cancellation and Rescheduling"
    assert results[0].similarity_score > results[1].similarity_score
    assert results[0].chunk.source_name == "appointments_and_cancellations.md"
    assert len(results) == 2


@pytest.mark.asyncio
async def test_policy_retriever_limit_controls_result_count() -> None:
    chunk_a = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="alpha text",
    )
    chunk_b = PolicyChunk(
        chunk_id="doc::b",
        document_id="doc",
        document_title="Doc",
        section_title="Beta",
        source_name="doc.md",
        text="beta text",
    )
    mapping = {
        "Doc\nAlpha\nalpha text": [1.0, 0.0],
        "Doc\nBeta\nbeta text": [0.0, 1.0],
        "question": [1.0, 0.0],
    }
    provider = FakeEmbeddingProvider(mapping)
    retriever = PolicyRetriever(provider)
    await retriever.index([chunk_a, chunk_b])

    results = await retriever.search("question", limit=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc::a"


@pytest.mark.asyncio
async def test_policy_retriever_tie_breaks_by_chunk_id() -> None:
    first = PolicyChunk(
        chunk_id="doc::z",
        document_id="doc",
        document_title="Doc",
        section_title="Zebra",
        source_name="doc.md",
        text="same",
    )
    second = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="same",
    )
    mapping = {
        "Doc\nZebra\nsame": [1.0, 0.0],
        "Doc\nAlpha\nsame": [1.0, 0.0],
        "query": [1.0, 0.0],
    }
    provider = FakeEmbeddingProvider(mapping)
    retriever = PolicyRetriever(provider)
    await retriever.index([first, second])

    results = await retriever.search("query")
    assert [result.chunk.chunk_id for result in results] == ["doc::a", "doc::z"]


@pytest.mark.asyncio
async def test_search_before_indexing_fails_clearly() -> None:
    retriever = PolicyRetriever(FakeEmbeddingProvider({}))

    with pytest.raises(PolicyRetrievalError, match="No chunks have been indexed yet"):
        await retriever.search("question")


@pytest.mark.asyncio
async def test_blank_queries_fail_clearly() -> None:
    provider = FakeEmbeddingProvider({
        "Doc\nAlpha\nalpha": [1.0, 0.0],
    })
    retriever = PolicyRetriever(provider)
    await retriever.index([
        PolicyChunk(
            chunk_id="doc::a",
            document_id="doc",
            document_title="Doc",
            section_title="Alpha",
            source_name="doc.md",
            text="alpha",
        )
    ])

    with pytest.raises(PolicyRetrievalError):
        await retriever.search("   ")


@pytest.mark.asyncio
async def test_retriever_rejects_duplicate_chunk_ids() -> None:
    retriever = PolicyRetriever(FakeEmbeddingProvider({}))
    chunks = [
        PolicyChunk(
            chunk_id="doc::a",
            document_id="doc",
            document_title="Doc",
            section_title="Alpha",
            source_name="doc.md",
            text="alpha",
        ),
        PolicyChunk(
            chunk_id="doc::a",
            document_id="doc",
            document_title="Doc",
            section_title="Alpha",
            source_name="doc.md",
            text="duplicate",
        ),
    ]

    with pytest.raises(PolicyRetrievalError):
        await retriever.index(chunks)


@pytest.mark.asyncio
async def test_retriever_rejects_wrong_embedding_counts() -> None:
    chunk = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="alpha",
    )

    class BadProvider:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0], [0.0, 1.0]]

    retriever = PolicyRetriever(BadProvider())
    with pytest.raises(PolicyRetrievalError):
        await retriever.index([chunk])


@pytest.mark.asyncio
async def test_retriever_rejects_inconsistent_vector_dimensions() -> None:
    chunk_a = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="alpha",
    )
    chunk_b = PolicyChunk(
        chunk_id="doc::b",
        document_id="doc",
        document_title="Doc",
        section_title="Beta",
        source_name="doc.md",
        text="beta",
    )

    class BadProvider:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0], [1.0]]

    retriever = PolicyRetriever(BadProvider())
    with pytest.raises(PolicyRetrievalError):
        await retriever.index([chunk_a, chunk_b])


@pytest.mark.asyncio
async def test_failed_reindex_does_not_destroy_previous_valid_index() -> None:
    valid_provider = FakeEmbeddingProvider({
        "Doc\nAlpha\nalpha": [1.0, 0.0],
        "query": [1.0, 0.0],
    })
    retriever = PolicyRetriever(valid_provider)
    valid_chunk = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="alpha",
    )
    await retriever.index([valid_chunk])

    class BadProvider:
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0], [1.0]]

    retriever._embedding_provider = BadProvider()
    with pytest.raises(PolicyRetrievalError):
        await retriever.index([
            PolicyChunk(
                chunk_id="doc::b",
                document_id="doc",
                document_title="Doc",
                section_title="Beta",
                source_name="doc.md",
                text="beta",
            )
        ])

    assert retriever._index[0].chunk.chunk_id == "doc::a"


@pytest.mark.asyncio
async def test_search_results_keep_source_metadata() -> None:
    chunk = PolicyChunk(
        chunk_id="doc::a",
        document_id="doc",
        document_title="Doc",
        section_title="Alpha",
        source_name="doc.md",
        text="alpha",
    )
    provider = FakeEmbeddingProvider({
        "Doc\nAlpha\nalpha": [1.0, 0.0],
        "query": [1.0, 0.0],
    })
    retriever = PolicyRetriever(provider)
    await retriever.index([chunk])

    results = await retriever.search("query")
    assert results[0].chunk.source_name == "doc.md"
    assert results[0].chunk.document_title == "Doc"
    assert results[0].chunk.section_title == "Alpha"
