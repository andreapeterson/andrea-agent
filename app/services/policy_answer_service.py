"""Grounded clinic-policy answer service with relevance gating and verified citations."""

from __future__ import annotations

from app.integrations.policy_answer_generator import (
    PolicyAnswerGenerator,
    PolicyGenerationResponseError,
)
from app.models.policy_answer import (
    PolicyAnswerResponse,
    PolicyAnswerStatus,
    PolicyCitation,
)
from app.models.retrieval import PolicySearchResult
from app.services.policy_retriever import PolicyRetrievalError, PolicyRetriever

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I don’t have enough clinic policy information to answer that. "
    "I can help connect you with a clinic team member."
)


class PolicyAnswerService:
    """Retrieve policy context, gate low-similarity results, and produce verified citations."""

    def __init__(
        self,
        retriever: PolicyRetriever,
        generator: PolicyAnswerGenerator,
        top_k: int = 3,
        minimum_similarity: float = 0.45,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if minimum_similarity < 0 or minimum_similarity > 1:
            raise ValueError("minimum_similarity must be between 0 and 1.")

        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._minimum_similarity = minimum_similarity

    @staticmethod
    def _normalize_chunk_ids(values: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for value in values:
            candidate = str(value).strip()
            if not candidate:
                continue
            if candidate not in seen:
                seen.add(candidate)
                normalized.append(candidate)
        return normalized

    async def answer(self, question: str) -> PolicyAnswerResponse:
        if question is None or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string.")

        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank.")

        results = await self._retriever.search(cleaned, limit=self._top_k)
        if not results or results[0].similarity_score < self._minimum_similarity:
            return PolicyAnswerResponse(
                question=cleaned,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                status=PolicyAnswerStatus.INSUFFICIENT_CONTEXT,
                citations=[],
            )

        generated = await self._generator.generate_answer(cleaned, results)
        if not generated.answerable:
            return PolicyAnswerResponse(
                question=cleaned,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                status=PolicyAnswerStatus.INSUFFICIENT_CONTEXT,
                citations=[],
            )

        answer = str(generated.answer or "").strip()
        if not answer:
            raise PolicyGenerationResponseError("Answerable generation returned a blank answer.")

        supporting_ids = self._normalize_chunk_ids(generated.supporting_chunk_ids)
        if not supporting_ids:
            raise PolicyGenerationResponseError("Answerable generation returned no supporting chunk IDs.")

        valid_ids = {result.chunk.chunk_id for result in results}
        invalid_ids = [chunk_id for chunk_id in supporting_ids if chunk_id not in valid_ids]
        if invalid_ids:
            raise PolicyGenerationResponseError(
                "Supporting chunk IDs include results that were not in the retrieved context."
            )

        citation_order = {result.chunk.chunk_id: result for result in results}
        citations: list[PolicyCitation] = []
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in supporting_ids:
                citations.append(
                    PolicyCitation(
                        chunk_id=chunk_id,
                        document_id=result.chunk.document_id,
                        document_title=result.chunk.document_title,
                        section_title=result.chunk.section_title,
                        source_name=result.chunk.source_name,
                    )
                )

        return PolicyAnswerResponse(
            question=cleaned,
            answer=answer,
            status=PolicyAnswerStatus.ANSWERED,
            citations=citations,
        )


__all__ = ["INSUFFICIENT_CONTEXT_MESSAGE", "PolicyAnswerService"]
