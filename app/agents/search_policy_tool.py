"""Retrieve clinic-policy evidence for the Policy Agent."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from app.services.policy_retriever import PolicyRetriever


class SearchPolicyArguments(BaseModel):
    """Validate the administrative policy query extracted by the model."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, description="Administrative clinic-policy question to search for.")


SEARCH_POLICY_TOOL: dict[str, object] = {
    "type": "function",
    "name": "search_policy",
    "description": (
        "Use this tool to find evidence for administrative clinic-policy questions. "
        "Search before answering a factual policy question. "
        "It provides clinic policy, not medical advice."
    ),
    "strict": True,
    "parameters": SearchPolicyArguments.model_json_schema(),
}


def create_search_policy_handler(
    retriever: PolicyRetriever,
    *,
    top_k: int = 3,
    minimum_similarity: float = 0.45,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Use an initialized index; return found=false, evidence=[] when no chunk qualifies.

    Evidence uses PolicySearchResult's JSON fields: chunk (including its source
    identifiers and text) and similarity_score. Retrieval failures propagate.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not math.isfinite(minimum_similarity) or not 0 <= minimum_similarity <= 1:
        raise ValueError("minimum_similarity must be between 0 and 1.")

    async def handler(arguments: dict[str, object]) -> object:
        validated = SearchPolicyArguments.model_validate(arguments)
        results = await retriever.search(validated.query, limit=top_k)
        evidence = [
            result.model_dump(mode="json")
            for result in results
            if result.similarity_score >= minimum_similarity
        ]
        return {"found": bool(evidence), "evidence": evidence}

    return handler


__all__ = ["SEARCH_POLICY_TOOL", "SearchPolicyArguments", "create_search_policy_handler"]
