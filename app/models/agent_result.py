"""Internal orchestration result for one processed caller turn."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationState
from app.models.policy_answer import PolicyAnswerResponse


class AgentTurnResult(BaseModel):
    """The saved conversation state and assistant response after one turn."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    state: ConversationState
    assistant_message: str = Field(..., min_length=1)
    policy_answer: PolicyAnswerResponse | None = None


__all__ = ["AgentTurnResult"]
