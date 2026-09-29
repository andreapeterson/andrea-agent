"""In-memory conversation-store boundary for PawLine agent memory."""

from __future__ import annotations

import copy
from typing import Protocol

from app.models.conversation import ConversationState


class ConversationStoreError(RuntimeError):
    """Base exception for conversation-store failures."""


class ConversationAlreadyExistsError(ConversationStoreError):
    """Raised when a conversation ID is created more than once."""


class ConversationNotFoundError(ConversationStoreError):
    """Raised when a conversation ID is not present in the store."""


class ConversationStore(Protocol):
    """Synchronous, process-local conversation-state storage for the agent."""

    def create(self, conversation_id: str) -> ConversationState:
        """Create and return a new empty conversation state."""

    def get(self, conversation_id: str) -> ConversationState:
        """Return a deep copy of the stored state."""

    def save(self, state: ConversationState) -> ConversationState:
        """Persist a state update and return a deep copy."""


class InMemoryConversationStore:
    """Process-local conversation memory persisted only in the current process.

    This store is intentionally simple and is not a database. It exists to hold
    conversation snapshots for the future orchestrator and disappears when PawLine
    restarts.
    """

    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}

    def create(self, conversation_id: str) -> ConversationState:
        if conversation_id is None or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        normalized = conversation_id.strip()
        if not normalized:
            raise ValueError("conversation_id cannot be blank.")
        if normalized in self._store:
            raise ConversationAlreadyExistsError(f"Conversation already exists: {normalized}")

        state = ConversationState(conversation_id=normalized)
        self._store[normalized] = copy.deepcopy(state)
        return copy.deepcopy(state)

    def get(self, conversation_id: str) -> ConversationState:
        if conversation_id is None or not isinstance(conversation_id, str):
            raise ConversationNotFoundError("Conversation not found.")

        normalized = conversation_id.strip()
        if normalized not in self._store:
            raise ConversationNotFoundError(f"Conversation not found: {normalized}")
        return copy.deepcopy(self._store[normalized])

    def save(self, state: ConversationState) -> ConversationState:
        if state is None or not isinstance(state, ConversationState):
            raise ValueError("state must be a ConversationState instance.")

        if not state.conversation_id or not state.conversation_id.strip():
            raise ValueError("state.conversation_id cannot be blank.")

        normalized = state.conversation_id.strip()
        if normalized not in self._store:
            raise ConversationNotFoundError(f"Conversation not found: {normalized}")

        self._store[normalized] = copy.deepcopy(state)
        return copy.deepcopy(self._store[normalized])


__all__ = [
    "ConversationAlreadyExistsError",
    "ConversationNotFoundError",
    "ConversationStore",
    "ConversationStoreError",
    "InMemoryConversationStore",
]
