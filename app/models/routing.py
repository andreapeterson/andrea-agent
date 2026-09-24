"""Fictional intake-routing protocol for PawLine demo simulation.

This module is intentionally limited to a deterministic portfolio example.
It does not provide veterinary guidance and must not be used in a real clinical
workflow without review and approval by licensed veterinary professionals.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class IntakeAnswers(BaseModel):
    """Represents the current state of a fictional intake triage questionnaire.

    None means the question has not yet been answered.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    difficulty_breathing: bool | None = None
    uncontrolled_bleeding: bool | None = None
    collapsed_or_unresponsive: bool | None = None
    known_toxin_exposure: bool | None = None
    rapidly_worsening: bool | None = None


class RoutingLevel(str, Enum):
    NEEDS_MORE_INFORMATION = "needs_more_information"
    URGENT = "urgent"
    SAME_DAY = "same_day"
    ROUTINE = "routine"


class RoutingAction(str, Enum):
    ASK_INTAKE_QUESTION = "ask_intake_question"
    CREATE_HANDOFF = "create_handoff"
    SEARCH_SAME_DAY_APPOINTMENT = "search_same_day_appointment"
    SEARCH_ROUTINE_APPOINTMENT = "search_routine_appointment"


class RoutingDecision(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    routing_level: RoutingLevel
    next_action: RoutingAction
    matched_rule_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
