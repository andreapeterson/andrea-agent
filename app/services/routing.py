"""Deterministic routing logic for a fictional veterinary intake simulation.

This module exists as a portfolio-only demonstration. It intentionally does not
perform any real veterinary triage, use an LLM, call external services, or make
medical recommendations. A real implementation would require licensed veterinary
review and formal version control.
"""

from app.models.routing import (
    IntakeAnswers,
    RoutingAction,
    RoutingDecision,
    RoutingLevel,
)

URGENT_RULE_IDS: dict[str, str] = {
    "difficulty_breathing": "urgent-difficulty-breathing",
    "uncontrolled_bleeding": "urgent-uncontrolled-bleeding",
    "collapsed_or_unresponsive": "urgent-collapse-unresponsive",
    "known_toxin_exposure": "urgent-known-toxin",
}

URGENT_FIELDS = list(URGENT_RULE_IDS.keys())


def assess_routing(answers: IntakeAnswers) -> RoutingDecision:
    """Return an intake-routing decision using a deterministic, rule-based process.

    This function is pure and does not access any network, database, AI service,
    or environment configuration.
    """
    matched_urgent = [
        rule_id
        for field_name, rule_id in URGENT_RULE_IDS.items()
        if getattr(answers, field_name) is True
    ]
    if matched_urgent:
        return RoutingDecision(
            routing_level=RoutingLevel.URGENT,
            next_action=RoutingAction.CREATE_HANDOFF,
            matched_rule_ids=matched_urgent,
            missing_fields=[],
        )

    unanswered_urgent = [
        field_name
        for field_name in URGENT_FIELDS
        if getattr(answers, field_name) is None
    ]
    if unanswered_urgent:
        return RoutingDecision(
            routing_level=RoutingLevel.NEEDS_MORE_INFORMATION,
            next_action=RoutingAction.ASK_INTAKE_QUESTION,
            matched_rule_ids=[],
            missing_fields=unanswered_urgent,
        )

    if answers.rapidly_worsening is True:
        return RoutingDecision(
            routing_level=RoutingLevel.SAME_DAY,
            next_action=RoutingAction.SEARCH_SAME_DAY_APPOINTMENT,
            matched_rule_ids=["same-day-rapidly-worsening"],
            missing_fields=[],
        )

    if answers.rapidly_worsening is None:
        return RoutingDecision(
            routing_level=RoutingLevel.NEEDS_MORE_INFORMATION,
            next_action=RoutingAction.ASK_INTAKE_QUESTION,
            matched_rule_ids=[],
            missing_fields=["rapidly_worsening"],
        )

    if all(
        getattr(answers, field_name) is False
        for field_name in [
            "difficulty_breathing",
            "uncontrolled_bleeding",
            "collapsed_or_unresponsive",
            "known_toxin_exposure",
            "rapidly_worsening",
        ]
    ):
        return RoutingDecision(
            routing_level=RoutingLevel.ROUTINE,
            next_action=RoutingAction.SEARCH_ROUTINE_APPOINTMENT,
            matched_rule_ids=["routine-no-escalation-indicators"],
            missing_fields=[],
        )

    return RoutingDecision(
        routing_level=RoutingLevel.NEEDS_MORE_INFORMATION,
        next_action=RoutingAction.ASK_INTAKE_QUESTION,
        matched_rule_ids=[],
        missing_fields=[],
    )
