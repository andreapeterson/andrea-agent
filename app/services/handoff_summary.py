"""Deterministic, auditable summary generation for escalation cases."""

from app.models.handoff import HandoffSummary
from app.models.routing import IntakeAnswers, RoutingDecision

_INTAKE_FIELD_NAMES = [
    "difficulty_breathing",
    "uncontrolled_bleeding",
    "collapsed_or_unresponsive",
    "known_toxin_exposure",
    "rapidly_worsening",
]


def build_handoff_summary(
    original_concern: str,
    intake_answers: IntakeAnswers,
    routing_decision: RoutingDecision,
) -> HandoffSummary:
    """Build a structured summary from trusted intake answers and routing data."""
    positive_signals: list[str] = []
    negative_signals: list[str] = []
    unanswered_signals: list[str] = []

    for field_name in _INTAKE_FIELD_NAMES:
        value = getattr(intake_answers, field_name)
        if value is True:
            positive_signals.append(field_name)
        elif value is False:
            negative_signals.append(field_name)
        elif value is None:
            unanswered_signals.append(field_name)

    return HandoffSummary(
        original_concern=original_concern,
        routing_level=routing_decision.routing_level,
        positive_signals=positive_signals,
        negative_signals=negative_signals,
        unanswered_signals=unanswered_signals,
        matched_rule_ids=list(routing_decision.matched_rule_ids),
    )
