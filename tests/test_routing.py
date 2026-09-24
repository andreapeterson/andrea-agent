import pytest

from app.models.routing import IntakeAnswers, RoutingAction, RoutingLevel
from app.services.routing import assess_routing

URGENT_FIELD_RULES = {
    "difficulty_breathing": "urgent-difficulty-breathing",
    "uncontrolled_bleeding": "urgent-uncontrolled-bleeding",
    "collapsed_or_unresponsive": "urgent-collapse-unresponsive",
    "known_toxin_exposure": "urgent-known-toxin",
}


@pytest.mark.parametrize(
    ("field_name", "rule_id"),
    sorted(URGENT_FIELD_RULES.items()),
)
def test_each_urgent_field_routes_to_urgent_handoff(field_name: str, rule_id: str) -> None:
    answers = IntakeAnswers(**{field_name: True})

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.URGENT
    assert decision.next_action == RoutingAction.CREATE_HANDOFF
    assert decision.matched_rule_ids == [rule_id]
    assert decision.missing_fields == []


def test_urgent_true_routes_immediately_even_when_other_answers_are_none() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=True,
        uncontrolled_bleeding=None,
        collapsed_or_unresponsive=None,
        known_toxin_exposure=None,
        rapidly_worsening=None,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.URGENT
    assert decision.next_action == RoutingAction.CREATE_HANDOFF
    assert decision.matched_rule_ids == ["urgent-difficulty-breathing"]
    assert decision.missing_fields == []


def test_multiple_urgent_indicators_return_all_matching_rule_ids() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=True,
        uncontrolled_bleeding=True,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=True,
        rapidly_worsening=False,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.URGENT
    assert decision.next_action == RoutingAction.CREATE_HANDOFF
    assert decision.matched_rule_ids == [
        "urgent-difficulty-breathing",
        "urgent-uncontrolled-bleeding",
        "urgent-known-toxin",
    ]
    assert decision.missing_fields == []


def test_urgent_takes_precedence_over_rapidly_worsening() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=True,
        rapidly_worsening=True,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.URGENT
    assert decision.next_action == RoutingAction.CREATE_HANDOFF
    assert decision.matched_rule_ids == ["urgent-known-toxin"]


def test_unanswered_urgent_question_requires_more_information() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=None,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=False,
        rapidly_worsening=False,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.NEEDS_MORE_INFORMATION
    assert decision.next_action == RoutingAction.ASK_INTAKE_QUESTION
    assert decision.missing_fields == ["difficulty_breathing"]
    assert decision.matched_rule_ids == []


def test_missing_fields_identifies_unanswered_fields() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=None,
        collapsed_or_unresponsive=None,
        known_toxin_exposure=False,
        rapidly_worsening=False,
    )

    decision = assess_routing(answers)

    assert decision.missing_fields == ["uncontrolled_bleeding", "collapsed_or_unresponsive"]


def test_all_urgent_false_and_rapidly_worsening_none_needs_more_information() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=False,
        rapidly_worsening=None,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.NEEDS_MORE_INFORMATION
    assert decision.next_action == RoutingAction.ASK_INTAKE_QUESTION
    assert decision.missing_fields == ["rapidly_worsening"]


def test_all_urgent_false_with_rapidly_worsening_true_is_same_day() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=False,
        rapidly_worsening=True,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.SAME_DAY
    assert decision.next_action == RoutingAction.SEARCH_SAME_DAY_APPOINTMENT
    assert decision.matched_rule_ids == ["same-day-rapidly-worsening"]
    assert decision.missing_fields == []


def test_all_five_false_is_routine() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=False,
        rapidly_worsening=False,
    )

    decision = assess_routing(answers)

    assert decision.routing_level == RoutingLevel.ROUTINE
    assert decision.next_action == RoutingAction.SEARCH_ROUTINE_APPOINTMENT
    assert decision.matched_rule_ids == ["routine-no-escalation-indicators"]
    assert decision.missing_fields == []


def test_separate_calls_do_not_share_lists() -> None:
    first = assess_routing(
        IntakeAnswers(
            difficulty_breathing=True,
            uncontrolled_bleeding=False,
            collapsed_or_unresponsive=None,
            known_toxin_exposure=False,
            rapidly_worsening=False,
        )
    )
    second = assess_routing(
        IntakeAnswers(
            difficulty_breathing=False,
            uncontrolled_bleeding=None,
            collapsed_or_unresponsive=False,
            known_toxin_exposure=False,
            rapidly_worsening=False,
        )
    )

    first.matched_rule_ids.append("should-not-impact-second")
    second.missing_fields.append("should-not-impact-first")

    assert first.matched_rule_ids == ["urgent-difficulty-breathing", "should-not-impact-second"]
    assert second.matched_rule_ids == []
    assert first.missing_fields == []
    assert second.missing_fields == ["uncontrolled_bleeding", "should-not-impact-first"]
