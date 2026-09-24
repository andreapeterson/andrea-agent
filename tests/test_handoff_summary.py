from app.models.handoff import HandoffSummary
from app.models.routing import IntakeAnswers, RoutingAction, RoutingDecision, RoutingLevel
from app.services.handoff_summary import build_handoff_summary


def test_summary_classifies_true_false_and_none_fields() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=True,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=None,
        known_toxin_exposure=False,
        rapidly_worsening=True,
    )
    decision = RoutingDecision(
        routing_level=RoutingLevel.URGENT,
        next_action=RoutingAction.CREATE_HANDOFF,
        matched_rule_ids=["urgent-difficulty-breathing", "same-day-rapidly-worsening"],
        missing_fields=["collapsed_or_unresponsive"],
    )

    summary = build_handoff_summary("labored breathing", answers, decision)

    assert summary.original_concern == "labored breathing"
    assert summary.routing_level == RoutingLevel.URGENT
    assert summary.positive_signals == ["difficulty_breathing", "rapidly_worsening"]
    assert summary.negative_signals == ["uncontrolled_bleeding", "known_toxin_exposure"]
    assert summary.unanswered_signals == ["collapsed_or_unresponsive"]
    assert summary.matched_rule_ids == ["urgent-difficulty-breathing", "same-day-rapidly-worsening"]


def test_every_intake_field_appears_in_exactly_one_signal_list() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=True,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=None,
        known_toxin_exposure=False,
        rapidly_worsening=None,
    )
    decision = RoutingDecision(
        routing_level=RoutingLevel.URGENT,
        next_action=RoutingAction.CREATE_HANDOFF,
        matched_rule_ids=["urgent-difficulty-breathing"],
        missing_fields=[],
    )

    summary = build_handoff_summary("respiratory distress", answers, decision)

    flattened = (
        summary.positive_signals
        + summary.negative_signals
        + summary.unanswered_signals
    )
    assert set(flattened) == {
        "difficulty_breathing",
        "uncontrolled_bleeding",
        "collapsed_or_unresponsive",
        "known_toxin_exposure",
        "rapidly_worsening",
    }
    assert len(flattened) == 5
    assert len(set(flattened)) == 5


def test_summary_does_not_share_list_objects() -> None:
    answers = IntakeAnswers(
        difficulty_breathing=True,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=None,
        known_toxin_exposure=False,
        rapidly_worsening=None,
    )
    decision = RoutingDecision(
        routing_level=RoutingLevel.NEEDS_MORE_INFORMATION,
        next_action=RoutingAction.ASK_INTAKE_QUESTION,
        matched_rule_ids=[],
        missing_fields=["collapsed_or_unresponsive"],
    )

    first = build_handoff_summary("uncertain", answers, decision)
    second = build_handoff_summary("uncertain", answers, decision)

    assert first.positive_signals is not second.positive_signals
    assert first.negative_signals is not second.negative_signals
    assert first.unanswered_signals is not second.unanswered_signals
    assert first.matched_rule_ids is not second.matched_rule_ids

    first.positive_signals.append("unexpected")
    first.negative_signals.append("unexpected")
    first.unanswered_signals.append("unexpected")
    first.matched_rule_ids.append("unexpected")

    assert second.positive_signals == ["difficulty_breathing"]
    assert second.negative_signals == ["uncontrolled_bleeding", "known_toxin_exposure"]
    assert second.unanswered_signals == ["collapsed_or_unresponsive", "rapidly_worsening"]
    assert second.matched_rule_ids == []


def test_handoff_summary_model_forbids_extra_fields() -> None:
    try:
        HandoffSummary(
            original_concern="breathing issue",
            routing_level=RoutingLevel.URGENT,
            positive_signals=["difficulty_breathing"],
            negative_signals=[],
            unanswered_signals=[],
            matched_rule_ids=["urgent-difficulty-breathing"],
            extra_field="unexpected",
        )
    except ValueError:
        return

    raise AssertionError("HandoffSummary should forbid extra fields")
