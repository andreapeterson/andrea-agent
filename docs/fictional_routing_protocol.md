# Fictional Veterinary Intake Routing Protocol

This document describes a fictional demonstration protocol for PawLine. It is not veterinary advice, not a clinical decision support system, and not a real treatment guideline. This simulation is for portfolio demonstration only.

A real veterinary intake protocol must be reviewed, approved, and version-controlled by licensed veterinary professionals before use in any operational or clinical context.

## Intake questions

The fictional intake protocol asks five questions:

1. difficulty_breathing
2. uncontrolled_bleeding
3. collapsed_or_unresponsive
4. known_toxin_exposure
5. rapidly_worsening

Each answer is a boolean or None. None means the question has not been answered yet.

## Rule IDs

Urgent rules:
- urgent-difficulty-breathing
- urgent-uncontrolled-bleeding
- urgent-collapse-unresponsive
- urgent-known-toxin

Same-day rule:
- same-day-rapidly-worsening

Routine rule:
- routine-no-escalation-indicators

## Rule precedence

1. Any urgent True field immediately creates a handoff.
2. If no urgent field is True but an urgent question is unanswered, the system asks for more information.
3. After all urgent fields are False, rapidly_worsening True creates a same-day search.
4. If all urgent fields are False and rapidly_worsening is unanswered, the system asks for more information.
5. If all five answers are False, the system searches for a routine appointment.

## Why urgent True creates a handoff

Urgent True indicators represent the demonstration's highest-priority signals. When a patient presents with any serious escalation signal, the protocol immediately routes to a handoff instead of waiting for more information or continuing to appointment search.

## Why unanswered urgent questions prevent appointment routing

A None answer means the system does not know whether the urgent signal is present. Because the protocol is intentionally conservative, unanswered urgent questions block appointment routing until those questions are answered.

## Why an LLM is not permitted to override the routing decision

This protocol is deliberately deterministic and rule-based. It is not allowed to use an LLM, external API, or any autonomous reasoning layer to override the result, because the purpose of this checkpoint is to demonstrate a fixed, reviewable, testable decision tree without hidden or non-deterministic behavior.
