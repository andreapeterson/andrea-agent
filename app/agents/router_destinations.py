"""Shared Router destination type and validation of explicitly supplied choices."""

from typing import Literal, get_args

RouterDestination = Literal["verification", "intake", "policy", "scheduling", "human_handoff"]
ROUTER_DESTINATIONS: tuple[RouterDestination, ...] = get_args(RouterDestination)


def validate_eligible_destinations(
    destinations: tuple[RouterDestination, ...],
) -> tuple[RouterDestination, ...]:
    if (
        not isinstance(destinations, tuple) or not destinations
        or any(value not in ROUTER_DESTINATIONS for value in destinations)
        or len(set(destinations)) != len(destinations)
    ):
        raise ValueError("Eligible destinations must be a nonempty tuple of unique Router destinations.")
    return destinations
