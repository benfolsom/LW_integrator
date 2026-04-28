"""Helpers for handling enum-backed and string-backed simulation modes."""

from __future__ import annotations

from typing import Any

from core.types import SimulationType


def is_bunch_to_bunch(simulation_type: Any) -> bool:
    """Return whether a simulation type value represents BUNCH_TO_BUNCH mode."""
    if simulation_type == SimulationType.BUNCH_TO_BUNCH:
        return True
    if simulation_type == "BUNCH_TO_BUNCH":
        return True
    return getattr(simulation_type, "name", None) == "BUNCH_TO_BUNCH"
