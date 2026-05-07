"""Run-mode helpers shared by GUI and CLI sweep config loading."""

from __future__ import annotations

from typing import Any


SWEEP_MODE = "blind_sweep"
OPTIMIZATION_MODE = "optimization"
LEGACY_SWEEP_MODE = "sweep"
SWEEP_OR_OPTIMIZATION_MODES = {
    SWEEP_MODE,
    LEGACY_SWEEP_MODE,
    OPTIMIZATION_MODE,
}


def normalize_sweep_or_optimization_mode(mode: Any) -> Any:
    """Map legacy sweep mode names to the current canonical mode."""
    if mode == LEGACY_SWEEP_MODE:
        return SWEEP_MODE
    return mode
