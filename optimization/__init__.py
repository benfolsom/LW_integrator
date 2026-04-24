"""Optimization package exports.

The package keeps heavyweight optional dependencies lazy so pure helpers and
tests can import ``optimization`` submodules without requiring SciPy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ObjectiveFunction",
    "ParameterGrid",
    "adaptive_grid_search",
    "compute_energy_at_position",
    "compute_energy_gain_near_aperture",
    "compute_max_energy_gain",
    "compute_percent_energy_gain",
    "compute_relative_energy_gain",
    "compute_trajectory_metrics",
    "create_energy_aperture_grid",
    "detect_transverse_deflection",
    "multi_start_optimize",
    "optimize_parameters",
]

_EXPORTS = {
    "compute_energy_at_position": ("optimization.metrics", "compute_energy_at_position"),
    "compute_energy_gain_near_aperture": (
        "optimization.metrics",
        "compute_energy_gain_near_aperture",
    ),
    "compute_max_energy_gain": ("optimization.metrics", "compute_max_energy_gain"),
    "compute_percent_energy_gain": (
        "optimization.metrics",
        "compute_percent_energy_gain",
    ),
    "compute_relative_energy_gain": (
        "optimization.metrics",
        "compute_relative_energy_gain",
    ),
    "compute_trajectory_metrics": (
        "optimization.metrics",
        "compute_trajectory_metrics",
    ),
    "detect_transverse_deflection": (
        "optimization.metrics",
        "detect_transverse_deflection",
    ),
    "ObjectiveFunction": ("optimization.optimizer", "ObjectiveFunction"),
    "adaptive_grid_search": ("optimization.optimizer", "adaptive_grid_search"),
    "multi_start_optimize": ("optimization.optimizer", "multi_start_optimize"),
    "optimize_parameters": ("optimization.optimizer", "optimize_parameters"),
    "ParameterGrid": ("optimization.parameter_sweep", "ParameterGrid"),
    "create_energy_aperture_grid": (
        "optimization.parameter_sweep",
        "create_energy_aperture_grid",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


__version__ = "0.1.0"
