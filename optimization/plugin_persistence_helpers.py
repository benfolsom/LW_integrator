"""Shared helpers for optimization plugin config persistence."""

from __future__ import annotations

from typing import Any, Dict

from input_output.config import (
    apply_persisted_config_overrides,
    build_saved_config_payload,
)


def metrics_export_settings_from_data(data: Dict[str, Any]) -> tuple[str, str]:
    """Resolve persisted metrics export settings."""
    return (
        data.get("metrics_export_format", "both"),
        data.get("metrics_export_scope", "all"),
    )


def resolve_loaded_sweep_state(
    sweep_state: Dict[str, Dict[str, Any]], param_name: str
) -> Dict[str, Any] | None:
    """Return saved sweep state, including legacy driver Pz conversion."""
    if param_name in sweep_state:
        return sweep_state[param_name]

    if param_name == "driver_energy_gev" and "driver_starting_Pz" in sweep_state:
        old_state = sweep_state["driver_starting_Pz"]
        return {
            "enabled": old_state.get("enabled", False),
            "min": "50.0",
            "max": "200.0",
            "points": old_state.get("points", "3"),
            "log": old_state.get("log", False),
            "fixed_value": "112.5",
        }

    return None


__all__ = [
    "apply_persisted_config_overrides",
    "build_saved_config_payload",
    "metrics_export_settings_from_data",
    "resolve_loaded_sweep_state",
]
