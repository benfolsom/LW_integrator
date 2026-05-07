"""Helpers for parsing and normalizing optimization plugin config inputs."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from optimization.sweep_helpers import calculate_starting_pz_from_energy


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid list format: {value}") from exc


def parse_float_range(value: str) -> tuple[float, float] | None:
    """Parse a ``min,max`` string into a numeric range."""
    if not value or not value.strip():
        return None

    try:
        parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid range format: {value}") from exc

    if len(parts) != 2:
        raise ValueError(
            f"Invalid range format: {value} - Range must have exactly 2 values (min, max)"
        )
    if parts[0] >= parts[1]:
        raise ValueError(
            f"Invalid range format: {value} - Range min must be less than max"
        )
    return (parts[0], parts[1])


def parse_offset_pair(offset_str: str) -> tuple[float, float]:
    """Parse ``x,y`` input and fall back to ``(0.0, 0.0)`` on invalid input."""
    try:
        values = [float(part.strip()) for part in offset_str.split(",")]
    except (ValueError, AttributeError):
        return (0.0, 0.0)

    if len(values) >= 2:
        return (values[0], values[1])
    if len(values) == 1:
        return (values[0], 0.0)
    return (0.0, 0.0)


def apply_sweep_parameter_overrides(
    config: Any,
    sweep_params: Mapping[str, Mapping[str, Any]],
    *,
    driver_negative: bool,
    linked_energy_sweep: bool,
    debug: Callable[[str], None] | None = None,
) -> Any:
    """Apply sweep-enabled plugin controls onto an ``OptimizationConfig``."""

    def log(message: str) -> None:
        if debug is not None:
            debug(message)

    config.linked_energy_sweep = linked_energy_sweep
    if linked_energy_sweep:
        log(
            "[DEBUG] _gather_config: Linked energy sweep ENABLED - driver energy will follow rider energy"
        )

    log("[DEBUG] _gather_config: Checking sweep parameters...")
    for param_key, controls in sweep_params.items():
        enabled = controls["sweep_var"].get()
        log(f"  {param_key}: sweep_var={enabled}")
        if enabled:
            log(
                "    Range: "
                f"[{controls['min_var'].get()}, {controls['max_var'].get()}], "
                f"points={controls['points_var'].get()}"
            )

    _apply_optional_range(
        config,
        sweep_params,
        "rider_transv_mom",
        "transverse_momentum_range",
        "transverse_momentum_points",
        "Added rider_transv_mom",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "rider_transv_dist",
        "transverse_spread_range",
        "transverse_spread_points",
        "Added rider_transv_dist",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "macroparticle_charge_multiplier",
        "macroparticle_charge_range",
        "macroparticle_charge_points",
        None,
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "macroparticle_sigma_multiplier",
        "macroparticle_sigma_range",
        "macroparticle_sigma_points",
        None,
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "rider_m_particle",
        "particle_mass_range",
        "particle_mass_points",
        "Added rider_m_particle",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "rider_charge_sign",
        "particle_charge_range",
        "particle_charge_points",
        "Added rider_charge_sign",
        debug=log,
    )
    _apply_integer_range(
        config,
        sweep_params,
        "rider_pcount",
        "particle_count_range",
        "particle_count_points",
        "Added rider_pcount",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "rider_stripped_ions",
        "rider_stripped_ions_range",
        "rider_stripped_ions_points",
        "Added rider_stripped_ions",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "driver_m_particle",
        "driver_mass_range",
        "driver_mass_points",
        "Added driver_m_particle",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "driver_charge_sign",
        "driver_charge_sign_range",
        "driver_charge_sign_points",
        "Added driver_charge_sign",
        debug=log,
    )
    _apply_integer_range(
        config,
        sweep_params,
        "driver_pcount",
        "driver_pcount_range",
        "driver_pcount_points",
        "Added driver_pcount",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "driver_transv_mom",
        "driver_transv_mom_range",
        "driver_transv_mom_points",
        "Added driver_transv_mom",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "driver_transv_dist",
        "driver_transv_dist_range",
        "driver_transv_dist_points",
        "Added driver_transv_dist",
        debug=log,
    )
    _apply_optional_range(
        config,
        sweep_params,
        "driver_stripped_ions",
        "driver_stripped_ions_range",
        "driver_stripped_ions_points",
        "Added driver_stripped_ions",
        debug=log,
    )

    controls = sweep_params["driver_starting_distance"]
    if controls["sweep_var"].get():
        config.driver_starting_distance_range = _range_from_controls(controls)
        config.driver_starting_distance_points = int(controls["points_var"].get())
        config.driver_starting_distance_log_scale = bool(controls["log_var"].get())
        log(
            "[DEBUG] Added driver_starting_distance: "
            f"{config.driver_starting_distance_range}, "
            f"{config.driver_starting_distance_points} points, "
            f"log={config.driver_starting_distance_log_scale}"
        )

    energy_controls = sweep_params["driver_energy_gev"]
    if energy_controls["sweep_var"].get():
        energy_min = abs(float(energy_controls["min_var"].get()))
        energy_max = abs(float(energy_controls["max_var"].get()))
        if energy_min > energy_max:
            energy_min, energy_max = energy_max, energy_min

        driver_mass = float(sweep_params["driver_m_particle"]["fixed_var"].get())
        pz_min = calculate_starting_pz_from_energy(
            energy_min, driver_mass, negative=True
        )
        pz_max = calculate_starting_pz_from_energy(
            energy_max, driver_mass, negative=True
        )

        config.driver_starting_Pz_range = (pz_min, pz_max)
        config.driver_starting_Pz_points = int(energy_controls["points_var"].get())
        config.driver_direction = "-z" if driver_negative else "+z"
        config.driver_energy_range = (energy_min, energy_max)
        config.driver_energy_points = config.driver_starting_Pz_points
        log(
            "[DEBUG] Added driver_energy: "
            f"{config.driver_energy_range} GeV -> Pz: "
            f"{config.driver_starting_Pz_range} amu·mm/ns, "
            f"{config.driver_starting_Pz_points} points"
        )
    else:
        energy_gev = abs(float(energy_controls["fixed_var"].get()))
        driver_mass = float(sweep_params["driver_m_particle"]["fixed_var"].get())
        config.driver_starting_Pz = calculate_starting_pz_from_energy(
            energy_gev,
            driver_mass,
            negative=driver_negative,
        )
        config.driver_energy_gev = energy_gev

    log("[DEBUG] _gather_config: Config building complete")
    return config


def _range_from_controls(controls: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(controls["min_var"].get()),
        float(controls["max_var"].get()),
    )


def _apply_optional_range(
    config: Any,
    sweep_params: Mapping[str, Mapping[str, Any]],
    sweep_key: str,
    range_attr: str,
    points_attr: str,
    debug_label: str | None,
    *,
    debug: Callable[[str], None],
) -> None:
    controls = sweep_params[sweep_key]
    if not controls["sweep_var"].get():
        return

    setattr(config, range_attr, _range_from_controls(controls))
    setattr(config, points_attr, int(controls["points_var"].get()))
    if debug_label:
        debug(
            f"[DEBUG] {debug_label}: {getattr(config, range_attr)}, "
            f"{getattr(config, points_attr)} points"
        )


def _apply_integer_range(
    config: Any,
    sweep_params: Mapping[str, Mapping[str, Any]],
    sweep_key: str,
    range_attr: str,
    points_attr: str,
    debug_label: str,
    *,
    debug: Callable[[str], None],
) -> None:
    controls = sweep_params[sweep_key]
    if not controls["sweep_var"].get():
        return

    min_val = float(controls["min_var"].get())
    max_val = float(controls["max_var"].get())
    setattr(config, range_attr, (int(min_val), int(max_val)))
    setattr(config, points_attr, int(controls["points_var"].get()))
    debug(
        f"[DEBUG] {debug_label}: {getattr(config, range_attr)}, "
        f"{getattr(config, points_attr)} points"
    )
