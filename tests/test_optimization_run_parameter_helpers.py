"""Tests for pure optimization run-control helpers."""

from core.types import SimulationType
from optimization.config import OptimizationConfig
from optimization.run_parameter_helpers import (
    calculate_transverse_offset,
    collect_optimization_parameter_selection,
    is_bunch_to_bunch,
    resolve_objective_metric,
)


def test_collect_optimization_parameter_selection_preserves_existing_order():
    config = OptimizationConfig(
        simulation_type=SimulationType.BUNCH_TO_BUNCH,
        aperture_points=2,
        energy_points=3,
        transverse_momentum_range=(0.1, 0.2),
        transverse_momentum_points=2,
        timestep_range=(1e-7, 2e-7),
        timestep_points=2,
        transverse_spread_range=(1e-6, 2e-6),
        transverse_spread_points=2,
        driver_energy_range=(0.5, 0.8),
        driver_energy_points=2,
    )

    selection = collect_optimization_parameter_selection(config)

    assert selection.names == [
        "aperture_radius",
        "initial_energy_gev",
        "transverse_momentum",
        "timestep",
        "rider_transv_dist",
        "driver_energy_gev",
    ]
    assert selection.bounds == [
        config.aperture_range,
        config.energy_range,
        config.transverse_momentum_range,
        config.timestep_range,
        config.transverse_spread_range,
        config.driver_energy_range,
    ]
    assert selection.log_lines == [
        f"    Added: initial_energy_gev, range={config.energy_range}, points=3",
        (
            "    Added: transverse_momentum, "
            f"range={config.transverse_momentum_range}, points=2"
        ),
        (
            "    Added: rider_transv_dist, "
            f"range={config.transverse_spread_range}, points=2"
        ),
    ]


def test_collect_optimization_parameter_selection_skips_disabled_parameters():
    config = OptimizationConfig(
        aperture_points=1,
        energy_points=1,
        transverse_momentum_range=(0.1, 0.2),
        transverse_momentum_points=1,
        driver_energy_range=None,
        driver_energy_points=3,
    )

    selection = collect_optimization_parameter_selection(config)

    assert selection.names == []
    assert selection.bounds == []
    assert selection.log_lines == []


def test_resolve_objective_metric_keeps_historical_defaults():
    assert resolve_objective_metric("max_energy_gain") == ("max_energy_gain_gev", True)
    assert resolve_objective_metric("max_percent_energy_gain") == (
        "max_percent_energy_gain",
        True,
    )
    assert resolve_objective_metric("min_transverse_spread") == (
        "max_energy_gain_gev",
        False,
    )


def test_calculate_transverse_offset_handles_enum_and_string_modes():
    assert is_bunch_to_bunch(SimulationType.BUNCH_TO_BUNCH)
    assert is_bunch_to_bunch("BUNCH_TO_BUNCH")
    assert not is_bunch_to_bunch(SimulationType.CONDUCTING_WALL)

    assert calculate_transverse_offset(
        SimulationType.BUNCH_TO_BUNCH, offset_value=0.25, aperture=0.001
    ) == 0.25
    assert calculate_transverse_offset(
        "BUNCH_TO_BUNCH", offset_value=0.25, aperture=0.001
    ) == 0.25
    assert calculate_transverse_offset(
        SimulationType.CONDUCTING_WALL, offset_value=0.25, aperture=0.001
    ) == 0.00025
