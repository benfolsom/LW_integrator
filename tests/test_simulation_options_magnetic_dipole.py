from __future__ import annotations

import pytest

from lw_integrator.testbed_runner import (
    SimulationOptions,
    _build_all_particle_tracks,
    build_external_field_config,
    build_magnetic_dipole_config,
)


def test_magnetic_dipole_nested_config_round_trip() -> None:
    options = SimulationOptions(
        magnetic_dipole_enabled=True,
        magnetic_dipole_spin_precession_enabled=True,
        magnetic_dipole_stern_gerlach_force_enabled=True,
        rider_magnetic_species="neutron",
        rider_rest_spin=(1.0, 2.0, 3.0),
        rider_polarization=0.75,
        driver_magnetic_species="antiproton",
        driver_rest_spin=(0.0, -1.0, 0.0),
        external_field_enabled=True,
        external_magnetic_field_gradient_t_per_m=(
            (-4.0, 2.0, 3.0),
            (4.0, -5.0, 6.0),
            (7.0, 8.0, 9.0),
        ),
    )

    restored = SimulationOptions.from_dict(options.to_dict())
    config = build_magnetic_dipole_config(restored)
    external = build_external_field_config(restored)

    assert config.enabled is True
    assert config.spin_precession_enabled is True
    assert config.stern_gerlach_force_enabled is True
    assert config.rider.species == "neutron"
    assert config.rider.polarization == pytest.approx(0.75)
    assert config.rider.rest_spin == pytest.approx(
        (1.0 / 14.0**0.5, 2.0 / 14.0**0.5, 3.0 / 14.0**0.5)
    )
    assert config.driver.species == "antiproton"
    assert config.driver.rest_spin == pytest.approx((0.0, -1.0, 0.0))
    assert external is not None
    assert external.magnetic_field_gradient_t_per_m == (
        (-4.0, 2.0, 3.0),
        (4.0, -5.0, 6.0),
        (7.0, 8.0, 9.0),
    )


def test_old_config_defaults_magnetic_dipoles_off() -> None:
    options = SimulationOptions.from_dict({"steps": 4})

    assert options.magnetic_dipole_enabled is False
    assert build_magnetic_dipole_config(options).enabled is False


def test_custom_moment_round_trip_preserves_sign() -> None:
    options = SimulationOptions(
        magnetic_dipole_enabled=True,
        rider_magnetic_species="custom",
        rider_magnetic_moment_j_per_t=-1.25e-27,
        rider_spin_quantum_number=1.5,
    )

    restored = SimulationOptions.from_dict(options.to_dict())

    assert restored.rider_magnetic_moment_j_per_t == pytest.approx(-1.25e-27)
    assert restored.rider_spin_quantum_number == pytest.approx(1.5)


def test_all_particle_visualization_tracks_preserve_particle_axis() -> None:
    import numpy as np

    states = []
    for step in range(3):
        states.append(
            {
                "t": np.array([step, step + 0.5], dtype=float),
                "x": np.array([step, 10.0 + step], dtype=float),
                "y": np.array([0.0, 1.0], dtype=float),
                "z": np.array([-step, -10.0 - step], dtype=float),
                "spin_x": np.array([1.0, 0.0]),
                "spin_y": np.array([0.0, 1.0]),
                "spin_z": np.array([0.0, 0.0]),
                "local_magnetic_field_x_t": np.array([0.1, 0.2]),
                "local_magnetic_field_y_t": np.array([0.0, 0.0]),
                "local_magnetic_field_z_t": np.array([1.0, 2.0]),
            }
        )

    tracks = _build_all_particle_tracks(states, interval=2)

    assert len(tracks) == 2
    assert tracks[0]["time_ns"] == [0.0, 2.0]
    assert tracks[1]["positions_mm"]["x"] == [10.0, 12.0]
    assert tracks[0]["spin"] == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert tracks[1]["local_magnetic_field_t"] == [
        [0.2, 0.0, 2.0],
        [0.2, 0.0, 2.0],
    ]
