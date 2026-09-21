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
        magnetic_dipole_source_model="covariant_retarded_point",
        magnetic_dipole_exact_retarded_backend="numba_roots_exact_serial",
        magnetic_dipole_exact_retarded_update=("second_order_start_taylor_endpoint"),
        magnetic_dipole_source_minimum_separation_mm=7.0e-9,
        magnetic_dipole_source_relative_stencil_step=2.0e-3,
        magnetic_dipole_source_minimum_stencil_step_mm=3.0e-15,
        magnetic_dipole_source_root_tolerance_mm=4.0e-21,
        magnetic_dipole_source_max_root_iterations=80,
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
    assert config.spin_model == "rfs_minimal_2021"
    assert config.stern_gerlach_model == "rfs_full_g"
    assert config.source.model == "covariant_retarded_point"
    assert config.exact_retarded_backend == "numba_roots_exact_serial"
    assert config.exact_retarded_update == ("second_order_start_taylor_endpoint")
    assert config.source.minimum_separation_mm == pytest.approx(7.0e-9)
    assert config.source.relative_stencil_step == pytest.approx(2.0e-3)
    assert config.source.minimum_stencil_step_mm == pytest.approx(3.0e-15)
    assert config.source.root_tolerance_mm == pytest.approx(4.0e-21)
    assert config.source.max_root_iterations == 80
    source_payload = restored.to_dict()["magnetic_dipole"]["source"]
    assert source_payload["model"] == "covariant_retarded_point"
    assert source_payload["history_model"] == "causal_frozen_c1"
    assert source_payload["minimum_separation_mm"] == pytest.approx(7.0e-9)
    assert source_payload["relative_stencil_step"] == pytest.approx(2.0e-3)
    assert source_payload["minimum_stencil_step_mm"] == pytest.approx(3.0e-15)
    assert source_payload["root_tolerance_mm"] == pytest.approx(4.0e-21)
    assert source_payload["max_root_iterations"] == 80
    assert source_payload["local_jet_primary_half_width_ns"] is None
    assert restored.to_dict()["magnetic_dipole"]["exact_retarded_backend"] == (
        "numba_roots_exact_serial"
    )
    assert restored.to_dict()["magnetic_dipole"]["exact_retarded_update"] == (
        "second_order_start_taylor_endpoint"
    )
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
    magnetic_payload = options.to_dict()["magnetic_dipole"]

    assert options.magnetic_dipole_enabled is False
    assert options.magnetic_dipole_spin_model == "rfs_minimal_2021"
    assert options.magnetic_dipole_stern_gerlach_model == "rfs_full_g"
    assert magnetic_payload["spin_model"] == "rfs_minimal_2021"
    assert magnetic_payload["stern_gerlach_model"] == "rfs_full_g"
    assert magnetic_payload["exact_retarded_backend"] == "python"
    assert magnetic_payload["exact_retarded_update"] == "first_order_endpoint"
    assert magnetic_payload["source"]["model"] == "off"
    assert "backend" not in magnetic_payload["source"]
    assert magnetic_payload["source"]["minimum_separation_mm"] == pytest.approx(2.0e-9)
    assert build_magnetic_dipole_config(options).enabled is False
    assert build_magnetic_dipole_config(options).source.active is False


def test_legacy_source_backend_alias_serializes_to_canonical_location() -> None:
    options = SimulationOptions.from_dict(
        {"magnetic_dipole": {"source": {"backend": "numba_full_strict_serial"}}}
    )

    assert options.magnetic_dipole_exact_retarded_backend == (
        "numba_full_strict_serial"
    )
    assert build_magnetic_dipole_config(options).exact_retarded_backend == (
        "numba_full_strict_serial"
    )
    magnetic_payload = options.to_dict()["magnetic_dipole"]
    assert magnetic_payload["exact_retarded_backend"] == ("numba_full_strict_serial")
    assert "backend" not in magnetic_payload["source"]


def test_matching_canonical_and_legacy_backends_are_accepted() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "exact_retarded_backend": "numba_roots_exact_serial",
                "source": {"backend": "numba_roots_exact_serial"},
            }
        }
    )

    assert options.magnetic_dipole_exact_retarded_backend == (
        "numba_roots_exact_serial"
    )


def test_conflicting_canonical_and_legacy_backends_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts with legacy"):
        SimulationOptions.from_dict(
            {
                "magnetic_dipole": {
                    "exact_retarded_backend": "python",
                    "source": {"backend": "numba_full_strict_serial"},
                }
            }
        )


def test_flat_testbed_source_fields_are_preserved_for_legacy_serializers() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole_source_model": "full_retarded_point",
            "magnetic_dipole_exact_retarded_backend": ("numba_roots_exact_serial"),
            "magnetic_dipole_source_minimum_separation_mm": 5.0e-9,
            "magnetic_dipole_source_relative_stencil_step": 1.5e-3,
            "magnetic_dipole_source_minimum_stencil_step_mm": 2.0e-15,
            "magnetic_dipole_source_root_tolerance_mm": 3.0e-21,
            "magnetic_dipole_source_max_root_iterations": 70,
        }
    )

    restored = SimulationOptions.from_dict(options.to_dict())
    config = build_magnetic_dipole_config(restored)

    assert config.source.model == "covariant_retarded_point"
    assert config.exact_retarded_backend == "numba_roots_exact_serial"
    assert config.source.minimum_separation_mm == pytest.approx(5.0e-9)
    assert config.source.relative_stencil_step == pytest.approx(1.5e-3)
    assert config.source.minimum_stencil_step_mm == pytest.approx(2.0e-15)
    assert config.source.root_tolerance_mm == pytest.approx(3.0e-21)
    assert config.source.max_root_iterations == 70


def test_rfs_config_without_rr_mode_defaults_to_off() -> None:
    options = SimulationOptions.from_dict({"magnetic_dipole": {"enabled": True}})

    assert options.radiation_reaction_mode == "off"
    assert options.adaptive_timestep_enabled is False


def test_rfs_config_preserves_explicit_rr_mode_for_validation() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {"enabled": True},
            "radiation_reaction_mode": "medina_lad",
        }
    )

    assert options.radiation_reaction_mode == "medina_lad"


def test_rfs_config_preserves_explicit_adaptive_mode_for_validation() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {"enabled": True},
            "adaptive_timestep_enabled": True,
        }
    )

    assert options.adaptive_timestep_enabled is True


def test_legacy_diagnostic_models_round_trip_through_testbed_config() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "spin_model": "bmt_frenkel",
                "stern_gerlach_model": "static_rest_gradient",
            }
        }
    )

    restored = SimulationOptions.from_dict(options.to_dict())
    config = build_magnetic_dipole_config(restored)

    assert restored.magnetic_dipole_spin_model == "bmt_frenkel"
    assert restored.magnetic_dipole_stern_gerlach_model == "static_rest_gradient"
    assert config.spin_model == "bmt_frenkel"
    assert config.stern_gerlach_model == "static_rest_gradient"


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


def test_causal_local_source_options_round_trip_and_build() -> None:
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "enabled": True,
                "source": {
                    "model": "covariant_retarded_point",
                    "history_model": "causal_local_jet",
                    "local_jet_narrow_half_width_ns": 1.0e-8,
                    "local_jet_primary_half_width_ns": 1.2e-8,
                    "local_jet_wide_half_width_ns": 1.5e-8,
                    "local_jet_inertial_prehistory": "assumed_inertial",
                },
            }
        }
    )

    restored = SimulationOptions.from_dict(options.to_dict())
    source = build_magnetic_dipole_config(restored).source

    assert source.history_model == "causal_local_jet"
    assert source.local_jet_narrow_half_width_ns == 1.0e-8
    assert source.local_jet_primary_half_width_ns == 1.2e-8
    assert source.local_jet_wide_half_width_ns == 1.5e-8
    assert source.local_jet_acceleration_samples == "interval_mean"
    assert source.local_jet_window_alignment == "past"
    assert source.local_jet_inertial_prehistory == "assumed_inertial"


def test_causal_local_named_scales_round_trip_and_build() -> None:
    scales = [
        {
            "name": "near",
            "narrow_half_width_ns": 2.0e-9,
            "primary_half_width_ns": 3.0e-9,
            "wide_half_width_ns": 5.0e-9,
        },
        {
            "name": "far",
            "narrow_half_width_ns": 5.0e-9,
            "primary_half_width_ns": 1.2e-8,
            "wide_half_width_ns": 1.5e-8,
        },
    ]
    options = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "source": {
                    "history_model": "causal_local_jet",
                    "local_jet_scales": scales,
                    "local_jet_maximum_cross_scale_relative_spread": 2.5e-4,
                }
            }
        }
    )

    restored = SimulationOptions.from_dict(options.to_dict())
    source = build_magnetic_dipole_config(restored).source

    assert restored.magnetic_dipole_source_local_jet_scales == scales
    assert tuple(scale.name for scale in source.local_jet_scales) == ("near", "far")
    assert source.local_jet_maximum_cross_scale_relative_spread == 2.5e-4


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
