"""Integration-seam tests for the coupled RFS magnetic-dipole model."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.external_fields import ExternalFieldConfig, NATIVE_FORCE_UNIT_NEWTON
from core.integration_runner import retarded_integrator
from core.magnetic_dipole import HBAR_NATIVE, boost_rest_polarization
from core.rfs import minkowski_dot
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    StartupMode,
)


def _particle_state(
    species_name: str,
    *,
    position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    species = get_species(species_name)
    beta_vector = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - float(beta_vector @ beta_vector))
    momentum = gamma * species.mass_amu * C_MMNS * beta_vector
    charge = float(species.charge_e) * ELEMENTARY_CHARGE
    return {
        "x": np.array([position_mm[0]], dtype=float),
        "y": np.array([position_mm[1]], dtype=float),
        "z": np.array([position_mm[2]], dtype=float),
        "t": np.array([0.0]),
        "Px": np.array([momentum[0]]),
        "Py": np.array([momentum[1]]),
        "Pz": np.array([momentum[2]]),
        "Pt": np.array([gamma * species.mass_amu * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.array([beta_vector[0]]),
        "by": np.array([beta_vector[1]]),
        "bz": np.array([beta_vector[2]]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        # q remains the legacy single-particle value. The explicit fields are
        # included so this test exercises source/observer charge separation.
        "q": np.array([charge]),
        "q_species": np.array([charge]),
        "q_observer": np.array([charge]),
        "q_source": np.array([charge]),
        "m": np.array([species.mass_amu]),
        "m_species": np.array([species.mass_amu]),
        "char_time": np.array([0.0]),
    }


def _empty_state() -> dict[str, np.ndarray]:
    return {
        key: np.array([], dtype=float)
        for key in (
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
            "bdotx",
            "bdoty",
            "bdotz",
            "q",
            "q_species",
            "q_observer",
            "q_source",
            "m",
            "m_species",
            "char_time",
        )
    }


def _rfs_config(*, rider_polarization: float = 1.0) -> MagneticDipoleConfig:
    return MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(
            species="neutron",
            rest_spin=(0.0, 0.0, 1.0),
            polarization=rider_polarization,
        ),
        driver=MagneticDipoleParticleConfig(
            species="proton",
            rest_spin=(0.0, 0.0, 1.0),
            polarization=0.0,
        ),
    )


def _run(
    *,
    rider: dict[str, np.ndarray],
    driver: dict[str, np.ndarray] | None,
    magnetic_dipole: MagneticDipoleConfig,
    steps: int = 2,
    h_step: float = 1.0e-3,
    startup_mode: StartupMode = StartupMode.COLD_START,
    radiation_reaction_mode: str = "off",
    external_field: ExternalFieldConfig | None = None,
):
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=startup_mode,
        radiation_reaction_mode=radiation_reaction_mode,
        external_field=external_field,
        magnetic_dipole=magnetic_dipole,
        self_consistency=SelfConsistencyConfig(enabled=False),
        use_numba=False,
    )


def test_neutral_neutron_gets_signed_rfs_impulse_in_vacuum_axial_gradient() -> None:
    neutron = get_species("neutron")
    assert neutron.magnetic_moment_j_t is not None
    gradient_t_m = 1.0e12
    h_step_ns = 1.0e-3
    field = ExternalFieldConfig(
        magnetic_field_gradient_t_per_m=(
            (-0.5 * gradient_t_m, 0.0, 0.0),
            (0.0, -0.5 * gradient_t_m, 0.0),
            (0.0, 0.0, gradient_t_m),
        )
    )

    trajectory, _, soa, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=_rfs_config(),
        h_step=h_step_ns,
        external_field=field,
    )

    assert soa is not None
    expected_force_newton = neutron.magnetic_moment_j_t * gradient_t_m
    expected_pz = expected_force_newton / NATIVE_FORCE_UNIT_NEWTON * h_step_ns
    assert float(soa.Pz[-1, 0]) == pytest.approx(expected_pz, rel=2.0e-15)
    assert float(soa.Pz[-1, 0]) < 0.0
    np.testing.assert_array_equal(soa.Px, 0.0)
    np.testing.assert_array_equal(soa.Py, 0.0)
    np.testing.assert_array_equal(soa.spin_x, 0.0)
    np.testing.assert_array_equal(soa.spin_y, 0.0)
    np.testing.assert_array_equal(soa.spin_z, 1.0)

    final_state = trajectory[-1]
    beta = np.array([final_state[axis][0] for axis in ("bx", "by", "bz")], dtype=float)
    gamma = float(final_state["gamma"][0])
    spin_four = boost_rest_polarization((0.0, 0.0, 1.0), beta)
    four_velocity = C_MMNS * np.concatenate(([gamma], gamma * beta))
    spin_magnitude = neutron.spin_quantum_number * HBAR_NATIVE

    # The stored rest spin has no spatial precession at B=0, but its boosted
    # time component must become nonzero once the gradient accelerates it.
    assert spin_four[0] < 0.0
    assert minkowski_dot(four_velocity, spin_four) / C_MMNS == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert spin_magnitude > 0.0
    assert minkowski_dot(spin_four, spin_four) == pytest.approx(-1.0, rel=2.0e-15)


def test_cold_start_moving_charge_drives_neutral_rfs_response() -> None:
    rider = _particle_state("neutron", position_mm=(0.0, 0.1, 0.0))
    driver = _particle_state("proton", beta=(0.1, 0.0, 0.0))

    _, _, soa, _, *_ = _run(
        rider=rider,
        driver=driver,
        magnetic_dipole=_rfs_config(),
        steps=24,
        h_step=1.0e-4,
    )

    assert soa is not None
    np.testing.assert_array_equal(soa.q_observer, 0.0)
    assert float(soa.local_magnetic_field_z_t[0, 0]) == 0.0
    assert np.max(soa.local_magnetic_field_z_t[:, 0]) > 0.0
    # For +Q moving along +x, a neutron at +y with spin +z and signed mu_n<0
    # is pushed toward +y by the full-G response once retarded history exists.
    assert float(soa.Py[-1, 0]) > 0.0
    assert np.count_nonzero(soa.Py[:, 0]) > 0


def test_rfs_off_is_an_exact_neutral_baseline() -> None:
    gradient_t_m = 1.0e12
    gradient_field = ExternalFieldConfig(
        magnetic_field_gradient_t_per_m=(
            (-0.5 * gradient_t_m, 0.0, 0.0),
            (0.0, -0.5 * gradient_t_m, 0.0),
            (0.0, 0.0, gradient_t_m),
        )
    )
    disabled = MagneticDipoleConfig(enabled=False)
    gradient_trajectory, _, _, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=disabled,
        external_field=gradient_field,
    )
    zero_trajectory, _, _, _, *_ = _run(
        rider=_particle_state("neutron"),
        driver=_empty_state(),
        magnetic_dipole=disabled,
        external_field=ExternalFieldConfig(),
    )

    for gradient_state, zero_state in zip(
        gradient_trajectory, zero_trajectory, strict=True
    ):
        for key in (
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
        ):
            np.testing.assert_array_equal(gradient_state[key], zero_state[key])


def test_rfs_rejects_dynamic_radiation_reaction() -> None:
    with pytest.raises(
        NotImplementedError, match="no validated radiation-reaction completion"
    ):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(),
            radiation_reaction_mode="medina_lad",
        )


def test_rfs_rejects_approximate_charge_source_history() -> None:
    with pytest.raises(ValueError, match="requires COLD_START"):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(),
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        )


def test_rfs_history_guard_includes_charged_rider_sources() -> None:
    reversed_config = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(species="proton", polarization=0.0),
        driver=MagneticDipoleParticleConfig(species="neutron", polarization=1.0),
    )

    with pytest.raises(ValueError, match="requires COLD_START"):
        _run(
            rider=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            driver=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            magnetic_dipole=reversed_config,
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        )


def test_rfs_rejects_shrunken_partial_polarization() -> None:
    with pytest.raises(
        ValueError, match="Partial polarization requires a weighted ensemble"
    ):
        _run(
            rider=_particle_state("neutron", position_mm=(0.0, 0.1, 0.0)),
            driver=_particle_state("proton", beta=(0.1, 0.0, 0.0)),
            magnetic_dipole=_rfs_config(rider_polarization=0.5),
        )
