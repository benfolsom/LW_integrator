"""Production-seam tests for intrinsic-dipole Maxwell source fields."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    DipoleSourceConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    StartupMode,
)


def _particle_state(
    species_name: str,
    *,
    position_mm: tuple[float, float, float],
    beta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    macro_population: float = 1.0,
) -> dict[str, np.ndarray]:
    species = get_species(species_name)
    beta_vector = np.asarray(beta, dtype=float)
    gamma = 1.0 / np.sqrt(1.0 - float(beta_vector @ beta_vector))
    momentum = gamma * species.mass_amu * C_MMNS * beta_vector
    charge = float(species.charge_e) * ELEMENTARY_CHARGE
    return {
        "x": np.array([position_mm[0]]),
        "y": np.array([position_mm[1]]),
        "z": np.array([position_mm[2]]),
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
        "q": np.array([charge]),
        "q_species": np.array([charge]),
        "q_observer": np.array([charge]),
        "q_source": np.array([charge]),
        "macro_population": np.array([macro_population]),
        "m": np.array([species.mass_amu]),
        "m_species": np.array([species.mass_amu]),
        "char_time": np.array([0.0]),
    }


def _source_config(
    *,
    rider_species: str,
    driver_species: str,
    rider_polarization: float,
    driver_polarization: float,
    coupled_moment_force: bool,
    source_model: str = "covariant_retarded_point",
) -> MagneticDipoleConfig:
    return MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=coupled_moment_force,
        source=DipoleSourceConfig(
            model=source_model,
            minimum_separation_mm=2.0e-9,
            relative_stencil_step=5.0e-4,
        ),
        rider=MagneticDipoleParticleConfig(
            species=rider_species,
            rest_spin=(0.0, 0.0, 1.0),
            polarization=rider_polarization,
        ),
        driver=MagneticDipoleParticleConfig(
            species=driver_species,
            rest_spin=(0.0, 0.0, 1.0),
            polarization=driver_polarization,
        ),
    )


def _run(
    rider: dict[str, np.ndarray],
    driver: dict[str, np.ndarray],
    magnetic: MagneticDipoleConfig,
    *,
    steps: int,
    h_step: float,
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
        startup_mode=StartupMode.COLD_START,
        radiation_reaction_mode="off",
        magnetic_dipole=magnetic,
        self_consistency=SelfConsistencyConfig(enabled=False),
        use_numba=False,
    )


def test_charge_only_observer_feels_neutral_dipole_source() -> None:
    separation_mm = 1.0e-4
    rider = _particle_state(
        "electron",
        position_mm=(separation_mm, 0.0, 0.0),
        beta=(0.01, 0.0, 0.0),
    )
    driver = _particle_state("neutron", position_mm=(0.0, 0.0, 0.0))
    magnetic = _source_config(
        rider_species="electron",
        driver_species="neutron",
        rider_polarization=0.0,
        driver_polarization=1.0,
        coupled_moment_force=False,
    )

    _, _, rider_soa, _, *_ = _run(
        rider,
        driver,
        magnetic,
        steps=12,
        h_step=1.0e-7,
    )

    assert rider_soa is not None
    # At +x, signed mu_n<0 gives Bz>0.  v x B points -y and q_e<0,
    # so the ordinary qF_mu response points +y even though the observer's
    # own polarization is zero and no RFS moment force is active.
    assert float(rider_soa.Py[-1, 0]) > 0.0
    assert float(rider_soa.by[-1, 0]) > 0.0
    assert np.count_nonzero(rider_soa.Py[:, 0]) > 0


def test_source_off_does_not_enter_provider_or_change_kinematics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.dipole_source_interactions as source_interactions

    def _unexpected_provider_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("the dipole provider must stay cold when source=off")

    monkeypatch.setattr(
        source_interactions,
        "evaluate_retarded_dipole_source_interaction_native",
        _unexpected_provider_call,
    )
    rider = _particle_state(
        "electron",
        position_mm=(1.0e-4, 0.0, 0.0),
        beta=(0.01, 0.0, 0.0),
    )
    driver = _particle_state("neutron", position_mm=(0.0, 0.0, 0.0))
    magnetic = _source_config(
        rider_species="electron",
        driver_species="neutron",
        rider_polarization=0.0,
        driver_polarization=1.0,
        coupled_moment_force=False,
        source_model="off",
    )

    _, _, rider_soa, _, *_ = _run(
        rider,
        driver,
        magnetic,
        steps=6,
        h_step=1.0e-7,
    )

    assert rider_soa is not None
    np.testing.assert_array_equal(rider_soa.Py[:, 0], 0.0)
    np.testing.assert_array_equal(rider_soa.by[:, 0], 0.0)


def test_full_rfs_total_field_produces_mutual_neutral_dipole_force() -> None:
    separation_mm = 1.0e-8
    rider = _particle_state("neutron", position_mm=(separation_mm, 0.0, 0.0))
    driver = _particle_state("neutron", position_mm=(0.0, 0.0, 0.0))
    magnetic = _source_config(
        rider_species="neutron",
        driver_species="neutron",
        rider_polarization=1.0,
        driver_polarization=1.0,
        coupled_moment_force=True,
    )

    _, _, rider_soa, driver_soa, *_ = _run(
        rider,
        driver,
        magnetic,
        steps=16,
        h_step=1.0e-11,
    )

    assert rider_soa is not None
    assert driver_soa is not None
    # Equal parallel moments separated transverse to their common axis repel.
    assert float(rider_soa.Px[-1, 0]) > 0.0
    assert float(driver_soa.Px[-1, 0]) < 0.0
    assert float(rider_soa.Px[-1, 0]) == pytest.approx(
        -float(driver_soa.Px[-1, 0]),
        rel=5.0e-5,
    )


def test_retarded_dipole_source_rejects_macro_moment_scaling() -> None:
    rider = _particle_state(
        "electron",
        position_mm=(1.0e-4, 0.0, 0.0),
        beta=(0.01, 0.0, 0.0),
        macro_population=2.0,
    )
    driver = _particle_state("neutron", position_mm=(0.0, 0.0, 0.0))
    magnetic = _source_config(
        rider_species="electron",
        driver_species="neutron",
        rider_polarization=0.0,
        driver_polarization=1.0,
        coupled_moment_force=False,
    )

    with pytest.raises(ValueError, match="macro_population=1"):
        _run(rider, driver, magnetic, steps=2, h_step=1.0e-7)
