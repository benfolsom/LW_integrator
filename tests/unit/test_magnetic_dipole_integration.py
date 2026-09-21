from __future__ import annotations

import copy

import numpy as np
import pytest

import core.vectorized_interactions as vectorized_interactions
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.external_fields import ExternalFieldConfig, magnetic_field_tesla_to_native
from core.integration_runner import retarded_integrator
from core.magnetic_dipole import (
    magnetic_moment_j_per_t_to_native,
    stern_gerlach_rest_impulse_native,
)
from core.self_consistency import SelfConsistencyConfig
from core.species import get_species
from core.types import (
    DriverTrainConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    PseudoGridConfig,
    SimulationType,
    StartupMode,
)


def _single_particle_state(
    *, mass_amu: float, charge_e: int = 0
) -> dict[str, np.ndarray]:
    charge = float(charge_e) * ELEMENTARY_CHARGE
    return {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([0.0]),
        "Pt": np.array([mass_amu * C_MMNS]),
        "gamma": np.array([1.0]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([0.0]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([charge]),
        "m": np.array([mass_amu]),
        "char_time": np.array([0.0]),
    }


def _empty_driver_state() -> dict[str, np.ndarray]:
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
            "m",
            "char_time",
        )
    }


def _run_neutron(
    *,
    steps: int,
    h_step: float,
    field: ExternalFieldConfig,
    magnetic: MagneticDipoleConfig,
    use_numba: bool = False,
    self_consistency: SelfConsistencyConfig | None = None,
):
    neutron = get_species("neutron")
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_single_particle_state(mass_amu=neutron.mass_amu),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=magnetic,
        self_consistency=self_consistency,
        use_numba=use_numba,
    )


def test_neutral_neutron_precesses_in_uniform_magnetic_field() -> None:
    neutron = get_species("neutron")
    field = ExternalFieldConfig(
        magnetic_field_native=(0.0, 0.0, magnetic_field_tesla_to_native(1.0))
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=True,
        stern_gerlach_force_enabled=False,
        rider=MagneticDipoleParticleConfig(
            species="neutron", rest_spin=(1.0, 0.0, 0.0)
        ),
    )

    trajectory, _, soa, _, *_ = _run_neutron(
        steps=11,
        h_step=1.0e-3,
        field=field,
        magnetic=magnetic,
    )

    assert soa is not None
    elapsed_s = float(trajectory[-1]["t"][0] - trajectory[0]["t"][0]) * 1.0e-9
    angle = -neutron.gyromagnetic_ratio_rad_s_t * elapsed_s
    assert float(trajectory[-1]["spin_x"][0]) == pytest.approx(np.cos(angle))
    assert float(trajectory[-1]["spin_y"][0]) == pytest.approx(np.sin(angle))
    assert float(trajectory[-1]["spin_z"][0]) == pytest.approx(0.0, abs=1.0e-15)
    spin_norm = np.sqrt(soa.spin_x**2 + soa.spin_y**2 + soa.spin_z**2)
    np.testing.assert_allclose(spin_norm, 1.0, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(soa.local_magnetic_field_z_t, 1.0)
    np.testing.assert_allclose(soa.Px, 0.0)
    np.testing.assert_allclose(soa.Py, 0.0)
    np.testing.assert_allclose(soa.Pz, 0.0)


def test_charged_electron_precesses_without_translation_when_at_rest() -> None:
    electron = get_species("electron")
    field = ExternalFieldConfig(
        magnetic_field_native=(0.0, 0.0, magnetic_field_tesla_to_native(1.0))
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        rider=MagneticDipoleParticleConfig(
            species="electron", rest_spin=(1.0, 0.0, 0.0)
        ),
    )

    trajectory, _, soa, _, *_ = retarded_integrator(
        steps=11,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_single_particle_state(
            mass_amu=electron.mass_amu, charge_e=electron.charge_e
        ),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=magnetic,
        use_numba=False,
    )

    assert soa is not None
    elapsed_s = float(trajectory[-1]["t"][0] - trajectory[0]["t"][0]) * 1.0e-9
    angle = -electron.gyromagnetic_ratio_rad_s_t * elapsed_s
    assert float(trajectory[-1]["spin_x"][0]) == pytest.approx(np.cos(angle))
    assert float(trajectory[-1]["spin_y"][0]) == pytest.approx(np.sin(angle))
    np.testing.assert_allclose(soa.Px, 0.0)
    np.testing.assert_allclose(soa.Py, 0.0)
    np.testing.assert_allclose(soa.Pz, 0.0)


def test_neutral_stern_gerlach_rest_gradient_changes_momentum() -> None:
    neutron = get_species("neutron")
    gradient = (
        (-5.0e5, 0.0, 0.0),
        (0.0, -5.0e5, 0.0),
        (0.0, 0.0, 1.0e6),
    )
    field = ExternalFieldConfig(magnetic_field_gradient_t_per_m=gradient)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(
            species="neutron", rest_spin=(0.0, 0.0, 1.0)
        ),
    )

    trajectory, _, _, _, *_ = _run_neutron(
        steps=2,
        h_step=1.0e-3,
        field=field,
        magnetic=magnetic,
    )

    expected = stern_gerlach_rest_impulse_native(
        (0.0, 0.0, float(neutron.magnetic_moment_j_t)),
        gradient,
        1.0e-3,
    )
    assert float(trajectory[-1]["Pz"][0]) == pytest.approx(expected[2])
    assert float(trajectory[-1]["Pz"][0]) < 0.0
    assert float(trajectory[-1]["Px"][0]) == pytest.approx(0.0)
    assert float(trajectory[-1]["Py"][0]) == pytest.approx(0.0)


def test_fixed_geometry_keeps_prescribed_gradient_at_start_position() -> None:
    gradient = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0e12, 0.0, 0.0),
    )
    field = ExternalFieldConfig(magnetic_field_gradient_t_per_m=gradient)
    magnetic = MagneticDipoleConfig(
        enabled=True,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(
            species="neutron", rest_spin=(1.0, 0.0, 1.0)
        ),
    )
    common = dict(
        steps=2,
        h_step=1.0e-3,
        field=field,
        magnetic=magnetic,
    )

    fixed, *_ = _run_neutron(
        **common,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="fixed_geometry",
            max_iterations=2,
        ),
    )
    neutron = get_species("neutron")
    variable, *_ = retarded_integrator(
        steps=2,
        h_step=1.0e-3,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_single_particle_state(mass_amu=neutron.mass_amu),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=magnetic,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="variable_geometry",
            max_iterations=2,
        ),
        use_numba=False,
    )

    assert float(fixed[-1]["spin_y"][0]) == 0.0
    assert float(variable[-1]["spin_y"][0]) != 0.0
    assert float(fixed[-1]["local_magnetic_field_z_t"][0]) == pytest.approx(
        1.0e12 * float(fixed[-1]["x"][0]) * 1.0e-3
    )
    assert float(variable[-1]["local_magnetic_field_z_t"][0]) == pytest.approx(
        1.0e12 * float(variable[-1]["x"][0]) * 1.0e-3
    )


def test_static_rest_gradient_rejects_relativistic_particle() -> None:
    neutron = get_species("neutron")
    state = _single_particle_state(mass_amu=neutron.mass_amu)
    beta = 0.1
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    state["Pz"][:] = gamma * neutron.mass_amu * C_MMNS * beta
    state["Pt"][:] = gamma * neutron.mass_amu * C_MMNS
    state["gamma"][:] = gamma
    state["bz"][:] = beta
    gradient = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0e9, 0.0, 0.0),
    )

    with pytest.raises(NotImplementedError, match=r"\|beta\| <= 0.01"):
        retarded_integrator(
            steps=2,
            h_step=1.0e-3,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=state,
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
            radiation_reaction_mode="off",
            external_field=ExternalFieldConfig(
                magnetic_field_gradient_t_per_m=gradient
            ),
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True,
                stern_gerlach_force_enabled=True,
                spin_model="bmt_frenkel",
                stern_gerlach_model="static_rest_gradient",
                rider=MagneticDipoleParticleConfig(
                    species="neutron", rest_spin=(0.0, 0.0, 1.0)
                ),
            ),
            use_numba=False,
        )


def test_static_rest_gradient_rejects_step_that_exits_low_beta_domain() -> None:
    neutron = get_species("neutron")
    gradient = (
        (-5.0e18, 0.0, 0.0),
        (0.0, -5.0e18, 0.0),
        (0.0, 0.0, 1.0e19),
    )

    with pytest.raises(NotImplementedError, match="this step would span"):
        retarded_integrator(
            steps=2,
            h_step=1.0e-3,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_single_particle_state(mass_amu=neutron.mass_amu),
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
            startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
            radiation_reaction_mode="off",
            external_field=ExternalFieldConfig(
                magnetic_field_gradient_t_per_m=gradient
            ),
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True,
                stern_gerlach_force_enabled=True,
                spin_model="bmt_frenkel",
                stern_gerlach_model="static_rest_gradient",
                rider=MagneticDipoleParticleConfig(
                    species="neutron", rest_spin=(0.0, 0.0, 1.0)
                ),
            ),
            self_consistency=SelfConsistencyConfig(enabled=False),
            use_numba=False,
        )


@pytest.mark.skipif(
    not vectorized_interactions.NUMBA_AVAILABLE,
    reason="Numba is unavailable, so distinct force kernels cannot be selected",
)
def test_magnetic_dipole_path_is_identical_across_forced_python_and_numba_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    electron = get_species("electron")
    proton = get_species("proton")
    field = ExternalFieldConfig(
        magnetic_field_native=(0.0, 0.0, magnetic_field_tesla_to_native(0.75))
    )
    magnetic = MagneticDipoleConfig(
        enabled=True,
        rider=MagneticDipoleParticleConfig(
            species="electron", rest_spin=(1.0, 0.0, 0.0)
        ),
        driver=MagneticDipoleParticleConfig(
            species="proton", rest_spin=(0.0, 1.0, 0.0)
        ),
    )
    rider = _single_particle_state(
        mass_amu=electron.mass_amu, charge_e=electron.charge_e
    )
    rider["z"][:] = -1.0
    driver = _single_particle_state(mass_amu=proton.mass_amu, charge_e=proton.charge_e)
    driver["z"][:] = 1.0
    common = dict(
        steps=3,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.COLD_START,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=magnetic,
        use_numba=True,
    )

    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    python_rider, python_driver, *_ = retarded_integrator(
        init_rider=copy.deepcopy(rider), **common
    )
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", True)
    numba_rider, numba_driver, *_ = retarded_integrator(
        init_rider=copy.deepcopy(rider), **common
    )

    for python_trajectory, numba_trajectory in (
        (python_rider, numba_rider),
        (python_driver, numba_driver),
    ):
        for python_state, numba_state in zip(python_trajectory, numba_trajectory):
            for key in (
                "x",
                "y",
                "z",
                "Px",
                "Py",
                "Pz",
                "Pt",
                "gamma",
                "spin_x",
                "spin_y",
                "spin_z",
            ):
                np.testing.assert_array_equal(numba_state[key], python_state[key])


def test_observer_moment_is_not_scaled_by_macroparticle_population() -> None:
    electron = get_species("electron")
    state = _single_particle_state(
        mass_amu=electron.mass_amu, charge_e=electron.charge_e
    )
    population = 1.0e12
    state["q_species"] = np.array([-ELEMENTARY_CHARGE])
    state["q_observer"] = np.array([-ELEMENTARY_CHARGE])
    state["q_source"] = np.array([-ELEMENTARY_CHARGE * population])
    state["macro_population"] = np.array([population])

    trajectory, *_ = retarded_integrator(
        steps=2,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=state,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            rider=MagneticDipoleParticleConfig(species="electron"),
        ),
        use_numba=False,
    )

    assert float(trajectory[0]["magnetic_moment_j_per_t"][0]) == pytest.approx(
        electron.magnetic_moment_j_t
    )
    assert float(trajectory[0]["magnetic_moment_native"][0]) == pytest.approx(
        magnetic_moment_j_per_t_to_native(electron.magnetic_moment_j_t)
    )
    assert float(trajectory[0]["magnetic_moment_native"][0]) < 0.0


def test_named_magnetic_species_rejects_mismatched_particle_mass_or_charge() -> None:
    proton_state = _single_particle_state(
        mass_amu=get_species("proton").mass_amu,
        charge_e=get_species("proton").charge_e,
    )

    with pytest.raises(ValueError, match="rider magnetic species 'electron' expects"):
        retarded_integrator(
            steps=2,
            h_step=1.0e-6,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=proton_state,
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True,
                rider=MagneticDipoleParticleConfig(species="electron"),
            ),
        )


def test_enabled_run_does_not_leak_spin_state_into_reused_disabled_input() -> None:
    electron = get_species("electron")
    state = _single_particle_state(
        mass_amu=electron.mass_amu, charge_e=electron.charge_e
    )
    kwargs = dict(
        steps=2,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        use_numba=False,
    )

    enabled, *_ = retarded_integrator(
        init_rider=state,
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            rider=MagneticDipoleParticleConfig(species="electron"),
        ),
        **kwargs,
    )
    disabled, *_ = retarded_integrator(
        init_rider=state,
        magnetic_dipole=MagneticDipoleConfig(enabled=False),
        **kwargs,
    )

    assert "spin_x" in enabled[0]
    assert "spin_x" not in state
    assert "spin_x" not in disabled[0]


def test_disabled_restart_strips_magnetic_state_from_enabled_output() -> None:
    electron = get_species("electron")
    field = ExternalFieldConfig(
        magnetic_field_native=(0.0, 0.0, magnetic_field_tesla_to_native(1.0))
    )
    common = dict(
        steps=2,
        h_step=1.0e-6,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=field,
        use_numba=False,
    )
    enabled, *_ = retarded_integrator(
        init_rider=_single_particle_state(
            mass_amu=electron.mass_amu, charge_e=electron.charge_e
        ),
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            rider=MagneticDipoleParticleConfig(
                species="electron", rest_spin=(1.0, 0.0, 0.0)
            ),
        ),
        **common,
    )

    restarted, *_ = retarded_integrator(
        init_rider=enabled[-1],
        magnetic_dipole=MagneticDipoleConfig(enabled=False),
        **common,
    )

    assert "spin_x" in enabled[-1]
    assert "spin_x" not in restarted[0]
    assert "magnetic_dipole_active" not in restarted[0]


def test_disabled_dipoles_leave_legacy_state_shape_and_values_unchanged() -> None:
    state = _single_particle_state(
        mass_amu=get_species("electron").mass_amu, charge_e=-1
    )
    kwargs = dict(
        steps=3,
        h_step=1.0e-4,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=ExternalFieldConfig(
            enabled=True,
            magnetic_field_native=(
                0.0,
                0.0,
                magnetic_field_tesla_to_native(1.0),
            ),
        ),
        use_numba=False,
    )

    baseline, *_ = retarded_integrator(init_rider=copy.deepcopy(state), **kwargs)
    disabled, *_ = retarded_integrator(
        init_rider=copy.deepcopy(state),
        magnetic_dipole=MagneticDipoleConfig(enabled=False),
        **kwargs,
    )

    assert baseline[-1].keys() == disabled[-1].keys()
    for key in baseline[-1]:
        if isinstance(baseline[-1][key], np.ndarray):
            np.testing.assert_array_equal(baseline[-1][key], disabled[-1][key])


def test_enabled_stern_gerlach_with_zero_gradient_is_exact_kinematic_noop() -> None:
    state = _single_particle_state(
        mass_amu=get_species("electron").mass_amu, charge_e=-1
    )
    kwargs = dict(
        steps=3,
        h_step=1.0e-4,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=ExternalFieldConfig(
            enabled=True,
            magnetic_field_native=(
                0.0,
                0.0,
                magnetic_field_tesla_to_native(1.0),
            ),
        ),
        use_numba=False,
    )
    spin_only = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=False,
        rider=MagneticDipoleParticleConfig(species="electron"),
    )
    zero_gradient = MagneticDipoleConfig(
        enabled=True,
        spin_precession_enabled=False,
        stern_gerlach_force_enabled=True,
        rider=MagneticDipoleParticleConfig(species="electron"),
    )

    without_sg, *_ = retarded_integrator(
        init_rider=copy.deepcopy(state), magnetic_dipole=spin_only, **kwargs
    )
    with_sg, *_ = retarded_integrator(
        init_rider=copy.deepcopy(state), magnetic_dipole=zero_gradient, **kwargs
    )

    for step_without, step_with in zip(without_sg, with_sg):
        for key in ("x", "y", "z", "t", "Px", "Py", "Pz", "Pt", "gamma"):
            np.testing.assert_array_equal(step_with[key], step_without[key])


@pytest.mark.parametrize("spin_quantum_number", [float("nan"), float("inf")])
def test_magnetic_particle_config_rejects_nonfinite_spin_quantum_number(
    spin_quantum_number: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MagneticDipoleParticleConfig(
            species="custom",
            magnetic_moment_j_per_t=1.0e-26,
            spin_quantum_number=spin_quantum_number,
        )


def test_pseudo_grid_rejects_magnetic_spin_reconstruction() -> None:
    neutron = get_species("neutron")
    with pytest.raises(NotImplementedError, match="spin reconstruction"):
        retarded_integrator(
            steps=2,
            h_step=1.0e-3,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_single_particle_state(mass_amu=neutron.mass_amu),
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
            pseudo_grid=PseudoGridConfig(enabled=True),
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
        )


def test_integrator_rejects_nonpositive_proper_time_step() -> None:
    neutron = get_species("neutron")
    with pytest.raises(ValueError, match="h_step must be finite and positive"):
        retarded_integrator(
            steps=2,
            h_step=-1.0e-3,
            wall_z=0.0,
            aperture_radius=1.0e9,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_single_particle_state(mass_amu=neutron.mass_amu),
            init_driver=_empty_driver_state(),
            mean=0.0,
            cav_spacing=0.0,
            z_cutoff=1.0e9,
        )


def test_driver_train_prehistory_local_field_matches_each_synthetic_state() -> None:
    neutron = get_species("neutron")
    rider = _single_particle_state(mass_amu=neutron.mass_amu)
    driver = _single_particle_state(mass_amu=neutron.mass_amu)
    beta = -1.0e-3
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    driver["z"][:] = 10.0
    driver["Pz"][:] = gamma * neutron.mass_amu * C_MMNS * beta
    driver["Pt"][:] = gamma * neutron.mass_amu * C_MMNS
    driver["gamma"][:] = gamma
    driver["bz"][:] = beta
    field = ExternalFieldConfig(
        magnetic_field_gradient_t_per_m=(
            (-0.5, 0.0, 0.0),
            (0.0, -0.5, 0.0),
            (0.0, 0.0, 1.0),
        )
    )

    _, driver_trajectory, _, driver_soa, *_ = retarded_integrator(
        steps=2,
        h_step=1.0,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=rider,
        init_driver=driver,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        radiation_reaction_mode="off",
        external_field=field,
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            stern_gerlach_force_enabled=False,
            rider=MagneticDipoleParticleConfig(species="neutron"),
            driver=MagneticDipoleParticleConfig(species="neutron"),
        ),
        driver_train=DriverTrainConfig(
            enabled=True,
            bunch_count=2,
            z_offsets_mm=(0.0, 2.0),
            prehistory_steps=2,
            preserve_prehistory_in_output=True,
        ),
        use_numba=False,
    )

    assert driver_soa is not None
    expected_bz_t = np.asarray(driver_trajectory[0]["z"]) * 1.0e-3
    np.testing.assert_allclose(
        driver_trajectory[0]["local_magnetic_field_z_t"], expected_bz_t
    )
    np.testing.assert_allclose(driver_soa.local_magnetic_field_z_t[0], expected_bz_t)
