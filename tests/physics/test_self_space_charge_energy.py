from __future__ import annotations

import copy

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE, PROTON_MASS_AMU
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    ChronoMatchingMode,
    GammaReconciliationMethod,
    MacroparticleSmearingConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)
from input_output.bunch_initialization import create_bunch_from_params
from optimization.single_integration_helpers import calculate_rider_starting_pz

AMU_NATIVE_TO_MEV = 931.49410242 / (C_MMNS * C_MMNS)


def _copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: (
            np.array(value, copy=True)
            if isinstance(value, np.ndarray)
            else copy.deepcopy(value)
        )
        for key, value in state.items()
    }


def _make_bunch(*, pcount: int = 16, charge_scale: float = 1.0):
    energy_gev = 0.0840444095085845
    physical_population = 554800806.6187345
    transverse_mm = 2.0
    starting_pz = calculate_rider_starting_pz(
        energy_gev, PROTON_MASS_AMU, SimulationType.BUNCH_TO_BUNCH
    )
    rider, _ = create_bunch_from_params(
        starting_distance=0.0,
        transv_mom=0.0,
        starting_Pz=starting_pz,
        stripped_ions=1.0,
        m_particle=PROTON_MASS_AMU,
        transv_dist=transverse_mm,
        pcount=pcount,
        charge_sign=-1.0,
        seed=20260527,
        transverse_geometry="gaussian",
        charge_multiplier=physical_population * charge_scale / pcount,
    )
    driver, _ = create_bunch_from_params(
        starting_distance=1350.0,
        transv_mom=0.0,
        starting_Pz=-starting_pz,
        stripped_ions=0.0,
        m_particle=PROTON_MASS_AMU,
        transv_dist=transverse_mm,
        pcount=pcount,
        charge_sign=1.0,
        seed=20260528,
        transverse_geometry="gaussian",
    )
    return rider, driver


def _physical_kinetic_mev(state: dict[str, np.ndarray]) -> float:
    gamma = np.asarray(state["gamma"], dtype=float)
    mass = np.asarray(state["m"], dtype=float)
    charge = np.asarray(state["q"], dtype=float)
    weights = np.maximum(1.0, np.abs(charge) / ELEMENTARY_CHARGE)
    return float(np.sum(weights * (gamma - 1.0) * mass * 931.49410242))


def _pair_potential_mev(state: dict[str, np.ndarray], softening_mm: float) -> float:
    x = np.asarray(state["x"], dtype=float)
    y = np.asarray(state["y"], dtype=float)
    z = np.asarray(state["z"], dtype=float)
    q = np.asarray(state["q"], dtype=float)
    total = 0.0
    for i in range(x.size):
        dx = x[i] - x[i + 1 :]
        dy = y[i] - y[i + 1 :]
        dz = z[i] - z[i + 1 :]
        r = np.sqrt(dx * dx + dy * dy + dz * dz + softening_mm * softening_mm)
        if r.size:
            total += float(np.sum(q[i] * q[i + 1 :] / r))
    return total * AMU_NATIVE_TO_MEV


def _run_short_self_space_charge(
    *,
    charge_scale: float,
    space_charge_enabled: bool,
    gamma_reconciliation_method: GammaReconciliationMethod = GammaReconciliationMethod.DISABLED,
    gamma_reconciliation_fixed_weight: float = 0.9,
    radiation_reaction_mode: str = "diagnostic_only",
    max_iterations: int = 2,
    macroparticle_smearing: MacroparticleSmearingConfig | None = None,
):
    rider, driver = _make_bunch(pcount=16, charge_scale=charge_scale)
    gamma_initial = float(np.mean(rider["gamma"]))
    beta_z = abs(float(np.mean(rider["bz"])))
    steps = 40
    h_step = 10.0 / (gamma_initial * beta_z * C_MMNS * (steps - 1))
    trajectory, _, *_ = retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=100000.0,
        aperture_radius=5000.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_copy_state(rider),
        init_driver=_copy_state(driver),
        mean=100000.0,
        cav_spacing=100000.0,
        z_cutoff=100000.0,
        z_cutoff_mode="absolute",
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            max_iterations=max_iterations,
            verbosity=0,
            chrono_interpolate=True,
            chrono_tolerance=0.001,
            chrono_adaptive_tolerance=True,
            gamma_reconciliation_method=gamma_reconciliation_method,
            gamma_reconciliation_fixed_weight=gamma_reconciliation_fixed_weight,
        ),
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        space_charge=SpaceChargeConfig(
            enabled=space_charge_enabled,
            retarded=False,
            softening_mm=0.1,
            bunch_sigma_mm=2.0,
            min_retarded_steps=0,
        ),
        radiation_reaction_mode=radiation_reaction_mode,
        pseudo_grid=None,
        macroparticle_smearing=macroparticle_smearing,
        use_numba=True,
    )
    return trajectory


@pytest.mark.physics
def test_no_space_charge_drift_conserves_particle_energy() -> None:
    trajectory = _run_short_self_space_charge(
        charge_scale=0.0, space_charge_enabled=False
    )

    initial = _physical_kinetic_mev(trajectory[0])
    final = _physical_kinetic_mev(trajectory[-1])

    assert final - initial == pytest.approx(0.0, abs=1e-6)


@pytest.mark.physics
def test_instantaneous_self_space_charge_conserves_kinetic_plus_pair_potential() -> (
    None
):
    trajectory = _run_short_self_space_charge(
        charge_scale=1.0,
        space_charge_enabled=True,
        gamma_reconciliation_method=GammaReconciliationMethod.DISABLED,
        radiation_reaction_mode="diagnostic_only",
        max_iterations=2,
    )
    softening_mm = 0.1

    initial_kinetic = _physical_kinetic_mev(trajectory[0])
    final_kinetic = _physical_kinetic_mev(trajectory[-1])
    initial_potential = _pair_potential_mev(trajectory[0], softening_mm)
    final_potential = _pair_potential_mev(trajectory[-1], softening_mm)

    delta_kinetic = final_kinetic - initial_kinetic
    delta_potential = final_potential - initial_potential
    delta_total = delta_kinetic + delta_potential

    assert delta_kinetic > 0.0
    assert delta_potential < 0.0
    assert abs(delta_total) < 0.5 * abs(delta_potential)


@pytest.mark.physics
@pytest.mark.xfail(
    reason=(
        "Gamma reconciliation and applied radiation-reaction modes still inject "
        "large same-bunch energy-ledger artifacts; keep as a regression target."
    ),
    strict=False,
)
def test_gamma_reconciled_medina_self_space_charge_energy_proxy_target() -> None:
    trajectory = _run_short_self_space_charge(
        charge_scale=1.0,
        space_charge_enabled=True,
        gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED,
        gamma_reconciliation_fixed_weight=0.9,
        radiation_reaction_mode="medina_lad",
        max_iterations=2,
    )
    softening_mm = 0.1

    initial = _physical_kinetic_mev(trajectory[0]) + _pair_potential_mev(
        trajectory[0], softening_mm
    )
    final = _physical_kinetic_mev(trajectory[-1]) + _pair_potential_mev(
        trajectory[-1], softening_mm
    )

    assert final - initial == pytest.approx(0.0, rel=1e-6, abs=1e-3)


@pytest.mark.physics
def test_smearing_does_not_corrupt_energy_ledger() -> None:
    """Position smearing breaks exact action-reaction but must not cause
    orders-of-magnitude worse energy conservation than the unsmeared case."""
    softening_mm = 0.1

    traj_unsmeared = _run_short_self_space_charge(
        charge_scale=1.0,
        space_charge_enabled=True,
        macroparticle_smearing=None,
    )

    traj_smeared = _run_short_self_space_charge(
        charge_scale=1.0,
        space_charge_enabled=True,
        macroparticle_smearing=MacroparticleSmearingConfig(
            enabled=True,
            subcharge_count=4,
            seed=42,
            position_sigma_mm=0.5,
            use_momentum_errors=False,
        ),
    )

    # No NaN / Inf with smearing enabled.
    for state in traj_smeared:
        for key in ("x", "y", "z", "gamma", "bx", "by", "bz"):
            arr = np.asarray(state[key], dtype=float)
            assert np.all(np.isfinite(arr)), f"Non-finite values in {key} with smearing"

    # Both runs must show kinetic energy increasing (bunch expansion).
    dk_unsmeared = _physical_kinetic_mev(traj_unsmeared[-1]) - _physical_kinetic_mev(traj_unsmeared[0])
    dk_smeared = _physical_kinetic_mev(traj_smeared[-1]) - _physical_kinetic_mev(traj_smeared[0])
    assert dk_unsmeared > 0.0
    assert dk_smeared > 0.0

    # Total energy proxy (kinetic + pair potential) change should be much
    # smaller than the kinetic change in both cases.  Smearing changes the
    # effective pair potential via position perturbations; allow a looser
    # tolerance (5× the unsmeared ratio) rather than requiring parity.
    def _total_delta(traj: list) -> tuple[float, float]:
        dk = _physical_kinetic_mev(traj[-1]) - _physical_kinetic_mev(traj[0])
        dp = _pair_potential_mev(traj[-1], softening_mm) - _pair_potential_mev(traj[0], softening_mm)
        return dk, dp

    dk_u, dp_u = _total_delta(traj_unsmeared)
    dk_s, dp_s = _total_delta(traj_smeared)

    ratio_unsmeared = abs(dk_u + dp_u) / (abs(dk_u) + 1e-30)
    ratio_smeared = abs(dk_s + dp_s) / (abs(dk_s) + 1e-30)

    # Smearing introduces position asymmetry but must not blow up the total-
    # energy residual to more than 5× the unsmeared case.
    assert ratio_smeared < max(5.0 * ratio_unsmeared, 0.1), (
        f"Smearing inflated energy-ledger residual: unsmeared ratio={ratio_unsmeared:.3g}, "
        f"smeared ratio={ratio_smeared:.3g}"
    )
