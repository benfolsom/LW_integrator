"""Same-bunch fields on the exact (RFS) path with INERTIAL_PREHISTORY."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)
from tests.unit.test_inertial_prehistory import _species_state, _state

pytestmark = pytest.mark.unit

_NO_MOMENT = MagneticDipoleParticleConfig(
    species="custom", magnetic_moment_j_per_t=0.0, spin_quantum_number=0.5
)
_MAGNETIC = MagneticDipoleConfig(
    enabled=True,
    spin_precession_enabled=True,
    stern_gerlach_force_enabled=False,
    rider=_NO_MOMENT,
    driver=_NO_MOMENT,
)


def _merge(*states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    merged = {}
    for key in states[0]:
        merged[key] = np.concatenate([np.atleast_1d(s[key]) for s in states])
    return merged


def _run(
    rider,
    driver,
    *,
    steps,
    h_step,
    space_charge=None,
    startup=StartupMode.INERTIAL_PREHISTORY,
):
    return retarded_integrator(
        steps=steps,
        h_step=h_step,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=copy.deepcopy(rider),
        init_driver=copy.deepcopy(driver),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=0.0,
        startup_mode=startup,
        radiation_reaction_mode="off",
        magnetic_dipole=_MAGNETIC,
        space_charge=space_charge,
        self_consistency=SelfConsistencyConfig(enabled=False),
        use_numba=False,
    )


def _final_positions(trajectory) -> np.ndarray:
    last = trajectory[-1]
    return np.column_stack([np.asarray(last[k], dtype=float) for k in ("x", "y", "z")])


def _pair(separation_mm: float):
    electron = _species_state(
        "electron", position_mm=(-separation_mm / 2, 0.0, 0.0), beta=(0.0, 1.0e-3, 0.0)
    )
    proton = _species_state("proton", position_mm=(separation_mm / 2, 0.0, 0.0))
    # Uncharged spectator (the joint run needs a driver bunch). Keep it close:
    # the inertial prehistory knot spacing follows the largest separation, and
    # coarse prehistory segments smear the t = 0 acceleration onset.
    spectator = _state(
        position_mm=(0.0, 0.0, 2.0 * separation_mm),
        mass_amu=1.0,
        observer_charge=0.0,
        source_charge=0.0,
    )
    return electron, proton, spectator


def test_joint_bunch_matches_split_rider_driver_pair() -> None:
    """Two charges as one bunch (new path) equal them as rider + driver."""
    electron, proton, spectator = _pair(1.0e-7)
    steps, h = 9, 2.5e-12
    split_rider, split_driver, *_ = _run(electron, proton, steps=steps, h_step=h)
    joint_rider, _, *_ = _run(
        _merge(electron, proton),
        spectator,
        steps=steps,
        h_step=h,
        space_charge=SpaceChargeConfig(enabled=True),
    )
    split = np.vstack((_final_positions(split_rider), _final_positions(split_driver)))
    joint = _final_positions(joint_rider)
    start = np.array([[-0.5e-7, 0.0, 0.0], [0.5e-7, 0.0, 0.0]])
    moved = np.max(np.abs(split - start))
    assert moved > 0.0
    np.testing.assert_allclose(joint, split, rtol=0.0, atol=1e-9 * moved + 1e-24)


def test_close_pair_inside_one_light_step_runs_and_converges() -> None:
    """Bunch-mates closer than c h need the extrapolated retarded point."""
    electron, proton, spectator = _pair(2.0e-10)
    horizon = 4.0e-12
    finals = []
    for intervals in (4, 8, 16):
        rider, _, *_ = _run(
            _merge(electron, proton),
            spectator,
            steps=intervals + 1,
            h_step=horizon / intervals,
            space_charge=SpaceChargeConfig(enabled=True),
        )
        finals.append(_final_positions(rider))
    coarse = np.max(np.abs(finals[0] - finals[2]))
    medium = np.max(np.abs(finals[1] - finals[2]))
    assert np.all(np.isfinite(finals[2]))
    assert medium < coarse


def test_same_bunch_exact_fields_reject_cold_start() -> None:
    electron, proton, spectator = _pair(1.0e-7)
    with pytest.raises(NotImplementedError, match="INERTIAL_PREHISTORY"):
        _run(
            _merge(electron, proton),
            spectator,
            steps=3,
            h_step=2.5e-12,
            space_charge=SpaceChargeConfig(enabled=True),
            startup=StartupMode.COLD_START,
        )


def test_exact_retarded_bunch_default_clock_is_bit_identical_to_legacy() -> None:
    """Exact same-bunch forces never use instantaneous startup or fallback."""
    electron, proton, spectator = _pair(1.0e-7)
    histories = []
    for clock in ("lab_time", "legacy_same_index"):
        histories.append(
            _run(
                _merge(electron, proton),
                spectator,
                steps=9,
                h_step=2.5e-12,
                space_charge=SpaceChargeConfig(enabled=True, instantaneous_clock=clock),
            )[:2]
        )
    for lab_role, legacy_role in zip(*histories):
        for lab, legacy in zip(lab_role, legacy_role):
            for key in lab:
                if isinstance(lab[key], np.ndarray):
                    np.testing.assert_array_equal(lab[key], legacy[key], err_msg=key)
