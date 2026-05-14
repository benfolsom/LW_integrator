from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.external_fields import (
    ELEMENTARY_CHARGE_COULOMB,
    NATIVE_FORCE_UNIT_NEWTON,
    compute_uniform_external_field_impulse,
    electric_field_v_per_m_to_native,
)
from core.integration_runner import retarded_integrator
from core.types import ExternalFieldConfig, SimulationType, StartupMode


def _char_time(charge: float, mass: float) -> float:
    return (2.0 / 3.0) * charge**2 / (mass * C_MMNS**3)


def _single_particle_state(gamma: float = 2.0) -> dict:
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    return {
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "t": np.array([0.0]),
        "Px": np.array([0.0]),
        "Py": np.array([0.0]),
        "Pz": np.array([gamma * ELECTRON_MASS_AMU * C_MMNS * beta_z]),
        "Pt": np.array([gamma * ELECTRON_MASS_AMU * C_MMNS]),
        "gamma": np.array([gamma]),
        "bx": np.array([0.0]),
        "by": np.array([0.0]),
        "bz": np.array([beta_z]),
        "bdotx": np.array([0.0]),
        "bdoty": np.array([0.0]),
        "bdotz": np.array([0.0]),
        "q": np.array([-ELEMENTARY_CHARGE]),
        "m": np.array([ELECTRON_MASS_AMU]),
        "char_time": np.array([_char_time(ELEMENTARY_CHARGE, ELECTRON_MASS_AMU)]),
    }


def _empty_driver_state() -> dict:
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


def test_si_electric_field_converter_matches_native_force_units() -> None:
    expected = (
        ELEMENTARY_CHARGE_COULOMB
        / NATIVE_FORCE_UNIT_NEWTON
        / ELEMENTARY_CHARGE
    )

    assert electric_field_v_per_m_to_native(1.0) == pytest.approx(expected)
    assert electric_field_v_per_m_to_native(-1.5e9) == pytest.approx(
        -1.5e9 * expected
    )


def test_uniform_electric_field_impulse_uses_proper_time_lorentz_form() -> None:
    field = ExternalFieldConfig(electric_field_native=(0.0, 0.0, 10.0))
    charge = -ELEMENTARY_CHARGE
    gamma = 2.0
    beta = (0.0, 0.0, 0.5)
    h_step = 0.25

    delta_px, delta_py, delta_pz, delta_pt = compute_uniform_external_field_impulse(
        field,
        charge=charge,
        gamma=gamma,
        beta=beta,
        h_step=h_step,
        position=(0.0, 0.0, 0.0),
        time=0.0,
    )

    assert delta_px == pytest.approx(0.0)
    assert delta_py == pytest.approx(0.0)
    assert delta_pz == pytest.approx(h_step * charge * gamma * 10.0)
    assert delta_pt == pytest.approx(h_step * charge * gamma * 10.0 * beta[2])


def test_uniform_magnetic_field_impulse_bends_transversely() -> None:
    field = ExternalFieldConfig(magnetic_field_native=(0.0, 3.0, 0.0))
    charge = ELEMENTARY_CHARGE
    gamma = 4.0
    beta = (0.0, 0.0, 0.75)
    h_step = 0.5

    delta_px, delta_py, delta_pz, delta_pt = compute_uniform_external_field_impulse(
        field,
        charge=charge,
        gamma=gamma,
        beta=beta,
        h_step=h_step,
        position=(0.0, 0.0, 0.0),
        time=0.0,
    )

    assert delta_px == pytest.approx(h_step * charge * gamma * (-0.75 * 3.0))
    assert delta_py == pytest.approx(0.0)
    assert delta_pz == pytest.approx(0.0)
    assert delta_pt == pytest.approx(0.0)


def test_external_field_respects_spatial_temporal_window() -> None:
    field = ExternalFieldConfig(
        electric_field_native=(0.0, 0.0, 10.0),
        z_min=1.0,
        z_max=2.0,
        t_min=0.5,
        t_max=1.0,
    )

    inactive = compute_uniform_external_field_impulse(
        field,
        charge=ELEMENTARY_CHARGE,
        gamma=1.0,
        beta=(0.0, 0.0, 0.0),
        h_step=1.0,
        position=(0.0, 0.0, 0.5),
        time=0.75,
    )
    active = compute_uniform_external_field_impulse(
        field,
        charge=ELEMENTARY_CHARGE,
        gamma=1.0,
        beta=(0.0, 0.0, 0.0),
        h_step=1.0,
        position=(0.0, 0.0, 1.5),
        time=0.75,
    )

    assert inactive == (0.0, 0.0, 0.0, 0.0)
    assert active[2] != 0.0


def test_external_field_runs_through_numba_kernel_mode_and_soa() -> None:
    field = ExternalFieldConfig(electric_field_native=(0.0, 0.0, -100.0))
    no_field = ExternalFieldConfig(enabled=False)
    common_kwargs = dict(
        steps=4,
        h_step=0.001,
        wall_z=0.0,
        aperture_radius=1.0e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=_single_particle_state(),
        init_driver=_empty_driver_state(),
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1.0e9,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        use_numba=True,
    )

    baseline, _, baseline_soa, _ = retarded_integrator(
        **common_kwargs,
        external_field=no_field,
    )
    accelerated, _, accelerated_soa, _ = retarded_integrator(
        **common_kwargs,
        external_field=field,
    )

    assert baseline_soa is not None
    assert accelerated_soa is not None
    assert accelerated[-1]["Pz"][0] > baseline[-1]["Pz"][0]
    assert accelerated[-1]["Pt"][0] > baseline[-1]["Pt"][0]
    np.testing.assert_allclose(accelerated_soa.Pz[-1], accelerated[-1]["Pz"])
    np.testing.assert_allclose(accelerated_soa.Pt[-1], accelerated[-1]["Pt"])
