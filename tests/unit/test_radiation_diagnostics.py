from __future__ import annotations

import pytest

from core import equations
from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE


def test_radiation_mode_defaults_to_off_and_rejects_legacy_bdot() -> None:
    assert equations._canonicalize_radiation_reaction_mode(None) == "off"
    assert equations._canonicalize_radiation_reaction_mode("none") == "off"
    assert (
        equations._canonicalize_radiation_reaction_mode("power-damping")
        == "power_matched_damping"
    )

    with pytest.raises(ValueError):
        equations._canonicalize_radiation_reaction_mode("legacy_bdot")


def test_lienard_power_zero_for_unaccelerated_motion() -> None:
    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=(0.5, 0.0, 0.0),
        beta_dot_t=(0.0, 0.0, 0.0),
        gamma=1.2,
    )

    assert power == pytest.approx(0.0)


def test_lienard_power_uses_coordinate_time_beta_dot() -> None:
    gamma = 2.0
    beta = (0.5, 0.0, 0.0)
    beta_dot_t = (0.0, 3.0, 0.0)

    power = equations._compute_lienard_radiated_power(
        ELEMENTARY_CHARGE,
        beta=beta,
        beta_dot_t=beta_dot_t,
        gamma=gamma,
    )

    expected_transverse_term = 3.0**2 - (0.5 * 3.0) ** 2
    expected = (
        2.0
        * ELEMENTARY_CHARGE**2
        / (3.0 * C_MMNS)
        * gamma**6
        * expected_transverse_term
    )
    assert power == pytest.approx(expected)


def test_power_matched_damping_removes_energy_from_mechanical_momentum() -> None:
    gamma = 2.0
    mass = ELECTRON_MASS_AMU
    momentum_mag = mass * C_MMNS * (gamma**2 - 1.0) ** 0.5
    radiated_energy = 0.1 * mass * C_MMNS**2

    (new_px, new_py, new_pz), new_gamma, applied = (
        equations._apply_power_matched_radiation_damping(
            (momentum_mag, 0.0, 0.0),
            mass,
            gamma,
            radiated_energy,
        )
    )

    assert applied == pytest.approx(radiated_energy)
    assert new_gamma == pytest.approx(gamma - 0.1)
    assert new_py == pytest.approx(0.0)
    assert new_pz == pytest.approx(0.0)
    assert new_px < momentum_mag


def test_power_matched_damping_does_not_cross_rest_energy() -> None:
    gamma = 1.01
    mass = ELECTRON_MASS_AMU
    momentum_mag = mass * C_MMNS * (gamma**2 - 1.0) ** 0.5
    requested_energy = 10.0 * mass * C_MMNS**2

    (_, _, _), new_gamma, applied = equations._apply_power_matched_radiation_damping(
        (momentum_mag, 0.0, 0.0),
        mass,
        gamma,
        requested_energy,
    )

    assert new_gamma == pytest.approx(1.0)
    assert applied == pytest.approx((gamma - 1.0) * mass * C_MMNS**2)
