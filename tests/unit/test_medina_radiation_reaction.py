"""Focused validation of the complete Medina force-derivative kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.external_fields import AMU_KG
from core.medina_radiation_reaction import (
    compute_medina_radiation_reaction,
    medina_response_time,
)

_NATIVE_ENERGY_UNIT_J = AMU_KG * (1.0e-3) ** 2 / (1.0e-9) ** 2
_ELECTRON_VOLT_J = 1.602176634e-19


def _native_energy_from_ev(value_ev: float) -> float:
    return float(value_ev) * _ELECTRON_VOLT_J / _NATIVE_ENERGY_UNIT_J


def _ev_from_native_energy(value_native: float) -> float:
    return float(value_native) * _NATIVE_ENERGY_UNIT_J / _ELECTRON_VOLT_J


def test_response_time_uses_native_gaussian_coefficient_and_charge_squared() -> None:
    expected = 2.0 * ELEMENTARY_CHARGE**2 / (3.0 * ELECTRON_MASS_AMU * C_MMNS**3)

    positive = medina_response_time(
        charge=ELEMENTARY_CHARGE,
        mass=ELECTRON_MASS_AMU,
    )
    negative = medina_response_time(
        charge=-ELEMENTARY_CHARGE,
        mass=ELECTRON_MASS_AMU,
    )

    assert positive == pytest.approx(expected)
    assert negative == pytest.approx(positive)


def test_zero_charge_returns_zero_force_work_and_radiation() -> None:
    result = compute_medina_radiation_reaction(
        external_force=(3.0, -2.0, 1.0),
        external_force_time_derivative=(5.0, 7.0, -11.0),
        beta=(0.2, -0.1, 0.3),
        acceleration=(4.0, 5.0, -6.0),
        gamma=1.0 / math.sqrt(1.0 - 0.14),
        mass=ELECTRON_MASS_AMU,
        charge=0.0,
        coordinate_dt=0.25,
    )

    assert result.radiation_reaction_force == (0.0, 0.0, 0.0)
    assert result.radiation_reaction_impulse == (0.0, 0.0, 0.0)
    assert result.reaction_work == 0.0
    assert result.far_radiated_energy == 0.0
    assert result.cross_field_energy == 0.0
    assert result.energy_balance_residual == 0.0


def test_force_derivative_is_present_in_rest_frame() -> None:
    force_derivative = (2.5, -4.0, 1.25)
    coordinate_dt = 0.125
    response_time = medina_response_time(charge=1.5, mass=2.0)

    result = compute_medina_radiation_reaction(
        external_force=(7.0, 8.0, 9.0),
        external_force_time_derivative=force_derivative,
        beta=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        gamma=1.0,
        mass=2.0,
        charge=-1.5,
        coordinate_dt=coordinate_dt,
    )

    expected_force = tuple(response_time * value for value in force_derivative)
    expected_impulse = tuple(value * coordinate_dt for value in expected_force)
    assert result.gamma_force_time_derivative == pytest.approx(force_derivative)
    assert result.radiation_reaction_force == pytest.approx(expected_force)
    assert result.radiation_reaction_impulse == pytest.approx(expected_impulse)
    assert result.reaction_power == 0.0
    assert result.far_radiated_power == 0.0


def test_constant_longitudinal_force_has_schott_boundary_power_not_drag() -> None:
    gamma = 10.0
    beta_z = math.sqrt(1.0 - 1.0 / gamma**2)
    force_z = 5.0
    mass = 2.0
    acceleration_z = force_z / (mass * gamma**3)

    result = compute_medina_radiation_reaction(
        external_force=(0.0, 0.0, force_z),
        external_force_time_derivative=(0.0, 0.0, 0.0),
        beta=(0.0, 0.0, beta_z),
        acceleration=(0.0, 0.0, acceleration_z),
        gamma=gamma,
        mass=mass,
        charge=1.0,
        coordinate_dt=1.0,
    )

    assert result.radiation_reaction_force == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-24
    )
    assert result.reaction_power == pytest.approx(0.0, abs=1.0e-21)
    assert result.far_radiated_power > 0.0
    assert result.cross_field_energy_rate == pytest.approx(
        -result.far_radiated_power,
        rel=2.0e-14,
    )
    assert result.energy_balance_residual == pytest.approx(0.0, abs=1.0e-20)


def test_force_pulse_recovers_larmor_after_boundary_accounting() -> None:
    sample_count = 4001
    duration = 1.0
    angular_frequency = 2.0 * math.pi / duration
    acceleration_amplitude = 1.0
    times = np.linspace(0.0, duration, sample_count)
    reaction_powers = np.empty(sample_count)
    radiated_powers = np.empty(sample_count)
    cross_energies = np.empty(sample_count)

    for index, time_value in enumerate(times):
        phase = angular_frequency * time_value
        acceleration_x = acceleration_amplitude * math.sin(phase)
        acceleration_derivative_x = (
            acceleration_amplitude * angular_frequency * math.cos(phase)
        )
        velocity_x = (
            acceleration_amplitude / angular_frequency * (1.0 - math.cos(phase))
        )
        beta_x = velocity_x / C_MMNS
        gamma = 1.0 / math.sqrt(1.0 - beta_x**2)
        gamma_derivative = gamma**3 * beta_x * acceleration_x / C_MMNS

        # Exact longitudinal dp/dt and its complete lab-time derivative.
        force_x = ELECTRON_MASS_AMU * gamma**3 * acceleration_x
        force_derivative_x = ELECTRON_MASS_AMU * (
            3.0 * gamma**2 * gamma_derivative * acceleration_x
            + gamma**3 * acceleration_derivative_x
        )
        result = compute_medina_radiation_reaction(
            external_force=(force_x, 0.0, 0.0),
            external_force_time_derivative=(force_derivative_x, 0.0, 0.0),
            beta=(beta_x, 0.0, 0.0),
            acceleration=(acceleration_x, 0.0, 0.0),
            gamma=gamma,
            mass=ELECTRON_MASS_AMU,
            charge=ELEMENTARY_CHARGE,
            coordinate_dt=0.0,
        )
        reaction_powers[index] = result.reaction_power
        radiated_powers[index] = result.far_radiated_power
        cross_energies[index] = result.cross_field_energy

    reaction_work = float(np.trapezoid(reaction_powers, times))
    radiated_energy = float(np.trapezoid(radiated_powers, times))
    larmor_energy = float(
        2.0
        * ELEMENTARY_CHARGE**2
        / (3.0 * C_MMNS**3)
        * acceleration_amplitude**2
        * duration
        / 2.0
    )

    assert radiated_energy == pytest.approx(larmor_energy, rel=1.0e-6)
    assert reaction_work == pytest.approx(-radiated_energy, rel=1.0e-10)
    assert cross_energies[0] == pytest.approx(0.0, abs=1.0e-30)
    assert cross_energies[-1] == pytest.approx(0.0, abs=1.0e-25)


def test_complete_derivative_recovers_half_mev_hyperbolic_capture_loss() -> None:
    """Regression for the proposed 0.5 meV, 10 pm electron flyby.

    This integrates Medina's instantaneous diagnostics on the unperturbed
    nonrelativistic Coulomb hyperbola.  It is an analytic-orbit benchmark, not
    an integration claim: the expected charge-radiation loss is about
    0.757 meV, while omitting ``gamma dF_ext/dt`` leaves only 0.0411 percent.
    """

    energy_infinity = _native_energy_from_ev(0.5e-3)
    periapsis = 10.0e-9  # 10 pm in mm
    coulomb_coupling = ELEMENTARY_CHARGE**2
    speed_infinity = math.sqrt(2.0 * energy_infinity / ELECTRON_MASS_AMU)
    speed_periapsis = math.sqrt(
        speed_infinity**2 + 2.0 * coulomb_coupling / (ELECTRON_MASS_AMU * periapsis)
    )
    specific_angular_momentum = periapsis * speed_periapsis
    semi_latus_rectum = (
        specific_angular_momentum**2 * ELECTRON_MASS_AMU / coulomb_coupling
    )
    eccentricity = semi_latus_rectum / periapsis - 1.0
    asymptotic_anomaly = math.acos(-1.0 / eccentricity)

    nodes, weights = np.polynomial.legendre.leggauss(512)
    radiated_integrand = np.empty(nodes.size)
    reaction_integrand = np.empty(nodes.size)
    truncated_integrand = np.empty(nodes.size)
    response_time = medina_response_time(
        charge=ELEMENTARY_CHARGE,
        mass=ELECTRON_MASS_AMU,
    )

    for index, unit_node in enumerate(nodes):
        anomaly = asymptotic_anomaly * float(unit_node)
        cosine = math.cos(anomaly)
        sine = math.sin(anomaly)
        radial_unit = np.asarray((cosine, sine, 0.0))
        transverse_unit = np.asarray((-sine, cosine, 0.0))
        radius = semi_latus_rectum / (1.0 + eccentricity * cosine)
        radial_speed = (
            coulomb_coupling
            * eccentricity
            * sine
            / (ELECTRON_MASS_AMU * specific_angular_momentum)
        )
        angular_speed = specific_angular_momentum / radius**2
        velocity = (
            radial_speed * radial_unit
            + specific_angular_momentum / radius * transverse_unit
        )
        beta = velocity / C_MMNS
        gamma = 1.0 / math.sqrt(1.0 - float(beta @ beta))
        force = -coulomb_coupling * radial_unit / radius**2
        acceleration = force / ELECTRON_MASS_AMU
        force_derivative = -coulomb_coupling * (
            transverse_unit * angular_speed / radius**2
            - 2.0 * radial_unit * radial_speed / radius**3
        )

        result = compute_medina_radiation_reaction(
            external_force=force,
            external_force_time_derivative=force_derivative,
            beta=beta,
            acceleration=acceleration,
            gamma=gamma,
            mass=ELECTRON_MASS_AMU,
            charge=-ELEMENTARY_CHARGE,
            coordinate_dt=0.0,
        )
        dt_d_anomaly = radius**2 / specific_angular_momentum
        radiated_integrand[index] = result.far_radiated_power * dt_d_anomaly
        reaction_integrand[index] = result.reaction_power * dt_d_anomaly

        # Reproduce the superseded omission for a quantitative regression.
        radiated_momentum_rate = np.asarray(result.radiated_momentum_rate)
        truncated_force = response_time * (
            result.gamma_time_derivative * force
            - radiated_momentum_rate / response_time
        )
        truncated_integrand[index] = float(truncated_force @ velocity) * dt_d_anomaly

    quadrature_weights = asymptotic_anomaly * weights
    radiated_energy = float(quadrature_weights @ radiated_integrand)
    reaction_work = float(quadrature_weights @ reaction_integrand)
    truncated_work = float(quadrature_weights @ truncated_integrand)
    radiated_mev = 1.0e3 * _ev_from_native_energy(radiated_energy)

    assert radiated_mev == pytest.approx(0.7571, rel=2.0e-4)
    assert reaction_work == pytest.approx(-radiated_energy, rel=2.0e-13)
    assert abs(truncated_work / reaction_work) == pytest.approx(
        4.1097e-4,
        rel=2.0e-4,
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mass": 0.0}, "mass"),
        ({"gamma": 0.5}, "gamma"),
        ({"coordinate_dt": -1.0}, "coordinate_dt"),
        ({"beta": (1.1, 0.0, 0.0)}, "beta"),
        ({"external_force": (1.0, 2.0)}, "external_force"),
    ],
)
def test_invalid_inputs_fail_explicitly(overrides: dict, message: str) -> None:
    arguments = {
        "external_force": (0.0, 0.0, 0.0),
        "external_force_time_derivative": (0.0, 0.0, 0.0),
        "beta": (0.0, 0.0, 0.0),
        "acceleration": (0.0, 0.0, 0.0),
        "gamma": 1.0,
        "mass": 1.0,
        "charge": 1.0,
        "coordinate_dt": 0.0,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        compute_medina_radiation_reaction(**arguments)
