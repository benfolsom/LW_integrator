"""Slow provider-level balance checks for the linear spin self-force oracle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS
from core.radiation_flux_oracle import (
    evaluate_retarded_radiation_sphere_native,
    gauss_legendre_sphere_quadrature,
    integrate_radiation_sphere_flux_history_native,
)
from core.spin_self_force_oracle import (
    evaluate_jakobsen_linear_spin_self_force_native,
)


def _periodic_intrinsic_source_history(
    *,
    charge_native: float,
    position_amplitude_mm: float,
    magnetic_moment_native: float,
    angular_frequency_per_ns: float,
    period_ns: float,
) -> list[dict[str, np.ndarray]]:
    history = []
    for time_ns in np.linspace(-0.1 * period_ns, 1.1 * period_ns, 1201):
        phase = angular_frequency_per_ns * time_ns
        cosine = math.cos(phase)
        sine = math.sin(phase)
        velocity_x = -position_amplitude_mm * angular_frequency_per_ns * sine
        acceleration_x = -position_amplitude_mm * angular_frequency_per_ns**2 * cosine
        history.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([position_amplitude_mm * cosine]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([velocity_x / C_MMNS]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([acceleration_x / C_MMNS**2]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([charge_native]),
                "q_source": np.array([charge_native]),
                "spin_x": np.array([0.0]),
                "spin_y": np.array([cosine]),
                "spin_z": np.array([sine]),
                "magnetic_moment_native": np.array([magnetic_moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return history


def _local_linear_spin_impulse(
    *,
    charge_native: float,
    mass_amu: float,
    g_factor: float,
    position_amplitude_mm: float,
    magnetic_moment_native: float,
    angular_frequency_per_ns: float,
    period_ns: float,
) -> np.ndarray:
    times_ns = np.linspace(0.0, period_ns, 65)
    forces = np.empty((times_ns.size, 3))
    for index, time_ns in enumerate(times_ns):
        phase = angular_frequency_per_ns * time_ns
        cosine = math.cos(phase)
        sine = math.sin(phase)
        acceleration_x = -position_amplitude_mm * angular_frequency_per_ns**2 * cosine
        jerk_x = position_amplitude_mm * angular_frequency_per_ns**3 * sine
        snap_x = position_amplitude_mm * angular_frequency_per_ns**4 * cosine
        moment = magnetic_moment_native * np.array((0.0, cosine, sine))
        moment_derivative = (
            magnetic_moment_native
            * angular_frequency_per_ns
            * np.array((0.0, -sine, cosine))
        )
        spin_scale = 2.0 * mass_amu * C_MMNS / (g_factor * charge_native)
        result = evaluate_jakobsen_linear_spin_self_force_native(
            charge_native=charge_native,
            mass_amu=mass_amu,
            four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
            four_acceleration_mm_ns2=(0.0, acceleration_x, 0.0, 0.0),
            four_jerk_mm_ns3=(0.0, jerk_x, 0.0, 0.0),
            four_snap_mm_ns4=(0.0, snap_x, 0.0, 0.0),
            spin_four_vector_native=np.r_[0.0, spin_scale * moment],
            spin_four_derivative_native=np.r_[0.0, spin_scale * moment_derivative],
            magnetic_moment_four_vector_native=np.r_[0.0, moment],
            magnetic_moment_four_derivative_native=np.r_[0.0, moment_derivative],
        )
        forces[index] = result.linear_spin_self_force_native[1:]
    intervals = np.diff(times_ns)
    return np.sum(
        0.5 * (forces[:-1] + forces[1:]) * intervals[:, np.newaxis],
        axis=0,
    )


@pytest.mark.slow
def test_fully_retarded_periodic_intrinsic_spin_flux_balances_local_impulse() -> None:
    """Compare the local oracle with the maintained retarded field providers.

    A slow one-dimensional charge oscillation is combined with a
    fixed-magnitude moment rotating in the transverse plane.  This geometry
    retains nonzero intrinsic spin and q-mu momentum while making the
    spin--radiative-electric-field correction in Jakobsen's supplemental
    Eq. (33) vanish because acceleration and that electric field are
    collinear.  The remaining total derivative is periodic.
    """

    charge = 0.8
    mass = 1.0
    g_factor = 2.3
    position_amplitude_mm = 0.03
    moment_native = 1.4e-8
    angular_frequency_per_ns = 1.7
    period_ns = 2.0 * math.pi / angular_frequency_per_ns
    history = _periodic_intrinsic_source_history(
        charge_native=charge,
        position_amplitude_mm=position_amplitude_mm,
        magnetic_moment_native=moment_native,
        angular_frequency_per_ns=angular_frequency_per_ns,
        period_ns=period_ns,
    )
    local_impulse = _local_linear_spin_impulse(
        charge_native=charge,
        mass_amu=mass,
        g_factor=g_factor,
        position_amplitude_mm=position_amplitude_mm,
        magnetic_moment_native=moment_native,
        angular_frequency_per_ns=angular_frequency_per_ns,
        period_ns=period_ns,
    )
    expected_outward_z = (
        charge
        * position_amplitude_mm
        * moment_native
        * angular_frequency_per_ns**4
        * period_ns
        / (3.0 * C_MMNS**4)
    )
    np.testing.assert_allclose(
        local_impulse,
        (0.0, 0.0, -expected_outward_z),
        rtol=0.0,
        atol=2.0e-14 * expected_outward_z,
    )

    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=3,
        azimuthal_order=6,
    )
    radial_results = []
    for radius_mm in (400.0, 800.0):
        samples = [
            evaluate_retarded_radiation_sphere_native(
                quadrature=quadrature,
                observation_time_ns=source_time_ns + radius_mm / C_MMNS,
                sphere_center_mm=(0.0, 0.0, 0.0),
                radius_mm=radius_mm,
                charge_history=history,
                dipole_history=history,
                source_identities=("periodic-intrinsic-source",),
                dipole_stencil_step_mm=0.04,
                backend="python",
            )
            for source_time_ns in np.linspace(0.0, period_ns, 17)
        ]
        integrated = integrate_radiation_sphere_flux_history_native(samples)
        radial_results.append(integrated.q_mu_interference)
        assert integrated.q_mu_interference.momentum_native[2] == pytest.approx(
            expected_outward_z,
            rel=1.0e-5,
        )
        assert abs(integrated.q_mu_interference.momentum_native[0]) < (
            3.0e-7 * expected_outward_z
        )
        assert abs(integrated.q_mu_interference.energy_native) < (
            3.0e-10 * C_MMNS * expected_outward_z
        )

    # The leading transverse finite-radius/bound-field contribution decreases
    # as 1/R and therefore vanishes at infinity.  The radiative z component is
    # already radius invariant at the maintained tolerance.
    assert (
        radial_results[0].momentum_native[1] / radial_results[1].momentum_native[1]
    ) == pytest.approx(2.0, rel=2.0e-5)
    assert radial_results[1].momentum_native[2] == pytest.approx(
        radial_results[0].momentum_native[2],
        rel=3.0e-6,
    )
    assert local_impulse[2] + radial_results[1].momentum_native[2] == (
        pytest.approx(0.0, abs=3.0e-6 * expected_outward_z)
    )
