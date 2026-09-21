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
    integrate_radiation_sphere_flux_native,
)
from core.retarded_dipole_fields import evaluate_retarded_dipole_field_gradient_native
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native
from core.spin_self_force_oracle import (
    evaluate_jakobsen_intrinsic_spin_radiation_balance_native,
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


def _circular_intrinsic_state(
    time_ns: float,
    *,
    orbit_radius_mm: float,
    angular_frequency_per_ns: float,
    gamma: float,
) -> tuple[np.ndarray, ...]:
    phase = angular_frequency_per_ns * time_ns
    radial = np.array((math.cos(phase), math.sin(phase), 0.0))
    tangent = np.array((-math.sin(phase), math.cos(phase), 0.0))
    velocity = orbit_radius_mm * angular_frequency_per_ns * tangent
    four_velocity = np.r_[gamma * C_MMNS, gamma * velocity]
    four_acceleration = np.r_[
        0.0,
        -(gamma**2) * orbit_radius_mm * angular_frequency_per_ns**2 * radial,
    ]
    four_jerk = np.r_[
        0.0,
        -(gamma**3) * orbit_radius_mm * angular_frequency_per_ns**3 * tangent,
    ]
    four_snap = np.r_[
        0.0,
        gamma**4 * orbit_radius_mm * angular_frequency_per_ns**4 * radial,
    ]
    return radial, tangent, four_velocity, four_acceleration, four_jerk, four_snap


def _circular_intrinsic_source_history(
    *,
    times_ns: np.ndarray,
    charge_native: float,
    magnetic_moment_native: float,
    orbit_radius_mm: float,
    angular_frequency_per_ns: float,
) -> list[dict[str, np.ndarray]]:
    beta_magnitude = orbit_radius_mm * angular_frequency_per_ns / C_MMNS
    gamma = 1.0 / math.sqrt(1.0 - beta_magnitude**2)
    history = []
    for time_ns in times_ns:
        radial, tangent, _, _, _, _ = _circular_intrinsic_state(
            float(time_ns),
            orbit_radius_mm=orbit_radius_mm,
            angular_frequency_per_ns=angular_frequency_per_ns,
            gamma=gamma,
        )
        history.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([orbit_radius_mm * radial[0]]),
                "y": np.array([orbit_radius_mm * radial[1]]),
                "z": np.array([0.0]),
                "bx": np.array([beta_magnitude * tangent[0]]),
                "by": np.array([beta_magnitude * tangent[1]]),
                "bz": np.array([0.0]),
                "bdotx": np.array(
                    [
                        -orbit_radius_mm
                        * angular_frequency_per_ns**2
                        * radial[0]
                        / C_MMNS**2
                    ]
                ),
                "bdoty": np.array(
                    [
                        -orbit_radius_mm
                        * angular_frequency_per_ns**2
                        * radial[1]
                        / C_MMNS**2
                    ]
                ),
                "bdotz": np.array([0.0]),
                "q": np.array([charge_native]),
                "q_source": np.array([charge_native]),
                "spin_x": np.array([0.0]),
                "spin_y": np.array([0.0]),
                "spin_z": np.array([1.0]),
                "magnetic_moment_native": np.array([magnetic_moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return history


def _uniformly_moving_rotating_dipole_history(
    *,
    times_ns: np.ndarray,
    beta_x: float,
    magnetic_moment_native: float,
    proper_angular_frequency_per_ns: float,
    lab_cycle_duration_ns: float,
) -> list[dict[str, np.ndarray]]:
    """Return a neutral dipole rotating in its uniformly moving rest frame."""

    gamma = 1.0 / math.sqrt(1.0 - beta_x**2)
    history = []
    for time_ns in times_ns:
        proper_phase = proper_angular_frequency_per_ns * time_ns / gamma
        history.append(
            {
                "t": np.array([time_ns]),
                "x": np.array(
                    [beta_x * C_MMNS * (time_ns - 0.5 * lab_cycle_duration_ns)]
                ),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([beta_x]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([0.0]),
                "q_source": np.array([0.0]),
                "spin_x": np.array([0.0]),
                "spin_y": np.array([math.cos(proper_phase)]),
                "spin_z": np.array([math.sin(proper_phase)]),
                "magnetic_moment_native": np.array([magnetic_moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return history


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


@pytest.mark.slow
def test_circular_intrinsic_spin_energy_requires_radiative_field_correction() -> None:
    """Close the nonzero supplemental Eq. (33) term against retarded flux.

    A charge moves uniformly on a circle while its intrinsic spin and moment
    remain aligned with the normal to the orbit.  This is the motion produced
    by a uniform magnetic field with aligned spin, so the translational and
    spin histories satisfy the leading external equations of motion.  Unlike
    the linear-oscillation test above, ``S.[A x J]`` is nonzero.

    The projected self-force alone therefore does not balance the radiated
    q-mu energy.  Adding Jakobsen's radiative-field *balance term* does.  The
    latter is not an additional mechanical force.
    """

    charge = 0.8
    mass = 1.0
    g_factor = 2.3
    orbit_radius_mm = 0.03
    angular_frequency_per_ns = 1.7
    moment_native = 1.4e-8
    period_ns = 2.0 * math.pi / angular_frequency_per_ns
    beta_magnitude = orbit_radius_mm * angular_frequency_per_ns / C_MMNS
    gamma = 1.0 / math.sqrt(1.0 - beta_magnitude**2)
    spin_magnitude = 2.0 * mass * C_MMNS * moment_native / (g_factor * charge)

    history = _circular_intrinsic_source_history(
        times_ns=np.linspace(-0.2 * period_ns, 1.2 * period_ns, 1601),
        charge_native=charge,
        magnetic_moment_native=moment_native,
        orbit_radius_mm=orbit_radius_mm,
        angular_frequency_per_ns=angular_frequency_per_ns,
    )

    source_times = np.linspace(0.0, period_ns, 65)
    self_force_energy_rates = np.empty(source_times.size)
    balance_energy_rates = np.empty(source_times.size)
    outward_radiated_energy_rates = np.empty(source_times.size)
    bound_momenta = np.empty((source_times.size, 4))
    for index, time_ns in enumerate(source_times):
        (
            _,
            _,
            four_velocity,
            four_acceleration,
            four_jerk,
            four_snap,
        ) = _circular_intrinsic_state(
            float(time_ns),
            orbit_radius_mm=orbit_radius_mm,
            angular_frequency_per_ns=angular_frequency_per_ns,
            gamma=gamma,
        )
        local = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
            charge_native=charge,
            mass_amu=mass,
            g_factor=g_factor,
            four_velocity_mm_ns=four_velocity,
            four_acceleration_mm_ns2=four_acceleration,
            four_jerk_mm_ns3=four_jerk,
            four_snap_mm_ns4=four_snap,
            spin_four_vector_native=(0.0, 0.0, 0.0, spin_magnitude),
            spin_four_derivative_native=np.zeros(4),
            spin_four_second_derivative_native=np.zeros(4),
        )
        self_force_energy_rates[index] = (
            C_MMNS * local.self_force.linear_spin_self_force_native[0] / gamma
        )
        balance_energy_rates[index] = (
            C_MMNS
            * local.self_force.linear_spin_radiative_balance_rate_native[0]
            / gamma
        )
        outward_radiated_energy_rates[index] = (
            C_MMNS * local.outward_radiated_momentum_rate_native[0] / gamma
        )
        bound_momenta[index] = local.bound_field_momentum_native
        scale = max(
            np.linalg.norm(local.self_force.linear_spin_radiative_balance_rate_native),
            np.finfo(float).tiny,
        )
        assert np.linalg.norm(local.balance_residual_native) < 2.0e-13 * scale

    intervals = np.diff(source_times)
    self_force_work = float(
        np.sum(
            0.5
            * (self_force_energy_rates[:-1] + self_force_energy_rates[1:])
            * intervals
        )
    )
    local_balance_energy = float(
        np.sum(0.5 * (balance_energy_rates[:-1] + balance_energy_rates[1:]) * intervals)
    )
    direct_outward_energy = float(
        np.sum(
            0.5
            * (outward_radiated_energy_rates[:-1] + outward_radiated_energy_rates[1:])
            * intervals
        )
    )
    assert self_force_work < 0.0
    assert local_balance_energy < self_force_work
    assert direct_outward_energy == pytest.approx(
        -local_balance_energy,
        rel=3.0e-15,
    )
    np.testing.assert_allclose(
        bound_momenta[-1],
        bound_momenta[0],
        rtol=0.0,
        atol=2.0e-14 * np.linalg.norm(bound_momenta[0]),
    )

    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=3,
        azimuthal_order=6,
    )
    outward_energies = []
    for radius_mm in (400.0, 800.0):
        samples = [
            evaluate_retarded_radiation_sphere_native(
                quadrature=quadrature,
                observation_time_ns=source_time_ns + radius_mm / C_MMNS,
                sphere_center_mm=(0.0, 0.0, 0.0),
                radius_mm=radius_mm,
                charge_history=history,
                dipole_history=history,
                source_identities=("circular-intrinsic-source",),
                dipole_stencil_step_mm=0.04,
                backend="python",
            )
            for source_time_ns in np.linspace(0.0, period_ns, 17)
        ]
        outward = integrate_radiation_sphere_flux_history_native(
            samples
        ).q_mu_interference.energy_native
        outward_energies.append(outward)
        assert outward > 0.0
        assert outward == pytest.approx(direct_outward_energy, rel=2.0e-6)
        assert local_balance_energy + outward == pytest.approx(
            0.0,
            abs=2.0e-6 * outward,
        )
        # The projected mechanical self-force alone misses most of the
        # interference energy in this geometry.
        assert abs(self_force_work + outward) > 0.8 * outward

    assert outward_energies[1] == pytest.approx(outward_energies[0], rel=3.0e-6)


@pytest.mark.slow
def test_nonperiodic_bound_momentum_closes_on_matched_light_cones() -> None:
    """Resolve a nonzero bound-momentum change at null infinity.

    The interval is one quarter of the circular orbit.  For every angular
    ray and source time, the observer time is chosen so the exact retarded
    event is that source time.  Multiplication by
    ``dt_observer/dt_source = 1-n.beta`` then integrates one common emission
    interval over the sphere instead of one common observer-time interval.

    Finite-radius momentum is dominated by reversible near-field transport.
    A quadratic extrapolation in ``1/R`` isolates the constant radiative
    term and is compared with the local self-force plus the explicitly
    nonzero bound-field endpoint change.
    """

    charge = 0.8
    mass = 1.0
    g_factor = 2.3
    orbit_radius_mm = 0.03
    angular_frequency_per_ns = 50.0
    moment_native = 1.4e-8
    period_ns = 2.0 * math.pi / angular_frequency_per_ns
    beta_magnitude = orbit_radius_mm * angular_frequency_per_ns / C_MMNS
    gamma = 1.0 / math.sqrt(1.0 - beta_magnitude**2)
    spin_magnitude = 2.0 * mass * C_MMNS * moment_native / (g_factor * charge)
    history = _circular_intrinsic_source_history(
        times_ns=np.linspace(-0.25 * period_ns, 1.25 * period_ns, 1537),
        charge_native=charge,
        magnetic_moment_native=moment_native,
        orbit_radius_mm=orbit_radius_mm,
        angular_frequency_per_ns=angular_frequency_per_ns,
    )

    # Integrate the local side finely because it nearly cancels the much
    # larger endpoint change in bound momentum.
    local_times = np.linspace(0.0, 0.25 * period_ns, 1025)
    local_rates = np.empty((local_times.size, 4))
    bound_momenta = np.empty((local_times.size, 4))
    for index, source_time_ns in enumerate(local_times):
        (
            _,
            _,
            four_velocity,
            four_acceleration,
            four_jerk,
            four_snap,
        ) = _circular_intrinsic_state(
            float(source_time_ns),
            orbit_radius_mm=orbit_radius_mm,
            angular_frequency_per_ns=angular_frequency_per_ns,
            gamma=gamma,
        )
        local = evaluate_jakobsen_intrinsic_spin_radiation_balance_native(
            charge_native=charge,
            mass_amu=mass,
            g_factor=g_factor,
            four_velocity_mm_ns=four_velocity,
            four_acceleration_mm_ns2=four_acceleration,
            four_jerk_mm_ns3=four_jerk,
            four_snap_mm_ns4=four_snap,
            spin_four_vector_native=(0.0, 0.0, 0.0, spin_magnitude),
            spin_four_derivative_native=np.zeros(4),
            spin_four_second_derivative_native=np.zeros(4),
        )
        local_rates[index] = (
            local.self_force.linear_spin_radiative_balance_rate_native / gamma
        )
        bound_momenta[index] = local.bound_field_momentum_native

    local_intervals = np.diff(local_times)
    local_impulse = np.sum(
        0.5 * (local_rates[:-1] + local_rates[1:]) * local_intervals[:, np.newaxis],
        axis=0,
    )
    bound_change = bound_momenta[-1] - bound_momenta[0]
    expected_outward_four_momentum = bound_change - local_impulse
    assert np.linalg.norm(bound_change[1:]) > (
        1.0e3 * np.linalg.norm(expected_outward_four_momentum[1:])
    )

    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=3,
        azimuthal_order=6,
    )
    source_times = np.linspace(0.0, 0.25 * period_ns, 17)
    radii = np.array((400.0, 800.0, 1600.0))
    radial_outward_four_momenta = []
    for radius_mm in radii:
        source_parameterized_samples = []
        for source_time_ns in source_times:
            radial, tangent, _, _, _, _ = _circular_intrinsic_state(
                float(source_time_ns),
                orbit_radius_mm=orbit_radius_mm,
                angular_frequency_per_ns=angular_frequency_per_ns,
                gamma=gamma,
            )
            source_position = orbit_radius_mm * radial
            source_beta = beta_magnitude * tangent
            charge_electric = np.empty((quadrature.sample_count, 3))
            charge_magnetic = np.empty((quadrature.sample_count, 3))
            dipole_electric = np.empty((quadrature.sample_count, 3))
            dipole_magnetic = np.empty((quadrature.sample_count, 3))
            time_jacobian = np.empty(quadrature.sample_count)
            for sample_index, direction in enumerate(quadrature.directions):
                observer_position = radius_mm * direction
                separation = observer_position - source_position
                distance = float(np.linalg.norm(separation))
                source_to_observer = separation / distance
                observer_event = ObserverEvent(
                    time_ns=float(source_time_ns + distance / C_MMNS),
                    position_mm=tuple(float(value) for value in observer_position),
                )
                charge_result = evaluate_retarded_charge_field_native(
                    history,
                    observer_event,
                    require_complete_history=True,
                )
                dipole_result = evaluate_retarded_dipole_field_gradient_native(
                    history,
                    observer_event,
                    source_identities=("nonperiodic-circular-source",),
                    require_complete_history=True,
                    stencil_step_mm=0.04,
                    backend="python",
                )
                charge_electric[sample_index] = charge_result.electric_field_native
                charge_magnetic[sample_index] = charge_result.magnetic_field_native
                dipole_electric[sample_index] = dipole_result.electric_field_native
                dipole_magnetic[sample_index] = dipole_result.magnetic_field_native
                time_jacobian[sample_index] = 1.0 - float(
                    source_to_observer @ source_beta
                )
                assert charge_result.retarded_time_ns[0] == pytest.approx(
                    source_time_ns,
                    abs=1.0e-14,
                )
                assert dipole_result.hertz.retarded_time_ns[0] == pytest.approx(
                    source_time_ns,
                    abs=1.0e-14,
                )

            source_parameterized_samples.append(
                integrate_radiation_sphere_flux_native(
                    quadrature=quadrature,
                    radius_mm=radius_mm,
                    charge_electric_field_native=charge_electric,
                    charge_magnetic_field_native=charge_magnetic,
                    dipole_electric_field_native=dipole_electric,
                    dipole_magnetic_field_native=dipole_magnetic,
                    observation_time_ns=float(source_time_ns),
                    sample_time_jacobian=time_jacobian,
                )
            )

        outward = integrate_radiation_sphere_flux_history_native(
            source_parameterized_samples
        ).q_mu_interference
        radial_outward_four_momenta.append(
            np.r_[outward.energy_native / C_MMNS, outward.momentum_native]
        )

    radial_values = np.asarray(radial_outward_four_momenta)
    inverse_radius = 1.0 / radii
    null_infinity = np.array(
        [
            np.polyfit(inverse_radius, radial_values[:, component], 2)[-1]
            for component in range(4)
        ]
    )
    np.testing.assert_allclose(
        null_infinity[:3],
        expected_outward_four_momentum[:3],
        rtol=2.0e-2,
        atol=2.0e-6 * abs(expected_outward_four_momentum[0]),
    )
    assert abs(null_infinity[3]) < 1.0e-12 * abs(null_infinity[0])


@pytest.mark.slow
def test_uniformly_moving_rotating_dipole_flux_is_a_four_vector() -> None:
    """Recover the Lorentz transform of a pure-magnetic radiation cycle.

    Every angular ray is evaluated on the future light cone of the same
    source event. Multiplying by ``dt_observer/dt_source`` then integrates a
    common proper-time cycle even though those rays reach the fixed sphere at
    different observer times.

    This is a radiation-transport test, not a local recoil-force test. A
    periodic rest-frame source emits no net rest-frame linear momentum, so its
    cycle-integrated radiated four-momentum has a simple boost target.
    """

    beta_x = 0.55
    gamma = 1.0 / math.sqrt(1.0 - beta_x**2)
    moment_native = 1.4e-8
    proper_angular_frequency_per_ns = 20.0
    proper_period_ns = 2.0 * math.pi / proper_angular_frequency_per_ns
    lab_period_ns = gamma * proper_period_ns
    history = _uniformly_moving_rotating_dipole_history(
        times_ns=np.linspace(-0.2 * lab_period_ns, 1.2 * lab_period_ns, 2241),
        beta_x=beta_x,
        magnetic_moment_native=moment_native,
        proper_angular_frequency_per_ns=proper_angular_frequency_per_ns,
        lab_cycle_duration_ns=lab_period_ns,
    )
    quadrature = gauss_legendre_sphere_quadrature(
        polar_order=6,
        azimuthal_order=12,
    )
    zeros = np.zeros((quadrature.sample_count, 3))
    source_times_ns = np.linspace(0.0, lab_period_ns, 25)
    rest_power_native = (
        2.0 * moment_native**2 * proper_angular_frequency_per_ns**4 / (3.0 * C_MMNS**3)
    )
    rest_cycle_energy_native = rest_power_native * proper_period_ns
    expected_four_momentum = np.array(
        (
            gamma * rest_cycle_energy_native / C_MMNS,
            gamma * beta_x * rest_cycle_energy_native / C_MMNS,
            0.0,
            0.0,
        )
    )

    radial_four_momenta = []
    for radius_mm in (400.0, 800.0):
        samples = []
        for source_time_ns in source_times_ns:
            source_position = np.array(
                (
                    beta_x * C_MMNS * (source_time_ns - 0.5 * lab_period_ns),
                    0.0,
                    0.0,
                )
            )
            dipole_electric = np.empty((quadrature.sample_count, 3))
            dipole_magnetic = np.empty((quadrature.sample_count, 3))
            time_jacobian = np.empty(quadrature.sample_count)
            for sample_index, direction in enumerate(quadrature.directions):
                observer_position = radius_mm * direction
                separation = observer_position - source_position
                distance = float(np.linalg.norm(separation))
                source_to_observer = separation / distance
                result = evaluate_retarded_dipole_field_gradient_native(
                    history,
                    ObserverEvent(
                        time_ns=float(source_time_ns + distance / C_MMNS),
                        position_mm=tuple(float(value) for value in observer_position),
                    ),
                    source_identities=("uniformly-moving-rotating-dipole",),
                    require_complete_history=True,
                    stencil_step_mm=0.04,
                    backend="python",
                )
                dipole_electric[sample_index] = result.electric_field_native
                dipole_magnetic[sample_index] = result.magnetic_field_native
                time_jacobian[sample_index] = 1.0 - beta_x * source_to_observer[0]
                assert result.hertz.retarded_time_ns[0] == pytest.approx(
                    source_time_ns,
                    abs=2.0e-14,
                )

            samples.append(
                integrate_radiation_sphere_flux_native(
                    quadrature=quadrature,
                    radius_mm=radius_mm,
                    charge_electric_field_native=zeros,
                    charge_magnetic_field_native=zeros,
                    dipole_electric_field_native=dipole_electric,
                    dipole_magnetic_field_native=dipole_magnetic,
                    observation_time_ns=float(source_time_ns),
                    sample_time_jacobian=time_jacobian,
                )
            )

        outward = integrate_radiation_sphere_flux_history_native(samples).mu_squared
        radial_four_momenta.append(
            np.r_[outward.energy_native / C_MMNS, outward.momentum_native]
        )

    radial_values = np.asarray(radial_four_momenta)
    for measured in radial_values:
        np.testing.assert_allclose(
            measured,
            expected_four_momentum,
            rtol=3.0e-3,
            atol=3.0e-5 * expected_four_momentum[0],
        )
    np.testing.assert_allclose(
        radial_values[1],
        radial_values[0],
        rtol=2.0e-3,
        atol=2.0e-5 * expected_four_momentum[0],
    )
