from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.radiation_flux_oracle import (
    ElectromagneticFluxSector,
    RadiationSphereFluxResult,
    evaluate_retarded_radiation_sphere_native,
    gauss_legendre_sphere_quadrature,
    integrate_radiation_sphere_flux_history_native,
    integrate_radiation_sphere_flux_native,
)


def _zero_fields(sample_count: int) -> np.ndarray:
    return np.zeros((sample_count, 3), dtype=float)


def _outgoing_transverse_fields(
    directions: np.ndarray, *, amplitude: float
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array((0.0, 0.0, 1.0))
    electric = np.cross(axis[np.newaxis, :], directions)
    electric /= np.linalg.norm(electric, axis=1)[:, np.newaxis]
    electric *= amplitude
    magnetic = np.cross(directions, electric)
    return electric, magnetic


def _linear_flux_sector(time_ns: float, factor: float) -> ElectromagneticFluxSector:
    return ElectromagneticFluxSector(
        energy_rate_native=factor * (2.0 + 3.0 * time_ns),
        momentum_rate_native=factor
        * np.array((1.0 + time_ns, -2.0 + 0.5 * time_ns, 0.25 - 0.75 * time_ns)),
        angular_momentum_rate_native=factor
        * np.array((-0.5 + 2.0 * time_ns, 0.3 - time_ns, 4.0 + 0.2 * time_ns)),
    )


def _sum_flux_sectors(
    *sectors: ElectromagneticFluxSector,
) -> ElectromagneticFluxSector:
    return ElectromagneticFluxSector(
        energy_rate_native=sum(sector.energy_rate_native for sector in sectors),
        momentum_rate_native=np.sum(
            [sector.momentum_rate_native for sector in sectors], axis=0
        ),
        angular_momentum_rate_native=np.sum(
            [sector.angular_momentum_rate_native for sector in sectors], axis=0
        ),
    )


def _linear_flux_sample(
    time_ns: float,
    *,
    radius_mm: float = 2.0,
    sphere_center_mm: tuple[float, float, float] = (0.1, -0.2, 0.3),
) -> RadiationSphereFluxResult:
    q_squared = _linear_flux_sector(time_ns, 1.0)
    q_mu = _linear_flux_sector(time_ns, -0.4)
    mu_squared = _linear_flux_sector(time_ns, 0.2)
    return RadiationSphereFluxResult(
        q_squared=q_squared,
        q_mu_interference=q_mu,
        mu_squared=mu_squared,
        total=_sum_flux_sectors(q_squared, q_mu, mu_squared),
        observation_time_ns=time_ns,
        sphere_center_mm=np.asarray(sphere_center_mm),
        angular_momentum_origin_mm=np.array((-0.4, 0.5, 0.6)),
        radius_mm=radius_mm,
        quadrature_sample_count=32,
        maximum_charge_light_cone_residual_mm=1.0e-15 * (1.0 + time_ns),
        maximum_dipole_light_cone_residual_mm=2.0e-15 * (1.0 + time_ns),
        charge_retarded_time_range_ns=(time_ns - 0.2, time_ns - 0.1),
        dipole_retarded_time_range_ns=(time_ns - 0.3, time_ns - 0.05),
    )


def _static_charge_dipole_history(
    *,
    charge_native: float = ELEMENTARY_CHARGE,
    moment_native: float = 0.7 * ELEMENTARY_CHARGE,
    spin_function: Callable[[float], np.ndarray] | None = None,
    times_ns: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    if spin_function is None:

        def spin_function(_time_ns: float) -> np.ndarray:
            return np.array((0.0, 0.0, 1.0))

    if times_ns is None:
        times_ns = np.linspace(-0.03, 0.003, 35)
    result = []
    for time_ns in times_ns:
        spin = spin_function(float(time_ns))
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([0.0]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([0.0]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([charge_native]),
                "q_source": np.array([charge_native]),
                "spin_x": np.array([spin[0]]),
                "spin_y": np.array([spin[1]]),
                "spin_z": np.array([spin[2]]),
                "magnetic_moment_native": np.array([moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _accelerated_charge_history(
    *,
    charge_native: float,
    acceleration_mm_ns2: float,
    times_ns: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    result = []
    for time_ns in times_ns:
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([0.5 * acceleration_mm_ns2 * time_ns**2]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([acceleration_mm_ns2 * time_ns / C_MMNS]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([acceleration_mm_ns2 / C_MMNS**2]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([charge_native]),
                "q_source": np.array([charge_native]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def test_gauss_legendre_sphere_integrates_basic_angular_moments() -> None:
    quadrature = gauss_legendre_sphere_quadrature(polar_order=8, azimuthal_order=16)

    assert quadrature.sample_count == 128
    assert np.sum(quadrature.solid_angle_weights) == pytest.approx(4.0 * np.pi)
    np.testing.assert_allclose(
        quadrature.solid_angle_weights @ quadrature.directions,
        0.0,
        atol=2.0e-15,
    )
    second_moment = np.einsum(
        "n,ni,nj->ij",
        quadrature.solid_angle_weights,
        quadrature.directions,
        quadrature.directions,
    )
    np.testing.assert_allclose(
        second_moment,
        np.eye(3) * 4.0 * np.pi / 3.0,
        rtol=3.0e-15,
        atol=2.0e-15,
    )
    assert not quadrature.directions.flags.writeable
    assert not quadrature.solid_angle_weights.flags.writeable


def test_uniform_outgoing_spherical_wave_has_expected_energy_flux() -> None:
    radius_mm = 2.3
    amplitude = 1.7
    quadrature = gauss_legendre_sphere_quadrature(polar_order=16, azimuthal_order=32)
    electric, magnetic = _outgoing_transverse_fields(
        quadrature.directions, amplitude=amplitude
    )
    zeros = _zero_fields(quadrature.sample_count)

    result = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=radius_mm,
        charge_electric_field_native=electric,
        charge_magnetic_field_native=magnetic,
        dipole_electric_field_native=zeros,
        dipole_magnetic_field_native=zeros,
    )

    assert result.q_squared.energy_rate_native == pytest.approx(
        C_MMNS * amplitude**2 * radius_mm**2,
        rel=3.0e-15,
    )
    np.testing.assert_allclose(result.q_squared.momentum_rate_native, 0.0, atol=2e-14)
    np.testing.assert_allclose(
        result.q_squared.angular_momentum_rate_native, 0.0, atol=2e-14
    )
    assert result.q_mu_interference.energy_rate_native == 0.0
    assert result.mu_squared.energy_rate_native == 0.0


def test_flux_history_integrates_irregular_linear_samples_exactly() -> None:
    times_ns = np.array((0.0, 0.07, 0.31, 0.8, 1.4))
    samples = [_linear_flux_sample(float(time_ns)) for time_ns in times_ns]

    result = integrate_radiation_sphere_flux_history_native(samples)

    duration = times_ns[-1] - times_ns[0]
    for factor, sector in (
        (1.0, result.q_squared),
        (-0.4, result.q_mu_interference),
        (0.2, result.mu_squared),
    ):
        first = _linear_flux_sector(float(times_ns[0]), factor)
        last = _linear_flux_sector(float(times_ns[-1]), factor)
        assert sector.energy_native == pytest.approx(
            0.5 * duration * (first.energy_rate_native + last.energy_rate_native)
        )
        np.testing.assert_allclose(
            sector.momentum_native,
            0.5 * duration * (first.momentum_rate_native + last.momentum_rate_native),
            rtol=2.0e-15,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            sector.angular_momentum_native,
            0.5
            * duration
            * (first.angular_momentum_rate_native + last.angular_momentum_rate_native),
            rtol=2.0e-15,
            atol=2.0e-15,
        )

    assert result.total.energy_native == pytest.approx(
        result.q_squared.energy_native
        + result.q_mu_interference.energy_native
        + result.mu_squared.energy_native
    )
    np.testing.assert_allclose(
        result.total.momentum_native,
        result.q_squared.momentum_native
        + result.q_mu_interference.momentum_native
        + result.mu_squared.momentum_native,
    )
    assert result.observation_time_interval_ns == (0.0, 1.4)
    assert result.sample_count == times_ns.size
    assert result.charge_retarded_time_envelope_ns == pytest.approx((-0.2, 1.3))
    assert result.dipole_retarded_time_envelope_ns == pytest.approx((-0.3, 1.35))
    assert result.maximum_charge_light_cone_residual_mm == pytest.approx(2.4e-15)
    assert result.maximum_dipole_light_cone_residual_mm == pytest.approx(4.8e-15)
    assert not result.total.momentum_native.flags.writeable


def test_flux_history_radius_comparison_uses_matched_retarded_time_window() -> None:
    quadrature = gauss_legendre_sphere_quadrature(polar_order=4, azimuthal_order=8)
    source_times_ns = np.linspace(-0.2, 0.3, 31)
    zeros = _zero_fields(quadrature.sample_count)
    integrals = []

    for radius_mm in (2.0, 11.0):
        samples = []
        for source_time_ns in source_times_ns:
            amplitude = (1.0 + 0.4 * source_time_ns) / radius_mm
            electric, magnetic = _outgoing_transverse_fields(
                quadrature.directions, amplitude=amplitude
            )
            samples.append(
                integrate_radiation_sphere_flux_native(
                    quadrature=quadrature,
                    radius_mm=radius_mm,
                    charge_electric_field_native=electric,
                    charge_magnetic_field_native=magnetic,
                    dipole_electric_field_native=zeros,
                    dipole_magnetic_field_native=zeros,
                    observation_time_ns=source_time_ns + radius_mm / C_MMNS,
                )
            )
        integrals.append(integrate_radiation_sphere_flux_history_native(samples))

    assert integrals[1].q_squared.energy_native == pytest.approx(
        integrals[0].q_squared.energy_native,
        rel=4.0e-15,
    )
    np.testing.assert_allclose(
        integrals[1].q_squared.momentum_native,
        integrals[0].q_squared.momentum_native,
        rtol=0.0,
        atol=2.0e-15,
    )


def test_flux_history_rejects_inconsistent_geometry_and_time_order() -> None:
    first = _linear_flux_sample(0.0)
    second = _linear_flux_sample(0.2)

    with pytest.raises(ValueError, match="at least two"):
        integrate_radiation_sphere_flux_history_native((first,))
    with pytest.raises(ValueError, match="increasing"):
        integrate_radiation_sphere_flux_history_native((second, first))
    with pytest.raises(ValueError, match="same sphere radius"):
        integrate_radiation_sphere_flux_history_native(
            (first, _linear_flux_sample(0.2, radius_mm=3.0))
        )
    with pytest.raises(ValueError, match="same sphere center"):
        integrate_radiation_sphere_flux_history_native(
            (
                first,
                _linear_flux_sample(
                    0.2,
                    sphere_center_mm=(0.1, -0.2, 0.30000000000000004),
                ),
            )
        )


def test_oscillating_magnetic_dipole_radiation_matches_gaussian_power() -> None:
    radius_mm = 4.7
    moment_second_derivative = np.array((0.0, 0.0, 2.4))
    quadrature = gauss_legendre_sphere_quadrature(polar_order=32, azimuthal_order=64)
    directions = quadrature.directions
    magnetic = np.cross(
        directions,
        np.cross(
            directions,
            np.broadcast_to(moment_second_derivative, directions.shape),
        ),
    ) / (C_MMNS**2 * radius_mm)
    electric = -np.cross(directions, magnetic)
    zeros = _zero_fields(quadrature.sample_count)

    result = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=radius_mm,
        charge_electric_field_native=zeros,
        charge_magnetic_field_native=zeros,
        dipole_electric_field_native=electric,
        dipole_magnetic_field_native=magnetic,
    )

    expected_power = (
        2.0
        * float(moment_second_derivative @ moment_second_derivative)
        / (3.0 * C_MMNS**3)
    )
    assert result.mu_squared.energy_rate_native == pytest.approx(
        expected_power, rel=2.0e-14
    )
    np.testing.assert_allclose(result.mu_squared.momentum_rate_native, 0.0, atol=1e-20)


def test_quadratic_sectors_sum_to_the_direct_total_field_flux() -> None:
    generator = np.random.default_rng(20260831)
    quadrature = gauss_legendre_sphere_quadrature(polar_order=8, azimuthal_order=12)
    fields = [generator.normal(size=(quadrature.sample_count, 3)) for _ in range(4)]
    split = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=1.3,
        charge_electric_field_native=fields[0],
        charge_magnetic_field_native=fields[1],
        dipole_electric_field_native=fields[2],
        dipole_magnetic_field_native=fields[3],
        sphere_center_mm=(0.2, -0.4, 0.1),
        angular_momentum_origin_mm=(-0.3, 0.5, 0.7),
    )
    zeros = _zero_fields(quadrature.sample_count)
    direct = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=1.3,
        charge_electric_field_native=fields[0] + fields[2],
        charge_magnetic_field_native=fields[1] + fields[3],
        dipole_electric_field_native=zeros,
        dipole_magnetic_field_native=zeros,
        sphere_center_mm=(0.2, -0.4, 0.1),
        angular_momentum_origin_mm=(-0.3, 0.5, 0.7),
    )

    assert split.total.energy_rate_native == pytest.approx(
        direct.q_squared.energy_rate_native, rel=4.0e-15, abs=2.0e-13
    )
    np.testing.assert_allclose(
        split.total.momentum_rate_native,
        direct.q_squared.momentum_rate_native,
        rtol=4.0e-15,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        split.total.angular_momentum_rate_native,
        direct.q_squared.angular_momentum_rate_native,
        rtol=4.0e-15,
        atol=2.0e-13,
    )


def test_angular_momentum_flux_obeys_change_of_origin_identity() -> None:
    quadrature = gauss_legendre_sphere_quadrature(polar_order=8, azimuthal_order=16)
    electric, magnetic = _outgoing_transverse_fields(
        quadrature.directions, amplitude=0.8
    )
    electric *= 1.0 + 0.3 * quadrature.directions[:, :1]
    magnetic = np.cross(quadrature.directions, electric)
    zeros = _zero_fields(quadrature.sample_count)
    first_origin = np.array((0.0, 0.0, 0.0))
    second_origin = np.array((0.4, -0.3, 0.2))
    common = {
        "quadrature": quadrature,
        "radius_mm": 1.9,
        "charge_electric_field_native": electric,
        "charge_magnetic_field_native": magnetic,
        "dipole_electric_field_native": zeros,
        "dipole_magnetic_field_native": zeros,
    }
    first = integrate_radiation_sphere_flux_native(
        angular_momentum_origin_mm=first_origin, **common
    )
    second = integrate_radiation_sphere_flux_native(
        angular_momentum_origin_mm=second_origin, **common
    )

    expected = first.total.angular_momentum_rate_native - np.cross(
        second_origin - first_origin,
        first.total.momentum_rate_native,
    )
    np.testing.assert_allclose(
        second.total.angular_momentum_rate_native,
        expected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_static_charge_and_dipole_have_no_outward_energy_flux() -> None:
    history = _static_charge_dipole_history()
    quadrature = gauss_legendre_sphere_quadrature(polar_order=2, azimuthal_order=4)

    result = evaluate_retarded_radiation_sphere_native(
        quadrature=quadrature,
        observation_time_ns=0.0,
        sphere_center_mm=(0.0, 0.0, 0.0),
        radius_mm=1.0,
        charge_history=history,
        dipole_history=history,
        source_identities=("source",),
        dipole_stencil_step_mm=5.0e-4,
    )

    assert result.q_squared.energy_rate_native == 0.0
    assert abs(result.q_mu_interference.energy_rate_native) < 1.0e-21
    assert abs(result.mu_squared.energy_rate_native) < 1.0e-21
    assert result.maximum_charge_light_cone_residual_mm is not None
    assert result.maximum_charge_light_cone_residual_mm < 1.0e-14
    assert result.maximum_dipole_light_cone_residual_mm is not None
    assert result.maximum_dipole_light_cone_residual_mm < 1.0e-14


def test_retarded_rotating_dipole_flux_matches_far_zone_power() -> None:
    angular_frequency_per_ns = 5.0
    moment_native = 2.0
    emission_time_ns = 2.5e-4

    def rotating_spin(time_ns: float) -> np.ndarray:
        angle = angular_frequency_per_ns * time_ns
        return np.array((np.cos(angle), np.sin(angle), 0.0))

    history = _static_charge_dipole_history(
        charge_native=0.0,
        moment_native=moment_native,
        spin_function=rotating_spin,
        times_ns=np.linspace(-0.15, 0.15, 601),
    )
    quadrature = gauss_legendre_sphere_quadrature(polar_order=4, azimuthal_order=8)
    expected_power = (
        2.0 * (moment_native * angular_frequency_per_ns**2) ** 2 / (3.0 * C_MMNS**3)
    )
    measured = []
    for radius_mm in (100.0, 400.0):
        result = evaluate_retarded_radiation_sphere_native(
            quadrature=quadrature,
            observation_time_ns=emission_time_ns + radius_mm / C_MMNS,
            sphere_center_mm=(0.0, 0.0, 0.0),
            radius_mm=radius_mm,
            dipole_history=history,
            source_identities=("rotating-moment",),
            dipole_stencil_step_mm=0.04,
        )
        measured.append(result.mu_squared.energy_rate_native)
        assert result.mu_squared.energy_rate_native == pytest.approx(
            expected_power, rel=1.0e-6
        )
        assert result.maximum_dipole_light_cone_residual_mm is not None
        assert result.maximum_dipole_light_cone_residual_mm < 1.0e-13
        assert result.dipole_retarded_time_range_ns is not None
        np.testing.assert_allclose(
            result.dipole_retarded_time_range_ns,
            emission_time_ns,
            rtol=0.0,
            atol=6.0e-16,
        )

    assert abs(measured[1] - measured[0]) / expected_power < 2.0e-7


def test_retarded_accelerated_charge_flux_matches_larmor_power() -> None:
    charge_native = 1.3
    acceleration_mm_ns2 = 0.8
    history = _accelerated_charge_history(
        charge_native=charge_native,
        acceleration_mm_ns2=acceleration_mm_ns2,
        times_ns=np.linspace(-0.15, 0.15, 601),
    )
    quadrature = gauss_legendre_sphere_quadrature(polar_order=4, azimuthal_order=8)
    expected_power = 2.0 * charge_native**2 * acceleration_mm_ns2**2 / (3.0 * C_MMNS**3)
    momentum_x = []
    for radius_mm in (10.0, 100.0):
        result = evaluate_retarded_radiation_sphere_native(
            quadrature=quadrature,
            observation_time_ns=radius_mm / C_MMNS,
            sphere_center_mm=(0.0, 0.0, 0.0),
            radius_mm=radius_mm,
            charge_history=history,
        )
        assert result.q_squared.energy_rate_native == pytest.approx(
            expected_power, rel=1.0e-11
        )
        assert result.charge_retarded_time_range_ns is not None
        np.testing.assert_allclose(
            result.charge_retarded_time_range_ns,
            0.0,
            rtol=0.0,
            atol=6.0e-16,
        )
        momentum_x.append(result.q_squared.momentum_rate_native[0])

    # The finite-radius bound-field momentum flux falls as 1/R.  The emitted
    # radiation pattern at this instantaneous rest event has zero net momentum.
    assert momentum_x[0] / momentum_x[1] == pytest.approx(10.0, rel=1.0e-10)


def test_charge_dipole_interference_momentum_reaches_far_zone_limit() -> None:
    charge_native = 1.3
    acceleration_mm_ns2 = 0.8
    moment_native = 2.0
    angular_frequency_per_ns = 5.0
    emission_time_ns = 2.5e-4
    times_ns = np.linspace(-0.15, 0.15, 601)
    charge_history = _accelerated_charge_history(
        charge_native=charge_native,
        acceleration_mm_ns2=acceleration_mm_ns2,
        times_ns=times_ns,
    )

    def rotating_spin(time_ns: float) -> np.ndarray:
        angle = angular_frequency_per_ns * time_ns
        return np.array((0.0, np.cos(angle), np.sin(angle)))

    dipole_history = _static_charge_dipole_history(
        charge_native=0.0,
        moment_native=moment_native,
        spin_function=rotating_spin,
        times_ns=times_ns,
    )
    quadrature = gauss_legendre_sphere_quadrature(polar_order=4, azimuthal_order=8)
    electric_dipole_second_derivative = np.array(
        (charge_native * acceleration_mm_ns2, 0.0, 0.0)
    )
    magnetic_dipole_second_derivative = (
        -moment_native * (angular_frequency_per_ns**2) * rotating_spin(emission_time_ns)
    )
    expected_momentum_rate = (
        2.0
        * np.cross(
            electric_dipole_second_derivative,
            magnetic_dipole_second_derivative,
        )
        / (3.0 * C_MMNS**4)
    )

    relative_errors = []
    for radius_mm in (800.0, 1600.0):
        result = evaluate_retarded_radiation_sphere_native(
            quadrature=quadrature,
            observation_time_ns=emission_time_ns + radius_mm / C_MMNS,
            sphere_center_mm=(0.0, 0.0, 0.0),
            radius_mm=radius_mm,
            charge_history=charge_history,
            dipole_history=dipole_history,
            source_identities=("rotating-moment",),
            dipole_stencil_step_mm=0.04,
        )
        assert abs(result.q_mu_interference.energy_rate_native) < 1.0e-15
        np.testing.assert_allclose(
            result.q_mu_interference.momentum_rate_native[1:],
            expected_momentum_rate[1:],
            rtol=9.0e-3,
            atol=0.0,
        )
        relative_errors.append(
            abs(
                result.q_mu_interference.momentum_rate_native[2]
                / expected_momentum_rate[2]
                - 1.0
            )
        )

    # The leading finite-radius correction decreases by four on radius doubling
    # for this symmetric prescribed-source benchmark.
    assert relative_errors[0] / relative_errors[1] == pytest.approx(4.0, rel=2.0e-3)
