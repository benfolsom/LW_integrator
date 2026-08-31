from __future__ import annotations

import math

import numpy as np
import pytest

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.spinning_shell_self_torque import (
    evaluate_spinning_shell_angular_balance_native,
    evaluate_spinning_shell_local_self_torque_native,
)


def _harmonic_derivatives(
    *, moment_native: float, angular_frequency_per_ns: float, time_ns: float
) -> np.ndarray:
    return np.asarray(
        [
            moment_native
            * angular_frequency_per_ns**order
            * math.cos(angular_frequency_per_ns * time_ns + order * math.pi / 2.0)
            for order in range(9)
        ]
    )


def test_static_shell_stores_constant_field_angular_momentum() -> None:
    derivatives = np.zeros(9)
    derivatives[0] = 2.3e-8

    result = evaluate_spinning_shell_angular_balance_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.04,
        observation_radius_mm=12.0,
        shell_retarded_moment_derivatives_native=derivatives,
        observation_retarded_moment_derivatives_native=derivatives,
    )

    assert result.near_field_angular_momentum_native > 0.0
    assert result.wave_zone_angular_momentum_native == 0.0
    assert result.field_angular_momentum_native == (
        result.near_field_angular_momentum_native
    )
    assert result.self_torque_native == 0.0
    assert result.outward_angular_momentum_rate_native == 0.0
    assert result.field_angular_momentum_rate_native == 0.0
    assert result.balance_residual_native == 0.0


def test_retarded_shell_angular_momentum_balance_closes_for_harmonic_moment() -> None:
    moment_native = 1.7e-8
    angular_frequency_per_ns = 0.8
    observation_time_ns = 0.37

    for shell_radius_mm in (0.08, 0.04, 0.02):
        observation_radius_mm = 25.0
        shell_time_ns = observation_time_ns - shell_radius_mm / C_MMNS
        observer_time_ns = observation_time_ns - observation_radius_mm / C_MMNS
        result = evaluate_spinning_shell_angular_balance_native(
            charge_native=-ELEMENTARY_CHARGE,
            shell_radius_mm=shell_radius_mm,
            observation_radius_mm=observation_radius_mm,
            shell_retarded_moment_derivatives_native=_harmonic_derivatives(
                moment_native=moment_native,
                angular_frequency_per_ns=angular_frequency_per_ns,
                time_ns=shell_time_ns,
            ),
            observation_retarded_moment_derivatives_native=_harmonic_derivatives(
                moment_native=moment_native,
                angular_frequency_per_ns=angular_frequency_per_ns,
                time_ns=observer_time_ns,
            ),
        )

        scale = max(
            abs(result.self_torque_native),
            abs(result.outward_angular_momentum_rate_native),
            abs(result.field_angular_momentum_rate_native),
        )
        assert abs(result.balance_residual_native) < 4.0e-15 * scale


def test_local_shell_torque_separates_reversible_and_radiative_terms() -> None:
    derivatives = _harmonic_derivatives(
        moment_native=2.1e-8,
        angular_frequency_per_ns=1.3,
        time_ns=0.23,
    )
    first = evaluate_spinning_shell_local_self_torque_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.08,
        current_moment_derivatives_native=derivatives,
    )
    half = evaluate_spinning_shell_local_self_torque_native(
        charge_native=ELEMENTARY_CHARGE,
        shell_radius_mm=0.04,
        current_moment_derivatives_native=derivatives,
    )

    assert first.total_self_torque_native == pytest.approx(
        first.time_symmetric_torque_native + first.radiation_reaction_torque_native,
        rel=0.0,
        abs=0.0,
    )
    assert first.time_symmetric_torque_native != 0.0
    assert first.radiation_reaction_torque_native != 0.0

    # The leading time-symmetric term is proportional to 1/R, while the
    # leading radiation-reaction term is proportional to R^2.  The small
    # higher-order corrections make these ratios only asymptotically exact.
    assert half.time_symmetric_torque_native / first.time_symmetric_torque_native == (
        pytest.approx(2.0, rel=2.0e-7)
    )
    reaction_ratio = (
        half.radiation_reaction_torque_native
        / first.radiation_reaction_torque_native
    )
    assert reaction_ratio == pytest.approx(0.25, rel=2.0e-7)


def test_shell_oracle_rejects_invalid_geometry_and_derivatives() -> None:
    derivatives = np.zeros(9)
    with pytest.raises(ValueError, match="exceed"):
        evaluate_spinning_shell_angular_balance_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            observation_radius_mm=1.0,
            shell_retarded_moment_derivatives_native=derivatives,
            observation_retarded_moment_derivatives_native=derivatives,
        )
    with pytest.raises(ValueError, match="0 through 8"):
        evaluate_spinning_shell_local_self_torque_native(
            charge_native=ELEMENTARY_CHARGE,
            shell_radius_mm=1.0,
            current_moment_derivatives_native=np.zeros(8),
        )
