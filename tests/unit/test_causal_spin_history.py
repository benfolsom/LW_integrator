from __future__ import annotations

import numpy as np
import pytest

from core.causal_spin_history import (
    append_causal_frozen_spin_slopes_in_place,
    append_causal_frozen_spin_slopes_per_ns,
    causal_frozen_spin_slopes_per_ns,
)


def test_linear_spin_history_has_exact_constant_slope() -> None:
    time = np.array([0.0, 0.1, 0.4, 0.9])
    intercept = np.array([1.0, -2.0, 0.5])
    slope = np.array([0.25, 0.75, -1.5])
    spin = intercept + time[:, np.newaxis] * slope

    result = causal_frozen_spin_slopes_per_ns(spin, time)
    np.testing.assert_allclose(result, np.broadcast_to(slope, result.shape))


def test_newest_quadratic_slope_is_exact_on_nonuniform_times() -> None:
    time = np.array([0.0, 0.2, 0.7, 1.5, 2.0])
    coefficients = np.array([0.5, -1.0, 2.0])
    spin = time[:, np.newaxis] ** 2 * coefficients

    result = causal_frozen_spin_slopes_per_ns(spin, time)
    expected = 2.0 * time[:, np.newaxis] * coefficients
    np.testing.assert_allclose(result[2:], expected[2:], rtol=2.0e-15, atol=2.0e-15)


def test_appending_knots_never_changes_queryable_slope_prefix() -> None:
    time = np.array([0.0, 0.1, 0.25, 0.6, 1.0])
    spin = np.stack(
        (
            np.cos(2.0 * time),
            np.sin(2.0 * time),
            0.1 * time**2,
        ),
        axis=1,
    )

    slopes = causal_frozen_spin_slopes_per_ns(spin[:2], time[:2])
    for stop in range(3, time.size + 1):
        prefix = slopes.copy()
        slopes = append_causal_frozen_spin_slopes_per_ns(
            slopes,
            spin[:stop],
            time[:stop],
        )
        np.testing.assert_array_equal(slopes[: prefix.shape[0]], prefix)
        np.testing.assert_array_equal(
            slopes,
            causal_frozen_spin_slopes_per_ns(spin[:stop], time[:stop]),
        )


def test_one_knot_placeholder_can_be_finalized_when_first_segment_arrives() -> None:
    time = np.array([0.0, 0.25])
    spin = np.array([[1.0, 0.0, 0.0], [0.75, 0.5, 0.0]])
    one = causal_frozen_spin_slopes_per_ns(spin[:1], time[:1])
    assert np.array_equal(one, np.zeros((1, 3)))

    two = append_causal_frozen_spin_slopes_per_ns(one, spin, time)
    expected = np.broadcast_to((spin[1] - spin[0]) / 0.25, (2, 3))
    np.testing.assert_array_equal(two, expected)


@pytest.mark.parametrize(
    ("spin", "time", "message"),
    (
        (np.zeros((2, 3)), np.array([0.0]), "one value per spin knot"),
        (np.zeros((2, 3)), np.array([0.0, 0.0]), "strictly increasing"),
        (np.array([[np.nan, 0.0, 0.0]]), np.array([0.0]), "finite"),
    ),
)
def test_invalid_spin_history_is_rejected(
    spin: np.ndarray,
    time: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        causal_frozen_spin_slopes_per_ns(spin, time)


def test_append_rejects_noncausal_or_corrupt_prefix() -> None:
    time = np.array([0.0, 0.1, 0.2])
    spin = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
        ]
    )
    previous = causal_frozen_spin_slopes_per_ns(spin[:2], time[:2])
    previous[1, 0] += 1.0
    with pytest.raises(ValueError, match="causal accepted prefix"):
        append_causal_frozen_spin_slopes_per_ns(previous, spin, time)


def test_managed_in_place_append_updates_only_the_constant_size_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.causal_spin_history as causal_spin_history

    time = np.linspace(0.0, 1.0, 100_001)
    spin = np.stack(
        (
            np.cos(0.7 * time),
            np.sin(0.7 * time),
            0.2 * time**2,
        ),
        axis=1,
    )
    old_count = time.size - 1
    expected = causal_frozen_spin_slopes_per_ns(spin, time)
    slopes = np.empty_like(spin)
    slopes[:old_count] = expected[:old_count]
    prefix = slopes[:old_count].copy()
    visited: list[int] = []
    original = causal_spin_history._causal_slope_at_knot

    def recording_slope(
        spin_values: np.ndarray,
        time_values: np.ndarray,
        knot: int,
    ) -> np.ndarray:
        visited.append(knot)
        return original(spin_values, time_values, knot)

    monkeypatch.setattr(
        causal_spin_history,
        "_causal_slope_at_knot",
        recording_slope,
    )
    actual = append_causal_frozen_spin_slopes_in_place(
        slopes,
        spin,
        time,
        old_count=old_count,
    )

    assert visited == [old_count]
    np.testing.assert_array_equal(actual[:old_count], prefix)
    np.testing.assert_array_equal(actual, expected)


def test_rotating_spin_interpolation_has_third_order_interior_convergence() -> None:
    errors: list[float] = []
    angular_frequency = 3.1
    fraction = 0.37
    for knot_count in (17, 33, 65):
        time = np.linspace(0.0, 1.0, knot_count)
        spin = np.stack(
            (
                np.cos(angular_frequency * time),
                np.sin(angular_frequency * time),
                0.2 * np.cos(0.5 * angular_frequency * time),
            ),
            axis=1,
        )
        slopes = causal_frozen_spin_slopes_per_ns(spin, time)
        level_errors = []
        for segment in range(2, knot_count - 1):
            duration = time[segment + 1] - time[segment]
            u = fraction
            h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
            h10 = u**3 - 2.0 * u**2 + u
            h01 = -2.0 * u**3 + 3.0 * u**2
            h11 = u**3 - u**2
            interpolated = (
                h00 * spin[segment]
                + h10 * duration * slopes[segment]
                + h01 * spin[segment + 1]
                + h11 * duration * slopes[segment + 1]
            )
            sample_time = time[segment] + fraction * duration
            expected = np.array(
                [
                    np.cos(angular_frequency * sample_time),
                    np.sin(angular_frequency * sample_time),
                    0.2 * np.cos(0.5 * angular_frequency * sample_time),
                ]
            )
            level_errors.append(float(np.linalg.norm(interpolated - expected)))
        errors.append(max(level_errors))

    assert errors[0] / errors[1] > 7.5
    assert errors[1] / errors[2] > 7.5
