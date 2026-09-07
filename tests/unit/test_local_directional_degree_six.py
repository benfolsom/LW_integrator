"""Check the higher-degree candidate against a smooth analytic source history."""

import numpy as np
import pytest

from core.causal_local_source_history import CausalLocalSourceHistoryUnavailableError
from core.causal_local_source_jet import (
    LocalSourceJetFitConfig,
    evaluate_causal_local_source_jet_native,
)
from core.retarded_fields import ObserverEvent
from tests.unit.test_causal_local_source_jet import (
    _circular_history,
    _exact_circular_response,
)


@pytest.mark.parametrize("degree", [5, 6])
@pytest.mark.parametrize("samples", ["exact_start", "interval_mean"])
def test_smooth_directional_response_refines_toward_analytic_history(degree, samples):
    event = ObserverEvent(0.03, (0.8, 0.2, 0.3))
    direction = np.array([500.0, 100.0, -70.0, 50.0])
    expected = _exact_circular_response(
        event, observer_four_velocity_mm_ns=direction
    ).partial_antisymmetric_response_along_velocity
    errors = []
    for count in (241, 961):
        actual, _ = evaluate_causal_local_source_jet_native(
            _circular_history(sample_count=count),
            event,
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(
                0.005,
                acceleration_degree=degree,
                acceleration_samples=samples,
                window_alignment="past",
            ),
            observer_four_velocity_mm_ns=direction,
        )
        error = np.linalg.norm(
            actual.partial_antisymmetric_response_along_velocity - expected
        ) / np.linalg.norm(expected)
        errors.append(error)
    assert errors[-1] < 1e-4
    # Fixed-window bias can limit refinement; do not require a higher fit
    # degree to improve every already-small error component monotonically.
    assert errors[-1] < max(1e-8, 0.4 * errors[0])


def test_degree_six_retains_condition_number_guard():
    with pytest.raises(
        CausalLocalSourceHistoryUnavailableError, match="condition-number"
    ):
        evaluate_causal_local_source_jet_native(
            _circular_history(sample_count=961),
            ObserverEvent(0.03, (0.8, 0.2, 0.3)),
            magnetic_moment_native=-1.7,
            fit=LocalSourceJetFitConfig(
                0.005,
                acceleration_degree=6,
                window_alignment="past",
                maximum_condition_number=1e4,
            ),
            observer_four_velocity_mm_ns=(500.0, 100.0, -70.0, 50.0),
        )
