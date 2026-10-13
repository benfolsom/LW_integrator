"""Fixed-step cloud routing and degenerate-source arithmetic."""

import copy

import numpy as np
import pytest

from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_visibility_gates import crossing_run, gate
from tests.unit.test_inertial_prehistory import _state


@pytest.mark.parametrize("count,width", [(1, 0.0), (16, 0.0), (1, 0.2)])
def test_fixed_step_degenerate_cloud_is_bitwise_point_trajectory(count, width):
    point = crossing_run(2, gate(), duration=0.001)
    cloud = crossing_run(
        2, gate(), duration=0.001, macroparticle_smearing=cloud_config(count, width)
    )
    for plain, wrapped in zip(point[:2], cloud[:2]):
        for before, after in zip(plain, wrapped):
            for key, value in before.items():
                if isinstance(value, np.ndarray):
                    assert value.tobytes() == after[key].tobytes(), key


@pytest.mark.slow
def test_fixed_step_cloud_supports_joint_observer_gate_events():
    rider = _state(
        position_mm=(0, 0, 10),
        beta=(0, 0, 0.05),
        observer_charge=0.1,
        source_charge=0.0,
    )
    rider = {
        key: (
            np.concatenate((value, value))
            if isinstance(value, np.ndarray) and value.shape == (1,)
            else copy.deepcopy(value)
        )
        for key, value in rider.items()
    }
    rider["z"][1] += 1.0
    driver = _state(
        position_mm=(0, 0.5, 0),
        beta=(0, 0.1, 0),
        observer_charge=0.0,
        source_charge=0.1,
    )
    result = crossing_run(
        4,
        gate(),
        duration=0.08,
        rider_state=rider,
        driver_state=driver,
        macroparticle_smearing=cloud_config(4, 0.02),
    )
    assert len(result[0]) == len(result[1]) == 5
    for trajectory in result[:2]:
        for state in trajectory:
            for name in ("x", "y", "z", "t", "gamma"):
                assert np.isfinite(state[name]).all()
    # Both distinct observer arrival times must leave a potential ledger.
    last = result[0][-1]
    assert np.all(last["potential_inclusion_delta_A_t"] != 0)
    from core.potential_inclusion import decode_inclusion_state

    event_times = []
    for observer in range(2):
        knots = [
            knot
            for state in result[0]
            for record in [
                decode_inclusion_state(
                    str(state.get("potential_inclusion_state", ["", ""])[observer])
                )
            ]
            for knot in record.get("exact_gate_history", {}).get("knots", [])
        ]
        # Every shared split stores a knot for both observers. Only this
        # observer's visibility switches change its one-sided force.
        arrivals = {
            knot["t"]
            for knot in knots
            if not np.array_equal(knot["before"], knot["after"])
        }
        assert arrivals
        event_times.append(arrivals)
    assert event_times[0].isdisjoint(event_times[1])
