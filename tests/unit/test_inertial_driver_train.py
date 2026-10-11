"""One-bunch trains must retain the single-driver exact startup behaviour."""

import numpy as np
import pytest

from core.types import DriverTrainConfig
from core.types import SpaceChargeConfig
from tests.unit.test_exact_visibility_gates import crossing_run
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_inertial_prehistory import _state


@pytest.mark.parametrize("prehistory_steps", [0, 20])
@pytest.mark.parametrize("particles", [1, 2])
def test_one_bunch_train_is_bit_identical_to_single_driver(prehistory_steps, particles):
    driver = _merge(
        *[
            _state(
                position_mm=(2 * i, 0.5, 0),
                beta=(0, 0.1, 0),
                observer_charge=10,
                source_charge=10,
            )
            for i in range(particles)
        ]
    )
    options = dict(
        driver_state=driver,
        space_charge=SpaceChargeConfig(enabled=True, retarded=True, softening_mm=0),
    )
    single = crossing_run(8, None, **options)
    train = crossing_run(
        8,
        None,
        **options,
        driver_train=DriverTrainConfig(
            enabled=True,
            bunch_count=1,
            prehistory_steps=prehistory_steps,
        ),
    )
    for before_bunch, after_bunch in zip(single[:2], train[:2]):
        assert len(before_bunch) == len(after_bunch) == 9
        for before, after in zip(before_bunch, after_bunch):
            for key, values in before.items():
                if isinstance(values, np.ndarray):
                    assert values.tobytes() == after[key].tobytes(), key


def test_charged_multibunch_train_excludes_other_bunches_in_preflight_and_steps(
    monkeypatch,
):
    import core.charge_source_interactions as interactions
    import core.integration_runner as runner

    # Hold the analytic seed window fixed so translated control runs do not
    # acquire different startup derivative reconstruction windows.
    monkeypatch.setattr(
        runner, "_estimate_inertial_prehistory_duration_ns", lambda *args, **kwargs: 1.0
    )

    driver = _merge(
        *[
            _state(
                position_mm=(2 * i, 0.5, 0),
                beta=(0, 0.1, 0),
                observer_charge=1,
                source_charge=1,
            )
            for i in range(2)
        ]
    )
    rider = _state(
        position_mm=(0, 0, 10), beta=(0, 0, 0.05), observer_charge=0, source_charge=0
    )
    recorded = []
    evaluate = interactions.evaluate_retarded_charge_source_interaction_native

    def record(history, event, **kwargs):
        excluded = kwargs.get("excluded_source_indices", ())
        if event.time_ns == 0 and excluded:
            recorded.append(tuple(excluded))
        return evaluate(history, event, **kwargs)

    monkeypatch.setattr(
        interactions, "evaluate_retarded_charge_source_interaction_native", record
    )
    options = dict(
        rider_state=rider,
        driver_state=driver,
        space_charge=SpaceChargeConfig(enabled=True, retarded=True, softening_mm=0),
    )
    train = crossing_run(
        4,
        None,
        **options,
        driver_train=DriverTrainConfig(
            enabled=True, bunch_count=3, z_offsets_mm=(0, 20, 50)
        ),
    )[1]
    expected = [
        (0, 2, 3, 4, 5),
        (1, 2, 3, 4, 5),
        (0, 1, 2, 4, 5),
        (0, 1, 3, 4, 5),
        (0, 1, 2, 3, 4),
        (0, 1, 2, 3, 5),
    ]
    assert recorded[:6] == expected
    for bunch, offset in enumerate((0, 20, 50)):
        translated = {key: values.copy() for key, values in driver.items()}
        translated["z"] += offset
        single = crossing_run(4, None, **{**options, "driver_state": translated})[1]
        for combined, isolated in zip(train, single):
            for key in ("Px", "Py", "Pz", "Pt", "bx", "by", "bz", "gamma"):
                np.testing.assert_allclose(
                    combined[key][2 * bunch : 2 * bunch + 2],
                    isolated[key],
                    rtol=2e-14,
                    atol=1e-13,
                )
    # A charged two-particle bunch really evolves: equality is not a neutral
    # ballistic identity that could conceal omitted within-bunch forces.
    assert np.any(train[-1]["Px"] != train[0]["Px"])
