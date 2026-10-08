"""Persistent train sources, own-bunch exclusions, and checkpoint continuation."""

from dataclasses import fields

import numpy as np
import pytest

from core.integration_runner import IntegrationCancelled
from core.types import CheckpointConfig, DriverTrainConfig, SpaceChargeConfig
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_visibility_gates import crossing_run, gate
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_inertial_prehistory import _state


def test_five_cloud_bunches_have_independent_own_bunch_fields(monkeypatch):
    import core.integration_runner as runner

    monkeypatch.setattr(
        runner, "_estimate_inertial_prehistory_duration_ns", lambda *args, **kw: 1.0
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
    options = dict(
        rider_state=rider,
        driver_state=driver,
        duration=0.002,
        macroparticle_smearing=cloud_config(4, 0.02),
        space_charge=SpaceChargeConfig(enabled=True, retarded=True, softening_mm=0),
    )
    offsets = (0, 20, 40, 60, 80)
    combined = crossing_run(
        2,
        None,
        **options,
        driver_train=DriverTrainConfig(
            enabled=True, bunch_count=5, z_offsets_mm=offsets
        ),
    )[1]
    for bunch, offset in enumerate(offsets):
        translated = {key: value.copy() for key, value in driver.items()}
        translated["z"] += offset
        isolated = crossing_run(2, None, **{**options, "driver_state": translated})[1]
        for train, single in zip(combined, isolated):
            for key in ("Px", "Py", "Pz", "Pt", "bx", "by", "bz", "gamma"):
                np.testing.assert_allclose(
                    train[key][2 * bunch : 2 * bunch + 2],
                    single[key],
                    rtol=2e-14,
                    atol=1e-13,
                )
    assert np.any(combined[-1]["Px"] != combined[0]["Px"])


def test_five_bunch_cloud_gate_checkpoint_is_bitwise_uninterrupted(tmp_path):
    options = dict(
        backend="numba_analytic_charge_response_serial",
        duration=0.15,
        macroparticle_smearing=cloud_config(4, 0.02),
        driver_train=DriverTrainConfig(
            enabled=True, bunch_count=5, z_offsets_mm=(0, 2, 4, 6, 8)
        ),
    )
    baseline = crossing_run(32, gate(), **options)
    directory = str(tmp_path / "cloud-train")
    cancelled = False

    def progress(current, total):
        nonlocal cancelled
        if current >= 20:
            cancelled = True

    with pytest.raises(IntegrationCancelled):
        crossing_run(
            32,
            gate(),
            **options,
            checkpoint=CheckpointConfig(
                enabled=True, directory=directory, interval_steps=2, interval_seconds=0
            ),
            progress_callback=progress,
            cancel_callback=lambda: cancelled,
        )
    resumed = crossing_run(
        32,
        gate(),
        **options,
        checkpoint=CheckpointConfig(
            resume_from=directory, interval_steps=2, interval_seconds=0
        ),
    )
    for first, second in zip(baseline[2:4], resumed[2:4]):
        for descriptor in fields(type(first)):
            value = getattr(first, descriptor.name)
            if isinstance(value, np.ndarray):
                other = getattr(second, descriptor.name)
                assert value.dtype == other.dtype and value.shape == other.shape
                assert value.tobytes() == other.tobytes(), descriptor.name
    assert np.any(baseline[0][-1]["potential_inclusion_delta_A_t"] != 0)
    assert np.any(baseline[0][-1]["by"] != baseline[0][0]["by"])
