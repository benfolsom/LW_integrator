"""Persistent train sources, own-bunch exclusions, and checkpoint continuation."""

from dataclasses import fields

import numpy as np
import pytest

from core.integration_runner import IntegrationCancelled
from core.integration_checkpoint import CheckpointCompatibilityError
from core.types import CheckpointConfig, DriverTrainConfig
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_visibility_gates import crossing_run, gate


@pytest.mark.slow
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
    with pytest.raises(
        CheckpointCompatibilityError, match="physics/configuration fingerprint"
    ):
        crossing_run(
            32,
            gate(),
            **{
                **options,
                "driver_train": DriverTrainConfig(
                    enabled=True, bunch_count=5, z_offsets_mm=(0, 2, 4, 6, 9)
                ),
            },
            checkpoint=CheckpointConfig(resume_from=directory),
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
