"""Persistent train sources, own-bunch exclusions, and checkpoint continuation."""

from dataclasses import fields, replace

import numpy as np
import pytest

from core.integration_runner import IntegrationCancelled
from core.integration_checkpoint import CheckpointCompatibilityError
from core.types import CheckpointConfig, DriverTrainConfig
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_exact_source_reduction import magnetic
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_inertial_prehistory import _state
from tests.unit.test_exact_visibility_gates import crossing_run, gate


@pytest.mark.slow
@pytest.mark.parametrize("breathing", [False, True])
def test_five_bunch_cloud_gate_checkpoint_is_bitwise_uninterrupted(tmp_path, breathing):
    options = dict(
        backend="numba_analytic_charge_response_serial",
        duration=0.15,
        macroparticle_smearing=replace(
            cloud_config(4, 0.02), breathing_enabled=breathing
        ),
        driver_state=_merge(
            *[
                _state(
                    position_mm=(x, 0.5, 0),
                    beta=(x * 0.001 if breathing else 0, 0.1, 0),
                    observer_charge=5,
                )
                for x in (-0.05, 0.05)
            ]
        ),
        magnetic_dipole=replace(
            magnetic(1), exact_retarded_backend="numba_analytic_charge_response_serial"
        ),
        driver_train=DriverTrainConfig(
            enabled=True, bunch_count=5, z_offsets_mm=(0, 2, 4, 6, 8)
        ),
    )
    baseline = crossing_run(32, gate(), **options)
    # Reduce the two-particle template first, then expand to five bunches.
    assert len(baseline[1][0]["x"]) == 5
    assert baseline[1][0]["macro_population"].sum() == 10
    assert baseline[1][0]["q_source"].sum() == 50
    assert np.all(baseline[3].exact_source_original_count == 2)
    np.testing.assert_allclose(
        baseline[3].exact_source_cell_sigma_mm, 0.05 / np.sqrt(2)
    )
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
