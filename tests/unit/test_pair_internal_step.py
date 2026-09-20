"""Internal adaptivity must preserve native cadence, reaction totals and restart."""

import copy
from dataclasses import asdict
import json

import numpy as np
import pytest

from core import momentum_center_pair as pair
from core.pair_step_error import InternalStepSettings
from core.step_doubling import ErrorScale
from tests.unit.test_preserved_source import fresh_pair


def configured(mode="full_dipole_coupled"):
    scales = InternalStepSettings(*[ErrorScale(1e-8, 1e-6) for _ in range(6)])
    return dict(
        fresh_pair(mode),
        integration_method="dop853",
        internal_step_control=asdict(scales),
    )


@pytest.mark.parametrize("mode", ["off", "full_dipole_coupled"])
def test_native_cadence_matching_state_and_restart(mode):
    initial = configured(mode)
    original = copy.deepcopy(initial)
    width = 0.01 / pair.c
    first, records = pair.advance_pair(initial, width)
    assert first["accepted_steps"] == 1
    assert (
        len(first["histories"][0]["time"]) == len(initial["histories"][0]["time"]) + 1
    )
    assert records[0]["particles"][0]["internal_step_control"]["accepted_steps"] >= 1
    restored, _ = pair.advance_pair(json.loads(json.dumps(first)), width)
    direct, _ = pair.advance_pair(initial, width, 2)
    assert restored == direct and initial == original
    plain = dict(initial)
    del plain["internal_step_control"]
    fixed, _ = pair.advance_pair(plain, width)
    np.testing.assert_array_equal(first["states"], fixed["states"])
    assert first["source_precision"] == fixed["source_precision"]
    if mode != "off":
        for row, expected in zip(
            first["dipole_reaction_ledger"], fixed["dipole_reaction_ledger"]
        ):
            np.testing.assert_allclose(
                row["applied_impulse_native"],
                expected["applied_impulse_native"],
                rtol=1e-12,
                atol=1e-24,
            )
            np.testing.assert_allclose(
                row["applied_torque_native"],
                expected["applied_torque_native"],
                rtol=1e-12,
                atol=1e-24,
            )


def test_incompatible_method_and_invalid_scale_rejected():
    initial = configured()
    with pytest.raises(ValueError, match="requires preserved DOP853"):
        pair.advance_pair(dict(initial, integration_method="rk4"), 0.01 / pair.c)
    bad = copy.deepcopy(initial)
    bad["internal_step_control"]["impulse_native"]["absolute"] = -1
    with pytest.raises(ValueError, match="error scales"):
        pair.advance_pair(bad, 0.01 / pair.c)


def test_internal_cli_parity_and_settings_preservation(tmp_path):
    from lw_integrator.nonlinear_pair import main, configure_checkpoint
    from core.nonlinear_pair_adaptive import initialize_adaptive
    from tests.unit.test_nonlinear_pair_adaptive import settings

    initial = configured()
    source, output = tmp_path / "source.json", tmp_path / "out.json"
    encoded = json.dumps(initial)
    source.write_text(encoded)
    width = 0.01 / pair.c
    expected, _ = pair.advance_pair(initial, width)
    assert (
        main(
            [
                "--checkpoint",
                str(source),
                "--output",
                str(output),
                "--step-ns",
                str(width),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == expected
    assert source.read_text() == encoded
    changed = copy.deepcopy(initial["internal_step_control"])
    changed["impulse_native"]["relative"] *= 0.5
    with pytest.raises(ValueError, match="retains its internal"):
        configure_checkpoint(expected, internal_step_control=changed)
    with pytest.raises(ValueError, match="stacking both"):
        initialize_adaptive(initial, settings(width), width)
