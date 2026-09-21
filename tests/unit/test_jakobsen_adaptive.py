"""Maintained controller adapter and explicit-model checkpoint regression."""

import json

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.jakobsen_step import JakobsenParticle, initial_canonical_state
from core.jakobsen_adaptive import checkpoint, restore, integrate
from core.step_doubling import StepDoublingTolerances, ErrorScale
from core.rfs import electromagnetic_field_tensor_native


def provider(t, position):
    field = 0.02 * c * c
    x, y, _ = position
    a = np.array([0, -field * y / 2, field * x / 2, 0])
    da = np.zeros((4, 4))
    da[1, 2], da[2, 1] = field / 2, -field / 2
    return (
        a,
        da,
        electromagnetic_field_tensor_native([0, 0, 0], [0, 0, field]),
        np.zeros((4, 4, 4)),
    )


def setup():
    particle = JakobsenParticle(1, 1, 5.5856946893)
    w = np.array([0.8, 0, 0.3])
    state = initial_canonical_state(
        time_ns=0,
        position_mm=[0, 0, 0],
        four_velocity_mm_ns=c * np.r_[np.sqrt(1 + w @ w), w],
        rest_spin_angular_momentum=c * np.array([1e-6, 0, 2e-6]),
        particle=particle,
        provider=provider,
    )
    return particle, checkpoint(
        state, particle=particle, provider_id="uniform-B0.02-v1", next_step_ns=0.3 / c
    )


def test_checkpoint_resume_and_trial_rejection():
    particle, start = setup()
    options = dict(
        particle=particle,
        provider=provider,
        provider_id="uniform-B0.02-v1",
        tolerances=StepDoublingTolerances(
            ErrorScale(1e-9, 1e-9),
            ErrorScale(1e-8, 1e-10),
            ErrorScale(1e-12, 1e-9),
            ErrorScale(1e-8, 1e-10),
        ),
        maximum_step_ns=0.3 / c,
    )
    original = json.dumps(start)
    first, _ = integrate(start, 1 / c, **options)
    saved = json.loads(json.dumps(first))
    resumed, records = integrate(saved, 2 / c, **options)
    continued, _ = integrate(first, 2 / c, **options)
    np.testing.assert_array_equal(resumed["state"], continued["state"])
    assert resumed["accepted"] == continued["accepted"]
    assert resumed["rejected"] > 0
    assert records and max(row["error"] for row in records) <= 1
    assert json.dumps(start) == original


@pytest.mark.parametrize(
    "key,value", [("model", "rfs"), ("provider_id", "different"), ("next_step_ns", -1)]
)
def test_checkpoint_mismatch_rejected(key, value):
    particle, payload = setup()
    payload[key] = value
    with pytest.raises(ValueError):
        restore(payload, particle=particle, provider_id="uniform-B0.02-v1")


def test_reaction_model_cannot_resume_ordinary_checkpoint():
    from dataclasses import replace

    particle, payload = setup()
    with pytest.raises(ValueError, match="mismatch"):
        restore(
            payload,
            particle=replace(particle, reaction_mode="experimental_linear_spin"),
            provider_id="uniform-B0.02-v1",
        )


def test_reaction_requires_explicit_provider_derivative():
    from dataclasses import replace
    from core.jakobsen_step import canonical_rhs

    particle, payload = setup()
    with pytest.raises(ValueError, match="gradient_proper_rate"):
        canonical_rhs(
            np.asarray(payload["state"]),
            particle=replace(particle, reaction_mode="experimental_linear_spin"),
            provider=provider,
        )


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_checkpoint_rejects_invalid_counts(count):
    particle, payload = setup()
    with pytest.raises(ValueError, match="controller counts"):
        checkpoint(
            payload["state"],
            particle=particle,
            provider_id="test",
            next_step_ns=0.1 / c,
            accepted=count,
        )


def test_canonical_step_rejects_missing_potential_derivative():
    from core.jakobsen_step import canonical_rhs

    particle, payload = setup()

    def incomplete(t, x):
        a, _, f, df = provider(t, x)
        return a, np.zeros((4, 1)), f, df

    with pytest.raises(ValueError, match="potential derivative"):
        canonical_rhs(
            np.asarray(payload["state"]), particle=particle, provider=incomplete
        )
