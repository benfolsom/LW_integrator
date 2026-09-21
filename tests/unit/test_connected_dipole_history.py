"""Native connected history, drift refusal and exact checkpoint continuation."""

from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from core.full_dipole_history import FullDipoleHistory
from core.connected_dipole_history import SourceDipoleError
from core.preserved_source import rounded_state


def fixture(count=41, exponential=False):
    times = np.arange(-10, count - 10) / 64
    reference = np.zeros((4, 4))
    reference[1, 2], reference[2, 1] = 0.003, -0.003
    high = np.zeros((count, 4, 4))
    low = np.zeros_like(high)
    with localcontext() as context:
        context.prec = 80
        for i, t in enumerate(times):
            if t <= 0:
                continue
            value = Decimal("1e-12") * Decimal.from_float(t) ** 5
            if exponential:
                value *= Decimal.from_float(t).exp()
            head = float(value)
            tail = float(value - Decimal.from_float(head))
            high[i, 1, 2], high[i, 2, 1] = head, -head
            low[i, 1, 2], low[i, 2, 1] = tail, -tail
    return FullDipoleHistory(
        times,
        np.zeros((count, 3)),
        np.zeros((count, 3)),
        rounded_state(reference, high, low),
        1.0,
        inertial_until=0.0,
        startup_fit="one_sided",
        dipole_reference=reference,
        dipole_high=high,
        dipole_low=low,
    )


def connected(count=41):
    return fixture(count).with_connected_dipoles(1e-20)


def test_quintic_accuracy_and_low_order_joins():
    history = connected()
    for i, segment in enumerate(history.segments):
        if segment.start < 0:
            assert segment.dipole_change is None
            continue
        assert segment.dipole.shape == (11, 4, 4)
        assert segment.dipole_error < 1e-25
        expected = 1e-12 * segment.end**5
        actual = np.polynomial.polynomial.polyval(1.0, segment.dipole_change)[1, 2]
        assert abs(actual / expected - 1) < 1e-10
        if segment.start == 0:
            np.testing.assert_array_equal(segment.dipole_change[:4], 0.0)
        else:
            for order in (1, 2, 3):
                np.testing.assert_allclose(
                    history.segments[i - 1].sample(segment.start, order)[1],
                    segment.sample(segment.start, order)[1],
                    rtol=1e-13,
                    atol=0.0,
                )


def test_restart_and_append_are_exact():
    source = fixture()
    prefix = connected(30)
    payload = prefix.to_checkpoint_payload()
    assert payload["format"] == "full-dipole-history-v6"
    restored = FullDipoleHistory.from_checkpoint_payload(payload)
    for i in range(30, len(source.time)):
        old = prefix.segments
        kwargs = dict(dipole_parts=(source.dipole_high[i], source.dipole_low[i]))
        prefix = prefix.append(
            source.time[i],
            source.position[i],
            source.velocity[i],
            source.dipole[i],
            **kwargs,
        )
        restored = restored.append(
            source.time[i],
            source.position[i],
            source.velocity[i],
            source.dipole[i],
            **kwargs,
        )
        assert all(a is b for a, b in zip(old, prefix.segments))
    whole = connected()
    for a, b, d in zip(prefix.segments, restored.segments, whole.segments):
        np.testing.assert_array_equal(a.dipole, b.dipole)
        np.testing.assert_array_equal(a.dipole, d.dipole)
        if a.dipole_change is not None:
            np.testing.assert_array_equal(a.dipole_change, b.dipole_change)
            assert not a.dipole_change.flags.writeable


def test_tight_budget_refuses_without_mutation():
    source = fixture(exponential=True)
    with pytest.raises(SourceDipoleError, match="drift"):
        source.with_connected_dipoles(1e-35)
    assert source.segments == ()
    assert source.dipole_reconstruction == "endpoint"


@pytest.mark.parametrize("budget", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        fixture().with_connected_dipoles(budget)


def test_no_silent_rewrite_or_sampling_change():
    with pytest.raises(ValueError, match="publishing"):
        fixture().completed().with_connected_dipoles(1e-20)
    with pytest.raises(ValueError, match="consecutive"):
        connected().with_time_balanced_sampling()


def test_checkpoint_requires_mode_and_budget():
    payload = connected().to_checkpoint_payload()
    del payload["dipole_tolerance"]
    with pytest.raises(ValueError, match="v6"):
        FullDipoleHistory.from_checkpoint_payload(payload)
    legacy = fixture().completed().to_checkpoint_payload()
    assert legacy["format"] == "full-dipole-history-v4"
    legacy["dipole_reconstruction"] = "connected_direct"
    with pytest.raises(ValueError, match="v6"):
        FullDipoleHistory.from_checkpoint_payload(legacy)


def test_unpreserved_mode_is_rejected():
    with pytest.raises(ValueError, match="preserved"):
        replace(
            fixture(),
            dipole_reference=None,
            dipole_high=None,
            dipole_low=None,
            dipole_reconstruction="connected_direct",
            dipole_tolerance=1e-20,
        )
