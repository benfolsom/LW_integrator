import numpy as np
import pytest

from core.exact_pair_trial import ExactPairEOMOptions, make_exact_role_eom_advance
from core.types import MagneticDipoleConfig


@pytest.mark.parametrize("value", [False, 1, "midpoint"])
def test_diagnostic_requires_callable(value):
    with pytest.raises(ValueError, match="callable"):
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            moment_impulse_diagnostic=value,
        )


def test_diagnostic_rejects_unsupported_radiation():
    with pytest.raises(ValueError, match="only off or medina_lad"):
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            radiation_reaction_mode="larmor",
            moment_impulse_diagnostic=lambda **k: np.zeros(4),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_adapter_forwards_only_an_explicit_diagnostic(monkeypatch, enabled):
    import core.self_consistency as module

    received = {}

    def fake(*args, **kwargs):
        received["eom"] = args[0]
        received.update(kwargs)
        return args[2][0]

    monkeypatch.setattr(module, "self_consistent_step", fake)
    diagnostic = (lambda **kwargs: np.zeros(4)) if enabled else None
    callback = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            moment_impulse_diagnostic=diagnostic,
        )
    )
    callback(0.1, {}, {}, object())
    if enabled:
        assert received["eom"].keywords["moment_impulse_diagnostic"] is diagnostic
    else:
        from core.equations import retarded_equations_of_motion

        assert received["eom"] is retarded_equations_of_motion
    assert "moment_impulse_diagnostic" not in received


def test_medina_adapter_repeats_from_same_accepted_start(monkeypatch):
    import core.self_consistency as module

    start = {"x": np.array([0.0])}
    starts, estimates = [], []

    def fake(eom, h, trajectory, *args, **kwargs):
        starts.append(trajectory[0])
        estimates.append(eom.keywords["moment_radiation_force_native"].copy())
        return {
            "x": np.array([123.0]),
            "_moment_applied_medina_force_native": np.array([[1.0, 0.0, 0.0]]),
            "medina_force_derivative_ready": np.array([True]),
            "medina_impulse_capped": np.array([False]),
        }

    monkeypatch.setattr(module, "self_consistent_step", fake)
    callback = make_exact_role_eom_advance(
        ExactPairEOMOptions(
            aperture_radius_mm=1.0,
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            radiation_reaction_mode="medina_lad",
            moment_impulse_diagnostic=lambda **k: np.zeros(4),
        )
    )
    result = callback(0.1, start, {}, object())
    assert len(starts) == 2 and all(s is start for s in starts)
    np.testing.assert_array_equal(estimates, [[[0, 0, 0]], [[1, 0, 0]]])
    np.testing.assert_array_equal(start["x"], [0.0])
    assert result["_moment_medina_iterations"] == 2
