from __future__ import annotations

import numpy as np
import pytest

import core.images as images
from core.constants import C_MMNS


def _make_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = -1.0,
    t: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: float = 1.0,
    mass: float = 1.0,
    gamma: float = 1.0,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.0,
    char_time: float = 1e-3,
) -> dict[str, np.ndarray]:
    return {
        "x": np.array([x], dtype=float),
        "y": np.array([y], dtype=float),
        "z": np.array([z], dtype=float),
        "t": np.array([t], dtype=float),
        "Px": np.array([px], dtype=float),
        "Py": np.array([py], dtype=float),
        "Pz": np.array([pz], dtype=float),
        "Pt": np.array([gamma * mass * C_MMNS], dtype=float),
        "gamma": np.array([gamma], dtype=float),
        "bx": np.array([bx], dtype=float),
        "by": np.array([by], dtype=float),
        "bz": np.array([bz], dtype=float),
        "bdotx": np.array([0.0], dtype=float),
        "bdoty": np.array([0.0], dtype=float),
        "bdotz": np.array([0.0], dtype=float),
        "q": np.array([charge], dtype=float),
        "m": np.array([mass], dtype=float),
        "char_time": np.array([char_time], dtype=float),
    }


def test_random_sign_uses_half_probability_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(images.random, "random", lambda: 0.49)
    assert images._random_sign() == 1

    monkeypatch.setattr(images.random, "random", lambda: 0.5)
    assert images._random_sign() == -1


def test_zeros_like_state_preserves_layout_without_aliasing() -> None:
    state = _make_state(charge=2.0, mass=3.0, char_time=0.25)

    result = images._zeros_like_state(state)

    for key in ("x", "y", "z", "t", "Px", "Py", "Pz", "Pt", "gamma"):
        assert result[key].tolist() == pytest.approx([0.0])
        assert result[key] is not state[key]

    assert result["q"].tolist() == pytest.approx([2.0])
    assert result["m"].tolist() == pytest.approx([3.0])
    assert result["char_time"].tolist() == pytest.approx([0.25])

    result["q"][0] = 9.0
    result["m"][0] = 7.0
    result["char_time"][0] = 6.0
    assert state["q"][0] == pytest.approx(2.0)
    assert state["m"][0] == pytest.approx(3.0)
    assert state["char_time"][0] == pytest.approx(0.25)


def test_radial_weight_handles_degenerate_aperture_radius() -> None:
    weights = images._radial_weight(
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 0.0]),
        aperture_radius=0.0,
        shift=0.0,
        plateau=0.2,
    )

    assert weights.tolist() == pytest.approx([0.0, 0.5])


def test_generate_conducting_image_applies_macroparticle_spread_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_state(x=1.0, y=-0.5, z=-1.5, charge=0.8, mass=2.0)
    baseline = images.generate_conducting_image(
        source,
        wall_z=0.0,
        aperture_radius=0.4,
        subcharge_count=4,
        use_weighting=False,
    )

    calls: list[float] = []
    offsets = iter([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])

    def _fake_normal(mean: float, sigma: float) -> float:
        assert mean == 0.0
        calls.append(sigma)
        return next(offsets)

    monkeypatch.setattr(images.np.random, "normal", _fake_normal)

    perturbed = images.generate_conducting_image(
        source,
        wall_z=0.0,
        aperture_radius=0.4,
        subcharge_count=4,
        use_weighting=False,
        macroparticle_sigma_multiplier=2.0,
        bunch_transv_dist=0.1,
        bunch_transv_mom=0.2,
        timestep=0.5,
        step_number=3,
    )

    expected_sigma = np.sqrt((0.1 * 2.0) ** 2 + (0.2 * 2.0 / 2.0 * 0.5 * 3.0) ** 2)
    assert calls == pytest.approx([expected_sigma] * 8)
    assert (perturbed["x"] - baseline["x"]).tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7])
    assert (perturbed["y"] - baseline["y"]).tolist() == pytest.approx(
        [-0.2, -0.4, -0.6, -0.8]
    )


def test_generate_conducting_image_zero_aperture_zeroes_weighted_on_axis_charge() -> None:
    source = _make_state(z=-1.0, charge=1.0)

    image = images.generate_conducting_image(
        source,
        wall_z=0.0,
        aperture_radius=0.0,
        subcharge_count=4,
        use_weighting=True,
    )

    assert np.allclose(image["q"], 0.0)


def test_generate_conducting_image_uses_position_only_sigma_without_momentum_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_state(x=0.5, y=0.25, z=-1.0, charge=1.0, mass=2.0)
    calls: list[float] = []

    def _fake_normal(mean: float, sigma: float) -> float:
        assert mean == 0.0
        calls.append(sigma)
        return 0.0

    monkeypatch.setattr(images.np.random, "normal", _fake_normal)

    images.generate_conducting_image(
        source,
        wall_z=0.0,
        aperture_radius=0.4,
        subcharge_count=4,
        use_weighting=False,
        macroparticle_sigma_multiplier=3.0,
        macroparticle_use_momentum_errors=False,
        bunch_transv_dist=0.1,
        bunch_transv_mom=0.2,
        timestep=0.5,
        step_number=3,
    )

    assert calls == pytest.approx([0.3] * 8)


def test_generate_switching_image_reflects_particles_above_wall_before_cutoff() -> None:
    source = _make_state(z=0.2, pz=-0.5, bz=-0.2, charge=1.0)

    image = images.generate_switching_image(
        source,
        wall_z=0.0,
        aperture_radius=0.25,
        cut_z=0.5,
    )

    assert image["z"][0] == pytest.approx(-0.2)
    assert image["Pz"][0] == pytest.approx(0.5)
    assert image["bz"][0] == pytest.approx(0.2)
    assert image["q"][0] == pytest.approx(-1.0)
