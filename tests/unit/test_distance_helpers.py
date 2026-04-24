from __future__ import annotations

import numpy as np
import pytest

import core.distances as distances
from core.constants import NUMERICAL_EPSILON


def _make_state(
    *,
    x: list[float] | None = None,
    y: list[float] | None = None,
    z: list[float] | None = None,
    t: list[float] | None = None,
    bx: list[float] | None = None,
    by: list[float] | None = None,
    bz: list[float] | None = None,
    char_time: list[float] | None = None,
) -> dict[str, np.ndarray]:
    x_vals = [0.0] if x is None else x
    size = len(x_vals)

    def _arr(values: list[float] | None, default: float = 0.0) -> np.ndarray:
        if values is None:
            values = [default] * size
        return np.array(values, dtype=float)

    return {
        "x": _arr(x),
        "y": _arr(y),
        "z": _arr(z),
        "t": _arr(t),
        "bx": _arr(bx),
        "by": _arr(by),
        "bz": _arr(bz),
        "char_time": _arr(char_time, 1e-3),
    }


def test_dot_beta_nhat_projects_velocity_onto_line_of_sight() -> None:
    state = _make_state(bx=[0.2], by=[-0.3], bz=[0.5])
    nhat = {
        "R": np.array([1.0]),
        "nx": np.array([0.4]),
        "ny": np.array([0.5]),
        "nz": np.array([-0.6]),
    }

    result = distances._dot_beta_nhat(state, nhat, 0)

    assert result == pytest.approx(0.2 * 0.4 + (-0.3) * 0.5 + 0.5 * (-0.6))


def test_locate_retarded_index_handles_negative_and_bracketed_targets() -> None:
    trajectory_ext = [
        _make_state(t=[0.0]),
        _make_state(t=[1.0]),
        _make_state(t=[2.0]),
        _make_state(t=[3.0]),
    ]

    assert distances._locate_retarded_index(trajectory_ext, 3, 0, -1.0) == 3
    assert distances._locate_retarded_index(trajectory_ext, 3, 0, 1.5) == 2
    assert distances._locate_retarded_index(trajectory_ext, 3, 0, 10.0) == 0


def test_compute_retarded_distance_uses_per_particle_retarded_indices() -> None:
    trajectory = [
        _make_state(x=[0.0, 0.0]),
        _make_state(x=[1.0, 1.0]),
    ]
    trajectory_ext = [
        _make_state(x=[2.0, 4.0]),
        _make_state(x=[1.0, 7.0]),
    ]

    result = distances.compute_retarded_distance(
        trajectory,
        trajectory_ext,
        index_traj=1,
        index_part=0,
        indices_ret=np.array([1, 0]),
    )

    assert result["R"][0] == pytest.approx(NUMERICAL_EPSILON)
    assert result["nx"][0] == pytest.approx(0.0)
    assert result["ny"][0] == pytest.approx(0.0)
    assert result["nz"][0] == pytest.approx(0.0)
    assert result["R"][1] == pytest.approx(3.0)
    assert result["nx"][1] == pytest.approx(-1.0)
    assert result["ny"][1] == pytest.approx(0.0)
    assert result["nz"][1] == pytest.approx(0.0)


def test_compute_instantaneous_distance_handles_overlap_and_normalization() -> None:
    vector = _make_state(x=[1.0], y=[0.0], z=[0.0])
    vector_ext = _make_state(x=[1.0, 4.0], y=[0.0, 0.0], z=[0.0, 0.0])

    result = distances.compute_instantaneous_distance(vector, vector_ext, 0)

    assert result["R"][0] == pytest.approx(NUMERICAL_EPSILON)
    assert result["nx"][0] == pytest.approx(0.0)
    assert result["R"][1] == pytest.approx(3.0)
    assert result["nx"][1] == pytest.approx(-1.0)
    assert result["ny"][1] == pytest.approx(0.0)
    assert result["nz"][1] == pytest.approx(0.0)


def test_chrono_match_indices_returns_linear_interpolation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(t=[float(step)], x=[10.0]) for step in range(4)]
    trajectory_ext = [_make_state(t=[float(step)], x=[0.0]) for step in range(4)]

    monkeypatch.setattr(distances, "_compute_delta_t", lambda **_: 1.5)

    result = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=3,
        index_part=0,
        interpolate=True,
        tolerance=0.1,
    )

    assert result.indices.tolist() == [2]
    assert result.indices_next.tolist() == [1]
    assert result.weights.tolist() == pytest.approx([0.5])
    assert result.residuals.tolist() == pytest.approx([0.5])
    assert result.max_residual == pytest.approx(0.5)
    assert result.needs_interpolation.tolist() == [True]
    assert result.use_cubic is False


def test_chrono_match_indices_high_precision_returns_cubic_bracketing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(t=[float(step)], x=[10.0]) for step in range(6)]
    trajectory_ext = [_make_state(t=[float(step)], x=[0.0]) for step in range(6)]

    monkeypatch.setattr(distances, "_compute_delta_t", lambda **_: 2.5)

    result = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=5,
        index_part=0,
        interpolate=True,
        tolerance=0.1,
        high_precision=True,
    )

    assert result.indices_prev.tolist() == [1]
    assert result.indices_next.tolist() == [2]
    assert result.indices.tolist() == [3]
    assert result.indices_next2.tolist() == [4]
    assert result.weights.tolist() == pytest.approx([0.5])
    assert result.needs_interpolation.tolist() == [True]
    assert result.use_cubic is True


def test_chrono_match_indices_adaptive_tolerance_can_suppress_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(t=[float(step)], x=[10.0]) for step in range(4)]
    trajectory_ext = [_make_state(t=[float(step)], x=[0.0]) for step in range(4)]

    monkeypatch.setattr(distances, "_compute_delta_t", lambda **_: 1.5)

    result = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=3,
        index_part=0,
        interpolate=True,
        adaptive_tolerance=True,
        timestep_h=10.0,
    )

    assert result.indices.tolist() == [2]
    assert result.indices_next.tolist() == [2]
    assert result.weights.tolist() == pytest.approx([1.0])
    assert result.needs_interpolation.tolist() == [False]


def test_chrono_match_indices_uses_char_time_fallback_when_denominator_is_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(t=[float(step)], x=[1.0]) for step in range(4)]
    trajectory_ext = [
        _make_state(t=[0.0], x=[0.0], bx=[1.0], char_time=[0.05]),
        _make_state(t=[1.0], x=[0.0], bx=[1.0], char_time=[0.05]),
        _make_state(t=[2.0], x=[0.0], bx=[1.0], char_time=[0.05]),
        _make_state(t=[3.0], x=[0.0], bx=[1.0], char_time=[0.05]),
    ]

    def _unexpected_call(**_: object) -> float:
        raise AssertionError("_compute_delta_t should not be used in the singular path")

    monkeypatch.setattr(distances, "_compute_delta_t", _unexpected_call)

    result = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=3,
        index_part=0,
    )

    assert result.tolist() == [3]
