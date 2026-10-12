from __future__ import annotations

import numpy as np
import pytest

import core.distances as distances
from core.constants import NUMERICAL_EPSILON
from core.types import ChronoMatchingMode, TrajectoryBuilder


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


def _make_soa(trajectory: list[dict[str, np.ndarray]]):
    builder = TrajectoryBuilder(len(trajectory), len(trajectory[0]["x"]))
    for step, state in enumerate(trajectory):
        builder.set_step(step, state)
    return builder.build()


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
    assert distances._locate_retarded_index(trajectory_ext, 3, 0, 10.0) == 3


def test_locate_retarded_index_soa_matches_strict_chrono_bracket() -> None:
    t_col = np.array([0.0, 1.0, 2.0, 3.0])

    assert distances._locate_retarded_index_soa(t_col, 3, -1.0) == 3
    assert distances._locate_retarded_index_soa(t_col, 3, 1.5) == 2
    assert distances._locate_retarded_index_soa(t_col, 3, 2.0) == 3
    assert distances._locate_retarded_index_soa(t_col, 3, 10.0) == 3
    for times in (t_col, np.array([-2.0, -1.0, -1.0, 0.0, 1.0])):
        for side in ("left", "right"):
            for target in (-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 10.0, np.nan):
                assert distances._locate_retarded_index_soa(
                    times, len(times) - 1, target, side=side, monotone=True
                ) == distances._locate_retarded_index_soa(
                    times, len(times) - 1, target, side=side
                )

    class CountedTimes(np.ndarray):
        cells = 0

        def __getitem__(self, index):
            value = super().__getitem__(index)
            if isinstance(index, tuple):
                type(self).cells += np.size(value)
            return value

    times = np.broadcast_to(np.arange(4096)[:, None], (4096, 5)).view(CountedTimes)
    targets = np.array([-1.0, 0.0, 2.0, 4094.0, 5000.0])
    np.testing.assert_array_equal(
        distances._batched_time_upper_bound(times, targets), [0, 1, 3, 4095, 4096]
    )
    assert CountedTimes.cells <= 5 * 13


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


def test_compute_retarded_distance_soa_matches_legacy_path() -> None:
    trajectory = [
        _make_state(x=[0.0, 1.0, -2.0], y=[0.0, 0.5, 1.0], z=[0.0, 0.0, 0.2]),
        _make_state(x=[1.0, 2.0, -1.0], y=[0.0, 0.5, 1.0], z=[0.0, 0.0, 0.2]),
        _make_state(x=[2.0, 3.0, 0.0], y=[0.0, 0.5, 1.0], z=[0.0, 0.0, 0.2]),
    ]
    trajectory_ext = [
        _make_state(x=[2.0, 5.0, 0.0], y=[0.0, 0.0, 1.2], z=[0.0, 1.0, 0.2]),
        _make_state(x=[1.0, 4.0, -1.0], y=[0.0, 0.5, 2.0], z=[0.0, 1.5, 0.2]),
        _make_state(x=[4.0, 3.0, 0.0], y=[1.0, 0.5, 1.0], z=[2.0, 0.0, 0.2]),
    ]
    indices = np.array([1, 0, 2])

    legacy = distances.compute_retarded_distance(
        trajectory,
        trajectory_ext,
        index_traj=2,
        index_part=2,
        indices_ret=indices,
    )
    soa = distances.compute_retarded_distance_soa(
        _make_soa(trajectory),
        _make_soa(trajectory_ext),
        index_traj=2,
        index_part=2,
        indices_ret=indices,
    )

    for key in ("R", "nx", "ny", "nz"):
        np.testing.assert_allclose(soa[key], legacy[key])


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


def test_chrono_match_indices_handles_pruned_source_history_shorter_than_observer_history() -> (
    None
):
    trajectory = [_make_state(t=[0.0], x=[500.0]) for _ in range(3)]
    trajectory[1]["t"] = np.array([0.5], dtype=float)
    trajectory[2]["t"] = np.array([1.0], dtype=float)
    trajectory_ext = [_make_state(t=[1.0], x=[0.0], bx=[0.0])]

    result = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=2,
        index_part=0,
        mode=ChronoMatchingMode.FAST,
    )

    assert result.tolist() == [0]


def test_chrono_match_indices_soa_fast_matches_legacy_path() -> None:
    trajectory = [
        _make_state(t=[float(step), float(step)], x=[500.0, 700.0]) for step in range(6)
    ]
    trajectory_ext = [
        _make_state(
            t=[float(step), float(step)],
            x=[0.0 + 10.0 * step, 100.0 + 5.0 * step],
            bx=[0.0, 0.2],
        )
        for step in range(6)
    ]

    legacy = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=5,
        index_part=0,
        mode=ChronoMatchingMode.FAST,
    )
    soa = distances.chrono_match_indices_soa(
        _make_soa(trajectory),
        _make_soa(trajectory_ext),
        index_traj=5,
        index_part=0,
        mode=ChronoMatchingMode.FAST,
    )

    np.testing.assert_array_equal(soa, legacy)


@pytest.mark.parametrize("cell_limit", [0, 1_000_000])
@pytest.mark.parametrize("time_mode", ["ordinary", "negative", "ties", "rewrite"])
def test_chrono_match_indices_soa_fast_matches_legacy_with_varied_time_columns(
    monkeypatch, cell_limit, time_mode
) -> None:
    n_particles = 5
    trajectory = []
    trajectory_ext = []
    for step in range(8):
        trajectory.append(
            _make_state(
                t=[float(step) + 0.02 * j for j in range(n_particles)],
                x=[420.0 + 13.0 * j for j in range(n_particles)],
                y=[0.5 * j for j in range(n_particles)],
                z=[-0.25 * j for j in range(n_particles)],
                bx=[0.01 * j for j in range(n_particles)],
                by=[0.0] * n_particles,
                bz=[0.0] * n_particles,
            )
        )
        trajectory_ext.append(
            _make_state(
                t=[float(step) + 0.07 * j for j in range(n_particles)],
                x=[10.0 + 9.0 * step + 23.0 * j for j in range(n_particles)],
                y=[0.3 * j for j in range(n_particles)],
                z=[-0.1 * j for j in range(n_particles)],
                bx=[0.0, 0.05, -0.1, 0.2, 1.0],
                by=[0.0] * n_particles,
                bz=[0.0] * n_particles,
                char_time=[0.05, 0.04, 0.03, 0.02, 0.01],
            )
        )

    if time_mode == "ordinary" and cell_limit == 1_000_000:
        legacy = distances.chrono_match_indices(
            trajectory,
            trajectory_ext,
            index_traj=7,
            index_part=4,
            mode=ChronoMatchingMode.FAST,
        )
        soa = distances.chrono_match_indices_soa(
            _make_soa(trajectory),
            _make_soa(trajectory_ext),
            index_traj=7,
            index_part=4,
            mode=ChronoMatchingMode.FAST,
        )
        np.testing.assert_array_equal(soa, legacy)
    monkeypatch.setattr(distances, "_MAX_VECTORIZED_CHRONO_MATCH_CELLS", cell_limit)
    distances._MONOTONE_TIME_CACHE.clear()
    builder = TrajectoryBuilder(8, n_particles)
    observer = _make_soa(trajectory)
    for step, state in enumerate(trajectory_ext):
        if time_mode == "negative":
            state["t"] -= 4.0
        elif time_mode == "ties":
            state["t"][:] = float(step // 2)
        builder.set_step(step, state)
        source = builder.build_partial(step + 1)
        args = dict(index_traj=step, index_part=4, mode=ChronoMatchingMode.FAST)
        cached = distances.chrono_match_indices_soa(observer, source, **args)
        with monkeypatch.context() as patch:
            patch.setattr(
                distances,
                "_monotone_time_columns",
                lambda history: np.zeros(n_particles, dtype=bool),
            )
            scanned = distances.chrono_match_indices_soa(observer, source, **args)
        np.testing.assert_array_equal(cached, scanned)
    stats = distances._MONOTONE_TIME_CACHE.stats()
    assert stats.misses == 1
    assert stats.appends == 7
    if time_mode == "rewrite":
        changed = dict(trajectory_ext[3])
        changed["t"] = np.full(n_particles, -100.0)
        builder.set_step(3, changed)
        assert not np.any(distances._monotone_time_columns(builder.build_partial(8)))
        assert distances._MONOTONE_TIME_CACHE.stats().rebuilds == 1


def test_chrono_match_indices_soa_averaged_matches_legacy_path() -> None:
    trajectory = [
        _make_state(t=[float(step), float(step)], x=[500.0, 700.0]) for step in range(6)
    ]
    trajectory_ext = [
        _make_state(
            t=[float(step), float(step)],
            x=[0.0 + 10.0 * step, 100.0 + 5.0 * step],
            bx=[0.0, 0.2],
        )
        for step in range(6)
    ]

    legacy = distances.chrono_match_indices(
        trajectory,
        trajectory_ext,
        index_traj=5,
        index_part=0,
        mode=ChronoMatchingMode.AVERAGED,
        interpolate=True,
        tolerance=0.1,
    )
    soa = distances.chrono_match_indices_soa(
        _make_soa(trajectory),
        _make_soa(trajectory_ext),
        index_traj=5,
        index_part=0,
        mode=ChronoMatchingMode.AVERAGED,
        interpolate=True,
        tolerance=0.1,
    )

    np.testing.assert_array_equal(soa.indices, legacy.indices)
    np.testing.assert_array_equal(soa.indices_next, legacy.indices_next)
    np.testing.assert_allclose(soa.weights, legacy.weights)
    np.testing.assert_allclose(soa.residuals, legacy.residuals)
    np.testing.assert_array_equal(soa.needs_interpolation, legacy.needs_interpolation)


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
