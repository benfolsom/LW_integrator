"""Unit tests for correctness of the three perf-optimized core functions.

Coverage:
  - compute_instantaneous_distance  (vectorized NumPy replacement)
  - _locate_retarded_index           (searchsorted replacement)
  - _compute_forces_numba_kernel     (parallel=True Numba kernel)
  - _worker_run_combo                (pickle-ability for multiprocessing)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import core.distances as distances
from core.constants import NUMERICAL_EPSILON
from core.vectorized_interactions import NUMBA_AVAILABLE, _compute_forces_numba_kernel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# compute_instantaneous_distance
# ---------------------------------------------------------------------------

class TestComputeInstantaneousDistance:
    def test_vectorized_distance_matches_scalar_loop_for_large_array(self) -> None:
        rng = np.random.default_rng(123)
        n = 50
        vector = _make_state(x=[1.0], y=[-2.0], z=[3.0])
        vector_ext = {
            "x": rng.uniform(-5.0, 5.0, n),
            "y": rng.uniform(-5.0, 5.0, n),
            "z": rng.uniform(-5.0, 5.0, n),
        }

        result = distances.compute_instantaneous_distance(vector, vector_ext, 0)

        # Manual reference using numpy ops per element
        expected_R = np.empty(n)
        expected_nx = np.empty(n)
        expected_ny = np.empty(n)
        expected_nz = np.empty(n)
        for j in range(n):
            dx = vector["x"][0] - vector_ext["x"][j]
            dy = vector["y"][0] - vector_ext["y"][j]
            dz = vector["z"][0] - vector_ext["z"][j]
            d = (dx**2 + dy**2 + dz**2) ** 0.5
            if d < NUMERICAL_EPSILON:
                expected_R[j] = NUMERICAL_EPSILON
                expected_nx[j] = expected_ny[j] = expected_nz[j] = 0.0
            else:
                expected_R[j] = d
                expected_nx[j] = dx / d
                expected_ny[j] = dy / d
                expected_nz[j] = dz / d

        np.testing.assert_allclose(result["R"], expected_R, atol=1e-12)
        np.testing.assert_allclose(result["nx"], expected_nx, atol=1e-12)
        np.testing.assert_allclose(result["ny"], expected_ny, atol=1e-12)
        np.testing.assert_allclose(result["nz"], expected_nz, atol=1e-12)

    def test_vectorized_distance_epsilon_guard_on_multiple_coincident_particles(self) -> None:
        vector = _make_state(x=[2.0], y=[3.0], z=[4.0])
        # Three coincident with reference particle, two offset
        vector_ext = {
            "x": np.array([2.0, 2.0, 2.0, 5.0, 2.0]),
            "y": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
            "z": np.array([4.0, 4.0, 4.0, 4.0, 7.0]),
        }

        result = distances.compute_instantaneous_distance(vector, vector_ext, 0)

        for i in range(3):
            assert result["R"][i] == pytest.approx(NUMERICAL_EPSILON)
            assert result["nx"][i] == pytest.approx(0.0)
            assert result["ny"][i] == pytest.approx(0.0)
            assert result["nz"][i] == pytest.approx(0.0)

        # particle at (5,3,4): dx=-3, dy=0, dz=0 → R=3, nx=-1
        assert result["R"][3] == pytest.approx(3.0)
        assert result["nx"][3] == pytest.approx(-1.0)
        assert result["ny"][3] == pytest.approx(0.0)
        assert result["nz"][3] == pytest.approx(0.0)

        # particle at (2,3,7): dx=0, dy=0, dz=-3 → R=3, nz=-1
        assert result["R"][4] == pytest.approx(3.0)
        assert result["nx"][4] == pytest.approx(0.0)
        assert result["ny"][4] == pytest.approx(0.0)
        assert result["nz"][4] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# _locate_retarded_index
# ---------------------------------------------------------------------------

def _ref_locate_retarded_index(
    trajectory_ext: list, index_traj: int, sample_index: int, target_time: float
) -> int:
    if target_time <= 0.0:
        return index_traj
    for k in range(index_traj, -1, -1):
        candidate_index = index_traj - k
        if trajectory_ext[candidate_index]["t"][sample_index] >= target_time:
            return candidate_index
    return index_traj


class TestLocateRetardedIndex:
    def _build_trajectory(self, n_steps: int) -> list[dict[str, np.ndarray]]:
        return [_make_state(t=[float(k)]) for k in range(n_steps)]

    def test_searchsorted_matches_linear_scan_across_targets(self) -> None:
        n_steps = 50
        traj = self._build_trajectory(n_steps)
        index_traj = n_steps - 1

        targets = [0.0, 0.5, 1.0, 5.0, 12.3, 24.9, 25.0, 25.1, 40.0, 48.9, 49.0]
        for target in targets:
            expected = _ref_locate_retarded_index(traj, index_traj, 0, target)
            got = distances._locate_retarded_index(traj, index_traj, 0, target)
            assert got == expected, (
                f"Mismatch for target_time={target}: expected {expected}, got {got}"
            )

    def test_locate_retarded_index_returns_index_traj_for_nonpositive_target(self) -> None:
        traj = self._build_trajectory(10)
        index_traj = 9

        assert distances._locate_retarded_index(traj, index_traj, 0, 0.0) == index_traj
        assert distances._locate_retarded_index(traj, index_traj, 0, -1.0) == index_traj


# ---------------------------------------------------------------------------
# _compute_forces_numba_kernel
# ---------------------------------------------------------------------------

def _build_kernel_args(n_ext: int, seed: int = 99):
    from core.constants import C_MMNS

    rng = np.random.default_rng(seed)
    bx_e = rng.uniform(-0.05, 0.05, n_ext)
    by_e = rng.uniform(-0.05, 0.05, n_ext)
    bz_e = rng.uniform(0.7, 0.9, n_ext)
    g_e = 1.0 / np.sqrt(1.0 - (bx_e**2 + by_e**2 + bz_e**2))
    R = rng.uniform(0.5, 10.0, n_ext)
    nx = rng.uniform(-0.3, 0.3, n_ext)
    ny = rng.uniform(-0.3, 0.3, n_ext)
    nz = np.sqrt(np.maximum(1.0 - nx**2 - ny**2, 0.0))
    bdot = rng.uniform(-0.005, 0.005, n_ext)
    q_e = np.full(n_ext, -1.178734e-5)

    return (
        1e-4,           # h
        -1.178734e-5,   # charge_i
        5.485799e-4,    # mass_i
        3.0,            # gamma_i
        0.0, 0.0, 0.85, # bx_i, by_i, bz_i
        nx, ny, nz, R,
        bx_e, by_e, bz_e,
        bx_e * bdot, by_e * bdot, bz_e * bdot,
        q_e, g_e,
        C_MMNS,
    )


class TestComputeForcesNumbaKernel:
    def test_parallel_kernel_matches_sequential_for_n_ext_100(self) -> None:
        args = _build_kernel_args(100)

        if NUMBA_AVAILABLE:
            result_jit = _compute_forces_numba_kernel(*args)
            result_py = _compute_forces_numba_kernel.py_func(*args)

            assert len(result_jit) == 8
            for i, (jit_val, py_val) in enumerate(zip(result_jit, result_py)):
                assert jit_val == pytest.approx(py_val, rel=1e-6), (
                    f"Output index {i} mismatch: jit={jit_val}, py={py_val}"
                )
        else:
            result = _compute_forces_numba_kernel(*args)
            assert isinstance(result, tuple)
            assert len(result) == 8
            for val in result:
                assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _worker_run_combo picklability
# ---------------------------------------------------------------------------

class TestWorkerRunCombo:
    def test_worker_run_combo_is_picklable(self) -> None:
        from lw_integrator.sweep_runner import _worker_run_combo

        data = pickle.dumps(_worker_run_combo)
        reloaded = pickle.loads(data)
        assert callable(reloaded)
