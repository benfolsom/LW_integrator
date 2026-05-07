"""Benchmark script for three performance changes in LW_integrator/core/.

Run from the repository root:
    .venv/bin/python scripts/profile_perf_changes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.constants import C_MMNS, NUMERICAL_EPSILON
from core.distances import compute_instantaneous_distance, _locate_retarded_index
from core.vectorized_interactions import NUMBA_AVAILABLE, _compute_forces_numba_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**kwargs: list) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=float) for k, v in kwargs.items()}


def _hdr(title: str) -> None:
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def _time_fn(fn, n: int) -> float:
    """Return mean wall time per call in microseconds."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


# ---------------------------------------------------------------------------
# Reference implementations (old code, inlined)
# ---------------------------------------------------------------------------

def _ref_compute_instantaneous_distance(vector, vector_ext, index):
    result: dict[str, np.ndarray] = {}
    n = len(vector_ext["x"])
    R_arr = np.empty(n)
    nx_arr = np.empty(n)
    ny_arr = np.empty(n)
    nz_arr = np.empty(n)
    for j in range(n):
        dx = vector["x"][index] - vector_ext["x"][j]
        dy = vector["y"][index] - vector_ext["y"][j]
        dz = vector["z"][index] - vector_ext["z"][j]
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        if distance < NUMERICAL_EPSILON:
            R_arr[j] = NUMERICAL_EPSILON
            nx_arr[j] = 0.0
            ny_arr[j] = 0.0
            nz_arr[j] = 0.0
        else:
            R_arr[j] = distance
            nx_arr[j] = dx / distance
            ny_arr[j] = dy / distance
            nz_arr[j] = dz / distance
    result["R"] = R_arr
    result["nx"] = nx_arr
    result["ny"] = ny_arr
    result["nz"] = nz_arr
    return result


def _ref_locate_retarded_index(trajectory_ext, index_traj, sample_index, target_time):
    if target_time <= 0.0:
        return index_traj
    for k in range(index_traj, -1, -1):
        candidate_index = index_traj - k
        if trajectory_ext[candidate_index]["t"][sample_index] >= target_time:
            return candidate_index
    return index_traj


# ---------------------------------------------------------------------------
# Benchmark 1 – compute_instantaneous_distance
# ---------------------------------------------------------------------------

def bench_distance_vectorization(N: int = 500, n_particles: int = 200) -> None:
    _hdr("Benchmark 1: compute_instantaneous_distance  (n_ext=%d, N=%d)" % (n_particles, N))

    rng = np.random.default_rng(42)
    vector = _make_state(x=[0.0], y=[0.0], z=[0.0])
    vector_ext = {
        "x": rng.uniform(-10.0, 10.0, n_particles),
        "y": rng.uniform(-10.0, 10.0, n_particles),
        "z": rng.uniform(-10.0, 10.0, n_particles),
    }

    t_ref = _time_fn(lambda: _ref_compute_instantaneous_distance(vector, vector_ext, 0), N)
    t_new = _time_fn(lambda: compute_instantaneous_distance(vector, vector_ext, 0), N)

    print(f"  {'Implementation':<40s} {'Mean µs/call':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Reference (Python for-loop)':<40s} {t_ref:>12.3f}")
    print(f"  {'Vectorized NumPy (new)':<40s} {t_new:>12.3f}")
    print(f"  {'Speedup':<40s} {t_ref/t_new:>12.2f}x")


# ---------------------------------------------------------------------------
# Benchmark 2 – _locate_retarded_index
# ---------------------------------------------------------------------------

def bench_locate_retarded_index(N: int = 2000, n_steps: int = 1000) -> None:
    _hdr("Benchmark 2: _locate_retarded_index  (n_steps=%d, N=%d)" % (n_steps, N))

    trajectory_ext = [_make_state(t=[float(k)]) for k in range(n_steps)]
    index_traj = n_steps - 1
    target_time = float(n_steps) / 2.0

    t_ref = _time_fn(
        lambda: _ref_locate_retarded_index(trajectory_ext, index_traj, 0, target_time), N
    )
    t_new = _time_fn(
        lambda: _locate_retarded_index(trajectory_ext, index_traj, 0, target_time), N
    )

    print(f"  {'Implementation':<40s} {'Mean µs/call':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Reference (reverse linear scan)':<40s} {t_ref:>12.3f}")
    print(f"  {'Linear scan (current — SOA needed for searchsorted)':<40s} {t_new:>12.3f}")
    print(f"  {'Speedup':<40s} {t_ref/t_new:>12.2f}x")


# ---------------------------------------------------------------------------
# Benchmark 3 – _compute_forces_numba_kernel
# ---------------------------------------------------------------------------

def bench_numba_kernel(N: int = 200, n_ext: int = 500) -> None:
    _hdr("Benchmark 3: _compute_forces_numba_kernel  (n_ext=%d, N=%d)" % (n_ext, N))

    if not NUMBA_AVAILABLE:
        print("  [SKIP] Numba not available.")
        return

    rng = np.random.default_rng(7)
    c = C_MMNS

    def _make_arrays(n):
        bx = rng.uniform(-0.1, 0.1, n)
        by = rng.uniform(-0.1, 0.1, n)
        bz = rng.uniform(0.8, 0.95, n)
        gamma = 1.0 / np.sqrt(1.0 - (bx**2 + by**2 + bz**2))
        R = rng.uniform(0.1, 5.0, n)
        nx = rng.uniform(-0.5, 0.5, n)
        ny = rng.uniform(-0.5, 0.5, n)
        nz = np.sqrt(np.maximum(1.0 - nx**2 - ny**2, 0.0))
        bdot = rng.uniform(-0.01, 0.01, n)
        charge = np.ones(n) * (-1.178734e-5)
        return bx, by, bz, gamma, R, nx, ny, nz, bdot, charge

    bx_e, by_e, bz_e, g_e, R, nx, ny, nz, bdot_s, q_e = _make_arrays(n_ext)
    bdotx = bx_e * bdot_s
    bdoty = by_e * bdot_s
    bdotz = bz_e * bdot_s

    h = 1e-4
    charge_i = -1.178734e-5
    mass_i = 5.485799e-4
    gamma_i = 3.0
    bx_i, by_i, bz_i = 0.0, 0.0, 0.94

    args = (h, charge_i, mass_i, gamma_i, bx_i, by_i, bz_i,
            nx, ny, nz, R, bx_e, by_e, bz_e, bdotx, bdoty, bdotz, q_e, g_e, c)

    # JIT warm-up with tiny arrays
    bx2, by2, bz2, g2, R2, nx2, ny2, nz2, bd2, q2 = _make_arrays(2)
    _compute_forces_numba_kernel(
        h, charge_i, mass_i, gamma_i, bx_i, by_i, bz_i,
        nx2, ny2, nz2, R2, bx2, by2, bz2,
        bx2 * bd2, by2 * bd2, bz2 * bd2, q2, g2, c,
    )

    t_py = _time_fn(lambda: _compute_forces_numba_kernel.py_func(*args), N)
    t_jit = _time_fn(lambda: _compute_forces_numba_kernel(*args), N)

    print(f"  {'Implementation':<40s} {'Mean µs/call':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'Python fallback (.py_func)':<40s} {t_py:>12.3f}")
    print(f"  {'JIT-compiled Numba (new)':<40s} {t_jit:>12.3f}")
    print(f"  {'JIT speedup':<40s} {t_py/t_jit:>12.2f}x")


# ---------------------------------------------------------------------------
# Benchmark 4 – Sweep parallelism
# ---------------------------------------------------------------------------

def bench_sweep_parallelism() -> None:
    _hdr("Benchmark 4: SweepRunner parallel workers")
    print("  [SKIP] Requires a real sweep config — not suitable for quick profiling.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench_distance_vectorization()
    bench_locate_retarded_index()
    bench_numba_kernel()
    bench_sweep_parallelism()
    print()
