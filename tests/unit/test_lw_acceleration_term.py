"""Sampled-path force kernels against finite differences of the LW potential.

The canonical update is ``dP^alpha = h (q_o / c) V_o_beta d^alpha A^beta`` with
``A^beta = q_s V^beta / (V . R)`` at the retarded event.  The reference
differentiates that potential numerically for a uniformly accelerated source,
so it checks the source-acceleration term independently of its closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

import core.vectorized_interactions as vi
from core.constants import C_MMNS as C

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])


def _source(t: float, v0: np.ndarray, a0: np.ndarray):
    return v0 * t + 0.5 * a0 * t * t, v0 + a0 * t


def _retarded_time(event: np.ndarray, v0, a0, span: float) -> float:
    lo, hi = event[0] - span, event[0]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        position, _ = _source(mid, v0, a0)
        if C * (event[0] - mid) > np.linalg.norm(event[1:] - position):
            lo = mid
        else:
            hi = mid
    root = 0.5 * (lo + hi)
    assert root - (event[0] - span) > 1.0e-6 * span, "retarded root at bracket edge"
    return root


def _four_potential(event, charge, v0, a0, span):
    t_ret = _retarded_time(event, v0, a0, span)
    position, velocity = _source(t_ret, v0, a0)
    beta = velocity / C
    gamma = 1.0 / np.sqrt(1.0 - beta @ beta)
    four_velocity = gamma * C * np.array([1.0, *beta])
    separation = np.array([C * (event[0] - t_ret), *(event[1:] - position)])
    return charge * four_velocity / (four_velocity @ METRIC @ separation)


def _reference_update(h, q_o, beta_o, q_s, event, v0, a0, span, step):
    """``h (q_o/c) V_o_beta d^alpha A^beta`` by central differences."""

    gradient = np.zeros((4, 4))  # [nu, beta] = d_nu A^beta, x^0 = c t
    for nu in range(4):
        offset = np.zeros(4)
        offset[nu] = step / C if nu == 0 else step
        upper = _four_potential(event + offset, q_s, v0, a0, span)
        lower = _four_potential(event - offset, q_s, v0, a0, span)
        gradient[nu] = (upper - lower) / (2.0 * step)
    raised = METRIC @ gradient  # d^alpha A^beta
    gamma_o = 1.0 / np.sqrt(1.0 - beta_o @ beta_o)
    observer_lower = METRIC @ (gamma_o * C * np.array([1.0, *beta_o]))
    return h * q_o / C * (raised @ observer_lower)  # (t, x, y, z)


def _kernel_inputs(event, v0, a0, span):
    t_ret = _retarded_time(event, v0, a0, span)
    position, velocity = _source(t_ret, v0, a0)
    separation = event[1:] - position
    distance = float(np.linalg.norm(separation))
    beta = velocity / C
    # Kernel-level d(beta)/dt [1/ns]; compute_vectorized_contributions takes
    # the stored d(beta)/d(ct) and converts.
    return separation / distance, distance, beta, a0 / C


CASES = {
    "transverse-acceleration": dict(
        v0=np.array([0.0, 0.0, 0.5]) * C,
        a0=np.array([2.0e-2, 0.0, 0.0]) * C,
        event=np.array([30.0, 40.0, 25.0, 300.0]),
        span=40.0,
        step=1.0e-3,
    ),
    "longitudinal-acceleration": dict(
        v0=np.array([0.1, 0.0, 0.6]) * C,
        a0=np.array([0.0, 0.0, -1.0e-2]) * C,
        event=np.array([20.0, -80.0, 60.0, 500.0]),
        span=40.0,
        step=1.0e-3,
    ),
    "small-k-relativistic": dict(
        v0=np.array([0.0, 0.03, 0.9990]) * C,
        a0=np.array([5.0e-5, 0.0, 0.0]) * C,
        event=np.array([4.0, 0.3, 40.0, 1200.0]),
        span=40.0,
        step=1.0e-5,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("backend", ["numpy", "numba_serial", "numba_parallel"])
def test_force_kernel_matches_lw_potential_derivative(case, backend, monkeypatch):
    spec = CASES[case]
    h, q_o, q_s = 1.0e-3, 1.0, 1.0
    beta_o = np.array([0.0, 0.0, -0.3])
    gamma_o = float(1.0 / np.sqrt(1.0 - beta_o @ beta_o))
    nhat, distance, beta_s, beta_dot = _kernel_inputs(
        spec["event"], spec["v0"], spec["a0"], spec["span"]
    )
    reference = _reference_update(
        h,
        q_o,
        beta_o,
        q_s,
        spec["event"],
        spec["v0"],
        spec["a0"],
        spec["span"],
        spec["step"],
    )
    gamma_s = float(1.0 / np.sqrt(1.0 - beta_s @ beta_s))

    def array(value: float) -> np.ndarray:
        return np.array([value], dtype=float)

    if backend == "numpy":
        monkeypatch.setattr(vi, "NUMBA_AVAILABLE", False)
        samples = vi.ExternalSampleBatch(
            charge=array(q_s),
            gamma=array(gamma_s),
            bx=array(beta_s[0]),
            by=array(beta_s[1]),
            bz=array(beta_s[2]),
            # Stored history convention: d(beta)/d(ct).
            bdotx=array(beta_dot[0] / C),
            bdoty=array(beta_dot[1] / C),
            bdotz=array(beta_dot[2] / C),
            valid_mask=np.array([True]),
        )
        result = vi.compute_vectorized_contributions(
            h,
            q_o,
            1.0,
            gamma_o,
            tuple(beta_o),
            array(nhat[0]),
            array(nhat[1]),
            array(nhat[2]),
            array(distance),
            samples,
            apply_external=True,
        )
    else:
        if not vi.NUMBA_AVAILABLE:
            pytest.skip("numba unavailable")
        kernel = (
            vi._compute_forces_numba_serial_kernel
            if backend == "numba_serial"
            else vi._compute_forces_numba_kernel
        )
        result = kernel(
            h,
            q_o,
            1.0,
            gamma_o,
            *beta_o,
            array(nhat[0]),
            array(nhat[1]),
            array(nhat[2]),
            array(distance),
            array(beta_s[0]),
            array(beta_s[1]),
            array(beta_s[2]),
            array(beta_dot[0]),
            array(beta_dot[1]),
            array(beta_dot[2]),
            array(q_s),
            array(gamma_s),
            C,
        )
    update = np.array([result[3], result[0], result[1], result[2]])
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(update - reference)) <= 2.0e-6 * scale


def test_source_acceleration_preserves_observer_mass_shell():
    """``V_o . dp_mech = 0`` once the canonical potential change is removed."""

    spec = CASES["longitudinal-acceleration"]
    beta_o = np.array([0.0, 0.0, -0.3])
    gamma_o = float(1.0 / np.sqrt(1.0 - beta_o @ beta_o))
    four_velocity_o = gamma_o * C * np.array([1.0, *beta_o])
    nhat, distance, beta_s, beta_dot = _kernel_inputs(
        spec["event"], spec["v0"], spec["a0"], spec["span"]
    )
    gamma_s = float(1.0 / np.sqrt(1.0 - beta_s @ beta_s))

    def array(value: float) -> np.ndarray:
        return np.array([value], dtype=float)

    result = vi._compute_forces_numba_kernel.py_func(
        1.0,
        1.0,
        1.0,
        gamma_o,
        *beta_o,
        array(nhat[0]),
        array(nhat[1]),
        array(nhat[2]),
        array(distance),
        array(beta_s[0]),
        array(beta_s[1]),
        array(beta_s[2]),
        array(beta_dot[0]),
        array(beta_dot[1]),
        array(beta_dot[2]),
        array(1.0),
        array(gamma_s),
        C,
    )
    canonical = np.array([result[3], result[0], result[1], result[2]])
    # d A^alpha / d tau along the observer, by central differences.
    step = 1.0e-4
    upper = _four_potential(
        spec["event"] + step * four_velocity_o / np.array([C, 1, 1, 1]),
        1.0,
        spec["v0"],
        spec["a0"],
        spec["span"],
    )
    lower = _four_potential(
        spec["event"] - step * four_velocity_o / np.array([C, 1, 1, 1]),
        1.0,
        spec["v0"],
        spec["a0"],
        spec["span"],
    )
    mechanical = canonical - (upper - lower) / (2.0 * step) / C
    drift = four_velocity_o @ METRIC @ mechanical
    assert abs(drift) <= 1.0e-6 * np.linalg.norm(four_velocity_o) * np.max(
        np.abs(mechanical)
    )
