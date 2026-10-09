"""Unreduced breathing: each parent projects the ensemble on its OWN initial
velocity normal, so longitudinal extent leaks into the 'transverse' RMS of
macros with transverse velocity.  Single ballistic bunch, no train."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breathing_probe import C_MMNS, ExactCloudHistory, breathing_width_knots, cfg, header  # noqa: E402

rng = np.random.default_rng(2)
n, bz = 32, 0.6
times = np.linspace(0, 0.05, 101)
T = np.repeat(times[:, None], n, 1)
for sig_l, vt in ((0.5, 30.0), (5.0, 30.0), (5.0, 3.0)):
    x0 = np.column_stack([rng.normal(0, 0.3, n), rng.normal(0, 0.3, n), rng.normal(0, sig_l, n)])
    v = np.column_stack([rng.normal(0, vt, n), rng.normal(0, vt, n), np.full(n, bz * C_MMNS)])
    pos = x0[None] + T[..., None] * v[None]
    B = np.broadcast_to(v / C_MMNS, pos.shape).copy()
    w, _, _ = breathing_width_knots(ExactCloudHistory(header(n), cfg(tau=1e-4), None), T, pos, B, np.zeros_like(T, bool))
    xy = pos[..., :2] - pos[..., :2].mean(1, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(xy**2, -1), 1) / 2)
    truth = 0.1 * rms[-1] / rms[0]
    print(f"sig_l={sig_l} mm, beta_t rms={vt / C_MMNS:.3f}: lab-transverse self-similar width at t=0.05 ns = {truth:.3f} mm;"
          f" realised widths min/median/max = {w[-1].min():.3f}/{np.median(w[-1]):.3f}/{w[-1].max():.3f} mm")
