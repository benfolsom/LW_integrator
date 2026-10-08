# Review: `review/correction-long-bunch` vs `development`

Commit reviewed: `972068b Add opt-in per-axis correction lattice for elongated bunches`
(`core/pic/correction.py` +56/−4, `tests/test_pic_correction.py` +99/−1).
Reviewer scratch scripts are summarised under Evidence. None are committed.

## Summary

The lattice geometry is correct. It is a lab-frame snapshot rotated into the
observer bunch's mean-momentum basis, recentred and refitted at every sample,
and every observer is covered. Defaults are bit-identical, and the targeted
tests pass. The correction kernel matches an independent Liénard–Wiechert
calculation to ≤2e-6 relative error for both transverse and longitudinal
acceleration. A units or cancellation bug therefore does **not** explain the
small ratios.

However, the reported ΔE/PIC ratios (1.5e-5 for protons, 5.9e-7 for electrons)
are **not yet evidence about the physics**. Near each γ≈10.8 electron cloud,
ΔE is a pancake about b/γ wide along z (0.02–0.2 mm for b = 0.3–1 mm). A
lattice spanning a long bunch with a handful of z-nodes cannot resolve it.
Depending on where the nodes fall, the interpolated RMS ΔE is wrong by a
factor of 0.03 to 6. In addition, the mandatory one-cell padding means the
outer node layers always get zero interpolation weight. That wastes 78–88% of
the exact-LW node evaluations for typical shapes.

**Recommendation:** OK to merge as an opt-in feature once the padding minimum is
relaxed (P2-1), or merge now with that as an immediate follow-up. Do not cite
the crossing ratios until the z-resolution or longitudinal-smoothing
convergence check under P1-1 has been done.

## Checks performed

| Check | Result |
|---|---|
| `pytest -p no:anyio tests/test_pic_correction.py tests/test_native_pic.py` | 17 passed, 1 deselected (slow), 92 s |
| Defaults bit-identical | `example_native_pic_correction.json` (4 steps, refit 1, lattice 4×4×5) run on `development` and on this branch: output JSON is byte-identical (`cmp`). By inspection, `fixed` mode computes `local` with the same expression and leaves `half` unchanged, and `prehistory_duration_s=None` keeps `-1/C`. |
| Independent LW oracle (scratch `lw_oracle.py`) | One γ=10.785 electron cloud (1 subcharge), inertial prehistory, then du_x/dt = 1e8 s⁻¹ for 100 ps. I wrote my own retarded-time root and LW formula and compared them with `CloudCorrection.direct`. Relative error 1.4e-6, 6e-8, 1e-12 and 2e-8 at four observer points. With longitudinal acceleration instead: ≤1.6e-6. |
| black / mypy (changed file) | clean / no errors in `core/pic/correction.py` |
| ruff | +8 `C408` (`dict()` calls), matching the file's existing idiom; nothing new otherwise |

## Findings

### P0: none

### P1-1: ΔE from relativistic clouds is unresolved along z on a bunch-extent lattice, so the reported ratios are not converged

*Where:* `core/pic/correction.py:737-750`, where the lattice spans the whole
bunch with `lattice_shape[2]` nodes. The cloud model has no longitudinal width
(`correction.py:321-328`, `longitudinal_sigma_mm=0`, and the docstring says "no
longitudinal smoothing").

*Failure scenario:* The electron bunch is elongated (several cm to m) with, say,
17 z-nodes, so Δz is mm to cm. The ΔE of each γ=10.8 disc cloud is
concentrated within about b/γ of its plane: 0.03–0.1 mm for b = 0.3–1 mm. The
correction particles receive is then trilinear interpolation between nodes
that either straddle a pancake (over-estimate) or miss it (under-estimate by
orders of magnitude). This applies to electron observers (their own clouds)
and to proton observers (the electron clouds crossing them).

*Evidence (scratch `z_profile.py`, `z_profile_shift.py`):* the setup is one
accelerated γ=10.785 cloud with observers on a 6 mm z-line.

| b | FWHM of \|ΔE\| along z | RMS(interp)/RMS(true), node on pancake (n_z = 5 / 17 / 65) | same, nodes shifted by 0.37 Δz |
|---|---|---|---|
| 0.3 mm | 0.020 mm | 6.17 / 3.08 / 1.56 | 0.034 / 1.02 / 0.67 |
| 1.0 mm | 0.210 mm | 3.03 / 1.49 / 0.88 | 0.068 / 1.08 / 0.85 |

Even with 0.1 mm spacing (n_z = 65 over 6 mm), the error is still 12–56%. A
metre-scale lattice cannot reach the b/γ ≈ 0.03–0.1 mm spacing needed.

*Suggested action:* this is not a code bug in the branch, but it is the main
risk for the intended use of the feature. Before drawing physics conclusions,
pick one of these:

- (a) Give clouds a longitudinal width (an exact-cloud longitudinal quadrature)
  so that ΔE is smooth on the lattice scale.
- (b) Increase K until the cloud z-spacing is ≲ b/γ_e. This is probably
  unaffordable.
- (c) At minimum, run a z-node convergence check (e.g. n_z = 17 → 33 → 65 plus
  a half-cell lattice phase shift) on the actual crossing. If the peak RMS
  ratio changes by O(1), it is aliasing.

### P2-1: the mandatory ≥1-cell padding makes the outer node layers dead weight

*Where:* `correction.py:104-111` (`lattice_padding_cells >= 1`) and `:737-741`.

*Failure scenario:* With padding p, the maximum observer coordinate is
`(n-1)(1+interior)/2 = n-1-p`, and the minimum is ≥ p (the extent is symmetric
about the centroid). For CIC, layer 0 is therefore never used. Layer n-1 is
used only with weight exactly 0, at one boundary point. Still, every node gets
an exact retarded-root evaluation. Since the lattice is built from the same
positions it is gathered at, any p > 0 that absorbs rounding (e.g. 1e-6) is
enough.

*Evidence:* For a 500-observer bunch (σ = 1, 1, 30 mm) on a 4×4×17
`bunch_extent` lattice, coordinates ranged over [1, 1.05, 2.66] to
[1.94, 2, 15]. I filled every outer-layer node with 1e300 and gathered: the
maximum gathered value was 1.0 (exact weight 0). In that case only 60 of 272
nodes can contribute (78% wasted). For a 4³ lattice, 56 of 64 (87.5%) are
wasted, and the whole bunch is interpolated from a single trilinear cell.
Relaxing the padding would cut the reported 27.8 s per step by up to about 4–8×
at fixed resolution, or buy 1.5–3× more useful cells per axis at the same
cost.

*Fix:* allow `0 < lattice_padding_cells < 1` (with a tiny default), or
equivalently clamp the coordinates. Adapt the existing validation case
`lattice_padding_cells=0.5` accordingly.

### P3-1: the extent is symmetric about the population-weighted centroid

*Where:* `correction.py:741` (`np.max(np.abs(local), axis=0)`).

For skewed or tailed bunches, which are typical after a crossing or with a
chirp, and with zero-weight stragglers (which are deliberately included), the
far side sets both half-widths. That can waste up to half of each axis.
Centring the lattice on the local bounding-box midpoint (store an offset in
`origin_m`) keeps the basis and costs nothing.

### P3-2: the default `half_extent_m` silently acts as a 4 mm lower bound in `bunch_extent` mode

*Where:* `correction.py:54` and `:734`, `:741`.

For a sub-mm transverse bunch, a user who forgets to lower `half_extent_mm`
gets a transverse lattice ≥8 mm wide, with no error. The docstring states the
lower-bound rule, but the config surface does not. Consider requiring an
explicit `half_extent_mm` with `bunch_extent`, or using a negligible default
lower bound in that mode.

### P3-3: `prehistory_duration_ns` is bundled in without mention in the commit message

*Where:* `correction.py:66-73`, `:148-156`, `:334-338`.

The change itself is sound. It is a straight inertial prefix, and an
insufficient duration fails loudly with `RetardedHistoryError`, which the
adapted test checks. Two small points:

- `from_config` turns `null` into a `TypeError` rather than a `ValueError`, and
  accepts booleans.
- Please document the sizing rule. For observers a distance Δz *ahead* of an
  ultra-relativistic source, the retarded lag is ≈ 2γ²Δz/c. For 5 MeV electrons
  that is ≈ 233 Δz/c, so 0.78 µs at Δz = 1 m (the test uses 2000 ns).

### P3-4: tests (owner's rule)

No new test functions were added; the existing tests were extended or adapted.
That is acceptable under the rule. Notes:

- The `bunch_rms_k` guard in `test_uniform_cloud_cancellation_...` exercises
  pre-existing behaviour unrelated to this change. It is harmless but out of
  scope.
- The affine-field oracle proves that geometry, basis round-trip and gather are
  exact. By construction it cannot detect the resolution issue in P1-1. That
  is fine, but no test should be read as covering it.
- The `monkeypatch` of `_direct` is appropriate.

### P3-5 (informational, pre-existing): centroid clouds cannot see symmetric focusing

`_cohorts` bisects along the largest extent, which is z for elongated bunches,
so modest K produces z-slices. Cloud widths are also immutable. Radial focusing
by the other bunch therefore moves no cloud centroid, and the transverse ΔE
channel sees only coherent dipole motion (offsets, asymmetry) plus finite-
sampling noise. For long axisymmetric bunches, the physical ΔE from purely
radial motion is itself suppressed: in the z-invariant limit it vanishes by
Gauss's law. So this is probably a modest under-count, but I could not quantify
it. My per-particle comparison was dominated by graininess, so I do not cite it
as evidence. A K-convergence point (K = 8 → 16 → 32) on the crossing would
settle it.

## Frame and geometry details checked (no defect)

- `rest_basis` is orthonormal. Nodes are mapped by `nodes @ basis.T + center`,
  and observers by `(x - center) @ basis`, which are mutually inverse; the
  adapted test asserts the round trip.
- Extents are lab-frame (no γ stretch). This is correct because ΔE is a
  lab-frame field evaluated at lab-simultaneous nodes at the cloud time and
  gathered at lab positions. The γ-stretched PIC rest grid is independent of
  it.
- The moving centre is recomputed at every `sample`. With `midpoint_predictor`,
  the predicted copy recomputes the lattice at the half-step observer positions
  and the predicted cloud time, which is consistent. `bunch_extent` correctly
  forbids `evaluation_every > 1`: temporal extrapolation would mix node values
  from lattices of different geometry.
- Strongly anisotropic spacing is not a problem for trilinear interpolation in
  itself. The error is separable, Σ_i h_i²/8 ∂_i²ΔE. The issue is only that h_z
  must resolve the b/γ structure in P1-1.

## Expected magnitude of the correction

For an observer at near-zone distance b, roughly transverse to a source of
Lorentz factor γ_s, the oracle runs calibrate:

  |ΔE| / |E_boosted-Coulomb| ≈ c₁ γ_s² |dβ_s/dt| b / c,

with c₁ ≈ 1 for transverse acceleration (measured 3.58e-3 against
γ²ab/c² = 3.60e-3) and c₁ ≈ 0.5 for longitudinal acceleration. Here
dβ_⊥/dt = qF_⊥/(γmc) and dβ_∥/dt = qF_∥/(γ³mc). For this crossing (γ_e =
10.785, γ_p = 1.0213, F_⊥ = E(1+β_eβ_p) = 1.20 E for the other bunch's field),
per unit F·b of 1 kV/m·mm:

| Source cloud driven by | Ratio per (kV/m·mm) |
|---|---|
| electron, transverse, by the proton field | 2.5e-5 |
| electron, transverse, by its own space charge (1/γ² suppressed) | 1.8e-7 |
| electron, longitudinal | 8.7e-8 |
| proton, transverse, by the electron field | 1.3e-9 |
| proton, longitudinal | 5.0e-10 |

Observers *ahead* of an electron cloud see a much larger fractional ΔE,
≈ 2Δγ_e/γ_e. The field there has not yet "seen" the energy change. That field
is weak, however, and needs the 2γ²Δz/c prehistory.

**Estimate.** The proton-observer ratio is dominated by coherent transverse
deflection of electron-cloud centroids. Then 1.5e-5 corresponds to a coherent
F_⊥·b ≈ 0.6 kV/m·mm. That is plausible for a proton bunch of order tens of pC
over a ~0.1 m length with an offset or asymmetric crossing, and the ratio
scales linearly with bunch charge. For a centred, symmetric crossing the
coherent part is much smaller, and the expected ratio drops towards the
1e-7–1e-6 level of the longitudinal and self channels. Electron observers
should be lower than proton observers, because proton clouds are about 2e4×
harder to deflect per unit field, so 5.9e-7 is plausible in order of
magnitude. **Nothing in the numbers indicates a units or cancellation bug.**
Because of P1-1, however, the measured values could be off by more than 10×
in either direction.

**Direct self-check you can run on the existing output:** predict, per step,
`max_j γ_j² |beta_dot_s[j]| b / c`, with `beta_dot_s` stored on
`CloudCorrection`, γ_j from `history_u`, and b the observer-to-cloud
transverse distance. Compare it with the measured RMS ratio. If the measured
value is far below the prediction while the clouds are inside the observer
bunch, that points to z-aliasing (P1-1) rather than physics.
