# Follow-up review: `review/correction-long-bunch` vs `development`

Commit reviewed: `5aa6895 Address cloud review (padding, z-resolution option,
self-check); opt-in breathing K-clouds`, on top of `972068b`. The earlier notes
are on `review-notes/correction-long-bunch`. Line numbers refer to `5aa6895`.
Scratch scripts are summarised under each finding. None are committed.

## Summary

- **Defaults are bit-identical** (P0: none).
- **The lattice follow-ups are correct:** fractional padding, a bounding-box
  midpoint, a 1 pm implicit floor, opt-in z densification and a z phase shift.
- **The z-aliasing risk (old P1-1) is only partly addressed.** The branch now
  has the tools for a convergence check. Breathing clouds do not smooth ΔE,
  however, because their longitudinal quadrature is still made of point
  subcharges.
- **Breathing worldlines are C² and causal, as intended.** The prehistory is
  inertial, and retargeting never resets position, velocity or acceleration.
- **Breathing has two model artefacts that I could reproduce.** Each creates
  ΔE as large as the physical acceleration correction, or larger, with no
  physical cause:
  - **P1-A:** clouds start at the floor width instead of the fitted width, so
    a force-free bunch "breathes" for nanoseconds.
  - **P1-B:** a rigid transverse kick makes the model tilt each cloud by
    (γ−1)θ in the lab.
- The predicted-vs-measured self-check named in the commit subject is **not
  in this repository**. No file references `beta_dot_s` outside
  `core/pic/correction.py`, so I could not review it.

**Recommendation:** merge the lattice changes. Keep breathing opt-in, but fix
P1-A first; it is about a three-line change. Either fix P1-B or label
breathing as experimental: it is not valid for crossings with coherent
deflection. Do not cite the crossing ratios until the z/phase/dt convergence
ladder (P1-C) has been run.

## Checks performed

| Check | Result |
|---|---|
| `pytest -p no:anyio tests/test_pic_correction.py tests/test_native_pic.py` on `5aa6895` | 19 passed, 1 deselected (slow), 107 s |
| Same tests on a scratch merge `development`+`5aa6895` (branch is 9 commits behind; `git merge-tree` is clean) | 19 passed, 1 deselected, 105 s |
| Defaults bit-identical | `run_pic` output JSON is byte-identical (`cmp`) between `origin/development` and `5aa6895` for (a) `example_native_pic_correction.json`, (b) the same with default cloud width, `refit_every=1` and lattice 4×4×5, (c) (b) + `midpoint_predictor`. By inspection, the non-breathing paths of `_direct`, `quasi_static` and `__init__` are unchanged, and the semantics stay `preceding_interval`. |
| `black --check` (6 changed files) | clean |
| `python -m mypy core/pic/correction.py` | no errors in `correction.py` (errors elsewhere are pre-existing) |
| `ruff` (correction.py + test) | 38 vs 20 on `development`: +17 `C408` (file idiom) and +1 `SIM102` |
| `scripts/validate_pic_breathing.py --mode compare` | Total field within 5–8e-5 of the independent reference. The *correction* error is 3.8–5.5% at the default 64 steps; see P3-3. |

## Status of earlier findings

| Earlier | Status | Notes |
|---|---|---|
| P1-1 z-aliasing of b/γ pancakes | **Partly resolved: tooling only** | `lattice_max_spacing_z_m` / `lattice_max_nodes_z` and `lattice_phase_z_cells` make option (c), the convergence check, possible. Breathing does not smooth ΔE (P1-C). |
| P2-1 dead outer layers | **Resolved** | `0 < lattice_padding_cells` is allowed, and the test shows the outer layers get weight > 0.99. The default is still 1.0 in `bunch_extent` mode (P3-1). |
| P3-1 symmetric extent | **Resolved** | Bounding-box midpoint (`correction.py:1122-1147`), with a test. |
| P3-2 implicit 4 mm floor | **Resolved** | 1 pm in `bunch_extent` (`:186-190`). Nit: `dataclasses.replace(fixed_cfg, lattice_extent_mode="bunch_extent")` keeps the 4 mm already written into `half_extent_m`; the test has to pass `half_extent_m=None` explicitly. |
| P3-3 prehistory validation and docs | **Resolved** | `None` and booleans raise `ValueError`. The sizing rule is in the docstring; see P3-4 for the cloud z extent. |
| P3-5 symmetric focusing invisible | **Attempted via breathing** | The floor (P2-B) and the rate limit (P2-A) still hide most of a transverse pinch. No K-convergence point yet. |
| Self-check | **Not reviewable** | Not on this branch. |

## Breathing physics checklist

- **Continuous worldlines.** The three cascaded first-order filters advance
  with the exact held-target solution (`:617-627`). The derivatives are
  correct: d(3rd)/dt = (2nd−3rd)/τ and d²(3rd)/dt² = (1st−2·2nd+3rd)/τ²
  (`:604-615`). A target change makes only the first stage's velocity jump,
  so subcharge position, velocity and acceleration stay continuous; only the
  jerk jumps. There is no width reset. The test's numerical derivative and
  retarget-invariance checks confirm this. ✔
- **Rate limit.** `‖target−first‖_F ≤ v_max τ / max|node|` bounds the
  first-stage speed. The later low-pass stages cannot exceed that bound, so
  the subcharge offset speed is ≤ v_max at fixed centroid β. ✔ It is
  over-restrictive, however (P2-A).
- **Rest-frame widths.** The boost and coast-to-t′=0 in `_rest_rms`
  (`:527-549`) check out by hand, including the velocity-addition formula.
  The per-axis projection uses a basis frozen at t=0 (`:565`), while the
  contraction follows the current β (`:551-561`): see P1-B.
- **Inertial prehistory.** All filter stages start equal, so the offset
  velocity and acceleration are exactly 0. The prefix and t=0 knots and
  `inertial_charge_boundary_*` use the same initial offsets and centroid u
  (`:450-461`, `:685-694`). ✔ (This still holds after the P1-A fix.)
- **Retarded-root bracketing.** Subcharge speed ≤ |β| + 0.25(1−|β|) keeps
  κ ≥ 0.75(1−|β|). If headroom runs out, `accept` refuses and leaves state
  unchanged, with no clipping (`:812-837`); the test covers this. ✔ The
  longitudinal subcharges sit ahead of the centroid, which increases the
  required prefix (P3-4).
- **Exclusions.** The near/cloud split and the cheap and certified paths
  reject breathing. ✔

## Findings

### P0: none

### P1-A: breathing clouds start at the floor width, not the fitted width, so force-free bunches get a nanosecond-scale spurious expansion current

*Where:* `correction.py:594-602`. The filters are initialised with
`_width_maps(self.width_floor_m)`, and the target is only computed afterwards
by `_refit_breathing_target`. The floor is `max(cloud_width_m, scale ·
species_RMS / K^(1/3))` (`:566-581`). For z-sliced cohorts, the transverse
cohort RMS is about the *species* RMS, so the target is about K^(1/3) times
the starting width.

*Failure scenario:* a ballistic bunch with no forces should have ΔE ≡ 0. Fixed
clouds give ~1e-12. With breathing, the clouds expand from σ/2 towards σ (for
K=8). Because of the rate limit (P2-A), this lasts ~10 ns for γ≈10.8
electrons. Throughout that time the LW field carries an expansion current that
the centroid-velocity quasi-static subtraction (`:924-945`) leaves in ΔE.

*Evidence (scratch `e3_startup.py`):* 4000 electrons, γ=10.785, rest RMS
(1, 1, 100) mm, K=8, 16 subcharges, dt = 1 ps, no forces, 200 random in-bunch
observers.

| | floor x,y (mm) | target x,y (mm) | after 400 ps: RMS / max \|ΔE\|/\|E_qs\| | lab width x after 400 ps |
|---|---|---|---|---|
| fixed clouds | – | – | 2.1e-12 / 1.2e-11 | – |
| breathing | 0.50, 0.49 | 0.97–1.05 | **1.4e-3 / 2.4e-2** | 0.50 → 0.52 mm |

This spurious ΔE is about 100× the crossing ratios quoted earlier (1.5e-5,
5.9e-7).

*Fix:* initialise all three stages at the initial target (compute
`width_target_m` and `_width_maps(width_target_m)` before building the
filters). The prehistory stays exactly inertial. Add a "ballistic breathing
⇒ offset_velocity == 0 and ΔE ≈ 0" assertion to the existing breathing block
in `test_uniform_cloud_cancellation_and_persistent_moment_refits`. That
adapts an existing test; it does not need a new one.

### P1-B: a rigid transverse kick creates a spurious lab shear of (γ−1)θ, with a shear current comparable to the physical ΔE

*Where:* `correction.py:551-561` (`_width_maps`), `:565` (`breathing_basis`
frozen at t=0), and `:548-549` (only diagonal moments are kept).

*Failure scenario:* every particle receives the same transverse kick. The lab
shape is then a rigid translation, so the source current is unchanged apart
from the centroid. In the new rest frame (boost along the tilted β), that
upright lab bunch is genuinely sheared relative to the fixed axes.
`_rest_rms` keeps only the diagonal of the covariance in the *initial* basis,
and `_width_maps` re-contracts along the *current* β. The rest-frame
z column therefore comes back with an x component of −(γ−1)θ. The filters
then move the subcharges towards that tilt, and the expansion current enters
ΔE. In a crossing, coherent deflection of the electron clouds is exactly the
dominant ΔE channel, so this artefact sits on top of the signal.

*Evidence (scratch `e7_deflect.py`):* 2000 electrons, γ=10.785, rest RMS
(0.4, 0.4, 4) mm, K=1, 4 transverse subcharges × 3 longitudinal, uniform
du_x/dt = 1e8 s⁻¹ for 100 ps (θ = 0.93 mrad).

- The target z column has x/z = −9.11e-3, against −(γ−1)θ = −9.1e-3. The
  actual particle lab covariance is unchanged.
- The maximum subcharge expansion speed is 2.0e-4 c, almost purely
  transverse, at the z-end nodes.
- At 4 probes, ΔE(breathing, τ=10 ps) − ΔE(same, τ=1 s, i.e. frozen) is
  (0.44–0.64, 0.86, 1.5, 5.7)e-3·|E_qs|.
- The fixed-cloud acceleration ΔE at the same probes is
  (0.4–4, 14.7, 4.0–5.1, 13.6)e-3·|E_qs|.
- The artefact is independent of dt (identical to 3 digits for dt = 2, 1 and
  0.5 ps), so timestep refinement will not reveal it.

*Fix:* fit the full rest covariance in a basis aligned with the *current* β,
and map back with its symmetric square root or Cholesky factor. The filters
already act on full 3×3 maps. Alternatively, fit the lab-snapshot covariance
directly, since the subcharges are lab-snapshot offsets. Either way, the
round trip lab → rest → lab is then the identity for a rigid translation.
Regression: extend the same existing breathing block with a uniform kick and
assert that the offset velocity stays ≲ roundoff.

### P1-C (carry-over of P1-1): z-aliasing is not fixed by breathing; only the opt-in dense z lattice addresses it

*Where:* `correction.py:582-593`. The breathing longitudinal quadrature is
`cloud_breathing_longitudinal_order` point subcharges, default 3.
`:1128-1142` is the dense-z option.

*Evidence (scratch `e2_zprofile.py`):* one accelerated γ=10.785 cloud
(du_x/dt = 1e8 s⁻¹, 100 ps), with observers on a z-line at b = 0.5 mm sampled
every 5 µm. The cohort's lab σ_z is 1.9 mm.

| model | ΔE peaks | share of Σ\|ΔE\|² within ±0.1 mm of peaks (share of line) | RMS(interp)/RMS(true), h = 1 mm, phase 0 / 0.37 |
|---|---|---|---|
| fixed, transverse only | one plane at z≈0 | 0.995 (2.1%) | 2.60 / 0.064 |
| breathing, order 3 | planes at z = 0, ±3.25 mm (= ±√3 σ_z) | 0.994 (5.9%) | 2.39 / 0.34 |
| breathing, order 12 | 12 planes | 0.982 (8.3%) | 0.35 / 0.25 |

ΔE stays a set of b/γ-thin sheets at the quadrature planes. The dense-z
option resolves them, but the cost scales as L/(b/γ): about 33k z nodes for a
1 m bunch at b = 0.3 mm. That is within the 65537-node limit, but roughly
2000× the cost of the earlier 4×4×17 run per step.

*Action:* before quoting ratios, run the ladder the new options allow:
n_z (or `lattice_max_spacing_z_mm`) × `lattice_phase_z_cells` ∈ {0, 0.5} × dt.

Informational: even for fixed clouds, ΔE at an in-plane probe was not
converged in dt. In `e7`, probe (0.8, 0, 0) mm gave 0.37, 2.6 and 4.1e-3
·|E_qs| at dt = 2, 1 and 0.5 ps, while off-plane probes agreed to 1–2%. I did
not find the root cause; this is pre-existing behaviour.

### P2-A: the rate limit applies the longitudinal headroom to every axis, so relativistic clouds cannot follow a transverse pinch

*Where:* `correction.py:647-658`. The speed is
`min(max_speed_c, 0.25 (1−|β|))`, divided by `max|node|`, and the whole map
update `delta` is scaled by one factor.

*Failure scenario:* at γ = 10.785, 0.25(1−β) = 1.08e-3 c. With the default
4×4×3 nodes, max|node| = 3.73, so every axis's width can change at most
2.9e-4 c ≈ 87 µm/ns. Transverse subluminality would allow ~1/γ = 0.093 c,
which is 86× more. A transverse pinch of 0.5 mm/ns, typical for a crossing
lasting about a nanosecond, lags by about 6×. The P1-A startup transient
takes ~10 ns for the same reason. Any longitudinal change also throttles the
transverse change, because of the common factor. Once headroom runs out,
`accept` raises (the test covers this), which aborts the run.

*Fix:* bound the subcharge velocity directly, or use per-axis bounds in the
boost basis: longitudinal ≲ k(1−|β|), transverse ≲ k/γ. Document the time a
cloud needs to follow a given pinch.

### P2-B: the width floor dominates the z axis of z-sliced cohorts and blocks compression below RMS/K^(1/3)

*Where:* `correction.py:566-581` and `:630-641`. The target is
`max(floor, scale · cohort_rms)`.

*Evidence:* in the `e3` bunch, `width_target_z = floor_z = 49.9 mm`
(rest), i.e. 4.6 mm in the lab. The 1/8-population slices themselves have
roughly 5× smaller z RMS, and the z subcharges sit at ±8 mm in the lab. The
z axis therefore never breathes. A transverse compression below 0.5× of the
initial σ (K=8) is invisible.

This contradicts the summary that "widths follow cohort second moments". It
also places charge well outside the cohort, so the source second moments no
longer match the particles. Consider flooring at `cloud_width_m` only, or
documenting the floor as a deliberate longitudinal smoothing length (which it
effectively is).

### P3-1: `bunch_extent` still defaults to one-cell padding

*Where:* `correction.py:84`. Only explicit fractional padding avoids the dead
outer layers. `bunch_extent` is new on this branch, so a small default
(e.g. 1e-6) would not affect `development` defaults.

### P3-2: overflow warning in the rate limit for large τ

*Where:* `correction.py:657`. `limit / max(norm, tiny)` overflows when
`delta == 0` and τ is large: it warns for τ = 1 s, the "frozen" control I
used in `e7`. The result is still correct (`min(1, inf)`). Use
`np.where(norm > limit, limit / norm, 1)`.

### P3-3: the validation script's default resolution reports a timestep error, not model error, and it covers only β=0

*Where:* `scripts/validate_pic_breathing.py`.

At the default 64 steps (order 4), the correction error is 3.8–5.5% and the
B error is 2.8–4.0%. The reference uses a continuous target, while the code
samples and holds it each step, so this is a lag, not a quadrature error:

| order, steps | correction error | B error |
|---|---|---|
| 4, 256 | 0.18% | 0.15% |
| 8, 64 | 2.1% | 1.5% |
| 8, 256 | 0.18% | 0.13% |

Default to ≥256 steps, or report a step ladder. The script never moves the
centroid, so it cannot detect P1-B or contraction effects. A γ≈10 rigid-kick
case would make it an informative oracle.

### P3-4: prehistory sizing should include the cloud's own z extent

*Where:* `correction.py:56-58` (docstring).

Breathing z subcharges ahead of the centroid add about 2γ²·Δz_node/c. That
is ~6 ns for the ±8 mm nodes in `e3`. Mention the node extent in the sizing
rule.

### P3-5: centroid acceleration under `instantaneous` semantics is a backward secant

*Where:* `correction.py:818-821` and `:682-684`.

The breathing history passes `(β_k − β_{k−1})/dt` as the instantaneous
β′(t_k), which lags by dt/2. The non-breathing path reconstructs β′ from the
β knots instead. On the same history at dt = 0.5 ps, the two differ by
4.9e-4 relative in ΔE (`e6`). That is small and first order in dt; document
it, or reconstruct the centroid part as in `preceding_interval` mode.

### P3-6: lint

`ruff` reports +17 `C408` (matching the file's idiom) and +1 `SIM102`. The
`TRY004` suppression at `:97` is fine.

## Tests (owner's rule)

No new `def test_*` functions were added. The changes are 2 parametrisations
and extended blocks. Runtime is +14 s over the earlier review.

| Change | Verdict |
|---|---|
| `breathing` parametrisation of the two midpoint tests (+2 cases) | **keep**: cheap, and covers causal isolation of the subcharge trial history |
| Breathing block in `test_uniform_cloud_cancellation_and_persistent_moment_refits` (C² derivative, retarget invariance, independent LW oracle, headroom rejection, moving floor) | **keep**; **merge into it** the P1-A (ballistic ⇒ zero) and P1-B (rigid kick ⇒ no shear) regressions rather than adding new tests |
| `bunch_rms_k` guard in the same test | **drop or keep**: pre-existing behaviour, out of scope (as noted before) |
| Prehistory-duration block | **keep** |
| Lattice blocks in `test_acceleration_interpolation_and_correction_bookkeeping` (affine oracle, bounding-box midpoint, fractional padding, analytic pancake densification, phase shift) | **keep**. The pancake check tests interpolation only, not LW physics, and says so. |
| Warm-start invalidation when the z node count changes (`test_node_batch_parity_...`) | **keep**: guards a real cache-shape hazard |
| Runner breathing smoke in `test_optional_runner_and_disabled_compatibility` | **keep** |

## Merge recommendation

- **Lattice/padding/z-densification/phase changes:** OK to merge. Defaults are
  bit-identical, and the tests pass on the branch and on a scratch merge with
  `development`.
- **Breathing:** do not use it for physics until P1-A is fixed, which is
  trivial. Also fix P1-B, or document it as invalid with coherent deflection.
  P2-A and P2-B decide whether breathing can show symmetric focusing at all,
  which was its motivation. Treat them as required before any claim that
  rests on breathing.
- **Crossing ratios:** still not citable until the P1-C convergence ladder has
  been run. The commit names a predicted-vs-measured self-check, but it is not
  in this repository and was not reviewed.
