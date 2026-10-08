# Review: `review/boundary-coupling` vs `development`

Scope: `git diff development...review/boundary-coupling` (commits `85c6fa8`,
`85f5d02`): conformal-wall PEC backend (`core/boundary/conformal*.py`), opt-in
two-way axial coupling (`core/boundary/coupling.py`, `axial_particle.py`), a
`mesh.pyi` stub, a one-line `native_rr` extension, two new test files and one
fixture. Line numbers refer to the branch head `85f5d02`.

Evidence was gathered by reading every changed file, running the fast tests
locally (`12 passed, 10 deselected`), running the four slow gates on the Mac in
a throwaway clone (`/tmp/lwreview`), and running independent scripts:
a three-level refinement of the conformal taper work (short 30 mm taper), a
retarded-kernel vs FDTD on-axis comparison at two resolutions, an energy-drift
and stability run, an LW-correction check against the analytic boosted
Coulomb field, and a wall-slope contract probe. Numbers are quoted inline.

## Merge recommendation

**Merge, after fixing P2-1 (slope contract) and the P3 test cleanups.** No P0
or P1 findings. Default paths are untouched: all pre-existing solver files are
byte-identical to `development`, `select_pec` defaults to the unchanged
`PECWakeSolver`, `TwoWayBoundaryCoupling(enabled=False).run()` returns `None`
without touching its factory, and `native_rr(gamma=None)` reproduces the old
expression exactly. The conformal discretisation is correct, energy-adjoint and
second order; the coupling seam uses the observer charge and avoids
self-field double counting. The remaining issues are robustness/labeling.

## Findings (ranked)

### P0 — wrong default behaviour

None found.

* Files `solver.py`, `wake.py`, `mesh.py`, `materials.py`, `incident.py`,
  `observers.py`, `diagnostics.py`, `history.py`, `_retarded.py` have an empty
  diff against `development`; hence the aligned pipe/pillbox/contraction/
  expansion regressions are bit-identical by construction.
* `core/boundary/particle.py:36`: `gamma if gamma is not None else
  1/np.sqrt(1-velocity*velocity)` — legacy callers unchanged.
* `core/boundary/conformal.py:374-380`: `boundary="original"` returns
  `PECWakeSolver(g, dt, segments)`; `wall` is rejected unless `"conformal"` is
  chosen explicitly. `tests/test_boundary_conformal.py:98` checks the type.
* `core/boundary/__init__.py` only re-orders imports and adds exports; no
  import side effects beyond modules already imported on `development`.

### P1 — wrong physics when enabled

None found. Items checked and verified:

* **Mapped metric** (`conformal.py:58-94`). Stored covariant components
  `eu = R_u Er`, `ev = Ez + R_z Er`, `b = R_u Bθ` with `√g = R R_u`; the
  weights `hr = 2πR dr dz / R_u`, `hb` likewise, `hz` = exact mapped ring area,
  reproduce `½∫(E²+B²) dV` in physical variables. `metric_inverse`
  (`:197-210`) is the exact block elimination of the non-diagonal energy metric;
  the Schur complement at the wall row (`:103-108`) is the correct tridiagonal
  left by eliminating the constrained `ev[-1]=0`. Verified by derivation.
* **PEC condition**: `zmask[-1]=False` (`:99`) forces `ev = E·(R_z,1)` — the
  tangential covariant component — to zero on the `u=const` wall; end caps are
  PEC via `rmask`. Correct.
* **Faraday/Ampère signs**: `ḃ = +C e`, `He ė = −Cᵀ Hb b − J`; consistent
  with the original cylindrical solver and verified by the adjoint test and by
  my 400-step no-source runs: relative energy drift −2.8e-13 (fixed point,
  dt=0.01) and −8.7e-15 (CG, dt=0.04) on a sheared wall `1.5+0.1 sin z`.
* **Second order**: independent refinement of `conformal_wake("in",
  length=30)` at (dr,dz,dt) = (0.1,0.25,0.1), (0.05,0.125,0.05),
  (0.025,0.0625,0.025): work 0.55521, 0.56108, 0.56260 eV/e; successive-difference
  ratio 3.85 (gather 3.85). Richardson limit 0.56311 eV/e against the analytic
  adiabatic EM-energy difference `N e ln2 /(2π ε0 · 2√π σ)` = 0.563122 eV/e.
  "pair" at length 30: −2.56e-5, −2.75e-5, −2.80e-5 eV/e, ratio 3.78. So the
  scheme is second order and converges to the physical value; the 0.36 %
  offset at the gate resolution (`out` fine: −0.561084) is discretisation
  error, not taper physics (the L=30 and L=300 mid-resolution values coincide
  to 4 digits).
* **Stability**: CG path is the Cayley/midpoint update (unconditionally
  energy-stable); in practice `dt` is capped by the axial symbol check
  `dt·k_max < π/2` (`:149`), i.e. `dt < dz/2`, not by a CFL. Fixed-point path
  enforces `dt ≤ 0.4·min(R_u dr, dz)` and a contraction bound (`:148-156`).
  Both verified to raise.
* **Observer charge in the two-way seam** (`coupling.py:294`):
  `return p.charge * e, p.charge * derivative`; the source charge
  (`population*charge` by default) enters only the `RespondingDrive`
  (`:149-150`). `tests/test_boundary_coupling.py:93` pins this. Work is
  `∫ f v dt` with `f = q_obs Ez` (`axial_particle.py:128-129`); the energy and
  momentum ledgers close to < 1.5e-8, which fixes the sign convention.
* **Double counting**: the particle is pushed only by the P/J retarded field
  (`_retarded.field`), never by its own incident LW field; the material is
  driven by `scattered + incident` in the constitutive update only
  (`solver.py:170`, unchanged). No double counting.
* **Causality / time-centering**: the coupling is a serial Picard iteration
  (`coupling.py:352-512`): material pass *n* is driven by the complete, sealed
  trajectory of pass *n−1* (with inertial prehistory), and the particle is
  integrated continuously (DOP853) through the sealed record. There is no
  interleaved push/field update to mis-centre. Retarded times are solved
  consistently (`incident.point_fields`, `_retarded.field`).
* **LW correction of the responding drive**: `point_fields` on an
  `AxialTrajectory` with a smooth velocity ramp reproduces the analytic
  boosted-Coulomb field at the present position (post-ramp retarded time)
  to all printed digits, and the pre-ramp "virtual position" field where the
  retarded time lies in the prehistory (hand check: −0.004157 vs −0.00415663).
* **Retarded kernel vs FDTD scattered field** (sign/scale of the force):
  on-axis `Ez` from `AxialBoundaryForce` vs `solver.snapshot().sample` for the
  iris geometry: ratios 0.90–0.97 at h=0.1 and 0.974–0.983 at h=0.05 (dt=h/10).
  Same sign, converging; see P2-3 for the implication.

### P2 — robustness

**P2-1. Wall slope contract is unchecked** — `core/boundary/conformal.py:61-76`.
`wall(z)` returns `(radius, slope)` and the slope is trusted. Passing a slope
of zero or of the wrong sign is accepted silently and the solver then solves a
*different* geometry: `shear = R_z/R_u` is wrong, so the PEC condition is
imposed on the wrong tangent and `physical_ez` is wrong, while the energy
identity still holds exactly (it is structural), so the built-in diagnostic
cannot detect it. Evidence: `ConformalPEC(g, 0.01, lambda z: (1.5+0.1*sin z,
0*z), 0.2)` → accepted, `max|shear| = 0.0`; with `-0.1*cos z` → accepted,
`shear[-1,0] = +0.0552`. Fix: in the constructor compare `slope` with a
central finite difference of `radius` at `z_half`/`z_node` (tolerance ~1e-6
relative to `max|slope|`+`dr/dz`) and raise.

**P2-2. Finite-window continuation is an unquantified model input**
— `core/boundary/_retarded.py:181-196`, `coupling.py:263,328`. After the
record ends at `source_end`, J is tapered to zero over `stop_duration` (1.0)
and P is frozen for ever; the particle is then integrated to `observation_end
= 1e6`, so the final impulse/work include the static tail of a frozen
polarization that a Drude wall (ν=2) would actually relax. The code documents
the scope string, but `run()` reports no sensitivity to `source_end` or
`stop_duration` and the gate test fixes both. Failure scenario: a user
enabling coupling and reporting "work per electron" obtains a number that
depends on two arbitrary window parameters at an unknown level. Fix: expose
the terminal-continuation contribution separately (the one-way path already
has `terminal_continuation_work` in `MaterialHistory.complete_ballistic_work`)
and/or require a documented `source_end` convergence check before a result is
labelled `converged`.

**P2-3. Force kernel and field solver disagree at the gate resolution**
— `coupling.py:283-293`, `history.py:55-81`. The particle feels the dipole-sum
reconstruction of P/J (quadrature `order`, default 4), whereas all surface and
energy ledgers use the FDTD scattered field. My comparison (iris, γ=10) shows
10 % (h=0.1) and 2 % (h=0.05) differences in on-axis `Ez`, converging with h.
The coupled gate (`dr=0.05, dz=0.025`) reproduces the fixture to 3e-7, which is
reproducibility, not accuracy; the physical work carries a few-percent
discretisation uncertainty that is not stated anywhere in the result dict.
Fix: add a single scalar (e.g. kernel/FDTD on-axis ratio at `source_end`) to
the returned diagnostics, or document the expected magnitude.

**P2-4. Picard "converged" is a weak-coupling statement** —
`coupling.py:433-471`. Convergence is declared when pass-to-pass changes of
work, impulse, position and γ fall below `tolerance=1e-3` (default). With
`feedback_updates=1` this only shows the first correction is small, which is
guaranteed whenever the wall field is weak; the fixed-point error is bounded
by the change times an unknown contraction ratio. Not wrong, but the default
1e-3 is loose relative to the 3e-7 claims in the gate. Suggest default 1e-4
and reporting the achieved change in the summary (it is already in `rows`).

**P2-5. The coupled gate is blind to the feedback term** —
`tests/test_boundary_coupling.py:336-381`, `coupling.py:221-238`. In the
`smooth_d02` gate the second (fed-back) pass changes the work by 4.9e-8
relative (`relative_work_change`), with `peak_incident_correction` 2.9e-11,
while each row is compared to the fixture at `rel=3e-7`. A sign error or an
omitted `corrections` term in `RespondingDrive.material_volume` would still
pass (`peak_incident_correction > 0` and `relative_work_change < 2e-7` both
hold either way). Only the one-way machinery is actually regression-tested.
Fix: assert the fed-back row against its fixture value at a tolerance below
the feedback effect (the fixture rows differ by 1.1e-7 relative between
iterations; `rel=3e-7` is coarser than that), or add a cheap sign test of
`material_volume` on a trajectory whose correction is O(1) (my local check:
`point_fields` on an `AxialTrajectory` matches the analytic boosted field, so
the kernel itself is right; the untested piece is the assembly at `:221-238`).

**P2-6. CG stopping and NaN handling** — `conformal.py:322-345, 268-280`.
`tolerance = 2e-14·‖rhs‖` with a 100-iteration cap and a recomputed true
residual (`> 5·tol` → raise). Observed 3–7 iterations and residuals ≤ 7e-15
in all my runs, so this is fine, but preconditioned CG can stagnate near
1e-14 relative for strongly stretched walls or large `dz/dr`; the failure mode
is a `RuntimeError`, not a silent accept — acceptable. NaN propagates to a
false comparison and the same `RuntimeError` in both solvers; `AxialTrajectory`
and `integrate_axial_particle` reject non-finite inputs/outputs. OK.

### P3 — nits

* `tests/test_boundary_coupling.py:205-331`: `inspect_coupled_records` (125
  lines) is never called by any test — dead code. Drop it or wire it into the
  slow gate.
* `tests/test_boundary_coupling.py:74-77`: integer-vs-float input equality
  sub-assert adds nothing; trim.
* `core/boundary/conformal_wake.py:150`: `reference_eV_per_electron =
  0.563121173` is the analytic adiabatic EM-energy difference (verified:
  0.5631218 with CODATA constants) — say so in the docstring; "reference"
  currently reads as an external simulation.
* `tests/test_boundary_conformal.py:143-148`: the `pair < 1.11e-6` and
  `0.00381` bounds are resolution-specific; the pair residual is a physical,
  converging loss (−2.8e-5 eV/e at L=30), not a cancellation identity. Comment
  this so the bound is not tightened later for the wrong reason.
* `coupling.py:155`: exact float equality `trajectory.beta != self.beta`
  rejects a user-built `AxialTrajectory` whose `beta` was computed as
  `sqrt(1-1/g**2)` instead of `sqrt(1-g**-2)`. Use `np.isclose(..., rtol=0,
  atol=1e-15)`.
* `coupling.py:120-129` + `incident.py:203-209`: a Newton overshoot past
  `x[-1]+1e-8` inside `point_fields` raises the trajectory's `ValueError`
  rather than the "unsaved future source history" message. Cosmetic.
* `conformal.py:72-76`: the map is only C0 at `u=core` (`R_u` jumps 1→scale;
  shear set to 0 at the node `u=core` while its dual cell straddles the mapped
  region) and the source ring at `u=core` uses the reference volume
  `source_hz` (`:86,266`) while its physical volume is `hz`. Local first-order
  inconsistency in one ring; the refinement study shows it does not spoil
  global second order. Document it.
* `mesh.pyi` duplicates runtime signatures and will drift; consider annotating
  `mesh.py` instead.
* mypy on the new files: `coupling.py:283` `Too many arguments for "__call__"
  of "Dispatcher"` (numba); add a `# type: ignore[call-arg]` or cast.

## New tests — keep / merge / drop

`tests/test_boundary_conformal.py`

* `test_mapped_metric_adjoint_and_midpoint_energy` — **keep** (adjoint, energy
  identity, amplitude linearity; the amplitude part is the regression for the
  absolute-floor fix).
* `test_physical_mapped_gather_and_rejections` — **keep**.
* `test_independently_mapped_uniform_pipe` — **keep** (this is the
  default-selector / unchanged-solver guard).
* `test_thread_c_taper_gate` (slow, 2 cases, 3 + 6 min) — **keep** as the
  frozen reproduction; add the comment from P3 about resolution-specific
  bounds.

`tests/test_boundary_coupling.py`

* `test_disabled_coupling_has_no_effect` — **keep** (default-off guard).
* `test_boundary_force_uses_observer_charge_and_complete_derivative` —
  **keep** (regression for the omitted-observer-charge port bug).
* `test_smooth_axial_integrator_ledgers_and_gamma_1000_work` — **keep**
  (regression for the short-window extrapolation fix).
* `test_trajectory_c2_prehistory_and_accelerated_lw` — **merge/trim**: drop
  the int-vs-float sub-check; the rest is useful.
* `test_coupling_rejects_abrupt_or_reused_material_solver` — **keep**, it is
  already one test for the three rejection paths plus non-convergence.
* `test_thread_c_step2m_coupled_gate` (slow, 2 ramps) — **keep one ramp**
  (`smooth_d02`) as the frozen gate unless the two ramps are meant to be
  compared; the second doubles a long run for a reproducibility check.
* `inspect_coupled_records` — **drop** (dead).

Fixture `tests/fixtures/boundary2_reference.json` — keep; it carries the
source commit and SHA-256 of the originating summaries.

## Validation performed

* Fast suite on the branch: `pytest -p no:anyio -m "not slow"
  tests/test_boundary_conformal.py tests/test_boundary_coupling.py
  tests/test_boundary_module.py` → 12 passed.
* Slow conformal gates (Mac): `test_thread_c_taper_gate[pair]` 181 s,
  `[out]` 372 s → 2 passed.
* Slow coupled gate (Mac): `test_thread_c_step2m_coupled_gate[smooth_d02]` →
  1 passed in 1209 s (material pass ~1.2 min, particle integration to 1e6
  ~10 min per pass). `converged=True`, `feedback_updates=1`; work
  −9.2268018e-7 (fixture −9.2268008e-7, 1.1e-7 relative), impulse
  −9.2735030e-7, `relative_work_change` 4.9e-8, peak energy residual 6e-11,
  `medina_over_lienard` 0.99996.
* Independent refinement, drift, kernel-vs-FDTD, LW-correction and slope-probe
  scripts as quoted above (`/tmp/lwreview_*.py` on the Dell and Mac; outputs in
  `/tmp/lwreview-out` on the Mac).
* `black --check` clean; `ruff` not installed in the review env.
