# Review: `review/exact-audit-sources-2` vs `development`

Reviewed `git diff development...review/exact-audit-sources-2` (3 commits,
`c42bbfb`, `ce972db`, `20c9807`; 15 files, +1479/−18). The scope is persistent
exact clouds on the fixed-step route, the fixed initial source reduction
(`core/exact_source_reduction.py`), and clouds plus checkpoints for driver
trains with `INERTIAL_PREHISTORY`.

I could not see the owner's report (`codex_report_exact_audit_sources.md`), the
`light_heavy.json` / `train.json` inputs, or their results. All three live
outside the repo, under `~/work/...` on the Mac. The convergence diagnosis
below is therefore based on the code and on independent probes I wrote with
synthetic bunches. It is not a re-run of the owner's study.

Evidence scripts are in `review_evidence/` on this branch. Each one runs in
under 5 minutes from that directory.

## Summary

* **Reduction conservation works as documented.** Charge, population,
  population-weighted position (dipole moment) and population-weighted
  mechanical momentum are conserved to round-off. Source charge `N q_s` and
  observer charge/inertia `q_s`, `m_s` stay correctly separated. Current and
  kinetic energy are *not* conserved; the error is second order in the
  velocity spread inside a cell. The quadrupole moment is also not conserved
  (8 % relative error at 8 macros in my probe).
* **The integrator gets the retarded fields of the reduced ensemble right.**
  At t=0 the exact path reproduces the closed-form uniform-motion field of the
  reduced clouds to 7e-5 (the one-step O(dt) residual) at every count. That
  rules out errors in root finding, subcharge placement and cloud wrapping.
* **Default paths are unchanged.** All guards are opt-in. Degenerate clouds
  are bitwise identical to point sources, the checkpoint fingerprint changes
  only when a train is enabled, and the regression suites pass. The one
  failure seen also fails on `development`.
* **Non-convergence comes from the model and the metrics, not a code bug.**
  Details are in the last section. The main causes:
  * the cloud width is fixed and independent of cell size;
  * the partition uses a lab-mm metric;
  * same-bunch near fields are granular;
  * the light/heavy comparison uses an rms-across-macros metric over only
    2 steps;
  * the train probe has an on-axis observer whose crossing time is far
    shorter than the step.

No P0 findings. No P1 bug in the code. One P1-level *model* issue (P1-a): it
is not a defect in arithmetic, but it makes the "macros" axis of the study
unable to converge as designed.

---

## P0 — wrong default behaviour

None found.

* `core/exact_source_cloud.py:99-103`: the short-circuit is now
  `count == 1 or sigma*multiplier == 0`. Before, a one-child cloud with
  nonzero width, or a zero-width cloud with many children, produced an
  `ExactCloudHistory` whose nodes all sit at the centre. Now it returns the
  raw history. The new parametrized tests show this is bitwise identical at
  both field level and trajectory level, and I reran them. The
  `proper_velocity` guard (`core/integration_runner.py:3429-3442`) was relaxed
  to match.
* `core/integration_runner.py:3779-3800`: on the fixed-step exact route,
  nonzero smearing widths were rejected before and are now accepted. The new
  `INERTIAL_PREHISTORY` check only applies to adaptive runs, and those already
  require inertial prehistory (`:3558`). No reachable default changes.
* `core/integration_runner.py:4692`: `driver_train` enters the checkpoint
  fingerprint only when enabled, so fingerprints of existing checkpoints are
  unchanged.
* `lw_integrator/testbed_runner.py`: the three new options default to
  off/0, and `build_magnetic_dipole_config` returns
  `exact_source_reduction=None` when disabled.
* Tests I ran:
  * all new and changed test files, slow ones included: 71 passed;
  * `test_source_history_representation_config`,
    `test_integration_runner_control_flow`, `test_potential_inclusion`,
    `test_exact_trial_field_cache`, `test_magnetic_dipole_integration`,
    `test_pseudo_grid_passive_state`, both checkpoint suites and
    `test_cli_gui_parity`: 125 passed, 1 failed.
  * The failure is
    `test_magnetic_dipole_path_is_identical_across_forced_python_and_numba_kernels`.
    It also fails on a clean `development` worktree in this environment, so it
    predates this branch and is probably a numba/platform difference.

## P1 — wrong physics when enabled

### P1-a (model): reduced macros keep a global fixed width, not their cell's size
`core/exact_source_reduction.py:107-150`, `core/exact_source_cloud.py:121`,
`scripts/check_sources2.py:129,292`.

The reduction collapses each cell to its centroid. It keeps the cell's charge
and momentum but drops its spatial extent: the second moment is lost, and the
longitudinal extent is lost entirely. Each new macro then gets the same
transverse Gaussian as every other macro, set by the single global
`position_sigma_mm` (0.05/0.1/0.2 mm in the study). The description "fewer,
wider clouds" is **not what the code does**: the width does not grow as the
macro count falls.

* **Failure scenario:** the light/heavy "macros" refinement (8→16→24→48 at
  fixed `w=0.1`). This walks between models with different charge
  distributions, not along a sequence converging to a fixed source. Wherever
  the observer is not far from a cell compared with the cell size, the field
  error is O(1) rather than O((a/d)²). That holds in the same bunch, and also
  across bunches when the γ-stretched longitudinal size matters.
* **Evidence** (`review_evidence/reduction_fields.py`, `samebunch.py`, with
  48 macros, a γ=3 driver and point observers):
  * **Cross-bunch** reduction converges roughly like a multipole expansion.
    The relative rms force error at 8/16/24/47 macros is
    2.5e-2 / 6.8e-3 / 5.9e-3 / 1.4e-5 for σ_l = 0.5 mm. For γ=10 and
    σ_l = 2 mm it is 0.20 / 0.030 / 0.015 / 1e-4. Convergence is not
    monotone between neighbouring counts (24 vs 32).
  * **Same-bunch** reduction at fixed 0.1 mm width does not converge. The
    cell-projected per-macro force error stays at 20–60 % of the rms force at
    every count. With w=0.1 the field is granular: the nearest-neighbour
    distance is 0.15 mm, max/median |F| = 12, and three observers carry 75 %
    of Σ|F|². A point observer next to a neighbour's quadrature child
    dominates.
* **Fix direction:**
  * per macro, set the width from the cell's population-weighted transverse
    covariance (perpendicular to its velocity), floored at the original
    width, and record it in the mapping;
  * alternatively, refine width and count together with w ∝ cell size, so
    that count → N recovers the original model;
  * until then, do not label the "macros" axis as a convergence axis for
    same-bunch-dominated observables.

### Not a bug: what was checked in (1) and (2)

* **Conservation** (`review_evidence/reduction_fields.py`, last block; a γ=3
  driver with 1 % spread in proper velocity):
  * charge, dipole moment Σ N x and momentum Σ N p are exact to ≤6e-11;
  * the current Σ N q β is off by 9e-5 relative at 8 macros and 6e-5 at 24;
  * kinetic energy Σ N(γ−1) is off by −1.3e-4 and −8e-5 (a Jensen deficit,
    as the docs state);
  * the quadrupole moment is off by 8 % and 1.3 %.
  * The docs mention the energy deficit but not the current deficit. Add one
    line to the docs: the magnetic source term is second order in the spread
    inside a cell.
* **Source and observer separation:** `q_observer = q_species` and
  `m = m_species` are required on input (`:76-82`) and preserved. `q_source`
  and `macro_population` are summed per cell, and legacy `q` is summed only
  when it differs from `q_species`. This matches the representative-particle
  accounting in AGENTS.md.
* **Integrator fidelity** (`review_evidence/integ_check.py`, 12+12 particles,
  w=0.1 mm with 4 children, radiation reaction off):
  * the first-step kicks of the exact path match the analytic
    uniform-motion field of the **reduced** clouds to 6.7–7.3e-5 at
    counts 3/6/9/11, and the full run matches to 8.4e-5;
  * the population-weighted total kick relative to the full run drops
    monotonically: 1.7e-1, 8.0e-3, 1.8e-3, 1.3e-3.
  * This rules out retarded-root, child-placement and charge-fraction errors
    in the reduced path.
* **Reduced checkpoints:** a run with reduction that is cancelled and resumed
  is bitwise identical to the uninterrupted run (`review_evidence/ckpt_check.py`).
* **Wide clouds versus prehistory length:** the prehistory estimate
  (`core/integration_runner.py:899-955`) uses only centre separations. A
  20 mm cloud at 0.2 mm separation still matches the analytic field
  (`review_evidence/wide.py`, python and analytic backends), so coasting
  extrapolation covers the children. See P3-e.

## P2 — robustness and methodology

### P2-a: the lab-mm partition metric is wrong for relativistic sources
`core/exact_source_reduction.py:20-45`. Farthest-point seeds and
nearest-seed cells use isotropic lab millimetres. For a source moving with
γ ≫ 1, the field error depends on the rest-frame size, so a longitudinal
spread of ℓ counts as γℓ. Scaling the longitudinal coordinate by γ of the
cell before partitioning gives much faster cross-bunch convergence
(`review_evidence/restframe.py`, mean of 3 seeds, w=0.1 with 16 children):

| case | metric | 8 | 16 | 24 |
|---|---|---|---|---|
| γ=3  | lab mm      | 0.104 | 0.030 | 0.0082 |
| γ=3  | rest frame  | 0.089 | 0.012 | 0.0074 |
| γ=10 | lab mm      | 0.190 | 0.059 | 0.0129 |
| γ=10 | rest frame  | 0.167 | 0.018 | 0.0036 |

This is a cheap change, and it directly targets the macro-count convergence
the audit needs.

### P2-b: the light/heavy convergence metric compares rms values across different macro counts
`scripts/summarize_sources2.py:70-74,88-90`. `weighted_rms_kick_native`
over 8 macros and over 48 particles are different statistics. By the
variance decomposition, rms²(48) = rms²(cell means) + the variance inside
cells, so the "macros" comparison does not tend to zero even with exact
physics. In my probe this metric was not monotone (γ=3:
1.4e-2, 1.0e-3, 3.4e-3) and understated the cell-projected error by up to
10×.

* **Fix:** project the reference run's kicks onto the coarse run's
  `parent_cells`, using the population-weighted mean per cell. That mapping
  is already saved in `*_mapping.json`. Compare particle by particle, and
  keep the population-weighted total Σ N Δp, which is a valid metric.
* Note that the saved mapping is for the coarse run's own partition. Cells
  are not nested, even though the seeds are.

### P2-c: the light/heavy probe is only 2 proper steps long
`scripts/check_sources2.py:98`. `steps=3` in light mode measures the
startup force at about t=0, not an encounter. The macro and timestep axes
therefore probe the initial field plus the first-step startup transient.
This is fine as a field check. It is not an encounter-level reference, and
the report should say so if it does not already.

### P2-d: the train timestep study cannot converge as set up
`scripts/check_sources2_train_inertial.py:53,78`, `scripts/check_sources2.py`
(train: 160 fixed proper steps).

* The rider is a single on-axis point (`driver.x == rider.x`), and it passes
  *through* each driver bunch's transverse quadrature cloud.
* The longitudinal impulse from one inertial pass is the integral of a
  sharply peaked, odd function of dz. It has width
  `σ/(γ v_rel)` (`width_time`, which the script computes), and its peak scales
  as γ·Σ wᵢ/rᵢ, which tends to γ·√(π/2)/σ.
* A fixed step much larger than `width_time` aliases this odd peak. The
  result is an O(1) residual that depends on the step phase, and it neither
  vanishes nor converges until h_lab ≪ σ/(γ v_rel). The same peak converges
  slowly in child count (the script notes this for Hermite rules).
* **Fix:** report `width_time / (γ_r·h_step)` next to the study, and use
  enough steps, a nonzero impact parameter, or gate/adaptive splitting around
  each crossing. Until then, treat the train timestep study as unresolved,
  not as a bug signal.

### P2-e: no test of reduction combined with a train
`core/integration_runner.py:3925-3955`. The docs say reduction is applied
to the template before train expansion. The code order is correct: reduce,
then `_driver_train_bunch_slices` on the reduced count, then
`_build_driver_train_initial_state`. Nothing covers it, though. Add the
reduction to the existing train checkpoint test instead of writing a new one.

## P3 — nits

* **a.** The docs at `docs/source/exact_macroparticle_sources.rst:395` cite
  `scripts/check_exact_audit_sources.py`, which exists neither on this branch
  nor on `development`. Lines `:435` and `:449` cite `codex_report_*.md`
  files that are not in the repo. Either commit them or point to their
  location under `LW_feasibility_studies`.
* **b.** `scripts/check_sources2.py:55,93` hard-codes
  `/Users/benjaminfolsom/...` and `/private/tmp/compute_cap`. Make the input
  path an argument with that path as the default. The script writes large
  `.npz` files into `--output`; per AGENTS.md, that directory should be under
  a study's `results_local/`.
* **c.** `retarded_integrator` discards the reduction mapping
  (`core/integration_runner.py:3941-3945`, `_`). The returned trajectories do
  not record that they are reduced. Attach the mapping, or at least
  `original_count`, to the run metadata so that downstream population
  ledgers cannot mix reduced and unreduced runs silently.
* **d.** `core/exact_source_reduction.py:107` (and the existing
  `_build_driver_train_initial_state`) treats any array of shape `(total,)`
  as a per-particle array. A per-bunch vector whose length happens to equal
  the particle count would be wrongly seed-sliced. Low risk; an explicit
  key list would be safer.
* **e.** `:148-150` zeroes `source_position_{low,tail}_*` but copies
  `source_time_{low,tail}_ns` from the seed parent. This is harmless while
  t is common and the low parts are zero, but for symmetry zero them too.
  Likewise, `_estimate_inertial_prehistory_duration_ns` ignores the size of
  the cloud's children. This is safe today because of analytic coasting
  extrapolation, but a short comment would help.
* **f.** The docstring at `:53` says "charge-weighted position", but the
  code weights by population. They are equal only when `q_source = N q_s`;
  the neutral control uses population weighting.
* **g.** The reduction config sits under `MagneticDipoleConfig` even though
  it is a source-model option. If the field is kept there, give the
  testbed's flat `exact_source_reduction_*` keys the `magnetic_dipole_`
  prefix or a nested block, to match the neighbouring options.
* **h.** ruff C408 at `core/exact_source_reduction.py:62` (`dict(...)` →
  literal).

## Tests (owner's rule: new tests only when strongly justified)

| test | verdict |
|---|---|
| `test_exact_source_reduction.py::test_reduction_preserves_initial_population_charge_position_and_momentum` | **Keep.** Core conservation invariant. |
| `…::test_full_count_returns_original_object_without_arithmetic` | **Merge** into the identity-route test below. Same claim. |
| `…::test_partition_is_deterministic_nested_and_handles_coincident_parents` | **Merge/trim.** Keep only the coincident-parents assertion (the guard at `:44`). Nothing relies on nestedness. |
| `…::test_heterogeneous_or_asynchronous_initial_states_reject` | **Merge** into one parametrized rejection test with the invalid-count cases. |
| `…::test_invalid_counts_reject` (8 values × 2 roles × 2 enabled) | **Keep, trimmed.** It covers the previous review's strict-parsing fix, but 3–4 representative values are enough. |
| `…::test_exact_full_count_route_is_bitwise_identity_and_reduced_route_uses_clouds` | **Keep.** It guards the integrator wiring and the bitwise identity at full count. Consider adding a uniform-motion field check in the style of `review_evidence/integ_check.py`, which is stronger than "equals pre-reduced input". |
| `test_exact_audit_source_routes.py::test_fixed_step_degenerate_cloud_is_bitwise_point_trajectory` | **Merge** with the extended `test_exact_source_cloud.py` parametrization. One file for degenerate-cloud identity. |
| `…::test_fixed_step_cloud_supports_joint_observer_gate_events` (slow) | **Keep.** New capability: fixed-step clouds with many observers. |
| `test_exact_train_clouds.py::test_five_bunch_cloud_gate_checkpoint_is_bitwise_uninterrupted` (slow) | **Keep.** It justifies relaxing the checkpoint guard. Add `exact_source_reduction` here to cover P2-e. Consider fewer steps. |
| `test_exact_source_cloud.py` parametrization (+3 cases) | **Keep.** Cheap; covers the short-circuit change. |
| `test_inertial_driver_train.py` cloud parametrizations | **Keep.** Cheap; covers excluding the bunch's own cloud. |
| `test_simulation_options_magnetic_dipole.py` additions | **Keep.** Round trip and defaults off. |

## Merge recommendation

**Mergeable as an opt-in, explicitly experimental source model**, after the
cheap fixes: P3-a (dangling doc references) and P3-b (hard-coded paths); P3-c
(recording that the run is reduced) is strongly recommended. Default
behaviour is unchanged, the conservation identities hold, and the exact path
evaluates the reduced sources' retarded fields correctly.

**Do not cite the light/heavy or train results as references** until P1-a,
P2-a and P2-b are addressed for light/heavy and P2-d for the train. The
branch's own "not converged" label is accurate.

## Why convergence fails (my view)

1. **Light/heavy, macro count.** This is a model and metric problem, not a
   code bug.
   * The width is fixed while the cell size shrinks, so each count is a
     different charge distribution, and the count → 48 limit is
     "48 × w=0.1 clouds", not the coarse models' continuum.
   * Wherever same-bunch or near-field contributions matter (likely in a
     screened-source case), point observers next to 0.1 mm quadrature
     children make the field granular and dominated by a few near
     encounters. Cell-projected errors stay at 20–60 % at every count.
   * The lab-mm partition under-resolves the longitudinal direction by
     about γ.
   * The rms-across-macros metric cannot converge because of the variance
     inside cells.
   * Cross-bunch fields from the reduction do converge, and the integrator
     reproduces them to 1e-4.
2. **Train, timestep.** A single on-axis observer passes through transverse
   quadrature clouds. The impulse is an odd function of dz, peaked over a
   time `σ/(γ v_rel)` that is far shorter than the fixed step, so the
   residual is aliasing noise.
3. **No evidence of retarded-root, subcharge-placement or charge-fraction
   bugs.** At every count tested, the integrator's fields match the closed
   form to the one-step residual.
