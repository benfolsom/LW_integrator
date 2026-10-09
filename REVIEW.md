# Follow-up review: `review/exact-audit-sources-2` (through `1f4a456`) vs `development`

This review covers the new commit `1f4a456`, which answers the earlier review
(branch `review-notes/exact-audit-sources-2`) and adds opt-in "breathing"
exact source clouds. I checked three things:

1. whether each earlier finding is resolved, and whether the macro-count axis
   can now converge;
2. the breathing physics on the exact path;
3. that defaults are bit-identical.

All evidence scripts are in `review_evidence/` on this branch and run from the
repo root:

* `count_convergence.py` (≈10 s)
* `breathing_probe.py` (part A ≈5 s; `--integrator` ≈3 min)
* `breathing_tilt.py` (≈2 s)
* `breathing_vs_rigid.py` (≈1 min)
* `default_identity.py <checkout> <out.json>` (≈2 min)

`reduction_fields.py` is copied from the earlier review; the convergence
script imports it.

## Summary

* **Defaults are bit-identical (no P0).** I hashed every array of the full
  trajectories on `development` and on this branch:
  * the exact point route (python and numba backends);
  * two shipped B2B example configs, run through `run_testbed`.

  The digests are identical. Rigid (non-breathing) unreduced fixed-step clouds
  are also bit-identical to the previously reviewed commit `20c9807`.
  * Test runs: all new and changed test files, slow ones included: 116
    passed.
  * Broader regression suites (checkpoint, trial cache, inertial prehistory,
    control flow, potentials, CLI/GUI parity, same-bunch, gates): 134 passed.
* **The earlier findings are addressed.** Two are partly open: the doc
  references (P3-a) and two old nits (P3-d, P3-g). The per-cell width is
  implemented as `max(σ_orig, σ_cell)`, not in quadrature, so the docs'
  "retains the second-moment trace" claim is not exact (P2-b).
* **Macro-count convergence:**
  * Cross-bunch: converges monotonically.
  * Same-bunch per-cell kicks: still do not converge.
  * Same-bunch smooth observable: the transverse virial Σ N x⊥·F⊥ converges
    only about linearly in the fraction of particles merged, because the
    reduction drops the self-field inside each cell.
  * Per-cell widths barely move either result. This is a model limit, not a
    code bug. Details are in the convergence section below.
* **Breathing, reduced route:** sound as specified.
  * The ballistic covariance closure is exact for free-streaming members.
  * Width, rate and acceleration knots are mutually consistent.
  * At constant width, a breathing child reproduces the rigid child bit for
    bit, even on an accelerated centre.
  * Inertial prehistory and the checkpoint/resume path are consistent.
* **Breathing, unreduced route:** two physics defects when enabled.
  * P1-a: each macro projects the ensemble onto its *own* initial velocity
    normal, so longitudinal extent leaks into the "transverse" RMS.
  * P1-b: a driver train is pooled into one covariance. This is accepted at
    runtime with no guard.
* **Breathing cost (P2-c):** the whole child history is rebuilt on every
  append, so cost is O(K²) per run.

---

## P0: wrong default behaviour

None found.

* `default_identity.py` hashes every numeric array of the full trajectories
  (and testbed metrics) for:
  * `crossing_run` exact point sources (python and
    `numba_analytic_charge_response_serial`);
  * `example_b2b_counter_propagating_proton_bunches.json` and
    `example_b2b_relativistic_proton_stationary_lead_ion.json`, each at 40
    steps through `run_testbed`.

  Digests on `development` and on `1f4a456` are identical. Fixed-step rigid
  clouds are rejected on `development`; their digests on `1f4a456` equal
  those on `20c9807`.
* Breathing is gated at every entry point:
  * `MacroparticleSmearingConfig.breathing_enabled=False` by default
    (`core/types.py:276`);
  * the testbed emits `breathing_*` keys only when enabled
    (`lw_integrator/testbed_runner.py:753-760`);
  * the checkpoint fingerprint drops `breathing_*` fields while disabled
    (`core/integration_runner.py:118-122`), so existing fingerprints are
    unchanged;
  * `EXACT_SOURCE_REDUCTION_FIELDS` are copied or published only when
    present or nonzero (`core/equations.py:356-360`, `core/types.py:1815-1823`).
* Checkpoint constants now include the six `exact_source_*` arrays (zeros on
  default runs). Older archives without them still resume:
  `test_resolved_knot_checkpoint.py` strips them and passes.

## P1: wrong physics when enabled

### P1-a: unreduced breathing projects the ensemble onto each parent's own velocity normal
`core/exact_source_cloud.py:285-287, 324-333`

For each parent, `normal = betas[0, parent] / |…|`. The ensemble deviation
`delta` (all members) is then projected perpendicular to that normal. When a
macro has a transverse velocity, its normal is tilted. The bunch's
longitudinal extent then projects into its "transverse" RMS, in the baseline
and at every later time.

* **Failure scenarios:**
  * Different macros of one self-similar flow get different widths.
  * Long bunches with angular spread get widths that are far too small,
    because the tilted baseline is inflated by σ_l·θ and does not grow.
* **Evidence** (`breathing_tilt.py`): a ballistic, force-free bunch with 32
  macros, β_z=0.6 and τ=1e-4 ns. "Truth" is the lab-transverse self-similar
  width at t=0.05 ns.

| σ_l | β_t rms | truth | realised min / median / max |
|---|---|---|---|
| 0.5 mm | 0.10 | 0.489 | 0.430 / 0.475 / 0.504 |
| 5 mm | 0.10 | 0.518 | **0.125 / 0.195 / 0.427** |
| 5 mm | 0.01 | 0.121 | 0.114 / 0.118 / 0.123 |

  * The effect scales with σ_l·θ/σ_t. It is negligible for mrad beams and
    large for wide-angle or low-γ ensembles.
  * `test_whole_parent_exclusion_in_field_and_mutual_provider[True]` cannot
    see it, because its two members share z (the test's comment notes that the
    projection "cancels").
* **Fix:**
  * Compute the ensemble transverse variance once, on a common axis (the
    population-mean proper-velocity direction, as the reduction metric
    already does). Apply the resulting ratio to every parent.
  * Keep each parent's own normal only for placing its nodes.
  * Add a z offset to the existing exclusion test, rather than writing a new
    test.

### P1-b: unreduced breathing pools a whole driver train (or any multi-bunch history) into one covariance
`core/exact_source_cloud.py:277-278, 325-328`; no guard in
`core/integration_runner.py:3519-3527, 3807-3825`.

The unreduced target is the population-weighted transverse RMS of *every*
source in the history. A driver-train history holds all bunches. Each
macro's width therefore follows the train-average expansion, not its own
bunch.

* **Failure scenario:** in a train whose bunches evolve differently (leading
  vs trailing bunch, or different encounter histories), a cold bunch's clouds
  expand and an expanding bunch's clouds lag.
* **Evidence:**
  * In `breathing_probe.py` A2, two coaxial bunches 50 mm apart share one
    history; A is ballistic with β_t=0.1 and B is cold.
    * B's macros grow from 0.100 to **0.541 mm**; their own-bunch truth is
      0.100.
    * A's macros reach 0.118 mm against an own-bunch truth of 0.744 mm. Here
      P1-a also contributes.
  * In `breathing_probe.py --integrator`, an unreduced breathing run with
    `DriverTrainConfig(bunch_count=2)` is accepted and runs.
  * The reduced route is unaffected, because its covariance is per cell.
* **Fix:** do one of the following:
  * compute the unreduced covariance per `driver_train_bunch_ranges` slice
    (the ranges already exist for same-bunch exclusions);
  * or reject unreduced breathing with trains until that is done.

## P2: robustness and methodology

### P2-a (model): the same-bunch macro axis still cannot converge per cell, and smooth same-bunch observables converge only slowly
`core/exact_source_reduction.py:162-192`, and the docs at
`docs/source/exact_macroparticle_sources.rst:570-588`.

`count_convergence.py` uses uniform-motion LW fields at t=0, 48 macros, w=0.1
with 16 children, and the mean of 3 seeds. It reports the population-weighted
cell-projected RMS force error (the metric `summarize_sources2.py` now uses)
and three width models:

* `fixed`: σ_orig, the old behaviour;
* `branch`: max(σ_orig, σ_cell), as shipped;
* `quad`: √(σ_orig² + σ_cell²).

| case | width | 6 | 12 | 24 | 36 | 47 |
|---|---|---|---|---|---|---|
| cross γ=3, gap 2 | branch | 3.9e-2 | 1.9e-2 | 4.8e-3 | 1.4e-3 | 4.9e-5 |
| cross γ=10, σ_l=2 | branch | 0.21 | 6.9e-2 | 1.1e-2 | 1.3e-3 | 6.7e-5 |
| cross γ=3, gap 1, σ_t=0.5 | fixed / branch | 0.54 / 0.38 | 0.23 / 0.26 | 0.18 / 0.15 | 7.5e-2 / 7.8e-2 | 2.4e-3 |
| same γ=3 | branch | 0.12 | 8.9e-2 | 0.12 | 0.16 | 0.12 |
| same γ=1.05 | branch | 0.12 | **2.1** | 0.16 | 0.24 | 3.2e-2 |
| same γ=3, virial Σ N x⊥·F⊥ | branch | 0.99 | 0.78 | 0.37 | 0.17 | 1.7e-2 |
| same γ=1.05, virial | branch | 0.62 | 0.39 | 0.11 | 4.0e-2 | 4.3e-3 |

* **Cross-bunch** converges monotonically. The γ-scaled partition makes this
  hold for γ=10 as well. Per-cell widths change the error by ≤5 %, except at
  a near gap, where they help at 6 macros and slightly hurt at 12.
* **Same-bunch per-cell kicks** stay at 10–25 % even at 47 of 48 macros, and
  they show spikes. The 2.1 is a reduced centre landing next to a widened
  neighbour's quadrature child. Point observers among point children in
  transverse-only discs remain granular. Widening the clouds does not fix
  this.
* **Same-bunch smooth observable.** The transverse virial drives rms-size
  growth. It converges monotonically but roughly linearly in the merged
  fraction: at γ=3, 99 % / 78 % / 37 % of it is missing at 6 / 12 / 24
  macros. This is the dropped self-field inside each cell; members of a cell
  no longer repel. Breathing cannot restore it, because its covariance is
  ballistic or kinematic and not force-driven.
* **Recommendation:**
  * Use the macro axis only for observables dominated by cross-bunch fields.
  * Report same-bunch results as model-dependent.
  * Any light/heavy quantity sensitive to self-expansion should not be read
    off reduced runs without a correction for the self-field inside cells.
  * Note this in the docs next to "cell self-fields are unresolved".

### P2-b: the per-cell width is `max`, not quadrature; the docs' second-moment claim is inexact
`core/exact_source_cloud.py:133`, `core/exact_source_reduction.py:189-192, 233`,
`docs/source/exact_macroparticle_sources.rst:581-586`.

Every original member is itself a cloud of per-axis width σ_orig. The
represented transverse second moment of a cell is therefore
σ_cell² + σ_orig². `max(σ_orig, σ_cell)` always under-represents it: by
30 % in width when σ_cell ≈ σ_orig, and entirely by σ_orig when the floor is
active. The docs say the trace is retained "when the floor is inactive". That
is true only for point members.

* The convergence limit is unaffected (σ_cell → 0), and the field impact in
  P2-a is small. Quadrature is not uniformly better in those probes.
* **Fix:** either switch to quadrature, or reword the docs ("floored cell
  RMS; does not add member width").
* The reduced breathing target (`exact_source_cloud.py:256-262`) uses the
  same convention.

### P2-c: the breathing history is rebuilt from scratch on every append, so cost is O(K²)
`core/retarded_fields.py:1621-1632, 1984-1993`, and
`core/exact_source_cloud.py:279-323`.

Every accepted append, and every trial, changes the row count, which misses
the `macro.clouds` cache. Each miss then repeats the whole reconstruction:

* `breathing_width_knots` runs over the whole history;
* the Python `np.ndindex` loop of Decimal `_translate_resolved_position` runs
  over K × children;
* every child calls `_prepare_source_history` from row 0.

The asynchronous unreduced predictor is additionally an O(N²K) Python triple
loop per call. Any momentum spread under a fixed proper step makes rows
asynchronous.

* **Evidence** (`breathing_probe.py`):
  * One asynchronous width call takes 0.04 / 0.07 / 0.14 / 0.29 s at
    N=8/8/16/16 and K=100/200/100/200, which is N²K scaling.
  * With 2 macros and 4 children (numba backend), the breathing overhead over
    the rigid run is +0.9 / +5.5 / +12 s at 16 / 32 / 64 intervals.
* At the light/heavy scale (96 macros × 16 children, 1,200 states) this
  extrapolates to days. The rigid path appends incrementally.
* **Fix:**
  * Width knots are append-only by construction. Cache widths, rates and
    child sources per accepted prefix, and rebuild only the trial tail, as
    `_expand_cloud_prepared_history` does with `previous`/`provisional`.
  * Vectorise the asynchronous alignment over members.

### P2-d: light/heavy reference cost
`scripts/check_sources2.py:136-152`

Running the input's 1,200 states resolves the earlier P2-c. By the script's
own `expected_seconds`, though, `a48_n16_w0.1_h1` is about 266.8 × 1199/2 ≈
1.6e5 s (about 44 h serial). Every charged trial now also has a matched
neutral run, so the grid roughly doubles. Plan the runs with `--prepare-only`
and say in the report which comparisons were actually completed.

## P3: nits

* **a. (partly open)** The published docs now cite
  `~/work/extracurr/...` paths and two local, untracked reports
  (`codex_report_sources2.md` and `codex_report_breathing.md`, at
  `exact_macroparticle_sources.rst:439, 481`). This is acceptable if those
  sections are labelled as local, but a reader of the built docs cannot follow
  them.
* **b.** `breathing_response_time_ns` defaults to 0.01 ns
  (`core/types.py:277`, `testbed_runner.py:340`). The docs ask for "an
  explicit positive" value. Either require it when breathing is enabled, or
  document the default.
* **c.** Breathing child proper velocity is rebuilt from float β
  (`retarded_fields.py:1985-1988`, `gamma = 1/sqrt(1-β²)`). Above γ≈10³ this
  loses the precision the centre's resolved `u` carried. A better form is
  `u_child = u_centre + γ-consistent correction`, or computing the deficit
  from the centre's resolved parts.
* **d.** The subluminal guard is correct and fails closed. For relativistic
  beams with high-order rules, though, nodes at about 3–4σ multiply σ′ and
  can trip it while every represented particle is physical. Mention this in
  the docs next to the error.
* **e. (old P3-d, open)** `exact_source_reduction.py:123-127` still treats
  any array of length `total` as per-particle.
* **f. (old P3-g, open)** Reduction and breathing still live under
  `MagneticDipoleConfig` and `macroparticle_smearing`, using flat testbed keys.
* **g.** `mypy core/exact_source_cloud.py` reports `float | None` arithmetic
  errors at `:133`, `:140` and `:243`. `validate_exact_cloud` has already
  rejected `None`; narrow it locally. Ruff is clean on both new modules.
* **h.** `retarded_integrator` still discards the full reduction mapping
  (`integration_runner.py:3968-3974`). `original_count` and `cell_sigma` now
  travel with the trajectories, which is enough for ledger safety.

## Earlier findings: status

| earlier | status |
|---|---|
| P1-a fixed width | **Resolved as proposed.** Each macro's width comes from its cell's transverse covariance, floored at σ_orig and recorded in the mapping and trajectories. Remaining issues: `max` instead of quadrature (P2-b), no longitudinal extent, and same-bunch non-convergence (P2-a). |
| P2-a lab-mm metric | **Resolved.** The metric uses the population-mean proper-velocity γ stretch, with a seed-order test. |
| P2-b rms metric | **Resolved.** Kicks are cell-projected, population-weighted, against an identity reference with checked populations. Neutral runs are now matched per trial. |
| P2-c 2-step probe | **Resolved.** The default is the input's 1,200 states; the cost is in P2-d. |
| P2-d train timestep | **Resolved in set-up.** The lab step is h ≤ σ/(γ_s v_rel S), `samples_per_width` is reported, an optional impact parameter is available, and the pulse window is labelled. With the default impact 0, the peak still converges slowly in child count; the script states this. |
| P2-e reduction + train | **Resolved.** The train checkpoint test reduces the template (asserts σ_cell = 0.05/√2). |
| P3-a docs refs | Partly resolved (above). |
| P3-b hard-coded paths | **Resolved** (`--input`, `--compute-cap`). |
| P3-c record reduction | **Resolved** for count and σ. |
| P3-d shape slicing | Open. |
| P3-e low/tail, prehistory comment | **Resolved.** |
| P3-f docstring | **Resolved.** |
| P3-g config placement | Open. |
| P3-h ruff C408 | **Resolved.** |

## Breathing physics: what I checked that is correct

* **Kinematics.** Child position is x_c + σ(t)·n̂, β is β_c + σ′ n̂/c, and
  β′ is β′_c + σ″ n̂/c². Units match `beta_prime_per_mm` (= a/c²).
  * The critically damped update (`exact_source_cloud.py:347-362`) matches the
    closed form for a piecewise-linear target. I re-derived it.
  * Numerically, the rate matches a finite-difference width to 1e-4 and the
    acceleration matches a finite-difference rate to 7e-3 (FD truncation).
  * The width lags the target by about 2τ, as expected.
* **Accelerated centre.** At constant width (one macro, so zero ensemble
  variance), breathing kicks equal rigid-cloud kicks to 0 and 3.6e-17
  (`breathing_vs_rigid.py`, python and numba). I confirmed that 36 breathing
  expansions ran. So reusing the centre's reconstructed β′ under
  `instantaneous` child semantics reproduces the centre worldline. The repo
  test covers breathing only on an inertial centre.
* **Reduced closure.** For free-streaming members, σ_cell² + 2t·xv + t²·vv
  equals the members' transverse RMS about the accepted centre at ages 0,
  0.02, 0.05 and 0.1 ns (ratio 1.0000, `breathing_probe.py` A3).
* **Prehistory and boundary.** Widths are held at the initial value for
  t ≤ epoch (`:336`). Rate and acceleration start at 0, so the child is C²
  across the inertial boundary. The child boundary is the centre's boundary
  plus `offsets[0]` (`retarded_fields.py:2003-2006`).
* **Resolved knots.** These are filled consistently for children: u, ready,
  low/tail translation and speed deficit. Time low/tail are inherited, which
  is correct because a child shares its centre's lab time.
* **History and checkpoint.** Widths are a pure function of the immutable
  constants and the accepted or trial knots. The asynchronous predictor reads
  only each row's prefix, so appends do not repaint earlier knots. I checked
  that synchronous rows reproduce exactly through the cubic path: s = 1 and
  dt = 0 give exact endpoints. Resume therefore equals an uninterrupted run.
  The slow train test checks this bit for bit for the reduced route; the
  unreduced route has no checkpoint test, but its determinism follows from the
  same argument.
* **Trial caching.** Trials are cached by `id(history)` with a weakref
  (`retarded_fields.py:923-1002`), so a self-consistency iteration that keeps
  the row count gets a fresh cloud.

## Tests (owner's rule: new tests only when strongly justified)

| test (new or extended in `1f4a456`) | verdict |
|---|---|
| `test_exact_source_cloud.py::test_cloud_matches_individually_translated_worldlines_and_roots`, `breathing` / `breathing_zero` × `cell_sigma` | **Keep.** This is the analytic child-worldline oracle. **Merge** a constant-width assertion into the existing `accelerated=True` case (breathing ≡ rigid on an accelerated centre, as in `breathing_vs_rigid.py`) instead of writing a new test. |
| `…::test_whole_parent_exclusion_in_field_and_mutual_provider[breathing]` | **Keep.** Extend it with a longitudinal offset once P1-a is fixed; as written it cannot detect P1-a. |
| `…::test_trial_cloud_and_endpoint_use_same_persistent_offsets_without_publication[breathing]` | **Keep.** It covers trial and endpoint without publication. |
| `…::test_cloud_fails_closed_for_proper_velocity` | **Keep.** Cheap guard. |
| `test_exact_source_reduction.py::test_reduction_preserves…` (× `width` × `breathing`) | **Keep.** Both widths exercise the floor on and off. |
| `…::test_partition_is_deterministic…`, γ-metric addition | **Keep.** |
| `…::test_exact_full_count_route…[breathing]` | **Keep.** It covers identity and the metadata through indexed views. |
| `test_exact_train_clouds.py::…checkpoint_is_bitwise_uninterrupted[breathing]` (slow) | **Keep.** This is the only breathing checkpoint test. Add an unreduced-train variant only after P1-b is resolved (or as a rejection test if trains are disallowed). |
| `test_resolved_knot_checkpoint.py` additions | **Keep.** Legacy archives without the new constants. |
| `test_simulation_options_magnetic_dipole.py` additions | **Keep.** Round trip and default off. |
| `test_sources2_scripts.py::test_summary_uses_study_metadata…` (2 × 3) | **Keep, trimmed.** The projection metric is now the convergence criterion. The three `metadata` variants test legacy-file detection; one legacy case is enough. |
| `test_sources2_scripts.py::test_serial_capacity_needs_one_slot` (× 4) | **Drop.** It tests a local wrapper-cap polling helper, not physics or numerics. |

No new tests are proposed. Each P1 fix should extend an existing test, as
noted above.

## Merge recommendation

**Mergeable for the reduced-source changes and as an opt-in experimental
model.** Defaults are bit-identical, and rigid clouds are unchanged since
`20c9807`. The reduced-route breathing is internally consistent, and the
earlier review's fixes are in place.

Before unreduced breathing is used for any result, either:

* fix **P1-a** (a common projection axis) and **P1-b** (per-bunch covariance
  for trains); or
* make the integrator reject unreduced breathing until they are fixed, which
  is cheap.

Address **P2-c** (incremental rebuild) before any 1,200-step breathing study;
otherwise the cost is impractical. Reword or fix **P2-b**.

The macro-count axis can be cited for **cross-bunch** observables. It should
not be cited for same-bunch kicks or for self-expansion (P2-a).
