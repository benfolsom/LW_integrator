# Many-particle magnetic dynamics and pseudogrid integration

Started 2026-09-21, after the v0.9.0 pair-solver release.

## Objective

Bring the full-spin motion and radiation-reaction calculations into practical
many-particle simulations. Keep the established two-particle results as a
regression reference. A merged pair solver was not completion of this work.

## Implementation sequence

1. Generalize source collection, simultaneous history publication, preserved
   state increments, and reaction accounting to arbitrary particle counts.
   Sum source potentials and their derivatives before evaluating each particle's
   nonlinear response; do not sum separately evaluated pair forces. Each source
   retains its own retarded time. Exclude the observer's own source.
2. Verify pair equivalence, three-particle interactions, exact restart, input
   immutability on failure, analytical derivative sums, and per-particle reaction
   records. Extend fixed-clock history selection and physical drift budgets.
3. Connect the existing pseudogrid subset selector in **exact reference mode**:
   record its selections, but continue solving every particle. This tests
   scheduling and restart without introducing a physical approximation, and
   provides no speedup. It is not reduced pseudogrid support.
4. Design and test passive-particle motion and spin updates against that exact
   reference, initially with reaction off. Do not copy charge-only weighted
   updates onto constrained spin states without validating the resulting states
   and source histories. Keep identities when selections rotate.
5. Validate reduced histories with magnetic reaction, including high derivatives
   after newly evolved signals arrive. Compare trajectories, spin, and reaction
   impulses independently; trajectory agreement alone is insufficient.
6. Connect ordinary bunch initialization, CLI/GUI controls, checkpoint options,
   and sweep workflows. Profile exact and reduced runs before choosing additional
   optimization. The existing two-particle CLI remains a pair-only interface.

## Boundaries

- Preserve the pair checkpoint model and reject many-particle checkpoints in the
  pair entry point. Use a separate model identifier for many-particle checkpoints.
- Keep exact pair arithmetic unchanged when there is only one non-self source.
- Published histories remain immutable; an unsuccessful step publishes nothing.
- Fixed history recording and adaptive internal DOP853 steps remain distinct.
  This work does not authorize arbitrary adaptive history recording.
- Full-spin reaction retains its conservation and strong-spin limitations.
- Do not infer many-particle radiation accuracy from source superposition tests.
- First tests are bounded regression cases, not long campaigns.

## Current implementation

The initial API is `core.momentum_center_particles`: `initialize_particles`,
`advance_particles`, and `configure_pseudogrid_reference`. Checkpoint list order
is persistent particle identity. There is no particle insertion, removal,
macroparticle weighting, or reduced passive update in this first implementation.
The numerical engine is shared with the pair solver, including local reaction
and history reconstruction. Validation results are recorded as work proceeds.

The prepared-checkpoint command is:

```sh
python -m lw_integrator.nonlinear_particles --checkpoint initial.json \
  --output continued.json --step-ns <recording-interval-ns> --steps 1
```

Use `--capabilities` for the supported modes. The history, drift-budget,
reaction, and internal DOP853 options reuse the pair CLI. Optionally add
`--pseudogrid-reference-active-count N` once to attach a checkpointed reference
schedule. Omit that option on resume; the saved selection counters continue.
Every particle still receives an exact update. The selector currently treats
the collection as one group, not separately configured rider/driver bunches.
The existing pair GUI cannot launch this new runner yet.

## Verification on 2026-09-21

- Focused regression suite: **111 passed** (94.94 seconds), covering the new
  many-particle tests, existing pair stepping, history options, preserved source
  precision, internal step control, pair CLI/GUI tests, and adaptive pair tests.
- After adding an explicit many-particle rejection to the pair-only adaptive
  controller, reran the many-particle and adaptive pair files: **30 passed**
  (49.56 seconds). These overlap the suite above; do not add the counts.
- Two-particle RK4 and preserved DOP853 results match the pair API exactly.
- Three-particle tests cover reaction records, analytical source sums, internal
  adaptivity, automatic history selection, relative/absolute drift budgets,
  checkpoint restart, failure atomicity, and CLI continuation.
- Four-particle results agree under particle reordering within the test's
  floating-point tolerance (relative 2e-14, absolute 1e-24 in native state units).
- Reference subset scheduling changes neither trajectory nor physical checkpoint
  data, and its scheduling counters survive restart. Invalid counters are rejected.
- Ruff, Black, whitespace checks, and the Sphinx warnings-as-errors build pass.
  Repository-wide type checking and the complete unit suite were not rerun.

These are short startup checks, not a many-particle accuracy campaign. The next
physics check should advance three particles until signals from newly evolved
motion reach the others and compare recording resolutions at a fixed physical
history-fitting span. Then use that exact result to test passive motion/spin
updates. Reduced pseudogrid, ordinary bunch input, GUI launch, and production
scale performance remain open; no speedup or conservation closure is claimed.

Local test logs: `/tmp/lw-multiparticle-final-tests.log`,
`/tmp/lw-multiparticle-guards-tests.log`, and `/tmp/lw-multiparticle-docs.log`.

## Three-particle continuation on 2026-09-22

Work was resumed from the Dell, with all calculations on the Mac. No changes
were made to the separate aneutronic worktree. Added reproducible run and
comparison scripts in `scripts/validate_multiparticle_postarrival.py` and
`scripts/compare_multiparticle_postarrival.py`.

The first comparison uses three identical, initially stationary spinning
particles on a circle of radius 0.6 mm, from the existing regression fixture.
Charge is 0.003 times the native light-speed constant, mass is 1 amu, and initial
rest spin is 0.01 times that constant along z in native units. The imposed smooth
startup lasts 1/c ns; this is a driven preparation, not an isolated equilibrium.
Reaction is explicitly off to isolate mutual-source and history behavior.
The target time is 2.5/c ns, past the first arrival at approximately 1.04/c ns.
Both recording grids use preserved DOP853 and the same physical fitting span.

The default relative moment-reconstruction allowance remains 1e-10, giving
an absolute allowance of 1.271912e-12 native dipole units per particle. No guard
was relaxed and no failed checkpoint was silently continued.

- With fitting-sample spacing 0.025/c ns, the 100-step and 200-step runs stopped
  after 13 and 25 accepted steps, respectively, before evolved signals arrived.
  The rejected endpoint drifts were 1.554395e-12 and 1.302818e-12.
- With shorter fitting-sample spacing 0.01/c ns, the 250-step coarse run reached
  113 accepted steps, including 10 steps sampling evolved source intervals.
  It then rejected drift 1.283570e-12. The paired 500-step run uses half the
  recording interval but the identical physical fitting span.
  The fine run subsequently stopped after 227 accepted steps, including 20
  post-arrival steps, rejecting the same drift 1.283570e-12. Both runs are
  incomplete; the identical refusal is not a resolution pass.

At nine matching post-arrival times for each of three particles (27 samples),
accepted velocities differ by 6.134378e-16 relative to the combined fine norm.
Preserved changes in the dipole tensor differ by 4.445930e-15 using their fine
change norm, not the large initial moment, as denominator. The combined absolute
position difference is 4.375527e-22 mm. Near-machine-precision agreement in this
symmetric, weak-coupling test does not establish that level of physical accuracy:
the same reconstruction bias can be present on both grids.

This is evidence against particle-step error as the immediate cause of refusal.
Both grids construct their histories from nearly identical physical samples and
encounter the same reconstruction error. No further timestep-only refinement is
scheduled. The next targeted check is the fitted history against those accepted
samples, including how enforcing continuity accumulates endpoint drift. The
existing source-history guard remains active while reduced pseudogrid design
continues; successful startup tests are not relabeled as completed long runs.

Small machine-readable results are archived under
[`evidence/multiparticle_2026-09-22`](evidence/multiparticle_2026-09-22), including
all four run summaries and the matched post-arrival comparison. Runtime was
115.96 seconds for the shorter-span coarse run and 422.13 seconds for its fine
partner; these are elapsed run times, not controlled performance benchmarks.

The comparison tool distinguishes completed endpoint comparisons from matched
accepted-history comparisons on incomplete runs. Endpoint momentum excludes
the rest-energy component from its normalization. Partial comparisons report
position, velocity, and preserved moment changes, not reconstructed derivatives,
kinetic momentum, radiation accuracy, or successful completion. Future harness
runs also retain per-step state and momentum records; the already-running
checks predate that addition.

Nine analysis-tool tests pass, covering incomplete-run rejection, mismatched
initial conditions, physical-time matching, and momentum normalization. Ruff
and Black pass for the added scripts and tests. This continuation does not yet
enable reduced pseudogrid updates. The next decision is based on the matching
post-arrival histories, not another automatic round of timestep refinement.

Full checkpoints and logs remain outside the synchronized tree on the Mac at
`/Users/benjaminfolsom/compute/multiparticle-evidence-20260922`.

## Startup reconstruction fix under live verification

Read-only decomposition of the saved shorter-span histories isolated the main
cause. At the last published coarse endpoint (ct = 1.08 mm), particle zero's
drift is 1.248475e-12. The propagated startup correction has norm 1.243167e-12;
later fitting-window corrections have combined norm 5.673308e-15. These tensor
contributions can cancel, so their norms are not an additive error budget.
The unconstrained fit itself misses that accepted endpoint by only 6.769191e-19.

The initial fit did not know that the moment and its first three derivatives
must join the constant inertial past. Overwriting those four fitted coefficients
afterwards enforces the join, but introduces a cubic correction that persists
and grows across later intervals. This is a reconstruction approximation issue,
not evidence of a missing force term or a particle trajectory blowup.

Implemented candidate: fit the first sample window with those four conditions
built into the least-squares problem. Fit powers four through ten to the same
eleven accepted samples. Rational fitting weights and compensated sample
arithmetic preserve small changes. Outside windows anchored at the inertial
boundary, the old fitting calculation is unchanged. Continuity checks, sample
spacing, and drift limits are unchanged.

- Offline reconstruction with the actual candidate map reduces the maximum
  coarse saved-data drift from 1.248475e-12 to 1.299458e-14, about 96-fold.
  This is not a new self-consistent trajectory result.
- The candidate is recorded as `startup_dipole_fit="constrained"` in history
  format v8. Existing histories still write their previous formats and retain
  `interpolate`; old trajectories are not silently reconstructed differently.
- The benchmark script exposes `--startup-dipole-fit constrained` for fresh
  runs. General-purpose default promotion is pending the live comparison.
- **119 focused tests passed** in 44.89 seconds, including polynomial
  coefficients through degree ten, fourth/fifth derivatives of a nonpolynomial
  reference, exact restart/append, unchanged higher coefficients outside startup,
  pair-history controls, and multiparticle stepping. The analytic nonpolynomial
  derivative check requires relative error below 1e-6 over its sampled interval.
- Broader testing also found an error-message precedence regression introduced
  by the earlier pair-model guard. Reordered the checks without removing either
  safety condition; the affected regression now passes.
- An additional prepared-v8 multiparticle CLI check passes and preserves both
  the fitting policy and budget policy; the candidate test file now has 13
  passing cases. These overlap the broader suite, so the counts are not added.
  The Sphinx warnings-as-errors build also passes.

Fresh 250-step and 500-step replays are running on the Mac with the original
0.01/c-ns fitting spacing and original drift budgets. The coarse replay has
passed 210 steps, including 107 post-arrival steps, beyond the previous 113-step
refusal. The fine run has also reached evolved signals. Completion and endpoint comparison remain
pending. An unattended watcher writes `constrained-comparison.json` in the
evidence directory when both runs finish; failed runs are explicitly reported
as incomplete, with a partial-history comparison if available.

Inspection scripts: `scripts/inspect_multiparticle_history_drift.py` and
`scripts/probe_startup_history_fit.py`. Archived decomposition and offline
candidate evidence are in `docs/evidence/multiparticle_2026-09-22`.
No magnetic-reaction accuracy or reduced-pseudogrid validation is claimed by
these reaction-off replays.

## Completed reaction-off comparison

Both constrained-startup replays completed with the original drift limits:
250 coarse steps (147 post-arrival) and 500 fine steps (293 post-arrival).
Elapsed times were 506.08 and 1939.07 seconds. The endpoint comparison passed
the predeclared position, spatial-momentum, and spin criteria. The spatial
kinetic-momentum difference is 5.169762e-16 relative to the fine spatial norm;
the absolute position difference is 9.529121e-22 mm. This very close agreement
in a weak, symmetric case is not an absolute physical-accuracy estimate.

The machine-readable result is archived as
`evidence/multiparticle_2026-09-22/constrained_comparison.json`.
The earlier running-status notes above describe the sequence of investigation.
Next: expose the checkpointed startup selection through the CLI, test fresh
selection and unchanged restart behavior, and run the separate reaction-on
comparison. Reduced pseudogrid and default promotion remain open.

## CLI integration and reaction-on continuation

Committed the multiparticle engine, constrained-startup implementation, and
completed reaction-off evidence as `05ca6dc` on `feature/multiparticle-full-spin`.
No push or merge was performed.

The subsequent CLI work exposes `--startup-dipole-fit constrained` for both
the pair and many-particle runners. Omission preserves the existing policy.
Changing it requires zero accepted steps and histories ending at the declared
inertial boundary. Resume accepts the same policy but rejects changes. The
benchmark now uses this shared configuration function instead of its own
conversion. **56 focused tests passed** in 3.80 seconds, including policy
selection, input preservation, refusal to change evolved data, CLI selection,
and separate reaction-sector comparison.

Fresh reaction-on coarse/fine runs use `full_dipole_coupled`, constrained startup,
250/500 recording steps, the same 0.01/c-ns physical fitting spacing, and the
original drift budgets. Evidence goes to `reaction-coarse` and `reaction-fine`
in the existing Mac evidence directory. These are independent fresh starts,
not reaction-mode changes to the completed reaction-off trajectories.

Before launch, the comparison was extended to total and charge-squared,
charge–dipole, and dipole-squared applied impulses and torques. Each contribution
uses its own fine-result norm and a 1e-2 relative comparison target; if the fine
norm is exactly zero, only an exactly zero difference passes. This avoids
hiding a small contribution behind a larger total. The test is numerical
recording-resolution agreement, not a closed conservation proof.

Both detached reaction jobs were verified running, with 11 accepted steps each
at the initial status check. A detached watcher will write
`reaction-comparison.json` in the evidence directory after both stop. It reports
incomplete or failed analysis explicitly rather than treating early stops as
success. The final focused test rerun passed all 56 tests in 3.30 seconds, and
the documentation build passed with warnings treated as errors. No reaction-on
result or reduced-pseudogrid validation is claimed yet.
