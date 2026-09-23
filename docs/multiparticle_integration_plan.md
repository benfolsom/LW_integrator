# Many-particle magnetic dynamics and pseudogrid integration

Started 2026-09-21, after the v0.9.0 pair-solver release.

Current status, 2026-09-23: the optional many-particle and mixed RK4/RK3 mode
has been merged into development at `cce1cf0`. The feature branch additionally
supports preserved source increments in mixed stepping, with passing restart,
fallback, and bounded history tests. Matched reaction accounting is implemented
for mixed RK4/RK3 stepping. Internal error control is now connected to mixed
stepping, with restart, failure-recovery, and short post-arrival evidence.
The combined single-fit/reaction recording comparison is still running;
ordinary bunch input, GUI launch, and larger-system validation remain open.
The full solver is still the
default. The agreed target and chronological results follow below.

## Objective

Bring the full-spin motion and radiation-reaction calculations into practical
many-particle simulations. Keep the established two-particle results as a
regression reference. A merged pair solver was not completion of this work.

## Agreed default and merge sequence, 2026-09-23

The intended everyday mode is mixed stepping: selected active particles use
RK4, and passive particles use RK3 with a recorded RK4 fallback. Every particle
continues moving, evolving its spin, and supplying its own causal history.
All-particle RK4 remains the higher-order alternative. This is the agreed
implementation direction; the mixed method is still optional today.

First merge the validated optional mode into development after branch review.
Then add preserved source increments, matched reaction impulse and torque
accounting, and compatibility with internal error control. Complete ordinary
input and GUI support before default promotion. Compare both trajectories and
reaction contributions after newly evolved signals arrive, using the same
physical selection schedule at each tested recording resolution.

The pre-merge review passed 167 broader regression tests covering pair and
many-particle stepping, startup fitting, precision-preserved histories,
internal controls, restart, and CLI/GUI pair behavior. Review found and fixed
an invalid recording interval reaching the selection-clock division before
validation. The 60-test follow-up passed, including five invalid-interval
cases. Targeted type checking reports the same 197 existing errors on the
branch and development, with no additional errors after normalizing source
line numbers. This is a known repository type-check backlog, not a clean
type-check result. Sphinx work is deferred until the default-mode milestone.

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
- Fixed history recording and adaptive internal RK4/RK3 or DOP853 steps remain distinct.
  This work does not authorize arbitrary adaptive history recording.
- Full-spin reaction retains its conservation and strong-spin limitations.
- Do not infer many-particle radiation accuracy from source superposition tests.
- First tests are bounded regression cases, not long campaigns.

## Initial implementation record

The initial API is `core.momentum_center_particles`: `initialize_particles`,
`advance_particles`, and `configure_pseudogrid_reference`. Checkpoint list order
is persistent particle identity. There was no particle insertion, removal,
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
the documentation build passed with warnings treated as errors. The results
below supersede the earlier pending-run status.

## Completed reaction-on comparison and first reduced-motion experiment

The two `full_dipole_coupled` three-particle runs completed 250 and 500
recording steps, with 147 and 293 steps after signals from evolved motion
arrived. The archived [comparison](evidence/multiparticle_2026-09-22/reaction_comparison.json)
passed every predeclared position, spatial kinetic-momentum, spin, and
reaction-sector recording-resolution check. The largest relative sector
difference was $1.64\times10^{-5}$ for the dipole-squared impulse, relative to
the fine-run norm of that *same* small sector. This validates agreement between
two recording grids for this weak, symmetric case. It does not establish
conservation, strong-spin validity, or radiation accuracy in general.

An opt-in **reduced-motion** mode now uses the existing subset selector. The
selected particles take fourth-order Runge–Kutta steps (RK4); other particles
take cheaper, second-order midpoint steps. Every particle still has its own
accepted position and magnetic-moment history and remains an individual
retarded source. A checkpointed physical selection interval keeps the active
rotation at the same times when the recording grid is refined; restoring a
run reproduces the schedule. This is *not* the older bunch pseudogrid's weighted
source representation or passive-neighbor reconstruction.

The first implementation deliberately accepts only reaction-off, unpreserved
RK4 checkpoints without internal adaptive steps. Enable it on a prepared
many-particle checkpoint using `--pseudogrid-midpoint-active-count N` and
`--pseudogrid-selection-spacing-ns T`, where $T$ must be an integer multiple
of the recording step. The
existing `--pseudogrid-reference-active-count N` remains exact and is the
comparison control; it accepts the same physical selection interval for fair
comparisons. Neither mode changes the pair solver. Unsupported
reaction or preserved-source settings fail explicitly; there is no silent
fallback to a different model.

The reproducible [short comparison](evidence/multiparticle_2026-09-22/midpoint_comparison.json)
uses three particles, one active particle per step, and 50 steps, of which
30 are after evolved-source signals arrive. The schedules matched exactly.
Midpoint was about $1.3\times$ faster than *scheduled exact* RK4 in two timed
repeats. Final differences were $9.61\times10^{-9}\,\mathrm{mm}$ in position,
$3.88\times10^{-5}$ relative in spatial canonical momentum, and
$2.55\times10^{-8}$ relative in stored spin. However, the maximum mass and
spin constraint residuals were approximately 100 times larger than in the
exact run. A separate [irregular four-particle comparison](evidence/multiparticle_2026-09-22/midpoint_four_particle_comparison.json)
had 38 post-arrival steps, matching selections, about a $1.2\times$ speedup, and
the same pattern of larger residuals. These residuals measure how well the
numerical state preserves the model's required mass relation and the condition
that spin is orthogonal to kinetic momentum. They are not yet acceptable
production error bounds. Both speed measurements are small-case results; the
source evaluation remains all-to-all, so large-particle scaling is unproven.

With the same physical selection interval held fixed, halving the recording
step reduced the three-particle position, momentum, and spin differences
against the exact control by factors of approximately 3.7, 3.1, and 4.0.
The archived selection traces show the same 50 particle subsets at identical
physical selection times on both grids in each case.
The [fine three-particle result](evidence/multiparticle_2026-09-22/midpoint_three_particle_fine_comparison.json)
and [fine four-particle result](evidence/multiparticle_2026-09-22/midpoint_four_particle_fine_comparison.json)
also show smaller constraint residuals, though still substantially above exact
RK4. A first attempted grid comparison rotated the active set once per step,
which changed the physical approximation when the step was halved. Those
numbers were discarded; the fixed physical selection clock is necessary for
a meaningful refinement comparison.

Next: establish a physical constraint/error policy for passive updates, test
larger and less symmetric configurations, and determine whether the lower
integration order remains worthwhile with preserved single-fit histories and
reaction enabled. Only then consider reducing the source evaluations or
promoting the mode beyond experimental CLI use. Ordinary bunch input and GUI
launch remain separate integration tasks.

## Guarded passive updates and third-order candidate

The experimental reduced mode now checks dimensionless mass and spin-orthogonality
residuals after each passive candidate step. The initial checkpointed limit is
$10^{-6}$ for each, measured against the particle's own mass, momentum, and spin
scales in the simulation frame. This is a **provisional numerical guard**, not
a universal physical-accuracy tolerance. If a candidate fails, its particle is
recomputed with RK4 against the same frozen source histories, before any
history is published. A reduced candidate that reaches an invalid velocity
state is handled the same way; a failure in a stage shared with RK4 still stops
the step. Fallbacks and accepted reduced updates are counted per
particle in the checkpoint and reported per step. If even RK4 exceeds the
limit, the whole step fails without publishing a partial checkpoint. Existing
experimental reduced checkpoints without this policy keep their previous
unguarded behavior on resume; new reduced configurations record the policy.

The [guarded midpoint three-particle run](evidence/multiparticle_2026-09-22/midpoint_guarded_three_coarse.json)
recomputed 11 of 100 passive updates and retained about a $1.27\times$ speedup.
The [irregular four-particle run](evidence/multiparticle_2026-09-22/midpoint_guarded_four_coarse.json)
recomputed 12 of 100, retained about a $1.18\times$ speedup, and reduced its
constraint residuals. Its final trajectory, however, was slightly *farther*
from all-RK4 than the unguarded midpoint result. A constraint guard therefore
cannot be used as a substitute for a trajectory-accuracy test.

A third-order Runge–Kutta passive candidate is available as
`--pseudogrid-rk3-active-count N`, with the same physical selection clock and
guard. In the weak three- and four-particle coarse comparisons, it needed no
fallback, kept mass and spin constraints close to all-RK4, and was about
$1.16\times$ and $1.11\times$ faster, respectively. The
[three-particle](evidence/multiparticle_2026-09-22/rk3_guarded_three_coarse.json)
and [four-particle](evidence/multiparticle_2026-09-22/rk3_guarded_four_coarse.json)
results are archived with their [fine](evidence/multiparticle_2026-09-22/rk3_guarded_three_fine.json)
[counterparts](evidence/multiparticle_2026-09-22/rk3_guarded_four_fine.json).
Doubling charge and spin in both layouts still gave close agreement. In the
bounded four-particle counter-propagating case at $0.8c$, with doubled charge
and spin, the discrepancy in outgoing transverse *kinetic* momentum relative
to the exact result fell from $5.55\times10^{-7}$ to $1.06\times10^{-8}$ when
the recording step was halved. The selected particle subsets and physical
selection times matched on both grids. See the [coarse](evidence/multiparticle_2026-09-22/rk3_guarded_four_counter08_scale2.json)
and [fine](evidence/multiparticle_2026-09-22/rk3_guarded_four_counter08_scale2_fine.json)
evidence. This is a specific post-arrival relativistic check, not a general
high-$\beta$ validation.

A fivefold charge-and-spin scale was also attempted, but the **all-RK4 exact
control** stopped on its existing source-position reconstruction guard:
integrated-position error $1.166058\times10^{-8}\,\mathrm{mm}$ exceeded the
$10^{-8}\,\mathrm{mm}$ configured limit. That is not evidence of an RK3 failure,
and this setup needs its own history-accuracy investigation before comparison.

An eight-particle ring with two RK4-active and six RK3-passive particles also
completed after source signals from evolved motion arrived. Its radius was
$0.08\,\mathrm{mm}$, so adjacent particles were farther apart than the
$0.05\,\mathrm{mm}$ startup-history window; the initial $0.06\,\mathrm{mm}$
radius would have required unavailable source history in the very first step.
The [coarse](evidence/multiparticle_2026-09-22/rk3_guarded_eight_ring_coarse.json)
and [fine](evidence/multiparticle_2026-09-22/rk3_guarded_eight_ring_fine.json)
comparisons used identical physical selection times and needed no RK4 fallback.
Their final transverse kinetic-momentum discrepancies relative to all-RK4
were $4.29\times10^{-9}$ and $5.20\times10^{-10}$, respectively, a reduction by
about $8.24$ on halving the recording step. Each run was timed once and gave
about a $1.17\times$ speedup; that figure is indicative, not a stable
large-system performance estimate.

The next production gate is to test the third-order candidate with more
varied geometries, preserved single-fit histories, and reaction-on dynamics. The
current code deliberately rejects the latter two rather than silently using
an incompatible history or reaction policy. Source evaluation is still
all-to-all, so this motion-only optimization cannot by itself solve
thousand-particle scaling. The full RK4 solver remains the general default;
RK3 is the leading opt-in reduced-motion candidate, and midpoint remains a
useful lower-cost comparison rather than the recommended path.

## Preserved increments for mixed stepping, 2026-09-23

Mixed RK4/RK3 and RK4/midpoint now accumulate state changes separately from
the large reference state and preserve the corresponding dipole changes.
Each method uses its own integration weights. An RK4 retry starts from the
original accepted increments and replaces the rejected candidate; it cannot
add both candidate and retry increments to the history. The existing causal
publication and history-drift checks remain active.

Direct tests force failures at both an RK3 stage and its proposed endpoint.
For the affected particle, the accepted state, preserved components, and
published history exactly match an all-RK4 step. CLI restart tests cover both
ordinary and preserved increments. A single-fit history test uses the archived
weak three-particle setup, constrained startup fitting, and fitting-sample
spacing `0.01 / c` ns. It advances 120 steps to `ct = 1.2 mm`, samples evolved
source history, and reproduces the whole run exactly after JSON checkpoint
restart at step 60.

Two initially attempted history configurations were rejected, including in
their all-RK4 controls. The compact geometry with startup duration `0.05 / c`
ns and fitting-sample spacing `0.005 / c` ns rejected dipole drift
`8.191479e-11` in the mixed run and `8.191528e-11` in its control, against the
unchanged `1.271912e-12` budget. The larger weak geometry at fitting-sample
spacing `0.025 / c` ns reproduced the known wide-span refusal: `1.315229e-12`
for mixed stepping and `1.315212e-12` for all-RK4. Halving only the mixed
recording step gave `1.314911e-12` and did not repair that refusal. The passing
test uses the already archived `0.01 / c` physical fitting spacing. These
rejections are recorded as history-resolution limits, not RK3 failures or
successful radiation-accuracy checks.

Short checkpoint comparisons against development `cce1cf0` are byte-identical
after canonical JSON serialization for ordinary RK4, preserved RK4, preserved
DOP853, and preserved RK4 with coupled dipole reaction. This checks unchanged
paths; reaction in the mixed mode is still rejected. The next implementation
task is consistent reaction impulse and torque accumulation for the accepted
RK3 or RK4 stages, followed by a comparison after causal signals arrive.

Final regression validation: 182 tests passed in 176.68 seconds, including
the pair GUI tests, preserved histories, DOP853, internal controls, reaction
paths, and the new mixed-method restart and fallback cases. Black, Ruff, and
the diff whitespace check pass. Targeted type checking reports 196 existing
errors, one fewer than the merge baseline and no new error messages.

## Reaction accounting in mixed RK4/RK3 stepping, 2026-09-23

The mixed RK4/RK3 path now accepts the existing zero-spin `charge_ll`,
`full_dipole_coupled`, and legacy `full_dipole_rr` options. The equations used
at each stage are unchanged. Accepted RK3 stages use weights `(1, 4, 1) / 6`
for both the state update and accumulated reaction quantities. An RK4 fallback
removes the rejected third stage and uses `(1, 2, 2, 1) / 6` throughout. A
stage/weight count mismatch raises an error before a checkpoint is returned.

This covers applied impulse, applied torque, external-action mass exchange,
and each charge-squared, charge–dipole, and dipole-squared contribution. For
charge-only reaction it also covers outward radiation and bound-momentum
integrals. Forced stage and endpoint failures reproduce the all-RK4 particle
state, history, and every accumulated reaction quantity exactly, with and
without preserved increments. A manufactured quadratic time dependence tests
the integration weights independently of the physical equations. CLI tests
resume coupled-reaction mixed runs with their saved reaction records.

The first physical comparison uses three particles, preserved endpoint
histories, coupled reaction, one active particle, and 25 steps of `0.005 / c`
ns to `ct = 0.125 mm`. It includes sampling evolved source history and an
exact checkpoint replay. Total reaction impulse and torque meet a 1% relative
comparison limit against all-RK4 at the same recording step. This is a bounded
integration test; it does not establish radiation convergence or high-order
history-derivative accuracy. All eight impulse/torque comparisons passed,
including the separate charge-squared, charge–dipole, and dipole-squared
contributions. The maximum nonzero relative discrepancy is `3.930023e-4`
for dipole-squared torque; total impulse differs by `8.984207e-5`, and total
torque by `5.517900e-6`. Each denominator is the corresponding all-RK4 norm
at the same step size. Charge-squared torque is exactly zero in both runs.
There are five post-arrival steps and no RK4 fallbacks. The result is archived
in [mixed reaction evidence](evidence/multiparticle_2026-09-23/mixed_reaction_postarrival.json).

Validation: 107 regression tests passed; the explicitly selected slow
post-arrival test also passed in 217.24 seconds. The slow test is excluded from
the ordinary suite by repository policy and must be selected with `-m slow`.
Black, Ruff, and diff whitespace checks pass. Targeted type checking remains
at 196 existing errors, with no new messages in the changed code.

Next: run mixed stepping with coupled reaction and constrained single-fit
histories on the archived weak setup, compare recording resolutions at fixed
physical fitting and selection intervals, and then integrate internal error
control. The all-RK4 option remains available. The midpoint experiment remains
reaction-off; the intended default is RK4 for active particles and RK3 for
passive particles. No default promotion is made by this change.

## Combined history/reaction comparison and adaptive preparation, 2026-09-23

The next comparison is running on the Mac in
`/Users/benjaminfolsom/compute/mixed-reaction-20260923-GvDakv`, with `coarse`
and `fine` subdirectories. Both use the archived weak three-particle fixture,
coupled reaction, preserved increments, constrained startup fitting, one active
RK4 particle, and two passive RK3 particles. They advance to `ct = 2.5 mm`
in 250 and 500 recording steps. Both the fitting-sample spacing and active
selection interval are `0.01 / c` ns. Thresholds are unchanged. These runs
are pending evidence, not a passed accuracy check.
An automatic comparison process is waiting for both completion summaries and
will write `comparison.json` in that directory. It stops without claiming a
comparison if either physical run fails.

The validation script now accepts the integration method and mixed selection
settings explicitly and records each actual selection time and active set.
The comparison rejects missing, incomplete, or unequal selection schedules.
This prevents a change in which particles receive RK4 from being mistaken
for a recording-resolution effect. It still checks total and separate
charge-squared, charge–dipole, and dipole-squared reaction quantities. Earlier
archived DOP853 results are a different integration method, not an all-RK4
control for these runs.

Internal adaptive stepping is **not yet enabled for mixed runs**. The current
controller uses DOP853; removing the compatibility guard would not provide
adaptive RK4/RK3. A new numerical building block,
`core/preserved_rk_error.py`, instead compares one full RK3 or RK4 step with
two half steps of the same method. It returns the two-half-step result and
an error estimate, without publishing any source history or changing defaults.
State and appended reaction quantities use identical weights. Small errors
are calculated from preserved increments, rather than subtracting rounded
absolute states. No extrapolated state is accepted.

The initial 34 focused tests pass: nonlinear analytic solutions verify the
expected error order and sign, tiny-increment tests check matching reaction
weights, invalid rates and domain failures leave inputs unchanged, existing
DOP853 tests remain passing, and comparison tests reject incompatible schedules.
This validates the building block, not the physical adaptive integration.
The broader regression selection passed 123 tests in 202.16 seconds, with
one slow physical test deselected by the repository's normal test policy.
Black, Ruff, and the diff whitespace check pass.

Remaining sequence:

1. Analyze the completed two-resolution runs, including actual selections,
   fallback counts, trajectories, and each reaction contribution.
2. Connect the RK3/RK4 estimate to the existing physical error scales and an
   internal accept/reject loop. Keep the selected particle set fixed throughout
   a recording interval and publish histories only after all particles pass.
   Verify that conversion to physical error scales does not round away small
   preserved estimates when constructing a nearby comparison state.
   Preserve the current constraint checks and retry a rejected passive method
   from its original accepted state, including its reaction integrals.
3. Test rejected trials, RK4 fallback, checkpoint replay, and a physical
   reaction-on comparison before exposing the adaptive combination. Step
   doubling costs nine RK3 or twelve RK4 right-hand-side evaluations per trial;
   it is an accuracy feature, not an assumed speedup. Error estimates require
   smooth intervals; history-boundary behavior needs explicit verification.
4. Complete CLI/GUI and general particle-input compatibility, then revisit
   defaults and merge readiness. Do not change Sphinx documentation yet.

## Adaptive mixed stepping connected, 2026-09-23

The preceding error-estimator preparation is now connected to the native
solver and prepared-checkpoint CLI. Active particles use adaptive RK4, and
passive particles use adaptive RK3. Each method compares one trial step with
two half steps, keeps the latter when its estimated error is acceptable, and
reduces the internal step otherwise. Particle selection is held fixed during
each recording interval. Only the final accepted endpoint is added to source
history; internal trials and half steps are not published.

An RK3 domain failure, exhausted accuracy-trial budget, or excessive endpoint
constraint residual triggers an adaptive RK4 retry from the original accepted
increments. The retry replaces both the particle state and every reaction
integral. If the retry fails, no particle's history or selection counters are
published. Domain/history errors other than the existing invalid-velocity
failure remain explicit errors; they are not disguised as convergence.

This requires preserved increments and supports reaction off, coupled dipole
reaction, and the legacy full-dipole reaction. Internal control still rejects
`charge_ll`, whose additional radiation records need a separate adapter.
Fixed-step mixed RK3 still supports that charge-only option. Midpoint remains
fixed-step and reaction-off. No integration or tolerance defaults change.

The new RK error calculation preserves differences when constructing a nearby
comparison state and transforming spin to the kinetic-momentum rest frame.
Synthetic tests with errors of `1e-23` demonstrate why this matters: ordinary
absolute-state subtraction can erase the estimate. Decimal arithmetic is
limited to error bookkeeping, not field evaluation or the reaction solve.
The existing DOP853 error path is unchanged in this commit; the new check does
not claim sub-float64 potential accuracy.

For prepared, precision-preserved RK4 data, the existing CLI controls combine:

```sh
python -m lw_integrator.nonlinear_particles --checkpoint initial.json \
  --output continued.json --step-ns <recording-interval-ns> --steps 10 \
  --pseudogrid-rk3-active-count <active-count> \
  --pseudogrid-selection-spacing-ns <selection-interval-ns> \
  --internal-error-settings tolerances.json
```

The tolerance file uses the existing six explicit absolute/relative physical
scales. On resume, omit new selection options and retain the saved tolerances.
Internal adaptivity is an accuracy option, not an assumed performance gain.
It requires 10 RK3 or 13 RK4 right-hand-side evaluations per trial, including
the endpoint domain check. The general many-particle GUI remains pending.

### Completed short continuation

The fixed-step coarse run completed 250 steps to `ct = 2.5 mm`, including
147 steps sampling evolved histories, without any RK4 fallbacks. Its existing
constrained single-fit histories provided the starting point for a diagnostic
fork: three further recording intervals of `0.01 / c` ns using adaptive
all-RK4 versus adaptive mixed RK4/RK3. Only accumulated reaction records were
reset, so old totals cannot hide errors in newly calculated reaction.

Both continuations completed to `ct = 2.53 mm`, sampled evolved histories,
and required no fallback. The largest relative difference among the eight
impulse/torque comparisons was `3.633748e-10` in charge–dipole impulse.
Total impulse differed by `3.746574e-14`, and dipole-squared impulse by
`1.026588e-11`. Each denominator is the corresponding adaptive all-RK4 norm
over these three intervals, not a finer recording grid. Principal trajectory
components agree to their stored floating-point resolution. These very small
differences are bounded compatibility evidence in a weak case, not an
equivalent claim of physical accuracy or complete radiation convergence.
The separately tracked external-action mass exchange has a larger relative
difference, `1.505328e-3`: absolute difference `5.392518e-28` against reference
norm `3.582288e-25` in its native units. This is recorded separately rather than
hidden by the much closer impulse/torque comparisons. The spin-state difference
is `2.349525e-33` relative to its all-RK4 norm, again limited by the stored
floating-point components, not a physical-accuracy claim.

The explicit diagnostic scales were absolute `1e-8` in each group's native
units and relative `1e-6`. Neither branch rejected an internal trial in this
weak continuation. Separate manufactured-rate tests force internal rejection
and verify that every reaction contribution is integrated correctly while
exactly one history sample is published per requested recording interval.

Reproduction utility: `scripts/validate_mixed_internal_continuation.py`.
The compact result is archived in
[adaptive continuation evidence](evidence/multiparticle_2026-09-23/mixed_internal_continuation.json);
its metadata identifies the full checkpoint artifacts.

### Remaining work

Final validation: 157 selected regression tests passed in 421.37 seconds,
including mixed internal fallback/restart, rejected-trial accounting, preserved
histories, fixed-step multiparticle behavior, existing DOP853, and pair GUI
tests. One slow test was excluded by the repository's normal policy; the
separate post-arrival continuation above supplies new physical evidence.
Black, Ruff, and diff whitespace checks pass. An initial regression run caught
an outdated expected error-message string; that assertion was corrected,
rerun, and included in the clean final run. No physical tolerance was relaxed.

1. Finish the running 500-step fixed-grid comparison. The short adaptive
   continuation does not replace that recording-resolution check.
2. Implement ordinary many-particle input and GUI launch controls, then assess
   default promotion and merge readiness. Retain the separate larger-system,
   strong-spin, and radiation-conservation limitations.

Targeted type checking has the same 196 errors as `4fd0760`, with no additional
messages after source-line normalization. This is still a known repository
backlog, not a clean type check.
