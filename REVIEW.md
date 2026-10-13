# Review: `review/startup-guard` (c5ff7ad) against `development`

Reviewer: independent review, 2026-10-13. Scope: the startup-history preflight
(`core/startup_history_guard.py`), its wiring (runner, CLI, testbed, config) and
the ~90 test call sites that now pass `allow_incomplete_startup_history=True`.
The review branch was not modified. Probe scripts are in
`review_probes/startup_guard/` (run from a checkout of the review branch).

**Recommendation: do not merge as is.** The motivation is right: COLD_START
silently drops interaction, and a preflight is a good place to say so. But the
check models a mechanism the integrator does not use. It rejects a shipped
example and the AGENTS.md default test geometry, and it applies to
APPROXIMATE_BACK_HISTORY a criterion that does not describe that mode. It also
passes a configuration where COLD_START drops the rider's entire interaction,
and its |F| measure hides net-impulse errors. Because of the false positives,
about half the overrides added to tests (47 of 91 tests) are needed only to
get past the check, so the guard is not exercised where it matters. Suggested
path: (1) make it warn-only, (2) base it on the integrator's real COLD_START
gate, (3) drop ABH from the error path, (4) add an absolute floor and a signed
measure, (5) then make it an error. Details below.

Branch hygiene: c5ff7ad is based on 94dbb48. `development` has since moved
(10 commits at review start, now `e9feee2`), including CHANGELOG and runner
changes. Rebase before merging.

## Evidence summary

| Probe | Result |
|---|---|
| AGENTS.md default B2B (head-on, 1000 mm, 1200 steps, closing-scale h, 0.01 mm offset, N=200, gamma 1.05/2/100) | PASS, worst ratio 6e-6 to 2e-5 |
| Same geometry with a 10 mm offset at beta=0.4 (`tests/physics/test_startup_inclusion_continuity.py::run_gate`) | **FAIL**, ratio 3.45e-3 |
| `configs/run_configs/example_b2b_counter_propagating_proton_bunches.json` (COLD_START) | PASS, 1.2e-5 |
| `configs/run_configs/example_b2b_relativistic_proton_stationary_lead_ion.json` (ABH) | **FAIL**, ratio 1 |
| Long driver (41 protons over 2 m), rider 50 mm ahead, integrator run | Rider force exactly 0 for the whole run; guard's rider-side estimate 2.8e-3 |
| Pipe occluder that always contains both bunches | **FAIL** (gate), at 1 m and at 100 m separation |
| `tolerance=nan` / `inf` | Always PASS |
| Guard spy over the 91 overridden tests | 47 would fail, 5 pass, 39 skipped (not B2B, empty bunch, or inertial) |
| Runtime, N = 1e2 / 1e4 / 2e5 per bunch | 1.6 / 1.2-1.7 / 2.3 s, nearly flat in N |
| Tests: guard test plus all touched test files plus `test_cli.py` and `test_cli_gui_parity.py` | 324 passed, 1 xfailed, 1 failed. The failure is `test_magnetic_dipole_path_is_identical_across_forced_python_and_numba_kernels` (1-ulp numba/Python difference). It also fails on the base commit, so it is pre-existing and environment-dependent. |

## P0: wrong default behaviour

### P0-1. APPROXIMATE_BACK_HISTORY is rejected by a criterion that does not apply to it; a shipped example now fails
`core/startup_history_guard.py:312-314`, `core/integration_runner.py:4172`

ABH has no missing-history window. On every step it bypasses sampled chrono
matching for external sources and uses the *current* source state with a
constant-velocity retarded-distance correction (`core/equations.py:2615`,
`_compute_approximate_retarded_distance`; `core/potential_inclusion.py:233`;
the docstring at `core/types.py:162,177`). Before t0 that is the same
inertial-extrapolation assumption the guard itself exempts under
`INERTIAL_PREHISTORY`. Any ABH error is spread over the whole run (it ignores
acceleration and uses the instantaneous geometry), and `∫_{t0}^{t_c}|F|dt` does
not measure it.

Failure scenario: `lw_integrator --testbed-config
configs/run_configs/example_b2b_relativistic_proton_stationary_lead_ion.json`
now raises `StartupHistoryError` ("rider[2] -> driver[0] ... missing-history
interval 3.336 ns of a 0.00125 ns run ... miss 1"). The stationary ion's
nominal horizon is `(steps-1)*h*gamma_ion`, i.e. 1.25 ps. ABH gives that ion
the rider's field from step 0. Evidence: `probe_configs.py`; also
`probe_guard.py` section I (the unit-test pair fails identically under ABH).

Fix: exclude ABH from the error path, or replace its check with one that
bounds ABH's actual approximation (for example, acceleration over `R/c`).

### P0-2. Purely relative metric: short, truncated or well-separated runs fail at ratio 1
`core/startup_history_guard.py:231-236, 275, 334`

The ratio has no absolute floor. If the run ends before `t_c`, missing equals
total and the ratio is 1, however weak the field. In `probe_guard.py` section B
the same 1000 mm head-on pair passes at 650 steps and fails at 500 steps
("1.788 ns of a 1.602 ns run ... miss 1"). In that case the missing force is
1/R² at a metre and contributes almost nothing. This is the main reason 47
test call sites need the override (steps=2-5 smoke tests, 0.3 ps driver-train
smoke runs, `test_same_bunch_plummer`). It also catches preview runs, staged
runs and the first chunk of checkpointed runs.

The AGENTS.md defaults pass only when the transverse offset is small. With a
10 mm offset at beta=0.4, the repository's own default-geometry test
(`run_gate`: 1000 mm, 1200 steps, closing-scale h) fails at 3.45e-3. A
larger impact parameter weakens the encounter impulse, so the far-field
approach dominates the denominator.

Fix: normalise against a run-independent scale, such as the impulse of the
full straight-line encounter or a physically relevant kick (`Δp/p`,
`ΔK/K`). Add an absolute floor so a run whose total interaction is negligible
cannot fail.

### P0-3. The visibility-gate test fires whenever the source is visible at all
`core/startup_history_guard.py:224-228`

`gate_hits = any(visible & (s_ret < t_j0) & (mag > 0))`, and `mag` is already
masked by `visible`, so the test is true whenever the source is visible at any
retarded point before t0. That includes the normal case of a pipe that
contains both bunches throughout. The docs describe a *window* (a gate that
opens or closes inside the missing region). The code tests visibility, not a
transition. It also ignores epsilon, so a 1e-12 contribution fails the run.

Failure scenario (`probe_guard.py` D): the same 1000 mm pair passes without
geometry and fails with a 50 mm-radius pipe enclosing both bunches. At 100 m
separation and a 10x longer run it still fails. In practice every
geometry-enabled COLD_START B2B run is rejected.

Fix: flag only a visibility *change* on the sampled `s_ret < t_j0` segment,
or a missing-region contribution above tolerance whose visibility differs
from the visibility after `t_c`.

### P0-4. Driver-train coasting prehistory is ignored
`core/startup_history_guard.py:287-345`; `core/integration_runner.py:1515-1533, 4147, 4233, 4442-4450`

With `driver_train.enabled` and `prehistory_steps > 0`, the sampled path already
prepends an inertial coasting history to *both* bunches
(`_build_coasting_history`). It also moves `origin_*`, which the COLD_START
travel gate uses, to the start of that history. The guard never receives
`prehistory_steps` and evaluates the missing window from the active t0, so it
rejects driver-train runs whose prehistory covers `R/c`. The new doc page
(`docs/source/startup_history.rst:64-78`) says giving the sampled path a
prehistory "is not a cheap change". Driver trains already do exactly that.

Fix: pass `active_start * h_step` (per bunch, lab time `gamma*h`) into the
check and shift `t_c` accordingly. Correct the doc paragraph.

### P0-5. GUI drops the override and tolerance
`lw_integrator/gui_config_mixins.py:1651` (`SimulationOptions(...)` built from
widgets) does not pass `allow_incomplete_startup_history` or
`startup_history_tolerance`, and there is no widget for them. If a JSON with
`"allow_incomplete_startup_history": true` is loaded and run from the GUI, it
reverts to `False` and raises. The CLI honours it, so the two paths diverge.
Fix: carry both fields through `self.options` as is done for
`source_history_representation`. Add a CLI/GUI parity test.

## P1: wrong physics

### P1-1. The modelled window is not the integrator's COLD_START gate, so false negatives are severe for extended source bunches
`core/startup_history_guard.py:196-206, 231`

The guard treats each pair as missing for its own `tau_c(i,j)`. The integrator
works differently. `core/potential_inclusion.py:196-210`, `core/equations.py:2535-2595`
and `_should_apply_external_forces`/`_compute_gating_threshold`
(`core/equations.py:1116-1240`) apply one all-or-nothing gate per observer,
covering every source at once. The gate opens only when
`travel >= max_j beta_avg R_j/(1 - beta_avg·n_j)`, so the *farthest* source
controls when the nearest source is switched on. Once a source has been
crossed, `beta·n -> +beta` and its threshold grows to `beta R/(1-beta)`, so
the gate can stay closed.

Evidence (`probe_long_bunch.py`, `probe_single.py`): rider at z=0, x=1 mm,
beta=0.866. With one driver proton at 50 mm, the rider's `Px` first changes at
0.06 ns (step 6). With 41 driver protons from 50 to 2050 mm, the rider's
`Px` stays exactly 0 for the whole run (320 and 1200 steps), and `Pz` is
unchanged throughout the 320-step run, although by 3.19 ns the rider has
crossed about 33 drivers. The guard's rider-side
estimate is 2.8e-3 ("driver[0] -> rider[0] ... miss 0.00278"). In
`probe_guard.py` section C the guard ratio is 1.6e-5, against about 0.9 when
the same integrand is taken over the integrator's gate window. In the
integrator runs the guard still raised (ratio 1), but only on the reverse
direction (far driver observing the rider), not because it saw the
rider-side loss. A longer run that passes the reverse direction keeps this
blind spot.

The integrator's behaviour (a gate that stays closed after a crossing) is a
pre-existing problem in its own right and should be checked separately. The
guard must model the gate the integrator actually applies, not an idealised
per-pair light cone. Simplest correct option: compute the gate-open time per
observer with the same `_compute_gating_threshold` formula on the
straight-line trajectories, and integrate every source's |F| up to that time.

Also: at `beta_avg = 0` the integrator's threshold is 0, so the gate opens at
t0 and uses the clamped first sample (a frozen source), not "no field". The
error there is `F_frozen - F_inertial`, not `F_inertial`. This is minor at low
beta, but it is a further sign the error model is not the integrator's.

### P1-2. |F| impulse hides net-impulse and energy errors
`core/startup_history_guard.py:12-15, 146-151`

`∫|F|dt` is described as conservative because it "cannot cancel by sign". For
the observables this project reports (net `ΔK`, `Σ N_i Δp_i`), the reverse
holds: the net longitudinal impulse of a crossing is a small residual of large
opposite-sign halves. COLD_START removes only the approach half, so a small
|F| fraction can be a large share of the net effect.

Evidence (`probe_signed.py`, `run_gate` geometry: beta=0.4, 1000 mm, 10 mm
offset, 1200 steps): |F| ratio is 3.45e-3. The signed missing `F_z` impulse
is -4.67e-6 against a signed run total of -3.52e-6, i.e. **133% of the net
longitudinal impulse** (`F_x`: 9e-5). With `--startup-history-tolerance 0.01`
this run passes, and its energy ledger is dominated by the dropped approach.

Fix: also bound the signed missing impulse per component against the signed
total (or against `|Δp|` of the full encounter). Fail on either measure.
State in the docs that a pass is not an energy-accuracy certificate.

### P1-3. Same-bunch, wall/image, external-field and dipole sources are silently unchecked
`core/startup_history_guard.py:314-320`

Non-B2B modes return `checked=False` and only the report records it. Nothing
is logged and no warning is issued. `demo_single_run_with_monitoring.json`
(CONDUCTING_WALL, COLD_START) and the native-PIC examples report "PASS" in
practice. Same-bunch space charge is documented as having "its own explicit
start-up treatment". It is the source the AGENTS.md macroparticle notes care
about most, and `test_same_bunch_plummer`/`test_self_space_charge_energy`
needed the override because of the cross-bunch part only. Log one line
whenever the check is skipped for a COLD_START/ABH run, so the user knows it
was not evaluated.

### P1-4. Straight-line extrapolation ignores external fields and bound motion
Undocumented limitation: there is no handling of `external_field`, cavity or
dipole fields, or bound pairs. For a slow bound pair (`probe_guard.py` G,
beta=1e-3, R0=1 mm, run 100 R0/c) the guard reports 1.0e-2, which is
plausible. For orbiting or field-steered sources the straight-line `t_c` and
geometry can be arbitrarily wrong. Say so in the error and docs, or skip with
a logged reason when `external_field`/`magnetic_dipole` is active.

## P2: robustness

- **P2-1 Subsampling can miss the worst observer.** `core/startup_history_guard.py:254`.
  Observers are strided by array index, not by risk. `probe_guard.py` H
  happened to catch a planted halo particle only because a different strided
  observer also failed. The nearest-8 sources are added per observer, but
  there is no equivalent for observers. Include the observers closest to the
  other bunch (the minimum over the 8-nearest distances) and the head and
  tail along the closing direction.
- **P2-2 Nearest sources are double-weighted.** `core/startup_history_guard.py:256-271`.
  The 24 strided and 8 nearest sources are summed with the same per-particle
  weight. The nearest ones are over-represented relative to the bunch, which
  biases both numerator and denominator. Weight the strided sample by
  `N_live/24` and the nearest set by 1, excluding overlap.
- **P2-3 Tolerance is unvalidated.** `cli.py:2506`, `testbed_runner.py:1918`,
  `core/types.py` (`IntegratorConfig.startup_history_tolerance`). A NaN or
  `inf` tolerance disables the check without notice (`probe_guard.py` E).
  Negative values reject everything. Require a finite value in `(0, 1]`.
- **P2-4 Sweeps lose points without notice.** `lw_integrator/sweep_runner.py:753-764`
  records a failed point as `success: False` and continues. In driver-distance
  sweeps (`example_b2b_linked_energy_vs_driver_distance*.json`, COLD_START,
  1500 steps) the guard is most likely to reject the close and short points.
  That is a distance-dependent hole in the results, not an error. Run the
  preflight once per sweep and report which points it would reject before the
  sweep starts.
- **P2-5 Checkpoint resume skips the check.** `core/integration_runner.py:4172`.
  This is correct only if the original run was checked. A run started with the
  override resumes with no warning. Store the override, the report and the
  ratio in the checkpoint manifest and repeat the warning on resume.
- **P2-6 Horizon uses nominal `steps`.** The horizon ignores adaptive
  refinement, cavity-exit and relative-cutoff early exits and particle loss.
  An early exit shortens the run, so the missing fraction is understated. This
  is acceptable for a screen, but the docs should say so.

## P3: nits

- `core/trajectory_integrator.py:197` and `cli.py:2506` hard-code `1e-3`.
  Use `DEFAULT_STARTUP_HISTORY_TOLERANCE`.
- The unit test is one function covering four behaviours
  (`tests/unit/test_startup_history_guard.py`). Split it, and add regression
  tests for each finding above: ABH, a short run, an always-open pipe, a long
  bunch, NaN tolerance, driver-train prehistory and GUI round-trip.
- Cost: about 1.2-2.3 s per run, nearly independent of N (24 x 32 x 2 pairs x
  about 3300 samples). That is fine for production. In unit tests the
  constant cost is paid even for 1-step runs, which the many `steps=2` tests
  now avoid only through the override. Consider short-circuiting when
  `t_end < min tau_c` (always ratio 1), which would also make P0-2 explicit.
- `warnings.warn(..., stacklevel=3)` points into `retarded_integrator`, not the
  user's call site. With `run_integrator` and the testbed, the depth differs.
- The error message suggests "start the bunches farther apart". For co-moving
  and bound configurations (`probe_guard.py` F: gamma=100, 1 mm, `tau_c`
  66.7 ns in a 3.3 ns run) a larger separation makes `tau_c` longer. Suggest
  INERTIAL_PREHISTORY or driver-train prehistory there instead.

## Do the test overrides hide real problems?

The spy (`guardspy.py`) re-ran the guard without the override at every
overridden call in the 10 touched test files (91 tests, 120 calls):

- **39 overrides are no-ops**: non-B2B, empty-bunch or INERTIAL tests.
  Remove them, so the flag means something where it appears.
- **5 overrides are unneeded** (the guard passes): `test_pseudo_grid_zero_charge_crossing_matches_inertial_baseline`,
  `test_no_space_charge_drift_conserves_particle_energy`,
  `test_rider_exit_mode_mutes_driver_train_bunches_before_rider_halt`,
  `test_driver_train_prehistory_local_field_matches_each_synthetic_state`,
  `test_full_rfs_total_field_produces_mutual_neutral_dipole_force`.
- **47 would fail.** Most are the short-run ratio-1 artefact (P0-2), but not
  all. Some are physics tests whose conclusions depend on COLD_START:
  - `test_startup_inclusion_continuity.py::test_cold_start_gate_opens_within_one_step_force_work`
    and `test_checkpoint_restart_after_gate_is_bitwise_identical` (3.45e-3).
    These deliberately test the COLD_START gate, so the override is right
    there, but see P1-2: their energy assertions sit on a dropped approach.
  - `test_pseudo_grid_feasibility.py` crossings (0.06-0.16), and
    `test_rfs_integration.py::test_cold_start_moving_charge_drives_neutral_rfs_response`
    (0.165). These are "tracks full solver" comparisons. Both sides use
    COLD_START, so the comparison remains valid, but neither side is
    physical.
  - `test_self_consistency_options.py` (0.211) and
    `test_retarded_dipole_source_integration.py` (0.31-0.67).

  The overrides do not hide regressions in what these tests assert, because
  they compare like with like. They do mean no test runs the default-on
  guard against a realistic B2B configuration. Add at least one test that
  runs `retarded_integrator` on the AGENTS.md default geometry *without* the
  override and asserts it passes. Mark the deliberate COLD_START physics tests
  with a comment that says why they override.

## Passing runs unchanged?

Yes, as far as checked. The guard reads copies (`_per_particle` copies;
`_unpack` builds new arrays) and returns before integration, so passing runs
are unchanged. `test_cli.py`, `test_cli_gui_parity.py` and the touched files
pass. The only failure is the pre-existing dipole 1-ulp mismatch, which
reproduces on the base commit.

## Suggested order of work

1. Land as **warn-only** (log plus `RuntimeWarning`, report stored in run
   metadata). This removes about 85 test edits and the P0 regressions at once.
2. Fix the error model: use the integrator's gate (P1-1), add a signed measure
   (P1-2), a run-independent normalisation with an absolute floor (P0-2), gate
   transitions only (P0-3), and driver-train prehistory (P0-4). Exempt ABH
   (P0-1).
3. Validate it against the integrator: compare COLD_START with
   INERTIAL_PREHISTORY on the exact path for a few configurations near the
   threshold, and confirm the guard's ratio predicts the measured `Δp`
   difference.
4. Then make it an error by default.
