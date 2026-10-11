# Review: `review/trim-tests` against `development`

- Branch under review: `review/trim-tests` @ `dd2ac03` (one commit, "Remove tests
  that duplicate implementation algorithms (audit rank 10)")
- Base: `development` @ `3032cb4` (this is also the merge base, so there is no
  drift)
- Scope: 12 files, +324 / −843 (about 519 net lines removed in the diff stat).
  Only `tests/` and `CHANGELOG.md` change. `core/`, `lw_integrator/`, configs
  and defaults are unchanged, so default runtime behaviour cannot change.

## Verdict

**Mergeable.** I found no lost behavioural assertion protecting physics,
numerics, checkpoint behaviour or CLI/GUI parity. No P0 or P1 findings.
The dead-particle coverage is **stronger** than on `development`: five of
nine `core/` mutations are caught only by the trimmed branch. Two P2 gaps
are older than this branch and are not regressions. Both sit next to the new
assertions, so this is a cheap time to close them. The P3 items are
formatting and lint. I recommend running `black` and `ruff --fix` on the
touched files before merging.

## What changed

| Change | Assessment |
|---|---|
| Delete `tests/test_dead_particle_loop_fix.py` (6 tests, 376 lines) | 4 of the 6 tests ran copies of the runner and EOM loops inside the test, never the real code. The other 2 (`mark` twice, `propagate`) are now covered by parametrised cases in `tests/test_dead_particle_handling.py`. The 3 scenario tests become real-runner cases in `test_retarded_integrator_adaptive_retry_uses_reduced_timestep`. |
| New `tests/particle_states.py` with `distance_state`, `single_particle_state` and `empty_driver_state` | Each helper is byte-identical to the copies it replaces in 6 test files. Checked in the diff. |
| `test_equations_helpers.py`: merge `test_final_mass_shell_projection_uses_mechanical_momentum` and `test_sc_iteration_projection_uses_mechanical_momentum` into `test_sc_projection_uses_mechanical_momentum[True/False]` | The two originals had identical setup and assertions. They differed only in the `_patch_mass_shell_convergence_sequence` call and `mass_shell_relaxation=1.0`, and the parametrisation reproduces both exactly. |
| `test_equations_helpers.py`: add `mixed_bunch` parameter to `..._skips_already_dead_particles` | Adds a 3-particle case. See P2-2. |
| `test_integration_runner_control_flow.py`: merge the two "accept energy jump" tests and the two probe/cooldown tests | I diffed the original bodies against each other. They differ only in the gamma sequence, `min_timestep_factor` or `cooldown_steps`, and the final message and `h_calls` assertions. Every one of those assertions is kept, inside its parameter branch. |
| `_driver_exit_step` hoisted to module level | Identical body. It was duplicated verbatim in two tests. |
| Adaptive-retry test parametrised with `previous_deaths`, `first_deaths` and `retry_deaths` | The `original` case is equivalent to the old test. `_make_bunch_state(x=[0], z=-1)` matches `_make_particle_state(z=-1, gamma=1)`, and the `h_calls`, `t` and `gamma` assertions are unchanged. The new cases drive the real `retarded_integrator` retry path. |

The collected-ID diff (`pytest --collect-only`, ignoring parameter IDs)
matches the table: −6 loop-fix tests, 2+2+2 tests merged into 3, and
nothing else removed. No test in `test_cli_gui_parity*`, checkpoint,
`tests/physics/` (apart from the helper import), sweep or archive files was
touched.

## Validation performed

- `pip install -e ".[dev]"` in a scratch venv, Python 3.13.
- All 9 modified test files (plus `test_dead_particle_loop_fix.py` on dev):
  - trim: **171 passed** (157 s)
  - dev: **172 passed** (155 s)
- Full non-slow suite (`-m "not slow" tests`): **not completed**. It reached
  about 2% (no failures so far) before the 10-minute budget ran out. Because
  `core/` is identical, the 9 touched files plus the collection diff are what
  carry the evidence. `pytest --collect-only tests` collects 2149 IDs on trim
  with no import or collection errors, against 2158 on dev.
- Mutation testing. Each mutation was applied to `core/` in that branch's own
  worktree, and the dead-particle, retry, mark and propagate tests were run
  against it. `-x` stops at the first failure, so only one failing test is
  listed per run.

| # | Mutation | dev | trim |
|---|---|---|---|
| M1 | `integration_runner.propagate_deaths_to_base` → no-op (first-attempt deaths dropped on energy-jump retry) | **survives** | caught (`adaptive_retry[multiple_losses]`, `[additional_retry_loss]`) |
| M2 | delete `propagate_dead_particle_status(temp_trajectory_base, trajectory[i-1])` in the runner | survives | survives |
| M3 | `mark_particle_dead` does not zero `q` | caught (loop_fix idempotent) | caught (`test_mark_single_particle[q]`) |
| M4 | `mark_particle_dead` does not zero `q_source` | **survives** | caught |
| M5 | `propagate_dead_particle_status` does not zero `q` | **survives** | caught (`test_propagate_to_next_step[q]`) |
| M6 | `propagate` aliases `_dead_particles` instead of copying it | **survives** | caught |
| M7 | `propagate` aliases failure-info dicts (shallow) | **survives** | caught |
| M8 | `propagate` does not zero `q_source` | **survives** | caught |
| M9 | runner `temp_trajectory_base` shallow-copies `trajectory[i-1]` | survives | survives |
| M10 | `equations.py:2382` skip test changed to `result["_dead_particles"].any()` (skips *every* particle once any is dead) | survives | survives |
| M11 | `equations.py:391` aliases `current_state["_dead_particles"]` | survives (2 tests run) | survives |

I also checked that the `assert` inside the fake `self_consistent_step` is
not swallowed. The runner catches only `GammaBlowupError` around those calls
(`core/integration_runner.py:2739` and `:2921`), and under M1 the test fails
on that inner assert.

## Findings

### P0: wrong default behaviour or lost coverage

None. No runtime code or default changed. No physics, numerics, checkpoint
or CLI/GUI-parity assertion was removed. Every removed assertion is either
reproduced exactly by a parametrised case, or it ran only against a copy of
the algorithm inside the test (the loop-fix "pattern" tests). The mutation
table shows that those copies could not catch any of the runner or EOM
mutations above.

### P1: wrong physics

None. The mass-shell projection tests keep the same mechanical-momentum
assertions (`Px == A_x`, `Pt == C + qΦ term`, `gamma == 1`) under both
projection paths.

### P2: robustness / weak replacement coverage (pre-existing, not regressions)

**P2-1: The runner's own copy/propagate of the retry base is untested.**
- Location: `tests/unit/test_integration_runner_control_flow.py:1395-1397`,
  against `core/integration_runner.py:2620-2627`.
- What happens: M2 (no `propagate_dead_particle_status` onto
  `temp_trajectory_base`) and M9 (shallow copy of `trajectory[i-1]` into the
  base) both pass on dev and on trim.
- Evidence: the new assertions `trajectory[0]["x"] is not trajectory[-1]["x"]`
  and the `999.0` write-back check look like replacements for the deleted
  `test_temp_trajectory_copy_and_propagate`. They cannot detect M9, because
  the fake step itself `_clone_state`s `trajectory[-1]` before returning. The
  deleted test was no better, since it only copied a dict inside the test.
- Why it matters: if a future refactor shares arrays between the accepted
  state and the retry base, a rejected attempt could write into history that
  is already accepted, and no test would notice.
- Suggestion: have one fake return `trajectory[-1]` mutated **in place** on
  the rejected attempt (for example, write `x[:] = 999` when `h_step >= 1.0`).
  Then assert that `trajectory[1]["x"]` is unchanged.

**P2-2: The `mixed_bunch` EOM case does not prove that live particles are
still integrated.**
- Location: `tests/unit/test_equations_helpers.py:1086-1135`.
- What happens: M10 (skip every particle once any is dead) passes the whole
  dead-particle subset on both branches.
- Evidence: with zero charges the live particles feel no force, and the case
  asserts only `gamma == [1, 999, 1]`. Under the real code their `t` does
  advance: a direct run gives `t_in=[0,0,0]` and `t_out=[0.001, 0, 0.001]`.
  So the `.any()` mutation leaves every asserted value unchanged.
- Why it matters: the deleted `test_dead_particle_skip_pattern` asserted
  `particles_processed == [0, 2]`, but on a loop inside the test, so this is
  not lost coverage. The new case is still the natural place to close the gap.
- Suggestion: add
  `assert result["t"][0] > state["t"][0] and result["t"][2] > state["t"][2]`
  and `assert result["t"][1] == state["t"][1]`.

**P2-3 (minor): In the retry test, the `q == 0` assertions check the fake
step, not the runner.**
- Location: `test_integration_runner_control_flow.py:1331` and `:1384`.
- What happens: the fake calls `mark_particle_dead`, which zeros `q`, and
  then `propagate_dead_particle_status(state, state)`, which zeros `q` again
  using the state's own mask. So `final["q"][idx] == 0.0` holds whatever the
  runner does with charge.
- Why it matters: it is harmless, but it reads as runner coverage. Drop the
  self-propagate call, or add a comment saying the charge check exercises
  `particle_status`.

### P3: nits

**P3-1: `black --check` fails on 8 of the 10 touched files.**
- Affected: `particle_states.py`, `test_electromagnetic_physics.py`,
  `test_beamline_geometry_integration.py`, `test_distance_helpers.py`,
  `test_perf_optimizations.py`, `test_external_fields.py`,
  `test_radiation_diagnostics.py` and `test_integration_runner_control_flow.py`.
- On dev, only `test_perf_optimizations.py` failed.
- Causes: 3–4 blank lines are left where helpers were deleted, there is one
  blank line before `def distance_state`, and the new parametrize lists
  (`pytest.param(0.5, [..],` and the multi-line conditional expressions near
  line 1388) are not in black style.

**P3-2: `ruff` reports new `I001` (unsorted imports) in five files.**
- Affected: `particle_states.py`, `test_electromagnetic_physics.py`,
  `test_beamline_geometry_integration.py`, `test_distance_helpers.py` and
  `test_external_fields.py`.
- Cause: `from tests.particle_states import ...` sits in the third-party block
  above `core` imports, and in `test_integration_runner_control_flow.py`
  `core.particle_status` comes before `core.external_fields`.
- Totals: 28 → 30 errors over the same file set. The rest are already on dev.

**P3-3: Fragile local in the `mixed_bunch` test.**
- Location: `test_equations_helpers.py:1090-1096`.
- The test builds the 1-particle `state` and then overwrites it when
  `mixed_bunch` is true. Branching once on the parameter would read more
  clearly.

**P3-4: Inaccurate line count in the brief.**
- The brief says "about 620 net lines". The diff stat shows −519 net
  (+324 / −843). This affects only the description; the CHANGELOG entry
  itself is accurate.

**P3-5: Relaxation setting kept in the merged projection test.**
- `test_sc_projection_uses_mechanical_momentum[False]` still passes
  `mass_shell_relaxation=1.0`. AGENTS.md says to omit relaxation from new
  settings. It is kept verbatim from the original test, so this is not a
  change, but it could be dropped in a follow-up because relaxation is unused
  on the corrected path.

## Merge recommendation

Merge after `black` and `ruff check --fix` on the touched test files (P3-1
and P3-2). P2-1 and P2-2 are not blockers, since the gaps are older than
this branch. They are each a 2–3 line strengthening of tests this branch
already touches, so folding them in now would be cheap.
