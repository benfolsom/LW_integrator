# Review: `review/trim` (894fb4a) against `development`

Scope: one commit, `894fb4a Remove low-risk unused code and inert
self-consistency controls (audit ranks 1-6)`. Its merge base is `faad1ae`.
`development` has since moved to `3032cb4` (v0.11.0). Diff against the merge
base: 37 files, +292/-1010.

## Verdict

**Merge, after the trivial CHANGELOG conflict is resolved.** There are no P0 or
P1 findings. The removed code has no live callers. Sampled default runs are
bit-identical to the base. Old configs still load, warn once, and produce the
same solver configuration. One P2 finding concerns direct-API compatibility.
Fix it in this branch or accept it explicitly. The rest are P3 tidy-ups.

## What was checked

1. **Callers of removed symbols.** I ran `git grep` over `core/`,
   `lw_integrator/`, `optimization/`, `input_output/`, `configs/`, `scripts/`,
   `examples/`, `docs/`, `tests/` and `local/` in a trial merge with current
   `development`. I searched for each removed name:
   `_get_current_particle_gamma_and_beta`, `initialize_particle_bunches`,
   `build_field_representative_space_charge_source_charges`,
   `_evaluate_cli_stability`, `_CliStabilityOutcome`,
   `save_partial_optimization_results`, `_check_gamma_consistency`,
   `on_shell_kinematic_boundary_selected`, `position_change_*`, `sc_relaxation_*`,
   `self_consistency_mass_shell_relaxation_var`, and the
   `SelfConsistencyConfig.chrono_matching_mode` field.
   - Production code has no remaining references, including string-based
     `getattr`/`hasattr`/`monkeypatch` uses. The GUI still calls
     `_save_partial_optimization_results`, but that is a separate method in
     `optimization/results_mixins.py:235` and is kept. The removed
     `result_io.save_partial_optimization_results` was an unused duplicate.
   - The only remaining hit is `local/archive/2026-02/repo-root/run_optimization.py:102`
     (`self_consistency_tolerance=`), which is archived and not run.
   - Since `faad1ae`, `development` has not added any use of the removed names.
     The trial merge compiles. Ruff `F` shows no new undefined or unused names.
2. **The "inert" claim.** On `faad1ae`, `sc_mass_shell_relaxation` was unpacked
   once (`core/equations.py:2285`) and never read again. No other `core/` module
   read `mass_shell_relaxation`. `self_consistency_tolerance` was stored and
   saved but never reached the solver. `SelfConsistencyConfig.chrono_matching_mode`
   was never read, and the runtime mode comes from the outer `chrono_mode`.
3. **The constant-true branch.** `on_shell_kinematic_boundary_selected` was set
   to `True` once (`faad1ae:core/equations.py:2429`) and never reassigned. Every
   collapsed branch keeps the arithmetic of its `True` arm: `coordinate_dt`,
   beta from on-shell momentum, `mechanical_proper_velocity=predictor_u`,
   `gamma_from_velocity = hypot(1, |u|)`, and the `source_u_*` writes.
4. **Bit identity, measured.** `scratchpad/scripts/bitcheck.py` (not committed)
   runs `SimulationOptions.from_dict` and `run_testbed` with 120 steps on three
   shipped run configs:
   - `example_b2b_counter_propagating_proton_bunches.json` (Medina/LAD)
   - `example_b2b_relativistic_proton_stationary_lead_ion.json`
   - `demo_single_run_with_monitoring.json` (conducting wall)

   It also runs the three `uniform_run` Medina trajectories from
   `tests/unit/test_medina_validity_guard.py`. It ran once against the base and
   once against the trim worktree. The rider and driver trajectory SHA-256
   values, final gamma and Medina digests were **identical**:
   - `9f7fc1b8…` / `d92fe5dd…`
   - `94044be1…` / `1e3dfbad…`
   - `6a0a7629…`
   - Medina digests `0023f08a…`, `178909fd…`, `97ea3562…`

   The only difference is the new deprecation warning.
5. **Old configs.** I loaded a payload with `self_consistency_mass_shell_relaxation`,
   `self_consistency_tolerance` and `chrono_matching_mode: AVERAGED` through
   `cli.build_request`. Apart from the inert relaxation value (0.3 before, 0.7
   now), it gives the same `SelfConsistencyConfig` fields as the base. It still
   resolves `ChronoMatchingMode.AVERAGED`, and it emits exactly one
   `InertSelfConsistencyKeyWarning`. The testbed loader, sweep loader and plugin
   loader are covered the same way by the new parametrized test. All four
   loaders are wired: `cli.py:1761,3206`, `testbed_runner.py:974`,
   `sweep_runner.py:1309` and `plugin_persistence_helpers.py:135`.
   `sweep_runner` also filters keys to dataclass fields before building
   `OptimizationConfig(**…)` (`sweep_runner.py:1608`), so an unmigrated key
   could not raise there either.
6. **Tests run** (Linux, Python 3.13, `pip install -e ".[dev]"`, `pytest -p no:anyio`):
   - These changed or adjacent files gave 358 passed, 7 failed and 8 errors:
     - `tests/test_self_consistency_options.py`, `test_cli_gui_parity.py`,
       `test_optimization.py`, `test_optimization_config_helpers.py`,
       `test_single_integration_helpers.py`, `test_simulation_options.py`,
       `test_sweep_runner_logging.py`, `test_repository_surface.py`
     - `tests/unit/test_equations_helpers.py`, `test_pseudo_grid.py`,
       `test_high_gamma_stored_history.py`, `test_medina_validity_guard.py`,
       `test_rfs_integration.py`

     All 15 failures and errors also occur on `faad1ae` in this environment. They
     are Medina snapshot hashes (this platform's digests above differ from the
     macOS fixture) and `SelfConsistencyNonConvergenceError` in the γ≥1e5
     stored-history cases. **They are pre-existing and environment-specific.**
     The digests themselves are identical between base and trim.
   - Every test importing `causal_c5_dipole_provider` or
     `causal_local_source_history` (exact pair, inertial prehistory, spin pair,
     directional degree-6, and others) passed, plus
     `tests/physics/test_same_bunch_plummer.py`,
     `test_startup_inclusion_continuity.py` and `test_free_crossing_null.py`:
     **208 passed** (`-m "not slow"`).
   - Trial merge with `development` (`3032cb4`): the self-consistency options,
     single-integration, simulation-options, equations-helpers and
     external-fields tests gave **144 passed**.
   - Not run: `tests/test_optimization_plugin.py` and the GUI, because tkinter is
     not available in this container. The GUI changes were checked statically.
     The removed widgets and variables have no remaining references, and ruff `F`
     is clean.

## Findings

### P0: wrong default behaviour

None.

### P1: wrong physics when enabled

None. The collapsed branch was unreachable, and outputs are bit-identical on the
sampled default, Medina/LAD and conducting-wall paths.

### P2: robustness

**P2-1. `SelfConsistencyConfig(chrono_matching_mode=...)` now raises
`TypeError`. The commit keeps the equally inert `mass_shell_relaxation` field
"for direct study callers" but drops this one.**
- Where: `core/self_consistency.py:208` (kept) and around `:222` (where
  `chrono_matching_mode` was removed).
- Failure scenario: a study script or notebook outside this repo does
  `SelfConsistencyConfig(chrono_matching_mode="AVERAGED")` or reads
  `sc.chrono_matching_mode`. This can come from the `LW_feasibility_studies`
  defaults audit or from code that copied the base docstring. On `faad1ae` it
  works. On `review/trim` it raises
  `TypeError: SelfConsistencyConfig.__init__() got an unexpected keyword argument 'chrono_matching_mode'`
  (reproduced). Saved JSON is protected by the migration shim, but direct
  Python callers are not.
- Related evidence: `scripts/check_exact_macroparticle_*.py` record
  `asdict(SelfConsistencyConfig())` into evidence summaries. New summaries lose
  the `chrono_matching_mode` key but keep `mass_shell_relaxation: 0.7`. Anything
  that diffs old and new summary key sets will see a schema change.
- Suggested fix, pick one:
  - Keep `chrono_matching_mode: str = "FAST"` as an inert compatibility field,
    matching `mass_shell_relaxation`.
  - Or drop both, and note the API break explicitly in the CHANGELOG.

  Either way, the two inert fields should get the same treatment. I could not
  check the external studies repository from here.

**P2-2. CHANGELOG conflict with current `development`.** A trial
`git merge origin/review/trim` into `3032cb4` conflicts only in `CHANGELOG.md`,
because v0.11.0 moved the "Unreleased" entries. `core/equations.py` merges
cleanly. Resolve the conflict by moving the three new bullets under the new
`Unreleased` heading.

### P3: nits

1. **Shipped configs still carry the removed keys, so every example now warns
   on load.** 15 files under `configs/` contain
   `self_consistency_mass_shell_relaxation` and/or `self_consistency_tolerance`:
   - `configs/run_configs/demo_single_run_with_monitoring.json`
   - `configs/run_configs/example_b2b_*.json`
   - every `configs/sweep_configs/*.json` example
   - `configs/sweep_configs/manual_smoke/*`

   Loading any of them emits `InertSelfConsistencyKeyWarning` (observed in the
   bitcheck runs). `tests/physics/test_free_crossing_null.py:106` has the same
   problem. Scrub these keys from the shipped examples. Keep the test fixtures
   under `tests/fixtures/sweep_configs/` and `tests/unit/fixtures/` as
   legacy-load evidence.
2. **A self-assignment is left over from the collapse.**
   `core/equations.py:4803-4805` reads
   `coordinate_dt = proper_to_coordinate_dt; if coordinate_dt == 0.0: coordinate_dt = proper_to_coordinate_dt`.
   The `if` does nothing and can be deleted. The `sc_verbosity >= 3` debug text
   at `:4849` still says "γ_velocity (from β over Δt=…)". It now comes from
   `|u|`.
3. **`_calculate_gamma_from_beta` is now dead in production.**
   `core/equations.py:1311` was reachable only from the removed `else` arm. Its
   only remaining callers are tests. Three tests
   (`tests/unit/test_equations_helpers.py:1316,1347,1430`) still monkeypatch it
   to `lambda *args: 1.0`. That patch already did nothing on the base and still
   does nothing, so the tests pass for other reasons and the patch misleads
   readers. Remove the monkeypatches. Optionally delete the helper and its
   direct unit tests, or keep it as a documented utility.
4. **Lint and format regressions in touched files.** These did not exist on the
   base:
   - `black --check` would now reformat `core/equations.py` (the
     `beta_denominator` parenthesisation at about :4808),
     `core/causal_local_source_history.py` and
     `tests/test_self_consistency_options.py`.
   - ruff reports new `I001` import-order errors in
     `core/causal_c5_dipole_provider.py:23` and
     `core/causal_local_source_history.py:30`, which each have a stray double
     blank line. It also reports `I001` where `input_output.config_migration` is
     imported before `core` in `lw_integrator/cli.py:23`,
     `lw_integrator/testbed_runner.py:34` and
     `tests/test_self_consistency_options.py`.
   - ruff reports `E501` (89 characters) for the
     `_accepted_step_start_beta_prime_per_mm as _accepted_interval_start_beta_prime_per_mm`
     alias line in `core/causal_local_source_history.py`, and `E501` in the new
     test.

   Running `black` plus `ruff --fix --select I` on the touched files fixes all
   of these. A simpler fix for the alias is to rename the call sites, or to name
   the shared function once.
5. **The nested-object migration has no consumer.**
   `input_output/config_migration.py:32` strips
   `self_consistency.mass_shell_relaxation` and
   `self_consistency.chrono_matching_mode` from a nested `self_consistency`
   object. No loader in the repo reads a nested `self_consistency` object. It
   only appears in the `asdict(sc)` evidence summaries written by the scripts
   above. The code is harmless and defensive, but the CHANGELOG suggests saved
   configs of that shape load somewhere. Either say it exists for evidence
   summaries or drop it.
6. **A blank row is left in the GUI grid.** Removing the relaxation entry leaves
   `sc_frame` row 6 empty in `lw_integrator/gui_tab_mixins.py` (verbosity is
   still at row 7). It is cosmetic and could not be checked visually here
   because there is no tkinter.
7. **The historical audit doc is stale.** `docs/self_consistency_option_audit.md:81-92`
   still lists the relaxation weight, `self_consistency_tolerance` and the
   carried `SelfConsistencyConfig.chrono_matching_mode` as live rows. It is a
   dated audit, so a one-line "superseded by 894fb4a" note is enough.
   `docs/source/self_consistency.rst:54` is consistent with the branch.

## Test-coverage assessment

None of the removed tests guarded live behaviour:
- `_evaluate_cli_stability` (3 tests), `_get_current_particle_gamma_and_beta`,
  `_check_gamma_consistency` and `build_field_representative_space_charge_source_charges`
  (2 tests) only tested the deleted helpers themselves.
- `test_relaxation_weight_does_not_change_mechanical_solution` is moot now that
  the solver cannot read the field. The new parametrized loader test asserts
  that the inert keys do not change live controls.
- The repository tombstone tests (no legacy `.py`, no stale doc phrases, no
  `legacy:` marker, no nested `pytest.ini`) and the dead-dialog absence test
  guard against regressions, not behaviour. Dropping them fits the owner rule.
  The only marginal loss is `test_pytest_configuration_is_root_only`, since a
  nested `pytest.ini` would silently change the test configuration. It is not
  worth keeping on its own.
- The numerical reference tests (Medina snapshots, the stored-history decimal
  references, Plummer and startup continuity) are retained and unchanged apart
  from dropping the `mass_shell_relaxation=` kwargs.

The new tests are justified. One parametrized test over the four loaders, plus
one idempotence and alias test, is the minimum needed to guard the migration
shim. No further tests are needed.
