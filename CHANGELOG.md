# Changelog

All notable changes and updates to the LW Integrator project are documented in this file.

## Unreleased

### Radiation Bookkeeping Cleanup (June 2026)

- **Physics** — Removed the legacy component-wise `bdot` radiation-reaction correction. It was gated on step-to-step acceleration-magnitude changes and only modified stored acceleration history, not particle mechanical momentum or energy.
- **API** — `radiation_reaction_mode` now defaults to `off`; passive Liénard radiation diagnostics remain available through `radiation_power` and `radiation_energy`, while `power_matched_damping` is the only current opt-in self-force approximation.
- **Bug** — Initial legacy trajectory states now receive zero-filled radiation bookkeeping fields during integration startup, matching the SoA trajectory builder and later integration steps.
- **Tests** — Relabelled old radiation-reaction activation tests as high-acceleration / large-`bdot` diagnostics so they no longer imply that a self-force was applied. Added SoA/Numba-path regression coverage for radiation bookkeeping fields.
- **Files modified** — `core/equations.py`, `core/integration_runner.py`, `core/self_consistency.py`, `core/diagnostics.py`, `docs/source/radiation_reaction_plan.rst`, `tests/unit/test_radiation_diagnostics.py`, `tests/unit/test_trajectory_arrays.py`, `tests/unit/test_numba_mode_features.py`, `tests/physics/test_radiation_reaction_activation.py`, `tests/physics/test_extreme_radiation_reaction.py`

### Integrator Architecture Simplification (June 2026)

- **Refactor** — Removed `core/performance.py` and its alternate integrator wrapper (`retarded_integrator_numba` / `run_optimised_integrator`) to eliminate redundant orchestration paths
- **Refactor** — `retarded_integrator` now always runs the canonical integration orchestration; Numba acceleration remains at the force-kernel layer in `core/vectorized_interactions.py`
- **Tests** — Updated control-flow and Numba feature tests to validate canonical-path behavior without monkeypatching `core.performance`
- **Docs** — Removed the performance API page from Sphinx toctrees and updated overview wording to describe kernel-level Numba acceleration
- **Note** — This supersedes the historical `retarded_integrator_numba` wrapper notes below; feature support now applies to the canonical path.

### Historical: Numba Path Full Feature Support (June 2026)

- **Feature** — Energy monitoring (`EnergyMonitorConfig`) is now supported in the Numba path: `retarded_integrator_numba` performs the same per-step energy check (warn or raise `EnergyJumpDetected`) as the Python path; the `use_numba_path = False` gate removed
- **Feature** — Macroparticle mode (`macroparticle_charge_multiplier != 1.0`) is now supported in the Numba path: all `generate_conducting_image` calls inside `retarded_integrator_numba` forward `macroparticle_charge_multiplier`, `macroparticle_sigma_multiplier`, and `macroparticle_use_momentum_errors`; gate removed
- **Feature** — Space charge (`SpaceChargeConfig`) is now supported in the Numba path: `space_charge` kwarg forwarded to all `retarded_equations_of_motion` calls inside `retarded_integrator_numba`; gate removed
- **Feature** — Adaptive timestep (`AdaptiveTimestepConfig`) is now supported in the Numba path: extracted ~400 lines of per-step adaptive logic (substep subdivision, proximity refinement, energy-jump retries, gamma blowup handling, cooldown hysteresis) from `retarded_integrator` into a new `_run_adaptive_step(...)` helper and `_AdaptiveStepState` dataclass; both `retarded_integrator` and `retarded_integrator_numba` now call this shared helper; the final `use_numba_path = False` gate removed
- **Tests** — Added `tests/unit/test_numba_mode_features.py` with 9 tests covering each newly-enabled mode individually, in combination, and verifying Python/Numba parity
- **Files modified** — `core/performance.py`, `core/integration_runner.py`, `tests/unit/test_numba_mode_features.py`, `tests/unit/test_integration_runner_control_flow.py`

### SOA Trajectory Refactor — Phase 3–5 Completion (June 2026)

- **Perf** — Vectorized `gather_external_samples_soa` inner gather and interpolation loops: replaced 10 per-field Python list comprehensions with NumPy fancy indexing (`traj_ext.bx[indices, particle_indices]`) and replaced the per-particle interpolation loop with a masked vectorized blend; eliminates all remaining Python-level loops in the SOA gather path
- **Refactor** — Removed `state_at()` dict shim from `chrono_match_indices_soa`: replaced `traj.state_at()` + `compute_instantaneous_distance()` with inline direct SOA field access (`traj.x[index_traj, index_part] - traj_ext.x[index_traj, :]`); no dict allocation per chrono-match call
- **API** — `retarded_integrator` now returns a 4-tuple `(trajectory, trajectory_drv, traj_soa, traj_drv_soa)` where the last two elements are `TrajectoryArrays | None`; the Numba fast-path returns `None` for the SOA slots; all call sites updated to `traj, drv, *_soa_out = retarded_integrator(...)`
- **API** — `diagnostics.py` functions (`analyze_trajectory_energy`, `check_superluminal_velocities`, `check_gamma_consistency`, `validate_trajectory`, `find_radiation_reaction_activations`) now accept `TrajectoryInput = Union[Trajectory, TrajectoryArrays]`; SOA branches use fully vectorized NumPy operations; legacy dict-list paths preserved
- **Files modified** — `core/vectorized_interactions.py`, `core/distances.py`, `core/integration_runner.py`, `core/diagnostics.py`, `core/performance.py`, `core/trajectory_integrator.py`, `lw_integrator/cli.py`, `lw_integrator/testbed_runner.py`, `examples/adaptive_timestep_example.py`, `examples/energy_monitoring_example.py`, all affected test files

- **Bug** — SC particle-slice `{k: v[[j]] for k, v in step.items()}` failed with `IndexError` when any state-dict array had a different length than `n_particles` (e.g. metadata or per-step scalars); replaced with `_slice_step()` which only indexes arrays whose first-axis length equals `n_particles`
- **Files modified** — `core/equations.py`

### Space-Charge Startup Fix and Physics-Driven Retarded Threshold (May 2026)

- **Bug** — Intra-bunch SC forces were never applied: the `apply_forces` driver-startup gate blocked the rider→rider SC block, and the `len(trajectory) > 1` guard was always `False` because the substep buffer starts with one entry per main step
- **Impact** — All SC-enabled runs were silently equivalent to SC-off; paired SC-on/SC-off smoke runs produced bit-identical results regardless of pcount or charge weight
- **Fix 1** — Removed `apply_forces` from the SC guard; rider→rider SC is independent of whether the external driver force is currently gated by startup logic
- **Fix 2** — Replaced the `len(trajectory) > 1` guard with `len(trajectory) >= 1`; instantaneous Coulomb is now used as a startup approximation from step 0 onward
- **Fix 3** — Fixed `IndexError` in the `retarded=False` branch of SC: `sc_traj_ext` had length 1 but `compute_retarded_distance` was indexed with `index_traj` (which grows > 0); fixed by padding `sc_traj_ext = [sc_step] * (index_traj + 1)`
- **Feature** — Added physics-driven instantaneous→retarded transition threshold to `SpaceChargeConfig`: new fields `bunch_sigma_mm` (default `0.01` mm) and `min_retarded_steps` (`None` = auto). The resolver `resolve_min_retarded_steps(h_step)` computes `ceil(bunch_sigma_mm / (c × h_step))` — the number of steps for light to cross the bunch — as the minimum history required before switching from instantaneous Coulomb to the full retarded LW kernel. Setting `min_retarded_steps=0` restores the old minimal behaviour; `retarded=False` keeps instantaneous permanently.
- **Verified** — SC-on now produces measurably different `Px` and `x` from SC-off at `Q=1e10` total bunch charge with pcount=8 and σ=0.01 mm
- **Files modified** — `core/equations.py`, `core/types.py`

### Intra-Bunch Space Charge (May 2026)

- **Feature** — Added retarded intra-bunch space-charge forces: each rider particle now receives Liénard-Wiechert fields from all other rider particles (j ≠ i) in addition to the driver/image forces already computed
- **New type** — `core.types.SpaceChargeConfig` dataclass with `enabled`, `retarded` (default `True`), and `softening_mm` (Plummer softening length, default `0.0`) fields
- **Core physics** — Second accumulation pass in `retarded_equations_of_motion` (`core/equations.py`) reuses the existing `compute_retarded_distance` / `gather_external_samples` / `compute_vectorized_contributions` pipeline; no new force kernel required
- **Integration runner** — `retarded_integrator` and `run_integrator` accept a new `space_charge: Optional[SpaceChargeConfig]` parameter; when enabled, the Numba hot-path is bypassed (same behaviour as self-consistency and adaptive-timestep modes)
- **Self-consistency threading** — `self_consistent_step` forwards `space_charge` to the step function via keyword argument, preserving backward compatibility with all existing call sites and test fakes
- **Config surface** — `SimulationOptions` gains `space_charge_enabled`, `space_charge_retarded`, and `space_charge_softening_mm` fields; all serialised through `to_dict` / `from_dict`; `build_space_charge_config` builder follows the established `build_*_config` pattern
- **CLI** — `--space-charge` flag (store_true) and `--space-charge-softening-mm` float option added to `lw-simulate` / `python -m lw_integrator`
- **GUI** — New "Intra-Bunch Space Charge" section in the Stability tab with enable checkbox, retarded-fields toggle, and softening-length entry; wired through `_apply_options_to_ui` / `_build_options_from_ui` round-trip
- **Defaults** — Feature is off by default (`space_charge_enabled = False`); all existing runs and configs are unaffected
- **Performance note** — Intra-bunch space charge is not Numba-optimised and runs on the Python path only. The per-step cost scales as O(N²) in rider pcount. This is acceptable for the small pcounts used in feasibility studies (1–16 particles) but will be slow for large bunches.
- **Files modified** — `core/types.py`, `core/equations.py`, `core/self_consistency.py`, `core/integration_runner.py`, `lw_integrator/testbed_runner.py`, `lw_integrator/cli.py`, `lw_integrator/gui.py`, `lw_integrator/gui_tab_mixins.py`, `lw_integrator/gui_config_mixins.py`


### Sweep/Optimization Logging Policy Fixes (April 2026)

- **Bug** — CLI sweep `--quiet` and `log_verbosity` policy paths still emitted per-run progress/detail/debug lines through direct `print()` and uncapped callbacks
- **Impact** — Headless sweeps could produce noisy stdout and oversized logs even when users requested quiet or truncated logging, making post-processing and live-monitoring output harder to trust
- **Fix** — Routed CLI sweep output through the runner logging policy, preserving compact metric lines for truncated logging while limiting full timestep/progress/stability diagnostics to `log_verbosity="full"`
- **Bug** — The GUI sweep stability confirmation dialog silently converted self-consistency verbosity `0` to `1` and adaptive-timestep debug `False` to `True`
- **Impact** — Loading or confirming otherwise quiet sweep configs could unexpectedly enable lower-level debug output for later full-logging runs
- **Fix** — Preserve loaded/current stability logging values by default, keep sweep-default enabling only behind the explicit override path, and remove duplicate/unconditional GUI config debug prints
- **Regression coverage** — Added focused tests for quiet CLI sweep logging and stability-dialog logging defaults
- **Files modified** — `lw_integrator/sweep_runner.py`, `optimization/plugin_control_mixins.py`, `optimization/plugin_config_mixins.py`, `optimization/plugin_ui_mixins.py`, `optimization/config.py`, `tests/test_sweep_runner_logging.py`, `tests/test_optimization_plugin.py`

### Critical: Macroparticle Image Charge Multiplier Fix (April 2026)

- **Bug** — `generate_conducting_image()` applied `macroparticle_charge_multiplier` twice, so image charges scaled as `multiplier²` instead of `multiplier`
- **Impact** — Macroparticle conducting-wall runs could over-amplify image-charge strength by large factors; for example, a multiplier of `2` produced a `4×` image charge
- **Fix** — Removed the second post-loop scaling path so the multiplier is applied exactly once per generated image charge
- **Regression coverage** — Added unit coverage for single-application scaling, geometry-driven charge suppression, and the surrounding integration control-flow paths
- **Files modified** — `core/images.py`, `tests/unit/test_trajectory_integrator_helpers.py`, `tests/unit/test_integration_runner_control_flow.py`

### Critical: Equation State Copy Isolation Fix (April 2026)

- **Bug** — `_initialize_result_state()` in `core.equations` reused the previous state's `q` array instead of copying it
- **Impact** — Marking a particle dead in the new step could silently mutate the previous step as well, corrupting trajectory history and retry logic by retroactively zeroing old charges
- **Fix** — Copy `q`, `m`, and `char_time` when building the next-step state so dead-particle handling and later mutations remain isolated to the new state
- **Regression coverage** — Added helper and control-flow coverage for state copying, scalar extractors, retarded-distance helpers, gamma reconciliation, convergence logging, cancellation, blowup handling, and final mass-shell projection
- **Files modified** — `core/equations.py`, `tests/unit/test_equations_helpers.py`

### Numba Force-Kernel Parity Fix (April 2026)

- **Bug** — `_compute_forces_numba_kernel()` in `core.vectorized_interactions` computed local `bdot_scalar` as `bdot·bdot` instead of the maintained NumPy path's `beta·bdot`
- **Impact** — The default JIT-accelerated force path could drift from the validated Python implementation on nonzero-acceleration trajectories, producing inconsistent momentum updates depending on whether Numba was active
- **Fix** — Corrected `bdot_scalar` to `bx*bdotx + by*bdoty + bz*bdotz`, aligning the Numba kernel with the NumPy implementation, and added parity coverage for hard-cutoff, small-k, verbose diagnostics, interpolation branches, and nonzero-acceleration kernels
- **Files modified** — `core/vectorized_interactions.py`, `tests/unit/test_vectorized_interactions_helpers.py`, `tests/unit/test_images_helpers.py`

### Adaptive Gamma-Blowup Retry Fix (April 2026)

- **Bug** — The adaptive gamma-blowup recovery path in `core.integration_runner` could raise `UnboundLocalError` before retrying with a smaller timestep
- **Impact** — Instead of recovering or cleanly marking a particle dead, some gamma blowups aborted the integration loop from the control-flow layer itself
- **Fix** — Removed the invalid `trial_state` propagation in the retry branch and added regression coverage for no-adaptive, minimum-timestep, and hard-blowup retry paths
- **Files modified** — `core/integration_runner.py`, `tests/unit/test_integration_runner_control_flow.py`

### Adaptive Refinement Bookkeeping Fixes (April 2026)

- **Bug** — Adaptive gamma-blowup retries were not incrementing `refinement_attempt`, so the configured max-retry limit was not actually enforced
- **Impact** — Some repeated gamma-blowup cases could keep refining until minimum timestep rather than honoring the intended retry cap
- **Fix** — Count gamma-blowup refinement attempts the same way energy-jump retries are counted, and added regression coverage for max-retry fallback

- **Bug** — Probe-stability checks in reduced-timestep mode compared the accepted step against the already-updated `previous_energy`, which collapsed the measured `ΔE/E` to zero
- **Impact** — The “unstable during probing” path was effectively unreachable, making timestep recovery look stable even when step-to-step energy drift remained large
- **Fix** — Preserve the pre-step reference energy for probing decisions, and added regression coverage for both stable return-to-normal and unstable-cooldown restart behavior

- **Files modified** — `core/integration_runner.py`, `tests/unit/test_integration_runner_control_flow.py`

### BUNCH_TO_BUNCH Transverse Offset Mode Fix (April 2026)

- **Bug** — Optimization and sweep run-control code sometimes compared `SimulationType.BUNCH_TO_BUNCH` enum values to the string `"BUNCH_TO_BUNCH"`
- **Impact** — Enum-backed BUNCH_TO_BUNCH configs could take the conducting-wall offset path, treating an absolute bunch offset as an aperture fraction and scaling it by aperture radius
- **Fix** — Centralized simulation-mode detection in `optimization.simulation_type_helpers.is_bunch_to_bunch()` and routed transverse-offset, sweep-grid, timestep, result-export, CLI sweep, and sweep run-control branches through the normalized check
- **Regression coverage** — Added tests covering enum and string mode values so BUNCH_TO_BUNCH offsets remain absolute, BUNCH_TO_BUNCH sweeps keep driver parameters, CLI sweep grids stay in BUNCH_TO_BUNCH mode, and auto-distance timestep calculations use driver distance
- **Files modified** — `lw_integrator/sweep_runner.py`, `optimization/config.py`, `optimization/plugin_config_mixins.py`, `optimization/plugin_control_mixins.py`, `optimization/results_mixins.py`, `optimization/run_mixins.py`, `optimization/run_parameter_helpers.py`, `optimization/simulation_type_helpers.py`, `optimization/sweep_helpers.py`, `optimization/sweep_result_helpers.py`, `tests/test_cli_gui_parity.py`, `tests/test_optimization.py`, `tests/test_optimization_config_helpers.py`, `tests/test_optimization_run_parameter_helpers.py`, `tests/test_sweep_result_helpers.py`

### Optimization Soft-Penalty Threshold Fix (April 2026)

- **Bug** — The optimization run path used a duplicate soft-penalty implementation with a hard-coded high-energy threshold instead of the tested mass-aware helper
- **Impact** — Proton and heavier-ion optimizations could be penalized at electron-like energies, biasing objective values away from otherwise valid high-energy parameter regions
- **Fix** — Routed optimization evaluations through `optimization.penalties.compute_soft_penalty()` and removed the duplicate mixin method
- **Regression coverage** — Kept focused penalty coverage for electron/proton threshold behavior and added an API guard so the duplicate control-mixin penalty method is not reintroduced
- **Files modified** — `optimization/run_mixins.py`, `optimization/plugin_control_mixins.py`, `tests/test_optimization_plugin.py`

### CLI Conducting-Wall Energy Convention Fix (April 2026)

- **Bug** — The CLI sweep runner converted conducting-wall particle energy with the BUNCH_TO_BUNCH kinetic-energy convention when building `SimulationOptions.rider_params["starting_Pz"]`
- **Impact** — Headless conducting-wall sweeps could initialize riders with too-large longitudinal momentum relative to the GUI/single-run convention, breaking CLI/GUI parity for wall-mode runs
- **Fix** — Reused the shared single-integration Pz helper in `lw_integrator.sweep_runner`, preserving kinetic-energy semantics for BUNCH_TO_BUNCH and total-energy semantics for wall modes
- **Regression coverage** — Added CLI option-construction coverage that captures the generated `SimulationOptions` and asserts conducting-wall Pz matches the shared GUI helper, not the BUNCH_TO_BUNCH convention
- **Files modified** — `lw_integrator/sweep_runner.py`, `tests/test_cli_gui_parity.py`

### Maintained Plotting and Validation Surface Cleanup (April 2026)

- **Plotting surface** — Added focused CLI coverage for `lw-generate-sweep-heatmap`, `lw-plot-latest-live`, `lw-plot-from-logcache-live`, and `lw-plot-trajectory`
- **Sweep plotting behavior** — Stopped auto-generating sweep heatmaps from the GUI save path; sweep saves now point users to `lw-generate-sweep-heatmap` for explicit post-processing
- **Legacy isolation** — Removed standalone legacy comparison and legacy plotting Python scripts from active examples and the `legacy/` tree; legacy notebooks remain as historical reference material
- **Config surface** — Removed stale legacy/overlay/difference comparison keys from tracked example configs while keeping loader tolerance for old user configs
- **Test discovery** — Fixed pytest configuration to collect from the actual `tests/` tree instead of stale `lw_integrator/tests`
- **Files modified** — `lw_integrator/sweep_heatmap.py`, `tests/test_plotting_tools.py`, `tests/test_adaptive_timestep_interactions.py`, `tests/test_repository_surface.py`, `docs/source/validation.rst`, `docs/source/notebooks.rst`, `docs/source/overview.rst`, `docs/source/theory.rst`, `docs/source/recent_changes.rst`, `pyproject.toml`, `examples/validation/`, `legacy/`

### CLI Config Compatibility Fix (April 2026)

- **Bug** — CLI JSON config parsing stopped accepting integer `SimulationType` flags (`0`, `1`, `2`) and historical chrono/startup aliases even though master accepted these values
- **Fix** — Restored integer mode parsing while rejecting boolean values and invalid integer flags; restored config-file aliases for older chrono/startup values without re-advertising them in help text
- **Regression coverage** — Added CLI parser coverage for all integer simulation modes, invalid integer inputs, and historical chrono/startup aliases
- **Files modified** — `lw_integrator/cli.py`, `tests/test_cli.py`

### Optimization Top-N Trajectory Regeneration Fix (April 2026)

- **Bug** — Top-N optimization trajectory regeneration used a stale local parameter mapper instead of the same resolver used by objective evaluations
- **Impact** — Regenerated trajectories for swept rider/driver parameters could differ from the parameter set that was actually optimized, especially for BUNCH_TO_BUNCH driver energy, mass, and starting-distance sweeps
- **Fix** — Route top-N regeneration through `resolve_optimization_run_parameters()` and always restore the temporary trajectory-saving flag after reruns
- **Regression coverage** — Extended BUNCH_TO_BUNCH top-N trajectory coverage to assert swept rider and driver parameters are passed through
- **Files modified** — `optimization/results_mixins.py`, `tests/test_optimization.py`

## v0.6.0 — March 2026

### CLI / GUI Parity (March 2026)

- **Unified code paths** — The CLI sweep runner (`sweep_runner.py`) now calls the same `run_testbed()` / `SimulationOptions` code paths as the GUI, eliminating divergent particle initialisation, integrator invocation, and metric extraction between the two interfaces
- **Identical results** — `lw-simulate --sweep-config …` produces the same output as the GUI's Blind Sweep mode for a given configuration
- **Files modified** — `lw_integrator/sweep_runner.py` (major refactor), `lw_integrator/cli.py`

### Incomplete-Sweep Archiving (March 2026)

- **Automatic relocation** — Sweeps with fewer than 100 completed runs are moved to `results/archive/incomplete/<sweep_dir_name>` immediately after saving
- **All save points wired** — CLI runner (after save and on `KeyboardInterrupt`), GUI mixin (`results_mixins.py`), GUI plugin (`optimization_plugin.py`), and library API (`parameter_sweep.py`)
- **Collision handling** — If the destination already exists, a `_1`, `_2`, … suffix is appended
- **New function** — `optimization.result_io.relocate_incomplete_sweep(sweep_dir, min_runs=100, log_fn=None)`

### Heatmap Contour Improvements (March 2026)

- **Contour alpha** reduced from 0.35 → 0.18 for less visual clutter
- **Edge-aware label clamping** — Labels whose centres fall outside the axes data limits are hidden; a one-shot `draw_event` callback shifts overflowing labels inward after the final Matplotlib layout pass
- **Overlap culling** — A second pass hides labels that genuinely intersect previously-accepted labels (negative pixel padding of −4 px, so merely-touching labels are kept)
- **Files modified** — `lw_integrator/sweep_heatmap.py`

### Driver Energy Sweep Fix (February–March 2026)

- **Bug** — Sweeping `driver_energy_gev` in BUNCH_TO_BUNCH mode had no effect; all runs used the hard-coded default Pz of −4925.0
- **Fix** — Check for `driver_energy_gev` in the parameter dictionary first and convert to Pz via `calculate_starting_pz_from_energy()`, falling back to legacy `driver_starting_Pz` key
- **Files modified** — `lw_integrator/optimization_plugin.py`, `optimization/run_mixins.py`

### Driver Pz / KE Calculation Fix (March 2026)

- **Bug** — Sweep runner used rider mass instead of driver mass when converting energy to Pz, producing incorrect results for ion-driver / electron-rider configurations
- **Files modified** — `lw_integrator/sweep_runner.py`, `lw_integrator/optimization_plugin.py`

### CLI Sweep Verbosity Overrides (March 2026)

- **`--log-verbosity {none,truncated,full}`** — Override the config's `log_verbosity` field from the command line
- **`--sc-verbosity {0,1,2,3}`** — Override self-consistency verbosity
- **`--adaptive-debug` / `--no-adaptive-debug`** — Toggle adaptive-timestep debug output
- Passed through to `run_sweep_from_config()` as a `verbosity_overrides` dictionary

### CLI / GUI Parity Tests (March 2026)

- **New test suite** — `tests/test_cli_gui_parity.py` (1 582 lines) verifying that CLI and GUI sweep paths produce identical configurations and results for real sweep configs

### Plot Generator CLI Sweep Bug Fix (March 2026)

- Fixed a bug in the sweep heatmap plot generator that caused incorrect parameter axis labelling when invoked from the CLI

### Version Bump

- Version bumped from 0.5.8 → **0.6.0**
- `.bumpversion.cfg` and `core/_version.py` updated

---

## February 2026

### Critical: Driver Energy Sweep Not Applied (February 26, 2026)

- **Bug** - Sweeping `driver_energy_gev` in BUNCH_TO_BUNCH mode had no effect on simulation results; all runs produced identical rider energy gains regardless of driver energy
- **Root cause** - The sweep code path in `_run_sweep_background()` built `driver_params_dict` using `params_dict.get("driver_starting_Pz", -4925.0)`, but the sweep grid populated the key `"driver_energy_gev"` (in GeV). Since `"driver_starting_Pz"` was never in `params_dict`, every run used the hardcoded default Pz of -4925.0
- **Scope** - Affected the sweep run-control path when building BUNCH_TO_BUNCH driver parameters from sweep-grid values
- **Fix** - When building driver parameters, check for `"driver_energy_gev"` in `params_dict` first and convert to Pz via the shared energy-to-Pz helper, falling back to `"driver_starting_Pz"` only for older configs
- **Files modified** - `optimization/run_mixins.py`, `optimization/run_parameter_helpers.py`, `optimization/sweep_run_helpers.py`, `optimization/sweep_helpers.py`, and related regression tests

### Optimization Plugin Fixes (February 26, 2026)

- **KeyError on config load** - Fixed crash when loading BUNCH_TO_BUNCH configuration from main GUI into optimization plugin
- **Root cause** - UI was updated to use `driver_energy_gev` parameter instead of `driver_starting_Pz`, but config loading code still referenced the old parameter name
- **Solution** - Added `calculate_energy_from_pz()` conversion function and updated `_on_load_from_main_config()` to convert legacy Pz values to energy (GeV)
- **Starting position field clarification** - Changed "Starting z Positions" field to control only rider starting position (not driver)
- **Impact** - Eliminated redundancy where driver position could be set in two places (field vs sweepable parameter)
- **Result** - Driver starting position now controlled exclusively by `driver_starting_distance` sweepable parameter; rider position set independently
- **Files modified** - `lw_integrator/optimization_plugin.py` (added conversion function, fixed config loading, updated UI labels)
- **Backward compatibility** - Old configs with `starting_Pz` values are automatically converted to energy on load

### Plotting Absolute Position Fix (February 26, 2026)

- **Plotting issue** - Energy plots showed z-positions relative to each particle's starting position rather than absolute lab-frame positions
- **Impact** - In BUNCH_TO_BUNCH simulations, rider starting at z=0 and driver starting at z=200mm both appeared to start from 0 in their respective plots, hiding the 200mm spatial separation
- **Root cause** - Code computed `z_rel = z - z[0]` to make positions relative, likely inherited from single-particle scenarios
- **Solution** - Changed to use absolute positions directly: `z_rel = z` (variable name kept for compatibility)
- **Result** - Energy plots now show true lab-frame positions, making spatial relationships between particles visible
- **Files modified** - `lw_integrator/testbed_runner.py` (lines ~1519, 1707, 1767, 1787)
- **Note** - Backward compatibility: old saved PNG files show relative positions, new ones show absolute positions

### GUI Button Visibility Fix (February 26, 2026)

- **Layout issue** - RUN and CANCEL buttons could become completely obscured when window was resized vertically to small sizes
- **Root cause** - Configuration panel used mixed pack() layout where scrollable canvas with expand=True could push fixed control frames below visible area
- **Solution** - Restructured config panel to use grid layout with explicit weight distribution:
  - Row 0 (weight=1): Scrollable canvas container - expands to fill space
  - Row 1-3 (weight=0): Control elements (Run Mode, RUN/CANCEL buttons, Status) - fixed height, always visible
- **Testing** - Added a local resize check to verify buttons remain visible at various window sizes
- **Files modified** - `lw_integrator/gui.py` (\_build_config_panel method, lines ~2608-2910)

### CLI Logging Fixes (February 25, 2026)

- **Debug flag parsing** - Fixed `--debug` and `--log-level` CLI options that were not being properly parsed
- **Logcache output** - CLI sweep runner now outputs optimization metrics to logcache files for live plotting compatibility
- **Format alignment** - Ensured CLI log format matches GUI expectations for plotting scripts

### COLD_START Gating Formula Fixes (February 20-25, 2026)

**Critical bug fixes** - The COLD_START gating mechanism had two fundamental errors in computing when retarded forces should be applied:

#### Fix 1: Division vs Multiplication (February 2026)

- **Incorrect formula** - Used multiplication `R × (1 - β·n̂)` instead of division `R / (1 - β·n̂)`
- **4× error** - For relativistic particles approaching sources (β·n̂ = -1), threshold was 4× too large (40km instead of 10km)
- **Hardcoded limitation** - Used hardcoded `estimated_max_R = 10000 mm`, failing for separations > 10km
- **Edge case handling** - Now properly handles receding particles (β·n̂ > 0) with threshold → ∞ as β·n̂ → 1

**Impact**: All relativistic simulations with β > 0.5 were affected. The bug caused forces to be gated for too long, then activate with insufficient causal history, resulting in energy losses of 250-3200 GeV (orders of magnitude larger than physical).

#### Fix 2: Missing β Factor for Low-Velocity Particles (February 2026)

**Second critical bug** - The formula `threshold = R / (1 - β·n̂)` calculated the distance **light travels**, not the distance the **particle travels**:

- **Missing β factor** - Formula should be `threshold = β·R / (1 - β·n̂)` to account for particle velocity
- **100× error for low-β** - For β = 0.01, threshold was 198mm instead of 2mm (forces suppressed until particle passed interaction region)
- **10× error for moderate-β** - For β = 0.1, threshold was 182mm instead of 18mm
- **Masked by relativistic cases** - For β ≈ 1, the error was negligible (factor of β ≈ 1), so bug went unnoticed in high-energy simulations

**Physical Derivation**: When particle and light approach:

- Initial separation: R
- Light speed: c (toward particle)
- Particle speed: v = β·c (toward light)
- Relative closing speed: c(1 - β·n̂)
- Time to meet: t = R / [c·(1 - β·n̂)]
- Distance **particle** travels: d = v·t = β·c·t = **β·R / (1 - β·n̂)** ✓
- Distance **light** travels: d = c·t = R / (1 - β·n̂) (old formula ✗)

**Corrected formula**: `threshold = β·R / (1 - β·n̂)` where:

- **Approaching** (β·n̂ < 0): denominator > 1 → threshold < β·R (particles and light meet quickly)
- **Perpendicular** (β·n̂ = 0): denominator = 1 → threshold = β·R (light travels full distance)
- **Receding** (β·n̂ > 0): denominator < 1 → threshold > β·R (light takes longer to catch up)
- **Receding at c** (β·n̂ → 1): denominator → 0 → threshold → ∞ (forces never apply)

**Dynamic Calculation**: The threshold is **recalculated every integration step** using current values:

- Distance R updates as particles move and images reposition
- Velocity β updates as particles accelerate/decelerate
- Threshold automatically decreases as particle approaches sources
- Two-stage check: (1) fast conservative estimate to skip expensive calculations, (2) precise per-source threshold
- Ensures physical causality at every step based on evolving geometry

**Example Timeline** (β = 0.5, initial R = 200mm, approaching):

```
Step 0:   travel = 0mm,   R = 200mm, threshold = 67mm  → forces OFF
Step 50:  travel = 25mm,  R = 175mm, threshold = 58mm  → forces OFF
Step 130: travel = 65mm,  R = 135mm, threshold = 45mm  → forces ON ✓
Step 200: travel = 100mm, R = 100mm, threshold = 33mm  → forces ON
```

**Impact**: Low-velocity simulations (β < 0.5) had severely incorrect gating. Non-relativistic particles would have forces suppressed until far past physical interaction regions, producing wrong results. High-β simulations (β > 0.9) unaffected.

### Transverse Offset Sweep Bug Fix (February 23, 2026)

- **Sweep parameter handling** - Fixed transverse offset being swept over multiple values instead of using single beam position
- **Performance impact** - Reduced sweep size by 2-3× for configs with multiple offset values
- **Physical correctness** - Transverse offset now correctly represents beam center position, not a parameter to optimize
- **Backward compatible** - Configs with multiple offset values now use only the first value

### Live Plotting Tools (February 19-24, 2026)

- **Unphysical gain filtering** - Live plotter now filters out non-physical gain values from visualization
- **CLI log parsing** - Fixed plotting scripts to handle both GUI and CLI logcache formats
- **Monitoring scripts** - Added tools for real-time sweep and optimization monitoring

### Stripped Ion Support (February 18, 2026)

- **Arbitrary ion species** - Added support for ions with configurable charge states (e.g., Ar^8+, C^6+)
- **Sweep configurations** - Included example configs for stripped ions in sweep library

### Critical Bug Fixes (February 18, 2026)

- **Transverse momentum parameter** - Fixed optimization silently ignoring transverse momentum parameter
- **Parameter logging** - Fixed only 3 of 7-9 parameters being logged during optimization runs
- **Driver energy UI** - Improved driver bunch energy configuration interface to eliminate confusion
- **Gamma reconciliation persistence** - Fixed gamma reconciliation settings not loading from saved configs
- **Optimization config saving** - Fixed validation errors preventing optimization configs from being saved

### GUI and Logging Improvements (February 11, 2026)

- **Parameter visibility** - Fixed driver parameter sweep visibility and loading issues
- **GUI greying** - Corrected greyout behavior for context-dependent parameters
- **Log convergence bug** - Fixed `log_convergence` option causing crashes

### Adaptive Timestep Refactoring (February 9-10, 2026)

**Auto-calculated parameters** - The adaptive timestep system now automatically calculates derived parameters to prevent inconsistent configurations:

- **`max_refinement_attempts`** - Computed from `timestep_reduction_factor` and `min_timestep_factor` using formula: `ceil(log(1/min_factor) / log(reduction_factor))`
- **`max_substeps_per_step`** - Computed from `min_timestep_factor` with 10% safety margin: `ceil(1/min_factor) × 1.1`
- **Reduced default reduction factor** - Changed from 10 to 3 for more gradual refinement, reducing oscillation in pathological cases
- **GUI improvements** - Max attempts shown as read-only calculated value with visual feedback
- **Time discontinuity prevention** - Automatic substep cap ensures full timestep coverage even at minimum refinement level

**Impact**: Eliminates overdetermined parameter combinations that could cause time skipping or excessive refinement. Users only set two independent parameters (`reduction_factor` and `min_timestep_factor`), with derived values calculated automatically for consistency.

### Batched Logging Implementation (February 9, 2026)

**Performance optimization** - Inner-loop debug logging now uses batched updates to prevent GUI unresponsiveness:

- **Batch aggregation** - Debug messages accumulated in memory and flushed in batches (default: 50 messages per flush)
- **Throttled GUI updates** - Reduces event queue flooding by ~100× in pathological cases (e.g., 750 messages → 8 GUI updates)
- **Logger parameter** - New optional `logger` parameter on `retarded_integrator()` accepts callable for custom logging
- **Backward compatible** - Falls back to `print()` if no logger provided; existing code unaffected
- **GUI responsiveness** - Prevents multi-minute freezes when `adaptive_timestep_debug = True` during challenging runs

**Impact**: GUI remains responsive during verbose debugging. Users can enable full adaptive timestep diagnostics without performance penalty.

### Gamma Reconciliation Default Changed (February 9, 2026)

**Disabled by default** - Gamma reconciliation feature now defaults to `DISABLED` for v0.4.8 compatibility:

- **Energy conservation** - Original reconciliation implementation violated energy conservation by overwriting `Pt` without preserving scalar potential contribution
- **Momentum rescaling issue** - Spatial momentum rescaling altered particle trajectories incorrectly
- **Opt-in feature** - Reconciliation methods (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, etc.) still available but require explicit enablement
- **Legacy behavior restored** - Default configuration matches v0.4.8 stable behavior: `gamma_reconciliation_method = DISABLED`

**Impact**: Eliminates silent energy non-conservation for users upgrading from v0.4.8. Feature requires redesign before safe re-enablement.

### Sweep Plotting and Heatmap Tools (February 5-8, 2026)

- **Sweep visualization** - New plotting tools for parameter sweep results with contour plots
- **Heatmap generation** - Automated heatmap creation with configurable color schemes
- **Live updates** - Real-time plot updates during long-running sweeps
- **Transparency controls** - Adjustable marker transparency for dense data visualization

### Particle Tracking and Failure Handling (February 4-5, 2026)

- **Blowup detection** - Improved detection and handling of particle trajectory failures
- **Cancellation improvements** - Better graceful shutdown for interrupted simulations
- **Death penalty scaling** - Fixed particle death penalty to use 1:1 scaling (10% lost → 10% penalty)
- **Failure metrics** - Added particle failure tracking to optimization results

### Verbose Logging in Sweep/Optimization (February 2026)

When running sweeps or optimizations, verbose diagnostic logs (SC iterations, adaptive timestep refinements) are now streamed to the GUI in real-time when verbosity settings are enabled:

- **Self-Consistency Verbosity** (`self_consistency_verbosity > 0`): SC convergence diagnostics are displayed in the GUI log window during runs
- **Adaptive Timestep Debug** (`adaptive_timestep_debug = True`): Timestep refinement actions are displayed in the GUI log window during runs

**Key behaviors:**

1. These logs appear in **real-time** during sweep/optimization execution
2. Logs are visible in the GUI's **Detailed** log view (toggle Summary/Detailed in the log controls)
3. Verbose output appears **even when not saved to file** (controlled separately by `log_verbosity` setting)
4. The `log_verbosity` setting controls what gets saved to disk:
   - `"none"`: No logs saved, SC/adaptive verbosity disabled
   - `"truncated"`: Brief logs only, SC/adaptive verbosity disabled
   - `"full"`: Complete debug logs saved, SC/adaptive verbosity enabled
   - `"top_n_only"`: Logs saved only for top N trajectories, SC/adaptive verbosity enabled

**Example:** If you set `log_verbosity="full"` and `self_consistency_verbosity=2`, you'll see detailed SC convergence messages like:

```
[VERBOSE] Particle 0: converged in 3 iter, E_ms=1.234e-08
[VERBOSE] Particle 1: converged in 2 iter, E_ms=5.678e-09
```

This ensures that diagnostic information is always visible during runs when requested, independent of file-saving preferences.

## January 2026

### Optimization System Enhancements (January 14-16, 2026)

- **Optimization plugin refactor** - Major restructuring of optimization system for maintainability
- **Smoothness penalties** - Refined optimizer penalties for trajectory smoothness
- **Top-N results bug** - Fixed bug where top-N runs were using incorrect default parameters
- **Output directory structure** - Improved organization of optimization results

### GUI Usability Improvements (January 6-11, 2026)

- **Trajectory output frame** - Fixed bugs in trajectory saving and display
- **Top-N controls** - Added proper greying out of top-N trajectory options for sweep mode
- **Pillow plot display** - Fixed issues with plot rendering in GUI
- **View output buttons** - Corrected functionality of result viewing buttons
- **Heatmap removal** - Removed unnecessary heatmap generation that slowed GUI

### Parameter Sweep Enhancements (January 6-7, 2026)

- **Wall position sweeps** - Made wall_z parameter sweepable for aperture studies
- **Auto-timestep debugging** - Added debugging options for auto-calculated timestep issues
- **Range parsing** - Fixed tuple/dict parsing bugs in parameter range fields
- **Output results** - Improved results directory structure and metadata

### Installation and Documentation (January 5, 2026)

- **System dependencies** - Improved documentation for tkinter and system-level dependencies
- **Bump2version integration** - Added automated version management workflow
- **Development guide** - Created comprehensive guide for contributors

## December 2025

### GUI Organization and Layout (December 2025)

- **Config menu persistence** - Made configuration menu a persistent pane instead of popup
- **Vertical resizing** - Added GUI vertical resizing handles for better space management
- **Log window sizing** - Adjusted default log window height for better visibility
- **Non-ANSI keyboard support** - Fixed keyboard shortcuts for non-ANSI layouts
- **Run button behavior** - Improved run button state management and feedback

### Optimization Implementation (December 10-21, 2025)

- **GUI optimization mode** - Implemented full optimization workflow in GUI
- **Four optimization methods** - Genetic Algorithm, Differential Evolution, Nelder-Mead, Multi-start
- **Convergence detection** - Early stopping when fitness plateaus (saves 40-70% computation)
- **Top-N trajectory saving** - Automatic saving of best results from optimization runs
- **Progress tracking** - Real-time optimization progress display

### Chrono-Match Interpolation (December 17, 2025)

- **Sub-timestep accuracy** - Retarded field calculations with chrono-match interpolation
- **Time residual reduction** - 10-100× improvement for ultra-relativistic simulations (γ > 100)
- **Advanced SC options** - Chrono-matching integrated with self-consistency iterations
- **Configurable interpolation** - Optional feature enabled via `SelfConsistencyConfig(chrono_interpolate=True)`

### Self-Consistency Improvements (December 9-16, 2025)

- **Mass-shell constraint** - Enforces Pt² = P² + (mc)² through iterative projection
- **Dual self-consistency** - Added dual weighting methods for gamma reconciliation
- **Variable geometry SC** - Self-consistency iterations account for changing particle positions
- **Debug logging** - Comprehensive logging of SC convergence for diagnostics
- **Step number tracking** - Added step numbers to all log output for easier debugging

### Critical Physics Corrections (December 2025)

- **Scalar potential fix** - Corrected dimensional error in electromagnetic potential calculation
- **Kinetic energy separation** - Properly subtracts potential energy (q·Φ) from conjugate energy
- **Gamma calculation** - Fixed inconsistency between energy-derived and velocity-derived gamma
- **Charge sign handling** - Corrected charge sign usage in field calculations
- **Float64 precision** - Upgraded all calculations to double precision throughout
- **k_factor threshold** - Relaxed to 1e-20 for extreme angle handling

### GUI and Configuration (December 11-14, 2025)

- **Config save/load** - Simplified configuration persistence behavior
- **Directory structure** - Improved organization of configs/ and results/ directories
- **Stability tab** - Reorganized stability controls with proper parameter greying
- **Mass-shell tolerance** - Added configurable tolerance to GUI stability settings
- **Graceful shutdown** - Better cleanup on Ctrl+C and GUI close events

## November 2025

### Image Charge Weighting (November 2025)

- **Radial weighting** - Basic radially asymmetric weighting of image subcharges
- **Distance-based attenuation** - Stricter limits for subcharge weighting distances
- **API exposure** - Weighting options exposed to API and GUI
- **GUI plot sizing** - Fixed window sizing issues in plot displays

### License and Project Setup (November 2025)

- **GPL license** - Changed project license to GPL
- **License file** - Added LICENSE file to repository

## Summary (February 2026)

### Adaptive Timestep Auto-Calculation (February 10, 2026)

- **Auto-calculated max attempts** - `max_refinement_attempts` now computed from `timestep_reduction_factor` and `min_timestep_factor` to ensure minimum timestep is always reachable
- **Auto-calculated substep cap** - `max_substeps_per_step` computed from `min_timestep_factor` with safety margin to prevent time discontinuities
- **Simplified configuration** - Only 2 independent parameters required (reduction_factor, min_factor); derived values calculated automatically
- **GUI improvements** - Read-only displays show calculated values with explanatory tooltips
- **Parameter consistency** - Eliminates configurations where min_timestep is unreachable within max_attempts
- **Optimization plugin fixed** - Removed obsolete `adaptive_timestep_max_attempts` parameter causing TypeError in sweeps

### Batched Logging for GUI Responsiveness (February 10, 2026)

- **Batch aggregation** - Debug messages buffered and flushed in batches (default 50 messages) instead of individual GUI updates
- **Logger parameter** - `retarded_integrator()` accepts optional `logger` callable for custom logging backends
- **Throttled updates** - Reduces GUI event queue flooding by ~100× during verbose debugging
- **Preserved diagnostics** - All debug messages still captured; only GUI update frequency reduced
- **Backward compatible** - Falls back to print() if no logger provided

### Gamma Reconciliation Default Changed (February 10, 2026)

- **Now DISABLED by default** - Changed from ADAPTIVE_WEIGHTED to DISABLED for v0.4.8 compatibility
- **Energy conservation issue** - Original implementation overwrote Pt without preserving scalar potential (q·Φ), violating energy conservation
- **Opt-in feature** - Five methods still available (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, DISABLED) but require explicit enablement
- **Momentum rescaling removed** - Spatial momentum no longer rescaled by default, preventing trajectory alterations
- **Legacy behavior restored** - Default matches v0.4.8 stable version behavior
- **Detailed documentation** - Changelog and configuration notes document the safe migration path

## January 2025

### Gamma Reconciliation Configuration (January 2025)

- **Configurable reconciliation methods** - Five methods available: ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, and DISABLED (now default)
- **Velocity-dependent weighting** - ADAPTIVE_WEIGHTED method uses β-dependent weights: trust energy at low β (<0.9), trust velocity at high β (>0.99), balanced in mid-range
- **Custom threshold tuning** - All thresholds and weights configurable via API and GUI for ultra-relativistic particles or specific physics regimes
- **GUI controls** - Gamma Reconciliation panel in Stability → Self-Consistency with method dropdown and parameter fields that show/hide dynamically
- **Backward compatibility** - Old `gamma_reconciliation_enabled` boolean replaced with method enum; historical configs should now use the enum directly
- **Important note** - Feature disabled by default (Feb 2026) due to energy conservation issues; requires redesign before safe re-enablement

### Transverse Offset GUI Improvements (January 2025)

- **Context-aware visibility** - Transverse offset fields now grayed out (disabled) when not in BUNCH_TO_BUNCH mode
- **Visual feedback** - Labels turn gray and entries disable automatically when simulation type changes
- **Usage guidance** - Informational notes and tooltips explain that offsets define bunch center positions and are only used in BUNCH_TO_BUNCH simulations
- **Improved clarity** - Reduces user confusion about when/how transverse offset parameters are used
- **Original demo compatibility** - More flexible than legacy (independent x/y for each bunch) while maintaining backward compatibility

### Transverse Offset and Legacy Code Isolation (January 21, 2025)

- **Transverse offset parameters** - New `transv_offset_x` and `transv_offset_y` fields separate beam center position from beam spread
- **Beam positioning** - Particles now distributed in `[offset ± spread]` allowing off-axis beams with controllable size
- **Core bunch initialization** - New `input_output.bunch_initialization.create_bunch_from_params()` replaces legacy initialization for normal operation
- **Legacy code isolation** - Legacy initialization was isolated from normal operation; active legacy comparison code has since been removed in favor of maintained core paths and reference notebooks
- **GUI integration** - Offset fields automatically appear in Particles tab for both rider and driver bunches
- **Optimization plugin fix** - "Transverse Offset" now correctly sets beam **position** (not spread), with separate `transv_dist` for beam size
- **Backward compatibility** - Old configs without offset parameters default to 0.0 (on-axis), no breaking changes

### Macroparticle Simulation (January 20, 2025)

- **Macroparticle charge scaling** - Test particle and image charges can be multiplied by configurable factor for bunch simulations
- **Stochastic position errors** - Gaussian position spread (σ_x in mm) applied to image subcharges
- **Cumulative momentum spread** - Transverse momentum errors accumulate over timesteps: σ_total(step) = sqrt(σ_x² + (σ_p × timestep × step / mass)²)
- **Pre-attenuation error application** - Errors applied before radial weighting calculations for physical accuracy
- **GUI integration** - Controls in Particles tab (single runs) and sweep/optimization sections with automatic greying for non-CONDUCTING_WALL modes
- **Configuration persistence** - All macroparticle parameters saved/loaded with simulation configs

### Optimization and Convergence (January 17, 2025)

- **Early stopping for Genetic Algorithm** - Automatic convergence detection stops optimization when fitness plateaus, saving 40-70% computation time
- **Configurable convergence parameters** - GUI controls for tolerance (default: 1e-6) and patience (default: 10 generations)
- **Comprehensive optimization guide** - New documentation covering sweep vs optimization workflows, metrics, and performance tuning

### Critical Physics Corrections (December 2025)

- **Corrected scalar potential calculation** - Fixed dimensional error in electromagnetic potential computation
- **Proper kinetic energy separation** - Now correctly subtracts potential energy (q·Φ) from conjugate energy to obtain kinetic gamma
- **Fixed self-consistency convergence** - Iterations now enforce the mass-shell constraint Pt² = P² + (mc)² through projection
- **Improved numerical precision** - Float64 throughout, relaxed k_factor threshold to 1e-20 for extreme angles
- **Self-consistency enabled by default** - Essential for energy conservation in high-energy simulations
- **Chrono-match interpolation** - Sub-timestep accuracy for retarded field calculations, providing 10-100× reduction in time residual. Critical for ultra-relativistic simulations (γ > 100). Enabled via `SelfConsistencyConfig(chrono_interpolate=True)`.

**Overall Impact**: The LW Integrator has evolved from a research prototype to a production-ready tool with comprehensive GUI, robust numerical methods, and extensive validation. Energy conservation improved by 3+ orders of magnitude. COLD_START gating fixes ensure correct physics across all velocity regimes. Optimization system enables practical parameter searches. GUI provides intuitive access to all features with real-time monitoring. Self-consistency iterations maintain physical correctness in challenging scenarios. The codebase now includes significant numerical methods and features beyond the original publication.
