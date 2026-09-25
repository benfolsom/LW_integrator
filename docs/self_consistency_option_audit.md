# Self-consistency and related option audit

**Status:** audit completed, and the user-approved decisions are now
implemented (see "Implemented changes" and "Migration / effect on saved
configs" at the end). The option-trace, default and issue sections describe
the audited revision **before** those changes, so they remain the record of
what was found; each issue's resolution is listed in the implemented-changes
table.

**Runtime revision inspected:** `development` at `7c8e235` (v0.10.0, "Merge
many-particle full-spin milestone"). The audit worktree is on branch
`audit/self-consistency-options`. Line numbers below refer to that revision
plus the first-pass fixes (SC-4, SC-5); the implementation pass shifted some
lines in the edited files. Probes used
the integrator's `.venv`: Python 3.12.12, NumPy 2.4.2, Numba 0.64.0. Date:
2026-09-23.

**Scope:** the deferred item "Deferred self-consistency and GUI-options audit"
in the aneutronic study's `dependency_audit.md`. It builds on
[coupled integration and option audit](coupled_integration_and_option_audit.md).
That audit found that the older controls have live consumers and do not
belong in the nonlinear pair runner. Its conclusion still holds.

The options audited are self-consistency (SC) iteration, gamma reconciliation,
chrono matching (interpolation, tolerance, high precision, adaptive
tolerance) and the chrono/retardation matching mode.

## Summary

- **SC iteration is live only in the older charge/Jakobsen equations of
  motion** (`core/equations.py`). The same routine runs for the exact
  RFS/dipole path, but that path is restricted to `fixed_geometry`, bypasses
  chrono matching and uses an on-shell kinematic boundary. There, SC on/off
  differed only at round-off (6e-15) in a bounded probe. The nonlinear pair,
  many-particle and shared-time adaptive runners do not read any of these
  options.
- **Effective defaults depend on the entry point.** A config without SC keys
  runs with SC **disabled** in `python -m lw_integrator.cli`. The same config
  runs with SC enabled and 10 iterations through the GUI/testbed, and with SC
  enabled and 5 iterations through sweep/optimization (SC-1).
- **Gamma reconciliation is structurally inert** with `medina_lad` (the default
  radiation mode) and on the exact path. In both cases β comes from the on-shell
  mechanical momentum, so γ_velocity equals γ_energy. Elsewhere it only
  reseeds the next SC iteration. In a bounded `diagnostic_only` probe, its
  effect was ≤ 4e-8 relative (SC-6).
- **`chrono_high_precision` does nothing in the maintained runner.** The runner
  always uses the structure-of-arrays (SOA) chrono path, which never enables
  cubic or position interpolation. The GUI checkbox is misleading (SC-7).
- **`max_iterations < 2` with SC enabled always fails** at the first step.
  The mass-shell "safety net" projection is unreachable with any shipped
  default. Documentation described both behaviors incorrectly, and that has
  been fixed (SC-2, SC-3).
- Two parsing bugs were fixed: sweep/plugin alias precedence (SC-4) and the
  testbed's silent fallback to `ADAPTIVE_WEIGHTED` for unknown names (SC-5).
- Removing SC in the older charge path is **not** a no-op. In the bounded
  probes it changed final states by about 1e-5 relative and changed
  population-weighted kinetic energy by about 17%. This supports the current
  AGENTS guidance to retain SC.

## Solver paths

| Label | Selection | Code |
| --- | --- | --- |
| **Older charge path** | Any `retarded_integrator` run that is not the exact path below. This includes conducting wall, switching wall, bunch-to-bunch (B2B) with `COLD_START` or `APPROXIMATE_BACK_HISTORY`, RFS with `COLD_START`, and pseudo-grid. | `core/equations.py` `retarded_equations_of_motion` via `core/self_consistency.py:self_consistent_step` |
| **Exact RFS/dipole path** | `magnetic_dipole.enabled`, `spin_model == "rfs_minimal_2021"`, `INERTIAL_PREHISTORY`, `BUNCH_TO_BUNCH` | `core/equations.py:1858-1863` (`exact_endpoint_recomposition_selected`), `core/exact_pair_integration.py`, `core/exact_pair_trial.py` |
| **Nonlinear runners** | `lw_integrator.nonlinear_pair`, `nonlinear_particles`, `nonlinear_pair_adaptive` (shared lab time; DOP853 or RK4/RK3) | `lw_integrator/nonlinear_pair.py:63-67` declares `inapplicable_controls` |

## Option trace

Save keys are top-level JSON keys unless stated. "Alias" means the legacy
`self_consistency_chrono_*` spelling. The GUI writes both spellings with equal
values.

| Option (GUI label) | Saved key(s) | CLI `lw_integrator.cli` | GUI / testbed `SimulationOptions` | Sweep / optimization `OptimizationConfig` | Runtime consumer |
| --- | --- | --- | --- | --- | --- |
| Enable self-consistency iterations (recommended) | `self_consistency_enabled` | passthrough `cli.py:1770-1797`; `bool(...)`, default **False** (`cli.py:3032`); returns `None` when SC and all chrono flags are off | dataclass **True** (`testbed_runner.py:339`); `from_dict` **True** (`:1579`) | **True** (`optimization/config.py:279`); persisted default True | `equations.py:243`; loop exits after one pass when disabled (`:4913`) |
| Convergence mode | `self_consistency_convergence_mode` | default `fixed_geometry`; not validated | canonicalized aliases `mass_shell_only`, `full_iteration` (`self_consistency.py:56-64`); not validated | as testbed | `fixed_geometry` / `variable_geometry`; unknown → `ValueError` at SC iteration 1 (`equations.py:4020`); exact path rejects `variable_geometry` (`integration_runner.py:3195-3207`, `exact_pair_trial.py:104-109`) |
| Target MS tolerance | `self_consistency_target_ms_tolerance` | default 1e-6 | 1e-6 | 1e-6 | convergence check `equations.py:4924-4969` |
| Max iterations | `self_consistency_max_iterations` | default **2** (`cli.py:3063`) | dataclass **10** (`:345`); `from_dict` **10** (`:1587`); fresh GUI 10 | **5** (`config.py:283`); persisted default 5 (`plugin_persistence_helpers.py:13`) | `range(max_iterations)` `equations.py:2115`; exhaustion raises `SelfConsistencyNonConvergenceError` (`:5044-5049`) |
| Mass-shell tolerance (safety net) | `self_consistency_mass_shell_tolerance` | 1e-2 | 1e-2 | 1e-2 | post-loop projection `equations.py:5061-5120`, reachable only if `target_ms_tolerance > mass_shell_tolerance` (SC-3) |
| Relaxation weight | `self_consistency_mass_shell_relaxation` | 0.7 | 0.7 | 0.7 | Pt-only relaxation `equations.py:4011-4017` |
| Verbosity | `self_consistency_verbosity` | 0 | dataclass **2** (`:354`); `from_dict` **0** (`:1594`); fresh GUI 2 | **2** (`config.py:286`); persisted **0** | logging only |
| Sweep dialog "Tolerance" (legacy) | `self_consistency_tolerance` | ignored | stored, never consumed | copied from target MS var (`plugin_control_mixins.py:139-142`) | none (see SC-15) |
| Gamma reconciliation, Method | `self_consistency_gamma_reconciliation_method` | default `DISABLED`; unknown name raises `SimulationConfigError` | default `DISABLED`; unknown name **now raises** (was silent `ADAPTIVE_WEIGHTED`, SC-5) | `DISABLED` | `equations.py:4417-4468`; seed only; inert when β comes from on-shell momentum (SC-6) |
| Adaptive weighted parameters (low/high β threshold, low/high/mid weight) | `self_consistency_gamma_reconciliation_{low,high}_beta_threshold`, `..._{low,high,mid}_beta_weight` | **dropped**; no passthrough, builder ignores them (SC-11) | parsed | parsed | `equations.py:4441-4461` |
| Fixed weight (α) | `self_consistency_gamma_reconciliation_fixed_weight` | parsed, default 0.5 | 0.5 | 0.5 | `equations.py:4438-4442` |
| Enable chrono-match interpolation | `chrono_interpolate`, alias | canonical wins (`cli.py:3013-3017`); `--chrono-interpolate` | `from_dict`: canonical wins (`:1596`); direct construction ORs both (`:611-614`) | loader: canonical wins (**fixed**, SC-4); direct construction ORs both (`config.py:413-416`) | `equations.py:960-1002` (external), `:1996-2008`, `:2611-2636` (same-bunch space charge) |
| Chrono tolerance (ns) | `chrono_tolerance`, alias | canonical wins | canonical wins; direct construction uses alias when canonical equals 1e-3 | as testbed, loader fixed | interpolation threshold; warning threshold when verbosity ≥ 2 |
| High-precision mode (cubic + position interpolation) | `chrono_high_precision`, alias | canonical wins | canonical wins; OR on direct construction | as testbed, loader fixed | **no effect in maintained runner** (SC-7) |
| Adaptive tolerance (auto-scale with timestep) | `chrono_adaptive_tolerance`, alias | canonical wins | as above | as above | tolerance = 0.1 h (`distances.py:465-467`); only matters with interpolation |
| (no GUI control) runtime retardation mode | CLI `chrono_mode` (`"fast"`/`"averaged"`); testbed `chrono_matching_mode`, alias | `chrono_mode`, default `fast` (`cli.py:71`); **ignores `chrono_matching_mode`** (SC-12) | `chrono_matching_mode` → `build_chrono_mode_enum` (`testbed_runner.py:3472`); GUI always writes `FAST` (`gui_config_mixins.py:1750`) | carried to testbed options | `chrono_mode` argument of `retarded_integrator`; API default **AVERAGED** (`integration_runner.py:2803`) (SC-10) |
| (carried) `SelfConsistencyConfig.chrono_matching_mode` | from `chrono_matching_mode`/alias | set from `chrono_mode` (`cli.py:1849-1852`) | set | set | **never read by the solver** (docstring corrected) |

### Default per entry point, missing keys

(Audited revision. After the implementation pass every column is enabled /
`fixed_geometry` / 2 iterations / verbosity 0 / DISABLED / chrono off / FAST.)

| Setting | Core `SelfConsistencyConfig()` | CLI JSON | Fresh GUI | Testbed `from_dict` | Sweep JSON / `OptimizationConfig()` | Plugin persisted load | AGENTS/study guidance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| enabled | True | **False** (no SC config) | True | True | True | True | True |
| max_iterations | 10 | **2** | 10 | 10 | **5** | 5 | 2 |
| verbosity | 0 | 0 | 2 | 0 | 2 | 0 | 0 |
| gamma method | DISABLED | DISABLED | DISABLED | DISABLED | DISABLED | DISABLED | DISABLED |
| chrono interp/hp/adaptive | off | off | off | off | off | off | off |
| retardation mode | n/a | FAST | FAST (hard-coded) | FAST | FAST | FAST | FAST |
| radiation reaction | n/a | medina_lad | medina_lad | medina_lad (`off` if RFS enabled) | medina_lad | from data | medina_lad |

Reproduction: call `cli._build_self_consistency_config({})`, then
`build_self_consistency_config(SimulationOptions())`, then
`SimulationOptions.from_dict({})` and `OptimizationConfig()`, and print
`enabled`, `max_iterations` and `verbosity`. The observed values are exactly
the ones shown in the table.

## Applicability by solver path

| Option | Older charge path | Exact RFS/dipole path | Nonlinear runners |
| --- | --- | --- | --- |
| SC enabled / iterations | Live. Removing SC changed probe results by about 1e-5 relative. | Runs, but uses an on-shell kinematic boundary (`equations.py:2095-2097`). SC on vs off differed by 6.0e-15 (round-off). | Not read (`inapplicable_controls`). |
| convergence mode | fixed / variable both live; variable recomputes retarded geometry. | `variable_geometry` rejected (explicit error). | Not read. |
| target / safety tolerances, relaxation | Live (safety net see SC-3). | Live but trivially satisfied. | Not read. |
| gamma reconciliation | Seed only. Inert with `medina_lad`. ≤ 4e-8 relative with `diagnostic_only`/`off`. | Inert (bit-identical in probe). | Not read. |
| chrono interpolate / tolerance / adaptive | Live for external sources and retarded same-bunch space charge (about 1e-6 relative in probe). | External chrono matching bypassed (`equations.py:2257`); space charge rejected on this path (`integration_runner.py:3175-3180`). **No-op** (bit-identical). | Not read. The light cone is solved inside published polynomial histories. |
| chrono high precision | No-op (SOA path, SC-7). | No-op. | Not read. |
| retardation mode FAST/AVERAGED | Live for external sources, **except** under `APPROXIMATE_BACK_HISTORY`, which bypasses chrono matching (`equations.py:2257-2265`). Same-bunch space charge always uses FAST (`:2617`, `:2631`). | No-op (bit-identical). | Not read. |

The main window's Stability tab states that "These settings apply to BOTH single
runs AND sweeps/optimizations" (`gui_tab_mixins.py:1836`). It has no mode gating.
Every control remains active when an exact RFS/dipole run is configured. The
nonlinear pair GUI correctly states that the main-window controls do not affect
it (`nonlinear_pair_gui.py:196-197`).

## Bounded evidence

All probes ran in a few seconds each and used `retarded_integrator` directly.

**Probe A, exact path:** `tests/unit/test_inertial_prehistory.py` helpers,
with an electron and a proton at ±1e-5 mm, `covariant_retarded_point` dipole
source, `INERTIAL_PREHISTORY`, 8 steps with `h = 1e-6`, and SC `fixed_geometry`
with 2 iterations. The run changed Px/Py by up to 1.6%, so the interaction was
not negligible. Maximum absolute final-state difference against the reference
(`medina_lad`, no chrono options, DISABLED):

| Variant | max abs diff |
| --- | --- |
| chrono interpolate + high precision + adaptive | 0 |
| gamma FIXED_WEIGHTED α=0 | 0 |
| gamma USE_VELOCITY | 0 |
| retardation mode AVERAGED | 0 |
| RR off: gamma USE_VELOCITY vs DISABLED | 0 |
| RR off: SC disabled vs enabled | 6.0e-15 |

**Probe B, older charge path:** the harness from
`tests/physics/test_self_space_charge_energy.py` (16 macro-protons, B2B
`COLD_START`, 40 steps over 10 mm, instantaneous space charge, SC 2 iterations,
chrono interpolation and adaptive tolerance on). The table gives maximum
relative final-state difference against SC on/DISABLED:

| Variant | diagnostic_only | medina_lad |
| --- | --- | --- |
| SC off | 9.9e-07 | 4.4e-06 |
| FIXED_WEIGHTED 0.9 | 4.3e-09 | **0** |
| ADAPTIVE_WEIGHTED | 8.5e-09 | **0** |
| USE_VELOCITY | 4.3e-08 | **0** |
| USE_ENERGY | 0 | 0 |
| 10 iterations instead of 2 | 0 | 0 |
| FIXED_WEIGHTED, 10 iterations | 4.3e-09 | 0 |

**Probe C, older path with retarded fields:** the Probe B bunch with the driver
at 12 mm and retarded space charge. The baseline has chrono interpolation off.
Results are maximum relative final-state difference, with population-weighted
ΔK:

| Variant | diagnostic_only | medina_lad |
| --- | --- | --- |
| chrono interpolate + adaptive | 9.9e-07 | 9.8e-07 |
| + high precision | identical to row above (bit-for-bit) | identical |
| retardation AVERAGED | 0 in this geometry | 0 |
| FIXED_WEIGHTED 0.9 | 1.9e-09 | 0 |
| SC off | 1.4e-05; ΔK −8.74e5 vs −7.48e5 MeV | 1.6e-05; ΔK −8.74e5 vs −7.86e5 MeV |

Also, in Probe C all 4,242 chrono calls went through `chrono_match_indices_soa`.
The AVERAGED result shows only that this geometry was insensitive. It does not
show that AVERAGED is inert in general.

These are short, single-configuration probes. They show which controls
*can* act and give orders of magnitude. They do not establish encounter
convergence or energy accounting for production studies.

## Test coverage

**Plumbing, parsing and mechanism (no physical claim):**

- `tests/test_simulation_options.py`: round-trip of SC, gamma and chrono fields;
  legacy chrono keys; chrono without SC.
- `tests/test_cli.py`: chrono flags; a chrono-only request builds a disabled SC
  config; chrono mode parsing.
- `tests/test_optimization_config_helpers.py`, `tests/test_single_integration_helpers.py`,
  `tests/test_optimization_plugin.py`, `tests/test_optimization.py`:
  propagation of SC, gamma and chrono fields between GUI, sweep and testbed. The
  last two need Tk and could not run in this venv.
- `tests/test_cli_gui_parity.py`: sweep JSON conversion, including preserved
  gamma fields, plus deterministic testbed parity. Five conversion tests fail
  on unmodified `development` because `configs/sweep_configs/*` fixtures are
  missing.
- `tests/test_integrator_testbed.py`, `tests/unit/test_trajectory_integrator_helpers.py`:
  convergence-mode alias canonicalization and factory presets.
- `tests/unit/test_equations_helpers.py`: chrono kwargs forwarding (`:204-260`).
  Nonconvergence raising with `max_iterations=1` (`:364`). Gamma blowup
  (`:1121`). Final projection and all gamma reconciliation mode tests
  (`:1137`, `:1318-1480`), which **monkeypatch** `_check_mass_shell_convergence`
  and `_calculate_gamma_from_beta`, so they test bookkeeping rather than physics.
- `tests/unit/test_exact_pair_trial.py:644`, `tests/unit/test_inertial_prehistory.py:1965`:
  exact path rejects `variable_geometry`; fixed geometry reuses one exact field
  evaluation per direction.
- `tests/unit/test_magnetic_dipole_integration.py:245-300`: fixed vs variable
  geometry produce different spin on the non-exact dipole path. This is a
  mechanism test.
- `tests/test_self_consistency_options.py` (new): alias precedence, gamma-method
  validation, unified defaults, build-time validation, run-start warnings,
  FAST API defaults, CLI aliases/passthrough, strict booleans (see below).

**Physics assertions that involve these options:**

- `tests/physics/test_self_space_charge_energy.py`: energy-ledger checks with SC
  on (2 iterations, DISABLED, chrono interpolation on). The only
  gamma-reconciliation physics test (`FIXED_WEIGHTED` + `medina_lad`) is
  `xfail(strict=False)`, and its gamma setting is inert according to Probe B.
- `tests/physics/test_mass_shell_clamping_comparison.py`: mass-shell error
  below 1e-2 with and without SC. This is a loose bound.
- `tests/physics/test_electromagnetic_physics.py`, `test_extreme_radiation_reaction.py`,
  `test_radiation_reaction_activation.py`, `tests/unit/test_beamline_geometry_integration.py`:
  these pass `ChronoMatchingMode.AVERAGED` incidentally, so physics coverage
  mostly exercises AVERAGED, not the production FAST mode.

**Not covered:** no test asserts a physical benefit from gamma reconciliation,
chrono interpolation, chrono high precision or AVERAGED. No test compares
entry-point defaults. No test checks that the GUI hides or labels inapplicable
controls.

## Compatibility notes

- **Exact retarded histories.** The exact path allows only `fixed_geometry`
  (`integration_runner.py:3195-3207`; `exact_pair_trial.py:104-109`), bypasses
  chrono sampling and rejects space charge. SC iterations reuse the accepted
  exact field (test at `test_inertial_prehistory.py:1965`). Chrono and gamma
  options are no-ops there, but nothing rejects or flags them.
- **Shared lab time.** The shared-time nonlinear runners do not read these
  options. In the older runner, proper-time steps give per-particle lab times,
  and chrono matching or interpolation is the way retarded samples are aligned.
  That is the only role of the chrono options.
- **Adaptive stepping.** The older energy-jump controller catches
  `GammaBlowupError` from inside the SC loop and retries. Outside the
  pseudo-grid active solve (`integration_runner.py:1704`),
  `SelfConsistencyNonConvergenceError` is not retried; it aborts the run.
  The exact path rejects adaptive timestep (`:3190-3194`). DOP853 and step
  doubling in the nonlinear runners are independent of SC.
- **Canonical vs mechanical momentum.** The SC projection corrects canonical
  `Pt` only, from the mechanical mass shell with scalar and vector-potential
  terms (`_canonical_pt_from_mechanical_mass_shell`). `P` is never projected.
  γ_energy subtracts qΦ. With the on-shell boundary (`medina_lad` or exact
  path), β comes from mechanical momentum, so γ_velocity equals γ_energy by
  construction.
- **Medina RR.** The Medina force is reset each SC trial (`equations.py:2117-2119`),
  so only the final trial feeds the spin update. A Medina kick overrides the
  reconciliation seed (`:4792`). Together with the on-shell boundary, this makes
  gamma reconciliation inert under `medina_lad`.
- **Dipole formalism.** The RFS spin is advanced once per accepted step after
  the SC loop (`equations.py:5123`). On the non-exact dipole path,
  `variable_geometry` changes spin results. On the exact path it is rejected.
  Nonlinear full-spin runners have their own momentum–velocity constraint;
  there, "self-consistency" means something different (see the prior audit).

## Issues

Status is **confirmed** (reproduced or unambiguous in code) or **suspected**.
Recommendations are tentative. "Physics check" marks items that need a bounded
physics comparison before deciding.

1. **SC-1: entry-point defaults differ. Confirmed.** `cli.py:3032` and `:3063`
   use disabled/2; the testbed uses enabled/10 (`testbed_runner.py:339`, `:345`,
   `:1579`, `:1587`); sweep and optimization use enabled/5 (`optimization/config.py:279`,
   `:283`; `plugin_persistence_helpers.py:13`); verbosity is 0 or 2 depending on
   the path. The same keyless JSON gives different physics by entry point
   (reproduction under "Default per entry point"). *Recommendation:* doc fix now.
   Unifying defaults, for example enabled/2 everywhere, is a default change
   and needs a user decision. **Physics check** before changing the CLI
   default, because enabling SC changes CLI results. Probes B and C found
   2 vs 10 iterations bit-identical.
2. **SC-2: `max_iterations < 2` with SC enabled always fails. Confirmed.** The
   check runs only for `sc_iteration > 0` (`equations.py:4924`), and
   `not converged` raises (`:5044-5049`). Reproduction: Probe B with
   `max_iterations=1` raises `SelfConsistencyNonConvergenceError ... after 1
   iterations (mass-shell error=inf)` at step 1, with RR off and with
   `medina_lad`; `max_iterations=2` succeeds. Commit `985e946` (2026-06-17)
   introduced the raise. Before it, a single pass was followed by the safety
   net. There are 19 older saved study configs (May 2026 pseudo-grid "sc_cap1"
   profiles, SC enabled, `max_iterations: 1`) that can no longer run. The GUI
   accepts any integer. *Recommendation:* reject the combination when the
   config is built, with a clear message. Alternatively, restore explicit
   single-pass semantics. This is a user decision; the tooltip and docs are
   fixed now.
3. **SC-3: mass-shell "safety net" unreachable. Confirmed** by code reading
   (`equations.py:4924-4969`, `:5044-5049`, `:5061-5096`). Convergence requires
   `E_ms < target_ms_tolerance`; otherwise the step raises. With
   `target ≤ mass_shell_tolerance`, the post-loop projection cannot trigger.
   The unit test at `test_equations_helpers.py:1137` reaches it only by
   monkeypatching. *Recommendation:* retain; the docs are fixed. Consider
   labelling it as a guard for misconfiguration only.
4. **SC-4: sweep/plugin alias precedence. Confirmed and fixed.** Take a JSON
   with `chrono_interpolate: false` and a stale
   `self_consistency_chrono_interpolate: true`, or `chrono_tolerance: 1e-3`
   with alias `5e-3`. The CLI and testbed resolved it to off/1e-3; the sweep
   loader and saved sweep-plugin configs resolved it to on/5e-3
   (`OptimizationConfig.__post_init__` ORs the booleans and prefers a
   non-default alias tolerance). Fixed in `sweep_runner.py:1552-1561` and
   `plugin_persistence_helpers.py:149-164`. A scan of 46,131 JSON files in the
   integrator and study repositories found no conflicting pairs, so no
   existing result changes. Direct construction of `SimulationOptions` or
   `OptimizationConfig` with conflicting values still uses OR semantics
   (`testbed_runner.py:611-632`, `config.py:413-434`). That is documented but
   not changed.
5. **SC-5: testbed silently mapped unknown gamma methods to
   `ADAPTIVE_WEIGHTED`. Confirmed and fixed.** `build_self_consistency_config`
   now raises `ValueError`, like the CLI (`testbed_runner.py:2822-2833`). Before
   the fix, `"NONE"` produced `ADAPTIVE_WEIGHTED`. Saved configs contain only
   `DISABLED`, `FIXED_WEIGHTED` and `ADAPTIVE_WEIGHTED`.
6. **SC-6: gamma reconciliation is inert in the default production
   configuration. Confirmed.** It only changes the next-iteration seed
   (`equations.py:4412-4468`, used at `:4922`, `:2395`) and only when SC is
   enabled. `USE_ENERGY` is identical to `DISABLED`. With `medina_lad` or the
   exact path, β comes from on-shell momentum (`:2095-2097`, `:4350-4368`),
   so γ_velocity equals γ_energy, and Medina also overrides the seed (`:4792`).
   Probes A and B were bit-identical. With `diagnostic_only`, the effect is
   ≤ 4e-8 relative. Yet 3,796 saved configs, including the shipped
   `configs/run_configs/example_b2b_counter_propagating_proton_bunches.json`,
   select `FIXED_WEIGHTED`. *Recommendation:* restrict to diagnostics, as
   AGENTS already does. Warn when it is selected with `medina_lad` or the exact
   path. Consider deprecation after a **physics check** on RR-off,
   high-γ close-approach configurations.
7. **SC-7: `chrono_high_precision` has no effect in the maintained runner.
   Confirmed.** `chrono_match_indices_soa` allocates but never uses the cubic
   index arrays and always returns `use_cubic=False` (`distances.py:499-500`,
   `:629-637`). The SOA gather receives no cubic or position arguments
   (`equations.py:2419-2427`; the same applies to space charge). The runner
   always passes SOA arrays (`integration_runner.py:2236-2243`). Probe C was
   bit-identical, and every chrono call used the SOA path. Only the non-SOA
   fallback, reachable by direct API use without SOA, implements it. The GUI
   label promises "cubic + position interpolation". *Recommendation:* user
   decision. Either wire it into the SOA path, which changes numerics when
   enabled and needs a **physics check**, or relabel and deprecate it.
8. **SC-8: GUI shows active SC, chrono and gamma controls for runs where they
   do nothing. Confirmed.** On the exact path, chrono and gamma controls are
   bit-identical no-ops and SC is round-off-level (Probe A). No GUI gating exists
   (`gui_state_mixins.py:199-265` depend only on the SC and interpolation
   checkboxes). *Recommendation:* grey out or annotate these controls when the
   exact path is selected, or warn at run start. Keep the options for the
   older path. Rejecting the combination would break saved configs that carry
   default-on SC, so a warning is preferable.
9. **SC-9: chrono sub-options without interpolation are silent no-ops.
   Confirmed.** `chrono_high_precision` and `chrono_adaptive_tolerance` alone
   still create a disabled SC config (`cli.py:3033-3036`,
   `testbed_runner.py:2811-2815`), and `chrono_tolerance` only sets the
   verbosity-2 warning threshold. The GUI greys these controls out; JSON and the
   CLI accept them silently. *Recommendation:* doc, or warn.
10. **SC-10: AVERAGED as an API default. Confirmed.** `retarded_integrator`
    (`integration_runner.py:2803`), `retarded_equations_of_motion`
    (`equations.py:1756`), `chrono_match_indices(_soa)` (`distances.py:453`,
    `:646`) and `trajectory_integrator.py:183-185` when no config is given all
    default to AVERAGED. Every maintained entry point passes FAST explicitly
    (CLI `DEFAULT_SIMULATION`, testbed, `IntegratorConfig` `types.py:1118`,
    `ExactPairEOMOptions`), so AVERAGED is reached only through explicit
    selection (`--chrono-mode averaged` or testbed `chrono_matching_mode`),
    direct API calls, or tests. Many physics tests pass AVERAGED. The docs
    reserve it for `APPROXIMATE_BACK_HISTORY`, but that startup mode bypasses
    external chrono matching entirely (`equations.py:2257-2265`), and
    same-bunch space charge hard-codes FAST. *Recommendation:* doc fix for
    `ChronoMatchingMode` and the `retarded_integrator` docstring. Changing
    the API default to FAST would change direct-API and test numerics; that
    is a user decision and needs a **physics check** of the affected tests.
11. **SC-11: CLI drops adaptive gamma weights and thresholds. Confirmed.** The
    passthrough list (`cli.py:1770-1797`) and builder (`:3011-3089`) handle only
    method and fixed weight. Reproduction: a CLI JSON with
    `low_beta_threshold: 0.5` gives 0.9 in the CLI and 0.5 in the testbed.
    *Recommendation:* add the passthrough. This changes CLI numerics for such
    configs (inert with `medina_lad`), so it is a user decision.
12. **SC-12: the retardation-mode key differs by entry point. Confirmed.** The
    CLI reads `chrono_mode` and ignores `chrono_matching_mode`: a JSON with
    `chrono_matching_mode: "AVERAGED"` runs FAST in the CLI and AVERAGED in the
    testbed. The GUI writes FAST unconditionally (`gui_config_mixins.py:1750`),
    so a loaded AVERAGED config silently becomes FAST when run or saved from the
    GUI. `SelfConsistencyConfig.chrono_matching_mode` is never read by the
    solver (docstring corrected). *Recommendation:* doc now; unify keys later.
    Low impact, because AVERAGED is diagnostic.
13. **SC-13: truthiness parsing. Confirmed.** Both `bool(payload.get(...))`
    (`cli.py:3032`) and the testbed's `_bool` (`testbed_runner.py:949-950`)
    turn the string `"false"` into True. JSON booleans are unaffected.
    *Recommendation:* strict boolean parsing (reject non-bool values). Low
    priority.
14. **SC-14: unknown convergence modes are not validated when the config is
    built. Confirmed.** Neither the CLI nor the testbed validates the value. A
    value such as the former `dual_independent` fails with
    `ValueError: Unknown convergence_mode` only at the first SC iteration
    (`equations.py:4020`), after startup work. *Recommendation:* reject it
    when the config is built. This is a simple follow-up.
15. **SC-15: dead sweep stability dialog. Confirmed.**
    `_confirm_stability_options` (`optimization/plugin_control_mixins.py:565`)
    has no callers. If revived, its "Tolerance" field would copy the unused
    legacy `self_consistency_tolerance` (1e-4 in saved configs) into
    `target_ms_tolerance`. *Recommendation:* remove it, or fix it before any
    reuse.
16. **SC-16: FAST retardation formula in docstrings. Suspected doc drift.**
    `ChronoMatchingMode` and `SelfConsistencyConfig` say `Δt = R(1+β·n̂)/c`.
    The code computes `R(1+β·n̂)/(c(1-(β·n̂)²)) = R/(c(1-β·n̂))`
    (`distances.py:274-283`, `:521-526`). These agree only to first order in β.
    The sign convention of n̂ needs checking before the docstring is corrected.
17. **SC-17: shipped example uses a legacy-study bundle. Confirmed.** It uses
    `FIXED_WEIGHTED` 0.9, chrono interpolation, 10 iterations and verbosity 2,
    contrary to the AGENTS test-run guidance. It is not changed here, because
    that would change the example's results. *Recommendation:* user decision.
18. **SC-18: self-consistency documentation drift. Confirmed and fixed.**
    `docs/source/self_consistency.rst` described "dual independent" convergence
    with `target_gamma_tolerance`; both are gone, and the old examples raise
    `TypeError`. It also described the safety net as a nonconvergence
    fallback, said reconciliation updates stored γ and Pt, and called
    `ADAPTIVE_WEIGHTED` the default. The GUI max-iterations tooltip and the
    `SelfConsistencyConfig` docstrings had the same drift. These are corrected.
    The historical flowchart is kept under a "historical" note.

## Implemented changes

First pass (bug fixes) plus the user-approved decisions, all uncommitted in
the audit worktree. Focused tests are in
`tests/test_self_consistency_options.py` unless noted.

| Issue | Decision | Change | Test |
| --- | --- | --- | --- |
| SC-1 | Unify missing-key defaults | `SelfConsistencyConfig` (`max_iterations` 10→2) and `.standard()`; CLI builder (enabled default False→True); `SimulationOptions` dataclass and `from_dict` (10→2 iterations, verbosity 2→0); fresh GUI (via `SimulationOptions`, verbosity fallback 2→0); `OptimizationConfig` (5→2, verbosity 2→0); persisted plugin defaults and GUI gather fallback (5→2). `aggressive()`/`variable_geometry()` unchanged | `test_core_and_standard_preset_use_unified_defaults`, `test_keyless_cli_config_enables_self_consistency_with_unified_defaults`, `test_testbed_gui_and_sweep_entry_points_use_unified_defaults`; `tests/test_cli.py::test_build_request_carries_chrono_options_with_default_sc` (updated) |
| SC-2, SC-14 | Validate at build | `SelfConsistencyConfig.__post_init__` rejects `max_iterations < 2` when enabled and unknown modes after alias canonicalisation; CLI wraps the message in `SimulationConfigError` | `test_enabled_self_consistency_rejects_fewer_than_two_iterations`, `test_disabled_self_consistency_does_not_check_iteration_count`, `test_unknown_convergence_mode_is_rejected_after_alias_canonicalisation`; two `tests/unit/test_equations_helpers.py` tests moved from `max_iterations=1` to 2 (one now forces nonconvergence by monkeypatch) |
| SC-3 | Retain, document | Docs only | — |
| SC-4 | Fix | Canonical `chrono_*` wins in sweep and plugin loaders | 5 precedence tests |
| SC-5 | Fix | Testbed rejects unknown gamma methods | 6 tests |
| SC-6, SC-8, SC-9 | Diagnostic-only, warn | `self_consistency_runtime_warnings` / `emit_self_consistency_runtime_warnings` (`core/self_consistency.py`), called once in `retarded_integrator` (also written to the run log). GUI notes under Gamma Reconciliation and Chrono Matching | parametrized warning tests, `test_run_start_warns_once_for_gamma_reconciliation_with_medina` |
| SC-7 | Deprecate | `ChronoHighPrecisionDeprecationWarning` (a `FutureWarning`) at run start; GUI checkbox relabelled "deprecated: no effect"; docstrings updated; non-SOA fallback kept | `test_high_precision_emits_deprecation_warning`, `test_high_precision_has_no_effect_in_maintained_runner` (bit-identical) |
| SC-10 | FAST API default | `retarded_integrator`, `retarded_equations_of_motion`, `chrono_match_indices(_soa)`, `LienardWiechertIntegrator` fallback; `ChronoMatchingMode` docstring | `test_api_chrono_mode_defaults_are_fast`, `test_trajectory_integrator_without_config_uses_fast`; `tests/unit/test_trajectory_integrator_helpers.py::test_chrono_match_indices_returns_bounded_results` now passes AVERAGED explicitly (see Validation) |
| SC-11 | CLI passthrough | Five adaptive gamma keys | `test_cli_passes_through_adaptive_gamma_parameters` |
| SC-12 | CLI alias, GUI carry-through | `_resolve_file_chrono_mode` (conflicts raise); GUI stores the loaded mode (`_current_chrono_matching_mode`) | alias/conflict tests, `test_gui_carries_loaded_retardation_mode_through` |
| SC-13 | Strict booleans | CLI `_strict_optional_bool`; testbed `_strict_bool` for `self_consistency_enabled` and the chrono switches (both spellings) | `test_string_booleans_are_rejected`, `test_json_booleans_are_accepted` |
| SC-15 | Remove | `_confirm_stability_options` and `_stability_dialog_logging_defaults` (and its test) removed; now-unused `math`/`ttk` imports dropped | `test_dead_stability_dialog_is_removed` (needs Tk) |
| SC-16 | Doc fix | `n̂ = (x_obs − x_src)/R` points from the source sample to the observer; FAST computes `Δt = R(1+β·n̂)/(c(1−(β·n̂)²)) = R/(c(1−β·n̂))` (denominator clamped to ±1e-12), which equals the exact uniform-motion delay only for β parallel to n̂. Docstrings in `core/types.py` and `core/distances.py` | docs only |
| SC-17 | Update example | `example_b2b_counter_propagating_proton_bunches.json`: DISABLED, chrono off (both spellings), 2 iterations, verbosity 0, target 1e-6, explicit `medina_lad` | `test_shipped_b2b_example_uses_maintained_defaults` |
| SC-18 | Doc fix | Sphinx page, README, tooltips | — |

**SC-1 evidence (bounded probe, Probe C geometry, 16 macro-protons, 40
steps).** `cli._build_self_consistency_config({})` and
`build_self_consistency_config(SimulationOptions.from_dict({}))` are now equal
field-for-field. Running both through `retarded_integrator`:

| radiation mode | new CLI vs testbed | old CLI (SC off) vs new CLI | ΔK new / old (MeV, population-weighted) |
| --- | --- | --- | --- |
| medina_lad | 0 (bit-identical) | 1.6e-5 max rel | −7.856e5 / −8.744e5 |
| diagnostic_only | 0 (bit-identical) | 1.4e-5 max rel | −7.476e5 / −8.744e5 |

## Migration / effect on saved configs

- **Configs that set every self-consistency key explicitly** (everything the
  GUI, testbed and sweep plugin save) are interpreted as before, except for
  the rejections and warnings below.
- **Keyless CLI JSON:** now runs with SC enabled, `fixed_geometry`, 2
  iterations (previously SC off). Expect ~1e-5 relative changes in final
  states and visible changes in population-weighted energy ledgers (table
  above). Add `"self_consistency_enabled": false` to reproduce old CLI runs.
- **Keyless GUI/testbed JSON:** 2 iterations instead of 10, verbosity 0. In
  every probe 2 and 10 iterations were bit-identical (convergence at the
  second iteration), but that is not guaranteed for hard close encounters;
  nonconvergence now raises instead of iterating further. Set
  `self_consistency_max_iterations` explicitly to keep 10.
- **Keyless sweep JSON / plugin configs:** 2 iterations instead of 5,
  verbosity 0 instead of 2.
- **`max_iterations` 0 or 1 with SC enabled:** now rejected at build. The 19
  May-2026 `pseudo_grid_profile_matrix_configs/*sc_cap1*` study configs are
  affected; they already failed at step 1 since June 2026.
- **String booleans:** rejected. The scan of 46,131 JSON files in the
  integrator and study repositories found **0** non-boolean values for
  `self_consistency_enabled` or the chrono switches (either spelling).
- **Unknown gamma methods / convergence modes:** rejected. None found in the
  scan.
- **`chrono_high_precision: true`:** still loads, has no effect (as before in
  the maintained runner), and now warns.
- **Gamma reconciliation other than DISABLED:** unchanged numerics; a warning
  where it cannot act. 3,796 saved study configs select `FIXED_WEIGHTED` and 1
  `ADAPTIVE_WEIGHTED`; with `medina_lad` they are bit-identical to DISABLED.
  They were intentionally left untouched.
- **Retardation mode:** CLI JSON with `chrono_matching_mode: "AVERAGED"` now
  runs AVERAGED (previously FAST); conflicting keys raise. GUI round-trips keep
  a loaded AVERAGED value. Direct API calls that omitted `chrono_mode` now use
  FAST instead of AVERAGED (see "Validation" for the effect on the test
  suite).
- **CLI ADAPTIVE_WEIGHTED configs with custom thresholds/weights:** now honoured
  (previously replaced by defaults). Inert with `medina_lad`.
- **Shipped example** `example_b2b_counter_propagating_proton_bunches.json`
  now uses the maintained defaults; results of that example change.

## Validation

Run on 2026-09-24 in the audit worktree (with every change above), with a
clean temporary worktree of the base commit `7c8e235` as the comparison. The
base worktree was checked out with `git worktree add --detach` outside `~/work`
and removed afterwards. No packages were installed. Every run was bounded
(under ten minutes). Failures are listed as "also at base" only when the same
test ID fails on both trees.

**Environments.** Main: the integrator `.venv` (Python 3.12.12, NumPy 2.4.2,
Numba 0.64.0, SciPy 1.17.0, pytest 9.0.2, Sphinx 6.2.1; no `_ssl`, no
`_tkinter`). GUI: the shared Pixi `maxiv-dev`/`sardana-dev` environment
(Python 3.12.12, Tk 8.6, NumPy 1.26.4, Numba 0.63.1), used read-only with
`PYTHONPATH` set to the checkout.

**Commands** (from each checkout root; the default `-m 'not slow'` applies):

```sh
PY=~/work/extracurr/LW_integrator/.venv/bin/python
PYTHONPATH=$PWD MPLBACKEND=Agg $PY -m pytest -p no:anyio -p no:cacheprovider -q -rfEsx \
  tests/unit tests/physics tests/test_simulation_options.py \
  tests/test_single_integration_helpers.py tests/test_optimization_config_helpers.py \
  tests/test_integrator_testbed.py tests/test_cli_gui_parity.py tests/test_cli.py \
  tests/test_self_consistency_options.py \
  --ignore=tests/unit/test_nonlinear_pair_gui.py   # needs Tk; run below

PIXI=~/.local/share/pixi/environments/maxiv-dev-4513398708556649349/envs/sardana-dev/bin/python
PYTHONPATH=$PWD MPLBACKEND=Agg $PIXI -m pytest -p no:anyio -p no:cacheprovider -q \
  tests/test_gui.py tests/test_optimization.py tests/test_optimization_plugin.py \
  tests/test_self_consistency_options.py tests/unit/test_nonlinear_pair_gui.py
```

| Suite | Audit worktree | Base `7c8e235` |
| --- | --- | --- |
| Main (venv) | 2067 passed, 33 failed, 1 skipped, 1 xfailed, 21 deselected (slow) | 2025 passed, 33 failed, 1 xfailed, 21 deselected |
| GUI (Pixi, Tk) | 178 passed | 136 passed |
| `tests/unit/test_trajectory_integrator_helpers.py` + `test_distance_helpers.py`, after the SC-10 test edit below | 33 passed | — |

- **Main suite:** the 33 failures are the same test IDs on both trees. None
  is caused by this work. All reproduce at base: 16 in
  `test_finite_magnetic_source_force.py`/`test_translating_magnetic_shell_kinematics.py`
  (`numpy.linalg.solve` core-dimension mismatch under NumPy 2.x), 9 in
  `tests/physics/test_pseudo_grid_feasibility.py` ("too many values to unpack
  (expected 4)" in the test helper), 5 in `test_cli_gui_parity.py` (missing
  `configs/sweep_configs/005_06_b2b_sweep_E_spread.json` and
  `11topapertureE_sweep30.json`), 2 in `test_full_spin_initialization.py`
  (no `_tkinter`), and 1 in `test_medina_radiation_reaction.py`
  (`numpy.trapz` removed in NumPy 2.x). The xfail is the expected
  `test_gamma_reconciled_medina_self_space_charge_energy_proxy_target`. The
  one skip is `test_dead_stability_dialog_is_removed` (needs Tk; passes in the
  GUI run). The +42 passes are the 43 new tests less that skip.
- **GUI suite:** no failures on either tree. Audit minus base = 43 new tests
  − 1 removed test (`_stability_dialog_logging_defaults`, SC-15).
- **Removed dialog (SC-15):** at base, `_confirm_stability_options` had no
  callers, and `_stability_dialog_logging_defaults` was used only by the
  dialog and its test. A search of the integrator and study checkouts found no
  other code references (only chat logs and another integrator worktree's own
  copy).

**SC-10 (API default AVERAGED→FAST).** An AST scan of `tests/` found 87 calls
to `retarded_integrator`, `retarded_equations_of_motion` or
`chrono_match_indices(_soa)` without the mode argument, 30 that forward
`**kwargs`, and 6 config-less `LienardWiechertIntegrator` constructions
(excluding the new test file), in 17 test files. Those files were run twice in the
audit worktree: as shipped (FAST), and with a temporary pytest plugin (outside
the repository) that restores the four `AVERAGED` defaults.

| Default | Result |
| --- | --- |
| FAST (shipped) | 256 passed, 9 failed |
| AVERAGED (restored) | 256 passed, 9 failed |

The 9 failures are identical in both modes and are the pre-existing
pseudo-grid unpacking failures above. Every other test in those files passes
under both defaults, so **no test changed result**, and no tolerance was
changed. The pass/fail check cannot detect a numerical drift that stays within
tolerance. One test lost its intended coverage:
`tests/unit/test_trajectory_integrator_helpers.py::test_chrono_match_indices_returns_bounded_results`
named its default call `indices_averaged` and relied on the AVERAGED default.
It now passes `mode=ChronoMatchingMode.AVERAGED` explicitly, so both modes are
covered again. Many implicit callers do not reach sampled external chrono
matching: they mock `_compute_delta_t`, use the exact
`INERTIAL_PREHISTORY` path, or have no external source.

**Docs.** `python -m sphinx -W --keep-going -E -b html docs/source <tmp>`:
4 warnings on both trees, all intersphinx inventories that cannot be fetched
without `_ssl`. There are no other warnings, and none on the self-consistency
page. With nitpicky `-n`, the base and the audit tree have the same 206
warnings (identical after normalising paths). One new nitpicky warning,
an unresolvable `:class:` reference in the `chrono_match_indices` docstring,
was fixed by making it a literal.

**Static checks** on the changed modules. `mypy core lw_integrator`: 2820
errors at base and 2799 in the audit tree, with no new messages; the 21
removed messages were all in the deleted dialog. `pyflakes`: identical
output. `black --check`: 3 files would be reformatted, all in hunks that
already fail at base (`optimization/config.py`,
`optimization/plugin_persistence_helpers.py`,
`tests/unit/test_trajectory_integrator_helpers.py`). `ruff` is not installed
in either environment and was not run.

### Follow-up: pre-existing failures fixed (2026-09-25)

A separate commit fixes the pre-existing failures listed above:

- `core/translating_magnetic_shell_kinematics.py`: pass `np.linalg.solve` an
  explicit trailing axis. NumPy >= 2 treats a `(K, M)` right-hand side as a
  matrix (16 tests).
- `tests/physics/test_pseudo_grid_feasibility.py`: the run helpers keep the
  first four `retarded_integrator` results (9 tests).
- `tests/unit/test_medina_radiation_reaction.py`: use `np.trapezoid` when
  available (`np.trapz` was removed in NumPy 2.4).
- `tests/test_cli_gui_parity.py`: the two sweep configs are tracked in
  `tests/fixtures/sweep_configs/`, because `configs/sweep_configs/*.json` is
  git-ignored (5 tests).
- `optimization/results_mixins.py` imports Tk only inside its two dialog
  methods, so `lw_integrator.headless_optimization_runner` imports without
  Tk. `tests/conftest.py` skips the modules that need Tk at import, and two
  full-spin GUI tests `importorskip("tkinter")`.

Result: `pytest tests` (default markers) with the integrator venv (NumPy 2.4,
no Tk) and with the Pixi Python (NumPy 1.26, Tk). Tolerances are unchanged.
Before the Tk import changes, the venv run had 2376 passed and 1 failed
(`test_sweep_runner_logging.py`, which needed Tk through the headless
runner; it now passes). The Pixi run had 2562 passed and 5 failed. Those 5
are Numba bitwise-reproducibility checks
(`test_exact_retarded_charge_numba.py`, `test_retarded_dipole_fields.py`,
`test_retarded_dipole_numba_full_strict.py`) that pass with the project's
Numba 0.64 but differ in the last bits under Pixi's Numba 0.63. They were
left as they are.

## Not done

- No cubic interpolation in the SOA chrono path (SC-7 decision: deprecate).
- No GUI mode gating (greying out) for the exact path; static notes plus
  run-start warnings instead.
- The testbed's `build_chrono_mode_enum` still maps unknown mode strings to
  FAST silently, and direct `SimulationOptions`/`OptimizationConfig`
  construction with conflicting chrono aliases still ORs them; neither was in
  the approved scope.
- The 3,796 saved study configs using gamma reconciliation are unchanged.
