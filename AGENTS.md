# Repository Guidelines

## Worktree lifecycle

- Whenever you create a repository worktree, add its root directory to the
  current Zed project workspace, for example with `zed --add <worktree-path>`.
- After the worktree's changes have been merged and you have verified that the
  worktree is clean, inactive, and safe to remove, prompt the user to delete it.
  Do not silently leave a finished worktree behind or delete it without the
  user's confirmation.

## Local Qwen delegation

The supervising agent may delegate low-risk, mechanical work to the local Qwen
model on the Mac when this saves time or context.

Suitable delegated work includes:

- Running already-defined tests, linters, or formatters and returning the exact
  command, exit status, relevant output, and artifact paths.
- Performing explicitly scoped local Git operations such as status, diff, log,
  add, or commit, and reporting the resulting status or diff. Never delegate
  pushes, merges, rebases, resets, force updates, branch or worktree deletion,
  or other remote, destructive, or history-rewriting operations.
- Performing simple filesystem searches or inventories and reporting exact
  paths and matches.
- Retrieving large public web or PDF inputs, preserving source URLs, extracting
  text or images, rendering PDF pages to PNG when useful, and returning
  structured notes plus artifact paths.
- Making bounded mechanical edits or file-format transformations from explicit
  instructions.

Do not delegate critical physics judgment, codebase or architecture reasoning,
interpretation of evidence, safety decisions, or final recommendations. The
supervising agent must define the scope, inspect commands, diffs, outputs, and
artifacts, and independently verify important claims and changes before relying
on or applying them. A local-model report is not validation evidence.

Do not send credentials, personal data, confidential client or proprietary
material, or unrelated workspace contents unless Ben explicitly authorizes it.

## Project Structure & Module Organization
Core simulation logic lives in `core/`, with CLI and GUI entry points in `lw_integrator/`. Sweep and optimization workflows are in `optimization/`, and particle setup helpers are in `input_output/`. Tests are primarily under `tests/unit/`, `tests/physics/`, and targeted integration-style files in `tests/`. Example configs live in `configs/run_configs/` and `configs/sweep_configs/`. Documentation sources are in `docs/source/`; static images and screenshots are in `docs/assets/`. Treat `legacy/` as reference code for regression comparison, not the default place for new work.

## Build, Test, and Development Commands
Set up a local environment with `python -m venv .venv && source .venv/bin/activate`, then install with `pip install -e ".[dev]"` or `pip install -e ".[dev,docs]"` when working on docs. Run the default test suite with `pytest`. Use markers to narrow scope, for example `pytest -m unit` or `pytest -m physics`. Run one file with `pytest tests/test_cli_gui_parity.py`. Lint and format with `ruff check .`, `black .`, and type-check active packages with `mypy core lw_integrator`. Build docs with `bash docs/build_docs.sh`; add `-w` for live rebuilds.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python lines within 88 characters. `black` is the formatter, `ruff` handles lint fixes, and `mypy` is enforced mainly for `core/` and `lw_integrator/`. Follow existing naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_CASE` for constants. Prefer small, explicit functions in numerical code and keep GUI, CLI, and physics logic separated by package.

## Testing Guidelines
Pytest is the test runner. Name files `test_*.py`, classes `Test*`, and functions `test_*`. Put fast deterministic checks in `tests/unit/`, physics validation in `tests/physics/`, and broader behavior coverage in `tests/`. Add a regression test with every bug fix, especially for numerical stability, CLI/GUI parity, or sweep/archive behavior. Mark long-running cases with `@pytest.mark.slow`.

Use these numerical-stability defaults for any further generated configs, integration-style runs, and sweep tests unless the task explicitly says otherwise. A June 2026 audit against Medina/LAD radiation reaction, same-bunch space charge, and plotted B2B sweep samples found that removing self-consistency is still unsafe, but the older 10-iteration + gamma-reconciliation + chrono-interpolation bundle is overkill.

Default integration-style test-run parameters (unless the user explicitly overrides them): use `startup_mode: "COLD_START"` (but use `INERTIAL_PREHISTORY` for runs on the exact retarded path, which requires it; Ben, 2026-10-02), prehistory separation `1000 mm`, total steps around `1200`, auto timestep calculation from relative closing speed (rather than fixed manual timestep), and trajectory sampling/output every `100` steps. Use `APPROXIMATE_BACK_HISTORY` only as an explicitly labeled diagnostic. When bypassing the CLI/GUI and calling the proper-time integrator directly, convert a desired lab-frame propagation distance to proper-time step size using `h_step = distance_mm / (gamma * beta * C_MMNS * (steps - 1))` for single-bunch coasting beams; do not use the lab-time `distance / (beta c steps)` value directly. For direct BUNCH_TO_BUNCH probes with different rider/driver gammas, choose the step size from the solver's proper-speed closing scale, e.g. counter-propagating `h_step = separation_mm / ((gamma_rider*beta_rider + gamma_driver*beta_driver) * C_MMNS * crossing_steps)`, and include post-encounter steps rather than ending exactly at first encounter.

For new configs, tests, and examples, prefer full 3D bunch initialization over the legacy z-axis-only particle form. Use `momentum_axis`, `starting_position_mm`, `transverse_distance_mm`, and `longitudinal_span_mm` as the default surface for new work, and treat `starting_distance` / `starting_Pz` / `transv_offset_x`-style configs as compatibility or legacy-comparison paths unless a task explicitly calls for them.

Default radiation-reaction setting for test runs: use Medina/LAD radiation reaction (`radiation_reaction_mode: "medina_lad"`) unless a task explicitly asks to run without it for comparison.

```json
{
  "self_consistency_enabled": true,
  "self_consistency_convergence_mode": "fixed_geometry",
  "self_consistency_target_ms_tolerance": 1e-6,
  "self_consistency_max_iterations": 2,
  "self_consistency_mass_shell_tolerance": 0.01,
  "self_consistency_verbosity": 0,
  "self_consistency_gamma_reconciliation_method": "DISABLED",
  "chrono_interpolate": false,
  "chrono_tolerance": 0.001,
  "chrono_high_precision": false,
  "chrono_adaptive_tolerance": false,
  "radiation_reaction_mode": "medina_lad"
}
```

The October 2026 corrected-integrator audit (`LW_feasibility_studies/studies/lw_eom_audit/defaults_audit/report.md`) retains two fixed-geometry force trials conservatively; they no longer repair the mass shell. Spatial momentum determines on-shell gamma and beta in every radiation mode. Turning self-consistency off changes velocity seeding and some blowup checks, so use it only in explicitly validated comparisons with independent accuracy or timestep-convergence checks, finite states, and no particle losses.

A four-iteration maximum matched two in the audited clean cases because the stopping criterion checks the mass shell, not successive forces. A small mass-shell residual is not an accuracy certificate: check trajectory convergence and the applicable population-weighted conservation ledgers.

Mass-shell relaxation is unused on the corrected path and should not be tuned; omit it from new settings (0.1, 0.7 and 1.0 gave identical terminal states). Keep gamma reconciliation (`FIXED_WEIGHTED`, `ADAPTIVE_WEIGHTED`, etc.) disabled except as a diagnostic or legacy-study option. Keep chrono interpolation separate from self-consistency and disabled by default; enable `chrono_interpolate`/`chrono_adaptive_tolerance` only for an explicit retarded-time sampling study with a tested tolerance. State radiation reaction off explicitly when it is a diagnostic control.

The sampled (non-exact) retarded path is a labelled diagnostic: it failed timestep, source-resolution and zero-chirp convergence controls in the October 2026 moderate-β benchmarks. Do not base physics claims on it without independent convergence evidence (Ben, 2026-10-04).

## Macroparticle Accounting

For macroparticle studies, keep source, observer, inertia, and diagnostics explicitly separated.

Default representative/mean-field interpretation:

- species charge `q_s`: actual charge of one physical particle or ion
- species mass `m_s`: actual mass of one physical particle or ion
- macro population `N`: physical particles represented by one simulated macroparticle
- source charge: `N q_s`
- observer charge: `q_s`
- observer mass: `m_s`
- diagnostic weight: `N`

This gives representative-particle acceleration `a ~ (q_s/m_s) E` in the collective macro-weighted source field. It is appropriate for mean-field/test-particle trajectory studies, but not by itself for closed net-energy claims.

For any energy or power claim, report population-weighted particle totals such as `sum_i N_i ΔK_i` and `sum_i N_i p_i`, not only unweighted macro means. Also account, or explicitly state missing accounting, for near-field energy, radiated energy, material/wall heat, and external work. Do not infer net electrical generation from per-particle kinetic-energy gains alone.

Radiation diagnostics must state the assumed scaling: single-particle, incoherent population (`N × P_single`), coherent population (`N² × P_single`), or a form-factor model.

Source smearing/finite-size settings are part of the physical model for macro sources; validate important claims with convergence over active count, subcharge/smearing settings, timestep, and retarded-time sampling.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Bump version to 0.6.7` and release notes like `v0.6.7: add sweep result plots to README`. Keep commits focused and descriptive. Pull requests should explain the user-visible or numerical impact, list validation performed (`pytest`, targeted markers, docs build), and include screenshots when GUI or plotting output changes. Link related issues or configs when a change affects specific simulation scenarios.

## Legacy compute paths and checkout retirement

- The former compute checkouts now live under `~/work/extracurr/<checkout-name>`.
  Results and historical run snapshots live under
  `~/work/archives/lw-compute-results-20260924`. Use these canonical locations
  for new sessions and new work; do not create new compute forks.
- `~/compute` remains a compatibility link for existing threads, scripts and
  terminals. Its checkout entries point to the canonical checkouts. Do not
  remove these links or rewrite live thread history. Update project roots and
  working directories at a convenient checkpoint, without interrupting work.
- The retired dipole implementations and magnetic-rr-plan study are preserved
  in `~/work/archives/lw-dipole-20260924`, with source/evidence, patches, Git
  bundles and a hash manifest. Treat these as archives, not active checkouts.
- Finish and merge remaining topic branches when appropriate; do not merge
  unfinished code simply for cleanup. Preserve milestones, dirty/untracked
  files and result provenance. Verify the destination and applicable tests.
  Add newly created worktrees to Zed; existing threads may keep their old paths.
- Ask Ben before deleting a checkout, compatibility link, backup, environment,
  cache or irreplaceable result. Keep frozen baseline source and evidence
  reproducible. Full recovery copies and local caches are outside `~/work`.
- Use only one machine to edit or run Git in a shared checkout at a time. Wait
  for `work-mirror` to finish syncing before switching. There is no separate
  compute sync folder now. Git registrations may still use working old-path
  links or the other machine's home: never prune or reset them merely for that
  reason. Keep portable Git links relative and verify them on both machines.
- Keep materialized environments and caches outside synced `~/work`. Ignored
  links to machine-local caches are intentional. Sync specifications, not
  environments; this overrides older `.venv` setup examples. Check the shared
  Pixi specifications before creating an environment, and never install a
  checkout as editable into a shared environment.
