# Repository Guidelines

## Project Structure & Module Organization
Core simulation logic lives in `core/`, with CLI and GUI entry points in `lw_integrator/`. Sweep and optimization workflows are in `optimization/`, and particle setup helpers are in `input_output/`. Tests are primarily under `tests/unit/`, `tests/physics/`, and targeted integration-style files in `tests/`. Example configs live in `configs/run_configs/` and `configs/sweep_configs/`. Documentation sources are in `docs/source/`; static images and screenshots are in `docs/assets/`. Treat `legacy/` as reference code for regression comparison, not the default place for new work.

## Build, Test, and Development Commands
Set up a local environment with `python -m venv .venv && source .venv/bin/activate`, then install with `pip install -e ".[dev]"` or `pip install -e ".[dev,docs]"` when working on docs. Run the default test suite with `pytest`. Use markers to narrow scope, for example `pytest -m unit` or `pytest -m physics`. Run one file with `pytest tests/test_cli_gui_parity.py`. Lint and format with `ruff check .`, `black .`, and type-check active packages with `mypy core lw_integrator`. Build docs with `bash docs/build_docs.sh`; add `-w` for live rebuilds.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python lines within 88 characters. `black` is the formatter, `ruff` handles lint fixes, and `mypy` is enforced mainly for `core/` and `lw_integrator/`. Follow existing naming: `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_CASE` for constants. Prefer small, explicit functions in numerical code and keep GUI, CLI, and physics logic separated by package.

## Testing Guidelines
Pytest is the test runner. Name files `test_*.py`, classes `Test*`, and functions `test_*`. Put fast deterministic checks in `tests/unit/`, physics validation in `tests/physics/`, and broader behavior coverage in `tests/`. Add a regression test with every bug fix, especially for numerical stability, CLI/GUI parity, or sweep/archive behavior. Mark long-running cases with `@pytest.mark.slow`.

Use these self-consistency defaults for any further generated configs, integration-style runs, and sweep tests unless the task explicitly says otherwise. These defaults come from `configs/sweep_configs/005_08_b2b_sweep_E_spread.json` / result `20260309_121938_005_08_b2b_sweep_E_spread`:

Default integration-style test-run parameters (unless the user explicitly overrides them): use `startup_mode: "COLD_START"`, prehistory separation `1000 mm`, total steps around `1200`, auto timestep calculation from relative closing speed (rather than fixed manual timestep), and trajectory sampling/output every `100` steps. Use `APPROXIMATE_BACK_HISTORY` only as an explicitly labeled diagnostic. When bypassing the CLI/GUI and calling the proper-time integrator directly, convert a desired lab-frame propagation distance to proper-time step size using `h_step = distance_mm / (gamma * beta * C_MMNS * (steps - 1))` for single-bunch coasting beams; do not use the lab-time `distance / (beta c steps)` value directly. For direct BUNCH_TO_BUNCH probes with different rider/driver gammas, choose the step size from the solver's proper-speed closing scale, e.g. counter-propagating `h_step = separation_mm / ((gamma_rider*beta_rider + gamma_driver*beta_driver) * C_MMNS * crossing_steps)`, and include post-encounter steps rather than ending exactly at first encounter.

Default radiation-reaction setting for test runs: use the radiation reaction force mode (currently `power_matched_damping`) unless a task explicitly asks to run without it for comparison.

```json
{
  "self_consistency_enabled": true,
  "self_consistency_tolerance": 1e-7,
  "self_consistency_max_iterations": 10,
  "self_consistency_verbosity": 2,
  "self_consistency_chrono_interpolate": true,
  "self_consistency_chrono_tolerance": 0.001,
  "self_consistency_chrono_high_precision": false,
  "self_consistency_chrono_adaptive_tolerance": true,
  "self_consistency_gamma_reconciliation_method": "FIXED_WEIGHTED",
  "self_consistency_gamma_reconciliation_low_beta_threshold": 0.9,
  "self_consistency_gamma_reconciliation_high_beta_threshold": 0.99,
  "self_consistency_gamma_reconciliation_low_beta_weight": 0.8,
  "self_consistency_gamma_reconciliation_high_beta_weight": 0.2,
  "self_consistency_gamma_reconciliation_mid_beta_weight": 0.5,
  "self_consistency_gamma_reconciliation_fixed_weight": 0.9
}
```

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects such as `Bump version to 0.6.7` and release notes like `v0.6.7: add sweep result plots to README`. Keep commits focused and descriptive. Pull requests should explain the user-visible or numerical impact, list validation performed (`pytest`, targeted markers, docs build), and include screenshots when GUI or plotting output changes. Link related issues or configs when a change affects specific simulation scenarios.
