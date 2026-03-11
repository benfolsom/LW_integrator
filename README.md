# LW Integrator

## Recent Updates (v0.6.0 — March 2026)

- **CLI/GUI parity** — the CLI sweep runner now calls the same core code paths
  as the GUI (`run_testbed()`, `SimulationOptions`), eliminating divergent
  behaviour between the two entry points.
- **Incomplete-sweep archiving** — sweeps with fewer than 100 completed runs
  are automatically relocated to `results/archive/incomplete/` on save
  (CLI, GUI, and library API).
- **Heatmap contour improvements** — reduced contour-line alpha, added
  edge-aware label clamping, and overlap culling so contour labels no longer
  clip at plot boundaries or stack on top of each other.
- **Driver energy sweep fix** — `driver_energy_gev` is now correctly converted
  to starting Pz in all sweep paths, fixing a bug where driver energy sweeps
  had no effect on BUNCH_TO_BUNCH simulations.

For the full history see [CHANGELOG.md](CHANGELOG.md).

---

The LW Integrator is a covariant charged-particle tracking code that evaluates
retarded Liénard–Wiechert potentials to obtain first-principles beam dynamics.
The repository contains a modernised `core` implementation that mirrors the
validated legacy solver, an updated Sphinx documentation set, and a collection
of validation scripts and notebooks. The methodology is documented in the
peer-reviewed article _Relativistic beam loading, recoil-reduction, and
residual-wake acceleration with a covariant retarded-potential integrator_
([Nucl. Instrum. Methods Phys. Res. A 1069 (2024) 169988](https://doi.org/10.1016/j.nima.2024.169988),
[arXiv:2310.03850](https://arxiv.org/abs/2310.03850)).

**Documentation:** [lw-integrator.readthedocs.io/en/latest](https://lw-integrator.readthedocs.io/en/latest)

---

## Contents

1. [Project overview](#project-overview)
2. [Repository layout](#repository-layout)
3. [Environment setup](#environment-setup)
4. [Running simulations](#running-simulations)
5. [Validation and testing](#validation-and-testing)
6. [Documentation workflow](#documentation-workflow)
7. [Versioning and release notes](#versioning-and-release-notes)
8. [Development guidelines](#development-guidelines)
9. [Support](#support)

---

## Project overview

- **Physics focus.** The code integrates particle trajectories using
  retarded-vector potentials and conjugate-momentum dynamics. The `core`
  package is a faithful transcription of the proven legacy solver and is kept in
  numerical lockstep by an integration test suite.
- **Startup strategies.** The integrator now exposes
  :class:`core.types.StartupMode`, allowing cold-start runs that suppress
  retarded forces during the short-history transient (default) or an
  `APPROXIMATE_BACK_HISTORY` mode that reconstructs a constant-velocity past
  for better legacy alignment. All entry points—CLI, scripts, and notebooks—take
  the new enum so you can toggle behaviour without patching call sites.
- **Self-consistency and energy conservation.** The integrator enforces the
  relativistic mass-shell constraint Pt² = P² + (mc)² through iterative
  projection during each timestep. Two modes are available:
  - **mass_shell_only (default)**: Fast iteration with fixed geometry—retarded
    distances computed once per step. Suitable for most simulations.
  - **full_iteration**: Updates particle positions and recomputes retarded
    distances each iteration for maximum accuracy when particles move
    significantly (|Δx| ~ 0.1×R_separation). Computationally expensive but
    accounts for geometric changes during the timestep.
    Self-consistency is enabled by default and critical for energy conservation
    in high-energy simulations (γ > 10⁴). Implemented December 2025.
- **Gamma reconciliation.** The integrator computes the Lorentz factor γ in two
  ways: from conjugate momentum (γ_energy) and from velocity (γ_velocity).
  Numerical differences between these can cause energy jumps. Five reconciliation
  methods are available via `GammaReconciliationMethod`:
  - **ADAPTIVE_WEIGHTED (default)**: Velocity-dependent blending with configurable
    thresholds and weights. Trusts energy at low β, velocity at high β.
  - **FIXED_WEIGHTED**: Fixed 50/50 blend (or custom weight).
  - **USE_VELOCITY** / **USE_ENERGY**: Use one calculation exclusively.
  - **DISABLED**: No reconciliation (legacy, not recommended).
    Configurable via API (`self_consistency_gamma_reconciliation_method` and related
    parameters) and GUI (Stability → Self-Consistency → Gamma Reconciliation).
    See `local/gamma_reconciliation_config.md` for detailed usage.
- **Adaptive timestep and beta clamping.** The integrator includes numerical
  safety features for extreme relativistic regimes (γ > 10⁶):
  - **Beta clamping** prevents particle velocities from reaching the speed of
    light (β ≥ 1), ensuring the Lorentz factor remains finite even at extreme
    energies. Velocities are automatically limited to β < 0.99999999999999999
    (17 decimal places, near the float64 precision limit) corresponding to
    ~34 TeV for electrons.
  - **Adaptive timestep refinement** detects energy jumps during integration
    and automatically retries problematic steps with smaller timesteps. This
    is configurable via `AdaptiveTimestepConfig` and particularly useful for
    high-energy electron-wall simulations.
- **Trajectory stability analysis.** Post-integration validation assesses
  whether trajectories are numerically stable across multiple timesteps, even
  in regions with strong physical forces (radiation reaction, image charges).
  Rather than rejecting runs with large single-step jumps—which can represent
  valid physics—the analyzer checks for oscillatory instabilities, erratic
  evolution that cannot fit smooth polynomial trends, and multi-scale
  inconsistencies. This multi-step approach distinguishes numerical artifacts
  from physical behavior and is essential for unattended sweep and optimization
  runs. Configured via `SmoothnessConfig` with presets for strict,
  balanced, and permissive validation. See `core/smoothness_analyzer.py`
  and `local/smoothness_checking_implementation.md` for details.
- **Macroparticle simulation.** For conducting-wall simulations, the integrator
  supports macroparticle mode where test particle charges are scaled by a
  configurable multiplier and image subcharge positions receive stochastic errors
  based on transverse position and momentum spreads. Position spread applies
  constant Gaussian errors (σ_x), while momentum spread creates cumulative
  displacement that grows with each timestep: σ_total(step) = sqrt(σ_x² +
  (σ_p × h × step / m)²). These errors are applied before charge attenuation
  calculations to accurately model beam emittance effects. Configured via GUI
  controls in the Particles tab (single runs) and optimization/sweep parameter
  sections. Only active for CONDUCTING_WALL simulation type.
- **Reference publication.** For the scientific context, derivations, and
  benchmark scenarios, see the project paper referenced above; the codebase
  tracks the configurations described there.
- **Documentation.** The refreshed Sphinx site under `docs/` explains the
  theoretical background, quick-start workflows, validation procedures, and
  contributor guidance. A new `theory` page summarises the covariant
  derivations drawn from the in-repo technical note.
- **Validation assets.** The `examples/validation` tree provides both Python
  scripts and notebooks for reproducing benchmark comparisons between the
  modern and legacy implementations. The refreshed `integrator_testbed`
  notebook surfaces legacy overlays, difference plots, and live initial-state
  summaries so physics regressions are immediately visible while you tweak
  parameters. Its widget scaffolding now lives in
  `examples/validation/testbed_ui.py` so you can import
  `IntegratorTestbedApp` into other notebooks or scripts without duplicating
  the layout logic.
- **CLI entry point.** The `lw-simulate` console command (see the
  [Command-line entry point](#command-line-entry-point) section below) runs the
  core integrator with JSON-configurable inputs. A minimal demonstration lives
  in `examples/entrypoint_demo.py`.

---

## Repository layout

```
LW_integrator/
├── core/                 # Maintained integrator implementation and helpers
├── configs/              # JSON run and sweep configuration files
├── docs/                 # Sphinx configuration, sources, and build script
├── examples/
│   └── validation/       # CLI and notebook-based comparison studies
├── input_output/         # Particle bunch initialisation utilities
├── legacy/               # Archived original solver and notebooks
│                         # (deprecated — kept for regression comparisons)
├── lw_integrator/        # CLI, GUI, sweep runner, and testbed runner
├── optimization/         # Sweep/optimization engine, metrics, result I/O
├── results/              # Sweep and optimization output (git-ignored)
├── scripts/              # Monitoring and helper scripts
├── tests/                # Pytest suite covering physics and helper modules
├── .github/workflows/    # Continuous-integration pipelines (docs publishing)
├── core/_version.py      # Single source of truth for the project version
└── README.md             # You are here
```

---

## Environment setup

### System Dependencies

Before installing Python packages, ensure you have the required system dependencies:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install python3-tk python3-dev
```

**Fedora/RHEL:**

```bash
sudo dnf install python3-tkinter python3-devel
```

**macOS:**

```bash
# Tkinter is included with Python from python.org
# If using Homebrew Python:
brew install python-tk@3.11  # adjust version as needed
```

**Windows:**

```powershell
# Tkinter is included with standard Python installers from python.org
# No additional installation needed
```

> **Note:** The GUI components require Tkinter, which is **not** pip-installable and must be installed at the system level. If you see `ModuleNotFoundError: No module named 'tkinter'`, install the appropriate system package above.

### Python Environment

1. **Create and activate a virtual environment** (Python 3.8–3.13 are supported).

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install the package in editable mode** with commonly used extras.

   **For general usage (simulation + GUI):**

   ```bash
   pip install -e .
   ```

   **For development (includes testing/linting tools and bump2version):**

   ```bash
   pip install -e ".[dev]"
   ```

   **For full installation (dev + examples + documentation):**

   ```bash
   pip install -e ".[dev,examples,docs]"
   ```

   - `dev` adds pytest, black, flake8, mypy, and bump2version for development
   - `examples` installs Jupyter and ipywidgets for interactive notebooks
   - `docs` brings in Sphinx, `sphinx-autobuild`, and related extensions

3. **(Optional) register the kernel for Jupyter usage.**

   ```bash
   python -m ipykernel install --user --name lw-integrator --display-name "LW Integrator (.venv)"
   ```

### Troubleshooting

**Matplotlib backend issues:**

If you encounter matplotlib GUI errors, try setting a different backend:

```bash
export MPLBACKEND=TkAgg  # Linux/macOS
set MPLBACKEND=TkAgg     # Windows CMD
```

Or configure it in your script:

```python
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
```

**Version management (developers only):**

The `bump2version` tool is included in the `dev` extras. If you only installed the base package (`pip install -e .`), you'll need to install dev dependencies to use versioning tools:

```bash
pip install -e ".[dev]"
```

---

## Running simulations

### GUI Application

The project includes a Tkinter-based GUI for single runs, parameter sweeps, and optimization:

```bash
python -m lw_integrator.gui
```

> **Note:** The GUI and CLI now share the same core integration code paths
> (`run_testbed()` / `SimulationOptions`), so results are identical regardless
> of which interface you use.

The GUI provides three operational modes:

**Single Run Mode** (Main tab):

- Configure and run individual simulations with real-time progress tracking
- Full control over particle properties, boundary conditions, and physics parameters
- Interactive trajectory visualization and energy/position analysis
- Export results in multiple formats (CSV, JSON, NPZ)
- Self-consistency iteration controls for high-gamma physics

**Sweep / Optimization Modes** (Sweep/Optimization tab):

#### Blind Sweep Mode

- **Parameter sweeps** over aperture radius, particle energy, transverse offset, and starting positions
- **Sweepable fixed parameters** - mass, charge, transverse momentum, timestep, wall position
- **Auto-timestep calculation** to maintain consistent integration resolution across energy ranges
- **Trajectory saving** with configurable stride
- Results saved to timestamped directories with JSON summary and plots

#### Optimization Mode

- **Multiple algorithms**: Genetic Algorithm, Differential Evolution, Nelder-Mead, Multi-start
- **Convergence detection**: Early stopping when fitness plateaus (GA only, configurable tolerance and patience)
- **Objectives**: Maximize energy gain (%), minimize transverse deflection, or custom metrics
- **Real-time logging**: Progress tracking with generation/iteration updates
- **Top-N saving**: Automatically saves best configurations found

**Optimization Quick Start:**

1. Select "Optimization" mode
2. Choose optimizer (Genetic Algorithm recommended for global search)
3. Set convergence parameters:
   - Tolerance: 1e-6 (relative improvement threshold)
   - Patience: 10 generations (lookback window)
4. Define parameter ranges (at least 2 sweep dimensions required)
5. Run - optimizer automatically stops when converged or max iterations reached

**Performance Notes:**

- Early stopping can reduce runtime by 40-70% when convergence occurs
- For radiation reaction physics (stripped_ions > 10), use timestep ≤ 3e-7 ns with self-consistency enabled
- Nelder-Mead is fastest for local optimization (~15-50 min), GA/DE are thorough but slower (~1-3 hours)

Results are saved to `results/sweeps/YYYYMMDD_HHMMSS_configname/` with convergence history, best parameters, and optional trajectory data. See `local/SWEEP_AND_OPTIMIZATION_GUIDE.md` for detailed usage.

### Command-line entry point

Installing the project (`pip install -e .` or via a wheel) exposes the
`lw-simulate` executable. The CLI uses sensible defaults (35 MeV electron
approaching a conducting aperture) but accepts both inline parameter overrides
and JSON configuration files.

**Basic usage with defaults:**

```bash
lw-simulate --quiet
```

**Override specific parameters:**

```bash
lw-simulate --steps 250 --time-step 5e-4 --aperture-radius 0.5 --output run.json
```

**Use a configuration file:**

```bash
lw-simulate --config my_scenario.json --output results.json
```

**Run a parameter sweep from a sweep configuration:**

```bash
lw-simulate --sweep-config configs/sweep_configs/005_08_b2b_sweep_E_spread.json
```

The sweep runner writes results to `results/sweeps/YYYYMMDD_HHMMSS_configname/`
and detailed debug logs to `logcache/`. Sweeps with fewer than 100 completed
runs are automatically relocated to `results/archive/incomplete/`.

The JSON configuration file allows complete specification of simulation
parameters, particle bunches, and physics options. Example structure:

```json
{
  "simulation": {
    "steps": 1000,
    "time_step": 3e-7,
    "simulation_type": "conducting-wall",
    "aperture_radius": 0.01,
    "wall_position": 100.0
  },
  "rider": {
    "mass": 1.0,
    "charge": 1.0,
    "energy": 5.0,
    "x0": 0.0,
    "y0": 0.0,
    "z0": 0.0
  }
}
```

Additional CLI options include:

- `--chrono-mode`: Retardation sampling strategy (`averaged` or `fast`)
- `--startup-mode`: Early-step handling (`cold-start` or `approximate-back-history`)
- `--image-weighting` / `--no-image-weighting`: Control image charge distribution
- `--self-consistency`: Enable self-consistency iterations for ultra-relativistic particles
- `--sweep-config`: Path to a JSON sweep configuration (runs a full parameter sweep)
- `--log-verbosity`: Override sweep log verbosity (`none`, `truncated`, or `full`)
- `--sc-verbosity`: Override self-consistency verbosity (0–3)
- `--adaptive-debug` / `--no-adaptive-debug`: Toggle adaptive timestep debug output

Run `lw-simulate --help` for complete option listing.

Conducting-wall runs apply radial weighting to image subcharges by default for
better agreement with the aperture geometry. Pass `--no-image-weighting` to
recover the legacy uniform distribution when benchmarking or debugging.

Programmatic usage mirrors the console invocation: call
`lw_integrator.cli.main` with a list of CLI-style arguments. See
`examples/entrypoint_demo.py` for a ready-to-run demonstration that exercises
both patterns.

### Sweep heatmap generation

After a sweep completes you can generate publication-quality heatmaps with
`generate_sweep_heatmap.py`:

```bash
python generate_sweep_heatmap.py results/sweeps/<sweep_dir> \
    --no-title --output gains.png --absolute-gains --log-param2 \
    --energy-max 1000 --num-contours 8 --no-markers --grey-zero \
    --grey-centre 0 --gain-max 200 --energy-min 1 --log-colorbar
```

Contour lines use reduced alpha (0.18) and labels are automatically clamped
to stay within the axes. Overlapping labels are culled after the final
layout pass so they never stack on top of each other.

---

## Validation and testing

The canonical comparison between the core and legacy solvers lives in
`examples/validation/core_vs_legacy_benchmark.py`. Execute it directly:

```bash
python examples/validation/core_vs_legacy_benchmark.py --seeds 0 1 2 --steps 5000 --plot
```

The script accepts additional options for output paths, DPI control, and plot
styling. Consult `--help` for the full list. Companion notebooks in the same
directory expose an interactive widget-driven interface for exploratory work.
The notebook delegates all widget construction to
`examples/validation/testbed_ui.py`; instantiate `IntegratorTestbedApp` to
embed the UI in your own notebook or lab book without copying code cells.

> **Note:** The interactive notebooks are tested and supported in JupyterLab
> and the classic Jupyter Notebook or Jupyter Lab interface. The VS Code notebook editor is
> known to trigger duplicate plot rendering.

The `tests/` directory contains deterministic Pytest suites that ensure
physics parity across configurations:

```bash
pytest tests
```

---

## Documentation workflow

All documentation sources are under `docs/source/`. The helper script
`docs/build_docs.sh` wraps `sphinx-build` and `sphinx-autobuild`.

- **One-off build** (HTML):

  ```bash
  cd docs
  ./build_docs.sh --clean --type html
  ```

- **Live preview with automatic reload** (requires `sphinx-autobuild`):

  ```bash
  cd docs
  ./build_docs.sh --clean --watch
  ```

  The preview runs at `http://localhost:8000` as long as the process remains
  active.

GitHub Actions publishes the rendered site to GitHub Pages whenever the `main`
branch is updated. Every build also uploads the HTML artefact so intermediate
branches can download the output for review.

---

## Versioning and release notes

The project version is defined exactly once in `core/_version.py`. Both
`pyproject.toml` (via `setuptools.dynamic`) and `docs/source/conf.py` import
that value, ensuring the wheel metadata and Sphinx footer remain consistent.

This project uses **bump2version** for automated version management. To cut a new release:

```bash
# Bump patch version (0.6.0 → 0.6.1)
bump2version patch

# Bump minor version (0.6.0 → 0.7.0)
bump2version minor

# Bump major version (0.6.0 → 1.0.0)
bump2version major

# Push release
git push origin master development --tags
```

This automatically updates `core/_version.py`, creates a commit, and tags the release.

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development and release workflows.

---

## Development guidelines

- Maintain parity between the `core` and `legacy` solvers when modifying
  physics-critical code. New behaviours should be backed by updated validation
  plots and regression tests.
- Prefer the helper utilities in `input_output/` when constructing particle
  bunches. They guarantee the integrator receives correctly shaped state
  dictionaries.
- Run the Pytest suite and build the documentation before submitting changes.
  The repository treats Sphinx warnings as errors to keep the rendered site
  trustworthy.
- For high-energy simulations (γ > 10⁴), self-consistency is now enabled by
  default with `mass_shell_only` mode (fixed geometry, fast). For maximum
  accuracy when particle motion during timesteps is significant, use
  `SelfConsistencyConfig.full_iteration()`. To disable (not recommended),
  explicitly set `SelfConsistencyConfig(enabled=False)`.

---

## Support

Discussion of new physics scenarios, validation additions, or documentation
improvements is welcome via GitHub issues. When reporting a problem, please
include:

- the observed behaviour and expected outcome,
- the relevant configuration (energy range, simulation type, etc.), and
- reproduction steps or sample notebooks.

For background reading on the theoretical model, consult `docs/source/theory`
and the accompanying technical note under `LW_local_refs/main.tex`.
