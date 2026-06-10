# LW Integrator

## Recent Updates (June 2026 + v0.6.0 highlights)

- **Experimental pseudo-grid reduced solver** — pseudo-grid mode now has an opt-in reduced active/passive solve path for `BUNCH_TO_BUNCH` studies. Per-step schedules select active rider/driver subsets, aggregate passive source charge onto active representatives, and reconstruct passive particles from weighted active deltas while preserving full-state outputs. Adaptive-timestep runs are supported. Reduced same-bunch space charge is also supported when each bunch keeps at least two active particles, including retarded-space-charge validation cases, using observer-specific self-excluded source-charge matrices. When causal-history pruning is enabled, supported reduced B2B solves compact live histories and record retained-start/dropped-sample diagnostics. `scripts/pseudo_grid_feasibility_probe.py` provides a lightweight zero-charge/weak-charge sanity and `N > 100` timing probe; `scripts/pseudo_grid_feasibility_matrix.py` adds small matrix sweeps including crossing, adaptive crossing, stronger-charge, longer-window, and opt-in same-bunch space-charge cases; `scripts/pseudo_grid_microbenchmarks.py` times reduced-mode scheduling, slicing, active solve, space-charge matrix, and passive reconstruction phases; retarded same-bunch source histories are cached per source particle to reduce active-solve overhead. Maintained pseudo-grid unit and regression tests live in this `LW_integrator` repository; sibling feasibility-study workspaces should remain user-like screening/probe surfaces. The mode remains experimental, and too-small active sets still fall back to the canonical full solve.
- **Parallel blind sweeps** — the Sweep/Optimization GUI now exposes a worker-count control and reuses the maintained headless `SweepRunner` parallel path when `workers > 1`. Saved sweep configs can persist `workers`, and CLI sweeps still support `-j/--workers` overrides.
- **Radiation reaction surface** — radiation-reaction mode is now configurable from the main GUI Stability tab, the Sweep/Optimization tab, saved single-run configs, saved sweep configs, and the single-run CLI via `--radiation-reaction-mode`. The user-facing default is now `medina_lad`.
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
The repository contains a modernised `core` implementation that preserves the
validated reference physics, an updated Sphinx documentation set, and a collection
of validation scripts and notebooks. The methodology is documented in the
peer-reviewed article _Relativistic beam loading, recoil-reduction, and
residual-wake acceleration with a covariant retarded-potential integrator_
([Nucl. Instrum. Methods Phys. Res. A 1069 (2024) 169988](https://doi.org/10.1016/j.nima.2024.169988),
[arXiv:2310.03850](https://arxiv.org/abs/2310.03850)).

**Documentation:** [lw-integrator.readthedocs.io/en/latest](https://lw-integrator.readthedocs.io/en/latest)

---

![LW Integrator GUI](docs/assets/gui_screenshot.png)

---

## Contents

1. [Project overview](#project-overview)
2. [Repository layout](#repository-layout)
3. [Environment setup](#environment-setup)
4. [Running simulations](#running-simulations)

---

## Project overview

- **Physics focus.** The code integrates particle trajectories using
  retarded-vector potentials and conjugate-momentum dynamics. The `core`
  package is a faithful transcription of the validated reference implementation and is kept in
  numerical lockstep by an integration test suite.
- **Self-consistency and energy conservation.** The integrator enforces the
  relativistic mass-shell constraint Pt² = P² + (mc)² through iterative
  projection during each timestep. Two modes are available:
  - **fixed_geometry (default)**: Fast iteration with fixed geometry—retarded
    distances computed once per step. Suitable for most simulations.
  - **variable_geometry**: Updates particle positions and recomputes retarded
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
  - **DISABLED**: No reconciliation. This is the maintained default for the
    current solver path.
    Configurable via API (`self_consistency_gamma_reconciliation_method` and related
    parameters) and GUI (Stability → Self-Consistency → Gamma Reconciliation).
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
  for implementation details.
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
- **GUI and CLI entry points.** The GUI (`python -m lw_integrator.gui`) provides
  an interactive interface for single runs, parameter sweeps, and optimization,
  with real-time logging and an initial-state summary panel. The `lw-simulate`
  console command (see [Command-line entry point](#command-line-entry-point)
  below) runs the same core code paths with JSON-configurable inputs. A minimal
  CLI demonstration lives in `examples/entrypoint_demo.py`.

---

## Repository layout

```
LW_integrator/
├── core/                 # Maintained integrator implementation and helpers
├── configs/              # JSON run and sweep configuration files
├── docs/                 # Sphinx configuration, sources, and build script
├── examples/
│   └── validation/       # Reference notebooks and validation examples
├── input_output/         # Particle bunch initialisation utilities
├── legacy/               # Archived notebooks (historical reference only)
├── lw_integrator/        # CLI, GUI, sweep runner, and testbed runner
├── optimization/         # Sweep/optimization engine, metrics, result I/O
├── results/              # Sweep and optimization output (git-ignored)
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
- Direct radiation-reaction mode selection from the Stability tab (`medina_lad` default for new runs)
- Interactive trajectory visualization and energy/position analysis
- Export results in multiple formats (CSV, JSON, NPZ)
- Self-consistency iteration controls for high-gamma physics

**Sweep / Optimization Modes** (Sweep/Optimization tab):

#### Blind Sweep Mode

- **Parameter sweeps** over aperture radius, particle energy, transverse offset, and starting positions
- **Sweepable fixed parameters** - mass, charge, transverse momentum, timestep, wall position
- **Auto-timestep calculation** to maintain consistent integration resolution across energy ranges
- **Parallel execution** with configurable worker count for blind sweeps (start with a modest count such as `2-4`)
- **Radiation-reaction mode selection** persisted in saved sweep configs and mirrored into runtime execution
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

Results are saved to `results/sweeps/YYYYMMDD_HHMMSS_configname/` with convergence history, best parameters, and optional trajectory data.

### Command-line entry point

Installing the project (`pip install -e .` or via a wheel) exposes the
`lw-simulate` executable. The CLI uses sensible defaults (35 MeV electron
approaching a conducting aperture, `medina_lad` radiation reaction) but accepts
both inline parameter overrides and JSON configuration files.

**Basic usage with defaults:**

```bash
lw-simulate --quiet
```

**Override specific parameters:**

```bash
lw-simulate --steps 250 --time-step 5e-4 --aperture-radius 0.5 --output run.json
```

**Run a baseline without radiation-reaction recoil:**

```bash
lw-simulate --radiation-reaction-mode off --quiet
```

**Use a configuration file:**

```bash
lw-simulate --config my_scenario.json --output results.json
```

**Run a parameter sweep from a sweep configuration:**

```bash
lw-simulate --sweep-config configs/sweep_configs/example_b2b_linked_energy_vs_driver_distance.json -j 4
```

The sweep runner writes results to `results/sweeps/YYYYMMDD_HHMMSS_configname/`
and detailed debug logs to `logcache/`. Sweeps with fewer than 100 completed
runs are automatically relocated to `results/archive/incomplete/`.

The JSON configuration file allows complete specification of simulation
parameters, particle bunches, and physics options. Example structure:

```json
{
  "steps": 1000,
  "time_step": 3e-7,
  "simulation_type": "conducting-wall",
  "aperture_radius": 0.01,
  "wall_position": 100.0,
  "rider": {
    "kinetic_energy_mev": 5.0,
    "mass_amu": 1.0,
    "charge_sign": 1.0,
    "position_z": 0.0
  }
}
```

Additional CLI options include:

- `--chrono-mode`: Retardation sampling strategy (`averaged` or `fast`)
- `--startup-mode`: Early-step handling (`cold-start` or `approximate-back-history`)
- `--radiation-reaction-mode`: Single-run radiation-reaction handling (`off`, `diagnostic_only`, `power_matched_damping`, or `medina_lad`)
- `--image-weighting` / `--no-image-weighting`: Control image charge distribution
- `--self-consistency`: Enable self-consistency iterations for ultra-relativistic particles
- `--sweep-config`: Path to a JSON sweep configuration (runs a full parameter sweep)
- `-j/--workers`: Parallel worker-process count for sweeps (start with a modest value)
- `--log-verbosity`: Override sweep log verbosity (`none`, `truncated`, or `full`)
- `--sc-verbosity`: Override self-consistency verbosity (0–3)
- `--adaptive-debug` / `--no-adaptive-debug`: Toggle adaptive timestep debug output

Run `lw-simulate --help` for complete option listing.

Conducting-wall runs apply radial weighting to image subcharges by default for
better agreement with the aperture geometry. Pass `--no-image-weighting` to
recover the historical uniform distribution when benchmarking or debugging.

Programmatic usage mirrors the console invocation: call
`lw_integrator.cli.main` with a list of CLI-style arguments. See
`examples/entrypoint_demo.py` for a ready-to-run demonstration that exercises
both patterns.

### Sweep heatmap generation

After a sweep completes you can generate publication-quality heatmaps with
the maintained packaged command `lw-generate-sweep-heatmap`:

```bash
lw-generate-sweep-heatmap results/sweeps/<sweep_dir> \
    --output gains.png --absolute-gains --log-param2 \
    --param1-min 1 --param1-max 140 --axis-param1-max 120 \
    --gain-min -50 --gain-max 50 --color-min -30 --color-max 40 \
    --num-contours 8 --no-markers --grey-zero --grey-centre 0 \
    --no-title
```

Use `--param1-max` / `--param2-max` to choose the data included in the
interpolation, and `--axis-param1-max` to crop the displayed x-axis after
interpolation. The explicit color limits keep positive and negative gain
regions visually comparable across related plots. Contour lines use reduced
alpha (0.18) and labels are automatically clamped to stay within the axes.
Overlapping labels are culled after the final layout pass so they never stack
on top of each other.

### Live sweep plotting

The maintained live sweep plotting commands are:

```bash
lw-plot-latest-live
lw-plot-from-logcache-live logcache/<sweep_log>.log
```

`lw-plot-latest-live` follows the most recent sweep log automatically.
`lw-plot-from-logcache-live` can run in static mode by default or in live mode
with `--live`.

### Saved trajectory plotting

Saved single-run trajectory files from the GUI/testbed can be plotted from the
CLI with `lw-plot-trajectory`:

```bash
lw-plot-trajectory results/testbed_runs/<run_dir>/trajectory_data_<timestamp>.json
lw-plot-trajectory results/testbed_runs/<run_dir>/trajectory_data_<timestamp>.npz
```

JSON trajectory files preserve rider/driver separation and the full core-state
history needed by the maintained plotting tools.
NPZ files use the standard compact `z/r/pz/pr/gamma` format shared with the
optimization tooling.

---

## Startup-mode default

`COLD_START` is the default startup mode for generated configs, CLI/GUI runs, and integration-style tests. Use `APPROXIMATE_BACK_HISTORY` only for explicitly labeled diagnostics or for reproducing older examples that require approximate retarded history initialization.

## Example configurations

Simple example configs live under `configs/`. Load them via the GUI
(**Configuration & Control → Load**) or pass them directly to the CLI.

### Single-run examples (`configs/run_configs/`)

| File                                                       | Description                                                                                                                                                                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `example_b2b_counter_propagating_proton_bunches.json`      | Two counter-propagating proton-mass bunches with opposite charge signs (proton/H- in the current config; 5+5 particles, 300 M stripped ions each), cold start. Demonstrates bunch-to-bunch energy exchange at non-relativistic energies. |
| `example_b2b_relativistic_proton_stationary_lead_ion.json` | Ultra-relativistic proton rider (Pz = 1,010,000 amu·mm/ns, γ ≈ 3369) approaching a stationary lead ion driver (1 T stripped ions, m = 207.2 amu). Uses `APPROXIMATE_BACK_HISTORY` startup for accurate retarded-field initialisation. |

### Sweep examples (`configs/sweep_configs/`)

**`example_b2b_linked_energy_vs_driver_distance.json`** — 80×80 log-spaced sweep of initial energy (0.5–3000 GeV) vs driver starting distance (10–100,000 mm) for counter-propagating proton-mass, opposite-charge bunches with linked rider/driver energy. Rider and driver each have 1 µm transverse spot size.

![B2B proton/H- 1 µm spot size](docs/assets/proton_proton_1micron.png)

---

**`example_b2b_linked_energy_vs_driver_distance_35um_rider.json`** — Same sweep geometry but with an asymmetric spot size: rider 35 µm, driver 0.1 µm. The larger rider spot reduces near-collision blowups for the proton/H- style counter-propagating pair.

![B2B proton/H- 35 µm rider spot size](docs/assets/proton_proton_35micron.png)

The bunch-to-bunch examples illustrate a screening regime: after the interaction
point, the driver bunch is treated as proceeding through a virtual exit aperture
that blocks direct line of sight to the rider a short distance downstream. The
maps therefore visualize residual post-screening fields rather than an
unbounded, permanently visible counter-propagating bunch interaction.

![B2B screening example heatmap](docs/assets/b2b_screening_example_heatmap.png)

---

**`example_conducting_wall_electron_aperture_sweep.json`** — 50×50 log-spaced sweep of initial energy (1–500 GeV) vs aperture radius (15–75 µm) for a single electron rider approaching a conducting wall. Rider transverse spot size 1 µm. Macroparticle mode enabled (charge multiplier ×1000, sigma multiplier ×2).

![Conducting wall electron sweep, 1 µm spot size](docs/assets/electron_aperture_0.001micron_spotsize.png)

---

**`example_conducting_wall_electron_aperture_sweep_10um_spotsize.json`** — 100×100 log-spaced sweep of initial energy (1.8–90 GeV) vs aperture radius (40–550 µm) for a single electron rider with 10 µm transverse spot size approaching a conducting wall. Macroparticle mode enabled.

![Conducting wall electron sweep, 10 µm spot size](docs/assets/electron_aperture_0.01micron_spotsize.png)
