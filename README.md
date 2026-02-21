# LW Integrator

## Recent Updates

### COLD_START Gating Formula Fix (February 2026)

**Critical bug fix** - The COLD_START gating mechanism had a fundamentally incorrect formula for computing when retarded forces should be applied, causing massive unphysical energy losses in relativistic simulations:

- **Incorrect formula** - Used multiplication `R × (1 - β·n̂)` instead of division `R / (1 - β·n̂)`
- **4× error** - For relativistic particles approaching sources (β·n̂ = -1), threshold was 4× too large (40km instead of 10km)
- **Hardcoded limitation** - Used hardcoded `estimated_max_R = 10000 mm`, failing for separations > 10km
- **Edge case handling** - Now properly handles receding particles (β·n̂ > 0) with threshold → ∞ as β·n̂ → 1

**Corrected formula**: `threshold = R / (1 - β·n̂)` where:

- **Approaching** (β·n̂ < 0): denominator > 1 → threshold < R (particles and light meet quickly)
- **Perpendicular** (β·n̂ = 0): denominator = 1 → threshold = R (light travels full distance)
- **Receding** (β·n̂ > 0): denominator < 1 → threshold > R (light takes longer to catch up)
- **Receding at c** (β·n̂ → 1): denominator → 0 → threshold → ∞ (forces never apply)

**Impact**: All relativistic simulations with β > 0.5 were affected. The bug caused forces to be gated for too long, then activate with insufficient causal history, resulting in energy losses of 250-3200 GeV (orders of magnitude larger than physical). Now scales correctly from millimeters to hundreds of meters. See `local/COLD_START_FIX.md` for detailed analysis.

### Adaptive Timestep Refactoring (February 2026)

**Auto-calculated parameters** - The adaptive timestep system now automatically calculates derived parameters to prevent inconsistent configurations:

- **`max_refinement_attempts`** - Computed from `timestep_reduction_factor` and `min_timestep_factor` using formula: `ceil(log(1/min_factor) / log(reduction_factor))`
- **`max_substeps_per_step`** - Computed from `min_timestep_factor` with 10% safety margin: `ceil(1/min_factor) × 1.1`
- **Reduced default reduction factor** - Changed from 10 to 3 for more gradual refinement, reducing oscillation in pathological cases
- **GUI improvements** - Max attempts shown as read-only calculated value with visual feedback
- **Time discontinuity prevention** - Automatic substep cap ensures full timestep coverage even at minimum refinement level

**Impact**: Eliminates overdetermined parameter combinations that could cause time skipping or excessive refinement. Users only set two independent parameters (`reduction_factor` and `min_timestep_factor`), with derived values calculated automatically for consistency.

### Batched Logging Implementation (February 2026)

**Performance optimization** - Inner-loop debug logging now uses batched updates to prevent GUI unresponsiveness:

- **Batch aggregation** - Debug messages accumulated in memory and flushed in batches (default: 50 messages per flush)
- **Throttled GUI updates** - Reduces event queue flooding by ~100× in pathological cases (e.g., 750 messages → 8 GUI updates)
- **Logger parameter** - New optional `logger` parameter on `retarded_integrator()` accepts callable for custom logging
- **Backward compatible** - Falls back to `print()` if no logger provided; existing code unaffected
- **GUI responsiveness** - Prevents multi-minute freezes when `adaptive_timestep_debug = True` during challenging runs

**Impact**: GUI remains responsive during verbose debugging. Users can enable full adaptive timestep diagnostics without performance penalty.

### Gamma Reconciliation Default Changed (February 2026)

**Disabled by default** - Gamma reconciliation feature now defaults to `DISABLED` for v0.4.8 compatibility:

- **Energy conservation** - Original reconciliation implementation violated energy conservation by overwriting `Pt` without preserving scalar potential contribution
- **Momentum rescaling issue** - Spatial momentum rescaling altered particle trajectories incorrectly
- **Opt-in feature** - Reconciliation methods (ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, etc.) still available but require explicit enablement
- **Legacy behavior restored** - Default configuration matches v0.4.8 stable behavior: `gamma_reconciliation_method = DISABLED`

**Impact**: Eliminates silent energy non-conservation for users upgrading from v0.4.8. Feature requires redesign before safe re-enablement (see `local/GAMMA_RECONCILIATION_FIX.md`).

### Verbose Logging in Sweep/Optimization (v0.4.2+)

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
4. [Running validation workloads](#running-validation-workloads)
5. [Documentation workflow](#documentation-workflow)
6. [Versioning and release notes](#versioning-and-release-notes)
7. [Development guidelines](#development-guidelines)
8. [Support](#support)

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
    in high-energy simulations (γ > 10⁴). Implemented December 2024.
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
LW_windows/
├── core/                 # Maintained integrator implementation and helpers
├── docs/                 # Sphinx configuration, sources, and build script
├── examples/
│   └── validation/       # CLI and notebook-based comparison studies
├── input_output/         # Particle bunch initialisation utilities
├── legacy/               # Archived original solver and notebooks
│                         # The historical "static" integrator remains here for
│                         # completeness; it is deprecated and not used by the
│                         # modern docs or validation workflows.
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
`lw-simulate` executable. The CLI uses sensible defaults (35 MeV electron approaching
a conducting aperture) but accepts both inline parameter overrides and JSON configuration files.

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

The JSON configuration file allows complete specification of simulation parameters, particle bunches,
and physics options. Example structure:

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

Run `lw-simulate --help` for complete option listing.

Conducting-wall runs apply radial weighting to image subcharges by default for
better agreement with the aperture geometry. Pass `--no-image-weighting` to
recover the legacy uniform distribution when benchmarking or debugging.

Programmatic usage mirrors the console invocation: call
`lw_integrator.cli.main` with a list of CLI-style arguments. See
`examples/entrypoint_demo.py` for a ready-to-run demonstration that exercises
both patterns.

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

## Recent changes (February 2026)

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
- **Detailed documentation** - See `local/GAMMA_RECONCILIATION_FIX.md` for analysis and migration guide

## Recent changes (January 2025)

### Gamma Reconciliation Configuration (January 2025)

- **Configurable reconciliation methods** - Five methods available: ADAPTIVE_WEIGHTED, FIXED_WEIGHTED, USE_VELOCITY, USE_ENERGY, and DISABLED (now default)
- **Velocity-dependent weighting** - ADAPTIVE_WEIGHTED method uses β-dependent weights: trust energy at low β (<0.9), trust velocity at high β (>0.99), balanced in mid-range
- **Custom threshold tuning** - All thresholds and weights configurable via API and GUI for ultra-relativistic particles or specific physics regimes
- **GUI controls** - Gamma Reconciliation panel in Stability → Self-Consistency with method dropdown and parameter fields that show/hide dynamically
- **Backward compatibility** - Old `gamma_reconciliation_enabled` boolean replaced with method enum; legacy property still works for compatibility
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
- **Legacy code isolation** - Legacy initialization (`legacy/bunch_inits.py`) now ONLY runs when "Enable legacy comparison" is checked in GUI
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

### Critical Physics Corrections (December 2024)

- **Corrected scalar potential calculation** - Fixed dimensional error in electromagnetic potential computation
- **Proper kinetic energy separation** - Now correctly subtracts potential energy (q·Φ) from conjugate energy to obtain kinetic gamma
- **Fixed self-consistency convergence** - Iterations now enforce the mass-shell constraint Pt² = P² + (mc)² through projection
- **Improved numerical precision** - Float64 throughout, relaxed k_factor threshold to 1e-20 for extreme angles
- **Self-consistency enabled by default** - Essential for energy conservation in high-energy simulations
- **Chrono-match interpolation** - Sub-timestep accuracy for retarded field calculations, providing 10-100× reduction in time residual. Critical for ultra-relativistic simulations (γ > 100). Enabled via `SelfConsistencyConfig(chrono_interpolate=True)`. See `local/CHRONO_INTERPOLATION_SUMMARY.md` for details.

**Impact**: Energy conservation improved by 3+ orders of magnitude in high-energy electron-wall simulations. Adaptive timestep auto-calculation eliminates parameter inconsistencies and prevents time discontinuities. Batched logging maintains GUI responsiveness during verbose debugging. Gamma reconciliation disabled by default restores v0.4.8 energy conservation behavior. Improved GUI feedback for transverse offsets reduces user confusion and makes bunch-to-bunch positioning more intuitive. Early stopping enables practical parameter optimization for computationally expensive self-consistent simulations. Macroparticle simulation enables realistic modeling of beam emittance and collective effects in conducting-wall scenarios. Transverse offset functionality enables off-axis beam studies critical for aperture tolerance analysis and beam dynamics research. Legacy code isolation ensures modern core implementation is used by default while maintaining validation capability.

## Versioning and release notes

The project version is defined exactly once in `core/_version.py`. Both
`setup.py` and `docs/source/conf.py` import that value, ensuring the wheel
metadata and Sphinx footer remain consistent.

This project uses **bump2version** for automated version management. To cut a new release:

```bash
# Bump patch version (0.4.1 → 0.4.2)
bump2version patch

# Bump minor version (0.4.1 → 0.5.0)
bump2version minor

# Bump major version (0.4.1 → 1.0.0)
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
