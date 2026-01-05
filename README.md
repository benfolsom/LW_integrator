# LW Integrator

The LW Integrator is a covariant charged-particle tracking code that evaluates
retarded Liénard–Wiechert potentials to obtain first-principles beam dynamics.
The repository contains a modernised ``core`` implementation that mirrors the
validated legacy solver, an updated Sphinx documentation set, and a collection
of validation scripts and notebooks.  The methodology is documented in the
peer-reviewed article *Relativistic beam loading, recoil-reduction, and
residual-wake acceleration with a covariant retarded-potential integrator*
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

* **Physics focus.**  The code integrates particle trajectories using
  retarded-vector potentials and conjugate-momentum dynamics.  The ``core``
  package is a faithful transcription of the proven legacy solver and is kept in
  numerical lockstep by an integration test suite.
* **Startup strategies.**  The integrator now exposes
  :class:`core.types.StartupMode`, allowing cold-start runs that suppress
  retarded forces during the short-history transient (default) or an
  ``APPROXIMATE_BACK_HISTORY`` mode that reconstructs a constant-velocity past
  for better legacy alignment.  All entry points—CLI, scripts, and notebooks—take
  the new enum so you can toggle behaviour without patching call sites.
* **Self-consistency and energy conservation.**  The integrator enforces the
  relativistic mass-shell constraint Pt² = P² + (mc)² through iterative
  projection during each timestep. Two modes are available:
  * **mass_shell_only (default)**: Fast iteration with fixed geometry—retarded
    distances computed once per step. Suitable for most simulations.
  * **full_iteration**: Updates particle positions and recomputes retarded
    distances each iteration for maximum accuracy when particles move
    significantly (|Δx| ~ 0.1×R_separation). Computationally expensive but
    accounts for geometric changes during the timestep.
  Self-consistency is enabled by default and critical for energy conservation
  in high-energy simulations (γ > 10⁴). Implemented December 2024.
* **Adaptive timestep and beta clamping.**  The integrator includes numerical
  safety features for extreme relativistic regimes (γ > 10⁶):
  * **Beta clamping** prevents particle velocities from reaching the speed of
    light (β ≥ 1), ensuring the Lorentz factor remains finite even at extreme
    energies. Velocities are automatically limited to β < 0.99999999999999999
    (17 decimal places, near the float64 precision limit) corresponding to
    ~34 TeV for electrons.
  * **Adaptive timestep refinement** detects energy jumps during integration
    and automatically retries problematic steps with smaller timesteps. This
    is configurable via ``AdaptiveTimestepConfig`` and particularly useful for
    high-energy electron-wall simulations.
* **Trajectory stability analysis.**  Post-integration validation assesses
  whether trajectories are numerically stable across multiple timesteps, even
  in regions with strong physical forces (radiation reaction, image charges).
  Rather than rejecting runs with large single-step jumps—which can represent
  valid physics—the analyzer checks for oscillatory instabilities, erratic
  evolution that cannot fit smooth polynomial trends, and multi-scale
  inconsistencies. This multi-step approach distinguishes numerical artifacts
  from physical behavior and is essential for unattended sweep and optimization
  runs. Configured via ``SmoothnessConfig`` with presets for strict,
  balanced, and permissive validation. See ``core/smoothness_analyzer.py``
  and ``local/smoothness_checking_implementation.md`` for details.
* **Macroparticle simulation.**  For conducting-wall simulations, the integrator
  supports macroparticle mode where test particle charges are scaled by a
  configurable multiplier and image subcharge positions receive stochastic errors
  based on transverse position and momentum spreads. Position spread applies
  constant Gaussian errors (σ_x), while momentum spread creates cumulative
  displacement that grows with each timestep: σ_total(step) = sqrt(σ_x² +
  (σ_p × h × step / m)²). These errors are applied before charge attenuation
  calculations to accurately model beam emittance effects. Configured via GUI
  controls in the Particles tab (single runs) and optimization/sweep parameter
  sections. Only active for CONDUCTING_WALL simulation type.
* **Reference publication.**  For the scientific context, derivations, and
  benchmark scenarios, see the project paper referenced above; the codebase
  tracks the configurations described there.
* **Documentation.**  The refreshed Sphinx site under ``docs/`` explains the
  theoretical background, quick-start workflows, validation procedures, and
  contributor guidance.  A new ``theory`` page summarises the covariant
  derivations drawn from the in-repo technical note.
* **Validation assets.**  The ``examples/validation`` tree provides both Python
  scripts and notebooks for reproducing benchmark comparisons between the
  modern and legacy implementations.  The refreshed ``integrator_testbed``
  notebook surfaces legacy overlays, difference plots, and live initial-state
  summaries so physics regressions are immediately visible while you tweak
  parameters.  Its widget scaffolding now lives in
  ``examples/validation/testbed_ui.py`` so you can import
  ``IntegratorTestbedApp`` into other notebooks or scripts without duplicating
  the layout logic.
* **CLI entry point.**  The ``lw-simulate`` console command (see the
  [Command-line entry point](#command-line-entry-point) section below) runs the
  core integrator with JSON-configurable inputs.  A minimal demonstration lives
  in ``examples/entrypoint_demo.py``.

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

1. **Create and activate a virtual environment** (Python 3.8–3.13 are supported).

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the package in editable mode** with commonly used extras.

   ```bash
   pip install -e ".[dev,examples,docs]"
   ```

   * ``dev`` adds the lint/test toolchain.
   * ``examples`` installs notebook dependencies.
   * ``docs`` brings in Sphinx, ``sphinx-autobuild``, and related extensions.

3. **(Optional) register the kernel for Jupyter usage.**

   ```bash
   python -m ipykernel install --user --name lw-integrator --display-name "LW Integrator (.venv)"
   ```

---

## Running validation workloads

The canonical comparison between the core and legacy solvers lives in
``examples/validation/core_vs_legacy_benchmark.py``.  Execute it directly:

```bash
python examples/validation/core_vs_legacy_benchmark.py --seeds 0 1 2 --steps 5000 --plot
```

The script accepts additional options for output paths, DPI control, and plot
styling.  Consult ``--help`` for the full list.  Companion notebooks in the same
directory expose an interactive widget-driven interface for exploratory work.
The notebook delegates all widget construction to
``examples/validation/testbed_ui.py``; instantiate ``IntegratorTestbedApp`` to
embed the UI in your own notebook or lab book without copying code cells.

> **Note:** The interactive notebooks are tested and supported in JupyterLab
> and the classic Jupyter Notebook or Jupyter Lab interface.  The VS Code notebook editor is
> known to trigger duplicate plot rendering.

The ``tests/`` directory contains deterministic Pytest suites that ensure
physics parity across configurations:

```bash
pytest tests
```

### Command-line entry point

Installing the project (``pip install -e .`` or via a wheel) exposes the
``lw-simulate`` executable.  Run it with default settings:

```bash
lw-simulate --quiet
```

The CLI accepts additional overrides—for example, to shorten the integration
and capture a JSON summary:

```bash
lw-simulate --steps 250 --time-step 5e-4 --output run.json
```

Conducting-wall runs apply radial weighting to image subcharges by default for
better agreement with the aperture geometry.  Pass `--no-image-weighting` to
recover the legacy uniform distribution when benchmarking or debugging.

Programmatic usage mirrors the console invocation: call
``lw_integrator.cli.main`` with a list of CLI-style arguments.  See
``examples/entrypoint_demo.py`` for a ready-to-run demonstration that exercises
both patterns.

### Optimization GUI

The project includes a Tkinter-based GUI for parameter sweeps and optimization:

```bash
python -m lw_integrator.gui
```

The GUI provides two modes in the **Sweep / Optimization** tab:

#### Blind Sweep Mode
* **Parameter sweeps** over aperture radius, particle energy, transverse offset, and starting positions
* **Sweepable fixed parameters** - mass, charge, transverse momentum, timestep, wall position
* **Auto-timestep calculation** to maintain consistent integration resolution across energy ranges
* **Trajectory saving** with configurable stride
* Results saved to timestamped directories with JSON summary and plots

#### Optimization Mode
* **Multiple algorithms**: Genetic Algorithm, Differential Evolution, Nelder-Mead, Multi-start
* **Convergence detection**: Early stopping when fitness plateaus (GA only, configurable tolerance and patience)
* **Objectives**: Maximize energy gain (%), minimize transverse deflection, or custom metrics
* **Real-time logging**: Progress tracking with generation/iteration updates
* **Top-N saving**: Automatically saves best configurations found

**Optimization Quick Start:**
1. Select "Optimization" mode
2. Choose optimizer (Genetic Algorithm recommended for global search)
3. Set convergence parameters:
   - Tolerance: 1e-6 (relative improvement threshold)
   - Patience: 10 generations (lookback window)
4. Define parameter ranges (at least 2 sweep dimensions required)
5. Run - optimizer automatically stops when converged or max iterations reached

**Performance Notes:**
* Early stopping can reduce runtime by 40-70% when convergence occurs
* For radiation reaction physics (stripped_ions > 10), use timestep ≤ 3e-7 ns with self-consistency enabled
* Nelder-Mead is fastest for local optimization (~15-50 min), GA/DE are thorough but slower (~1-3 hours)

Results are saved to `results/sweeps/YYYYMMDD_HHMMSS_configname/` with convergence history, best parameters, and optional trajectory data. See `local/SWEEP_AND_OPTIMIZATION_GUIDE.md` for detailed usage.

---

## Documentation workflow

All documentation sources are under ``docs/source/``.  The helper script
``docs/build_docs.sh`` wraps ``sphinx-build`` and ``sphinx-autobuild``.

* **One-off build** (HTML):

  ```bash
  cd docs
  ./build_docs.sh --clean --type html
  ```

* **Live preview with automatic reload** (requires ``sphinx-autobuild``):

  ```bash
  cd docs
  ./build_docs.sh --clean --watch
  ```

  The preview runs at ``http://localhost:8000`` as long as the process remains
  active.

GitHub Actions publishes the rendered site to GitHub Pages whenever the ``main``
branch is updated.  Every build also uploads the HTML artefact so intermediate
branches can download the output for review.

---

## Recent changes (January 2025)

### Transverse Offset and Legacy Code Isolation (January 21, 2025)
* **Transverse offset parameters** - New `transv_offset_x` and `transv_offset_y` fields separate beam center position from beam spread
* **Beam positioning** - Particles now distributed in `[offset ± spread]` allowing off-axis beams with controllable size
* **Core bunch initialization** - New `input_output.bunch_initialization.create_bunch_from_params()` replaces legacy initialization for normal operation
* **Legacy code isolation** - Legacy initialization (`legacy/bunch_inits.py`) now ONLY runs when "Enable legacy comparison" is checked in GUI
* **GUI integration** - Offset fields automatically appear in Particles tab for both rider and driver bunches
* **Optimization plugin fix** - "Transverse Offset" now correctly sets beam **position** (not spread), with separate `transv_dist` for beam size
* **Backward compatibility** - Old configs without offset parameters default to 0.0 (on-axis), no breaking changes

### Macroparticle Simulation (January 20, 2025)
* **Macroparticle charge scaling** - Test particle and image charges can be multiplied by configurable factor for bunch simulations
* **Stochastic position errors** - Gaussian position spread (σ_x in mm) applied to image subcharges
* **Cumulative momentum spread** - Transverse momentum errors accumulate over timesteps: σ_total(step) = sqrt(σ_x² + (σ_p × timestep × step / mass)²)
* **Pre-attenuation error application** - Errors applied before radial weighting calculations for physical accuracy
* **GUI integration** - Controls in Particles tab (single runs) and sweep/optimization sections with automatic greying for non-CONDUCTING_WALL modes
* **Configuration persistence** - All macroparticle parameters saved/loaded with simulation configs

### Optimization and Convergence (January 17, 2025)
* **Early stopping for Genetic Algorithm** - Automatic convergence detection stops optimization when fitness plateaus, saving 40-70% computation time
* **Configurable convergence parameters** - GUI controls for tolerance (default: 1e-6) and patience (default: 10 generations)
* **Comprehensive optimization guide** - New documentation covering sweep vs optimization workflows, metrics, and performance tuning

### Critical Physics Corrections (December 2024)
* **Corrected scalar potential calculation** - Fixed dimensional error in electromagnetic potential computation
* **Proper kinetic energy separation** - Now correctly subtracts potential energy (q·Φ) from conjugate energy to obtain kinetic gamma
* **Fixed self-consistency convergence** - Iterations now enforce the mass-shell constraint Pt² = P² + (mc)² through projection
* **Improved numerical precision** - Float64 throughout, relaxed k_factor threshold to 1e-20 for extreme angles
* **Self-consistency enabled by default** - Essential for energy conservation in high-energy simulations
* **Chrono-match interpolation** - Sub-timestep accuracy for retarded field calculations, providing 10-100× reduction in time residual. Critical for ultra-relativistic simulations (γ > 100). Enabled via `SelfConsistencyConfig(chrono_interpolate=True)`. See `local/CHRONO_INTERPOLATION_SUMMARY.md` for details.

**Impact**: Energy conservation improved by 3+ orders of magnitude in high-energy electron-wall simulations. Early stopping enables practical parameter optimization for computationally expensive self-consistent simulations. Macroparticle simulation enables realistic modeling of beam emittance and collective effects in conducting-wall scenarios. Transverse offset functionality enables off-axis beam studies critical for aperture tolerance analysis and beam dynamics research. Legacy code isolation ensures modern core implementation is used by default while maintaining validation capability.

## Versioning and release notes

The project version is defined exactly once in ``core/_version.py``.  Both
``setup.py`` and ``docs/source/conf.py`` import that value, ensuring the wheel
metadata and Sphinx footer remain consistent.  To cut a new release:

1. Update ``__version__`` in ``core/_version.py``.
2. Commit the change alongside relevant release notes or change logs.
3. Tag and publish as needed; the packaging metadata is already aligned.

---

## Development guidelines

* Maintain parity between the ``core`` and ``legacy`` solvers when modifying
  physics-critical code.  New behaviours should be backed by updated validation
  plots and regression tests.
* Prefer the helper utilities in ``input_output/`` when constructing particle
  bunches.  They guarantee the integrator receives correctly shaped state
  dictionaries.
* Run the Pytest suite and build the documentation before submitting changes.
  The repository treats Sphinx warnings as errors to keep the rendered site
  trustworthy.
* For high-energy simulations (γ > 10⁴), self-consistency is now enabled by
  default with ``mass_shell_only`` mode (fixed geometry, fast). For maximum
  accuracy when particle motion during timesteps is significant, use
  ``SelfConsistencyConfig.full_iteration()``. To disable (not recommended),
  explicitly set ``SelfConsistencyConfig(enabled=False)``.

---

## Support

Discussion of new physics scenarios, validation additions, or documentation
improvements is welcome via GitHub issues.  When reporting a problem, please
include:

* the observed behaviour and expected outcome,
* the relevant configuration (energy range, simulation type, etc.), and
* reproduction steps or sample notebooks.

For background reading on the theoretical model, consult ``docs/source/theory``
and the accompanying technical note under ``LW_local_refs/main.tex``.
