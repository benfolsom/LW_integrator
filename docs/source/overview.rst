Overview
========

The LW Integrator is a covariant electromagnetic particle tracking code tuned for
accelerator physics studies.  The validated implementation still lives in the
``legacy/`` tree, but the modern ``core/`` package mirrors the same physics with
clearer structure, type hints, and unit-tested helper utilities.  The method is
described in *Relativistic beam loading, recoil-reduction, and residual-wake
acceleration with a covariant retarded-potential integrator*
(``https://doi.org/10.1016/j.nima.2024.169988`` / ``https://arxiv.org/abs/2310.03850``).
This page summarises the pieces you will encounter when navigating the
repository.

High-level anatomy
------------------

``core/``
    The maintained implementation of the retarded Liénard–Wiechert solver.
    ``trajectory_integrator.py`` retains the original algorithm but exposes it
    through typed helper functions (image charge generators, retarded-distance
    utilities, and the ``IntegratorConfig`` data class).  ``performance.py``
    bundles optional Numba kernels that accelerate large runs without changing
    the underlying physics.  ``self_consistency.py`` holds the fixed-point
    iteration used for radiation-reaction corrections and ensuring gamma
    consistency between energy and velocity calculations (enabled by default
    as of December 2025).  ``images.py`` implements conducting-wall image charge
    generation with optional macroparticle simulation support, applying stochastic
    position and momentum spread errors to model beam emittance effects.

``configs/``
    JSON configuration files for single runs (``run_configs/``) and parameter
    sweeps (``sweep_configs/``).  Sweep configs specify parameter grids, physics
    options, and verbosity settings consumed by both the CLI and GUI runners.

``legacy/``
  Archived notebooks and scripts from the original codebase.  They are kept
  for regression comparisons and historical reference.  Production workflows
  should use ``core/`` unless you are debugging a discrepancy.  The historical
  "static" integrator lives here for completeness; it is considered deprecated
  and is not exercised by the modern documentation or tooling.

``examples/``
    Ready-to-run validation material.  The ``validation/`` folder contains both
    Python scripts and Jupyter notebooks that reproduce the legacy vs. core
    comparisons.  The ``comparison/`` folder houses CLI benchmarks that report
    metrics (maximum Δγ, Δz, etc.) across seeded simulation suites.

``tests/``
    Automated regression coverage.  ``tests/integration/test_core_integrators.py``
    verifies equivalence between the core solver, its self-consistent variant,
    and the legacy implementation.  ``tests/unit`` hosts deterministic unit
    tests for helper functions such as ``generate_conducting_image``.

``input_output/``
    Utilities for constructing particle bunch dictionaries in the format the
    integrator expects.  ``bunch_initialization.py`` is the main entry point and
    is documented in the API section below.

``lw_integrator/``
    CLI entry point (``cli.py``), GUI application (``gui.py``), headless sweep
    runner (``sweep_runner.py``), testbed runner (``testbed_runner.py``), and the
    optimization GUI plugin (``optimization_plugin.py``).  As of v0.6.0 the CLI
    sweep runner calls the **same** ``run_testbed()`` / ``SimulationOptions``
    code paths as the GUI, so results are identical between interfaces.  Launch
    the GUI with ``python -m lw_integrator.gui`` to access three operational
    modes: Single Run (Main tab) for individual simulations with real-time
    visualization, Blind Sweep for parameter space exploration, and Optimization
    for global search using Genetic Algorithm, Differential Evolution,
    Nelder-Mead, or Multi-start methods.  Includes convergence detection, top-N
    result saving, and comprehensive trajectory export options.

``optimization/``
    Sweep and optimization engine shared by CLI and GUI.  Contains the parameter
    grid builder (``parameter_sweep.py``), metrics extraction (``metrics.py``),
    result I/O (``result_io.py``), and UI/run/results mixins that the GUI plugin
    composes.  ``result_io.relocate_incomplete_sweep()`` automatically moves
    sweep directories with fewer than 100 completed runs to
    ``results/archive/incomplete/``.

``scripts/``
    Monitoring helpers, smooth-heatmap generation utilities, and other
    operational scripts used during long-running sweeps.

``results/``
    Output location for sweep and optimization runs (git-ignored).  Completed
    sweeps land in ``results/sweeps/``; incomplete or cancelled sweeps are
    relocated to ``results/archive/incomplete/`` automatically.

``docs/``
    The refreshed documentation that you are currently reading.  Sphinx builds
    use the configuration in ``docs/source/conf.py`` and the helper script
    ``docs/build_docs.sh``.

Key ideas to keep in mind
-------------------------

* **Physics parity matters.**  The core code is intentionally a transcription of
  the legacy solver, with critical corrections applied in December 2025 to fix
  gamma calculation and scalar potential handling. Recent changes include proper
  separation of conjugate and kinetic energy, corrected self-consistency
  convergence tests, and improved numerical precision for extreme relativistic
  scenarios. Any behavioural change should come with matching updates to the
  validation scripts and the integration tests.
* **Particle states are dictionaries of NumPy arrays.**  Whenever you initialize
  particles manually, fill every expected key (``x``, ``Pz``, ``gamma``, ``q``,
  ``char_time`` …) or use ``input_output.create_bunch_from_energy`` to obtain a
  correctly shaped state.
* **Simulation modes are enumerated.**  ``SimulationType`` enumerates the three
  supported wall configurations.  The solver mirrors the legacy integer flags so
  comparison runs remain straightforward.
* **Startup modes are configurable.**  ``StartupMode`` switches between
  ``COLD_START`` (the default, suppressing early retarded forces) and
  ``APPROXIMATE_BACK_HISTORY`` (reconstructs a constant-velocity history that
  mirrors the legacy solver's behaviour).  CLI commands, scripts, and notebooks
  surface the enum so you can pick the right transient treatment per study.
* **CLI/GUI parity.**  As of v0.6.0 the CLI sweep runner
  (``lw-simulate --sweep-config``) and the GUI's Blind Sweep mode invoke the
  same ``run_testbed()`` function with the same ``SimulationOptions`` dataclass.
  This eliminates subtle differences in particle initialisation, metric
  extraction, or physics option handling between the two interfaces.
* **Incomplete-sweep archiving.**  Sweeps that finish with fewer than 100
  completed runs are automatically relocated to
  ``results/archive/incomplete/<sweep_dir_name>`` on save.  This applies to all
  save paths (CLI, GUI mixin, GUI plugin, library API) and also fires on
  ``KeyboardInterrupt`` in the CLI runner.
* **Self-consistency is enabled by default.**  As of December 2025, self-
  consistency iterations are enabled by default to ensure energy conservation in
  high-energy simulations. These iterations verify that gamma derived from
  energy matches gamma derived from velocity (γ = 1/√(1 - β²)), which is
  critical for physical correctness. See ``SelfConsistencyConfig`` in the API
  documentation.
* **Macroparticle simulation for conducting walls.**  The integrator supports
  macroparticle mode where test particle charges are scaled and image subcharges
  receive stochastic position/momentum errors. Position spread applies constant
  Gaussian errors (σ_x), while momentum spread creates cumulative displacement
  that grows with each timestep. This enables realistic modeling of beam
  emittance and collective effects. Configure via ``macroparticle_enabled``,
  ``macroparticle_charge_multiplier``, ``macroparticle_position_spread``, and
  ``macroparticle_momentum_spread`` parameters. Only active for CONDUCTING_WALL
  simulation type.
* **Transverse offset for off-axis beams.**  Beam center position is now
  separate from beam size. Use ``transv_offset_x`` and ``transv_offset_y`` to
  position beam center in mm, and ``transv_dist`` for beam spread (half-width).
  Particles are distributed uniformly in [offset ± spread] for both x and y.
  Critical for aperture tolerance studies and beam halo analysis. The
  optimization plugin's "Transverse Offset" fractions are converted to absolute
  positions (offset = fraction × aperture_radius). Legacy initialization
  (``legacy/bunch_inits.py``) now only runs when "Enable legacy comparison" is
  checked in the GUI or ``use_legacy=True`` is passed to
  ``prepare_particle_bunches()``. Default behavior uses modern core
  initialization (``input_output.bunch_initialization.create_bunch_from_params``).
* **GUI application for all workflows.**  The Tkinter-based GUI (``python -m
  lw_integrator.gui``) supports single runs, parameter sweeps, and optimization
  with real-time progress tracking and trajectory visualization. It provides
  full control over particle properties, boundary conditions, physics parameters,
  and numerical methods. Results can be exported in CSV, JSON, or NPZ formats.
  The GUI is the recommended interface for interactive work, with the CLI
  (``lw-simulate``) and notebook options available for scripting and batch
  processing.
* **Heatmap and contour tools.**  ``lw-generate-sweep-heatmap`` (and the
  compatibility script ``generate_sweep_heatmap.py``) produce
  publication-quality heatmaps from sweep results. Contour lines use a low
  alpha (0.18), labels are clamped to stay inside the axes after the final
  layout pass, and overlapping labels are culled automatically.
* **Saved trajectory plotting.**  ``lw-plot-trajectory`` turns saved single-run
  JSON or NPZ trajectory files into publication-ready PNG summaries, including
  core/legacy overlays for JSON payloads and compact momentum/gamma panels for
  NPZ payloads.
* **Notebook tooling is first-class.**  The validation notebooks are kept in
  sync with the scripts and expose colourblind-friendly plots, high-DPI export,
  and configuration widgets.  Use them to explore scenarios before committing to
  scripted sweeps.
* **Need the math?**  See :doc:`theory` for the derivation of the covariant
  equations of motion implemented in ``core/trajectory_integrator.py`` and the
  approximations used in the benchmark studies.

With the map in hand, continue to :doc:`quickstart` to set up a development
environment or jump to :doc:`validation` for the comparison workflows.
