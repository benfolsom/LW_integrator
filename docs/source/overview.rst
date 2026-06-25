Overview
========

The LW Integrator is a covariant electromagnetic particle tracking code tuned for
accelerator physics studies.  The maintained implementation lives in ``core/``,
with CLI and GUI workflows layered on top through ``lw_integrator/`` and
``optimization/``.  The method is
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
    utilities, and the ``IntegratorConfig`` data class).  Kernel-level Numba
    acceleration lives in ``vectorized_interactions.py`` and is used by the
    canonical integrator path without changing the underlying physics.
    ``pseudo_grid.py`` contains the experimental active/passive reduced-solver
    helpers for ``BUNCH_TO_BUNCH`` studies, including causal-history retention
    metadata for supported reduced runs.
    ``self_consistency.py`` holds the fixed-point
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
    Archived notebooks from the original codebase.  They are kept for
    historical reference only; production workflows should use ``core/`` and the
    maintained CLI/GUI entry points.

``examples/``
    Ready-to-run examples and reference validation material.  The
    ``validation/`` folder keeps historical notebooks and supporting examples,
    while maintained command-line workflows live under ``lw_integrator/``.

``tests/``
    Automated regression coverage.  ``tests/unit`` hosts deterministic unit
    tests for helper functions, ``tests/physics`` covers numerical behavior, and
    top-level tests cover CLI/GUI parity plus plotting entry points.

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

``results/``
    Output location for sweep and optimization runs (git-ignored).  Completed
    sweeps land in ``results/sweeps/``; incomplete or cancelled sweeps are
    relocated to ``results/archive/incomplete/`` automatically.

``docs/``
    The refreshed documentation that you are currently reading.  Sphinx builds
    use the configuration in ``docs/source/conf.py`` and the helper script
    ``docs/build_docs.sh``.

Pseudo-grid reduced solve sketch
---------------------------------

In ``BUNCH_TO_BUNCH`` pseudo-grid mode, each bunch is still stored and exported
as a full particle state, but only a selected active subset is advanced as
dynamic observers on a given step.  A separate field-representative set contains
weighted live particles used as retarded Liénard--Wiechert sources.  Non-field
live particles deposit source charge directly onto nearby field representatives,
which improves source-distribution fidelity without requiring every represented
particle to be solved as an observer.

.. code-block:: text

   Live particles
       |-- select active particles --------> Dynamic observers
       |                                      |
       |-- select field representatives --> Weighted LW sources
                                              |
                         Retarded LW source sum
                                              |
                                Advance active trajectories
                                              |
                         Reconstruct or propagate passive states
                                              |
                                      Live particles

.. only:: html

   .. raw:: html
      :file: _static/pseudo_grid_full_vs_reduced.svg.inc

.. code-block:: text

   One bunch, viewed as a 2D transverse slice for a single step

          passive p0 ○
                      \
                       \ w0,1
                        \
          active A1  ●----●  active A2
                    / \  / \
             w1,1  /   \/   \  w2,2
                  /    /\    \
      passive p1 ○    /  \    ○ passive p2
                      /    \
             active A3 ●    ○ passive p3

   Cross-bunch reduced force solve

      rider active observers      retarded LW fields      driver field reps
             ● A_r0  <----------------------------------  ● F_d0 + q_eff(d0)
             ● A_r1  <----------------------------------  ● F_d1 + q_eff(d1)

   Legend: ● = live particle selected for this role; ○ = passive live particle.
           w = passive interpolation/deposition weight.
           q_eff = representative charge plus deposited non-field source charge.

After the active solve, passive handling depends on ``passive_update_mode``.
The historical ``weighted_delta`` mode reconstructs each passive from the
weighted combination of its active anchors' changes in position, momentum,
velocity, gamma, and radiation bookkeeping.  For high-passive source-resolution
studies, ``external_interbunch`` is usually the safer reduced mode: passives are
integrated as observers of external fields and opposite-bunch field
representatives, but same-bunch/intrabunch space charge is omitted for those
passive solves.  This keeps passive source/evaluation samples responsive to
bending fields and interbunch acceleration without reintroducing the expensive
and numerically fragile passive same-bunch space-charge solve.  ``ballistic``
coasts passives force-free and ``frozen`` keeps static passives; both are best
treated as diagnostics.  Cross-bunch field-source representation uses the
separate field-representative deposition path.  If a selected active particle is
marked dead by the existing status machinery, pseudo-grid loss tracking removes
it from later schedules and renormalizes passive-anchor weights onto surviving
anchors when possible.

Key ideas to keep in mind
-------------------------

* **Physics parity matters.**  The core code is intentionally close to the
  validated reference solver, with critical corrections applied in December
  2025 to fix gamma calculation and scalar potential handling. Recent changes
  include proper separation of conjugate and kinetic energy, corrected
  self-consistency convergence tests, and improved numerical precision for
  extreme relativistic scenarios. Any behavioural change should come with
  focused regression tests.
* **Particle states are dictionaries of NumPy arrays.**  Whenever you initialize
  particles manually, fill every expected key (``x``, ``Pz``, ``gamma``, ``q``,
  ``char_time`` …) or use
  ``input_output.bunch_initialization.create_bunch_from_energy`` to obtain a
  correctly shaped state.
* **Simulation modes are enumerated.**  ``SimulationType`` enumerates the three
  supported wall configurations.  The enum still accepts the historical integer
  values for compatibility.
* **Startup modes are configurable.**  ``StartupMode`` switches between
  ``COLD_START`` (the default, suppressing early retarded forces) and
  ``APPROXIMATE_BACK_HISTORY`` (reconstructs a constant-velocity history that
  matches the archived reference treatment).  Maintained CLI and GUI workflows
  surface the enum so you can pick the right transient treatment per study.
* **Experimental pseudo-grid mode is now active for B2B studies.**
  ``PseudoGridConfig`` exposes an opt-in reduced active/passive solve path for
  ``SimulationType.BUNCH_TO_BUNCH`` runs.  The reduced path supports adaptive
  timestep retries and reduced same-bunch space charge when each bunch keeps at
  least two active particles.  High-passive convergence studies should normally
  use ``active_selection_mode="fixed_prefix"`` with
  ``passive_update_mode="external_interbunch"`` so passives respond to external
  and opposite-bunch fields while same-bunch space charge remains confined to
  the reduced active/field-representative model.  Supported reduced B2B runs can
  compact live causal histories while preserving full-state SoA/legacy outputs
  and retained-history diagnostics.  Unsupported configurations fall back to the
  canonical full solve.
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
* **Full 3D bunch initialization is the preferred path for new work.**  Use
  ``momentum_axis``, ``starting_position_mm``, ``transverse_distance_mm``, and
  ``longitudinal_span_mm`` when defining new bunches. The CLI accepts these
  fields directly in rider/driver JSON, and the main GUI now exposes them
  through a ``Manual Particle Config`` tab with rider/driver JSON editors.
  Legacy off-axis fields such as ``transv_offset_x``, ``transv_offset_y``, and
  ``transv_dist`` remain available for compatibility and older studies,
  including optimization-plugin offset fractions converted to absolute
  positions (offset = fraction × aperture_radius).
* **GUI application for all workflows.**  The Tkinter-based GUI (``python -m
  lw_integrator.gui``) supports single runs, parameter sweeps, and optimization
  with real-time progress tracking and trajectory visualization. It provides
  full control over particle properties, boundary conditions, physics parameters,
  and numerical methods. Results can be exported in CSV, JSON, or NPZ formats.
  The GUI is the recommended interface for interactive work, with the CLI
  (``lw-simulate``) available for scripting and batch processing.
* **Heatmap and contour tools.**  ``lw-generate-sweep-heatmap`` is the
  maintained command for producing
  publication-quality heatmaps from sweep results. Contour lines use a low
  alpha (0.18), labels are clamped to stay inside the axes after the final
  layout pass, and overlapping labels are culled automatically. Use
  ``--color-min`` / ``--color-max`` for comparable signed color scales across
  related plots, and ``--axis-param1-max`` when a displayed x-axis crop should
  preserve neighboring data columns for interpolation.
* **Live sweep plotting.**  ``lw-plot-latest-live`` follows the newest sweep
  log automatically, while ``lw-plot-from-logcache-live`` can watch a specific
  log file or render a one-shot static plot from it.
* **Saved trajectory plotting.**  ``lw-plot-trajectory`` turns saved single-run
  JSON or NPZ trajectory files into publication-ready PNG summaries, including
  rider/driver views for JSON payloads and compact momentum/gamma panels for
  NPZ payloads.
* **Reference notebooks remain available.**  Historical validation notebooks are
  retained for context, while maintained CLI and GUI workflows should be the
  baseline for new scripted sweeps.
* **Need the math?**  See :doc:`theory` for the derivation of the covariant
  equations of motion implemented in ``core/trajectory_integrator.py`` and the
  approximations used in the benchmark studies.

With the map in hand, continue to :doc:`quickstart` to set up a development
environment or jump to :doc:`validation` for the maintained validation workflow.
