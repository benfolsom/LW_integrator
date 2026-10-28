Overview
========

The LW Integrator (LWI) is a covariant electromagnetic particle tracking code
tuned for accelerator physics studies.  The maintained implementation lives in ``core/``,
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
    ``trajectory_integrator.py`` is a compatibility facade. ``equations.py``
    performs particle updates, ``integration_runner.py`` manages runs, and
    ``retarded_fields.py`` evaluates exact sources. Native PIC and boundary
    solvers live in separate packages; these include substantial developments
    beyond the historical reference algorithm.  Kernel-level Numba
    acceleration lives in ``vectorized_interactions.py`` and is used by the
    canonical integrator path without changing the underlying physics.
    ``pseudo_grid.py`` contains the experimental active/passive reduced-solver
    helpers for ``BUNCH_TO_BUNCH`` studies, including causal-history retention
    metadata for supported reduced runs.
    ``self_consistency.py`` configures the force trials and mass-shell check.
    Spatial mechanical momentum determines gamma and beta in every radiation
    mode; the check alone does not establish force convergence.  ``images.py`` implements conducting-wall image charge
    generation with optional macroparticle simulation support, applying stochastic
    position and momentum spread errors through a legacy image prescription.

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
    optimization GUI plugin (``optimization_plugin.py``).  The CLI sweep runner
    and the GUI call the same ``run_testbed()`` / ``SimulationOptions`` code
    paths.  The GUI (``python -m lw_integrator.gui``) offers single runs, blind
    sweeps and optimization; :doc:`quickstart` walks through them.

``optimization/``
    Sweep and optimization engine shared by CLI and GUI.  Contains the parameter
    grid builder (``parameter_sweep.py``), metrics extraction (``metrics.py``),
    result I/O (``result_io.py``), and UI/run/results mixins that the GUI plugin
    composes.  ``result_io.relocate_incomplete_sweep()`` moves sweep
    directories with fewer than 100 completed runs to
    ``results/archive/incomplete/``.

``results/``
    Output location for sweep and optimization runs (git-ignored).  Completed
    sweeps land in ``results/sweeps/``; incomplete or cancelled sweeps (fewer
    than 100 completed runs, including a ``KeyboardInterrupt`` in the CLI
    runner) are relocated to ``results/archive/incomplete/`` on save.

``docs/``
    The documentation you are reading.  Sphinx builds
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

Reconstruction fails before publishing an advanced live particle if its speed
has ``|beta| >= 1``, any reconstructed dynamical field is non-finite, or its
time does not strictly increase. ``PseudoGridStateError`` reports the step,
full-bunch particle index, speed, and old and new times. This check covers
weighted passive deltas, ballistic updates, and copied active or external
interbunch solve results. It never clips or repairs a state. Dead particles and
passives deliberately left unchanged by ``frozen`` mode are not advanced and
are excluded from this check. Passing the check establishes a finite,
subluminal state with increasing time; it does not establish energy conservation
or the physical accuracy of the passive approximation.

Key ideas to keep in mind
-------------------------

* **Regression boundaries matter.** Existing tests cover specific field,
  trajectory, configuration, and interface contracts. Historical notebooks
  explain the project's origin; they do not establish universal numerical
  lockstep with the current implementation. Adapt the relevant existing tests
  for behavioral changes, and add tests only when strongly justified.
* **Particle states are dictionaries of NumPy arrays.**  Whenever you initialize
  particles manually, fill every expected key (``x``, ``Pz``, ``gamma``, ``q``,
  ``char_time`` …) or use
  ``input_output.bunch_initialization.create_bunch_from_energy`` to obtain a
  correctly shaped state.
* **Simulation modes are enumerated.**  ``SimulationType`` enumerates the three
  configurations: conducting wall, switching wall, and bunch-to-bunch.  The enum still accepts the historical integer
  values for compatibility.
* **Startup modes are configurable.**  ``StartupMode`` switches between
  ``COLD_START`` (the default, suppressing early retarded forces; valid only
  where fields are negligible long enough to build history, see
  :doc:`startup_history`),
  ``INERTIAL_PREHISTORY`` (a finite synthetic coasting prefix that supports
  exact retarded light-cone solves from active time zero), and
  ``APPROXIMATE_BACK_HISTORY`` (reconstructs a constant-velocity history that
  matches the archived reference treatment).  Maintained CLI and GUI workflows
  surface the enum so you can pick the right transient treatment per study.
  The exact inertial mode belongs to ``BUNCH_TO_BUNCH`` RFS/retarded-dipole
  runs, including fixed-step charge driver trains and the guarded
  checkpointed shared-lab-time route. It is required for exact visibility
  gates. See :doc:`exact_path`. Its sparse coasting knots are
  hidden from output, and their duration is extended until all initial exact
  field stencils have bracketed light-cone roots; missing history thereafter
  is a hard error.
* **Native PIC is a separate lab-time solver.** It uses quasi-static
  rest-frame grids with NumPy/Numba, MLX, or CuPy backends. ``auto`` stays on
  CPU. The opt-in K-cloud LW correction requires CPU float64; near-field
  pairs, midpoint prediction, and restricted Medina/LAD remain experimental.
  See :doc:`native_pic` for defaults and limits.
* **The pseudo-grid reduced solve is experimental and opt-in.**
  ``PseudoGridConfig`` exposes an opt-in reduced active/passive solve path for
  ``SimulationType.BUNCH_TO_BUNCH`` runs.  The reduced path supports adaptive
  timestep retries and reduced same-bunch space charge when each bunch keeps at
  least two active particles.  High-passive convergence studies should normally
  use rotating or slow-rotating active selection with
  ``passive_update_mode="external_interbunch"`` so passives respond to external
  and opposite-bunch fields while same-bunch space charge remains confined to
  the reduced active/field-representative model.  ``slow_rotating_live`` replaces
  only a configurable fraction of the active set at a configured interval;
  ``fixed_prefix`` is best treated as a diagnostic control.  Supported reduced B2B runs can
  compact live causal histories while preserving full-state SoA/legacy outputs
  and retained-history diagnostics.  Unsupported configurations fall back to the
  canonical full solve.
* **CLI/GUI parity.**  The CLI sweep runner (``lw-simulate --sweep-config``)
  and the GUI's Blind Sweep mode invoke the same ``run_testbed()`` function with
  the same ``SimulationOptions`` dataclass, so particle initialisation, metric
  extraction and physics options do not differ between the two interfaces.
  Native PIC is a separate path with no such parity claim.
* **Self-consistency is enabled by default.** The maintained settings use two
  ``fixed_geometry`` trials, gamma reconciliation disabled, and chrono
  interpolation off. Mass-shell relaxation is unused on the corrected path.
  Check trajectory convergence and conservation separately; see
  :doc:`self_consistency`.
* **Boundary response is experimental.** :doc:`boundary` covers prescribed
  Drude drives, conformal perfect conductors, and opt-in finite-window axial
  feedback. It is separate from the production LW and PIC runners.
* **Smooth external fields use a provider.** For energy-loss studies, prefer
  the second-order exact route and verify timestep convergence; see
  :doc:`external_fields`.
* **Stochastic quantum emission is opt-in and experimental.** It runs on the
  sampled, general exact ``INERTIAL_PREHISTORY`` (recoils stored as split
  velocity knots) and guarded native pair routes, within documented
  restrictions; see :doc:`stochastic_quantum_emission`.
* **Photon transport is opt-in and experimental.** Photons fly at c and scatter
  or convert on the native lab-time runner only; see :doc:`photon_transport`.
* **Macroparticle simulation for conducting walls.** The legacy image-source
  multiplier scales generated image charges. The spread prescription uses
  ``macroparticle_sigma_multiplier``, ``macroparticle_use_momentum_errors``,
  and the rider bunch's transverse spread inputs. It does not change the
  observer charge or establish beam-emittance evolution; see :doc:`api/images`.
* **Full 3D bunch initialization is the preferred path for new work.**  Use
  ``momentum_axis``, ``starting_position_mm``, ``transverse_distance_mm``, and
  ``longitudinal_span_mm`` when defining new bunches. The CLI accepts these
  fields directly in rider/driver JSON, and the main GUI exposes them
  through a ``Manual Particle Config`` tab with rider/driver JSON editors.
  Legacy off-axis fields such as ``transv_offset_x``, ``transv_offset_y``, and
  ``transv_dist`` remain available for compatibility and older studies,
  including optimization-plugin offset fractions converted to absolute
  positions (offset = fraction × aperture_radius).
* **GUI and plotting tools.**  The Tkinter GUI is the usual interface for
  interactive work and ``lw-simulate`` suits scripting and batch runs.  Native
  PIC and the experimental boundary module use separate entry points.
  ``lw-generate-sweep-heatmap``, ``lw-plot-latest-live``,
  ``lw-plot-from-logcache-live`` and ``lw-plot-trajectory`` plot sweep results,
  live sweep logs and saved trajectories; see :doc:`quickstart` and
  ``--help`` for their options.
* **Reference notebooks remain available.**  Historical validation notebooks are
  retained for context, while maintained CLI and GUI workflows should be the
  baseline for new scripted sweeps.
* **Need the math?**  See :doc:`theory` for the derivation of the covariant
  equations of motion implemented in ``core/trajectory_integrator.py`` and the
  approximations used in the benchmark studies.

With the map in hand, continue to :doc:`quickstart` to set up a development
environment or jump to :doc:`validation` for the maintained validation workflow.
