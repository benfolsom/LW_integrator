Quick start
===========

Follow the steps below to prepare a development environment, run a minimal
simulation, and confirm that the regression tooling works on your machine.

1. Clone the repository and create a virtual environment (``.venv`` is assumed
   throughout the project scripts):

   .. code-block:: bash

      git clone https://github.com/benfolsom/LW_integrator/
      cd LW_integrator
      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip

2. Install the project in editable mode together with the optional extra used by
   the validation notebooks:

   .. code-block:: bash

      pip install -e .[dev]

   The ``dev`` extra mirrors the dependencies used in CI (NumPy, SciPy,
   Matplotlib, pytest, Sphinx, nbsphinx, etc.).

4. (Optional) Open the historical validation notebook for reference:

   .. code-block:: bash

      code examples/validation/core_vs_legacy_benchmark.ipynb

   Use it as a historical reference for the archived comparison workflow.  For
   current validation, prefer the pytest and CLI checks in :doc:`validation`.

5. Launch the GUI application:

   .. code-block:: bash

      python -m lw_integrator.gui

   The GUI provides three operational modes:

   * **Single Run Mode** (Main tab): Configure and execute individual simulations
     with real-time progress tracking, trajectory visualization, and interactive
     energy/position analysis. Export results in CSV, JSON, or NPZ formats. For
     new particle setups, prefer the ``Manual Particle Config`` tab so full 3D
     rider/driver JSON can be entered directly.

   * **Blind Sweep Mode** (Sweep/Optimization tab): Parameter sweeps over aperture
     radius, particle energy, transverse offset, and starting positions with
     auto-timestep calculation and configurable trajectory saving.

   * **Optimization Mode** (Sweep/Optimization tab): Global optimization using
     Genetic Algorithm, Differential Evolution, Nelder-Mead, or Multi-start
     methods with convergence detection and top-N result saving.

6. Exercise the command-line entry point:

   .. code-block:: bash

      lw-simulate --quiet

   The ``lw-simulate`` executable (also accessible via ``python -m
   lw_integrator.cli``) runs the core integrator with default settings
   (35 MeV electron approaching a conducting aperture). Override parameters
   inline or provide a JSON configuration file:

   **Inline parameter overrides:**

   .. code-block:: bash

      lw-simulate --steps 250 --time-step 5e-4 --aperture-radius 0.5 --output run.json

   **Using a native direct-integrator configuration file:**

   .. code-block:: bash

      lw-simulate --config my_scenario.json --output results.json

   **Run an existing GUI/testbed configuration unchanged:**

   .. code-block:: bash

      lw-simulate --testbed-config configs/run_configs/study_config.json \
        --output testbed_report.json

   Use ``--testbed-config`` for the full GUI/testbed JSON schema
   (``rider_params``, ``driver_params``, and ``core_params``). It loads the
   configuration through ``SimulationOptions`` and executes ``run_testbed()``,
   preserving the configured 3D particle setup, beamline geometry, source
   smearing, driver train, startup mode, self-consistency, and output settings.
   The JSON is authoritative for physics settings. Direct-run CLI overrides are
   not applied, but ``--checkpoint-dir``, ``--resume-from``, and the checkpoint
   interval flags may be supplied operationally. Use ``--config`` only for the
   separate native direct-integrator schema. See :doc:`checkpoints` for restart
   commands and the current fixed-step boundary.

   Example native direct-integrator JSON configuration structure:

   .. code-block:: json

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
          "particle_count": 5,
          "starting_position_mm": [0.05, 0.0, 0.0],
          "momentum_axis": [0.0, 0.0, 1.0],
          "transverse_distance_mm": 0.01,
          "longitudinal_span_mm": 0.02
        }
      }

   Additional options include ``--chrono-mode``, ``--startup-mode``,
   ``--image-weighting``, and ``--self-consistency``. Run ``lw-simulate --help``
   for the complete list.

   Intrinsic magnetic moments are experimental and off by default.  The
   selected RFS model is currently guarded to fixed-step
   ``BUNCH_TO_BUNCH`` point-charge runs using ``COLD_START`` or
   ``INERTIAL_PREHISTORY``, with same-bunch space charge disabled.
   Dynamic recoil is either off or the explicit charge-only ``medina_lad``
   hybrid.  For example, add these switches to a
   suitable electron--proton BUNCH_TO_BUNCH run:

   .. code-block:: bash

      lw-simulate --simulation-type bunch-to-bunch \
         --startup-mode inertial-prehistory \
         --radiation-reaction-mode off --no-adaptive-timestep \
         --magnetic-dipoles --stern-gerlach \
         --rider-magnetic-species electron --rider-spin 1 0 0 \
         --driver-magnetic-species proton --driver-spin 0 0 1

   ``--magnetic-dipoles`` without ``--stern-gerlach`` enables spin transport
   only.  Adding ``--stern-gerlach`` selects the fully coupled full-G response;
   ``--no-spin-precession`` leaves a frozen-spin force diagnostic rather than a
   complete RFS evolution.  To add the ordinary non-self field of each moment,
   also select ``--dipole-source full-retarded-point``.  The optional
   ``--dipole-source-cutoff-mm`` value is a strict point-singularity abort
   boundary, not softening.  ``--exact-retarded-backend
   numba_roots_exact_serial`` opts the exact charge and dipole providers into a
   cross-platform serial CPU root kernel while preserving Python source and
   stencil assembly; ``python`` is the default.
   ``numba_full_strict_serial`` additionally compiles the strict per-source
   charge and Hertz event paths.  It is faster but tolerance-validated rather
   than bitwise-identical.  Charge and dipole stencil centers stay on the
   Python reference path.  On Apple silicon,
   ``metal_certified_full_strict`` may accelerate sufficiently large dipole
   root batches, but only as float32 bracket proposals certified against the
   original float64 data; the strict CPU root and fields remain authoritative.
   Small calls stay on the CPU and there is no automatic platform dispatch.
   Neutral
   particles and prescribed gradients are best configured in a saved JSON.
   See :doc:`magnetic_dipole_moments` for the numerical contract, all hard
   scope guards, the diagnostic legacy models, and retarded-source limits.

   ``inertial-prehistory`` is the appropriate boundary model for an incoming
   particle that existed before active time zero.  It constructs eight sparse
   coasting knots, extends their causal span until every initial exact charge
   and dipole stencil is bracketed, hides those knots from normal output, and
   initializes canonical momentum once from the total retarded potential.
   Each later accepted step advances mechanical :math:`qF+\mu G` and then
   reconstructs canonical momentum from the accepted endpoint potential.  It
   does not reconstruct the earlier interacting trajectory or prime Medina's
   force derivative, so encounter results must converge as the active starting
   separation is moved outward.  Use ``cold-start`` instead for a physical
   field turn-on transient.

   Replacing ``--radiation-reaction-mode off`` with ``medina_lad`` enables the
   charge-only RFS/Medina hybrid.  It does not include intrinsic-dipole
   self-recoil or charge--dipole radiation-interference recoil, and any run
   with a capped Medina impulse is unsuitable as capture evidence.

   **Running a parameter sweep from the CLI:**

   .. code-block:: bash

      lw-simulate --sweep-config configs/sweep_configs/005_08_b2b_sweep_E_spread.json

   Sweep results are written to ``results/sweeps/YYYYMMDD_HHMMSS_configname/``
   with detailed debug logs in ``logcache/``.  Sweeps with fewer than 100
   completed runs are automatically relocated to
   ``results/archive/incomplete/``.
   Sequential CLI sweeps also honor the config's ``per_run_timeout`` value:
   a point that exceeds the timeout is saved as a failed timed-out run and the
   sweep continues to the next grid point.

   Bunch-to-bunch sweep examples model proton-mass, opposite-charge beams
   (proton/H- in the current configs) with the driver bunch passing through a
   virtual exit aperture shortly after the interaction point. This screens the
   rider from direct line of sight downstream, so the heatmaps show residual
   post-screening fields rather than indefinitely visible bunch-bunch coupling.

   Fine-tune diagnostic output during sweeps without editing the JSON config:

   .. code-block:: bash

      lw-simulate --sweep-config my_sweep.json --log-verbosity full --sc-verbosity 2 --adaptive-debug

   * ``--log-verbosity {none,truncated,full}`` — controls what is saved to disk.
   * ``--sc-verbosity {0,1,2,3}`` — self-consistency iteration detail level.
   * ``--adaptive-debug`` / ``--no-adaptive-debug`` — adaptive timestep diagnostics.

   .. note::

      As of v0.6.0 the CLI sweep runner calls the **same** ``run_testbed()`` /
      ``SimulationOptions`` code paths as the GUI, so results are identical
      between the two interfaces.

   You can replicate the same behaviour programmatically by calling
   ``lw_integrator.cli.main`` with a list of CLI-style arguments; see
   ``examples/entrypoint_demo.py`` for a minimal example.

7. Generate the HTML documentation locally:

   .. code-block:: bash

      cd docs
      ./build_docs.sh --clean --type html

   Open ``docs/build/html/index.html`` in a browser to browse the rendered pages.

8. Run the integration and CLI/GUI parity checks relevant to your change:

   .. code-block:: bash

      pytest tests/test_integration_e2e.py
      pytest tests/test_cli_gui_parity.py

   These tests exercise maintained end-to-end solver paths and the intended
   headless baseline for sweep behavior.

9. Run a macroparticle simulation (conducting-wall mode):

   .. code-block:: python

      from lw_integrator.testbed_runner import SimulationOptions, run_testbed
      from core.types import SimulationType

      options = SimulationOptions(
          simulation_type=SimulationType.CONDUCTING_WALL,
          steps=1000,
          macroparticle_enabled=True,
          macroparticle_charge_multiplier=10.0,     # 10× charge scaling
          macroparticle_position_spread=1e-5,       # 10 μm position σ
          macroparticle_momentum_spread=1e-6,       # Momentum spread
          core_params={
              'time_step': 3e-7,
              'wall_z': 100.0,
              'aperture_radius': 0.001,
          },
      )

      result = run_testbed(options)

   This example demonstrates beam emittance modeling with stochastic errors
   applied to image subcharges before charge attenuation calculations.

10. Run a full 3D off-axis beam simulation:

   .. code-block:: python

      from lw_integrator.testbed_runner import SimulationOptions, run_testbed
      from core.types import SimulationType

      rider_params = {
          'kinetic_energy_mev': 35.0,
          'mass_amu': 0.000548579909,
          'charge_sign': -1.0,
          'stripped_ions': 1.0,
          'particle_count': 5,
          'starting_position_mm': [5e-5, 0.0, 0.0],
          'momentum_axis': [0.0, 0.0, 1.0],
          'transverse_distance_mm': 1e-5,
          'transverse_momentum': 0.0,
          'longitudinal_span_mm': 2e-5,
      }

      options = SimulationOptions(
          simulation_type=SimulationType.CONDUCTING_WALL,
          steps=1000,
          manual_particle_config_enabled=True,
          rider_params=rider_params,
          core_params={
              'time_step': 1e-7,
              'wall_z': 100.0,
              'aperture_radius': 0.0001,  # 100 μm aperture
          },
      )

      result = run_testbed(options)

   This demonstrates the preferred maintained path for off-axis beam
   positioning. The bunch centroid starts 50 μm off-axis in ``x`` with a full
   3D spread: transverse disk radius 10 μm and longitudinal span 20 μm along
   the propagation axis.

Next steps
----------

* :doc:`validation` details the maintained pytest, CLI, GUI, and plotting checks
  used as the current regression baseline.
* :doc:`notebooks` provides guidance on using the interactive assets efficiently
  (plot styling, DPI control, output directories, etc.).
* :doc:`recent_changes` describes the macroparticle simulation feature and other
  recent enhancements including optimization convergence and physics corrections.
* :doc:`development/index` is the entry point for coding conventions, testing
  expectations, and contribution guidelines.
