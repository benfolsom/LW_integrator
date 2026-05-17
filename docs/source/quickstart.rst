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
     energy/position analysis. Export results in CSV, JSON, or NPZ formats.

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

   **Using a configuration file:**

   .. code-block:: bash

      lw-simulate --config my_scenario.json --output results.json

   Example JSON configuration structure:

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
          "position_z": 0.0
        }
      }

   Additional options include ``--chrono-mode``, ``--startup-mode``,
   ``--image-weighting``, and ``--self-consistency``. Run ``lw-simulate --help``
   for the complete list.

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

   For the March 2026 35 um B2B energy/spread reference domain, use
   ``configs/sweep_configs/005_08_b2b_sweep_E_spread_png_limits_40x40.json``
   when you want the same plotted 0.5-3000 GeV and 0.01-100 m limits with a
   reduced 40x40 grid.

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

10. Run an off-axis beam simulation with transverse offset:

   .. code-block:: python

      from lw_integrator.testbed_runner import SimulationOptions, run_testbed
      from core.types import SimulationType

      # Create an off-axis beam at 50 μm from center with ±10 μm spread
      rider_params = {
          'starting_distance': 0.0,
          'transv_mom': 0.0,
          'starting_Pz': 1e6,
          'stripped_ions': 1.0,
          'm_particle': 0.000548579909,  # electron mass
          'transv_dist': 1e-5,           # ±10 μm beam spread
          'transv_offset_x': 5e-5,       # 50 μm off-axis in x
          'transv_offset_y': 0.0,        # on-axis in y
          'pcount': 5,
          'charge_sign': -1.0,
      }

      options = SimulationOptions(
          simulation_type=SimulationType.CONDUCTING_WALL,
          steps=1000,
          rider_params=rider_params,
          core_params={
              'time_step': 1e-7,
              'wall_z': 100.0,
              'aperture_radius': 0.0001,  # 100 μm aperture
          },
      )

      result = run_testbed(options)

   This demonstrates off-axis beam positioning, useful for aperture tolerance
   studies and beam halo analysis. Particles are distributed uniformly in
   x ∈ [40, 60] μm and y ∈ [-10, 10] μm.

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
