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

4. (Optional) Launch the validation notebook for interactive comparison:

   .. code-block:: bash

      code examples/validation/core_vs_legacy_benchmark.ipynb

   From VS Code or Jupyter Lab, execute the cells and experiment with the
   widgets. The notebook runs the same ``run_benchmark`` helper that the CLI
   uses and can save overlay plots and ΔE scatter figures directly.

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

   For detailed optimization workflows, see ``local/SWEEP_AND_OPTIMIZATION_GUIDE.md``.

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

   Additional options include ``--chrono-mode``, ``--startup-mode``,
   ``--image-weighting``, and ``--self-consistency``. Run ``lw-simulate --help``
   for the complete list.

   You can replicate the same behaviour programmatically by calling
   ``lw_integrator.cli.main`` with a list of CLI-style arguments; see
   ``examples/entrypoint_demo.py`` for a minimal example.

7. Generate the HTML documentation locally:

   .. code-block:: bash

      cd docs
      ./build_docs.sh --clean --type html

   Open ``docs/build/html/index.html`` in a browser to browse the rendered pages.

8. Run the integration test suite to validate physics parity:

   .. code-block:: bash

      pytest tests/integration/test_core_integrators.py -k two_particle

   The tests confirm that the core and legacy solvers agree on canonical
   benchmarks to within floating-point tolerances.

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

* :doc:`validation` details the scripts and notebooks that compare the core and
  legacy implementations across multiple seeds and integration lengths.
* :doc:`notebooks` provides guidance on using the interactive assets efficiently
  (plot styling, DPI control, output directories, etc.).
* :doc:`recent_changes` describes the macroparticle simulation feature and other
  recent enhancements including optimization convergence and physics corrections.
* :doc:`development/index` is the entry point for coding conventions, testing
  expectations, and contribution guidelines.
