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

3. Run the integration test suite to confirm the core and legacy solvers agree
   on the canonical benchmark:

   .. code-block:: bash

      pytest tests/integration/test_core_integrators.py -k two_particle

   Expect to see both the pure core solver and the self-consistent wrapper match
   the legacy reference to within floating-point tolerances.

4. Launch the colourblind-friendly benchmark notebook:

   .. code-block:: bash

      code examples/validation/core_vs_legacy_benchmark.ipynb

   From VS Code or Jupyter Lab, execute the cells and experiment with the
   widgets.  The notebook runs the same ``run_benchmark`` helper that the CLI
   uses and can save overlay plots and ΔE scatter figures directly.

5. Exercise the command-line entry point:

   .. code-block:: bash

      lw-simulate --quiet

   The ``lw-simulate`` executable (also accessible via ``python -m
   lw_integrator.cli``) runs the core integrator with the default configuration.
   Override parameters as needed, for example to shorten the run and capture a
   JSON summary:

   .. code-block:: bash

      lw-simulate --steps 250 --time-step 5e-4 --output run.json

   You can replicate the same behaviour programmatically by calling
   ``lw_integrator.cli.main`` with a list of CLI-style arguments; see
   ``examples/entrypoint_demo.py`` for a minimal example.

6. Generate the HTML documentation locally:

   .. code-block:: bash

      cd docs
      ./build_docs.sh --clean --type html

   Open ``docs/build/html/index.html`` in a browser to browse the rendered pages.

Next steps
----------

* :doc:`validation` details the scripts and notebooks that compare the core and
  legacy implementations across multiple seeds and integration lengths.
* :doc:`notebooks` provides guidance on using the interactive assets efficiently
  (plot styling, DPI control, output directories, etc.).
* :doc:`development/index` is the entry point for coding conventions, testing
  expectations, and contribution guidelines.
