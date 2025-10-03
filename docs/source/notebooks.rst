Working with the notebooks
==========================

Several Jupyter notebooks ship alongside the benchmark scripts.  They are kept
in sync with the code so you can diagnose scenarios interactively and export
publication-ready figures without hand-tuning matplotlib settings.

Environment setup
-----------------

* Activate the project’s virtual environment (``source .venv/bin/activate``) and
  install the editable package with the ``dev`` extra to pull in ``nbsphinx`` and
  notebook-friendly dependencies.
* Launch VS Code or Jupyter Lab from the repository root so relative imports and
  output directories resolve correctly.

Core notebooks
--------------

``examples/validation/core_vs_legacy_benchmark.ipynb``
    Widget-driven interface to :func:`examples.validation.core_vs_legacy_benchmark.run_benchmark`.
    It exposes every rider/driver parameter, simulation-type toggle, and export
    option.  Overlay plots use the colourblind-friendly palette introduced in
    the refreshed scripts and can be saved at 150–600 dpi.

``examples/validation/integrator_testbed.ipynb``
    A slimmer UI focused on experimenting with the core solver directly.  The
    notebook greys out configuration controls that are irrelevant for the
    selected simulation type and prints a JSON summary of the effective
    parameters so you can paste them into regression tests.

``examples/validation/conducting_aperture_test.ipynb`` (legacy)
    Archived exploratory notebook; useful when cross-checking the 35 MeV
    conducting aperture scenario documented in :mod:`examples.validation.conducting_aperture_test`.

Best practices
--------------

* Keep executions deterministic by setting the `Seed` widget where available.
* Use the built-in DPI selector instead of ``plt.savefig`` overrides so the
  styling stays consistent between notebooks and scripts.
* Store exported figures in the per-notebook directories under
  ``test_outputs/``; these paths are ignored by Git and referenced in the docs
  and contributing guidelines.
* When you add a new notebook, link it here and ensure ``nbsphinx`` can import
  any custom modules without executing the cells (``nbsphinx_execute = 'never'``
  in ``conf.py`` avoids long builds).

Need a notebook-specific tweak?  Update both the notebook and the corresponding
script so users who prefer the CLI see the same behaviour.
