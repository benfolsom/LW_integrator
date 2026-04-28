Working with the notebooks
==========================

Several historical Jupyter notebooks ship with the repository as reference
material.  Maintained CLI and GUI workflows are the baseline for new
development, but the notebooks remain useful for understanding older studies.

Environment setup
-----------------

* Activate the project’s virtual environment (``source .venv/bin/activate``) and
  install the editable package with the ``dev,docs`` extras to pull in
  ``nbsphinx`` and notebook-friendly dependencies.
* Launch VS Code or Jupyter Lab from the repository root when inspecting
  historical notebooks.

Core notebooks
--------------

``examples/validation/core_vs_legacy_benchmark.ipynb``
    Historical widget-driven interface for archived-reference comparisons.
    It is retained as a reference artifact rather than a supported entry point.

``examples/validation/integrator_testbed.ipynb``
    Historical UI focused on experimenting with the core solver directly.
    Treat archived-reference overlays as reference material rather than the
    supported validation baseline.

``examples/validation/conducting_aperture_test.ipynb`` (archived)
    Archived exploratory notebook; useful when cross-checking the 35 MeV
    conducting aperture scenario documented in :mod:`examples.validation.conducting_aperture_test`.

The historical "static" integrator surfaces only inside archived reference
material.  Active validation should use the pytest and CLI checks described in
:doc:`validation`.

Best practices
--------------

* Keep executions deterministic by setting the `Seed` widget where available.
* Use the built-in DPI selector instead of ``plt.savefig`` overrides so the
  styling stays consistent between notebooks and scripts.
* Store exported figures outside tracked source directories.
* When you add a new notebook, link it here and ensure ``nbsphinx`` can import
  any custom modules without executing the cells (``nbsphinx_execute = 'never'``
  in ``conf.py`` avoids long builds).

Need a notebook-specific tweak?  Prefer moving reusable behavior into maintained
modules, then keep notebook code as a thin reference layer.
