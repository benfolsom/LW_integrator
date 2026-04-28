Validation and Regression
=========================

Supported validation now centers on the maintained core, CLI, GUI, and plotting
entry points.  The archived solver under ``legacy/`` and the historical
comparison notebooks remain in the repository as reference material, but they
are not the baseline for new workflows.

Automated Checks
----------------

Use the pytest suite as the primary regression gate:

* ``pytest -m unit`` for deterministic helper and control-flow coverage.
* ``pytest -m physics`` for numerical and stability checks.
* ``pytest tests/test_cli_gui_parity.py`` when changes touch CLI/GUI parity.
* ``pytest tests/test_plotting_tools.py tests/test_trajectory_plotter.py`` for
  maintained sweep and single-run plotting commands.

For broad local validation, run ``pytest`` from the repository root.

Maintained Plotting Validation
------------------------------

Sweep post-processing should use the packaged commands:

* ``lw-generate-sweep-heatmap`` for publication-quality sweep heatmaps.
* ``lw-plot-latest-live`` to follow the newest sweep log.
* ``lw-plot-from-logcache-live`` for a specific sweep log in static or live mode.
* ``lw-plot-trajectory`` for saved single-run JSON or NPZ trajectory files.

These tools are covered by focused CLI tests and should remain the supported
surface for plotting-related changes.

Reference Notebooks
-------------------

``examples/validation/core_vs_legacy_benchmark.ipynb`` and
``examples/validation/integrator_testbed.ipynb`` are kept as historical
reference notebooks.  They may be useful for understanding older analysis
paths, but active development should prefer the maintained CLI, GUI, and pytest
coverage described above.

When a validation run reveals a regression, preserve the config, command, and
result artifact that reproduce it, then add a focused pytest case before fixing
the implementation.
