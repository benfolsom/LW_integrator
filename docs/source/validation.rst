Validation and regression
==========================

The LW Integrator repository keeps parity with the archived physics by running a
suite of scripted comparisons.  This page explains the assets you can use to
confirm changes and to reproduce the plots that appear in publications.

Command-line benchmarks
-----------------------

The ``examples/comparison`` directory contains utilities aimed at comparing the
core solver with its legacy counterpart.

``legacy_vs_core_systematic_comparison.py``
    Runs a configurable sweep over random seeds and integration lengths.  Use
    the ``--seed`` and ``--steps`` flags (or pass lists via the ``run_suite``
    helper) to generate metrics such as maximum relative differences in ``z``,
    ``Pt``, and ``gamma``.

``legacy_vs_core_trajectory_comparison.py``
    Recreates the canonical two-particle demo from the legacy notebooks, plots
    the position and γ-series overlays, and highlights discrepancies.  The
    refreshed version uses colourblind-safe palettes and high-DPI scatter
    rendering by default.

``core_vs_legacy_benchmark.py``
    The central benchmarking CLI.  Key arguments:

    * ``--simulation-type {conducting,switching,bunch_to_bunch}`` chooses the
      wall configuration via :class:`~core.trajectory_integrator.SimulationType`.
    * ``--steps`` / ``--time-step`` / ``--seed`` control integration length and
      reproducibility.
    * ``--save-json`` writes metrics to disk; ``--save-fig`` exports the overlay
      plot using the palette shared with the notebooks; ``--plot-dpi`` sets the
      output resolution (150–600 dpi).
    * ``--skip-legacy`` runs only the modern solver when you are iterating on
      new physics and the ground truth is not available.

Notebook workflows
------------------

``examples/validation/core_vs_legacy_benchmark.ipynb``
    Interactive front-end to ``run_benchmark`` with widgets for rider and driver
    parameters, simulation type, and export options.  It produces ΔE scatter
    plots, and can save both figures and metrics.

``examples/validation/integrator_testbed.ipynb``
    Exploratory environment for all supported simulation types.  It disables
    irrelevant configuration controls dynamically and mirrors the plot styling
    used by the scripted tools so that figures remain consistent across entries.

Practical tips
--------------

* When updating physics, modify the validation scripts first so that the docs
  and notebooks inherit the new behaviour naturally.
* The helper :func:`core.performance.run_benchmark` returns both metrics and the
  raw trajectories.  Use the payload to generate custom diagnostics without
  re-running the solver.
* Keep the output directories (`test_outputs/notebook_runs/` and
  `test_outputs/testbed_runs/`) tidy; they are ignored by Git but referenced in
  the notebooks.

If a validation run reveals a regression, open an issue describing the scenario
and attach the metrics JSON along with any overlay plots that highlight the
breakdown.
