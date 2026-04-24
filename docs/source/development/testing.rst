Testing expectations
====================

The repository uses ``pytest`` for both unit and integration coverage.  The
``tests`` directory is split by scope:

* ``tests/unit`` covers deterministic helpers such as image charge generation or
  chronology matching.
* ``tests/integration`` exercises the end-to-end solvers, comparing core vs.
  archived reference behavior and validating the self-consistency wrapper.
* ``tests/physics`` retains regression checks for specific physics
  invariants (charge conservation, relativistic momentum, etc.).
* ``tests/test_cli_gui_parity.py`` verifies that the CLI sweep runner and the
  GUI's Blind Sweep mode produce identical configurations and results when given
  the same input.  Added in v0.6.0 alongside the CLI/GUI code-path
  unification.

Running tests
-------------

With the virtual environment activated, run:

.. code-block:: bash

   pytest

CI runs ``pytest`` with a pinned random seed to keep the stochastic benchmarks
stable.  When diagnosing failures locally, re-run the same subset as CI and
attach the pytest output when raising an issue.

Pre-commit hooks
----------------

Install the project’s hooks once:

.. code-block:: bash

   pre-commit install

The configuration formats Python files with ``black``, sorts imports via
``isort``, and runs ``ruff`` for linting.  Execute ``pre-commit run --all-files``
if you would like to mirror the CI checks manually.
