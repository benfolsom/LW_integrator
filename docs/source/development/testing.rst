Testing expectations
====================

The repository uses ``pytest`` for both unit and integration coverage.  The
``tests`` directory is split by scope:

* ``tests/unit`` covers deterministic helpers such as image charge generation or
  chronology matching.
* ``tests/test_integration_e2e.py`` and targeted root-level tests exercise
  end-to-end solver behavior and self-consistency wrapper paths.
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

The tracked workflow builds documentation; it does not run pytest or impose a
suite-wide seed. Set seeds explicitly in stochastic tests and configurations.
Run the subset relevant to a change and attach its command and output when
raising an issue. Adapt existing tests before adding new ones.

Pre-commit hooks
----------------

Install the project’s hooks once:

.. code-block:: bash

   pre-commit install

The configuration runs Black, Ruff, mypy, and file-format checks; it has no
isort hook. Execute ``pre-commit run --all-files`` for these local checks.
The tracked docs workflow does not run them.
