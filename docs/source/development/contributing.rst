Contributing
============

We welcome patches that improve the physics fidelity, performance, or ergonomics
of the LW Integrator.  Please keep the following ground rules in mind:

* **Stay regression safe.**  All changes that touch physics must keep the core
  and legacy solvers in agreement.  Update the validation scripts and add tests
  if necessary.
* **Document as you go.**  Extend the relevant page under ``docs/source/`` when
  you add or change behaviour.  The documentation is built as part of CI and
  warnings fail the build.
* **Prefer type hints and small helpers.**  The modern ``core/`` modules are a
  transcription of the legacy code; helper functions exist to keep the flow
  readable without altering the math.
* **Use the project tooling.**  Formatting, linting, and tests are orchestrated
  via ``pre-commit`` hooks and ``pytest``.  Run ``pre-commit install`` once and
  ``pre-commit run --all-files`` before sending a PR.

Process
-------

1. Fork and branch from ``main``.
2. Make changes with accompanying tests and documentation.
3. Run the validation scripts relevant to your change.
4. Submit a pull request summarising the physics context and validation output.

Reach out via the issue tracker if you plan a large refactor so we can align on
scope before you begin.
