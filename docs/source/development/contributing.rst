Contributing
============

We welcome patches that improve the physics fidelity, performance, or ergonomics
of the LW Integrator.  Please keep the following ground rules in mind:

* **Stay regression safe.**  All changes that touch physics need focused pytest
  coverage and, where relevant, CLI/GUI parity checks.  Historical notebooks can
  inform investigations, but they are not the active validation baseline.
* **Document as you go.**  Extend the relevant page under ``docs/source/`` when
  you add or change behaviour. The tracked docs workflow builds HTML without
  warning-as-error. The local ``bash docs/build_docs.sh`` adds ``-W`` and
  ``--keep-going``, so warnings fail that stricter build.
* **Prefer type hints and small helpers.**  The modern ``core/`` modules are a
  descendants of the historical solver, with separate exact-history, PIC,
  and boundary implementations. Keep their numerical contracts explicit.
* **Use the project tooling.**  Formatting, linting, and tests are orchestrated
  via ``pre-commit`` hooks and ``pytest``.  Run ``pre-commit install`` once and
  ``pre-commit run --all-files`` before sending a PR.

Process
-------

1. Branch from ``development``. Release history uses ``master``, while the
   tracked docs workflow builds on ``main`` and ``development`` and deploys
   only from ``main``. This configuration does not establish the live deployment
   state; release maintainers must reconcile that branch mismatch before relying
   on automatic publication.
2. Update documentation and adapt existing tests; add tests only when strongly justified.
3. Run the pytest, CLI, plotting, or docs checks relevant to your change.
4. Submit a pull request summarising the physics context and validation output.

Reach out via the issue tracker if you plan a large refactor so we can align on
scope before you begin.
