Overview
========

The LW Integrator is a covariant electromagnetic particle tracking code tuned for
accelerator physics studies.  The validated implementation still lives in the
``legacy/`` tree, but the modern ``core/`` package mirrors the same physics with
clearer structure, type hints, and unit-tested helper utilities.  The method is
described in *Relativistic beam loading, recoil-reduction, and residual-wake
acceleration with a covariant retarded-potential integrator*
(``https://doi.org/10.1016/j.nima.2024.169988`` / ``https://arxiv.org/abs/2310.03850``).
This page summarises the pieces you will encounter when navigating the
repository.

High-level anatomy
------------------

``core/``
    The maintained implementation of the retarded Liénard–Wiechert solver.
    ``trajectory_integrator.py`` retains the original algorithm but exposes it
    through typed helper functions (image charge generators, retarded-distance
    utilities, and the ``IntegratorConfig`` data class).  ``performance.py``
    bundles optional Numba kernels that accelerate large runs without changing
    the underlying physics.  ``self_consistency.py`` holds the fixed-point
    iteration used for radiation-reaction corrections.

``legacy/``
  Archived notebooks and scripts from the original codebase.  They are kept
  for regression comparisons and historical reference.  Production workflows
  should use ``core/`` unless you are debugging a discrepancy.  The historical
  "static" integrator lives here for completeness; it is considered deprecated
  and is not exercised by the modern documentation or tooling.

``examples/``
    Ready-to-run validation material.  The ``validation/`` folder contains both
    Python scripts and Jupyter notebooks that reproduce the legacy vs. core
    comparisons.  The ``comparison/`` folder houses CLI benchmarks that report
    metrics (maximum Δγ, Δz, etc.) across seeded simulation suites.

``tests/``
    Automated regression coverage.  ``tests/integration/test_core_integrators.py``
    verifies equivalence between the core solver, its self-consistent variant,
    and the legacy implementation.  ``tests/unit`` hosts deterministic unit
    tests for helper functions such as ``generate_conducting_image``.

``input_output/``
    Utilities for constructing particle bunch dictionaries in the format the
    integrator expects.  ``bunch_initialization.py`` is the main entry point and
    is documented in the API section below.

``docs/``
    The refreshed documentation that you are currently reading.  Sphinx builds
    use the configuration in ``docs/source/conf.py`` and the helper script
    ``docs/build_docs.sh``.

Key ideas to keep in mind
-------------------------

* **Physics parity matters.**  The core code is intentionally a transcription of
  the legacy solver.  Any behavioural change should come with matching updates
  to the validation scripts and the integration tests.
* **Particle states are dictionaries of NumPy arrays.**  Whenever you initialize
  particles manually, fill every expected key (``x``, ``Pz``, ``gamma``, ``q``,
  ``char_time`` …) or use ``input_output.create_bunch_from_energy`` to obtain a
  correctly shaped state.
* **Simulation modes are enumerated.**  ``SimulationType`` enumerates the three
  supported wall configurations.  The solver mirrors the legacy integer flags so
  comparison runs remain straightforward.
* **Notebook tooling is first-class.**  The validation notebooks are kept in
  sync with the scripts and expose colourblind-friendly plots, high-DPI export,
  and configuration widgets.  Use them to explore scenarios before committing to
  scripted sweeps.
* **Need the math?**  See :doc:`theory` for the derivation of the covariant
  equations of motion implemented in ``core/trajectory_integrator.py`` and the
  approximations used in the benchmark studies.

With the map in hand, continue to :doc:`quickstart` to set up a development
environment or jump to :doc:`validation` for the comparison workflows.
