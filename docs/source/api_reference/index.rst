API Reference
=============

Complete API documentation for the LW integrator refactor. This section
focuses on the production modules that now ship inside ``core/`` and their
related utilities.

.. toctree::
   :maxdepth: 2

   core

Quick Reference
---------------

Core Modules
~~~~~~~~~~~~

* :mod:`core.trajectory_integrator` – validated Python reference path
* :mod:`core.performance` – Numba-accelerated kernels with graceful fallback
* :mod:`core.self_consistency` – optional Δγ convergence helper
* :mod:`input_output.updated_bunch_initialization` – modern bunch factory

Key Entry Points
~~~~~~~~~~~~~~~~

.. autosummary::

   core.trajectory_integrator.retarded_integrator
   core.performance.retarded_integrator_numba
   core.performance.run_optimised_integrator
   core.self_consistency.self_consistent_step

Supporting Types
~~~~~~~~~~~~~~~~

.. autosummary::

   core.trajectory_integrator.IntegratorConfig
   core.trajectory_integrator.SimulationType
   core.performance.OptimisationOptions
   core.self_consistency.SelfConsistencyConfig
