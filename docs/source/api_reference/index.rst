API Reference
=============

Complete API documentation for the LW Integrator. This section provides detailed information about the core electromagnetic integration classes, mathematical foundations, and stability requirements.

.. toctree::
   :maxdepth: 2

   core

Quick Reference
---------------

Core Integration Classes
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   LW_integrator.covariant_integrator_library_heavyion.LienardWiechertIntegrator

**Key Features:**

* **Conjugate Momentum Formulation**: Canonical momentum P = γmv + qA ensuring gauge invariance
* **Explicit Integration Scheme**: Computationally efficient predictor-corrector method
* **Liénard-Wiechert Fields**: Exact electromagnetic field calculations with retardation
* **Energy Conservation Monitoring**: Advanced stability control and accuracy validation
* **Adaptive Time Stepping**: Energy-dependent integration parameters for numerical stability

**Critical Usage Requirements:**

* **Minimal Transverse Momentum**: Initialize with px_fraction, py_fraction ≤ 1e-6
* **Early Cutoff Mechanisms**: Terminate integration after physical events to prevent runaway behavior
* **Energy-Dependent Time Steps**: Use smaller Δt for lower energy particles
* **Conservation Monitoring**: Maintain |ΔE/E| < 10⁻⁶ for stable integration

Mathematical Foundation
~~~~~~~~~~~~~~~

.. autosummary::

   lw_integrator.physics.electromagnetic_fields
   lw_integrator.physics.radiation_reaction
   lw_integrator.physics.space_charge
   lw_integrator.physics.constants

Integration Methods
~~~~~~~~~~~~~~~~~~

.. autosummary::

   lw_integrator.integrators.adaptive_integrator
   lw_integrator.integrators.basic_integrator
   lw_integrator.integrators.self_consistent_integrator

I/O and Utilities
~~~~~~~~~~~~~~~~~

.. autosummary::

   lw_integrator.io.standard_input_format
   lw_integrator.io.output_formats
   lw_integrator.utilities.plotting
   lw_integrator.utilities.analysis
