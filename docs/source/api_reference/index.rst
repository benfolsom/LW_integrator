API Reference
=============

Complete API documentation for the LW Integrator. This section provides detailed information about all classes, functions, and modules.

.. toctree::
   :maxdepth: 2

   core
   physics
   integrators
   io
   utilities

Quick Reference
---------------

Core Classes
~~~~~~~~~~~~

.. autosummary::

   lw_integrator.StandardInputFormat
   lw_integrator.BeamParameters
   lw_integrator.LatticeElement
   lw_integrator.SimulationParameters
   lw_integrator.MacroParticle
   lw_integrator.MacroParticleEnsemble

Physics Modules
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
