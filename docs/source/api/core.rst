Core integrator
===============

.. automodule:: core.trajectory_integrator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: core.beamline_geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: core.particle_initialization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: core.species
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: core.magnetic_dipole
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: core.rfs
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: core.retarded_fields.evaluate_retarded_charge_field_native

.. autofunction:: core.retarded_fields.evaluate_retarded_charge_field_gradient_native

Public run and field types
--------------------------

.. autofunction:: core.integration_runner.retarded_integrator

.. autofunction:: core.equations.retarded_equations_of_motion

.. autoclass:: core.types.IntegratorConfig
   :no-members:

.. autoclass:: core.types.BeamlineGeometryConfig
   :no-members:

.. autoclass:: core.types.TrajectoryArrays
   :no-members:

.. autoclass:: core.types.IndexedTrajectoryArrays
   :no-members:

.. autoclass:: core.types.TrialTrajectoryHistory
   :no-members:

.. autoclass:: core.exact_source_cloud.ExactCloudHistory
   :no-members:

.. autoclass:: core.retarded_fields.ObserverEvent
   :no-members:

.. autoclass:: core.retarded_fields.RetardedChargeFieldResult
   :no-members:

.. autoclass:: core.retarded_fields.RetardedChargeFieldGradientResult
   :no-members:

.. autodata:: core.constants.C_MMNS

Additional configuration and result contracts
---------------------------------------------

The following names are the types referenced by the run signatures above.
They are defined in ``core/types.py`` unless another module is shown.

.. py:currentmodule:: None

.. py:class:: core.types.ChronoMatchingMode

   Sampled retardation strategy. See :doc:`../self_consistency`.

.. py:class:: core.types.StartupMode

   Startup history strategy. See :doc:`../exact_path`.

.. py:class:: core.types.PseudoGridConfig

   Experimental reduced-solver settings. See :doc:`../overview`.

.. py:class:: core.types.MacroparticleSmearingConfig

   Bounded source-smearing settings. See :doc:`../exact_macroparticle_sources`.

.. py:class:: core.types.CavityExitConfig

   Bunch-to-bunch early termination and tail settings; exact routes retain
   their restrictions in :doc:`../exact_path`.

.. py:class:: core.types.MagneticDipoleConfig

   Experimental intrinsic-moment response and source settings. See
   :doc:`../magnetic_dipole_moments`.

.. py:class:: core.types.CheckpointConfig

   Accepted-step restart controls. See :doc:`../checkpoints`.

.. py:class:: core.types.AdaptivePairReturnConfig

   Shared-lab-time adaptive settings. See :doc:`../multirate_return`.

.. py:class:: core.self_consistency.SelfConsistencyConfig

   Force-trial and mass-shell stopping settings, with two fixed-geometry trials
   by default. See :doc:`../self_consistency` for fields and accuracy limits.

.. py:data:: core.types.ParticleState

   Mapping from field names to NumPy arrays, one element per particle.

.. py:data:: core.types.Trajectory

   List of ``ParticleState`` snapshots.

.. autoclass:: core.distances.ChronoMatchResult
   :no-members:

.. autoexception:: core.integration_runner.IntegrationCancelled

.. autoexception:: core.integration_runner.EnergyJumpDetected

.. autofunction:: core.integration_runner.run_integrator

Directional visibility helper
-----------------------------

.. py:currentmodule:: None

.. py:function:: core.beamline_geometry.compute_directional_visibility_mask(source_positions, geometry, observer_direction)

   Return a Boolean array of shape ``(N,)`` for retarded source positions of
   shape ``(N, 3)``. ``geometry`` is a
   :class:`core.types.BeamlineGeometryConfig`; ``observer_direction`` is a
   three-component propagation direction. The helper selects the occluder most
   aligned with that direction and checks its transverse aperture and axial
   extent. Disabled geometry or no occluders leaves all sources visible.
