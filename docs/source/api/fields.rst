PIC, boundary, and external-field APIs
======================================

These interfaces have separate unit and stepping contracts. Native PIC uses SI
and lab time; boundary fields use normalized Heaviside–Lorentz units with c=1;
LW external providers use mm, ns, and native fields. Read :doc:`../native_pic`,
:doc:`../boundary`, and :doc:`../external_fields` before combining them.

Native PIC and correction
-------------------------

.. autoclass:: core.pic.grid.Grid
   :no-members:

.. autoclass:: core.pic.grid.Species
   :no-members:

.. autoclass:: core.pic.grid.ElectrostaticPIC
   :members: fields, sample, push

.. py:class:: core.pic.correction.CorrectionConfig

   Frozen keyword configuration for ``CloudCorrection``. Common controls are
   ``clouds_per_species=8``, ``refit_every=1``, ``subcharge_count=16``, and
   ``cloud_breathing=False``. Length inputs ending in ``_m`` use metres,
   and times ending in ``_s`` use seconds. See :doc:`../native_pic` and
   :doc:`../pic_correction_convergence` for complete configuration examples,
   source-resolution limits, and the experimental breathing model.

.. autoclass:: core.pic.correction.CloudCorrection
   :no-members:

.. autofunction:: core.pic.simulation.run_pic

Boundary construction and conversion
------------------------------------

.. autoclass:: core.boundary.mesh.AxisymmetricGrid
   :no-members:

.. autoclass:: core.boundary.materials.AlignedWall
   :no-members:

.. autoclass:: core.boundary.materials.DrudeMedium
   :no-members:

.. autoclass:: core.boundary.materials.DrudeWall
   :no-members:

.. autoclass:: core.boundary.incident.BallisticDrive
   :no-members:

.. autoclass:: core.boundary.solver.ScatteredFieldSolver
   :members: step, state, restore

.. autoclass:: core.boundary.observers.BoundaryUnits
   :no-members:

.. autoclass:: core.boundary.observers.BoundarySnapshot
   :members: pic_consumer

External provider contract
--------------------------

.. autoclass:: core.external_fields.ExternalFieldConfig
   :no-members:

.. autoclass:: core.external_field_provider.ExternalFieldSample
   :members: from_si

.. autoclass:: core.external_field_provider.ExternalFieldProvider
   :no-members:

.. autofunction:: core.external_field_provider.provider_force_derivative

.. autoclass:: core.pic.backend.PICBackend
   :no-members:

.. autoclass:: core.pic.grid.PICFields
   :no-members:
