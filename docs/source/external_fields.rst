Prescribed external fields
============================

``ExternalFieldConfig`` accepts uniform fields and a linear magnetic gradient,
or an opt-in Python ``provider`` for smooth spatial and time-dependent fields.
The provider replaces the configured base fields and gradient. Combining it
with nonzero base fields or hard spatial/time windows raises ``ValueError``;
the provider must implement its own smooth entrances and exits.

The provider hook is **experimental**. For energy-loss and channeling work use
the exact route below (``INERTIAL_PREHISTORY`` with the second-order endpoint
update); the ordinary update is first order in timestep.

Provider contract
-------------------

The callable receives keyword arguments ``position_mm`` (three coordinates)
and ``time_ns`` (lab time), and returns ``ExternalFieldSample`` from
``core.external_field_provider``. It must be pure: rejected solver trials can
evaluate the same event repeatedly. Potentials must be at least twice
continuously differentiable, including at region boundaries. The solver checks
finite values and array shapes, but does not certify Maxwell consistency or
differentiate an arbitrary callable numerically.

A sample contains scalar ``phi``, three-component ``vector_potential``,
``electric``, and ``magnetic``, plus ``partial_e`` and ``partial_b`` with shape
``(4, 3)``. Derivative rows refer to ``(ct, x, y, z)``, all in millimetres.
Electric and magnetic fields use native force-per-charge units; the magnetic
Lorentz term uses beta crossed with B. Potentials have field-times-length units.
``ExternalFieldSample.from_si`` converts SI values and derivatives whose rows
instead refer to seconds and metres. Returned sample arrays are copied and
made read-only.

The provider supplies analytic field derivatives to Medina/LAD and magnetic
gradients to the supported Stern–Gerlach response. Its potential values are
diagnostics: this hook advances mechanical momentum without adding a new
external canonical gauge offset. Supply the callable through the Python API;
there is no general JSON factory for arbitrary providers. Checkpointing
requires a dataclass provider with serializable fields; plain functions are
unsupported by that fingerprinting path.

Recommended integration route
-------------------------------

For channeling or energy-loss studies, use the exact ``BUNCH_TO_BUNCH`` route
with ``INERTIAL_PREHISTORY`` and
``magnetic_dipole.exact_retarded_update="second_order_start_taylor_endpoint"``.
Retain the exact route's RFS configuration and scope guards, even for
charge-only particles; see :doc:`exact_path` and
:doc:`magnetic_dipole_moments`. This does not enable a standalone exact
prescribed-field-only runner.

The ordinary non-exact provider update samples the start force and is first
order in timestep, including energy errors after leaving a field region.
Analytic Medina/LAD derivatives do not raise that trajectory order. Converge
the timestep and check external work and radiation accounting before making
energy-loss claims. The second-order exact update is opt-in; the release does
not change the default endpoint update.

Provider, unit-conversion, and derivative checks are in
``tests/unit/test_external_fields.py``.

Minimal provider and run
------------------------

This construction follows the uniform-provider parity control in
``tests/unit/test_external_fields.py``. It is a short API demonstration of the
first-order prescribed-field path, not an energy-loss benchmark. The field is
native, the positions are mm, and the step is proper ns.

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np
   from core.external_field_provider import ExternalFieldSample
   from core.external_fields import ExternalFieldConfig
   from core.integration_runner import retarded_integrator
   from core.self_consistency import SelfConsistencyConfig
   from core.types import SimulationType, StartupMode
   from input_output.bunch_initialization import create_bunch_from_energy

   @dataclass(frozen=True)
   class UniformElectric:
       ez: float = 1.0

       def __call__(self, *, position_mm, time_ns):
           electric = np.array([0.0, 0.0, self.ez])
           return ExternalFieldSample(
               phi=-float(electric @ np.asarray(position_mm)),
               vector_potential=np.zeros(3),
               electric=electric,
               magnetic=np.zeros(3),
               partial_e=np.zeros((4, 3)),
               partial_b=np.zeros((4, 3)),
           )

   rider, _ = create_bunch_from_energy(
       kinetic_energy_mev=1.0, mass_amu=0.000548579909, charge_sign=-1.0,
   )
   empty_driver = {key: value[:0].copy() for key, value in rider.items()}
   trajectory, driver, rider_arrays, driver_arrays, timings = retarded_integrator(
       steps=2, h_step=1e-8, wall_z=0.0, aperture_radius=1e9,
       sim_type=SimulationType.BUNCH_TO_BUNCH, init_rider=rider,
       init_driver=empty_driver, mean=0.0, cav_spacing=0.0, z_cutoff=0.0,
       startup_mode=StartupMode.COLD_START,
       self_consistency=SelfConsistencyConfig(),
       radiation_reaction_mode="medina_lad",
       external_field=ExternalFieldConfig(provider=UniformElectric()),
       use_numba=False,
   )

The five results are rider and driver state histories, optional array histories,
and timing records. A dataclass with serializable fields is suitable for the
provider fingerprint; a plain function is not. General JSON configuration
cannot construct arbitrary Python providers. The exact second-order route
requires the additional RFS and source-history setup described above.
