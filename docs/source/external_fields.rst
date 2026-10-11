Prescribed external fields
============================

``ExternalFieldConfig`` accepts uniform fields and a linear magnetic gradient,
or an opt-in Python ``provider`` for smooth spatial and time-dependent fields.
The provider replaces the configured base fields and gradient. Combining it
with nonzero base fields or hard spatial/time windows raises ``ValueError``;
the provider must implement its own smooth entrances and exits.

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
