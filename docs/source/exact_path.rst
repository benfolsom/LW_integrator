Exact visibility gates and driver trains
==========================================

The exact endpoint charge route requires ``BUNCH_TO_BUNCH``, an enabled
``rfs_minimal_2021`` magnetic response configuration, and
``startup_mode: "INERTIAL_PREHISTORY"``. A charge-only study can select zero
intrinsic moments while keeping that route enabled. Cold start does not
select the same endpoint charge translation. See
:doc:`magnetic_dipole_moments` for backend choices and response limits, and
:doc:`exact_macroparticle_sources` for persistent finite charge clouds.

Updates, same-bunch fields, and strict helpers
------------------------------------------------

``magnetic_dipole.exact_retarded_update`` defaults to
``first_order_endpoint``. The experimental
``second_order_start_taylor_endpoint`` uses accepted-start derivatives and is
required by the adaptive shared-lab-time route. Backend selection is independent
of update order. An exact light-cone solve describes the supplied interpolated
history; trajectories still need timestep and source-history convergence.

Exact same-bunch charge fields exclude the observer itself and enter force,
canonical potentials, startup preflight, and endpoint reconstruction. They
require inertial prehistory, retarded evaluation, and zero softening for actual
point-charge pairs. They do not supply same-bunch dipole sources. Close
same-bunch queries can continue the last accepted segment on the Python
provider; this bounded continuation needs timestep checks and is rejected
when gate-aware future history would be needed.

The experimental ``adaptive_pair_return`` route also supports charge bunches
with multiple particles and same-bunch charge fields. All particles reach a
shared lab-time barrier, step doubling checks every particle, and both bunches
commit their midpoint and endpoint together. Intrinsic dipole-source runs
remain limited to one particle per bunch; see :doc:`multirate_return`.

The default exact backend remains ``python``. Explicit serial Numba backends
compile roots, strict field evaluation, or analytic response derivatives;
see :doc:`magnetic_dipole_moments` for their separate contracts.
``core.strict_null_helpers`` compiles null residuals and polynomial sampling
without fast math. Powers and derivative normalization retain reference
rounding. Batched source jets preserve source-reduction order. These internal
helpers do not select a different physical model or enable automatic GPU
dispatch. Bounded warm-run checks preserved each backend's prior output bytes;
this does not make all backends bitwise identical to Python.

Stochastic emission is guarded off on this general exact path. Enabling it
with ``INERTIAL_PREHISTORY`` raises ``NotImplementedError`` until source
histories preserve recoil velocity jumps; see
:doc:`stochastic_quantum_emission`. Smooth prescribed fields are supported
through the separate :doc:`external_fields` provider contract.

Directional visibility
------------------------

Enabled ``beamline_geometry`` now gates exact charge fields, trajectory-based
dipole fields, and canonical potentials at the retarded source event. Each
persistent charge-cloud child has its own light-cone solve and visibility.
The selected pipe is the one whose axis is most aligned with the observer
velocity. Its radial aperture is strict, its axial ends are inclusive, and
an observer at rest has unrestricted visibility. Hidden sources contribute
neither fields nor potentials; missing source history remains an error.

The solver locates the arrival of source boundary light cones and splits
the force step at each gate change, including an entry and exit within one
step. Derivatives are evaluated on smooth branches rather than across the
discontinuous mask. Charge derivatives use analytic jets; dipole differences
use one-sided stencils near a boundary. An always-visible gate preserves
the ungated trajectory bytes in the reported controls.

At a switch, canonical momentum changes by
:math:`\Delta P=(q_{\mathrm{observer}}/c)\Delta A`, while mechanical momentum stays
unchanged. The ``potential_inclusion_delta_A_*`` fields record this potential
offset, which is excluded from force, radiation, and work diagnostics.
Internal split knots and the selected boundary side survive endpoint
reconstruction and supported checkpoints without changing public output
cadence. Medina/LAD restarts its force-derivative history on the new smooth
branch and applies reaction within the remaining interval.

This is an ideal discontinuous gate model. Material screening, screen heat,
and boundary radiation are not supplied by this convention.

The supported gates require ``INERTIAL_PREHISTORY`` and
``light_cone_quintic`` source histories. Further limits are:

* Adaptive gates support one observer per bunch, including its persistent
  cloud sources. Multiple observers need joint event histories.
* Dipole gates require trajectory histories; independent ``causal_c5`` and
  ``causal_local_jet`` histories are unsupported with gates.
* The selected observer pipe must stay fixed within a force step. A detected
  change is rejected.
* Endpoint light cones must lie in supplied history. Gate-aware continuation
  beyond the latest source knot is rejected; large fixed steps can fail here.
* The legacy adaptive substep controller remains unsupported. The separate
  checkpointed shared-lab-time route retains its own guards.

Inertial driver-train prehistory
----------------------------------

Fixed-step exact charge runs now support ``driver_train`` with
``INERTIAL_PREHISTORY``. The train is expanded before constructing history,
so every particle receives its translated anchor and its own velocity.
The prefix is hidden from normal output and extended until initial exact
field queries are bracketed. It assumes a coasting past, rather than
reconstructing an earlier interacting trajectory or priming Medina's force
derivative. Move the active starting separation outward as a convergence
control.

Driver same-bunch fields exclude self and other train bunches, retaining
interactions within each bunch. Real within-bunch point-charge pairs require
retarded evaluation and zero softening. The softening guard is waived only
when there are no such pairs. A pseudo-grid setting is allowed only when it
already falls back to the full direct solve; actual force reduction remains
unsupported on this exact route.

Adaptive or resumable trains, intrinsic dipole-source trains, and persistent
finite train clouds remain unsupported. The reported one-bunch controls are
byte-identical to single-driver runs. A five-bunch uniform-motion field
control agrees with an independently summed boosted-Coulomb field. These
checks establish the tested startup and field behavior, rather than
long-term train accuracy or energy gain.

Evidence
----------

``tests/unit/test_exact_visibility_gates.py`` and
``tests/unit/test_inertial_driver_train.py`` cover the maintained behavior.
The gate report records near-second-order point-charge gate convergence
with the Taylor endpoint update, both with radiation off and with Medina/LAD.
It also records an adaptive charge-cloud crossing at unchanged tolerances.
Dipole finite-difference roundoff grows at very small stencil widths; the
standalone dipole check does not establish trajectory convergence.
These bounded checks do not validate material screening or whole-system
energy closure.
