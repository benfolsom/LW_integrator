Magnetic dipole moments and spin
================================

Intrinsic magnetic-moment dynamics are experimental and disabled by default.
The selected physical model is the signed, minimal
Rafelski--Formanek--Steinmetz (RFS) model:

* ``rfs_minimal_2021`` transports spin with the minimal signed equation used in
  the `2021 RFS charged-particle paper
  <https://doi.org/10.1103/PhysRevA.103.052218>`_ (`arXiv:2103.02594
  <https://arxiv.org/abs/2103.02594>`_).
* ``rfs_full_g`` adds translational magnetic-gradient response through the full
  :math:`G^{\mu\nu}` tensor defined in the `2018 RFS paper
  <https://doi.org/10.1140/epjc/s10052-017-5493-2>`_
  (`arXiv:1712.01825 <https://arxiv.org/abs/1712.01825>`_).

The model responds to prescribed fields and to charge-generated, retarded
Lienard--Wiechert fields.  Charge and magnetic moment are independent particle
properties, so the same equations have a regular neutral-particle limit.  The
older ``bmt_frenkel`` plus ``static_rest_gradient`` pair remains available only
as a controlled diagnostic.  A separate full-retarded point-dipole source can
now add the ordinary non-self field of intrinsic moments.  That source is also
experimental and remains off unless selected explicitly.

Selected RFS equations
----------------------

The RFS papers state their equations in SI.  The production kernel implements
the algebraically equivalent scaled-Gaussian form in the integrator's native
amu--mm--ns--charge units.  Coordinates are
:math:`x^\mu=(ct,x,y,z)`, the metric is ``(+---)``, and
:math:`F^{0i}=-E_i`, so native :math:`\mathbf E` and :math:`\mathbf B` have the
same scale.  The stored dimensionless spin/polarization four-vector
:math:`a^\mu=s^\mu/S` obeys

.. math::

   u^2=c^2, \qquad u\mathbin{\cdot}a=0, \qquad a^2=-1,

for a fully polarized stretched state with

.. math::

   S=I\hbar, \qquad d=\frac{\mu_{\mathrm{signed}}}{cS}.

Keeping the measured sign of :math:`\mu` is essential.  In particular, the
electron and neutron moments are negative relative to spin.  The RFS magnetic
four-potential and its antisymmetric gradient tensor are

.. math::

   \mathcal B_\mu[a]=F^*_{\mu\nu}a^\nu, \qquad
   G^{\mu\nu}[a]=\partial^\mu\mathcal B^\nu[a]
   -\partial^\nu\mathcal B^\mu.

Here :math:`\mathcal B_\mu` is RFS notation and is not the ordinary three-vector
magnetic field.  The coupled equations implemented by the local response
kernel are

.. math::

   m\dot u^\mu=\left({q\over c}F^{\mu\nu}
   +{\mu_{\mathrm{signed}}\over c}G^{\mu\nu}[a]\right)u_\nu,

.. math::

   \dot a^\mu={q\over mc}F^{\mu\nu}a_\nu
   +\left({\mu_{\mathrm{signed}}\over S}-{q\over mc}\right)
   \left(F^{\mu\nu}a_\nu
   -{u^\mu\over c^2}u_\rho F^{\rho\lambda}a_\lambda\right)
   +{\mu_{\mathrm{signed}}\over mc}G^{\mu\nu}[a]a_\nu.

The homogeneous-field spin coefficients are the signed minimal 2021 choice.
For the final gradient term, the maintained model uses the full 2018
:math:`G^{\mu\nu}` tensor.  It agrees with the compact directional-derivative
form used in the 2021 paper in vacuum; inside a current distribution it is an
explicit full-G extension and should not be described as literally 2021
Eq. (11).

This normalized form is exactly the physical-spin form because
:math:`G[s]=S G[a]`.  The moment is converted once from the user-facing J/T
value to native magnetic-moment units; :math:`S` is converted to native action
units.  Source charges enter the native Lienard--Wiechert evaluator directly,
without a charge-to-SI-to-charge normalization round trip.

The dot denotes a proper-time derivative.  In code,
``partial_f[lambda, mu, nu]`` is
:math:`\partial_\lambda F^{\mu\nu}` with
:math:`\partial_0=c^{-1}\partial_t`; the observer spin is held fixed while this
field derivative is taken.  The implementation forms the full tensor instead
of replacing it with the compact source-free Gilbert expression.  That avoids
silently dropping the current-density term when a supplied field is not in
vacuum.

RFS model status
----------------

``minimal`` is part of the model name for a reason.  The `2018 RFS analysis
<https://doi.org/10.1140/epjc/s10052-017-5493-2>`_ describes a family of
covariant gradient corrections to spin transport; the `2021 charged-particle
model <https://doi.org/10.1103/PhysRevA.103.052218>`_ selects the simplest
member rather than establishing uniqueness.  The 2018 analysis also shows that
varying the natural interaction action produces an extra term proportional to
:math:`F^{*\mu\nu}\dot s_\nu` and then higher field orders.  It does not provide
a closed, all-orders action or Hamiltonian for the coupled system.

The maintained model should therefore be described as a covariant,
linear-in-field/gradient response model, not as a finished action-based theory.
Its covariant constraints and limiting cases are strong validation targets,
but they do not remove this theory boundary.

Charge-field and gradient evaluation
------------------------------------

The charge-generated RFS field is evaluated independently of observer charge,
so a neutron is not lost through the ordinary Lorentz-force ``q=0`` shortcut.
The integration seam currently applies this evaluator to the opposing bunch
only; it does not yet supply same-bunch RFS response.

For each observer event the dedicated evaluator:

1. interpolates each stored point-charge worldline with position, velocity,
   and acceleration continuous at the trajectory knots;
2. solves the light-cone equation against that interpolated worldline;
3. evaluates both the velocity and acceleration terms of the native Gaussian
   Lienard--Wiechert field; and
4. forms a centred spacetime finite difference of :math:`F^{\mu\nu}`.

Every one of the eight displaced stencil events re-solves every source light
cone.  The time stencil displaces :math:`ct`, not merely the stored source
sample.  Consequently the numerical derivative includes the variation of
retarded time.  Differencing a field while freezing a previously selected
retarded source state would omit that chain rule and is not used by RFS.

Prescribed native :math:`\mathbf E` and :math:`\mathbf B` pass through without
renormalization and are added to the charge field.  The current schema can also
express a static spatial magnetic-field gradient in T/m; that boundary is
converted to native field per mm before the gradient is applied to the native
position.  Unconfigured electric and time derivatives are zero.

The charge-canonical state definition derived in ``main.tex`` remains
authoritative.  Under ``INERTIAL_PREHISTORY``, one exact charge provider returns
:math:`A^\mu`, :math:`F^{\mu\nu}`, :math:`\partial_\lambda A^\nu`, and
:math:`\partial_\lambda F^{\mu\nu}`.  The step advances the mechanical
:math:`qF` response and, after both bunch endpoints are available, stores
:math:`P_{n+1}=p_{n+1}+qA_{n+1}/c`.  The same field supplies the charge part of
the RFS total field.  ``COLD_START`` retains the established canonical charge
kernel plus a separate exact RFS field/gradient sample.  In both modes RFS adds only
:math:`(\mu/c)G^{\mu\nu}[a]u_\nu`, including its temporal component.  This
avoids counting the Lorentz force twice, preserves the feature-off baseline,
and does not redefine canonical momentum by appending
:math:`d\mathcal B_\mu`.

Configuration and operating modes
---------------------------------

The GUI places the common switches with the particle controls.  The direct CLI
offers matching enable, species, spin-direction, spin-transport, and
gradient-force switches.  Saved testbed configurations use a nested block:

.. code-block:: json

   {
     "magnetic_dipole": {
       "enabled": true,
       "spin_precession_enabled": true,
       "stern_gerlach_force_enabled": true,
       "spin_model": "rfs_minimal_2021",
       "stern_gerlach_model": "rfs_full_g",
       "exact_retarded_backend": "python",
       "source": {
         "model": "off",
         "minimum_separation_mm": 2e-9,
         "relative_stencil_step": 1e-3,
         "minimum_stencil_step_mm": 1e-15,
         "root_tolerance_mm": 1e-21,
         "max_root_iterations": 96
       },
       "rider": {
         "species": "electron",
         "magnetic_moment_j_per_t": null,
         "spin_quantum_number": null,
         "rest_spin": [0.0, 0.0, 1.0],
         "polarization": 1.0
       },
       "driver": {
         "species": "proton",
         "magnetic_moment_j_per_t": null,
         "spin_quantum_number": null,
         "rest_spin": [0.0, 0.0, 1.0],
         "polarization": 1.0
       }
     }
   }

The switches give three intended user-facing states:

.. list-table:: RFS operating modes
   :header-rows: 1
   :widths: 22 18 22 38

   * - State
     - ``enabled``
     - ``spin_precession_enabled``
     - ``stern_gerlach_force_enabled``
   * - Off
     - ``false``
     - ignored
     - ignored
   * - Spin transport only
     - ``true``
     - ``true``
     - ``false``
   * - Fully coupled RFS
     - ``true``
     - ``true``
     - ``true``

A force run with spin transport disabled is possible as a frozen-spin
diagnostic, but it is not a complete coupled RFS evolution.  On the CLI,
``--magnetic-dipoles`` selects spin-only RFS and adding ``--stern-gerlach``
selects the fully coupled mode.  The RFS safety guards currently also require
``--simulation-type bunch-to-bunch``, either ``--startup-mode cold-start`` or
``--startup-mode inertial-prehistory``,
``--radiation-reaction-mode off``, and ``--no-adaptive-timestep`` for a pure
RFS run.  ``--radiation-reaction-mode medina_lad`` instead selects the explicit
charge-radiation-only RFS/Medina hybrid described below.  Other dynamic recoil
modes are rejected.  The direct CLI and GUI still select radiation reaction
``off`` when RFS is enabled unless the user explicitly requests the hybrid.

Intrinsic source fields are a separate selection.  The GUI offers ``Off`` and
``Full retarded point (experimental)``.  The matching direct CLI choice is
``--dipole-source off`` or ``--dipole-source full-retarded-point``.  The latter
is active only when magnetic-dipole dynamics are enabled.  The advanced
``--dipole-source-cutoff-mm`` option sets the strict minimum separation abort
boundary.  It does not soften the point field.

The exact retarded charge and dipole providers share five explicit backend
choices.  ``python`` is the default and remains the reference on every
platform.  The canonical JSON setting is
``magnetic_dipole.exact_retarded_backend`` and the direct CLI option is
``--exact-retarded-backend``.  The former
``magnetic_dipole.source.backend`` key is accepted only as an input
compatibility alias when the canonical key is absent or has the same value;
conflicting values are rejected, and saved configurations emit only the
canonical key.

``numba_roots_exact_serial`` is a cross-platform CPU opt-in.  It compiles only
the independent light-cone root searches.  The final quintic worldline sample,
light-cone residual, charge or Hertz event construction, source accumulation,
and finite-difference assembly retain the Python reference arithmetic and
order, giving complete-provider parity in the maintained tests.

``numba_full_strict_serial`` is the faster, tolerance-validated opt-in.  It
also compiles the final worldline sample, spin interpolation, moment boost,
Hodge dual, per-source Hertz tensor, and corresponding per-source charge-field
event work with strict binary64 arithmetic and ``fastmath=False``.  It is not
a bitwise-parity mode: an event-level change of one binary64 ULP can be
magnified by finite differences.  Charge and dipole stencil centers remain on
the Python reference path.  Source accumulation and finite-difference
construction also remain in Python and preserve the reference request and
reduction order.  Acceptance requires matching run status, tightly matching
physical trajectory arrays, and cumulative projection-energy disagreement
below ``0.025 meV``.  Raw derivative differences must be reported with their
field scale and propagated force/spin effect rather than interpreted alone.

``numba_analytic_charge_dipole_response_serial`` is the potential-first
analytical opt-in.  On a smooth source-history segment it solves one retarded
light cone and propagates a third-order four-coordinate Taylor jet through the
implicit light-cone equation and covariant Hertz tensor.  The first, second,
and third Hertz derivatives directly produce :math:`A^\mu`,
:math:`F^{\mu\nu}`, and :math:`\partial_\lambda F^{\mu\nu}` without displaced
observer events.  It supports relativistic motion; no slow-speed dipole-force
approximation is used.  Because the current spin history is only
:math:`C^1`, the backend strictly falls back to the full finite-difference
oracle at segment boundaries, on the mutable final spin segment, for a
one-knot history, and near a particle-loss wavefront.  Those fallbacks are
counted in run diagnostics.  Accepted-endpoint reconstruction uses the same
analytical Hertz response (or the same declared full-strict fallback) so the
stored canonical offset agrees with the potential used to decode mechanical
momentum at the next step start.  The smaller nine-event endpoint stencil is
retained by the finite-difference backends.

The analytical response is accepted against adjacent-stencil Richardson
limits, grouped physical trajectory variables, explicit radiation/projection
energy ledgers, and independent timestep refinement.  Individual Cartesian
components and the saved ``local_magnetic_field_*`` visualization arrays are
reported but do not override their complete vector or force-path response.
The coefficient audit also records response-level zeros and Bianchi
redundancies.  The exact-endpoint production path uses that audit to retain
only 144 of 210 Hertz-jet coefficients and emits the 34 values actually
consumed: four components of :math:`A^\mu`, six packed independent components
of :math:`F^{\mu\nu}`, and 24 packed components of
:math:`\partial_\lambda F^{\mu\nu}`.  It does not materialize
:math:`\partial A`, the full field tensor, or the full field-gradient tensor.
The dense analytical jet remains the comparison oracle, and the finite-
difference fallback is unchanged.
Four of the 24 packed field-derivative outputs are constrained by the
homogeneous Maxwell/Bianchi identities (the response map has rank 20).  They
are retained for now because the direct RFS contraction consumes the existing
``(4, 6)`` layout; removing them would require an explicit reconstruction or a
still narrower force/spin contraction kernel and therefore a separate
roundoff audit.

The finite-difference Numba kernels consume displaced events in the oracle's lazy first-use
order, preserving which history or singularity error is raised first.  They
are serial: neither selects a worker count nor consumes multiple cores, and
there is no automatic or operating-system-specific dispatch.  Explicit
selection raises a capability error if Numba is unavailable or initial
compilation fails; it never silently changes the recorded backend.  The
selection covers the exact charge one-event field, exact charge nine-event
gradient, dipole accepted-endpoint potential, and full dipole gradient.  The
finite-difference backends use the nine-event dipole endpoint provider.  The
finite-difference centers remain Python-reference evaluations; the analytical
mode instead uses one center root and one consistent Hertz response on smooth
segments.

``metal_certified_full_strict`` is an explicit Apple-silicon accelerator
option for large dipole batches.  Metal receives float32 history and observer
data and proposes light-cone segment indices only.  Each proposal must pass an
original-float64 two-endpoint check backed by a strict timelike-chord proof.
The strict CPU kernel then computes the float64 root and all field physics;
ambiguous proposals and runtime GPU failures use the exact CPU search.  The
raw proposal/root crossover begins near 1,024 observer events, but float64
certification and complete strict field work move the production crossover to
about 8,192 events per uploaded source-history batch. Smaller calls stay on
``numba_full_strict_serial`` without dispatching Metal.  There is no automatic
Metal selection, and float32 Hertz tensors or field gradients are explicitly
outside this backend because they do not meet the numerical contract.

``null`` selects the cited species value.  A custom species must provide both a
signed moment in J/T and its spin quantum number.  ``rest_spin`` is normalized
and interpreted as a rest-frame direction expressed in the lab coordinate
axes.  For the current RFS slice, ``polarization`` must be exactly zero or one.
Partial polarization must eventually be represented by a weighted ensemble of
unit-spin orientations, not by shrinking one particle's invariant spin.

Species presets and signs
-------------------------

``core.species`` is the single immutable registry used by the physics and user
interfaces.  It includes electron, positron, proton, antiproton, neutron,
deuteron, triton, helion (the helium-3 nucleus), and alpha particle.  Free-
particle masses and moments use the `2022 CODATA adjustment
<https://doi.org/10.1103/RevModPhys.97.025002>`_.  The antiproton preset also
records the direct `BASE measurement
<https://doi.org/10.1038/nature24048>`_.

A named magnetic preset must match the simulated particle's physical mass and
observer charge.  The run is rejected if, for example, an electron moment is
paired with a proton-mass particle.  Antiparticles are separate presets rather
than charge-sign shortcuts.  The H- entry is intentionally unsupported: its
bound-state moment is not inferred by adding constituent free-particle
moments.  Use ``species: "custom"`` with an explicit, documented moment and
spin when such a model is intentional.

Spin state and diagnostics
--------------------------

The trajectory stores dimensionless rest-frame spin-direction components
``spin_x``, ``spin_y``, and ``spin_z``.  For velocity
:math:`\boldsymbol\beta`, the corresponding dimensionless polarization
four-vector is

.. math::

   a^0=\gamma\,\boldsymbol\beta\mathbin{\cdot}\boldsymbol\zeta,

.. math::

   \mathbf a=\boldsymbol\zeta
   +{\gamma^2\over\gamma+1}
   (\boldsymbol\beta\mathbin{\cdot}\boldsymbol\zeta)\boldsymbol\beta.

The RFS kernel receives :math:`a^\mu` directly.  The accepted spin update is
projected back onto :math:`u\mathbin{\cdot}a=0` and its invariant magnitude is
restored to control numerical drift.

``local_magnetic_field_x_t``, ``local_magnetic_field_y_t``, and
``local_magnetic_field_z_t`` are state-aligned diagnostics.  For RFS they
contain the prescribed field plus the available charge-generated field
re-evaluated at the saved observer event.  A COLD_START event without enough
explicit source history is not extrapolated; the diagnostic remains zero and
does not yet export a separate readiness flag.  In particular, a timestep
longer than the source--observer light delay can leave an end-of-step field
diagnostic unavailable even when the accepted start-of-step RFS force used a
nonzero field.  ``INERTIAL_PREHISTORY`` instead requires every initial exact
stencil to be ready before the run begins.  Treat these arrays as visualization
aids, not as proof that a field was absent.

Backend comparisons allow these saved visualization diagnostics a named
absolute tolerance of ``1e-12 T``; ordinary physical state arrays retain the
``2e-12`` relative tolerance.  This diagnostic allowance does not relax the
force path: force-center fields and dynamical state are checked separately,
and ``local_magnetic_field_*`` must not be used as force-path validation.

The current saved local-field diagnostic does not yet add the intrinsic
dipole-source field.  That field is present in the accepted canonical and RFS
dynamics when its provider is enabled; use provider-readiness data and direct
field diagnostics, rather than the legacy ``local_magnetic_field_*`` arrays,
for source-field validation.

Hard scope guards
-----------------

The first coupled implementation deliberately rejects combinations whose
meaning has not yet been validated:

* RFS runs are limited to ``BUNCH_TO_BUNCH`` with point-charge cross-bunch
  sources.  Conducting and switching image-source modes are not enabled.
* Charge-source RFS requires ``COLD_START`` or ``INERTIAL_PREHISTORY`` and
  explicit history.  Approximate back-history is not treated as a complete
  retarded derivative.  Inertial startup is limited to exact-field
  ``BUNCH_TO_BUNCH`` RFS/retarded-dipole runs and rejects driver trains.
* Dynamic recoil is limited to ``medina_lad``.  It is an explicitly named
  charge-only hybrid, not a complete RFS radiation-reaction theory.  ``off``
  and read-only ``diagnostic_only`` also remain available; other recoil modes
  are rejected.
* Same-bunch RFS response is absent, so ``space_charge`` must be disabled.
* Nonzero macroparticle smearing is unsupported.  Each displaced subcharge would
  require its own light-cone solve before a smeared source could be supported.
* Beamline visibility boundaries are not applied to a finite-difference
  stencil, so beamline geometry must be disabled for charge-source RFS.
* Adaptive timestep substeps are not yet supported by the exact source-history
  evaluator.  Exact inertial endpoint reconstruction also requires
  ``fixed_geometry`` self-consistency.
* RFS polarization is restricted to zero or one.
* Pseudo-grid mode remains incompatible with spin-aware particle
  reconstruction.
* The full-retarded point source is initially limited to exactly one physical
  particle in each nonempty bunch, with ``macro_population`` equal to one.  It
  does not yet support a driver train or a cavity-exit synthetic coasting tail.

The legacy diagnostic pair has different limits.  ``bmt_frenkel`` performs
charged or neutral BMT/Larmor transport in prescribed fields.
``static_rest_gradient`` applies
:math:`\nabla(\boldsymbol\mu\mathbin{\cdot}\mathbf B)` from the configured
static gradient and rejects a nonzero impulse at
:math:`|\boldsymbol\beta|>0.01`.  It is useful for sign and unit checks, but it
must not be presented as a relativistic retarded-gradient calculation.

Retarded intrinsic-dipole source
--------------------------------

``covariant_retarded_point`` is the first ordinary Maxwell field provider for
an intrinsic moment.  It uses a conserved antisymmetric moment tensor and a
retarded Hertz potential to construct the ordinary four-potential, field
tensor, and full spacetime field gradient.  Every nested finite-difference
event solves its own source light cone.  The provider therefore includes the
near, induction, and radiation zones without freezing source acceleration,
spin evolution, or retarded time.

Source creation is independent of electric charge, so a neutral magnetic
particle remains a field source.  Stable particle identities exclude the
observer's own source.  The ordinary charge response consumes the returned
field as mechanical :math:`qF`; a nine-event endpoint provider supplies the
ordinary potential used to reconstruct accepted canonical momentum.  The RFS
response consumes the returned field and field gradient exactly once.  The
resulting total non-self field supplies charge--dipole and dipole--dipole force
and torque without adding another pair-force law.  Adding a textbook dipole
pair force on top would double-count the interaction.

For exact inertial startup, the charge and dipole providers use the same
explicit source histories and the same light-cone-root contract.  Within each
provider, :math:`A`, :math:`F`, and their spacetime derivatives at one stencil
event are evaluated from the same retarded event; no force component freezes a
previously selected source state.  Once all initial stencils preflight, the
integrator performs one mechanical-to-canonical initialization,

.. math::

   P^\mu(0)=p^\mu(0)+{q\over c}
   \left(A_q^\mu(0)+A_{\mathrm{dip}}^\mu(0)\right),

without changing the initialized velocity.  The eight-knot synthetic coasting
prefix is hidden from output and never supplies a Medina force sample.  If an
exact light cone later falls outside retained history, the run raises rather
than degrading to zero field.

At every later accepted step, both provisional endpoints are appended before
either canonical state is changed.  The endpoint potentials are then sampled
from those symmetric histories and :math:`q(A_{n+1}-A_n)/c` replaces the saved
start offset without changing mechanical momentum, position, spin, or Medina
work.  This is bookkeeping, not an additional force.

The exact-retarded translation update is selected independently of the field
backend.  ``first_order_endpoint`` remains the default.  The experimental
``second_order_start_taylor_endpoint`` option expands the ordinary Lorentz
four-force at the accepted start event,

.. math::

   K^\mu={q\over c}F^{\mu\nu}u_\nu,
   \qquad
   \dot K^\mu={q\over c}\left[
   u^\lambda(\partial_\lambda F^{\mu\nu})u_\nu
   +F^{\mu\nu}a_\nu\right],

and applies

.. math::

   \Delta p^\mu=hK^\mu+{h^2\over2}\dot K^\mu.

The force, field derivative, velocity, and acceleration in this expression all
belong to the same accepted start phase-space event.  In particular, a
self-consistency trial endpoint velocity is not substituted into the
start-event contraction.  Position and coordinate time use the matching
start/end trapezoidal update, and canonical momentum is still recomposed from
the accepted endpoint potential after both source endpoints are published.
The option differentiates the ordinary charge and dipole-source Lorentz
response.  It does not yet supply the higher field derivatives needed to make
the RFS moment force or Medina reaction intrinsically second order, so full
coupled runs still require timestep refinement and projection-energy audits.

This first implementation remains a full-retarded finite-difference oracle.
The shared optional exact-retarded backends accelerate charge and dipole
light-cone work without changing Python reference-order source or stencil
assembly.  ``relative_stencil_step``,
``minimum_stencil_step_mm``, ``root_tolerance_mm``, and
``max_root_iterations`` are advanced convergence controls preserved by the
CLI, GUI, and testbed JSON round trip.  A validation study should repeat the
calculation with half and twice the relative stencil step.  The normal GUI
exposes only the source model and ``minimum_separation_mm``.

The point source is singular.  ``minimum_separation_mm`` is a strict abort
boundary, not a particle radius, contact interaction, field clamp, or
softening length.  The default is 2 pm.  Crossing the boundary means that the
selected point model has left its declared domain; it does not authorize the
integrator to continue with a finite force.

The retarded construction and its static, oscillating, and moving limits are
consistent with the Green-function treatments by `Sautbekov
<https://doi.org/10.1016/j.jmmm.2019.04.012>`_
(`arXiv:1806.07089 <https://arxiv.org/abs/1806.07089>`_) and `Heras
<https://doi.org/10.1103/PhysRevE.58.5047>`_.  The implementation still has no
dipole self-field, contact term, finite-size source, conducting dipole image,
or dipole radiation-reaction completion.  It can emit an outgoing retarded
field that acts on the other particle, but it does not yet apply the associated
self-recoil to its source.

Charge-only Medina/RFS hybrid
-----------------------------

``radiation_reaction_mode="medina_lad"`` applies the corrected Medina
charge-self-force to mechanical momentum.  The production derivative retains
the full :math:`d(\gamma\mathbf F_{\rm ext})/dt`, using accepted midpoint
force samples; an unprimed first sample records far radiation but applies no
incomplete impulse.  If the numerical impulse guard caps the force, spin sees
the same post-cap force that translation received.

.. math::

   \mathbf F_{\rm RAD}={2q^2\over3mc^3}
   \left[
   {d\over dt}\left(\gamma\mathbf F_{\rm ext}\right)
   -{\gamma^3\over c^2}
   \left(\mathbf F_{\rm ext}\mathbin{\cdot}\mathbf a\right)\mathbf v
   \right].

The derivative is explicitly

.. math::

   {d\over dt}\left(\gamma\mathbf F_{\rm ext}\right)
   =\gamma{d\mathbf F_{\rm ext}\over dt}
   +{d\gamma\over dt}\mathbf F_{\rm ext}.

For that applied charge-radiation four-acceleration
:math:`A_{\rm RR}^\mu`, the normalized RFS spin equation receives

.. math::

   \delta\dot a^\mu=-{u^\mu\over c^2}
   \left(A_{\rm RR}\mathbin{\cdot}a\right).

This Fermi--Walker term is evaluated at both spin midpoint stages.  Together
with the applied translation it preserves :math:`u\mathbin{\cdot}a=0`
instantaneously instead of relying only on the final numerical projection.
The hybrid covers the charge ``q^2`` self-radiation sector.  It does not cover
charge--dipole ``q mu`` interference recoil, intrinsic-dipole ``mu^2`` recoil,
or self torque.  A capture-validation run must also reject every step whose
``medina_impulse_capped`` diagnostic is true.

Validation and capture boundary
-------------------------------

The maintained deterministic checks cover signed species data, tensor and
Hodge-dual conventions, the static-rest and neutral limits, covariant
constraint derivatives, the full :math:`G^{\mu\nu}` tensor in vacuum and
current regions, analytic light-cone roots, charge-field gradients, retarded
dipole static and uniform-motion limits, source self-exclusion, stencil
convergence, explicit model-pair validation, and feature-off equivalence.  See
:doc:`validation` for the focused commands.

Electron--proton capture is a classical characterization and sensitivity
study, not a validation of atomic stability.  The source option can compare
charge-only evolution with mutual retarded charge--dipole and dipole--dipole
response, including the outgoing field that reaches the other particle.  The
Medina hybrid adds charge-only recoil, but dipole self-recoil and ``q mu``
interference recoil remain absent, so it cannot yet close a total long-time
radiation balance.  A later balance study must track particle energy,
near-field energy, outgoing radiation, and every self-force without double
counting.  No classical point-particle result should be presented as
reproducing the quantum hydrogen spectrum.

The immediate capture target is a first-pass question: does an initially
unbound flyby leave the encounter with negative relative mechanical energy,
with no capped Medina step?  Because ``INERTIAL_PREHISTORY`` declares inertial
motion before a finite active start, that result must converge as the starting
separation is moved outward.  Following a weakly bound trajectory through
apoapsis and a return encounter is a separate long-time test; it needs a
history-preserving multirate strategy rather than carrying the periapsis
timestep across the entire orbit.  Such a return test will still lack a closed
radiation balance until the missing charge--dipole :math:`q\mu` and intrinsic
dipole :math:`\mu^2` self-recoil sectors are implemented.

The archived ``TUPAB218.tex`` equations remain a research input rather than the
implemented authority.  The maintained model follows the cited RFS sign,
normalization, neutral-limit, and full-gradient conventions explicitly.
