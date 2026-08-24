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

The existing charge-canonical trajectory update derived in ``main.tex`` remains
authoritative for the :math:`qF^{\mu\nu}u_\nu` response.  RFS independently
samples :math:`F` and :math:`\partial F` with the exact light-cone evaluator and
adds only :math:`(\mu/c)G^{\mu\nu}[a]u_\nu`, including its temporal
component.  This
avoids counting the Lorentz force twice, preserves the feature-off baseline,
and does not redefine canonical momentum by appending
:math:`d\mathcal B_\mu`.  It also means that the charge force and RFS response
currently use two numerical sampling paths rather than one unified field
kernel; that seam remains an explicit convergence and consistency target.

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
``--simulation-type bunch-to-bunch``, ``--startup-mode cold-start``,
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
nonzero field.  Treat these arrays as visualization aids, not as proof that a
field was absent.

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
* Charge-source RFS requires ``COLD_START`` and explicit history.  Approximate
  back-history is not treated as a complete retarded derivative.
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
  evaluator.
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
potential and its derivative through the existing canonical equation, while
the RFS response consumes the returned field and field gradient exactly once.
The resulting total non-self field supplies charge--dipole and dipole--dipole
force and torque without adding another pair-force law.  Adding a textbook
dipole pair force on top would double-count the interaction.

This first implementation is a full-retarded finite-difference oracle rather
than a fast production kernel.  ``relative_stencil_step``,
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

The archived ``TUPAB218.tex`` equations remain a research input rather than the
implemented authority.  The maintained model follows the cited RFS sign,
normalization, neutral-limit, and full-gradient conventions explicitly.
