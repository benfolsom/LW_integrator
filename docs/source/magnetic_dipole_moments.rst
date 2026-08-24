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
as a controlled diagnostic.

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
``--radiation-reaction-mode off``, and ``--no-adaptive-timestep`` for a dynamic
run.  The direct CLI and GUI select radiation reaction ``off`` when RFS is
enabled unless the user explicitly supplies a mode; a dynamic mode is rejected.

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

Hard scope guards
-----------------

The first coupled implementation deliberately rejects combinations whose
meaning has not yet been validated:

* RFS runs are limited to ``BUNCH_TO_BUNCH`` with point-charge cross-bunch
  sources.  Conducting and switching image-source modes are not enabled.
* Charge-source RFS requires ``COLD_START`` and explicit history.  Approximate
  back-history is not treated as a complete retarded derivative.
* Dynamic radiation reaction must be off.  ``diagnostic_only`` may record
  read-only radiation diagnostics, but Medina/LAD and other recoil modes are
  rejected because RFS does not supply an RR completion.
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

The legacy diagnostic pair has different limits.  ``bmt_frenkel`` performs
charged or neutral BMT/Larmor transport in prescribed fields.
``static_rest_gradient`` applies
:math:`\nabla(\boldsymbol\mu\mathbin{\cdot}\mathbf B)` from the configured
static gradient and rejects a nonzero impulse at
:math:`|\boldsymbol\beta|>0.01`.  It is useful for sign and unit checks, but it
must not be presented as a relativistic retarded-gradient calculation.

Deferred source physics
-----------------------

The present model describes **response** to prescribed and charge-generated
:math:`F^{\mu\nu}`.  It does not make an intrinsic particle moment a source of
its own retarded electromagnetic field.  Therefore it currently contains no
mutual dipole--dipole force, no hyperfine-type source field, and no intrinsic
dipole radiation or dipole radiation reaction.  Conducting images also remain
charge-only sources.

Those effects require a separately documented covariant magnetization-current
or retarded moving-dipole source model.  The retarded field and radiation of a
moving magnetic dipole are treated, for example, by `Sautbekov
<https://doi.org/10.1016/j.jmmm.2019.04.012>`_
(`arXiv:1806.07089 <https://arxiv.org/abs/1806.07089>`_).  If a validated
dipole-source field is later added to the total non-self
:math:`F^{\mu\nu}`, the existing :math:`qF` response and full RFS
:math:`G[F]` response generate charge--dipole and dipole--dipole interactions.
Thus the source model is a separate field provider, but dipole--dipole response
is **not** a separate force toggle.  Adding an independent textbook pair force
on top of that total-field response would double-count the interaction.  A
static near-field provider, a fully retarded provider, and any dipole
self-radiation/reaction completion must be named and validated separately.

Validation and capture boundary
-------------------------------

The maintained deterministic checks cover signed species data, tensor and
Hodge-dual conventions, the static-rest and neutral limits, covariant
constraint derivatives, the full :math:`G^{\mu\nu}` tensor in vacuum and
current regions, analytic light-cone roots, charge-field gradients, explicit
model-pair validation, and feature-off equivalence.  See :doc:`validation` for
the focused commands.

Electron--proton capture is a classical characterization and sensitivity
study, not a validation of atomic stability.  This first stage can compare
charge-only evolution with RFS spin and gradient response, but it cannot test
dipole--dipole or hyperfine physics while dipole sourcing is absent.  Dynamic
radiation reaction is also guarded off, so it cannot yet decide a proposed
long-time balance between emitted radiation and mutual retarded interactions.
A later balance study must track particle energy, near-field energy, outgoing
radiation, and any self-force without double counting.  No classical
point-particle result should be presented as reproducing the quantum hydrogen
spectrum.

The archived ``TUPAB218.tex`` equations remain a research input rather than the
implemented authority.  The maintained model follows the cited RFS sign,
normalization, neutral-limit, and full-gradient conventions explicitly.
