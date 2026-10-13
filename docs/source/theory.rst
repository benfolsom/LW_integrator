Theory primer
=============

This page distills the physics encoded in the LW Integrator and mirrors the
notation used in the internal design notes that accompany the project.  It connects the
covariant Liénard–Wiechert formalism to the concrete data structures exposed in
``core/trajectory_integrator.py`` and the validation studies under
``examples/validation``.

The main charge integrator uses a native amu--millimetre--nanosecond unit
system derived from the historical Gaussian-unit formulation.  The production
RFS kernel and its retarded charge-field gradient now remain in that native
scaled-Gaussian system.  User-facing J/T moments and T/m gradients cross an
explicit boundary once; configured native electric and magnetic fields pass
through unchanged.


Retarded fields
---------------

The solver models every source particle as a point charge whose fields are
sampled at the observer's *retarded* time.  Starting from Jackson's form of the
Liénard–Wiechert fields, the magnetic field is obtained from the electric field
via a cross product, while the electric field splits into a velocity term and an
acceleration term:

.. math::

   \mathbf{B} = \bigl[\mathbf{n} \times \mathbf{E}\bigr]_{\text{ret}},

.. math::

   \mathbf{E} = e\left[\frac{\mathbf{n} - \boldsymbol{\beta}}{\gamma^{2}\,\kappa^{3} R^{2}}\right]_{\text{ret}}
   + \frac{e}{c} \left[ \frac{\mathbf{n} \times \bigl((\mathbf{n} - \boldsymbol{\beta}) \times \dot{\boldsymbol{\beta}}\bigr)}{\kappa^{3} R} \right]_{\text{ret}},

where :math:`\kappa = 1 - \boldsymbol{\beta} \cdot \mathbf{n}`, :math:`R` is the
retarded source–observer separation, :math:`\boldsymbol{\beta} = \mathbf{v}/c`,
and :math:`\gamma = (1-\beta^{2})^{-1/2}`.  Each quantity is evaluated at the
retarded time :math:`t - R/c`.  The implementation samples these terms inside
:func:`core.integration_runner.retarded_integrator`, looping over all
available source trajectories.

A key limit for the benchmark problems is the near head-on configuration where
:math:`\mathbf{n}` aligns with :math:`\boldsymbol{\beta}`.  Neglecting transverse
components yields

.. math::
   :label: eq-headon-limit

   \mathbf{E}_{\parallel} \approx e\,\frac{1+\beta}{(1-\beta) R^{2}}\,\mathbf{n},

The growth as $\beta\rightarrow1$ is at fixed retarded separation $R$.
The inverse factor applies when the directions are opposite. A comparison at
fixed simultaneous separation requires converting the retarded distance first;
it does not inherit this growth statement.

Covariant potentials
--------------------

Instead of tracking fields directly, the integrator evolves the covariant
potential :math:`A^{\alpha}` for each source trajectory.  Using proper time
:math:`\tau` as the integration variable, the retarded potential reads

.. math::
   :label: eq-retarded-potential

   A^{\alpha}(x) = \left.\frac{e\, V^{\alpha}(\tau)}{V(\tau) \cdot [x - r(\tau)]}\right|_{\tau = \tau_{0}},

with :math:`V^{\alpha} = \{c\gamma, \gamma \mathbf{u}\}` the four-velocity,
:math:`r^{\alpha}(\tau)` the source worldline, and :math:`\tau_{0}` obtained from
light-cone constraint :math:`[x - r(\tau_{0})]^{2} = 0`.  The denominator reduces
to :math:`\gamma c R \kappa`, linking the potential back to the geometry used in
:eq:`eq-headon-limit`.

.. _equations_of_motion:

Conjugate momentum and equations of motion
------------------------------------------

Each observer particle carries a conjugate four-momentum

.. math::
   :label: eq-conjugate-momentum

   \mathcal{P}^{\alpha} = m V^{\alpha} + \frac{e}{c} A^{\alpha},

where :math:`m` and :math:`e` are the observer mass and charge.  Differentiating
:math:`\mathcal{P}^{\alpha}` with respect to proper time leads to the mixed-field
force law used inside the stepping kernel:

.. math::
   :label: eq-eom-momentum

   \frac{d\mathcal{P}^{\alpha}}{d\tau} = \frac{e}{c} V_{\beta} \, \partial^{\alpha} A^{\beta}.

The historical potential equation motivates the canonical representation.
The current momentum update is implemented by
:func:`core.equations.retarded_equations_of_motion`, re-exported by the facade.
The corrected path updates mechanical momentum, reconstructs on-shell energy,
and recomposes canonical momentum with the applicable potentials.

Position updates follow directly from the Hamiltonian identity

.. math::
   :label: eq-eom-position

   \frac{d x^{\alpha}}{d\tau} = \frac{1}{m}\left( \mathcal{P}^{\alpha} - \frac{e}{c} A^{\alpha} \right),

which the solver evaluates after each momentum update to keep particle states in
sync. Proper-time stepping alone does not exclude runaway behavior; the force
model, history coverage, and trajectory convergence require separate checks.

Intrinsic magnetic-moment response
----------------------------------

The experimental magnetic-moment path preserves the charge-canonical momentum
above.  Under exact ``INERTIAL_PREHISTORY`` evolution, the integrator advances
the gauge-invariant mechanical Lorentz response and the selected
Rafelski--Formanek--Steinmetz dipole response together.  Once both provisional
bunch endpoints are present in the retarded histories, it reconstructs

.. math::

   P^\mu_{n+1}=p^\mu_{n+1}+{q\over c}A^\mu(x_{n+1}).

This explicit endpoint representation is equivalent to the canonical
equation at the continuum level while avoiding an implicit pair solve for
:math:`A(x_{n+1})`.  The canonical potential-derivative contraction remains a
tested convention oracle.  The RFS contribution is only the dipole four-force

.. math::

   \left.\frac{dp^\mu}{d\tau}\right|_{\mathrm{dipole}}
   ={\mu_{\mathrm{signed}}\over c}G^{\mu\nu}[a]u_\nu,
   \qquad
   G^{\mu\nu}[a]=\partial^\mu(F^{*\nu\rho}a_\rho)
   -\partial^\nu(F^{*\mu\rho}a_\rho),

where :math:`a^\mu=s^\mu/(I\hbar)` is the dimensionless spin four-vector.  In
native Gaussian units the full Lorentz-plus-dipole equation is

.. math::

   m\dot u^\mu=\left({q\over c}F^{\mu\nu}
   +{\mu_{\mathrm{signed}}\over c}G^{\mu\nu}[a]\right)u_\nu.

The field includes prescribed fields and observer-charge-independent,
cross-bunch point-charge Lienard--Wiechert fields.  Every centred
spacetime-gradient stencil event performs a new light-cone solve, so the
derivative includes retarded-time variation.  Under
``INERTIAL_PREHISTORY``, the exact charge provider supplies
:math:`A^\mu`, :math:`F^{\mu\nu}`, :math:`\partial_\lambda A^\nu`, and
:math:`\partial_\lambda F^{\mu\nu}` from the same retarded event at each
stencil point.  Force evaluation uses :math:`F`; accepted canonical output uses
the potential evaluated at the accepted endpoint.  ``COLD_START`` retains the
established charge-force path and a separate exact RFS field/gradient sample.
Adding only the native
:math:`(\mu/c)G[a]u` term avoids a second Lorentz force and preserves the
feature-off baseline.  The signed minimal 2021 coefficients advance spin at
the same time.  Its gradient term uses the full 2018 :math:`G` tensor: this
matches the compact 2021 form in vacuum and is an explicit extension, rather
than literally 2021 Eq. (11), in a current region.

This is an experimental covariant response model with strict scope guards, not
a closed all-orders action theory.  The optional
``covariant_retarded_point`` provider adds the ordinary non-self field of each
intrinsic moment.  The existing :math:`qF` and full :math:`G[F]` response then
produce charge--dipole and dipole--dipole coupling without a second pair-force
law.  The provider includes outgoing dipole radiation, but not intrinsic
dipole self-recoil or charge--dipole radiation-interference recoil.  The
equations, configuration modes, primary references, and current validation
boundary are given in :doc:`magnetic_dipole_moments`.

Relativistic position updates in proper time
--------------------------------------------

The LW runner takes $h=\Delta\tau$ in nanoseconds. For a short step,
$\Delta t\simeq\gamma h$ and $\Delta\mathbf x\simeq h\mathbf p/m$, where
$\mathbf p=\mathbf P-q\mathbf A/c$ is mechanical spatial momentum.
Coordinate velocity is $\boldsymbol\beta=\mathbf p/(\gamma m c)$; dividing
proper-time displacement by $ch$ alone would miss the factor $\gamma$.

The second-order exact update uses the average of start and endpoint mechanical
momenta in its displacement. On the corrected path, gamma and beta come from
the same on-shell mechanical momentum. Native PIC is a separate solver that
steps in lab time; its timestep must not be substituted for this proper step.

Self-consistency iterations
---------------------------

The default two fixed-geometry force trials use a one-way mass-shell check
from the second trial onward. Force-updated mechanical spatial momentum sets
on-shell gamma and beta in every radiation mode. There is no independent
energy-versus-velocity gamma convergence criterion, and the check does not
measure successive-force differences.

Keep gamma reconciliation disabled and omit mass-shell relaxation, which is
unused on the corrected path. Changing iteration count cannot replace
trajectory, timestep, or conservation checks. See :doc:`self_consistency` for
current defaults, failure behavior, and chrono options.

Radiation pressure and reaction
-------------------------------

The validation notebooks explore scenarios where residual fields act on a test
particle once a conducting surface or driving bunch is withdrawn.  Two secondary
forces are monitored to confirm that their contribution is negligible for the
reported configurations:

* **Radiation pressure.**  Using Jackson's scaling, the momentum transfer to an
   observer with area :math:`a_{T}` receiving power :math:`P_{R}` across solid
   angle :math:`\Omega` is :math:`\dot{P}_{\text{RP}} = (P_{R}/c)\,(a_{T}/\Omega R^{2})`.
   For the millimetre-to-micron geometries in this repository, this quantity is
   orders of magnitude smaller than the Lorentz force recovered from
   :eq:`eq-headon-limit`.
* **Radiation reaction.**  Passive Liénard radiation diagnostics are tracked
  separately from any optional self-force.  The provisional
  ``power_matched_damping`` mode removes the computed radiated energy by
  scaling mechanical momentum magnitude; it is an energy-bookkeeping
  approximation, not a LAD model.  The experimental ``medina_lad`` mode applies
  Medina's reduced-order Lorentz–Abraham–Dirac force to mechanical momentum:

  .. math::

     \mathbf{F}_{\text{rad}} = \frac{2}{3}\frac{e^{2}}{m c^{3}}\left[\frac{d}{dt}\left(\gamma\mathbf{F}_{\text{ext}}\right) - \frac{\gamma^{3}}{c^{2}} (\mathbf{F}_{\text{ext}} \cdot \mathbf{a})\, \mathbf{v}\right].

  The complete first term is

  .. math::

     \frac{d}{dt}\left(\gamma\mathbf{F}_{\text{ext}}\right)
     =\gamma\frac{d\mathbf{F}_{\text{ext}}}{dt}
     +\frac{d\gamma}{dt}\mathbf{F}_{\text{ext}}.

  ``SimulationOptions`` and ``IntegratorConfig`` default to ``medina_lad``;
  direct ``retarded_integrator`` calls default to ``off`` and require explicit
  selection. Existing checks cover controlled prescribed-field cases.  Longitudinal acceleration should show the expected
  near-cancellation of the Medina terms, while transverse bending gives the
  synchrotron-style recoil mostly opposite the particle velocity.  Conducting
  boundary cases still require dedicated convergence checks before
  ``medina_lad`` should be treated as physics evidence.

Exact inertial prehistory
-------------------------

The ``INERTIAL_PREHISTORY`` startup mode constructs a finite synthetic history
in which each initialized particle coasts inertially before active time zero.
The maintained implementation uses eight sparse knots.  Uniform motion is
represented exactly between those knots, so their spacing is independent of
the much smaller active integration timestep.

The initial duration is conservatively estimated from the maximum cross-bunch
separation :math:`R_{\max}`, the largest initial speed
:math:`\beta_{\max}`, and the exact-field stencil scale :math:`\delta`:

.. math::

   T_{\mathrm{prefix}}
   =2\,{R_{\max}+3\delta\over c(1-\beta_{\max})}.

The factor two is a safety margin; the three stencil widths cover the nested
dipole derivative.  Before integration, the solver evaluates every initial
charge and enabled dipole potential/field stencil in both bunch directions.
If any displaced event lacks a bracketed light-cone root, it doubles the
duration and repeats the full preflight, for at most eight geometric
attempts.  A missing root after successful startup is an error rather than
a request to suppress the force.

At each stencil event, potential, field, and their derivatives are derived
from the same retarded source event within each provider.  Charge and dipole
providers use the same explicit worldline history and light-cone convention;
they do not rely on a frozen retarded state.  Once the preflight succeeds, the
public time-zero input is interpreted as mechanical momentum and rebased once
to

.. math::

   P^\mu(0)=p^\mu(0)+{q\over c}
   \left(A^\mu_{q}(0)+A^\mu_{\mathrm{dip}}(0)\right).

This initialization changes canonical coordinates only; it leaves the stated
mechanical momentum and velocity unchanged.

The synthetic prefix supplies history only.  It is omitted from normal
trajectory output, and active time zero remains the first reported event.  It
also does not invent a pre-simulation force sample for Medina/LAD radiation
reaction: the force derivative remains unprimed until accepted active-time
force samples exist.  Thus the mode specifies an inertial incoming state, not
an assertion that the preceding interacting trajectory has been solved.
Finite-start studies must move the active starting separation outward and
demonstrate convergence of the reported encounter observable; varying the
eight synthetic knot locations alone cannot test the omitted pre-start
interaction.

``INERTIAL_PREHISTORY`` differs from ``APPROXIMATE_BACK_HISTORY`` because the
former supplies finite, explicit source events to the exact retarded-time
solver.  The latter retains the archived analytic extrapolation and remains a
diagnostic mode. The exact mode supports fixed-step ``BUNCH_TO_BUNCH``
RFS/retarded-dipole runs, fixed-step charge driver trains, and the guarded
adaptive shared-lab-time route. See :doc:`exact_path` for train and gate
restrictions, and :doc:`multirate_return` for adaptive charge bunches.  ``COLD_START`` remains appropriate when the desired model is a genuine
turn-on transient.

COLD_START gating mechanism
---------------------------

``COLD_START`` has no source history before each source's first stored sample.
It therefore admits a source only once light from that sample has reached the
observer. This maintained startup rule models a turn-on approximation. It is not
a proof of a causal interacting history, and it is valid only where the fields
are negligible for long enough to build a source history; a preflight check
rejects other runs (:doc:`startup_history`). Prefer ``INERTIAL_PREHISTORY``.
Exact source evaluation separately solves light cones and requires explicit
history coverage.

Light-cone admission
~~~~~~~~~~~~~~~~~~~~

Let source :math:`j` have first stored sample :math:`(t_{j0}, \mathbf{x}_{j0})`.
At the observer's step-start event :math:`(t, \mathbf{x})` the source is
admitted when

.. math::

   c\,(t - t_{j0}) \;\ge\; \left|\mathbf{x} - \mathbf{x}_{j0}\right| .

Properties:

- **Per source.** Each source is judged on its own sample. A distant source
  never holds back a near one, and a near one never admits a distant one.
- **Stateless and monotone.** The mask depends only on the current event and
  the first sample, with no averaged velocity or direction. For a subluminal
  observer, :math:`c(t - t_{j0}) - |\mathbf{x} - \mathbf{x}_{j0}|` grows at
  least as fast as :math:`c(1 - \beta)`, so an admitted source stays admitted.
  The earlier travel-distance gate could close again after a crossing.
- **The physical statement of the startup rule.** For one head-on source at
  distance :math:`R_0` the observer, at speed :math:`\beta c`, is admitted at
  :math:`t = R_0 / ((1 + \beta) c)`. A source at rest is admitted at
  :math:`t = R_0 / c` for an observer that stays near its start.
- **The same interval as the preflight check.** The per-pair time
  :math:`\tau_c` in :doc:`startup_history` is exactly this admission time for a
  straight-line observer, so the guard and the integrator agree.

Admission is applied to the gathered external samples: a source that is not
admitted contributes no charge to the force, the radiation fields or the
canonical potential ledger. With smearing, each subcharge inherits its parent's
admission. When no source is admitted the retarded solve is skipped. Wall and
image sources use the same rule, so an observer at rest next to an image is
admitted when light from the image's first sample arrives, not at once.

Sources admitted at an event had no retarded point at the preceding event. The
canonical bookkeeping takes their ordinary change of vector potential over that
step from the observer's motion alone, with the source's sampled retarded point
held, so the mechanical energy stays continuous across admission.

Same-bunch space charge has its own startup rule and is not gated by this
mask. The exact path (``INERTIAL_PREHISTORY``) does not use it.

Implementation
~~~~~~~~~~~~~~

The mask is ``_cold_start_admission_mask`` in ``core/equations.py``. It is used
by :func:`core.equations.retarded_equations_of_motion` for the force, by
``core/potential_inclusion.py`` for the sampled canonical ledger, and by
``core/pseudo_grid_potential.py`` for reduced pseudo-grid solves.

For conducting walls, the distance :math:`R` is computed to image charges (not
the wall itself), ensuring correct handling of virtual source positions.

Bridging back to the code
-------------------------

The mathematical relationships above surface in the codebase as follows:

- :class:`core.types.IntegratorConfig` captures the physical
  parameters (:math:`\Delta\tau`, aperture radius, wall position) implied by the
  analytical terms.
- :func:`core.images.generate_conducting_image` and
  :func:`core.images.generate_switching_image` encode the
  boundary conditions assumed when taking the head-on limit to model conducting
  apertures and switching walls.
- The reference notebooks under ``examples/validation/`` document historical
  comparisons and exploratory studies related to :eq:`eq-headon-limit`.
- The notebooks under ``legacy/`` are retained for historical investigations,
  but they are not part of the modern retarded-field workflows.

For deeper derivations and experimental context, see `<https://doi.org/10.1016/j.nima.2024.169988>`_.
