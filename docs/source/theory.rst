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
:func:`core.trajectory_integrator.retarded_integrator`, looping over all
available source trajectories.

A key limit for the benchmark problems is the near head-on configuration where
:math:`\mathbf{n}` aligns with :math:`\boldsymbol{\beta}`.  Neglecting transverse
components yields

.. math::
   :label: eq-headon-limit

   \mathbf{E}_{\parallel} \approx e\,\frac{1-\beta}{(1+\beta) R^{2}}\,\mathbf{n},

which explains the asymptotic growth of the longitudinal field as
:math:`\beta \rightarrow 1`.  This is the regime probed by the aperture-loss
studies and the recoil-reduction scenarios in ``examples/validation``.

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

Expanding :math:`\partial^{\alpha} A^{\beta}` in terms of
:math:`V^{\alpha}`, :math:`R^{\alpha}`, :math:`\dot{V}^{\alpha}`, and
:math:`\kappa` yields the component-wise form implemented in
``core.trajectory_integrator._update_conjugate_momentum``.  The spatial
components couple velocity, acceleration, and retarded distance, ensuring that
head-on image-charge interactions reproduce the steep gradients reported in the
reference study.

Position updates follow directly from the Hamiltonian identity

.. math::
   :label: eq-eom-position

   \frac{d x^{\alpha}}{d\tau} = \frac{1}{m}\left( \mathcal{P}^{\alpha} - \frac{e}{c} A^{\alpha} \right),

which the solver evaluates after each momentum update to keep particle states in
sync.  Proper-time stepping avoids runaway behaviour at high :math:`\gamma`
while keeping the integration scheme close to the historical reference
implementation.

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

Relativistic position updates in coordinate time
------------------------------------------------

While the covariant formulation uses proper time :math:`d\tau`, the numerical
implementation steps forward in coordinate time with interval :math:`h = \Delta t`.
The spatial position update relates proper-time and coordinate-time derivatives:

.. math::

   \Delta \mathbf{x} = \mathbf{v} \, \Delta t = \frac{\mathbf{P}_{\text{kinetic}}}{\gamma m} \, h,

where :math:`\mathbf{P}_{\text{kinetic}} = \mathcal{P} - (e/c)\mathbf{A}` is the
kinetic (mechanical) momentum. The crucial :math:`1/\gamma` factor ensures that
velocity :math:`\mathbf{v} = \mathbf{P}_{\text{kinetic}}/(\gamma m)` remains
subluminal even as momentum grows with :math:`\gamma`.

The corresponding velocity is then computed from the coordinate-time displacement:

.. math::

   \boldsymbol{\beta} = \frac{\mathbf{v}}{c} = \frac{\Delta \mathbf{x}}{c \, h}.

Note that this formula does **not** include a :math:`\gamma` factor in the
denominator—the time dilation is already accounted for in the position update.

Self-consistency iterations
---------------------------

For ultra-relativistic particles (:math:`\gamma \gg 1`), forces depend strongly
on :math:`\gamma` through the retarded field geometry (via :math:`\kappa` and
field Lorentz contraction). This creates a circular dependency:

.. math::

   \gamma \rightarrow \text{forces} \rightarrow \mathcal{P} \rightarrow \gamma.

The integrator resolves this through self-consistency iterations at each timestep.
Within iteration :math:`n`:

1. Use :math:`\gamma_{n-1}` from the previous iteration to compute retarded forces
2. Update conjugate momentum :math:`\mathcal{P}_{n}` from those forces
3. Compute positions using the **same** :math:`\gamma_{n-1}`:
   :math:`\Delta \mathbf{x} = (\mathbf{P}_{\text{kinetic}}/(\gamma_{n-1} m)) h`
4. Compute velocity: :math:`\boldsymbol{\beta}_{n} = \Delta \mathbf{x}/(c h)`
5. Derive two independent estimates of :math:`\gamma_{n}`:

   * From energy: :math:`\gamma_{\text{E}} = (\mathcal{P}^{0} - e\Phi)/(mc)`
   * From velocity: :math:`\gamma_{\text{V}} = 1/\sqrt{1 - \beta^{2}}`

6. Check convergence: :math:`|\gamma_{\text{E}} - \gamma_{\text{V}}|/\gamma_{\text{E}} < \epsilon`

If not converged, iteration :math:`n+1` uses :math:`\gamma_{n} = \gamma_{\text{E}}`
and repeats. Typical tolerance :math:`\epsilon = 10^{-6}` achieves convergence
within 1–3 iterations even after large energy jumps.

The key to stable convergence is using a **consistent** :math:`\gamma` throughout
each iteration for both force calculation and position updates, ensuring that
the velocity extracted from :math:`\Delta \mathbf{x}` corresponds physically to
the momentum computed from those forces.

Implementation details are in :class:`core.self_consistency.SelfConsistencyConfig`
and :func:`core.equations.retarded_equations_of_motion`.

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

  This mode is opt-in and currently validated only against controlled
  prescribed-field cases.  Longitudinal acceleration should show the expected
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
benchmarking mode.  The exact mode is currently limited to fixed-step
``BUNCH_TO_BUNCH`` RFS/retarded-dipole runs and cannot be combined with driver
trains.  ``COLD_START`` remains appropriate when the desired model is a genuine
turn-on transient.

COLD_START gating mechanism
---------------------------

The COLD_START startup mode suppresses retarded forces during the initial phase
of a simulation until particles have traveled far enough from their origin for
light from external sources to have causally reached them. This ensures physical
causality is respected and avoids applying forces based on incomplete or
unphysical retarded histories.

Gating threshold formula
~~~~~~~~~~~~~~~~~~~~~~~~

The threshold distance a particle must travel before forces are applied is:

.. math::

   d_{\text{threshold}} = \frac{\beta \cdot R}{1 - \boldsymbol{\beta} \cdot \mathbf{n}},

where:

- :math:`\beta = |\boldsymbol{\beta}|` is the particle speed (in units of c)
- :math:`R` is the current distance from particle to external source
- :math:`\mathbf{n}` is the unit vector pointing from source to particle
- :math:`\boldsymbol{\beta} \cdot \mathbf{n}` is the velocity component along the separation

Physical interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

The threshold represents the distance the **particle** must travel before light
from the source's initial position could reach it. The formula accounts for the
relative closing speed between light and particle:

- **Approaching** (:math:`\boldsymbol{\beta} \cdot \mathbf{n} < 0`): particles
  and light meet quickly, threshold < :math:`\beta R`

  Example: :math:`\boldsymbol{\beta} \cdot \mathbf{n} = -\beta` (head-on) →
  threshold = :math:`\beta R / (1 + \beta)`

- **Perpendicular** (:math:`\boldsymbol{\beta} \cdot \mathbf{n} = 0`): light
  travels full distance, threshold = :math:`\beta R`

- **Receding** (:math:`\boldsymbol{\beta} \cdot \mathbf{n} > 0`): light takes
  longer to catch up, threshold > :math:`\beta R`

- **Receding at c** (:math:`\boldsymbol{\beta} \cdot \mathbf{n} \rightarrow 1`):
  light never catches particle, threshold → ∞ (forces never applied)

Dynamic threshold calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The threshold is **recalculated every integration step** using current values of
:math:`R` and :math:`\boldsymbol{\beta}`. This ensures causality is respected
as geometry evolves:

1. Distance :math:`R` changes as particles move and images reposition
2. Velocity :math:`\boldsymbol{\beta}` updates as particles accelerate
3. Threshold automatically decreases as particle approaches sources
4. Forces "turn on" dynamically when travel distance exceeds threshold

Two-stage implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

For computational efficiency, the code uses a two-stage check:

**Stage 1: Early conservative check** (performance optimization)

- Uses estimated maximum :math:`R` from external particle bounds
- Conservative threshold: :math:`\beta R_{\max} / (1 + \beta)` (assumes head-on approach)
- If travel distance < estimated threshold, skip expensive retarded distance calculations

**Stage 2: Precise check** (accurate gating)

- Uses actual retarded distance :math:`R` to each external source
- Per-source thresholds: :math:`\beta R / (1 - \boldsymbol{\beta} \cdot \mathbf{n})`
- Forces applied when travel distance ≥ threshold

Velocity regime behavior
~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`\beta` factor in the numerator is critical for non-relativistic particles:

.. list-table::
   :header-rows: 1
   :widths: 20 15 30 35

   * - Regime
     - :math:`\beta`
     - Threshold (approaching)
     - Physical meaning
   * - Non-relativistic
     - 0.01
     - :math:`0.01 R / 1.01 \approx 0.01 R`
     - Forces apply almost immediately
   * - Low velocity
     - 0.1
     - :math:`0.1 R / 1.1 \approx 0.09 R`
     - Early force application
   * - Moderate
     - 0.5
     - :math:`0.5 R / 1.5 \approx 0.33 R`
     - Forces apply at 1/3 distance
   * - Relativistic
     - 0.9
     - :math:`0.9 R / 1.9 \approx 0.47 R`
     - Near half-distance
   * - Ultra-relativistic
     - → 1
     - → :math:`R / 2`
     - Approach halfway limit (never exceed)

Without the :math:`\beta` factor, low-velocity particles would have forces
suppressed for distances far exceeding physical interaction regions, producing
incorrect physics.

Implementation
~~~~~~~~~~~~~~

The gating mechanism is implemented in ``core/equations.py``:

- :func:`_compute_gating_threshold` computes per-source thresholds
- :func:`_should_apply_external_forces` performs the Stage 2 check
- Early conservative check embedded in :func:`retarded_equations_of_motion`

For conducting walls, the distance :math:`R` is computed to image charges (not
the wall itself), ensuring correct handling of virtual source positions.

Bridging back to the code
-------------------------

The mathematical relationships above surface in the codebase as follows:

- :class:`core.trajectory_integrator.IntegratorConfig` captures the physical
  parameters (:math:`\Delta\tau`, aperture radius, wall position) implied by the
  analytical terms.
- :func:`core.trajectory_integrator.generate_conducting_image` and
  :func:`core.trajectory_integrator.generate_switching_image` encode the
  boundary conditions assumed when taking the head-on limit to model conducting
  apertures and switching walls.
- The reference notebooks under ``examples/validation/`` document historical
  comparisons and exploratory studies related to :eq:`eq-headon-limit`.
- The notebooks under ``legacy/`` are retained for historical investigations,
  but they are not part of the modern retarded-field workflows.

For deeper derivations and experimental context, see `<https://doi.org/10.1016/j.nima.2024.169988>`_.
