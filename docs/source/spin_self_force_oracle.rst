Point-particle spin self-force oracle
=====================================

Purpose and scope
-----------------

``evaluate_jakobsen_linear_spin_self_force_native`` is a diagnostic
translation of the point-particle self-force derived by Jakobsen
[Jakobsen2024]_.  It calculates the ordinary charge Abraham--Lorentz--Dirac
(ALD) term and the first correction that is linear in physical spin or
magnetic moment.  It does not apply either force to a trajectory.

The word *linear* is important.  Electromagnetic radiation bookkeeping can be
split into charge-only :math:`q^2`, charge--moment interference
:math:`q\mu`, and moment-only :math:`\mu^2` sectors.  Jakobsen's result is an
effective point-particle expansion through first order in spin and
magnetization.  It covers the :math:`q\mu` and related :math:`qS` terms but
does not contain the quadratic :math:`\mu^2` sector.  The finite rotating-shell
oracles show that :math:`\mu^2` radiation exists for a specified extended
source; matching that source-dependent result to an intrinsic point particle
remains a separate task.

Variables and project normalization
-----------------------------------

Jakobsen uses :math:`a^\mu` for four-acceleration.  The integrator already
uses :math:`a^\mu` for its dimensionless spin direction, so the oracle uses
the following names:

* :math:`u^\mu` is four-velocity, with :math:`u^2=c^2`;
* :math:`A^\mu=du^\mu/d\tau` is four-acceleration;
* :math:`J^\mu=dA^\mu/d\tau` is four-jerk; and
* :math:`K^\mu=dJ^\mu/d\tau` is four-snap.

All derivatives use proper time in nanoseconds.  The physical spin passed to
the oracle is

.. math::

   S^\mu=(I\hbar)a^\mu_{\rm normalized},

where ``spin_quantum_number`` is :math:`I`.  The current no-susceptibility
particle model similarly uses

.. math::

   M^\mu=\mu_{\rm signed}a^\mu_{\rm normalized}.

Thus the oracle must not receive the normalized RFS spin direction in place
of :math:`S^\mu`.  Spin is in native action units
``amu mm^2/ns``; magnetic moment is in native Gaussian ``charge mm``.

Native-unit equation
--------------------

Jakobsen works in rationalized Heaviside--Lorentz units with
:math:`c=\epsilon_0=\mu_0=1`.  Converting charge and moment to Gaussian units
contributes one factor of :math:`\sqrt{4\pi}` for each electromagnetic source.
Restoring the proper-time and velocity factors gives

.. math::

   F^\mu_{qS}
   = {2q\over3c^4}P^\mu{}_{\nu}
   \left\{
     (J\times\dot M)^\nu
     +{d\over d\tau}
      \left[J\times\left(M-{q\over mc}S\right)\right]^\nu
   \right\},

with

.. math::

   P^\mu{}_{\nu}
   =\delta^\mu{}_{\nu}-{u^\mu u_\nu\over c^2}.

The body-frame cross product is

.. math::

   (V\times W)^\mu
   =\epsilon^\mu{}_{\nu\rho\sigma}
    V^\nu W^\rho {u^\sigma\over c},
   \qquad \epsilon^{1230}=1.

Because that cross product uses the moving four-velocity, its derivative acts
on three factors, not two.  Defining
:math:`D=M-qS/(mc)`, the implementation expands it as

.. math::

   {d\over d\tau}(J\times D)
   =\dot J\times D+J\times\dot D
    +\epsilon^\mu{}_{\nu\rho\sigma}
      J^\nu D^\rho {A^\sigma\over c}.

Omitting the final moving-frame term would be a non-covariant approximation.

As a unit and sign check, the same conversion applied to the paper's
charge-only term gives

.. math::

   F^\mu_{q^2}={2q^2\over3c^3}P^\mu{}_{\nu}J^\nu.

At instantaneous rest this is the Gaussian ALD coefficient already used by
the Medina kernel.  The maintained test evaluates both implementations from
the same rest-frame force derivative and requires agreement.

What the result reports
-----------------------

The result retains the unsubtracted moment
:math:`D=M-qS/(mc)`, both contributions inside braces, their projected sum,
the linear-spin force, the charge ALD force, and their total.  It also reports
:math:`u\cdot F` and the input :math:`u\cdot S` and :math:`u\cdot M`
residuals.  The projector makes the returned four-force orthogonal to
:math:`u^\mu`, as required for fixed mechanical rest mass.

The returned self-torque correction is exactly zero.  This records the result
at the order calculated by Jakobsen; it is not a claim that all finite-size or
higher-order self-torques vanish.

Radiated momentum is not the mechanical force
----------------------------------------------

The expanded supplement to the published paper compares the local
self-force with Villarroel's instantaneous radiated four-momentum.  Its
Eq. (33) uses :math:`v^\mu` without separately defining that symbol.  The
main text consistently uses :math:`\dot z^\mu` for four-velocity; its
dimensions and the balance test below identify :math:`v^\mu` here with the
normalized four-velocity :math:`\bar u^\mu=u^\mu/c`.  No erratum or later
arXiv revision was found as of August 2026.  The implementation records this
interpretation explicitly rather than silently changing notation.

The leading charge radiative electric field is

.. math::

   E_{\rm rad}^\mu={2q\over3c^3}P^\mu{}_{\nu}J^\nu .

The additional term on the local side of the radiated-momentum identity is

.. math::

   \Delta_{\rm rad}^\mu
   ={q\over mc^3}\bar u^\mu
     \left[S\cdot(A\times E_{\rm rad})\right].

This term is parallel to the four-velocity.  It is therefore **not** another
mechanical force to apply to a fixed-mass particle.  It belongs only to the
comparison between local reaction and transported field momentum.

For an intrinsic moment with no susceptibility,
:math:`M=gqS/(2mc)`, the complete native-unit identity is

.. math::

   F_{qS}^\mu+\Delta_{\rm rad}^\mu
   =\dot P_{\rm rad,particle}^\mu
    +{dB_{\rm bound}^\mu\over d\tau},

where positive outward radiation is
:math:`-\dot P_{\rm rad,particle}^\mu` and

.. math::

   B_{\rm bound}^\mu
   ={q^2\over3mc^5}
   \left[g(A\times\dot S)^\mu
   +(g-2)(J\times S)^\mu\right],

.. math::

   \dot P_{\rm rad,particle}^\mu
   ={q^2g\over3m}
   \left{
   {\bar u^\mu[S\cdot(A\times J)]\over c^6}
   +{(\ddot S\times A)^\mu\over c^5}
   \right}.

``evaluate_jakobsen_intrinsic_spin_radiation_balance_native`` evaluates the
radiated term and the proper-time derivative of the bound momentum directly.
It does not define one by subtracting the other, so its residual checks the
signs, powers of :math:`c`, and moving-frame derivatives independently.

Validation and remaining gates
------------------------------

The focused tests presently establish:

* the instantaneous-rest reduction to ordinary three-vector cross products;
* the moving-frame derivative term and the paper's Levi--Civita sign;
* the charge coefficient against the independent Medina implementation;
* covariance under a finite Lorentz boost;
* :math:`u\cdot F=0` after projection;
* the neutral limit, which correctly has no term linear in moment because the
  outer charge is zero; and
* the static :math:`g=2` intrinsic-moment cancellation in the retained local
  expression.

A first independent conservation benchmark uses a charge oscillating along
one axis while its fixed-magnitude intrinsic spin and moment rotate in the
perpendicular plane.  The acceleration and charge radiative electric field
are collinear in this geometry, so the spin-specific radiative-field
correction in the paper's supplemental Eq. (33) vanishes geometrically.  The
interval is one full period, so the remaining local total derivative returns
to its initial value.  The maintained Maxwell-stress integrator independently
evaluates the cross term of the standard electric- and magnetic-dipole
radiation fields.  The local impulse and outward :math:`q\mu` momentum are
equal and opposite at both tested angular quadratures.

A slower provider-level version constructs the complete retarded charge and
dipole fields from the same periodic history.  At 400- and 800-mm spheres, its
radiative momentum agrees with the opposite local impulse within the declared
convergence tolerance.  An additional transverse finite-radius term decreases
by two when the radius doubles, identifying the expected :math:`1/R`
bound-field transport rather than far radiation.

A second provider-level benchmark uses uniform circular motion with the spin
and moment aligned normal to the orbit.  This is the leading motion generated
by a uniform magnetic field with aligned spin, rather than an arbitrary
prescribed spin history.  Here :math:`S\cdot(A\times J)` is nonzero.  The
projected mechanical self-force by itself accounts for only part of the
outward charge--moment interference energy.  After adding
:math:`\Delta_{\rm rad}`, the local loss agrees with the complete retarded
provider flux to about two parts per million at both tested radii.  The
direct Villarroel/Jakobsen radiated-loss expression gives the same result,
and the bound momentum returns to its initial value after one orbit.

An arbitrary-state unit test also closes
:math:`F_{qS}+\Delta_{\rm rad}=\dot P_{\rm rad,particle}+\dot B_{\rm bound}`
without relying on periodic cancellation.

The provider-level nonperiodic test is now also complete.  It selects one
quarter of a faster circular orbit, for which the endpoint change in spatial
bound momentum is more than one thousand times the small radiative momentum.
Each angular ray is evaluated at the observation time whose retarded event is
the declared source time, with the exact
:math:`dt_{\rm obs}/dt_{\rm source}=1-\mathbf n_{\rm ret}\cdot\boldsymbol\beta`
Jacobian.  The finite-radius momentum is intentionally dominated by
near-field transport.  A quadratic extrapolation in :math:`1/R` over three
radii recovers the constant radiative four-momentum and closes it against the
local impulse plus the nonzero bound-field endpoint change within two percent
in the small spatial components.  The much larger energy component closes
more tightly.

Sampled reduction-of-order oracle
---------------------------------

``evaluate_sampled_intrinsic_spin_reduction_native`` is the first diagnostic
bridge from the high-derivative formula to a causal production model.  Its
inputs are a short proper-time stencil of leading-order four-velocity,
**non-self** four-acceleration, and physical spin.  The non-self qualifier is
the reduction-of-order rule: the already-small self-reaction term is evaluated
on the ordinary lower-order motion, rather than being allowed to create new
independent acceleration modes.

The helper reconstructs jerk, snap, and the first two spin derivatives with
arbitrary-node finite-difference weights, then calls the intrinsic-spin
balance oracle.  It also differentiates the velocity samples and reports the
difference from the supplied center acceleration.  A nonzero residual warns
that the sampled leading trajectory is internally inconsistent.  Center
values are subtracted before every differentiation so the nearly constant
temporal velocity near :math:`c` is not damaged by subtracting large weighted
numbers.

This helper is intentionally **not** the production algorithm.  Its centered
stencil uses future samples, and numerical differentiation is less attractive
than the analytical potential/response jets already used by the exact
provider.  It supplies a reference target for a later causal implementation.
Tests recover polynomial derivatives on an irregular grid and show
fourth-order convergence to the exact circular-orbit self-force when the
proper-time spacing is halved.

The nested result still reports the charge ALD term for comparison, but a
future production caller must retain the existing Medina charge reaction and
apply only the new linear-spin contribution.  Adding both charge terms would
double count the :math:`q^2` sector.

Production use is intentionally blocked.  The unreduced equation contains
four-snap and spin/moment derivatives.  A production model must specify how
those derivatives are obtained without introducing runaway solutions, and it
must avoid double counting the charge term already supplied by Medina.  It
must also close mechanical impulse against far-zone flux plus the appropriate
bound-field, or Schott-like, change.  Jakobsen's supplemental comparison to
Villarroel [Villarroel1975]_ shows explicitly that the radiated momentum and
the local force differ by a total derivative and an additional radiative-field
term, so far flux alone is not the instantaneous local force.

The next milestone is to obtain the same derivative contractions from the
analytical non-self RFS response, or from a separately validated causal
accepted-history stencil, without future samples.  Compare that causal result
with both the sampled reduction oracle and the unreduced circular benchmark
under a controlled slow-reaction expansion.  Production injection remains
blocked until the comparison converges and the existing Medina charge term is
shown to enter exactly once.

References
----------

.. [Jakobsen2024] G. U. Jakobsen, "Spin and Susceptibility Effects of
   Electromagnetic Self-Force in Effective Field Theory," *Physical Review
   Letters* **132**, 151601 (2024),
   `doi:10.1103/PhysRevLett.132.151601
   <https://doi.org/10.1103/PhysRevLett.132.151601>`_,
   `arXiv:2311.04151 <https://arxiv.org/abs/2311.04151>`_.

.. [Villarroel1975] D. Villarroel, "Local characterization of massless
   radiation from point sources," *Annals of Physics*, 113--126 (1975),
   `doi:10.1016/0003-4916(75)90142-6
   <https://doi.org/10.1016/0003-4916(75)90142-6>`_.
