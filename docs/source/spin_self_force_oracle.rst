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

Production use is intentionally blocked.  The unreduced equation contains
four-snap and spin/moment derivatives.  A production model must specify how
those derivatives are obtained without introducing runaway solutions, and it
must avoid double counting the charge term already supplied by Medina.  It
must also close mechanical impulse against far-zone flux plus the appropriate
bound-field, or Schott-like, change.  Jakobsen's supplemental comparison to
Villarroel [Villarroel1975]_ shows explicitly that the radiated momentum and
the local force differ by a total derivative and an additional radiative-field
term, so far flux alone is not the instantaneous local force.

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
