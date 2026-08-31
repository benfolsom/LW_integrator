Radiation-flux oracle
=====================

Purpose
-------

The radiation-flux oracle is an independent diagnostic.  It does not change a
particle trajectory.  It measures electromagnetic energy, momentum, and
angular momentum crossing a spherical surface, so that a local
radiation-reaction force or torque can be checked against an external
conservation measurement.

Here *oracle* means a deliberately transparent reference calculation.  It is
allowed to be slower than the production integrator.  It is not a new force
law and it is not assumed to be exact merely because it is called an oracle.
It must still pass analytic and numerical convergence checks.

The two calculations to compare are:

.. code-block:: text

   source histories -> retarded fields on a sphere -> outward flux

   source histories -> local self-reaction model   -> work, impulse, torque

Instantaneous values do not generally agree because the bound near field can
temporarily store energy, momentum, and angular momentum.  Comparisons over a
time interval must include the corresponding bound-field, or Schott, change.
Bonga, Poisson, and Yang demonstrate this balance explicitly for a spinning
charged shell [Bonga2018]_.

What is calculated
------------------

In native scaled-Gaussian units, the outward energy-flux density is the radial
component of the Poynting vector,

.. math::

   \mathbf S = \frac{c}{4\pi}\,\mathbf E\times\mathbf B.

The outward momentum-flux density through a surface with unit normal
:math:`\mathbf n` is

.. math::

   \mathbf f_p = \frac{1}{4\pi}
   \left[
   \frac{E^2+B^2}{2}\mathbf n
   -\mathbf E(\mathbf E\cdot\mathbf n)
   -\mathbf B(\mathbf B\cdot\mathbf n)
   \right].

The angular-momentum flux about a chosen origin is obtained by integrating
:math:`(\mathbf x-\mathbf x_0)\times\mathbf f_p` over the sphere.  The full
Maxwell stress is used, rather than only the leading radial radiation term,
because subleading transverse stress can contribute to angular momentum.

This last quantity must be named carefully.  It is the angular-momentum
transport defined by the symmetric Maxwell stress tensor.  Bonga, Grant, and
Prabhu show that its null-infinity limit can depend on Coulombic as well as
radiative field components, while the corresponding Noether-current flux is
purely radiative [Bonga2020]_.  The oracle therefore reports stress-tensor
angular-momentum flux but does not automatically label all of it as angular
momentum radiated by the source.

The source field is split into charge and intrinsic magnetic-dipole parts.
Because the stress tensor is quadratic in the field, every reported quantity
is separated into:

* the charge-only :math:`q^2` sector;
* the signed charge--dipole :math:`q\mu` interference sector; and
* the dipole-only :math:`\mu^2` sector.

The three sectors add algebraically to the result obtained from the total
field.

Four-layer design
-----------------

``integrate_radiation_sphere_flux_native`` accepts fields already sampled on
a sphere.  This pure layer tests the Poynting-vector, Maxwell-stress, sector,
and angular-momentum accounting without invoking a retarded-field solver.

``evaluate_retarded_radiation_sphere_native`` is the reference sampling layer.
It evaluates the maintained retarded charge and magnetic-dipole providers at
each angular point and then calls the pure integrator.  Its default ``python``
backend is intentionally explicit.  Faster exact-retarded backends can be
selected for larger diagnostics, but must be compared with the Python result.

``integrate_radiation_sphere_flux_history_native`` is the pure time-integration
layer.  It accepts an ordered series of results for one fixed sphere and uses
the trapezoidal rule on an arbitrary strictly increasing observation-time
grid.  It returns transported energy, linear momentum, and angular momentum
for every field sector.  It also retains the observation interval, the
retarded-time envelope when available, and the largest provider light-cone
residual.

The time integral is still a surface-flux result, not a recoil force or
impulse.  Closing a conservation law against particle work or impulse requires
the change in bound, Schott-like field energy and momentum over the same source
interval.  Radius comparisons must shift their observation windows so that
the retarded-time envelopes match; integrating the same coordinate-time
window at two radii generally compares different emitted wavefronts.

``evaluate_radiation_reaction_balance_native`` is the accounting layer for a
proposed local self-reaction law.  With outward flux positive, it reports the
residual of

.. math::

   \Delta Q_{\rm mechanical}
   + Q_{\rm outward}
   + \Delta Q_{\rm bound},

for energy and linear momentum, and optionally for angular momentum.  The
function does not infer a missing bound term or decide that a residual is
small enough.  It makes the sign convention and missing inputs explicit so a
model can be tested under numerical refinement.

The current implementation is diagnostic-only: it is not called by the
equations of motion, Medina radiation reaction, checkpoint scheduler, CLI, or
GUI.

Required convergence checks
---------------------------

A finite sphere contains both radiative and bound-field transport.  Moving
the sphere outward while holding coordinate time fixed does not compare the
same emitted wavefront.  For radii :math:`R_1` and :math:`R_2`, shift the
observation times by approximately :math:`(R_2-R_1)/c` so that the same source
emission interval is sampled.

Before treating a result as radiation evidence, require:

* increasing sphere radius at matched retarded source time;
* increasing polar and azimuthal quadrature orders;
* tighter retarded-root and dipole-stencil settings;
* agreement with static no-radiation cases;
* agreement with the Gaussian oscillating-dipole power law; and
* closure of the total field against the sum of the :math:`q^2`,
  :math:`q\mu`, and :math:`\mu^2` sectors.

The first maintained provider-level benchmarks cover an accelerated charge at
instantaneous rest, a prescribed rotating magnetic moment at rest, and the
interference of independently prescribed electric- and magnetic-dipole
radiation.  The charge result reproduces the nonrelativistic Larmor power.  At
matched retarded time the measured :math:`\mu^2` power agrees with the
far-field limit of the retarded moving-dipole solution [Heras1998]_,

.. math::

   P_\mu = \frac{2\lvert\ddot{\boldsymbol\mu}\rvert^2}{3c^3}

to better than one part per million at both tested radii.  The
:math:`q\mu` benchmark has zero integrated interference power and approaches
the expected directional momentum flux as the sphere is enlarged.  This
validates the first energy- and linear-momentum-flux milestone.  The maintained
tests also show that time-integrated outgoing-wave energy is invariant under a
radius change when the observation window is shifted to cover the same source
times.  For a circularly rotating rest-frame moment,

.. math::

   \boldsymbol\mu(t)=\mu_0(\cos\omega t,\sin\omega t,0),

the provider reproduces both the power above and the emitted angular momentum

.. math::

   \dot J_z={P_\mu\over\omega}
   ={2\mu_0^2\omega^3\over3c^3}

to better than one part per million at both tested radii.  This validates the
first provider-level angular-momentum benchmark for the symmetric-stress
quantity.

The generic balance layer is separately checked against Medina's charge-only
bound energy and momentum.  The residual decreases quadratically when the
time spacing is halved, as expected for trapezoidal integration.  This checks
the accounting signs and the known charge-sector Schott term.

An independent end-to-end charge benchmark drives the retarded provider with
one complete period of harmonic motion.  The source state and its bound field
return to their initial values, so the net Schott boundary change is zero.
Sphere-flux integration at 20 and 80 mm agrees with Medina's integrated far
energy and closes reaction work plus outward energy to within the maintained
``5e-10`` relative tolerance.  This establishes the complete-period
sphere-versus-local route without assuming the local far-radiation value in
the surface calculation.

A nonzero-boundary charge interval, identification of the magnetic bound
contribution, comparison with a finite-size self-torque, and application to an
archived flyby remain later acceptance steps.

References
----------

.. [Heras1998] J. A. Heras, "Explicit expressions for the electric and
   magnetic fields of a moving magnetic dipole," *Physical Review E* **58**,
   5047 (1998), `doi:10.1103/PhysRevE.58.5047
   <https://doi.org/10.1103/PhysRevE.58.5047>`_.

.. [Bonga2018] B. Bonga, E. Poisson, and H. Yang, "Self-torque and angular
   momentum balance for a spinning charged sphere," *American Journal of
   Physics* **86**, 839 (2018), `doi:10.1119/1.5054590
   <https://doi.org/10.1119/1.5054590>`_, `arXiv:1805.01372
   <https://arxiv.org/abs/1805.01372>`_.

.. [Bonga2020] B. Bonga, A. M. Grant, and K. Prabhu, "Angular momentum at null
   infinity in Einstein--Maxwell theory," *Physical Review D* **101**, 044013
   (2020), `doi:10.1103/PhysRevD.101.044013
   <https://doi.org/10.1103/PhysRevD.101.044013>`_, `arXiv:1911.04514
   <https://arxiv.org/abs/1911.04514>`_.
