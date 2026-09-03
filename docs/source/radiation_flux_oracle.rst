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
Its optional ``sample_time_jacobian`` supports a matched-light-cone
parameterization.  Each angular ray may then be sampled at the observation
time whose retarded root is one declared source time, while the factor
:math:`dt_{\rm obs}/dt_{\rm source}` converts the surface flux to a rate per
source time.  The ordinary path omits this argument and is unchanged.

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

For a nonperiodic moving source, one constant observation time is not enough
to match source endpoints over the whole sphere: different angular rays have
different light-travel distances.  The maintained intrinsic-spin test uses

.. math::

   t_{\rm obs}(\mathbf n,t_s)
   =t_s+{|R\mathbf n-\mathbf z(t_s)|\over c},
   \qquad
   {dt_{\rm obs}\over dt_s}=1-\mathbf n_{\rm ret}\cdot\boldsymbol\beta.

It verifies every returned retarded root against :math:`t_s`, applies this
Jacobian per angular sample, and extrapolates the finite-radius result in
:math:`1/R`.  This is the matched null-boundary construction needed to compare
one common nonperiodic source interval with a local bound-field endpoint
change.

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

Finite spinning-shell benchmark
-------------------------------

``evaluate_spinning_shell_angular_balance_native`` implements the
slow-variation expansion for a uniformly charged, infinitesimally thin
spherical shell derived by Bonga, Poisson, and Yang [Bonga2018]_.  It reports
the shell's electromagnetic self-torque, outward angular-momentum flux,
near-field and wave-zone angular momentum, and the residual of

.. math::

   {dJ_{\rm field}\over dt}+N_{\rm outward}+T_{\rm self}=0.

The inputs are a signed axial magnetic moment and its first eight time
derivatives, evaluated at the two retarded source times required by the
finite-shell calculation.  A companion function evaluates the current-time
self-torque and separates its time-symmetric and radiation-sensitive terms.
This distinction matters physically: the largest self-torque terms store and
return field angular momentum or alter electromagnetic inertia.  They do not
represent irreversible radiation reaction.  The first radiation-sensitive
term is proportional to the fourth time derivative of the moment.

This benchmark is proportional to charge times magnetic moment.  It therefore
tests the :math:`q\mu` interference sector, **not** the pure :math:`\mu^2`
sector.  The earlier roadmap description of this paper as a pure magnetic
self-torque benchmark was incorrect.  A neutral intrinsic dipole still needs
a separate finite-size source model before a :math:`\mu^2` local self-torque
can be certified.

Exact harmonic shell response
-----------------------------

``evaluate_harmonic_spinning_shell_response_native`` implements the exact
single-frequency response of the same uniformly charged shell derived by
Mansuripur and Jakobsen [Mansuripur2020]_.  This is the next bridge between the
local shell torque and the independently measured :math:`\mu^2` energy flux.
For

.. math::

   \Omega(t)=\operatorname{Re}[\Omega_0e^{-i\omega t}],
   \qquad x={\omega R\over c},

define :math:`g(x)=\sin x-x\cos x`.  The exact response coefficient is

.. math::

   \Gamma(\omega)={Z_0q^2\over6\pi}
   {g(x)\over x^2}(1-i x)e^{i x},

and the complex self-torque amplitude is

.. math::

   T_{\rm self,0}=i\Gamma(\omega)\Omega_0.

The cycle-averaged outward power is evaluated separately as

.. math::

   \langle P_{\rm out}\rangle
   ={Z_0q^2|\Omega_0|^2\over12\pi}
   {g(x)^2\over x^2}.

The maintained test verifies the energy balance

.. math::

   {1\over2}\operatorname{Re}
   \left[T_{\rm self,0}\Omega_0^*\right]
   +\langle P_{\rm out}\rangle=0.

The paper absorbs a factor of :math:`\mu_0` into its magnetic-moment symbol.
The implementation instead returns the ordinary moment used by the
integrator,

.. math::

   \mu_{\rm amp}={qR^2\Omega_0\over3},

in native moment units.  This makes its point-size power directly comparable
with the radiation-sphere provider.  The finite-size correction relative to
that point result is

.. math::

   {P_{\rm shell}\over P_{\rm point}}
   =\left[{3(\sin x-x\cos x)\over x^3}\right]^2.

The code evaluates these expressions with small-:math:`x` series that avoid
subtracting nearly equal sine and cosine terms.  It also reports the shell's
maximum equatorial speed divided by :math:`c`; the underlying shell motion is
nonrelativistic even though the external LW integrator supports relativistic
translation.

This exact frequency-domain result is still an oracle, not a production
self-torque.  It assumes one fixed rotation axis and one prescribed harmonic
frequency.  The present milestone checks power balance, the point-dipole
sphere-flux limit, and convergence to the Bonga--Poisson--Yang slow-variation
series.

``evaluate_harmonic_spinning_shell_transfer_native`` extends the same exact
coefficient to complex frequency and evaluates the mechanical transfer
function

.. math::

   {\Omega_0\over T_0}
   ={i\over I\omega+\Gamma(\omega)+i\beta},
   \qquad I={2\over3}mR^2.

A pole of this response in the upper half of the complex-:math:`\omega` plane
would produce a nonzero impulse response before the applied torque.  The
companion ``count_harmonic_spinning_shell_transfer_poles_native`` uses
Cauchy's argument principle to count denominator zeros inside one explicitly
declared rectangle.  It evaluates the complete complex response, including
the reversible part of :math:`\Gamma`; discarding that apparently
inertia-like part changes the pole structure.

For the paper's electron-mass, electron-charge, 1-nm shell with zero ordinary
friction, the maintained test finds no exact-response poles in three expanding
upper-half-plane boxes reaching :math:`|\operatorname{Re}x|=400` and
:math:`\operatorname{Im}x=400`, where :math:`x=\omega R/c`.  The same counter
finds two upper-half-plane poles when Eq. (19), the small-radius derivative
truncation, replaces the exact response.  It also finds thirteen exact poles
in a smaller lower-half-plane box, demonstrating that the zero upper count is
not a counter that simply misses all roots.  Both the contour density and box
size are varied in the tests.

``reconstruct_harmonic_spinning_shell_impulse_response_native`` performs the
complementary time-domain check.  It reports the normalized response
:math:`I\Omega(t)/L_{\rm impulse}` as a function of :math:`ct/R`.  For the
exact model it subtracts the known bare-inertia transfer function before the
finite Fourier integral and restores its causal exponential analytically.
This avoids mistaking the slowly converging Fourier representation of the
instantaneous mechanical velocity jump for a pre-impulse signal.  The
small-radius truncation falls sufficiently rapidly at high frequency to be
integrated directly.

With dimensionless mechanical friction
:math:`\beta R/(Ic)=0.1`, the exact 1-nm electron shell has a maximum sampled
pre-impulse response below :math:`10^{-9}` at both tested frequency limits.
The truncated control retains a converged pre-impulse response above 0.3.
Increasing the exact frequency limit from 200 to 400 and the truncated limit
from 400 to 800, while retaining a dimensionless frequency spacing of 0.01,
leaves the post-impulse response stable within the maintained tolerances.

The pole count and time reconstruction together provide strong numerical
causality evidence and reproduce the contrast in Mansuripur--Jakobsen Fig. 2.
They are not a mathematical proof over the entire unbounded upper half-plane.

Neutral counter-rotating shell construction
-------------------------------------------

``evaluate_neutral_counterrotating_shell_response_native`` records the
paper's explicit charge-neutral realization.  Two nearly coincident shells
carry charges :math:`(+q/2,-q/2)`, masses :math:`(m/2,m/2)`, and angular
velocities :math:`(+\Omega,-\Omega)`.  Their net electric charge is zero, but
the charge-current products have the same sign, so the two magnetic moments
add:

.. math::

   \mu_{+}+\mu_{-}={qR^2\Omega\over3}.

Mansuripur and Jakobsen show that the collective equation of motion retains
the one-shell form with parameters :math:`q`, :math:`m`, and :math:`\Omega`.
The result object makes that equivalence explicit while preserving both
internal charges and rotations.  Tests verify exact charge cancellation,
equal per-shell moment contributions, the total moment, and equality with the
effective one-shell harmonic response.

This resolves an important sector distinction.  The external electric
monopole vanishes, so a charge--moment flux formed from *net* charge times
total moment cancels.  The varying total moment and its :math:`\mu^2` radiated
power survive.  The local torque also survives because the neutral source
contains oppositely moving internal charges; it is not obtained by inserting
``charge_native=0`` into the charged-shell formula.

This construction is a specific finite-size neutral-current model, not a
universal structureless point-dipole law.  Its response still depends on the
internal charge and radius used to realize a given magnetic moment.  Matching
that dependence to an intrinsic-particle effective theory remains necessary
before production use.

Neutral-shell pulse energy ledger
---------------------------------

``evaluate_neutral_spinning_shell_pulse_energy_balance_native`` extends the
same exact response from one harmonic to a sampled, fixed-axis rotation
pulse.  It Fourier-decomposes the prescribed angular velocity, evaluates the
exact shell self-torque coefficient at every frequency, and integrates its
work.  Independently, it evaluates the outward magnetic radiation from

.. math::

   \widetilde\mu(\omega)
   ={qR^2\over3}\widetilde\Omega(\omega)

with the finite-shell form factor

.. math::

   \left[{3(\sin x-x\cos x)\over x^3}\right]^2,
   \qquad x={\omega R\over c}.

For a one-shot pulse, the sampled window must include quiet intervals before
and after the rotation.  Then the initial and final source states agree and
there is no *net* bound-field endpoint change.  The maintained smooth-pulse
test consequently verifies

.. math::

   W_{\rm self}+E_{\rm out}=0.

The oracle also inverse-transforms the complete complex self-torque and the
finite-shell radiation amplitude.  It reports the instantaneous self-work and
outward power, then defines an **inferred** reversible field-energy change by

.. math::

   \Delta E_{\rm inferred}(t)
   =-\int_{t_0}^{t}
   \left[P_{\rm self}(t')+P_{\rm out}(t')\right]dt'.

For the maintained slow pulse, the largest reversible excursion is more than
200 times the final radiated energy, but it returns to zero when the shell
returns to rest.  This illustrates why radiation power alone cannot be used
as an instantaneous damping force.

The result reports the angular velocity at the window boundaries relative to
its peak and the fraction of radiated energy in the upper ten percent of the
sampled frequency band.  These expose a poorly isolated pulse or inadequate
time resolution instead of silently treating its periodic Fourier extension
as a physical transient.

This is an integrated :math:`\mu^2` dissipation check for the explicit neutral
two-shell source.  The time-dependent field-energy curve is inferred from the
balance above; it is not yet an independent near-field integral.  The oracle
therefore does **not** provide a certified time-local intrinsic-dipole
self-torque or a translational recoil law.
Earlier point- and accelerated-dipole calculations also find magnetic
radiation-reaction effects [Itoh1991]_ [Unruh1999]_, but their source models
and approximations do not by themselves perform this finite-source-to-
intrinsic-particle matching.

``evaluate_neutral_spinning_shell_finite_sphere_energy_native`` supplies the
independent near-field calculation for the same fixed-axis source.  It uses
the exact causal shell potential of Bonga--Poisson--Yang Eq. (26), valid on
both sides of the material shell,

.. math::

   A_\phi={\mu_0\over4\pi}{\cal G}(t,r)\sin\theta.

After performing the angular integral analytically, the remaining radial
energy density is

.. math::

   {dU_{\rm em}\over dr}={\mu_0\over12\pi}
   \left[
   {r^2\over c^2}{\cal G}_t^2
   +({\cal G}+r{\cal G}_r)^2+2{\cal G}^2
   \right].

Gaussian radial quadrature is split at :math:`r=R`; no point-dipole
singularity or extrapolation through the shell is used.  The Poynting flux is
evaluated independently on a finite observation sphere at the same coordinate
time.  The maintained pulse test closes

.. math::

   \Delta U_{\rm em}
   +\int P_{\rm self}\,dt
   +\int P_{\rm outward}\,dt=0.

The maximum residual decreases by approximately four on each factor-two time
refinement from 128 through 512 samples.  At the finest level it is below
:math:`3\times10^{-4}` of the largest stored field energy.  A static control
matches the analytic field energy inside the selected observation sphere to
:math:`10^{-12}` relative and has exactly zero flux and balance residual.

This establishes the energy ledger independently for the explicit finite
source.  A corresponding pure-:math:`\mu^2` angular-momentum ledger,
arbitrary-axis motion, and intrinsic-particle matching remain open.

Distinct-shell angular-momentum ledger
--------------------------------------

``evaluate_concentric_neutral_shell_angular_momentum_balance_native`` keeps
the two countercharged shells at distinct radii while allowing each angular
velocity to be a three-vector.  The exact retarded shell potential is summed
before calculating fields.  Direct angular integration of the surface
Lorentz force gives the electromagnetic torque on shell :math:`i`,

.. math::

   \boldsymbol N_i={\mu_0q_iR_i\over6\pi}
   \left[
   \boldsymbol\Omega_i\mathbin\times\boldsymbol{\cal G}(t,R_i)
   -\partial_t\boldsymbol{\cal G}(t,R_i)
   \right].

This is the total vector torque on that shell.  It is not the scalar
generalized torque conjugate to the relative counterrotation in the harmonic
energy model.

For distinct radii, charge neutrality holds outside both shells but not in
the annulus between them.  The radial field-angular-momentum density therefore
contains both a magnetic :math:`\mu^2` term and an internal reversible
:math:`q\mu` term,

.. math::

   {d\boldsymbol J_{\rm field}\over dr}
   ={\mu_0r^2\over6\pi c^2}
   \boldsymbol{\cal G}\mathbin\times\partial_t\boldsymbol{\cal G}
   -{\mu_0Q_{\rm enclosed}r\over6\pi}
   \left(\partial_r\boldsymbol{\cal G}
   +{\boldsymbol{\cal G}\over r}\right).

The second term is essential.  Omitting it gives a nonconvergent transient
ledger even though a quiet final boundary can misleadingly close.  With both
terms present, the maintained rotating-pulse test closes

.. math::

   \Delta\boldsymbol J_{\rm field}
   +\int\left(\sum_i\boldsymbol N_i
   +\dot{\boldsymbol J}_{\rm outward}\right)dt=0

with second-order time refinement.

A shell-separation ladder distinguishes the sectors.  The transverse internal
:math:`q\mu` torque and field momentum decrease linearly with
:math:`|R_+-R_-|`.  The axial torque, field momentum, and neutral exterior
:math:`\mu^2` flux approach finite values.  This supplies the first controlled
coincident-shell angular-momentum limit, but it remains a nonrelativistic
finite-source oracle rather than an intrinsic-particle torque law.

Retarded/advanced split and point-dipole limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full retarded shell torque contains two physically different effects.  A
time-symmetric part changes the object's electromagnetic inertia and exchanges
angular momentum reversibly with its nearby field.  A radiation-sensitive
part remains after subtracting the corresponding advanced-field solution.
The oracle reports these separately as

.. math::

   \boldsymbol N_{\rm symmetric}
   ={\boldsymbol N_{\rm ret}+\boldsymbol N_{\rm adv}\over2},\qquad
   \boldsymbol N_{\rm RR}
   ={\boldsymbol N_{\rm ret}-\boldsymbol N_{\rm adv}\over2}.

This split explains why comparing the complete retarded torque directly with
a point radiation-reaction law gives a worse result as the motion is slowed:
the reversible electromagnetic-inertia term has not been removed.  After the
split, the radiation-sensitive torque approaches the standard point-dipole
expression [StumpPollack1997]_

.. math::

   \boldsymbol N_{\rm RR}^{\rm point}
   ={\mu_0\over6\pi c^3}
   \boldsymbol\mu\mathbin\times\boldsymbol\mu^{(3)}.

For the maintained rotating pulse, increasing its duration from 80 to 160 to
320 shell light-crossing times reduces the axial relative error from 14.1% to
3.73% to 0.945%.  Thus each factor-two slowdown reduces the leading error by
approximately four, consistent with a finite-size correction of order
:math:`(R/cT)^2`.

The instantaneous reaction torque is not generally minus the simultaneous
far-zone angular-momentum flux.  Their difference is the time derivative of
near-field, or Schott, angular momentum [Duviryak2022]_.  For the point term,

.. math::

   \boldsymbol\mu\mathbin\times\boldsymbol\mu^{(3)}
   ={d\over dt}\left(
   \boldsymbol\mu\mathbin\times\ddot{\boldsymbol\mu}\right)
   -\dot{\boldsymbol\mu}\mathbin\times\ddot{\boldsymbol\mu}.

Consequently, a quiet-boundary pulse can agree after integration even when
the two rates differ substantially during the pulse.  This distinction must
be retained when the oracle is later matched to an intrinsic spin equation.

Inertial covariant point-torque oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate_inertial_point_magnetic_dipole_self_torque_native`` provides the
next deliberately limited bridge.  Along an inertial worldline it promotes
the rest-frame point torque to the four-vector

.. math::

   N^\mu_{\rm RR}={\mu_0\over6\pi c^3}
   \epsilon^\mu{}_{\nu\rho\sigma}\,
   \mu^\nu {d^3\mu^\rho\over d\tau^3}{u^\sigma\over c}.

Its rest frame is exactly the three-vector law above.  Maintained tests boost
arbitrary, nonparallel moment and third-derivative vectors to a frame with
:math:`|\boldsymbol\beta|\simeq0.77` and recover the Lorentz transform of the
rest-frame torque.  The construction also gives

.. math::

   u_\mu N^\mu_{\rm RR}=0,
   \qquad
   \mu_\mu N^\mu_{\rm RR}=0.

Thus it is compatible with the spin supplementary condition and, when
:math:`\boldsymbol\mu` is proportional to spin, does not change spin
magnitude.

The word *inertial* in the function name is an important physics boundary,
not merely an implementation status.  For an accelerated dipole, Unruh's
reaction field contains Fermi--Walker derivatives plus additional terms
involving proper acceleration and its derivative [Unruh1999]_.  The inertial
formula must not be inserted into a flyby until those terms have been
translated into this project's conventions and tested independently.

Planar-acceleration comparator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate_unruh_planar_accelerated_dipole_torque_comparator_native`` records
the torque-relevant part of Unruh's Eq. (60) without applying it.  Let
:math:`A^\mu=du^\mu/d\tau`, define
:math:`\kappa^2=-A_\mu A^\mu/c^2`, and let :math:`D_{\rm FW}` denote the
Fermi--Walker derivative with respect to proper time.  After fixing the
overall sign and normalization with the finite-shell inertial limit, the
candidate torque is

.. math::

   N^\mu_{\rm U}={\mu_0\over6\pi c^3}
   \epsilon^\mu{}_{\nu\rho\sigma}\mu^\nu
   \left[D_{\rm FW}^3\mu^\rho
   +{3\over2}\kappa^2D_{\rm FW}\mu^\rho\right]
   {u^\sigma\over c}.

Unruh's reaction magnetic field has one further term proportional to
:math:`\mu^\rho f\dot f`.  It is parallel to the moment, so it makes exactly
zero contribution to :math:`\boldsymbol\mu\mathbin\times\boldsymbol B_{RR}`.
It is omitted only from this torque comparator, not declared absent from the
complete self-field.

The implementation requires callers to supply the first and third
Fermi--Walker moment derivatives explicitly.  It checks that velocity,
acceleration, moment, and both derivatives satisfy their required
orthogonality relations.  Tests establish exact recovery of the inertial
oracle, covariance of the acceleration term under a finite boost, invariance
of :math:`\kappa^2`, and orthogonality of the resulting torque to velocity and
moment.

This still does not authorize an applied accelerated torque.  Unruh derives
the field for acceleration confined to a plane and explicitly notes that the
result was not independently proved from the conventional radiated
energy--momentum calculation.  The reaction electric field, translational
force, nonplanar motion, moving-source balance, and reduction of order remain
separate gates.

Sampled reduction-of-order reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate_sampled_fermi_walker_magnetic_torque_reduction_native`` supplies
an independent reference for the next gate.  It differentiates seven or more
samples from a leading-order trajectory in which magnetic self-reaction is
absent.  This is the meaning of *reduction of order*: high derivatives inside
the already-small self-reaction correction are evaluated from the ordinary
non-self dynamics, instead of becoming new dynamical variables that admit
runaway solutions.

Let :math:`w=u/c` and let

.. math::

   P^\mu{}_\nu=\delta^\mu{}_\nu-w^\mu w_\nu

project a four-vector into the particle's instantaneous rest space.  For a
spatial magnetic moment, one Fermi--Walker derivative is

.. math::

   {D_{\rm FW}\mu\over D\tau}=P{d\mu\over d\tau}.

The implementation differentiates this expression twice more, including the
first and second derivatives of :math:`P`; it does not merely project the raw
third derivative once.  That distinction matters whenever the worldline is
accelerated because the instantaneous rest space itself then changes.

The centered form deliberately uses samples on both sides of the evaluation
event and is therefore a validation oracle only.  A separately named causal
form uses the newest seven-or-more accepted samples and no future state.  It
is still diagnostic: callers must ensure that rejected trial steps and any
self-reaction-corrected states never enter its history.  Both forms report the
scaled interpolation-matrix condition number and violations of velocity
normalization and moment orthogonality, so an apparently precise torque can
be rejected when its derivative reconstruction is ill-conditioned.

The maintained accelerated-worldline test uses hyperbolic motion with a
moment prescribed in a Fermi--Walker-transported basis.  The reconstructed
third derivative converges at fourth order under step halving.  This sampled
route is the reference answer for a later sparse analytical implementation;
it is not connected to the trajectory update.

The corresponding analytical route is now available as
``evaluate_potential_directional_magnetic_torque_reduction_native``.  The
leading RFS spin equation is first order, so its third spin derivative needs
only two derivatives of that equation's right-hand side.  Consequently, the
existing directional potential data through fourth derivative order are
sufficient; no fifth-potential derivative, enlarged retarded Taylor jet, or
additional root solve is introduced.  The routine then applies three
successive rest-space projections and passes the resulting first and third
Fermi--Walker derivatives to the planar Unruh comparator.

A maintained test integrates the same non-self RFS equations independently,
samples that trajectory, and compares the sampled and analytical reductions.
The two agree down to the finite-difference cancellation floor.  This closes
the algebraic reduction-of-order sub-gate, but not the physical limitations
of the underlying planar accelerated-dipole law.  The analytical result is
still passive and is not used by the trajectory update.

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

The finite-shell analytic ledger now closes at floating-point precision for a
prescribed harmonic moment, and its local expansion verifies the expected
shell-radius scaling of the reversible and radiation-sensitive pieces.  The
exact harmonic response independently closes mean self-torque work against
outward power, its point limit matches the Maxwell-stress sphere oracle, and
its low-frequency torque converges to the local shell series.  The first
complex-frequency pole-count and refined impulse-response tests reproduce the
exact-versus-truncated causality distinction.  Together these establish the
finite-size shell bookkeeping benchmark across the :math:`q\mu` angular and
:math:`\mu^2` energy channels.  A nonzero-boundary charge interval, explicit
intrinsic-particle matching, identification of the pure :math:`\mu^2` bound
contribution, and application to an archived flyby remain later acceptance
steps.

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

.. [Mansuripur2020] M. Mansuripur and P. K. Jakobsen, "Electromagnetic
   radiation and the self torque of an oscillating magnetic dipole,"
   *Proceedings of SPIE* **11462**, 114620W (2020),
   `doi:10.1117/12.2569137 <https://doi.org/10.1117/12.2569137>`_,
   `arXiv:2008.11264 <https://arxiv.org/abs/2008.11264>`_.

.. [Itoh1991] N. Itoh, "Radiation reaction due to magnetic dipole
   radiation," *Physical Review A* **43**, 1002 (1991),
   `doi:10.1103/PhysRevA.43.1002
   <https://doi.org/10.1103/PhysRevA.43.1002>`_.

.. [Unruh1999] W. G. Unruh, "Radiation reaction fields for an accelerated
   dipole for scalar and electromagnetic radiation," *Physical Review A*
   **59**, 131 (1999), `doi:10.1103/PhysRevA.59.131
   <https://doi.org/10.1103/PhysRevA.59.131>`_.

.. [StumpPollack1997] D. R. Stump and G. L. Pollack, "Magnetic dipole
   oscillations and radiation damping," *American Journal of Physics* **65**,
   81 (1997), `doi:10.1119/1.18523
   <https://doi.org/10.1119/1.18523>`_.

.. [Duviryak2022] A. Duviryak, "On the free rotation of a polarized
   spinning-top as a test of the correct radiation reaction torque,"
   `arXiv:2204.01519 <https://arxiv.org/abs/2204.01519>`_ (2022).
