High-gamma retarded-field numerics
==================================

Audit of ``8d42ba6`` on ``feature/high-gamma-numerics``, 2026-10-02.
The initial audit below records the design-decision boundary. Ben's
2026-10-02 decision retains the production quintic position, beta, and
beta-prime contract. Stage 1 is implemented for exact local kinematics and
analytic inertial boundaries; stored high-gamma integrator histories remain
stage 2. The implementation and validation limits follow the original audit.

Precision requirements
----------------------

Here :math:`u=\gamma\beta` is dimensionless spatial proper velocity, distinct
from the dimensional four-velocity used elsewhere in the solver.
For a finite massive source, :math:`\gamma=\sqrt{1+|u|^2}` and
:math:`d=1-|\beta|=1/[\gamma(\gamma+|u|)]`. With :math:`b=|u|/\gamma`,
:math:`\hat\beta=u/|u|`, and unit observation direction :math:`n`,
:math:`\kappa=d+b|n-\hat\beta|^2/2`. Handle :math:`u=0` separately.
All these expressions must propagate their Taylor derivatives.

Stabilizing kappa alone is insufficient. The velocity field also uses
:math:`1-|\beta|^2=1/\gamma^2` and :math:`n-\beta=(n-\hat\beta)+d\hat\beta`.
The longitudinal part of :math:`n-\hat\beta` can itself lose precision; in
coordinates aligned with :math:`\hat\beta`, evaluate
:math:`n_\parallel-1=-|n_\perp|^2/(1+n_\parallel)` near alignment.
At :math:`\theta=1/\gamma`, a longitudinal field component can be much smaller
than the vector norm, so reference tests must distinguish component errors,
absolute errors near zeros, and vector relative errors.

Audit inventory
---------------

* ``charge_potential_jet.py:193–240``: beta comes from the position
  polynomial, implicit Newton divides by ``-1+n.beta``, and the potential
  divides by ``1-n.beta``. The gradient and Hessian inherit cancellation.
  Projecting the scalar light-cone residual to zero preserves an existing
  root; it does not repair that root or its derivative algebra.
* ``charge_potential_jet_numba.py:146–218``: the same second-order jet
  cancellation, including Newton's derivative. This compiled kernel has
  no local small-kappa guard.
* ``charge_response_jet.py:178–184,228–261,300–317`` and
  ``charge_response_jet_numba.py:142–146,189–217,244–260,381–398``:
  scalar and jet kappa, root gradient, ``1-beta_squared``, ``n-beta``,
  potential derivatives, field derivatives, and diagnostic residuals all
  need attention. The compiled provider also rejects rounded beta squared
  at one before entering its jet kernel.
* ``retarded_fields.py:185–220,332–389,681–785``: extraction, cached
  histories, and quintic coefficients retain beta and beta prime, but not
  gamma or mechanical momentum. Position coefficient differentiation
  therefore cannot recover the speed deficit. Endpoint remainders subtract
  large positions and displacements; the ballistic special case removes
  polynomial noise but cannot restore the missing deficit. Append buffers
  and provisional trial histories must preserve any added representation.
* ``retarded_fields.py:704–714,1137–1339`` and
  ``retarded_dipole_numba_roots.py:76–89,200–355``: knot bracketing,
  termination checks, extrapolation, scalar residuals, and Newton's
  derivative use distance differences or ``1-n.beta``. False zero residuals
  can accept a wrong root. Adjacent floating-point time brackets bound
  representable time, not physical root accuracy. Rationalizing a difference
  of norms alone cannot restore lost position or time information.
* ``retarded_fields.py:1341–1436,1860–1937`` and
  ``exact_retarded_numba.py:110–172``: direct fields and potentials have
  the same cancellation and reject beta squared at one. The analytical
  segment-margin proof divides by ``1-beta_bound`` and falls back when the
  Bernstein velocity bound rounds to one; fixing only the field kernels
  leaves the provider unable to certify its derivative path.
* ``equations.py:552–559,1191–1290,4382–4515,4678–4689,4908–4917``:
  four-velocity reconstruction loses gamma from beta; velocity updates
  use position/time differences except at the on-shell boundary, where
  mechanical momentum is already available. Recoil updates reconstruct
  beta from momentum, then clamp it. Kahan summation improves summing
  squares but cannot recover a discarded speed deficit. Both beta and
  beta squared are clamped, so both clamps require replacement. Diagnostic
  gamma and optional reconciliation also depend on this reconstruction.
* ``source_kinematics.py:27–55,96–117``: conversion from four-kinematics
  rejects beta rounded to one, subtracts longitudinal acceleration terms,
  and differentiates rounded beta samples. A proper-velocity path needs
  compatible acceleration derivatives, not just an added scalar gamma.
* ``integration_runner.py:905,1163,3972,4869``,
  ``particle_initialization.py:239``, and
  ``input_output/bunch_initialization.py:177,213–220,286–291``:
  prehistory duration and closing-speed estimates use speed deficits with
  floors, initialization forms beta from gamma, and exact startup
  reconstruction derives gamma from beta. These can limit an integration
  before a corrected kernel is called.

Related active paths also need explicit scope treatment:

* ``retarded_potential_directional_jet.py:485–550,827–854``,
  ``dipole_hertz_jet.py:607–613``,
  ``dipole_hertz_jet_numba.py:495–562,712–766``,
  ``retarded_dipole_fields.py:875,932–938,1204–1210``, and
  ``retarded_dipole_numba_roots.py:493–502`` repeat the jet, gamma,
  invariant-distance, and kappa losses for charge or dipole sources.
* ``distances.py:158,191,279,328,521–563,737``,
  ``equations.py:909,1090,1107``, and
  ``vectorized_interactions.py:860–884,1274–1293`` retain cancellation
  and cutoffs in the chrono-matching force path. Factoring with
  ``1-projection**2`` still subtracts nearly equal numbers.
* Beta-derived gamma also occurs in ``magnetic_dipole.py``,
  ``finite_magnetic_source_force.py``,
  ``translating_magnetic_shell_kinematics.py``, ``full_dipole_response.py``,
  and ``jakobsen_pair.py``. ``full_dipole_radiation.py`` and
  ``reciprocal_radiation.py`` use cancellation-prone angular Jacobians.
  ``macroparticle_smearing.py:263`` has a separate beta clamp.
  ``diagnostics.py``, ``capture_diagnostics.py``, and
  ``medina_radiation_reaction.py:222`` contain beta-derived consistency
  checks that cannot diagnose arbitrarily large gamma reliably.
* ``lw_integrator/sweep_runner.py:316``,
  ``lw_integrator/testbed_runner.py:2051``, and optimization configuration,
  timestep, and stress helpers reconstruct beta or gamma by subtraction.
  In particular, ``optimization/charge_response_jet_uniform_stress.py:25–28``
  uses a cancellation-prone uniform-motion retarded root; it is unsuitable
  as the new high-precision oracle. No ``legacy/`` changes are proposed.

Design decision required before stable kappa
--------------------------------------------

The current contract defines position, beta, and acceleration as one quintic
worldline and its derivatives. At gamma :math:`10^9` and :math:`10^{12}`, uniform motion
produces identical binary64 beta and position coefficients, although the
speed deficits differ by six orders of magnitude. Ordinary trajectory
storage retains gamma and canonical momentum, but prepared histories and
standalone jet calls discard them. Canonical momentum must have its field
contribution removed before it supplies mechanical proper velocity.

Adding independently interpolated proper velocity to the old position
polynomial would generally make :math:`dx/d(ct)\ne u/\sqrt{1+|u|^2}`. The root's
implicit derivatives would then disagree with the potential and field jets.
A constant gamma override would also give incorrect accelerated derivatives.

Recommendation for review: retain the existing moderate-gamma representation
and define an additional high-gamma segment representation that carries
proper velocity and a separately resolved light-cone coordinate
(:math:`ct-x_\parallel`), with consistent position and acceleration derivatives.
Thread that representation through roots, preparation caches, trial overlays,
and all charge jets. Preserve beta-only calls as a compatibility path with
their existing precision limits. The representation and its transition rule
need agreement before implementation; no threshold is selected by this audit.

An alternative is a proper-velocity interpolant whose integral defines the
position worldline. That changes the interpolation model and root evaluation
cost, and requires its own comparison against existing moderate-gamma results.
Removing guards before either coherent representation is validated would
replace diagnosed failures with incorrect fields.

Stage 1: precise kernels and analytic boundaries
------------------------------------------------

The four potential/response jet modules accept optional
``source_proper_velocity``: dimensionless mechanical :math:`u=\gamma\beta`
at the already-selected root. Without it their original arithmetic and
small-kappa guards remain unchanged. This option is supplied only when the
caller knows the source kinematics; general integrator histories keep their
existing quintic representation and solver.

``core/precise_charge.py`` supplies common Python and strict Numba local
Taylor algebra. It uses a frame aligned with the supplied velocity and
projects transverse geometry with compensated cross products of the raw
proper velocity. Ordinary dot products against a rounded transverse basis
invent a transverse displacement for some exactly parallel rays, which
can dominate the physical cone at high gamma. The compensated projection
preserves those parallel inputs, including directions away from the axes.
The kernel
resolves :math:`R-L` as :math:`P^2/(R+L)` ahead of the source. Its denominator
is :math:`R\kappa=\delta R+(1-\delta)(R-L)` at the root. The local Taylor
expansion of the longitudinal deficit includes the supplied acceleration
and jerk; its first derivative is minus longitudinal acceleration. The
transverse velocity terms are retained in the invariant and denominator.
The longitudinal difference is :math:`n_\parallel-\beta_\parallel
=\delta-(R-L)/R`, and the speed invariant uses the resolved deficit.
Only zero separation or a nonpositive stable denominator is degenerate.

Uniform motion also admits a closed-form simultaneous-separation potential
and field. Its differentiated expression avoids subtracting separately
large implicit-root derivative terms at the beaming-cone shoulder. Potential
jets include ordinary gradients and Hessians, and response jets include
ordinary field gradients. The internal second-order field jets are used for
inertial directional responses; an accelerated second-order field response
would additionally require the next acceleration derivative.

``core/ballistic_charge.py`` solves the positive uniform-motion quadratic
with 90-digit standard-library Decimal arithmetic. It resolves longitudinal
light-cone differences before computing the retarded point and uses the
rationalized quadratic branch behind the source. Its result includes the
retarded separation, residual at the returned time, and an absolute time
error bound. That bound covers arithmetic and binary64 time conversion for
its documented domain; it does not cover uncertainty in supplied observer
events or mechanical states. Charge field/response results expose these
bounds where available, leaving them unspecified for quintic roots.

``core/analytic_charge_root.py`` defines a stateless callable interface for
an analytic source's resolved :math:`w=ct-\hat e\cdot x`, transverse position,
and proper velocity. Its bracketed solve uses a stable interval residual.
The caller supplies coordinate error bounds and a positive lower bound on
kappa throughout the bracket. The returned coordinate error includes those
uncertainties; a small scalar residual alone never certifies a physical root.
No stored light-cone history or segment representation is introduced.

The ballistic root keeps its longitudinal and transverse separation parts
independently. The precise local kernel accepts these resolved frame
components as an optional geometry input. Recombining them into a large lab
vector before the field call can erase the transverse displacement: in a
rotated, exactly parallel ballistic probe at gamma about :math:`10^{12}`,
that conversion produced a false transverse field roughly :math:`10^8`
times the physical field. Keeping the parts separate reduces this probe's
relative field error to approximately :math:`3\times10^{-16}`. Additional
Decimal tests cover the full boundary/provider path in every decade.

Inertial-prehistory preparation keeps the initial mechanical momentum,
anchor position, and anchor time as optional particle constants. Builders,
indexed views, append caches, trial overlays, and native checkpoints preserve
this analytic boundary. The boundary has a finite interval: it never fills a
missing future source history. Its stable root and precise charge kernels
apply only before the initial anchor. A requested compiled batch that contains
this metadata uses the Python analytic-boundary evaluator; the standalone
precise Numba kernels and the analytic response provider are also validated.

Resolved beta histories retain their original derivative reconstruction.
An explicit inertial prefix whose beta rounds to one instead has exactly zero
acceleration; accepted samples following it retain the original timelike
checks. This also prevents distant artificial seed knots from making the
short-interval derivative weights singular. It does not remove or change the
beta clamp in ``core/equations.py``.

Decimal references and achieved precision
-----------------------------------------

``scripts/generate_high_gamma_references.py`` uses only the standard library
and 90 Decimal digits. The 96-row JSON fixture in ``tests/unit/data/`` covers
12 gamma decades, four angles, and both directions. It derives the uniform
retarded point from the closed-form root and supplies potentials, fields,
potential gradients/Hessians, and electric-field gradients. Tests consume the
fixture directly and never execute the generator.

Additional 90-digit Decimal tests cover exactly parallel rays with a rotated
velocity in every decade, ahead of and behind the source. These references
use the exact supplied binary64 proper velocity, rather than an ideal angle.

Potential/field vector values and potential Hessians are checked at
:math:`2\times10^{-13}` relative norm tolerance; Python/Numba agreement is
checked at :math:`5\times10^{-15}`. At :math:`\theta=1/\gamma` ahead of the
source, the potential gradient and field gradient reach approximately
:math:`1.3\times10^{-8}` relative norm error against ideal-angle references.
Their fixture tolerance is :math:`2\times10^{-8}`; other angles use the tighter
value tolerance. The nearly vanishing longitudinal numerator is sensitive to
rounded inputs and its final small difference. Component-relative accuracy
near that zero is not claimed. Existing tests and tolerances are unchanged.

Fig. 3 initialization diagnosis
--------------------------------

The read-only errata 0.5 TeV drift case reproduces the initialization failure.
The initial proton light cone has roughly :math:`5.7\times10^7` mm retarded
separation. The maintained centered provider therefore chooses about 5,700 mm
for its relative derivative stencil, or about 19 ns for the time displacement.
The observer event at positive 19 ns requires a source event after the
initial history endpoint at zero. Negative-time stencils first require more
past coverage, but extending that coverage eight times cannot cover the
positive-time event. At the last attempts both knot endpoint residuals are
positive for the failing event, rather than opposite signs. This is a stencil
coverage problem, not a high-gamma rounding failure. No stencil-sizing or
future-extrapolation fix is included in stage 1.

Remaining stage 2 work
----------------------

The general quintic roots, position coefficients, reconstructed history beta,
segment-margin proofs, and integrator beta clamp still have the limits
identified in the audit. Dipole kernels and chrono matching are also unchanged.
Stage 2 may supply resolved light-cone history coordinates and gamma to the
local precise kernels, but it still needs a coherent acceleration/jerk
contract, accepted/trial history handling, root-error accounting, and margin
validation. No interpolation model, transition threshold, or stage 2 storage
has been implemented here. Exact kinematics cannot recover observer event
precision or source information already discarded before the kernel call.
