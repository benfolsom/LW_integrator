# Experimental full-dipole radiation reaction

## What is implemented

`full_dipole_rr` applies a particle's own regular electromagnetic field to
both its momentum and spin. It retains all six components of the dipole
tensor, including its electric components, with no expansion in spin.
It includes charge-squared, charge–dipole, and dipole-squared force and torque.
The default remains off. This is a research mode, not validated production
physics or an established cure for the strong-spin velocity failure.

Two choices define this **specific experimental model**:

1. Use the time-antisymmetric point-current self-field (half retarded minus
   half advanced). Exclude the singular time-symmetric part. Conservative
   finite-size terms and their renormalization have not been derived here.
2. Retain the existing dipole/spin relation, but allow the external-only
   action mass to exchange energy with the self-field. Record that exchange
   rather than force the old radiation-off mass relation to remain constant.

These choices are explicit, not a claim that the existing action uniquely
determines a complete finite-size magnetic particle. Independent full
particle-plus-field energy and angular-momentum conservation remains open.

## Regular self-field without sampling a singularity

Use Gaussian units with $c=1$, metric $(+---)$, and proper-time derivatives.
The source is the same current used by the full-dipole retarded provider:

$$j^\mu(x)=\int d\tau\,[q u^\mu\delta^4(x-z)
+\partial_\nu(D^{\mu\nu}\delta^4(x-z))].$$

The Green function includes the Gaussian $4\pi$ normalization:

$$G_R(t,\mathbf r)=\frac{\delta(t-r)-\delta(t+r)}{2r}
=-\frac{1}{4\pi}\int d\Omega\,\delta'(t-\mathbf n\cdot\mathbf r).$$

The last identity follows by integrating over the polar angle. It lets us
differentiate the finite difference analytically, without subtracting two
numerically divergent fields. Define the null covector
$\ell_\mu=(1,-\mathbf n)$, $\kappa=\ell_\mu u^\mu$, and
$L=\kappa^{-1}d/d\tau$. At the worldline, every null-plane source root is
the current event. The potential is

$$A_R^\mu=-\left\langle L\left(\frac{q u^\mu}{\kappa}\right)
+\ell_\nu L^2\left(\frac{D^{\mu\nu}}{\kappa}\right)\right\rangle_\Omega.$$

An observer derivative adds $\ell_\alpha L$. The implementation forms
$F_R^{\mu\nu}=\partial^\mu A_R^\nu-\partial^\nu A_R^\mu$ and
$\partial_\alpha F_R^{\mu\nu}$ by polynomial differentiation. It needs
proper-time Taylor coefficients of $u$ and $D$ through order four.
It integrates angles in the instantaneous velocity rest frame and transforms
the response back. This avoids narrow high-speed laboratory angular peaks;
it does not establish accuracy of high-speed trajectory derivatives.

`regular_self_field` implements this identity. Tests compare the charge limit
to LAD, static-location electric/magnetic moments to their cubic-derivative
self-fields, and a full electric-plus-magnetic example to the independent
retarded provider and its time-reversed advanced solution. Lorentz transforms,
the homogeneous Maxwell identity, and source-sector scaling are checked.

## Force, torque, and momentum storage

For the chosen regular field, the local proper force and torque are

$$R^\mu=qF_R^{\mu\nu}u_\nu
+\tfrac12 D^{\alpha\beta}\partial^\mu F_{R\alpha\beta},$$

$$T_R= (F_R\eta)D+D(F_R\eta)^T.$$

Splitting the self-field into its charge and dipole sources gives the three
reported sectors. The charge–dipole force includes both charge acting on its
dipole field and dipole acting on its charge field. No squared sector is
dropped, and no first-order-spin recoil is reused.

The stepper solves the full momentum-to-velocity relation with the total
regular field and gradient. It keeps the existing constitutive rule
$D=K(p\cdot u)S/p^2$, $K=gq/2$. Thus force and torque are included in the
spin constraint solve, rather than appended after velocity has been chosen.

Stored momentum remains $P=p+qA_{\rm external}$, where external means other
particles plus imposed potentials, **excluding the particle's own potential**.
Only the actual force/torque receives $F_R$. The potential-offset derivative
remains $q u\cdot\partial A_{\rm external}$. This intentional distinction is
handled in `evaluate_with_self_field`, not disguised as a consistent external
potential provider.

Define the external action-mass excess

$$I=p^2-m_0^2-KF_{{\rm external}\,\mu\nu}S^{\mu\nu}.$$

Its rate is computed from the actual momentum/spin update:

$$\dot I=2p\cdot\dot p-K[(u\cdot\partial F_{\rm external}):S
+F_{\rm external}:\dot S].$$

It is generally nonzero. The checkpoint integrates this rate so numerical
mass-balance error is $I_{\rm end}-I_{\rm start}-\int\dot I\,d\tau$, not
$I$ itself. This records an explicit constitutive extension; it is not an
independent proof of field-energy conservation. The legacy diagnostic name
`mass_constraint` carries $I$ in this experimental mode and `mass_law` states
its different meaning. Spin and timelike-velocity checks are unchanged.

## Reduction of order and causal operation

`ordinary_source_coefficients` constructs a short auxiliary trajectory using
the reaction-off equations on both sides of the current trial state. This
centered fit replaces the less accurate, one-sided default. It does not replace
or publish accepted history. A forward auxiliary observer still samples other
particles' **retarded, already published** histories; missing source data cause
an error, not extrapolation. Eleven proper-time samples over $[-w,w]$ fit a
degree-eight polynomial, whose coefficients through order four supply the
analytic self-field calculation. The legacy `backward` option samples $[-w,0]$.

The predictor integrates small state changes directly. It also reconstructs
small dipole changes from momentum and spin increments, avoiding subtraction
of two nearly equal dipole tensors before taking four derivatives. Temporal
velocity coefficients follow from $u\cdot u=1$; this changes derivative
evaluation, not the velocity law or accepted state. A declared zero-field
preparation interval uses exact coasting derivatives at and before its start.

New full-reaction pair preparations default to a C7 potential ramp: its first
seven derivatives join continuously to the constant intervals. Its beta-CDF
evaluation and factored derivatives avoid cancellation near the upper endpoint.
**C7 and the older C5 ramp are different applied preparations**, although each
still reaches the same final mutual potential. This is not an apples-to-apples
physical comparison with older C5 runs. The checkpoint stores the choice.

This reduces order in **reaction strength**, not spin or speed. The predictor
also gives an ordinary timelike-domain check. It therefore cannot be used to
step across an already invalid ordinary state, even if the hypothetical
unreduced self-field might supply another branch. There is no speed clipping.

The companion study also tests a **separate, study-only** coupled local-field
root. It feeds a linear spacetime expansion of the self-field back into the
source motion and solves for matching field and gradient at the current event.
It passed a short strong-case continuation beyond the earlier reaction-on
failure, but neglects higher spacetime derivatives of that self-field. It is
not the full unreduced problem and has not replaced `full_dipole_rr`. See the
study's `planning/velocity_reaction_followup_2026-09-09.md` for the retained
limitations, refinement results, and independent native post-arrival campaign.

The caller must choose a derivative window and test its resolution separately
from the integration step. ODE state tolerance and polynomial fit residual
are not bounds on fourth-derivative error. History joins, rapidly varying
startup, and strong reaction are important unresolved accuracy limitations.
The reference implementation is computationally expensive and is not compiled.

The first C5/backward-fit startup comparison had a severely window-dependent
dipole-squared impulse. The improved C7/centered calculation has step and window
changes of $3.84\times10^{-4}$ and $4.83\times10^{-4}$, respectively, in the
two-particle pre-arrival diagnostic. These use the Euclidean norm of both
particles' concatenated four-impulses divided by the finer-step or smaller-window
reference norm. The separate bound/radiation reconstruction below differs by
$1.21\times10^{-3}$, using its own impulse norm as denominator. This is progress,
not a universal error bound or validation across evolved-history joins.

## Matched source four-momentum diagnostic

For the same smooth $u,D$ used as both source and receiver, integration by parts
in the angular identity separates the force into outgoing radiation and a total
derivative. Write $C^\mu=qu^\mu/\kappa$,
$V^\mu=D^{\mu\nu}\ell_\nu/\kappa$ (not an electric field), and use Minkowski
contractions for dots. The dipole-squared part is

$$B_{DD}^\mu=-\left\langle\ell^\mu
 [V\cdot L^3V-(LV)\cdot L^2V]\right\rangle,$$

$$J_{DD}^\mu=-\left\langle\ell^\mu\kappa
 (L^2V)\cdot(L^2V)\right\rangle,\qquad
R_{DD}^\mu+J_{DD}^\mu+\dot B_{DD}^\mu=0.$$

$J$ is outward radiated four-momentum per proper time. $B$ is the matched
reversible momentum term, analogous to the charge Schott momentum; it is not
yet an independently measured finite-volume bound-field integral. Thus a
separate impulse calculation is

$$\Delta p_{DD}^\mu=-\int J_{DD}^\mu d\tau
-[B_{DD}^\mu]_{\rm start}^{\rm end}.$$

This uses at most third derivatives at endpoints and second derivatives in
the radiation integral, rather than integrating a fourth derivative directly.
The other sectors follow from

$$B_{qq}^\mu=\langle-qLC^\mu+\ell^\mu C\cdot LC\rangle,$$

$$B_{qD}^\mu=\langle-qL^2V^\mu-\ell^\mu
[V\cdot L^2C-C\cdot L^2V-(LV)\cdot LC]\rangle,$$

$$J_{qq}^\mu=-\langle\ell^\mu\kappa(LC)^2\rangle,\qquad
J_{qD}^\mu=-\langle2\ell^\mu\kappa(LC)\cdot L^2V\rangle.$$

In the charge limit $B_{qq}^\mu=-2q^2\dot u^\mu/3$. The implementation tests
this limit, the local balance identity, dipole radiation normalization, Lorentz
transformation, and an endpoint derivative on exact uniform-field motion.

`source_balance` returns $B,\dot B,J$ separately for each sector. Crucially,
the reduced-order stepper obtains its source derivatives on an **ordinary**
auxiliary trajectory, while the accepted trajectory has reaction feedback.
An ordinary derivative of $B$ need not equal its derivative along that accepted
trajectory. The diagnostic therefore does **not** replace the applied impulse,
prove exact reaction-on conservation, or complete angular-momentum accounting.
The study runner retains raw force integrals and reconstructed impulses under
different names. The native checkpoint ledger still records the applied force.

## Usage and checkpoints

Use the maintained experimental checkpoint command:

```sh
python -m lw_integrator.nonlinear_pair \
  --checkpoint prepared.json --output reaction.json \
  --radiation-reaction full_dipole_rr \
  --reaction-window-ns 0.0001 --reaction-derivatives centered \
  --step-ns 0.00001 --steps 32
```

The numbers are illustrative, not universal defaults. The window is **proper
time**; the integration step is **laboratory time**. Particle configuration
stores `reaction_mode`, `reaction_window_ns`, and `reaction_derivative_method`.
Omitting the derivative CLI flag preserves the checkpoint setting. Old
checkpoints without that field restore `backward`, and those without
`startup_smoothness` restore C5. New particle objects default to `centered`.
The separate
`dipole_reaction_ledger` records native impulse, native integrated spin torque,
each source sector, and the action-mass exchange in length-time mass-squared
units. No unmeasured bound momentum or far radiation is filled with zeros and
presented as a completed balance. The charge-only ledger is separate.

Prepare reaction-consistent endpoint velocities and dipoles. Arbitrarily
switching an already interacting checkpoint can change these instantaneous
quantities; the strict restore check will reject the mismatch. Starting with
a zero-field driven preparation avoids rewriting a pre-existing history.
Checkpoint failure remains atomic. No established CLI/GUI defaults change.

## References and remaining validation

- [Dirac (1938), Classical theory of radiating electrons](https://doi.org/10.1098/rspa.1938.0124): origin of the radiative-field prescription for charge.
- [Poisson (1999), An introduction to the Lorentz–Dirac equation](https://arxiv.org/abs/gr-qc/9912045): radiative-field construction and reduction of order. Its metric convention differs from ours.
- [Kazinski (2006), Radiation reaction for multipole moments](https://arxiv.org/abs/hep-th/0604168): extended-system effective models and additional internal dynamics. It is not a derivation of our chosen spin constitutive extension.
- [Milton et al. (2020), Self-force on moving electric and magnetic dipoles](https://arxiv.org/abs/2006.15375): dipole self-force and radiative mass exchange for constant translational velocity, not our arbitrary-acceleration model.

The angular full-current formula above is derived here from the stated Green
function identity; the references are not claimed to contain this exact code.
Remaining checks are post-arrival derivative accuracy on causal full-spin
trajectories, independent bound-field and angular-momentum accounting, and
validity of reduction in reaction strength. Repeating the strong-case comparison
with improved derivatives still fails earlier with reaction on. Its relative
change of source velocity or dipole reaches about $2.84$ (Euclidean/Frobenius
norm divided by the ordinary-source norm), outside a small-reaction regime.
This is evidence against using this approximation there, not evidence that
a complete physical reaction model must destabilize the system.
