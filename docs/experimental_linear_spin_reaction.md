# Experimental magnetic self-reaction that changes particle motion

This is a first implementation, not a complete magnetic radiation-reaction
model. It adds the already derived **first-order-in-spin** force to the live
one-particle-per-role stepping path. The mode is off by default. The
existing Medina charge reaction is neither replaced nor applied twice.

## What is implemented

The starting formula is Jakobsen's projected self-force, Eqs. (19) and (20a)
of [Spin and Susceptibility Effects of Electromagnetic Self-Force in Effective
Field Theory](https://arxiv.org/html/2311.04151v2). Its additional rest-frame
self-torque vanishes at the retained order. This statement does not extend
to terms quadratic in the magnetic moment or to a resolved finite shell.

The code obtains the required acceleration derivatives from the existing
Rafelski--Formanek--Steinmetz (RFS) non-self dynamics. “Non-self” means the
motion driven by other particles and prescribed fields before adding the
particle's own radiation reaction. Using those derivatives avoids treating
higher derivatives as new independent initial conditions.

The RFS force model is not automatically the non-self model used in Jakobsen's
paper. The combination here is therefore explicitly an **experimental
RFS-based reduction**. Reusing a published self-force formula does not prove
the consistency of the combined dynamics.

There is a further ordering distinction. The published force expression is
first order in spin, but substituting the full spin-dependent RFS acceleration
and spin evolution can generate **partial higher-order terms**. This code does
not re-expand that substitution and discard every term beyond first order.
Those implicit contributions are not the complete missing magnetic-moment-
squared self-force. An offline fixed-event spin-scaling check of the coupled
10 pm, beta 0.8 example finds an even-under-spin-reversal force about 0.83% of
the odd part for the electron. It decreases in proportion to spin magnitude
relative to the odd part, consistent with a quadratic contribution. This is
a model-ordering diagnostic, not a measured physical error or a validation of
those higher-order terms. A strictly first-order reduction remains a separate
comparison before promotion beyond experimental use.

The implementation uses an analytical potential-derivative calculation when
available. At a source-history boundary, it can instead estimate derivatives
from earlier accepted samples. No future accepted sample is accessed.
Unbounded uniform prescribed fields now provide exact analytical derivatives,
including their mixed effects with retarded source fields. Bounded or
nonuniform prescribed fields and the independent causal C5/local dipole source
histories still use the earlier-sample route. The legacy analytical
dipole provider must not be substituted for the source history actually
driving the particle.

## How one step changes

First perform the ordinary force and Medina update. Evaluate the additional
spin-dependent four-force at the accepted start of the step. With proper-time
increment $h$ and mechanical momentum $\mathbf p$, apply

$$\Delta\mathbf p=h\mathbf F_{\mathrm{spin}},\qquad
E_{\mathrm{new}}=c\sqrt{m^2c^2+|\mathbf p+\Delta\mathbf p|^2}.$$

The extra position/time drift uses the average of the pre- and post-impulse
velocities, consistently with the underlying endpoint-average step. The
ordinary endpoint potential is recomputed after the pair is synchronized.
The added force integration is **first order in step size**; the name of the
underlying second-order update is not a second-order accuracy claim for the
combined experimental method. The controller compares one full step with
two half steps using its conservative first-order error model.

When velocity changes, the spin four-vector must remain orthogonal to it.
We use a rotation-free Lorentz transformation from the old velocity $u$ to
the new velocity $v$:

$$S_{\mathrm{new}}=S-
\frac{S\mathbin{\cdot}v}{c^2+u\mathbin{\cdot}v}(u+v).$$

Here the dot product has signature $(+---)$. This preserves both the spin
length and its orthogonality to the new velocity. It is transport caused by
the change of motion, not an invented magnetic self-torque. Tests compare its
small-step sign with the existing RFS transport and check boosted frames.

## Records and restart

Every evaluated step has a separate record containing the force, proper-time
increment, route used, actual kinetic-energy change and the energy impulse
suggested by the temporal force component. Their difference is an additional
numerical energy adjustment, not radiated energy. The adaptive comparison
includes both the new work and this adjustment.

Accepted records retain lifetime signed/absolute work and energy adjustments,
plus the integrated four-impulse. Detailed recent records are bounded to
avoid unbounded checkpoint growth. Medina work and force memory remain in
their original accounts. The older mass-consistency adjustment alone is not
the full adjustment ledger when this mode is active.

Full-step trials and rejected trials do not update accepted history. A
second half step receives only its private earlier-half history. Both
particles, proper-time clocks and cumulative records are published together
and stored in the normal accepted-pair checkpoint. A test interrupts a real
Medina-on run, reloads it from disk and compares it with an uninterrupted run.

## Selection and limits

Select `magnetic_dipole.intrinsic_spin_self_reaction_mode` as
`experimental_linear_spin` in JSON, or pass
`--intrinsic-spin-self-reaction-mode experimental_linear_spin` on the CLI.
The GUI label is “Experimental: first-order spin recoil”. Selection also
requires both RFS spin precession and Stern--Gerlach force to be enabled,
`second_order_start_taylor_endpoint`, the
checkpointed exact-pair adaptive controller and its existing startup rules.
It is not implemented for the ordinary fixed-step or many-particle paths.

- Charge reaction may be `medina_lad` or explicitly `off` for comparison.
- Zero charge or zero physical spin gives zero added force. For a neutral
  magnetic particle this does **not** imply zero physical self-radiation:
  the missing magnetic-moment-squared contribution is outside this mode.
- Charged spin with zero gyromagnetic factor is explicitly unsupported by
  the present derivative reduction; it is not silently treated as spinless.
- If an analytical derivative is unavailable before enough accepted samples
  exist, the record says `warmup_no_force`. This explicitly omits the early
  impulse. Refinement must assess that startup approximation; it is not a
  complete reaction history from the first sample.
- Once enough samples exist, an ill-conditioned derivative estimate stops
  the run rather than becoming a zero force. A force exceeding 10% of the
  recent non-self force, measured in the instantaneous rest frame, also
  stops the experiment. That is a perturbative-use guard, not a proven
  physical validity threshold, and it does not clip the force.
- Applied recoil invalidates any flag claiming the stored non-self start
  acceleration is the complete acceleration. Existing causal source-history
  fallback remains responsible for reconstructing complete motion.

Pure magnetic-moment-squared recoil, its torque, finite-size matching and a
complete energy/momentum balance remain open. The shell comparison is still
an independent reference calculation; passing software tests is not a
substitute for that physical validation.

See [the Medina-compatible timestep correction](moment_boundary_checkpoint_diagnostic.md)
for the separate flyby experiment. It tests a numerical force-integration
correction while general magnetic self-reaction is off, and should not be
confused with the new force described here.

## Initial charge history in high-speed tests

A short curved first step can expose a separate charge-only problem. If the
preceding coasting history has a very long last interval, reconstructing its
endpoint acceleration from the new step can make the interpolated velocity
exceed light speed inside that old interval, although all stored velocities
are valid. This was reproduced with recoil both off and on at beta 0.99.

The public checkpointed-pair runner now uses the existing geometrically tapered
coasting times even when the dipole source is off. Intervals become gradually
shorter toward the first live half step. The causal local-fit source keeps
its separate physical-window startup grid. The fix changes initial sampling,
not the force equation, and does not establish causal interpolation for every
possible future step-size change. Tests retain the failing sparse-history
example and check the generated-history route at beta 0.99 and 0.9999.

In strong-field Medina benchmarks, the charge-force derivative also needs a
previous force sample. The study supplies that sample from the known preceding
uniform-field orbit as explicit benchmark data. The default inertial prefix
does not invent a force history or silently prime Medina.

## Exact uniform-field derivatives

For an unbounded uniform prescribed field, use the linear potential
$\phi=-\mathbf E\cdot\mathbf x$ and
$\mathbf A=(\mathbf B\times\mathbf x)/2$. The potential's second and higher
coordinate derivatives vanish exactly. This provides the non-self motion's
derivatives from the first step, without sampled-history startup omissions.
It does not mean that the trajectory's acceleration derivatives vanish: the
changing velocity and spin still enter them.

When a retarded source also contributes, both fields enter the leading
acceleration used to differentiate the potential along the trajectory. This
retains mixed source/external terms. Adding two separately reduced self-forces
would generally be wrong. The prescribed potential is used for derivatives;
it does not silently change the canonical momentum convention of the stepper.

The provider rejects all enabled hard-window and magnetic-gradient configs,
even if the particle happens to be far from a window boundary. Those cases
keep the existing earlier-sample route. Tests compare the uniform case with
exact Lorentz-motion derivatives through beta 0.9999, compare a mixed
source/external trajectory with independently sampled derivatives, and check
that actual pair feedback uses the analytical route without startup omissions.

The [independent uniform-orbit and spin-transport check](uniform_orbit_spin_reference.md)
gives the elementary-rotation reference used to test the derivatives. Recoil
spin transport is evaluated directly in rest-frame components, avoiding the
precision loss of a boost to the lab and back at high speed. This is the same
rotation-free four-dimensional transport, not a newly added self-torque.

## A stationary-source boundary switch found by the coupled test

The coupled-source check exposed a small ordinary-force error that matters
more when differentiated. Near a source-history knot, the analytical charge
provider deliberately falls back to numerical differences if its comparison
stencil could cross the knot. For a stationary source represented by identical
constant polynomials on both sides, that switch is unnecessary. In the saved
80%-of-light-speed check it changes one acceleration by about 5.08e-8 relative;
the subsequent sampled self-force differs by about 1.1% at that resolution.

The provider now extends its smooth-region certificate through adjacent
segments only if every nonconstant coefficient is exactly zero and the
constant positions are exactly equal. For a stationary source, shifting an
observer coordinate by a distance $h$ moves the retarded time by at most $h/c$.
The whole resulting time interval must remain inside that certified stationary
span. This is not a tolerance that treats small motion as zero. An arbitrarily
small nonzero coefficient stops the extension. Exact-knot, missing-history,
moving-source and curved-source protections remain in place.

This narrow correction does not solve general calculation-method switches for
moving sources, or validate the complete sampled self-force. The five startup
half-step omissions and missing magnetic-moment-squared terms remain explicit.
