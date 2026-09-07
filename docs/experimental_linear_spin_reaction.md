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

The implementation uses an analytical potential-derivative calculation when
available. At a source-history boundary, it can instead estimate derivatives
from earlier accepted samples. No future accepted sample is accessed.
Prescribed external fields and the independent causal C5/local dipole source
histories currently use that earlier-sample route. The legacy analytical
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
