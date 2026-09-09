# Experimental nonlinear reciprocal pair

`core.momentum_center_pair` connects the nonlinear ordinary interaction model
to the shared-derivative source histories. Both particles produce charge and
full dipole potentials, respond to the other's accepted past, and advance in
the same lab-time step. This is a separate Python API, not a CLI/GUI default.

“Nonlinear” means the momentum–velocity relation is solved without discarding
terms quadratic or higher in spin. It does not mean the model contains every
spatial multipole of an extended particle, or its complete radiation reaction.
Only `reaction_mode="off"` is accepted. The older first-order Jakobsen pair
remains available under its own model and checkpoint format.

## State and unit contract

The 14 entries are lab time in ns, three positions in mm, four stored momentum
components in native momentum units, and six antisymmetric spin-tensor entries
in native angular-momentum units. Tensor component order is 01, 02, 03, 12, 13,
23. Rest spin at initialization is specified in the **momentum rest frame**.
Stored momentum is physical momentum plus the potential contribution; these
state coordinates are not asserted to have canonical Poisson brackets.

The internal algebra in `core.momentum_center` uses length-time coordinates
and Gaussian units with $c=1$. Let a tilde denote its variables and let
$c=299.792458\,\mathrm{mm/ns}$. The adapter uses

$$\widetilde{x}^0=ct,\qquad \widetilde{q}=q/c,\qquad
\widetilde{P}=P/c,\qquad \widetilde{S}=S/c.$$

$$\widetilde{A}=A/c,\qquad \widetilde{F}=F/c,\qquad
\widetilde{D}=D/c.$$

Both charge and potential are rescaled. Rescaling only one would give the
wrong normalization for a particle that is both source and observer. Mass
remains in amu; source positions remain in mm. All potential/field derivative
indices refer to $(ct,x,y,z)$, even though the provider accepts time in ns.

$$\widetilde{p}=\widetilde{P}-\widetilde{q}\widetilde{A},\qquad
\widetilde{p}^{,2}=m_0^2+(g\widetilde{q}/2)
\widetilde{F}_{\mu\nu}\widetilde{S}^{\mu\nu},\qquad
\widetilde{S}^{\mu\nu}\widetilde{p}_{\nu}=0.$$

$$\widetilde{D}^{\mu\nu}=k\widetilde{S}^{\mu\nu},\qquad
k=\frac{g\widetilde{q}}{2}
\frac{\widetilde{p}\cdot\widetilde{u}}{\widetilde{p}^{,2}}.$$

The metric is $(+---)$ and $\widetilde{u}^{,2}=1$. The solver rejects a
non-timelike momentum or velocity branch. It does not project the state,
silently change models, or interpret a domain failure as spin flipping.

## API and accepted past

- `MomentumCenterParticle(charge_native, mass_amu, g, reaction_mode="off")`
  specifies this model explicitly.
- `initial_state_native(event, momentum_direction, rest_spin_native, particle,
  provider)` prepares a 14-component state in a supplied potential.
- `dynamics_native(state, particle, provider)` returns its lab-time derivative
  and diagnostics. Native momentum, velocity and dipole values are named
  explicitly. The nested `length_time` diagnostics use the internal units
  above, including constraint residuals; they are not energy errors in meV.
- `FullDipoleProvider(history, charge_native)` supplies native potentials and
  analytical derivatives from published intervals only.
- `initialize_pair(particles, states, histories)` validates and packages two
  accepted states with their prepared native-unit histories. The last accepted
  time, position, instantaneous velocity and proper dipole must agree.
- `advance_pair(payload, width_ns, steps=1)` returns a new checkpoint and
  accepted-end diagnostics. The checkpoint can be serialized with JSON and
  supplied directly to the next call.

Initialization deliberately does **not** invent a self-consistent interacting
past. A caller must prepare it or explicitly label a prescribed startup. The
unit-test coasting past is prescribed; its initial mismatch is not a solved
warm-up problem. The historical five-interval publication delay still applies.

Every RK4 stage uses the same frozen pair of source histories. Both candidate
states and new histories must succeed before either is published. Any failure
leaves the supplied checkpoint unchanged. An unavailable retarded time is an
error, never extrapolated; reducing a step may not repair an insufficient past.
RK4 is not claimed to be symplectic and this API does not yet adapt its step.

The version tag is `experimental_momentum_center_pair_v1`, with an explicit
native-unit tag. Jakobsen's 11-component states cannot be loaded here.

## Validation and remaining work

Core tests cover native free motion, charge normalization, stationary/moving
magnetic agreement with the established provider, full-tensor unit conversion,
short charged reciprocal evolution, JSON restart equality, and failed-step
non-mutation. The companion study compares native trajectories with its
retained nonlinear equations for twelve spin/gyromagnetic-factor combinations.
Migration agreement is a regression check, not an independent proof of the
underlying physical model.

The new full-tensor source response reuses analytical differentiation machinery
but **not** the compiled sparse magnetic-source kernel. Earlier optimizations
remain active on their existing compatible paths. Benchmark and specialize the
full-tensor response after a matched post-arrival trajectory check, not by
substituting the old constrained source definition.

Next: reproduce the study's smooth-start, evolved-source reciprocal trajectory
through this native API; check step refinement and errors after newly evolved
source data reach the other particle. Then assess particle-plus-field momentum
and angular momentum. Neither those balances nor nonlinear charge–dipole and
dipole-squared self-radiation are closed by these integration tests.

Derivation, references and prior evidence are retained in the companion study
repository's `planning/finite_spin_primary_implementation_plan_2026-09-09.md`
and `planning/nonlinear_spin_prototypes_2026-09-09.md`.
