# Experimental matched Jakobsen core API

This API evolves one responding particle in a supplied electromagnetic
potential. Its ordinary force, spin evolution and stored momentum belong to
the same first-order-in-spin model. It is **not** a new CLI/GUI setting and is
not wired into the legacy coupled-pair runner. RFS defaults are unchanged.

## Physical scope

The implementation follows the intrinsic-spin, no-susceptibility part of
[Jakobsen, arXiv:2311.04151v2](https://arxiv.org/html/2311.04151v2), ordinary
equations (6)–(8) and reaction equations (19)–(20), with the canonical momentum
checked against independent constrained-action variation in the study repo.
It retains first order in physical spin and first order in reduced radiation
reaction. It is not a complete finite-size or moment-squared radiation model.

`core.jakobsen` uses physical spin angular momentum in native amu/mm/ns
units, **not** unit polarization. The stored four-momentum includes mechanical
momentum, the usual charge-potential offset, and this model's additional spin
offset. Never apply that offset to an RFS checkpoint.

Velocity is reconstructed from spatial stored momentum through first spin
order. The temporal stored momentum is a separate consistency diagnostic;
the code does not project away its error. Terms quadratic in spin are not
controlled, including errors of that order in the inverse momentum mapping.

## Entry points and state

- `JakobsenParticle(charge_native, mass_amu, g, reaction_mode="off")`.
- `initial_canonical_state(...)`, `canonical_rhs(...)`, `midpoint_step(...)`
  in `core.jakobsen_step`.
- `checkpoint(...)`, `restore(...)`, `integrate(...)` in
  `core.jakobsen_adaptive`.

The eleven state values are lab time, three positions, four canonical momentum
components, and three physical rest-frame spin components. Trial steps return
new arrays; rejected states are not written into accepted checkpoints.

The provider is called as `provider(time_ns, position_mm)` and returns
`(A, partial_A, F, partial_F)`. Coordinates for derivatives are $(ct,x,y,z)$.
The layouts are `(4,)`, `(4,4)`, `(4,4)`, `(4,4,4)`, respectively; the derivative
index comes first. All values use the existing native scaled-Gaussian units.

For `reaction_mode="experimental_linear_spin"`, the provider must additionally
implement `gradient_proper_rate(time_ns, position_mm, four_velocity_mm_ns)`.
It returns the proper-time derivative **of the field gradient**, shape
`(4,4,4)`. An absent derivative is an error, not a zero. Where a retarded source
cannot supply a derivative across a history boundary, the caller must stop or
choose a separately validated fallback; this adapter does not invent one.

Checkpoint provider IDs must encode the potential and its parameters/history
identity. The adapter cannot verify a caller-supplied name against arbitrary
provider code. JSON checkpoint data here are not the existing runner's
checkpoint format. There is no implicit conversion or cross-model resume.

## Radiation accounting

The local coupling uses [Medina's reduced-order charge reaction](https://doi.org/10.1088/0305-4470/39/14/021)
and the existing independently tested intrinsic-spin self-force kernel.
The ordinary force derivative includes the effect of spin acceleration on
the charge force, without inserting spin corrections inside terms already
proportional to spin. Local complex-step differentiation evaluates analytical
algebra; it does not sample displaced physical-space fields or future history.

The leading charge reaction enters the ordinary spin transport and the
derivative of the spin momentum offset. The spin-dependent charge reaction
and explicit intrinsic-spin reaction are each added once. The returned
`radiative_balance_correction` is **not an additional mechanical force**.
It explains why mechanical reaction work alone need not equal the change of
canonical particle energy. It must remain separate in conservation accounting.

## Optimization compatibility and validation limits

The dense and sparse compiled analytical dipole providers now both supply the
required data. Request `include_partial_a=True` from the sparse history-facing
provider to obtain the first potential derivative. Request
`observer_four_velocity_mm_ns=u` for its proper-time contraction. Both are
optional; the default 34-output numerical response is unchanged. Only six
additional Hertz coefficients are required (150 instead of 144). The
canonical stepper needs the small full derivative map because it reconstructs
velocity after summing source potentials; returning only a velocity-dependent
contraction would require repeating that source evaluation.

### Experimental reciprocal adapter

`core.jakobsen_pair.initialize_pair` and `advance_pair` provide a separately
checkpointed, fixed-shared-lab-time two-particle experiment. Both roles respond
to the same accepted retarded histories. No future source extrapolation is
permitted. Accepted position intervals are quintic with instantaneous endpoint
derivatives; spin intervals are cubic with endpoint spin rates. Appending a
new state does not change any accepted polynomial. The complete source
interval is checked for a timelike speed bound.

This adapter integrates the existing proper-time canonical equation using
fourth-order lab-time Runge–Kutta. It does not replace the midpoint API or
production pair controller. The cold-start coasting interval keeps its own
left-hand derivatives, separate from the initial interacting right-hand
derivatives. Measurement windows after a mutually evolved warm-up avoid
confusing the startup wave with encounter accuracy.

Pair reaction is deliberately rejected: the required higher source-derivative
contract has not yet been validated for this interval representation. The
potential-derivative addition alone does not enable pair radiation reaction.
The pair checkpoint stores accepted states and frozen histories and is not
interchangeable with CLI/GUI checkpoints. Force-integrated particle momentum
is a bookkeeping diagnostic, not total particle-plus-field conservation.

Local force tests, independent ordinary-action comparisons, source-current
variation checks, pulse/loop/passing-charge refinement, and JSON restart tests
are recorded in the study repo's
`planning/matched_core_campaign_2026-09-08.md`. Radiation trajectory references
share the local reaction kernel and therefore test stepping and momentum
mapping, not independent radiation physics. Prescribed-source tests do not
establish reciprocal two-body conservation. Existing strict-backend and one
slow flux-test failure remain recorded rather than relaxed.
