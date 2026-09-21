# Analytical reaction derivatives and time-controller status

**Current integration status, 10 September:** [higher-order stepping and the
adaptive connection](higher_order_pair_and_adaptive.md) supersede the pending
implementation items below. DOP853 state and reaction accounting, shared-time
adaptive acceptance, CLI restart and limited GUI controls are implemented.
Full post-arrival radiation accuracy is still being tested. The sections below
retain the earlier diagnostic results; they are not the current task list.

**Precision follow-up:** [preserved source-history arithmetic](preserved_source_history.md)
now connects to optional fresh-pair generation, general interval reconstruction
and v4 checkpoint restart. Both 128/256-step post-arrival runs complete; outgoing
momentum differs by 6.78282e-6 and dipole-squared reaction impulse by 2.81558e-1,
each relative to the fine-result norm. The next diagnostic keeps recorded source
times fixed and refines internal stepping. Radiation accuracy remains open.

## Source-history startup window follow-up

An explicit `FullDipoleHistory(..., inertial_until=0, startup_fit="one_sided")`
option prevents the first evolved derivative fits from mixing stationary past
with moving future samples. It waits for eleven accepted on-or-after-start knots;
there is no extrapolation. Startup history availability may therefore require
ten forward steps. Later windows retain the existing centered rule and all
accuracy/speed limits remain unchanged. History format v3 records the option;
v1/v2 restore their existing centered windows. Select this only in fresh
preparations, not by relabeling evolved checkpoints. It is not yet an automatic
default. New tests cover known-polynomial departures, publication and restart.

Fresh 128- and 256-step coupled runs both reach the post-arrival endpoint.
Their outgoing momentum differs by 3.10839e-5 relative, but dipole-squared
reaction impulse differs by 6.54542e-1 relative (fine-quantity norm denominator).
The stopping failure is repaired in these runs, not general radiation accuracy.
Fixed-observer tests find closely agreeing potentials/fields but strongly
disagreeing high time derivatives of the reconstructed response. A v3 checkpoint
also restarts with exactly matching CLI/API output and unchanged input.
Known-polynomial controls additionally demonstrate loss of small dipole changes
stored on a larger constant component, amplified by high differentiation orders.
Preserve such changes before rounding absolute samples; increasing fit precision
afterward cannot recover them. Stable fit arithmetic is also still required.
This history fit is distinct from the analytical reaction-derivative selector.
The reaction-local Taylor calculation should not be assumed to supply complete
endpoint derivatives of the coupled, re-solved trajectory without validating
the varying self-field solution as well.

The legacy COLD_START/chrono paths are not used by this nonlinear pair.
Legacy chrono estimates can differ substantially from the exact light-cone
solution in transverse relativistic geometry; their interpolation tolerance is
not a general retarded-time accuracy bound. The aneutronic shared-time branch
had cancellation/endpoint/secant fixes missing here. They have now been carried
over with regression tests and the existing 64-iteration budget retained. The
51 shared-time/controller tests pass; the nonlinear pair's adaptive-controller
connection is still a separate pending task.

## What is implemented

The maintained nonlinear pair now accepts `reaction_derivative_method="analytic"`.
This calculates derivatives directly through the full retarded charge/dipole
potential and the nonlinear equations of motion, without fitting neighbouring
trajectory samples. It supports moving source polynomials, not just the
stationary startup reference. The main state still stores $P=p+qA$.

`core/full_dipole_response.py` shares one Hertz/LW potential calculation between
ordinary responses and Taylor responses. `core/response_taylor.py` carries
four orders along observer motion together with the three observer derivatives
needed by the potential response. Its strict compiled products reuse the
previous order-three kernels; a Python fallback remains available without
Numba. These are derivative coefficients, not a spatial grid or sampled fields.

`core/momentum_center_derivatives.py` differentiates the same velocity-system
and force/spin-rate algebra used by `core/momentum_center.py`. It computes
local coefficients in kinetic momentum to avoid subtraction of small changes
in two large stored-momentum/potential terms. It does not change the integrated
state or introduce another production force law.

The full-spin coupled self-field model is unchanged: its field and first
spacetime gradient are solved together with the source motion. Higher
spacetime self-field derivatives remain omitted, and independent total
particle-plus-field conservation is still open. Analytical differentiation
does not remove that physical approximation.

## Provider and interface contract

- A length-time potential provider supplies `taylor_response(events)`, with
  `events.shape == (5,4)`. Row $n$ is the $n$th derivative divided by $n!$.
  Returned $A$, $\partial A$, $F$ and $\partial F$ arrays have a leading
  coefficient axis of length five. Values and shapes are checked.
- The native full-tensor history provider implements this contract. Boosted
  providers and trial local self-fields propagate it, including all derivative
  terms. Missing capability raises an error; no fit is silently substituted.
- C7 startup is required while a driven ramp is active. Evolved source joins
  with nonunique high derivatives are rejected. Internal knots of explicitly
  declared coasting prehistory have known derivatives and are permitted.
- `MomentumCenterParticle(..., reaction_mode="full_dipole_coupled",
  reaction_derivative_method="analytic")` needs no `reaction_window_ns`.
  Direct API defaults remain explicit/compatible; select analytic as shown.
- The CLI exposes `--reaction-derivatives analytic`. Fresh explicit coupled
  preparation selects it if no derivative selector was given. The GUI offers
  the same selection and a preserve option. Existing checkpoints retain their
  recorded method; changing an evolved checkpoint is rejected. C5 histories
  are not silently rewritten into C7 preparations.

## What the verification establishes

The maintained derivatives agree with the independent stationary startup
reference through all four orders. Additional checks cover a Lorentz-transformed
moving charge/full dipole through speed $0.99c$, a source with acceleration and
changing dipole, transformation of the full equation derivatives, and an
80-digit independent calculation with supplied self-fields. Six new core
tests include strict join/startup guards, checkpoint restart and real CLI/API
agreement. The affected pre-existing selection passed 65 tests.

A fresh 128-step-grid native run reached 106 accepted steps, $ct=1.03515625$
mm, including newly evolved source signals. Its local coupled solves passed.
It then rejected a source-position reconstruction error of
$1.3163\times10^{-8}$ mm against the unchanged $10^{-8}$ mm budget.
Four levels of bounded local step halving did not repair this history failure.
The rejected reconstructed interval was $ct=0.986328125$ to $0.99609375$ mm,
earlier than the attempted endpoint $ct=1.044921875$ mm. Thus halving only the
next step cannot be assumed to repair an already sampled history.

A fresh 256-step-grid run reached 207 accepted steps, $ct=1.0107421875$ mm,
then rejected $1.5525\times10^{-8}$ mm against the same budget. Its rejected
source interval also starts at $ct=0.986328125$ mm. These runs do not establish
post-arrival convergence; neither finished its planned $ct=1.25$ mm endpoint.

The direct 48-step maintained startup run agrees with the independent study
trajectory to $1.54\times10^{-17}$ relative kinetic spatial momentum and
$2.63\times10^{-10}$ relative dipole-squared impulse. Each denominator is the
maintained result's norm; the corresponding absolute differences are
$1.40\times10^{-20}$ and $7.47\times10^{-21}$ in normalized momentum units.
The maximum particle position difference is $1.36\times10^{-20}$ mm.
The study report archives the complete comparison and failed runs.
This is useful post-arrival evidence, not full trajectory-accuracy closure.
No source-position budget, self-consistency threshold, or velocity-domain
condition was relaxed. Full integration is not declared validated merely
because the derivative algebra and software interfaces pass tests.

## The earlier adaptive method

The earlier dipole work used **shared laboratory time with step doubling**:
both particles are compared at the same time, and one whole step is compared
with two half steps. Differences in physical position, mechanical momentum,
spin and accumulated diagnostics determine whether the step is accepted and
how large the next one should be. Rejected trial histories are not committed.

That machinery still exists in `core/shared_lab_time.py`,
`core/step_doubling.py` and `core/adaptive_pair_return.py`. The older Jakobsen
adapter also reuses it in `core/jakobsen_adaptive.py`. It has **not** been
connected to the new full-spin reaction pair.

The current nonlinear pair already puts both particles at the same laboratory
time, directly integrating laboratory-time rates. In length-time notation,

$$
\frac{dy}{dt}=\frac{c}{u^0}\frac{dy}{d\tau},\qquad x^0=ct.
$$

Here $u^0=dx^0/d\tau$ and the proper parameter $\tau$ is in length units.
This accomplishes the time-coordinate conversion without separately finding a
proper-time step for each particle. But the requested laboratory step is
fixed. `--max-step-halvings` only retries source-position failures; it is not
the step-doubling error controller and cannot guarantee recovery.

## Relationship to legacy adaptive timesteps

| Mechanism | What it controls | Used by the nonlinear reaction pair? |
| --- | --- | --- |
| Initial automatic timestep estimate | Chooses a starting step from speed, separation and requested run length | Not automatically imported from old bunch configurations |
| Legacy adaptive timestep | Shrinks steps after energy jumps or proximity triggers, with cooldown/recovery rules | No |
| Shared-time step doubling | Estimates numerical error by comparing whole and half steps at the same time | Shared time yes; adaptive acceptance/controller no |
| Chrono matching | How the older solver interpolates retarded source samples | No; the nonlinear provider solves the light cone in published polynomials |

These methods are compatible in purpose but are not interchangeable settings.
A real reaction-induced energy change must not be mistaken for numerical
error. The recommended connection is to reuse the step-doubling error scales
and controller, not to stack the old energy-jump heuristics on the new model.
It needs a full-spin adapter, sector-resolved reaction-increment checks,
transactional histories, and checkpointed controller state. In particular,
compare kinetic momentum rather than $P$, and physical spin rather than
blindly placing six tensor components in an older three-vector spin field.

## Pseudogrid readiness

A pseudogrid here means evolving a smaller active particle subset while
approximating the response of passive particles and using weighted source
representatives. It is not a Maxwell field mesh.

The existing charge/bunch implementation has selection, neighbour weights,
source aggregation, bounded pair bookkeeping and history-retention helpers
in `core/pseudo_grid.py` and `core/integration_runner.py`. This is reusable
infrastructure, but magnetic dipoles are explicitly rejected in that route
because passive-spin reconstruction is not implemented. The new nonlinear
API also explicitly requires exactly two particles/histories.

Before magnetic pseudogrid runs, implement and verify:

1. A direct small many-particle full-tensor calculation as the comparison
   reference, with summed causal potentials and per-particle local reaction.
2. Consistent passive momentum/spin updates and weighted dipole source history.
   Do not reuse scalar effective-charge weights as an unexamined rule for a
   particle's own nonlinear self-reaction.
3. Comparisons against all-active runs, including encounter observables and
   separate charge, interference and dipole reaction contributions.
4. Controller, checkpoint and rejected-trial bookkeeping for that state.

The next source-history investigation should check whether the newly available
analytical motion derivatives can supply the history's endpoint derivatives
directly. This may avoid inferring high derivatives from samples spanning a
rapid change in forcing. It is a proposed correction, not yet verified or
silently applied to existing histories.

The next practical sequence is source-history accuracy, shared-time adaptive
integration, then a small direct many-particle benchmark before pseudogrid
approximation. Neither old adaptivity nor old pseudogrid controls currently
enable those capabilities in the nonlinear pair by switching a flag.
