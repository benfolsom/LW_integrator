# Higher-order stepping and shared-time adaptive integration

**10 September follow-through:** [internal error control](internal_error_control.md)
now separates integration accuracy from source-recording cadence, with native
API/CLI/GUI support and restart guards. The outer step-doubling adapter below
remains a distinct method; stacking the two controllers is rejected.

## What changes

The native pair now supports fixed DOP853 steps with preserved increments.
This addresses a verified mechanism by which RK4 startup error can become large
after repeated differentiation of the source history. It changes the numerical
step, not the full-spin coupled force model or stored momentum definition.

Use `initialize_pair(..., preserve_source_increments=True,
integration_method="dop853")` for a fresh preparation. DOP853 requires the
preserved-source path. The checkpoint records `integration_method`; absent means
the existing RK4 route. The ordinary checkpoint CLI accepts
`--integration-method dop853` for a compatible fresh preparation, preserves the
recorded method by default, and rejects changing methods on evolved data.
The checkpoint GUI exposes the same method selector and preserves the recorded
method by default. The historical RK4 source-precision behavior is retained.

`core/preserved_runge_kutta.py` uses the installed SciPy DOP853 coefficients for
the twelve main stages. The matching weights also integrate all applied forces,
torques, charge/interference/dipole reaction contributions and external action-mass
exchange. Updating only the state while leaving RK4 diagnostic weights behind
would be incorrect. Fixed stepping uses neither SciPy's adaptive controller nor
its dense-output history. No new package dependency or environment was added.

The [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)
describes the eighth-order method. This order does not automatically describe
the full coupled algorithm with reconstructed causal histories.

## Adaptive connection

`core/nonlinear_pair_adaptive.py` reuses the existing step-doubling norm and
step-size controller. It compares one full interval with two half steps and
retains the latter only if accepted. Both particles remain at the same lab time;
there is no extra proper-time root solve for this lab-time integrator.

The required `AdaptivePairSettings` scales are:

| Field | Quantity and units |
| --- | --- |
| `position_mm` | Particle positions, mm |
| `momentum_native` | Four-component kinetic momentum, native momentum units |
| `spin_native` | Three spin components in the kinetic-momentum rest frame, native spin units |
| `impulse_native` | Total and charge-squared, charge–dipole, dipole-squared impulses, native momentum units |
| `torque_native` | Corresponding integrated torque components, native spin units |
| `mass_exchange` | Integrated external action-mass exchange, amu squared |

Every scale is an existing `ErrorScale(absolute, relative)`. There are no newly
invented default physics tolerances. Bounds `minimum_interval_ns` and
`maximum_interval_ns` apply to the whole comparison; an accepted interval
contains two native half steps. Default trial budget is eight.

The adapter compares interval reaction increments from zeroed trial ledgers,
then adds accepted increments to the stored totals. Thus a tiny current impulse
is not inferred by subtracting two large accumulated totals. All trials are
private copies. Domain, causal-history and local-root failures propagate rather
than being mislabeled as ordinary truncation error.

The error estimate conservatively uses order one until full-history convergence
establishes another order. Its largest scaled component must not exceed one;
diagnostic groups use their own units/scales before entering the shared norm.
The next interval can shrink by up to a factor of two or grow by up to 1.25.
This is separate from the old energy-jump/proximity heuristics.

API entry points are `initialize_adaptive(native_payload, settings, width)` and
`advance_adaptive_interval(checkpoint)`. The latter returns a new checkpoint and
a report for every attempted interval. The checkpoint format is
`adaptive-full-spin-pair-v1`, containing `native_payload`, settings, the next
interval width and accepted/rejected counts. It is deliberately not loaded as
an ordinary fixed-step checkpoint. Supported reaction modes are off and the
full-dipole modes; charge_ll adaptive bookkeeping is not included in this adapter.

## Command line and restart

Create a settings JSON from `dataclasses.asdict(AdaptivePairSettings(...))`, then:

```sh
python -m lw_integrator.nonlinear_pair_adaptive \
  --checkpoint native.json --settings settings.json \
  --initial-interval-ns WIDTH --intervals 10 --output adaptive.json
python -m lw_integrator.nonlinear_pair_adaptive \
  --checkpoint adaptive.json --intervals 10 --output continued.json
```

Use the project checkout with the compatible shared Python environment; do not
install it editable into that environment. Outputs must be new paths. Each
accepted interval is saved atomically, so a later failure retains the last
accepted result. Resume retains its settings; it does not accept replacements.
Accuracy refusal at the minimum interval or trial budget reports all attempted
widths and scaled errors. Invalid/nonfinite accumulated ledgers are rejected,
not silently discarded while forming interval increments.
The checkpoint GUI can resume a prepared adaptive checkpoint: choose
`adaptive_resume`, leave the timestep blank and keep the physics/method selectors
at `preserve`. The count then means adaptive intervals, not native half steps.
Preparing the explicit tolerance settings remains an API/CLI task; the GUI does
not invent physical tolerances or convert ordinary checkpoints automatically.

## Validation limits

The initial affected higher-order selection passes 81 tests, including legacy
paths and matched reaction quadrature. The 128-step DOP853 trajectory completes
the previous post-arrival endpoint; the 256-step run also completed. The relative
outgoing spatial momentum difference is 2.38769e-6, but the dipole-squared
reaction impulse difference is 4.26353e-1, each using the fine quantity's norm
across both particles. Completion therefore did not establish radiation accuracy. Adaptive tests
cover physical spin conversion, rejected trials, coupled restart, explicit
refusal reports, invalid ledgers and CLI checkpoint preservation. A real coupled CLI startup completes eight adaptive
intervals and rejects two trial widths before accepting a shorter eighth
interval. A post-arrival continuation initially reaches its minimum timestep
without meeting the dipole-squared impulse tolerance. Allowing smaller timesteps
with unchanged accuracy scales accepts an interval and gives exact API/CLI
checkpoint agreement. These are local acceptance checks, not a tolerance
recommendation or convergence result for already-recorded histories.
The following continuation is rejected by the history-fit conditioning limit:
the mixture of coarse old samples and newly shortened steps makes the derivative
fit too sensitive to numerical error. Sustained post-arrival adaptive operation
therefore remains blocked; this adapter is not yet a generally validated run mode.
The final adaptive/controller/CLI/GUI selection passes 54 tests. Software tests
do not override this real-trajectory refusal.
Local step acceptance cannot certify derivatives in already-published histories,
establish independent conservation, or enable magnetic pseudogrids.
