# Experimental full-dipole source history

`core.full_dipole_history.FullDipoleHistory` is a maintained, experimental
interpolation component. It does not select a force law or enable a new
CLI/GUI simulation mode. Nonlinear magnetic self-reaction is not implemented
by this component.

## Inputs and readiness

Supply accepted times, positions, **instantaneous** velocities, and full
antisymmetric proper-time dipole tensors. Do not pass preceding-step average
velocities or replace the tensor with a normalized rest-spin vector.

Length/time units must be consistent, and `speed_limit` is explicit. For
mm/ns geometry it is the existing `C_MMNS` constant; dimensionless study
fixtures use 1. No dipole-unit conversion is performed here.

Eleven accepted samples determine a knot's derivatives. Five samples on
either side are required, so the last five intervals are not yet published.
The default degree-ten fit uses all eleven samples; degree eight is retained
only for the documented accuracy comparison. Old published intervals remain
unchanged after an append. An unavailable retarded source time must cause
step rejection or a wait for accepted data, never extrapolation.

## Position consistency and error budget

The default constructs a velocity polynomial and integrates it for position.
This avoids encoding roundoff in absolute endpoint positions as large higher
derivatives. It matches velocity through its third derivative, so position
joins agree through the fourth derivative. Dipole joins agree through the
third derivative.

Every reconstructed endpoint is checked against the accepted position.
`position_tolerance` is an **absolute length allowance supplied by the caller**;
its default is zero, with only a small floating-point allowance added. A failed
check rejects the candidate without changing published history. The measured
`position_error` is stored on every segment. A nonzero allowance requires an
error budget appropriate to the experiment's separation and field variation
scales; a number suitable for a dimensionless test is not a native-unit default.

`integrate_velocity=False` retains endpoint-position interpolation as a
diagnostic comparison. Neither method may accept a source interval whose
polynomial speed bound reaches the supplied speed limit.

## Restart and scope

`to_checkpoint_payload()` and `from_checkpoint_payload()` preserve the accepted
samples, interpolation settings and published prefix. This is a history-only
payload: the owning stepper must also save particle states and its configuration.

This is an immutable reference implementation, not the optimized growable
history backend. It deliberately copies accepted arrays. It is suitable for
small reference tests, not yet the intended many-particle throughput path.
Smooth joins alone are insufficient validation: derivative accuracy, source
position error, retarded availability, and physical energy/momentum accounting
remain distinct checks. Study evidence is recorded in the companion study
repository's `planning/shared_derivative_gate_2026-09-09.md` report.
