# Choosing a dipole history drift budget

The budget limits how far the connected magnetic-moment curve may differ from
the moment calculated by the particle integrator at a saved endpoint. It is
a reconstruction check, not a physical loss of moment or a radiation-error bound.

For each particle, fresh connected histories without an existing budget use

$B_i = r\,\lVert D_{i,\mathrm{initial}}\rVert_F$, with $r=10^{-10}$.

Here $D$ is the full antisymmetric dipole tensor in the simulation frame.
The Frobenius norm is the square root of the sum of squared components,
including both antisymmetric halves. The preserved initial reference tensor
supplies the scale. Its value is frozen at preparation, so growth in the moment
does not automatically loosen the guard. The history code continues to enforce
an absolute native-unit budget, and the same budget is restored on resume.

## Practical selection

- For a first run in the supported fixed-recording solver, leave both budget
  options blank to use the provisional relative default. The startup log shows
  each reference norm and resulting absolute budget.
- Set `--dipole-drift-relative` to choose a different dimensionless allowance.
  The GUI has the corresponding relative allowance field.
- Set `--dipole-drift-budget` to reproduce a known absolute native-unit budget.
  This overrides automatic derivation and cannot be combined with an explicit
  relative allowance. The absolute option applies the same budget to both particles.
- Existing budgets are preserved when options are omitted. Relative selection
  is restricted to fresh setups; an evolved checkpoint does not recalculate its
  budget from its current moment.
- An initially zero dipole has no useful relative scale. Supply an explicit
  absolute budget based on the moment scale you intend to resolve. There is no
  universal native-unit floor.

For a new physical regime, check recording-resolution sensitivity of outgoing
momentum and the reaction contributions that matter. Keep the physical fit span
fixed during that comparison. If the history guard stops, examine the mismatch
and the effect of a reviewed budget change on those quantities before relaxing
it. Tightening this guard alone does not guarantee better radiation accuracy.

## Evidence and limits

The completed reciprocal benchmark had initial norms 1.35206 and 1.14472 native
units, and maximum fine-run mismatches 2.64261e-11 and 2.07001e-11. These are
approximately 1.95e-11 and 1.81e-11 of their initial scales. The provisional
1e-10 relative default allows about five times those observed mismatches.
It would permit the completed benchmark's published histories, whose late
dipole-squared impulse comparison was 1.034e-5 relative to the fine result.
The complete campaign used explicit absolute budgets; it was not rerun solely
to test this configuration policy.

The scale is frame-dependent, not Lorentz invariant. Agreement in this benchmark
does not calibrate the default for arbitrary ultra-relativistic boosts, long
trajectories, or initially tiny moments that later grow substantially. The default
is a documented starting point for those cases, with the guards and comparison
tests retained. Both reference norms and derived budgets are recorded in the
checkpoint's `dipole_budget_policy` and printed in the CLI/GUI job log.
