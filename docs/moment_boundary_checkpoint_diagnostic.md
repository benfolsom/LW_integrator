# Magnetic-force correction in actual checkpoint steps

## Latest extension: Medina-on diagnostic, 7 September

The diagnostic now supports `medina_lad` in addition to `off`. The older
radiation-off restrictions and results below describe the initial version.
Each uncommitted role step is repeated until its assumed Medina lab force
matches the force actually applied (relative tolerance 1e-6, maximum six
iterations). Invalid, unprimed, capped or nonconvergent trials fail explicitly.
The existing RFS helper supplies both the radiation acceleration and matching
spin transport. The study callback also adds the ordinary-force derivative's
missing reaction-acceleration contribution. Medina itself is still applied
exactly once per trial. Physical start-force storage is unchanged.

This uses the same step-averaged reaction force convention as the existing
spin update; it is not a proof of exact instantaneous acceleration. No
production default or general magnetic self-reaction mode has been enabled.
Historical/no-op controls pass on the first inbound Medina-on checkpoint.
The study's `decision_focused_continuation_2026-09-07.md` records the new
short-trajectory comparison and independent shell check.

All four bounded inbound continuations subsequently completed: baseline and
corrected paths at 1e-13 ns (four steps) and 5e-14 ns (eight steps). Corrected
electron cumulative absolute projection is 1.4513e-8 and 3.3473e-9 meV, versus
9.6172e-8 and 5.1915e-8 for baseline. Paths cover only 4e-13 ns and do not reach
closest approach. This validates the short continuation plumbing and observed
numerical improvement, not a full flyby or energy-conservation claim.
The expanded focused core suite passes 109 tests, with one deselected.

7 September 2026. Continues the [second-order moment-force plan](moment_force_second_order_plan.md).

## What changed

`ExactPairEOMOptions.moment_impulse_diagnostic` can explicitly bind a callback
to the equations of motion for a diagnostic pair trial. The ordinary path
retains its original callable and arguments when the option is absent.
The callback returns only the difference from the first-order magnetic
impulse already accumulated by the solver. It is added before the normal
momentum/velocity reconstruction and pair endpoint finalization.

The physical start force, stored start acceleration and force-memory inputs
are not replaced with an averaged force to imitate a higher-order impulse.
Callback outputs must be finite four-vectors. This route requires exact
second-order pair stepping, active magnetic force and spin, causal local
dipole history, no prescribed external field and radiation reaction off.
Unsupported combinations raise errors. There is no CLI/GUI option or
checkpoint-method promotion; the callback is not a serialized run setting.

The callback itself lives in the study's
`diagnostics/checkpoint_moment_correction.py`, not in the production solver.
It checks that its recomputed start force agrees with the force used by the
live step. It uses the analytical force derivative for sampled smooth steps,
and a piecewise midpoint impulse through detected charge-provider changes.
It keeps the explicit degree-six dipole fit and rejects detected dipole
fit-scale changes. Neither its nine-point method scan nor its midpoint/end
dipole checks prove absence of arbitrarily narrow transitions; these are
bounded diagnostic controls, not production event handling.

## What was tested

On the Mac, the 13 focused core modules pass **244 tests**. This includes
the existing source/force tests, actual pair-trial regressions and six new
diagnostic-option tests. Black/Ruff pass for changed Python files.

The study runner first reproduces each historical saved step with the option
disabled. At every tested step size, a zero callback leaves the 16 checked
position, momentum, velocity, spin and projection quantities bitwise unchanged.
It then applies the actual correction through the maintained trial solver.
The three saved events use lab steps 1e-13, 5e-14 and 2.5e-14 ns. An additional
inbound 2e-13 ns step exercises an actual source-knot crossing. Radiation is
off for these comparisons; the original historical reproduction retains its
original Medina setting with the callback disabled.

At 1e-13 ns the absolute electron projection discrepancy changes as follows:

| Event | Existing update, meV | Corrected update, meV |
| --- | ---: | ---: |
| Inbound 12 pm | 2.400e-8 | 3.908e-9 |
| Closest approach | 8.309e-8 | 1.042e-11 |
| Outbound 12 pm | 3.177e-8 | 3.827e-9 |

For the inbound steps 2e-13, 1e-13 and 5e-14 ns, corrected electron errors
are 3.105e-8, 3.908e-9 and 4.754e-10 meV, consistent with near-eightfold
reduction per halving. At 2.5e-14 ns the result is 9.806e-11 meV and the trend
is less clean. Closest-approach corrected values change sign around
1e-11--4e-11 meV rather than displaying a useful convergence sequence.
Proton errors are much smaller and do not improve systematically.
Do not interpret these small residuals as established roundoff bounds.

These are actual one-step projection checks, but not a global energy ledger,
complete self-reaction validation or a full flyby. The historical flyby
energy gates remain failed. No production default is changed.

## Remaining work

Check the small-residual sensitivity and midpoint-panel sensitivity before
promoting a trajectory claim. Replace sampled event searches with reliable
event handling before unrestricted stepping. Extending to Medina requires
an explicit treatment of the physical total acceleration used in the moment
derivative: the present start acceleration is complete only with reaction
off. Do not simply remove the radiation guard. Broader relativistic,
three-dimensional and external-field checks and restart identity remain.

The study's `planning/overall_progress_and_live_checkpoint_2026-09-07.md`
contains the archived reports, command and current shell status. The Mac
worktree is `/Users/benjaminfolsom/compute/LW_integrator-moment-boundary-checkpoint`
on `feature/moment-boundary-checkpoint`, based on `759997e`. The independent
shell core at `dbe2a21` remains untouched.
