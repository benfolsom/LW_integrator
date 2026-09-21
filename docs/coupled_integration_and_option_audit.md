# Coupled reaction integration and option audit

**Latest follow-through:** [internal error control](internal_error_control.md)
adds matched adaptive motion/reaction integration without shortening the source
recording cadence. API, CLI and checkpoint GUI share its settings and restart
rules. This is separate from the outer step-doubling controller described below.

**Latest integration follow-through, 10 September:** [higher-order stepping and
shared-time adaptive integration](higher_order_pair_and_adaptive.md) now provide
native DOP853 stepping with matched reaction accounting, an adaptive API/CLI,
and GUI method selection/adaptive checkpoint resume. This supersedes the older
missing-controller statement below. Existing charge-only self-consistency and
chrono options remain in their own solver; they are not duplicated here.
Radiation-accuracy validation and magnetic pseudogrids remain separate open items.

**10 September follow-through:** [analytical derivatives and controller status](analytic_reaction_and_controllers.md)
supersedes the centered-window recommendation below. The maintained provider
now supports analytical moving-source derivatives, with CLI/GUI selection and
no fitting window. This does not wire the nonlinear pair into the older
adaptive controller or pseudogrid path; both gaps are explicitly documented.

## Working method

`full_dipole_coupled` is the working full-spin radiation-reaction method for
the nonlinear pair API and checkpoint runner. It computes local motion and
the regular self-field together. It retains all charge-squared, charge–dipole
and dipole-squared terms. `full_dipole_rr` remains a legacy reduced-order
comparison/replay mode; it is not the recommended full-spin selection.

This is a development decision, not a claim that all physics checks passed.
Higher spacetime derivatives of the local self-field remain omitted. The
longer post-arrival comparisons, strong-domain coverage and independent total
particle-plus-field conservation remain open. No force, spin or velocity is
clipped to make a failed state acceptable. Radiation reaction remains optional.

## One solver behind the API, CLI and GUI

The coupled solver now lives in `core/coupled_dipole_reaction.py`; it does not
import study scripts. The native pair dispatches `full_dipole_coupled` to it
and preserves its mode, derivative window, centered derivative method and
full force/torque ledgers in checkpoints. New coupled preparations default to
the existing C7 startup. Existing startup histories are not rewritten.

```sh
python -m lw_integrator.nonlinear_pair --capabilities
python -m lw_integrator.nonlinear_pair \
  --checkpoint initial.json --output coupled.json \
  --step-ns STEP --steps COUNT \
  --radiation-reaction full_dipole_coupled \
  --reaction-window-ns WINDOW --reaction-derivatives centered
```

`STEP`, `COUNT` and `WINDOW` are placeholders, not recommended numerical values.
Choose appropriate units and resolution for the prepared initial state.
For an evolved coupled checkpoint, omit the reaction options to preserve them.

The main GUI has a **Nonlinear pair…** button. Its small checkpoint window
launches the same CLI in a detached process, shows its log and saves every
accepted step. Closing Zed or that GUI window leaves the launched job running.
The window can also be started with `python -m lw_integrator.nonlinear_pair_gui`.
It does not silently convert older bunch configurations or invent source pasts;
prepared whole-pair checkpoint input is still required. This is checkpoint
GUI integration, not a completed general-purpose initial-condition editor.

## Failure and restart policy

A velocity-domain failure in an older pair reaction mode now recommends
`full_dipole_coupled`. It does not retry that failed state under different
equations. Start from fresh prepared data or use a checkpoint that already
contains coupled histories. A coupled velocity failure is reported as such,
not followed by another undisclosed model change.

Changing reaction mode, window or derivative method on an evolved checkpoint
is rejected. Even a window change can alter endpoint velocities used by stored
histories. Fresh data may select a method, but must still pass the native
history/state consistency checks. The old study checkpoint envelope remains
distinct and is not automatically reinterpreted as a native checkpoint.

## Which older controls should remain?

| Control | Existing charge/Jakobsen solver | Nonlinear coupled pair | Decision |
| --- | --- | --- | --- |
| Self-consistency iterations | Still used by the existing equation/integration path | Not called; the momentum–velocity constraint and local self-field solve have their own equations | Retain for the older solver; do not duplicate it in the pair UI |
| Gamma reconciliation | Optional projection/blending in the older equation path | Not used; do not project the coupled velocity or dynamic mass | Keep as an older-solver diagnostic, not a new pair option |
| Chrono matching | Optional interpolation of sampled retarded data in the older path | The provider solves the light cone inside published polynomial histories | Do not add a second interpolation switch to the pair |
| `full_dipole_rr` | Separate nonlinear reduced-order implementation | Available for explicit replay/comparison | Legacy designation; recommend `full_dipole_coupled`, no removal yet |

The source audit found live consumers of the older settings in
`core/equations.py`, `core/integration_runner.py` and `core/self_consistency.py`.
They are not dead configuration. Removing them globally would change the
charge-only solver without establishing an equivalent replacement. No such
removal or change of older solver defaults is part of this integration.

The two kinds of "self-consistency" must not be confused: the old options
iterate an accepted-step correction; the coupled reaction calculation seeks
agreement between local source motion and the self-field it generates.

## Profiling and bounded optimization

A fixed accepted event from the completed study pilot was profiled before
changing arithmetic. Its input checkpoint was preserved and hashed. Of 13.19 s,
13.05 s lay in the retarded provider and 9.10 s in Python derivative-array
multiplication. No legacy self-consistency or chrono routines appeared.

The analytical derivative class now reuses the existing strict-float64
compiled multiplication and reciprocal kernels. Coefficient ordering and
summation order are unchanged; `fastmath` remains off. The original Python
operations remain the fallback without Numba and a reference for tests.
Only the full-tensor provider selects the accelerated derivative subclass;
existing Python-reference providers retain their original arithmetic backend.
This reuses earlier optimization work rather than introducing another kernel
family. It is not a complete compiled full-tensor response or GPU path.

The first promoted-core fixed-event result was bit-for-bit identical in all
14 state-rate components and the root residuals; the provider still made the
same 845 samples. Its profiled wall time was 3.50 s. Separate unprofiled repeat
timings give 11.19 s for Python and 1.99 s for the final provider-scoped
compiled evaluation, a 5.61× fixed-event speedup. The fresh three-step CLI
trajectory matches the original coupled study in states, histories and ledgers
bit-for-bit. This does not establish a full-run speedup.
The two previously launched long studies retain their
already-loaded original code; they were not restarted or silently updated.

## Keep the next work focused

1. Arithmetic, interface and fresh three-step CLI migration checks have passed.
   Keep the separately recorded older rounding-threshold failure and missing
   sweep fixtures visible; do not claim the entire repository suite is clean.
2. Keep the original long comparisons running. Assess any remaining history
   mismatch on coupled trajectories, not by retuning the legacy method.
3. After a successful CLI comparison, use the maintained runner for a longer
   coupled trajectory. Do not make unsupported total-conservation claims.
4. Only then consider further profiling-led work: full response compilation,
   repeated-event reuse or faster history lookup. Do not add more solver knobs
   or remove older charge-only machinery without a measured reason.

Current evidence is archived in the study repository under
`planning/evidence/coupled_integration_2026-09-10/`. The running comparison
plan is `planning/coupled_reaction_history_2026-09-10.md` in that repository.
