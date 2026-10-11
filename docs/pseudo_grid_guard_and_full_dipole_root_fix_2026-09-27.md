# Pseudo-grid state guard and full-dipole root tolerance: justification

Date: 2026-09-27. Branch `feature/bunch-self-field-check`, based on
`origin/development` at `96dfec4`. Commits:

| Commit | Subject |
| --- | --- |
| `e1bc9ed` | Reject invalid pseudo-grid reconstructed states |
| `8492bfe` | Scale full-dipole retarded-root tolerances |

Both problems were found by the aneutronic feasibility study
(`LW_feasibility_studies-aneutronic`, branch `study/aneutronic-cellular`). A
Codex executor implemented the fixes. The supervising agent reviewed them,
reran the changed tests and committed them. Detailed evidence and reproduction
scripts are in the git-ignored `local/` directory of this worktree; this page
records the decision-relevant facts.

## 1. Pseudo-grid passive reconstruction could publish unphysical states

### What was observed

A scoping study used the legacy charge/pseudo-grid path for a single
self-expanding electron bunch (1 nC, $\gamma = 10$, $\sigma_r = 10$ µm,
$\sigma_z = 30$ µm) with 8 or 16 active and 1,000 passive particles.

- The reconstructed passive particles reached $|\beta| \ge 1$: 40 final
  particles in the 8-active run.
- Some particles' time stopped increasing.
- No error was raised.
- The 32-active run blew up (γ ≈ $5\times10^{8}$ at step 7).

The report is `local/bunch_self_field_scoping_2026-09-26.md`, section 4.

### Why this justifies a hard error

A state with $|\beta| \ge 1$, a non-finite value, or non-increasing time is
not a physical state. Publishing one silently contaminates retarded histories
and every observable derived from them. Clipping would hide the failure, so
the correct response is to stop and report it.

### What changed

`core/pseudo_grid.py` adds `PseudoGridStateError` and validates every live,
advanced particle before `reconstruct_full_state_from_active_result` returns.
This covers active copies, weighted passive deltas, ballistic coasting and
external-interbunch passive solves.

- The error names the step, the particle index, $|\beta|$ and both times.
- Values are never modified.
- Explicitly frozen passives and dead particles keep their existing behaviour.

**Behaviour change:** a live weighted passive particle with no valid anchor
previously kept its old time silently. It now raises.

`core/integration_runner.py` passes the step index. The microbenchmark
script's synthetic result now advances time.

### Tests

The focused pseudo-grid, runner and physics set gives:

- **patched:** 124 passed, 1 expected failure;
- **clean `96dfec4`:** 88 passed, the same expected failure;
- **new tests:** 36, covering superluminal, NaN and infinite speeds, invalid
  time in the ballistic and external modes, time lost to roundoff, copied
  non-finite kinematics and lost anchors.

Existing normal, frozen, ballistic and external-interbunch cases are
unchanged.

### Consequence for past results

A read-only audit of about 1,560 saved pseudo-grid result bundles found that
none retains per-particle velocities or times. So whether past runs were
affected can be neither demonstrated nor excluded. See
`LW_feasibility_studies-aneutronic/results/audit/20260927T0919Z_pseudogrid_superluminal_audit.md`.
Rerunning the configurations behind past conclusions with this guard enabled
is the recommended way to settle it.

### Not fixed here

The scoping study also found that the legacy path cannot run a
first-principles self-expanding bunch:

- no inertial prehistory without dipoles;
- `COLD_START` uses stationary-source same-bunch forces;
- uncontrolled 15–43 µm smoothing;
- dense passive factorisation.

These need separate development. The proposal is in the scoping report,
section 5.

## 2. Full-dipole light-cone roots were rounded onto segment joins

### What was observed

An electron–carbon full-spin encounter (100 keV, impact parameter 1 pm,
$g = 2.00231930436$, `full_dipole_coupled`) failed in the fourth RK4 stage at
1.8 pm separation:

`ValueError: Retarded root must lie inside a smooth segment`

### Diagnosis

At the rejected event, in length-time units (mm):

| Quantity | Value |
| --- | --- |
| Segment duration | $1.294\times10^{-11}$ |
| Old absolute `brentq` tolerance | $2\times10^{-14}$ |
| Returned root: fraction, light-cone residual | 0.9999999999989729, $-6.6\times10^{-15}$ |
| Independently refined root: fraction, residual | 0.9996282150881548, $-3.9\times10^{-24}$ |

The true root lies $4.8\times10^{-15}$ mm from the segment end. That is inside
the old fixed tolerance but far outside the $10^{-8}$ fractional join guard.
The solver stopped on a point indistinguishable from the join, and the guard
then correctly rejected it.

**The failure was an implementation defect.** It was not a genuine join and
not a physical close-encounter limit.

### What changed

In `core/full_dipole_response.py`:

- **Root tolerance:** absolute $4\epsilon \times$ segment duration (floored at
  the smallest positive float), relative $4\epsilon$.
- **Endpoint recognition:** $8\epsilon$ times the largest coordinate magnitude
  involved, instead of fixed absolute values.
- **Unchanged:** the strict smooth-segment guard, the explicit
  boundary-policy requirement for true joins, and the callers'
  source-position budgets.

### Tests and evidence

- **New tests** (`tests/unit/test_full_dipole_response.py`): the rejected
  geometry at three length scales ($10^{-9}$, 1, $10^{9}$), with both boundary
  policies; the static Coulomb field; and genuine joins still rejected
  without an explicit policy.
- **Full-spin/pair suites:** 373 passed on the patch, against 365 on
  `96dfec4`, with the same two Tk skips. The eight added tests account for
  the difference.
- **Event replay:** the saved rejected event now resolves to fraction
  0.9996282150881548, with finite point and fourth-order Taylor responses.
- **Fresh encounter:** at the fine step (fraction 0.01) the encounter now
  completes: 2,784 steps, minimum separation 0.954 pm, 127 min. Its radiated
  charge-plus-dipole field energy (1.2348683 eV) agrees with the electron
  reaction work (1.2348891 eV) to $1.7\times10^{-5}$. A matched no-nucleus
  control radiates $<2\times10^{-42}$ of the signal over 1–100 keV.

### Not fixed here, deliberately

The coarse step (fraction 0.02) still fails at startup:
`SourcePositionError: 7.31e-15 mm > 1.0e-15 mm`. That budget is the study's
declared accuracy, $10^{-6}\,b$ with $b = 1$ pm, i.e. 0.001 fm. Loosening it
would change the accepted error, so it was left in place. The proposed
remedy is finer startup recording, with convergence checks.

- A step-refinement pair for the spin encounter is therefore still missing.
- Full-spin energy closure, including independently tracked spin and
  bound-field endpoints, is **not** established by this change.
- A checkpoint written by the old root solver may fail the endpoint
  consistency check on restore; regenerate it from its preparation.
