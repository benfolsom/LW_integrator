# Finite-spin milestone: integration review

## 21 September continuation

The broader unit run has now completed: 1,632 passed, five failed, and one
was deselected. All five failed test cases reproduce on the isolated
origin/development checkout in the same environment. The additional three
are retarded-dipole compiled-provider exact-equality checks; their baseline
report is `/tmp/lw-baseline-dipole-failures.log`. These remain failures,
but this comparison found no new failing unit tests from the milestone.

The straightforward type cleanup now reduces the repository-wide count from
2,569 to 2,521 errors (90 files). Taylor-series arithmetic and history sampling
pass their targeted type check without suppression. Explicit arguments replace
ambiguous unpacking in the physical error estimator. The affected numerical
tests passed 25 cases; Ruff and whitespace checks passed. Report:
`/tmp/lw-mypy-20260921.log`. The repository-wide type gate remains open.

The original interrupted unit run had no recoverable final report. A fresh
unit run now records `/tmp/lw-merge-unit-20260921.log` and JUnit output at
`/tmp/lw-merge-unit-20260921.xml` when it finishes. Merge remains pending.

Fixed the internal-step CLI parity regression: its expected result must use
the selected history setup now that fresh CLI runs default to single-fit.
The regression now checks both automatic selection and explicit preservation.
The focused history/internal-step suite passed 42 tests, and the subsequent
option/CLI suite passed 31 tests.

Added concrete history and CLI type annotations and made validated optional
history fields explicit to the type checker. The three history modules,
history-option module, and pair CLI no longer have direct diagnostics in the
targeted type-check report. Imported modules still produce failures; this is
not a clean repository-wide type check.

Fixed duplicate generated documentation for the adaptive-pair configuration.
The complete Sphinx HTML build now passes with warnings treated as errors.
Report: `/tmp/lw-docs-review-20260921.log`; output:
`/tmp/lw-full-docs-merge-20260921/`.

## Outcome

The numerical milestone and interface work are ready for a reviewable commit.
The development merge remains pending broader regression completion and
resolution or explicit disposition of type-check failures.

Fetched origin/development at 9ca9377. That commit is an ancestor of the feature
branch, which has 104 subsequent commits before the new milestone commit.
The local development checkout has unrelated changes in AGENTS.md and two
archived scripts; these have not been modified by this review.

## Checks

- All 56 Python files in the staged milestone changes pass Ruff and Black.
- Whitespace checks pass. The previously completed focused tests, GUI launch,
  and coarse–fine numerical comparison remain documented in the user guide.
- The full Sphinx HTML build succeeds using a dedicated environment outside
  the synchronized work directory. It reports one duplicate API description
  for IntegratorConfig.adaptive_pair_return. A warnings-as-errors build still
  fails on that warning.
- Repository-wide pre-commit is not clean. The type checker reports 2,569
  errors in 97 files on the feature checkout, versus 2,323 in 58 files on an
  isolated origin/development checkout using the same hook. The net increase
  is 246; it must not be described as entirely inherited baseline debt.
- The broader unit suite is running. Its two observed exact-charge-provider
  failures reproduce on origin/development: one exact-array comparison differs
  by at most 4.147e-18, and one normalized gradient comparison measures
  3.872e-12 against a 2e-12 limit. No tolerance was changed.

The all-files formatter touched unrelated files during inspection. Those
automatic changes were reverted to the staged snapshot; they are not included
in the milestone. No development merge or remote push has been performed.

## Evidence locations

Local logs: /tmp/lw-merge-precommit.log, /tmp/lw-feature-mypy.log,
/tmp/lw-development-mypy.log, and /tmp/lw-full-docs-build.log.
Built manual: /tmp/lw-full-docs-merge-20260920/.
Documentation environment: ~/compute/lw-docs-merge-venv/.
Baseline checkout: ~/compute/LW_integrator-merge-baseline-20260920/.
