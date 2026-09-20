# Finite-spin milestone: integration review

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
