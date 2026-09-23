# Many-particle v0.10.0 release review

## Decision

Ready for the optional-mode milestone merge into `development` and `master`.
The repository uses `master`, not `main`. Keep all solver defaults unchanged.
This release closes the bounded mixed-stepping comparison and ordinary-input/
GUI integration tasks; it does not establish universal reaction conservation,
strong-spin validity, weighted pseudogrid support, or large-system performance.

## Validation

- Full unit suite: **1,776 passed, five failed, and two deselected**, in
  636.74 seconds. The five failures reproduce on an isolated archive of
  `development` at `cce1cf0`, in the same Python environment. No tolerance
  was relaxed. The two deselections follow the repository's test selection.
- Final input/GUI suite: **32 passed**, including three reaction-input checks
  added after the full suite started. These verify coupled reaction, zero-spin
  charge reaction, and rejection of nonzero spin in charge-only mode.
- The documented example completed ten steps and another ten after restart.
  The GUI command test independently verifies exact resumed output and input
  preservation. Real Tk window creation and scrolling passed on the Mac.
- Sphinx HTML: passed with warnings treated as errors. README, Sphinx,
  changelog, example input, and version metadata now describe v0.10.0.
- Staged formatting, lint, JSON, whitespace, and other non-type pre-commit
  checks pass. The mypy hook remains failing: the targeted seven-file check
  reports 196 errors in 33 files, identical to `4fd0760` after normalizing
  source-line numbers. This is not a clean repository-wide type check.
  The commit hook skips mypy only because that failure was checked separately.
- The completed 250/500-step reaction comparison passes every declared check
  with identical physical selection schedules and no RK4 fallback. Its largest
  relative impulse/torque difference is `9.103794e-5`, using the corresponding
  fine-result norm, below the declared `1e-2` limit. See the
  [plan and evidence](multiparticle_integration_plan.md).

## Inherited failing tests

The same five compiled-provider numerical-equality tests fail on the feature
and development snapshots:

- `test_roots_exact_charge_provider_is_bitwise_reference_equal`
- `test_full_strict_charge_provider_is_deterministic_with_reference_center`
- `test_numba_roots_backend_is_bitwise_invariant_to_numba_thread_count`
- `test_numba_backend_recomputes_final_worldline_sample_in_python`
- `test_full_strict_event_hertz_is_deterministic_and_within_one_ulp`

They remain tracked numerical-test failures, not passing tests. The isolated
baseline run of their three files reports five failures and 39 passes.

## Local evidence

Logs are outside the synchronized source tree, under `~/compute/` on the Mac:
`multiparticle-release-unit-tests.log`, `lw-v010-baseline-failures.log`,
`multiparticle-release-mypy.log`, `lw-v0.10-sphinx.log`, and
`lw-v010-example-resume.log`. The compact physical comparison is committed in
`docs/evidence/multiparticle_2026-09-23/mixed_recording_resolution.json`.

Existing changes in the development checkout's `AGENTS.md` and two archived
scripts are unrelated to this release and must remain untouched by merging.
