# Review: `review/sampled-smearing` vs `development`

Branch tip `98dec54`; merge base `94dbb48`; `development` at `bb301e7`.

- `3cd1e29` is a cherry-pick of `d2dbac3`. Both have the same `git patch-id` (`2493e38c…`), and `d2dbac3` is **already on `development`** (merged in `b6e8ed2`).
- `98dec54` is the change under review: it matches each smeared child to its own light cone.

## Summary

The approach is sound. A child is the parent's stored worldline translated by its constant lab-frame cloud offset. The new matcher `match_cloud_child_indices` (`core/distances.py:181`) applies the same rule as the point-source matcher (first knot with residual ≤ 0 on a bracketed history). The child's own `x, β, β̇, γ` replace the parent's (`core/macroparticle_smearing.py:198-233`). Charge, mass and population still come from the parent, so normalisation (`q/K`, population repeated) is unchanged.

Default behaviour is unchanged:
- Smearing is off by default (`MacroparticleSmearingConfig.enabled=False`).
- `_smeared_child_samples` returns `None` when smearing is disabled, when `subcharge_count <= 1`, or when there is no fixed cloud. In those cases the old `np.repeat` expansion runs.

**Recommendation: merge after a rebase onto `development` and after fixing P1-1. Ideally also add the parity and integration tests listed under P2.** I could not reproduce the claimed 216 MeV convergence table (5.4 → 0.73 %) because the branch has no script or config for it. I am treating those numbers as unverified.

## Validation performed

- `pip install -e ".[dev]"`, Python 3.13.
- `pytest -p no:anyio`:
  - `tests/test_macroparticle_smearing.py`, `tests/unit/test_distance_helpers.py`, `tests/unit/test_retarded_fields.py`, `tests/physics/test_free_crossing_null.py`, `tests/physics/test_same_bunch_plummer.py`, `tests/physics/test_self_space_charge_energy.py`, `tests/unit/test_equations_helpers.py`, `tests/unit/test_exact_source_cloud.py`, `tests/unit/test_pseudo_grid_passive_state.py`, `tests/unit/test_potential_inclusion.py`: **203 passed, 1 xfailed** (the xfail was already marked).
  - `tests/test_integration_e2e.py`, `tests/test_pic_coupling.py`, `tests/test_pic_nearfield.py`, `tests/test_single_integration_helpers.py`, `tests/unit/test_inertial_driver_train.py`, `tests/unit/test_trajectory_integrator_helpers.py`, `tests/unit/test_integration_runner_control_flow.py`, `tests/unit/test_exact_train_clouds.py`, `tests/unit/test_exact_audit_source_routes.py`, `tests/unit/test_exact_same_bunch_fields.py`, `tests/unit/test_source_history_representation_config.py`, `tests/unit/test_exact_source_reduction.py`: **149 passed**.
- Cost micro-benchmark: 64 sources × 8 children, 1200-step history, counter-propagating, one observer.
  - List path: parent match 2.0 ms, child match 6.1 ms.
  - SoA path: child match 0.7 ms.
- `git merge-tree origin/development origin/review/sampled-smearing` reports **conflicts** in `CHANGELOG.md`, `core/distances.py`, `tests/unit/fixtures/medina_legacy_3673af5.json` and `tests/unit/test_distance_helpers.py`.

## P0 — wrong default behaviour

None found. The new code runs only with `macroparticle_smearing.enabled`, `subcharge_count > 1` and a `fixed_per_particle` cloud. The point-source path, and both the B2B and same-bunch matching, are textually unchanged.

## P1 — wrong physics

**P1-1. On B2B, fields are evaluated at a different observer position from the one the children are matched to.**
- **Where:** `core/equations.py:2868-2872` against `core/distances.py:236-237`.
- **Problem:** `match_cloud_child_indices` puts each child on the light cone of the *stored start-of-step* observer, `trajectory[index_traj][particle_idx]`. But `smear_source_samples` builds `R` and `n̂` from `observer_position=(working_x, working_y, working_z)`. After the first self-consistency trial, `working_x` becomes `result["x"]`, the end-of-step trial position (`core/equations.py:5811`).
- **Failure scenario:** the default settings use 2 fixed-geometry iterations, so the force that is finally applied pairs each child's event with an observer about one step away from the apex of the light cone it was matched on.
  - The mismatch in `R` is about `|β_obs| c Δt_lab`. This is first order in h. In the claimed convergence table it is mixed in with the sampling error the fix is meant to remove.
  - The point-source path uses `compute_retarded_distance` at the stored start-of-step position. So a smeared source with all-zero offsets does **not** reproduce the point source on the B2B path.
  - The same-bunch path is internally consistent: it uses `current_state` for both.
- **Note:** this inconsistency existed before, but the per-child "own light cone" claim is only true if both steps use the same observer event.
- **Fix:** pass the same observer event to both. Either pass `working_*` and the matching observer time to the matcher, or evaluate the smeared `n̂` at the stored position, as the point path does.
- **Regression test to add:** a smeared source with zero offsets, run through `retarded_equations_of_motion` with SC iterations ≥ 2, must equal the point-source force to round-off.

## P2 — robustness

**P2-1. Variable-geometry self-consistency silently reverts to the shared event.**
- **Where:** `core/equations.py:2889-2893`.
- **Problem:** for `sc_iteration > 0` under `variable_geometry`, `child_samples=None`. Iteration 0 therefore uses per-child events and later iterations use the parent event. The model changes between iterations, and the iteration difference used for convergence then measures that switch rather than convergence.
- **Fix:** call the matcher with `temp_trajectory` and `observer_particle_idx`, as the parent matching does at line 2683.

**P2-2. Several configurations silently fall back to the old shared-event expansion.**
- **Where:** `core/equations.py:2876-2895` and `3253-3268`.
- **Affected configurations:**
  - `chrono_interpolate=True`, which is exactly the configuration AGENTS.md reserves for retarded-time sampling studies;
  - `refresh_policy` other than `fixed_per_particle`, because `cloud is None`;
  - pseudo-grid same-bunch sources;
  - `APPROXIMATE_BACK_HISTORY`.
- **Problem:** these are documented in the CHANGELOG but nothing warns or logs at runtime, so a sampling-convergence study run with interpolation on gets the known-wrong expansion.
- **Fix:** add a one-time warning, or reject the combination.

**P2-3. Unbracketed children inherit the parent's index.**
- **Where:** `core/distances.py:259-260`.
- **Problem:** when a child's translated history does not bracket the crossing (early cold start, or a child whose offset moves the crossing before history start), it takes the parent's index plus its own offset. The parent index may itself be the delay-estimate fallback. Near startup this mixes matched and unmatched children within one cloud.
- **Status:** acceptable, but no test covers it, and the directional/COLD_START gate `_should_apply_external_forces` (`core/equations.py:2797`) is still decided from the *parent* `n̂` before smearing.

**P2-4. "Longitudinal" cloud offsets are hard-wired to z.**
- **Where:** `core/macroparticle_smearing.py:123-130`. This predates the branch.
- **Problem:** `(σ_T, σ_T, σ_L)` is applied to `(x, y, z)` regardless of `momentum_axis`. Per-child matching matters most for longitudinal offsets. For the full-3D configs AGENTS.md recommends with `momentum_axis` ≠ z, `longitudinal_sigma_mm` adds a transverse offset instead.
- **Fix:** before relying on this fix for such configs, rotate the offsets into the beam frame, or document the limitation.

**P2-5. Cost.**
- **Where:** `core/distances.py:222-232`.
- **Problem:** the list (non-SoA) `event()` builds a Python nested list of `n_src × K` for each of about log₂T bisection passes and each of 4 fields. The matcher and K gathers then run per observer, per SC iteration, and on both B2B and same-bunch. The list path cost about 3× the parent match in my benchmark; SoA was cheaper than the parent.
- **Impact:** fine for production, which uses SoA through `integration_runner`. List-path callers such as the GUI and direct calls see roughly an `N_obs × 2 × 2 × 6 ms` per-step overhead at 64×8.
- **Fix:** do `K` gathers through a single fancy-indexed gather of `(n_src, K)` indices.

**P2-6. Tests.** The only new test is the matcher unit test. Missing:
- (a) an end-to-end check that point sources are bit-identical (the claim is true by inspection, but untested);
- (b) a zero-offset smeared source equal to the point source (see P1-1);
- (c) a same-bunch test checking that self-exclusion survives child samples. It does by inspection, because the parent `valid_mask`/`charge` zeroing at `core/equations.py:3220-3227` is ANDed and repeated;
- (d) a slow-marked convergence regression for the claimed 216 MeV case.

**P2-7. The branch does not merge cleanly.**
- **Problem:** it conflicts with `development` in 4 files, and `3cd1e29` duplicates `d2dbac3`.
- **Fix:** rebase `98dec54` alone onto `development`, then re-run the tests above.

## P3 — nits

- `core/distances.py:259`: `np.where(first <= 0, 0, hi)`. Only `first == 0` can reach this under `bracketed`, so write `first == 0` for clarity.
- The same-bunch path uses `current_state` for the observer position, while B2B uses `working_*`. Unify them (see P1-1).
- Cloud offsets are lab-frame, drawn at `γ=1` in `_initial_cloud` and not Lorentz-contracted. This matches the exact persistent clouds, so it is a modelling convention, but the docstring should say that the `σ` values are lab-frame lengths.
- `smear_source_samples` returns the unsmeared samples (`return samples, {}`) if any child batch lacks positions. That cannot happen with `include_positions=True`, but raising would be safer than silently reverting to point sources.
- The momentum-error kick is applied on top of each child's own velocity. That is fine, but it means the effective cloud velocity spread now includes real history differences. Mention this in the docs.
