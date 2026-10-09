# Review: `review/external-field-provider` (520c1bf) vs `development` (7201440)

Scope: opt-in spatial external-field provider
(`core/external_field_provider.py`, `ExternalFieldConfig(provider=...)`) and its
use in the force stage and the Medina/LAD force derivative. Line numbers refer to
520c1bf. This branch (`review-notes/external-field-provider`) adds only this file
on top of the review branch. I did not modify the review branch or `development`.

## Summary

- **Default path:** no change. 247 output arrays were byte-identical to
  `development` (details under P0).
- **Medina/LAD derivative:** correct. It uses the right orders, units and signs
  (details under check 1).
- **Main limitation:** on the default (non-exact) route, channel trajectories
  are only first-order accurate in dt. The provider force is sampled once at
  the step-start event and integrated with a symplectic-Euler-like update. In a
  smooth tube, the particle leaves the field with a net kinetic-energy error
  that is O(dt). The exact second-order route conserves energy at O(dt²).
  Neither the branch's validation nor its documentation says this.
- **Other problems:**
  - A non-RFS Stern–Gerlach force silently ignores the provider's ∇B.
  - A plain-function provider combined with checkpointing raises `TypeError`.

**Recommendation: merge after the P1 is fixed or guarded**, and the documentation
for the first-order default route has been added. Details are at the end.

## Checks performed

| Check | Result |
|---|---|
| `pytest -p no:anyio tests/unit/test_external_fields.py tests/unit/test_external_potential_derivatives.py` | 25 passed (133 s) |
| `test_medina_radiation_reaction`, `test_medina_validity_guard`, `test_growable_trajectory_builder`, `test_trajectory_arrays`, `test_trial_trajectory_history`, `test_rfs_radiation_reaction` | 71 passed, 3 failed. The 3 failures are `test_legacy_trajectory_matches_3673af5_snapshot`. They also fail on `development`, with **identical** hashes (environment-dependent snapshot), so they are not a regression. |
| Default-path byte parity, `development` vs branch, 64 steps. Cases: no field + Medina; uniform E+B (γ=20) + Medina; linear ∇B, RR off; hard `z_max` window + Medina | 247/247 arrays identical, no keys added or removed |
| `validate.py --interface-only` | d(qF)/dt analytic vs numerical error 1.5e-11; SI conversion checks passed; a disabled provider is never called; hard windows are rejected |
| black / mypy on changed files | black clean. mypy shows no new errors (the same 52 pre-existing errors as `development`). |
| **Independent energy and timestep study (new, below)** | static channels, RR off and on, default and exact routes |

### Independent energy and timestep study

Setup: single positron/electron charge, `COLD_START`, default
`SelfConsistencyConfig()`, providers from
`diagnostics/external_field_provider/providers.py`. The conserved quantity is
H = (γ−1)mc² + qφ(x). A sketch of the script is in the appendix.

**A. Harmonic channel, RR off** (ω = 1e11 ns⁻¹, A = 1e-4 c/ω, 10 transverse
periods). E_tr = qkA²/2 is the transverse energy.

| steps | ω·dt_lab | max\|ΔH\|/E_tr | ΔH(end)/E_tr | p_x(end)/(mωA) (exact 0) |
|---|---|---|---|---|
| 151 | 0.419 | 0.265 | −0.154 | −0.462 |
| 301 | 0.209 | 0.117 | −0.024 | −0.116 |
| 601 | 0.105 | 0.055 | −0.0030 | −0.029 |
| 1201 | 0.052 | 0.027 | −3.8e-4 | −0.0072 |
| 2401 | 0.026 | 0.013 | −4.7e-5 | −0.0018 |

The energy error is bounded and oscillates with amplitude ≈ ω·dt/2. It is first
order with no secular drift, the signature of symplectic Euler. Phase converges
at O(dt²). The γ=20 longitudinal case, run over 10 transverse periods at the
lab frequency ω/√γ, gives the same table to 4 digits.

**B. Harmonic channel, Medina/LAD on** (same setup). Energy budget at 1201
steps:

- ΔH = −0.03894 E_tr
- Σ `radiation_reaction_work` = −0.03863 E_tr
- RR-off drift (from table A) = −0.00038 E_tr

The sum of the last two, −0.03901 E_tr, matches ΔH. The analytic loss over 10
periods, 1 − exp(−20πτω) = 0.03861, matches the RR work to 0.06%. RR work is
almost independent of the step size (0.0400 / 0.0390 / 0.0387 / 0.0386). This
confirms that the provider's derivative hook drives the right Medina damping.

**C. Smooth finite tube (`SmoothTube`, C2 quintic ramps), RR off, γ₀ = 2,
x₀ = 0.1 mm.** This tests region entry and exit; the field is zero outside the
tube.

| route | case | steps | max\|ΔH\|/K₀ | ΔK(exit)/K₀ |
|---|---|---|---|---|
| default | weak (max qφ/K = 2.8%, transmits) | 151 / 301 / 601 / 1201 | 1.2e-3 / 6.0e-4 / 3.0e-4 / 1.5e-4 | −1.7e-5 / −9.5e-6 / −4.5e-6 / −2.3e-6 |
| exact 2nd order | weak | 151 / 301 / 601 | 5.8e-5 / 1.5e-5 / 3.7e-6 | −2.0e-7 / −4.0e-7 / +1.1e-7 |
| default | strong (depth 7e7, reflects at entrance fringe) | 151 / 301 / 601 / 1201 | 5.1e-2 / 2.6e-2 / 1.3e-2 / 6.6e-3 | −2.9e-2 / −1.4e-2 / −7.2e-3 / −3.6e-3 |
| exact 2nd order | strong | 151 / 301 / 601 | 3.3e-3 / 8.1e-4 / 2.1e-4 | −3.5e-4 / −9.2e-5 / −8.6e-6 |

Interpretation:

- On the default route, the energy error is first order *and the particle exits
  with an O(dt) net energy error*. Because the τ-parametrised relativistic
  Hamiltonian is not separable (γ depends on p), the left-endpoint update is not
  symplectic. Once γ changes appreciably, the error no longer stays bounded.
- The exact `second_order_start_taylor_endpoint` route adds ½h²·dF/dτ using the
  provider's analytic ∂F and converges at second order. It is the route to use
  for channeling energy-loss claims.
- No problems at entry or exit with a C2 provider: there are no jumps or caps,
  and both routes converge smoothly.

## Check 1: Medina/LAD derivative terms, units, sign

- **Orders:** sufficient. `compute_medina_radiation_reaction`
  (`core/medina_radiation_reaction.py:155`) is reduced order. It needs only F
  and the lab-time dF/dt, so first spacetime derivatives of E and B are
  enough; the Hessian of E is not needed.
- **Formula:** `provider_force_derivative` (`core/external_field_provider.py:411`)
  computes q[∂ₜE + (v·∇)E + β̇×B + β×(∂ₜB + (v·∇)B)] through
  `event_rate = (c, v)` contracted with `partial_e[λ] = ∂E/∂x^λ`, where
  x⁰ = ct in mm. This is the complete convective derivative.
  - c·∂/∂(ct) = ∂/∂t is correct.
  - β̇ comes from `_derive_relativistic_kinematics_from_force` as dβ/dt (ns⁻¹).
    It is computed from the total non-RR force, which is correct for reduced
    order.
  - Units are native force per ns, which `compute_medina_radiation_reaction`
    expects.
  - Sign and value are confirmed by the 1.5e-11 numerical check and by the
    analytic damping in study B.
- **Index convention:** `partial_b[1:].T` equals the existing
  `magnetic_gradient[i, j] = ∂B_i/∂x_j` (`core/external_fields.py:473`). The
  `_external_tensor_gradient` rows (`core/equations.py:826-830`) match
  `mechanical_lorentz_four_force_derivative_native`, which expects x^λ in mm
  and no extra factor of c.
- **SI conversion** (`from_si`, lines 369-400): φ·1000·e_scale,
  A·1000·b_scale, the time row /(c·1000) and the space rows /1000 are all
  correct.

## Check 2: same event, accepted history, entry/exit

- **Same event:** the provider is evaluated once per force trial at
  `field_position` and `current_state.t`. With `fixed_geometry` (the default),
  every self-consistency trial uses the start event, so the force, the derivative
  sample and Medina's β/γ (`provider_beta`/`provider_gamma`, the working state)
  belong to one event and one velocity.
- **Rejected trials:** adaptive rejected trials never write `current_state`. The
  derivative of the sampled remainder uses only accepted history:
  `total − provider` is reconstructed from the accepted
  `medina_external_force_*` minus `medina_provider_force_*`.
- **Entry/exit:** hard windows are rejected when a provider is set
  (`core/types.py:1403-1418`). C2 smoothness is left to the provider, and the
  study above shows clean behaviour at entry and exit.
- Exceptions: P2-1 (time-dependent providers under `variable_geometry`) and the
  P3 notes.

## Findings

### P0: wrong default behaviour

None found. Evidence: the 247-array byte-parity check above. Disabled providers
are never called (interface check). `_initialize_medina_step_state` pops the
provider keys when no provider is set, and `TrajectoryArrays.state` emits them
only when they are nonzero.

### P1: wrong physics when enabled

**P1-1: Non-RFS Stern–Gerlach force silently ignores the provider's magnetic
gradient.**

- **Where:** `core/equations.py:3327-3350` builds
  `force_field = ExternalFieldConfig(E, B)`, a uniform config with zero
  gradient, and passes that to `evaluate_external_field_si`. The resulting
  `local_magnetic_gradient_t_per_m` is therefore always zero, and that is what
  `stern_gerlach_rest_impulse_native` receives
  (`core/equations.py:3440-3465`, `dipole_active and sg_active and not
  rfs_force_selected`). The provider branch added to `evaluate_external_field_si`
  (`core/external_fields.py:78-86`) is never reached on this path.
- **Failure scenario:** a spin-polarised beam with the default (non-RFS) SG
  model in a provider field with ∇B ≠ 0. The SG force is exactly zero, with no
  error or warning. The RFS route is unaffected, because it receives
  `provider_sample.partial_b` through `_external_tensor_gradient`.
- **Evidence:** code path above. `force_field` never carries
  `magnetic_field_gradient_t_per_m`.
- **Fix:** pass `gradient_t_per_m = provider_sample.partial_b[1:].T *
  1000 / magnetic_field_tesla_to_native(1)` into the SI tuple, or raise for
  provider + non-RFS SG. A raise is the smaller change. The channeling use case
  (charge only) is not affected today.

### P2: robustness or validation gaps

**P2-1: The default route is first order for spatially varying providers, and
this is not documented.**

- **Where:** `core/equations.py:3356-3390` applies h·qγ(E+β×B) sampled at the
  start position. The position update at `core/equations.py:4869-4877` uses the
  end momentum. The ½h²·dF/dτ correction at `core/equations.py:4523-4553` runs
  only inside `if second_order_exact_source_selected` (line 4342).
- **Failure scenario:** a channeling run with the AGENTS.md defaults (COLD_START,
  ~1200 steps). The trajectory, and net energy after the particle leaves a
  potential region, carry O(dt) errors. Example: −0.36% of K after a fringe
  reflection at 1201 steps (study C).
- **Evidence:** studies A and C.
- **Fix:** document in the `ExternalFieldConfig`/provider docstrings and in
  `retarded_integrator` that the provider is first order on the default route
  and second order only on the exact route. Better still, recommend the exact
  route for energy-loss claims.

**P2-2: The branch's harmonic validation cannot detect integrator energy
error.**

- **Where:** `diagnostics/external_field_provider/validate.py:361` fits damping
  to `e_rr / e_off`. This cancels the O(ω·dt) energy oscillation common to both
  runs. The only RR-off check (`:394`) requires that the error decreases, not
  that it is small or of a given order.
- **Failure scenario:** a regression that degrades the base integrator for
  provider fields would still pass the 0.093% damping check.
- **Evidence:** table A. At the branch's own 601/1201/2401 steps, the RR-off
  max|ΔH|/E_tr is 5.5% / 2.7% / 1.3%, comparable to the 3.9% RR signal being
  measured.
- **Fix:** report the RR-off max|ΔH| and the exit-energy order explicitly. Add
  the RR-on budget (ΔH − Σ RR work vs. RR-off drift), as in study B.

**P2-3: A plain-function provider combined with checkpointing raises
`TypeError`.**

- **Where:** `core/integration_runner.py:4616-4637` (and `:4231-4250`,
  adaptive exact pair) puts `external_field` into
  `_checkpoint_json_value`. That function (`:94-129`) rejects callables.
- **Failure scenario:** `ExternalFieldConfig(provider=my_function)` with
  `CheckpointConfig(enabled=True)` fails with `TypeError: unsupported checkpoint
  compatibility input function`.
- **Evidence:** reproduced. A dataclass provider such as `HarmonicChannel`
  passes, but it is fingerprinted only by field values, not by class identity
  (see P3-4).
- **Fix:** fingerprint the provider as `{qualname, module, repr or dataclass
  fields}`, or raise a clear `ValueError` that provider + checkpoint is
  unsupported.

**P2-4: Under `variable_geometry`, a time-dependent provider is sampled at an
inconsistent event.**

- **Where:** `core/equations.py:3306-3326`. When `sc_iteration > 0`,
  `field_position` is the working (trial) position, but `time_ns` stays at
  `current_state.t`.
- **Failure scenario:** a time-dependent provider combined with
  `self_consistency_convergence_mode="variable_geometry"` samples
  E(x_trial, t_start). The same mismatch already exists for windowed uniform
  fields. Static providers are unaffected.
- **Fix:** pair the working position with the working time, or reject
  time-dependent providers with `variable_geometry`.

### P3: nits

1. **Remainder timestamp moved from step midpoint to step start.**
   `core/equations.py:5240-5242` labels the remainder (sampled collective)
   force at the step start; the non-provider path labels it at the midpoint.
   This is consistent within the provider path, and with constant dt the
   derivative is unchanged. With adaptive dt it shifts the remainder derivative
   by O(Δh). Add a comment.
2. **Validity guard uses a different state.** The `medina_validity_guard`
   branch (`:5355-5356`) still passes the end-of-step `beta_tuple`/`result.gamma`,
   while Medina now uses `provider_beta`/`provider_gamma`. This only affects the
   O(h) guard threshold.
3. **Per-trial config construction.** `ExternalFieldConfig` is imported and
   constructed per particle per trial (`:3319-3330`), which reruns
   `__post_init__` validation. Consider building `force_field` with a light
   helper, or pass E and B directly.
4. **Dataclass provider fingerprint.** Checkpoint fingerprints of dataclass
   providers omit the class name, so two provider classes with identical fields
   would hash the same.
5. **Extra arrays on every Medina run.** `TrajectoryBuilder._MEDINA_FLOAT_FIELDS`
   (`core/types.py:2240-2242`) now allocates three (steps × particles) float64
   arrays on every Medina run (the default RR mode), even without a provider.
   This is harmless for parity, but costs memory and checkpoint size.
6. **Stale docstring.** `compute_uniform_external_field_impulse` still
   describes itself as "from a uniform field" but now handles providers.

## New tests on the branch

| Test | Verdict |
|---|---|
| `tests/unit/test_external_fields.py::test_uniform_provider_reproduces_existing_external_path[off, medina_lad]` | **Keep.** Cheap (64 steps). Guards the native-unit bridge and checks that the Medina hook is live (`radiation_energy_applied > 0`, ready flag set). |
| `tests/unit/test_external_potential_derivatives.py` new parameter `dict(provider=...)` | **Keep.** One line; ensures a provider is rejected by the uniform analytical-potential reduction. |
| `diagnostics/external_field_provider/validate.py`, `providers.py` | Not pytest tests. **Keep as diagnostics**; tighten them per P2-2. |

I added no tests. If P1-1 is fixed by raising, a three-line `pytest.raises`
test is justified. Otherwise, I don't think the owner's bar for new tests is
met.

## Merge recommendation

**Merge after:**

1. **P1-1:** fix the SG ∇B, or raise for provider + non-RFS SG.
2. **P2-1:** document the first-order default route and recommend the exact
   route for energy-loss claims.
3. **P2-3:** raise a clear error, or fingerprint callables.

P2-2, P2-4 and the P3 items can follow. The default path is byte-identical, the
Medina/LAD derivative is correct, and the RR damping matches analytic theory.
Physics claims about channeling energy loss made on the default route need the
exact route, or an explicit dt-convergence study of exit energy like study C.

## Appendix: reproduction sketch

```python
# python -I script.py <checkout>   (single particle, RR off, COLD_START)
from diagnostics.external_field_provider.validate import run, particle
from diagnostics.external_field_provider.providers import HarmonicChannel, SmoothTube
from core.types import ExternalFieldConfig
from core.constants import C_MMNS, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
prov = SmoothTube(depth=1e6, radius_mm=0.2, z_min_mm=0.05, z_max_mm=0.55, ramp_mm=0.1)
g0, L = 2.0, 0.7; beta = (1 - g0**-2) ** 0.5
for steps in (151, 301, 601, 1201):
    h = L / (g0 * beta * C_MMNS * (steps - 1))
    d = run(ExternalFieldConfig(provider=prov), steps=steps, h=h,
            state=particle(gamma=g0, x=0.1), mode="off", exact=False)  # exact=True: 2nd-order route
    kin = (d["Px"]**2 + d["Py"]**2 + d["Pz"]**2) / (M * (d["gamma"] + 1))
    print(steps, (kin[-1] - kin[0]) / kin[0])
```

The harmonic study uses `HarmonicChannel(M*ω²/Q)`, `x₀ = 1e-4·c/ω`, γ₀ = 1 (or
20), lab duration 10·2π√γ₀/ω, and `h = T/(γ₀(steps−1))`. H = kin + Q·φ.
