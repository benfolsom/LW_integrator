# Complete the magnetic-moment force's second-order time update

Status: 7 September 2026. Analytical force kernel and opt-in local dipole-source
derivative tested; **not enabled in the live stepper**. No radiation-reaction
or default changes. See the [source-derivative validation and remaining
fit-sensitivity failure](causal_directional_gradient_validation.md).

Follow-up: the rejected case's sensitivity is entirely in the acceleration
fit. A degree-six candidate now passes all six saved source checks with
unchanged limits and passes smooth-source controls. It has not been made the
default; the degree-five historical rejection remains recorded. Continue
with explicit candidate settings when testing the charge-source derivative
and combined force impulse.

## What this fixes

The ordinary Lorentz force already includes its first time derivative in the
second-order momentum update. The additional force on a magnetic moment still
uses only its initial value. Even with accurate source fields, this mismatch
leaves a local error proportional to the square of the timestep. Completing
the derivative should remove that leading contribution; it does not establish
global energy conservation or supply magnetic self-radiation.

The flyby study independently predicts the sign and size of that contribution
from the starting forces, without fitting the measured error. Its full record
is `studies/magnetic_dipole_electron_capture/planning/`
`moment_force_derivative_kernel_and_prediction_2026-09-07.md` in the sibling
feasibility-study repository. That record links the earlier isolation tests.

## Formula and data contract

Use coordinates $x^\alpha=(ct,x,y,z)$ and the metric $(+,-,-,-)$.
Let $u$ be four-velocity, $a=du/d\tau$ the physical total four-acceleration,
$s$ normalized spin, and $\mu$ the constant signed magnetic moment. Write
$G[\partial F,s]$ for the antisymmetric magnetic response already used by the
force routine. Because this response is linear in both arguments,

$$K_\mu^\alpha=\frac{\mu}{c}G[\partial F,s]^{\alpha\beta}u_\beta,$$

$$\frac{dK_\mu^\alpha}{d\tau}=\frac{\mu}{c}\left(
G[u^\kappa\partial_\kappa\partial F,s]^{\alpha\beta}u_\beta
+G[\partial F,\dot s]^{\alpha\beta}u_\beta
+G[\partial F,s]^{\alpha\beta}a_\beta\right).$$

These are, respectively, change of the source response along the particle's
motion, spin change, and observer acceleration. All three are required. The
caller must use acceleration and spin evolution from the selected physical
model, not a preceding-step average.

For a proper-time step $h$, the added impulse is $h^2\dot K_\mu/2$.
Coordinate-time stepping must retain the existing proper-time conversion.

The new `antisymmetric_response_moment_force_derivative_native` routine takes
the existing 4-by-6 packed field gradient plus its 4-by-6 directional rate.
Six entries store the independent components of an antisymmetric 4-by-4
response. The provider therefore needs only **24 additional contracted
coefficients**, not every component of the second field derivative. This
preserves the potential-based analytical route; separate electric and magnetic
three-vector fields are not introduced.

## Completed checks

- Compare against numerical differentiation of the independent dense force
  implementation at Lorentz factors 1, 2, 10, 100 and 1000.
- Check the differentiated orthogonality identity
  $u\cdot\dot K_\mu+a\cdot K_\mu=0$.
- Match the existing, broader potential-based derivative reference in
  `core/potential_jet_rfs.py`.
- On a manufactured smooth force path with an exactly integrable polynomial,
  confirm cubic local impulse error after including the derivative, versus
  quadratic error without it; also check input validation and zero moment.

The three focused core test modules passed **97 tests** on the Mac using an
isolated export of base `dbe2a2173e6ba9b2e78b7471d27a29910341d55d` plus this
kernel and its tests. These are local algebra tests, including high-gamma
inputs, not high-gamma trajectory validation or a performance benchmark.

## Next implementation steps and acceptance checks

1. Extend the **current causal local source provider** to calculate the
   directional rate of its field gradient. Reuse its selected source-history
   polynomial, retarded root, spin fit, and availability guards. Check that
   its existing potential, field response, and gradient remain unchanged.
   Implemented as an optional Python path. All ordinary outputs remain
   bitwise unchanged with versus without the option. The original degree-five
   fit accepts five saved-state checks and rejects the sixth. An explicit
   degree-six acceleration candidate accepts all six without weakening limits;
   this is not yet a default change.
2. Compare the new directional derivative against an independent check inside
   smooth regions and test boundaries and unavailable-history cases explicitly.
   Do not silently replace the current source model with the older quintic
   position/cubic spin provider just because that provider has higher
   derivatives. It represents different history data.
   Local polynomial, boundary, prefix-causality and collection tests now pass.
   The derivative differentiates the selected polynomial, not a refit at a
   shifted observer. Saved-state comparisons test this distinction but do not
   yet cover every history or scale transition. Also supply and validate the
   **charge-source contribution** to the same directional gradient: the
   moment responds to both charge-generated and dipole-generated fields.
   This charge contribution is now implemented and locally checked; see the
   [charge derivative report](charge_directional_gradient_validation.md).
   All six saved cases agree with same-segment numerical differentiation.
   Larger displacements crossing charge-history segments are not evidence for
   the local derivative and expose a boundary-handling concern for live steps.
3. After resolving the source checks, connect the derivative to the second-order moment impulse, including the
   selected acceleration and spin right-hand side. First run short, matched
   checkpoint comparisons with radiation reaction disabled, then enabled.
   Begin with a diagnostic combined charge-plus-dipole force correction using
   the explicit degree-six dipole candidate. Account for selected source
   segments and fit changes across the step; do not treat an unavailable
   derivative as zero or silently advance with only first-order moment force.
4. Require the isolated quadratic projection contribution to disappear under
   timestep refinement, preserve ordinary charge-only behaviour, and check
   general three-dimensional and relativistic cases. Only then reassess a
   bounded close-pass trajectory and cumulative error limits.

Do not subtract the predicted scalar projection error as a substitute for
correcting the four-force update. The historical flyby energy limits remain
failed, and the independent shell-conservation check remains open.

## Isolation and reproduction

The active worktree is `/home/benfol/compute/LW_integrator-moment-force-second-order`
on `feature/moment-force-second-order`, added to the Dell Zed workspace.
Mac tests used `/Users/benjaminfolsom/compute/lw-moment-force-prototype.cS99h4`
with the shared `sardana-dev` Pixi environment and explicit `PYTHONPATH`.
The shell campaign's clean reference checkout, `LW_integrator-radiation-flux-oracle`,
has not been changed.

Test command, run from the Mac export:

```sh
PYTHONPATH="$PWD" /Users/benjaminfolsom/.pixi/bin/pixi run \
  -m /Users/benjaminfolsom/work/environment-specs/maxiv-dev/pixi.toml \
  -e sardana-dev python -m pytest -q -o addopts="" \
  tests/unit/test_moment_force_derivative.py \
  tests/unit/test_antisymmetric_response_rfs.py \
  tests/unit/test_potential_jet_rfs.py
```
