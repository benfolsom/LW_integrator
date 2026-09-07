# Directional derivative of the local dipole source response

7 September 2026. Part of the [second-order moment-force plan](moment_force_second_order_plan.md).

## What is implemented

The existing local dipole source solver can optionally return the rate at
which its field gradient changes along the receiving particle's motion.
This supplies one of the inputs needed to complete the magnetic-moment force's
second-order time update. It does **not** apply that force correction yet.

Call the existing polynomial or causal-local provider with
`observer_four_velocity_mm_ns`. The returned
`partial_antisymmetric_response_along_velocity` has shape `(4, 6)` and contains
$u^k\partial_k\partial_l F_p$, with pair order 01, 02, 03, 12, 13, 23. Coordinates
are $(ct,x,y,z)$, so the four-velocity supplies the proper-time conversion.
Without this argument the new field is `None`, and no directional calculation
is performed. Empty or fully excluded collections return an explicit zero
array when a direction is requested.

Internally the existing third-order Taylor arithmetic carries a second
coefficient array representing one directional derivative. This is forward
automatic differentiation: each sum, product and reciprocal propagates its
derivative by the ordinary calculus rules. The scalar root, source polynomial,
spin representation and all ordinary arithmetic are retained. The root
iteration fixes the base root but must retain its directional variation.
Only the contracted gradient is returned, not a full second field-derivative
tensor. This transparent Python implementation is not a new compiled speedup;
the new internal coefficient calculations may be optimized after acceptance.

The returned derivative is of the **selected local polynomial**, not of the
whole moving-window fitting procedure. No higher-degree source model has been
silently substituted. Existing source derivatives already suffice for this
additional differentiation.

## Acceptance safeguards

When requested, the directional response participates in both nested-window
and adjacent-scale comparisons, using the existing relative-spread limits.
Thus a source that passes for fields alone may be rejected for the new
derivative. Scale selection can also change if the new comparison fails; the
reported scale remains authoritative. The optional result must not be paired
with an ordinary response from a different selected fit.

The separate `directional_light_cone_jet_residual` reports the residual Taylor
coefficients of the differentiated light-cone equation. Like the existing
unscaled residual, coefficients have different derivative units; it is a
diagnostic, not a single physical energy-uncertainty bound.

## Tests and saved-state checks

**176 focused tests passed on the Mac** (21.98 s). Black and Ruff pass for
all changed Python files. The seven modules cover the new directional
gradient, causal local sources and histories, Hertz responses, and the three
previous force-kernel/reference modules. Tests include:

- An independently root-solved observation-point stencil for fixed polynomial
  sources, at source speeds 0, 0.8c and 0.999c, with component and stereographic
  spin representations. Relative response-norm disagreement must be below
  2e-7; ordinary results must remain bitwise identical.
- Linearity and zero-direction checks, finite-input validation, source-order
  summation, exclusions, read-only collection output, and explicit rejection
  of inconsistent directional fits.
- Moving-window refits of an exact polynomial history, continuity across
  source segments and fit-window sample departures, and independence from
  later accepted samples for both acceleration-sampling options.

The read-only checkpoint probe then sampled both particles at inbound 12 pm,
closest approach and outbound 12 pm, using each event's saved source prefix.
Five of six cases pass the unchanged configured limits and preserve ordinary
outputs bitwise. The electron's shifted-observer comparisons improve under
refinement, reaching relative differences 7.18e-9, 1.92e-8 and 7.71e-9 at the
finest displacement. Proton comparisons are less monotone, reaching a few
parts per million; do not treat finer displacement as automatically better.

The sixth case, the outbound proton receiving the electron's dipole response,
is **unavailable**. The selected shortest window set (historical name
`atomic`) is internally stable, but its mandatory next-longer comparison
(`close`) fails its own nested-window test:

| Window-set comparison | Relative directional spread |
| --- | ---: |
| Shortest set, largest pairwise difference | 1.0613e-5 |
| Next-longer set, narrow versus primary | 2.6776e-5 |
| Next-longer set, primary versus wide | 1.1107e-3 |
| Next-longer set, narrow versus wide | 1.1195e-3 |
| Unchanged allowed spread | 1.0000e-3 |

The next-longer half-widths are 5e-10, 7.5e-10 and 1.25e-9 ns. The widest fit
causes most of the disagreement; still wider sets are worse. Fit condition
numbers remain similar, approximately 9.4e3--9.5e3, so these data do not show
a sudden loss of matrix conditioning. They do not yet distinguish finite
window bias from imperfections in the stored acceleration/spin histories.
Raw fixed-fit inspection does not override the rejection.

## Next checks before enabling the force correction

1. At the rejected event, separate acceleration-history and spin-history
   contributions to fit sensitivity. Use controlled smooth histories and
   window/degree comparisons; do not simply loosen the limit or discard the
   required adjacent comparison.
2. Validate the charge-generated contribution to the same directional
   gradient, retaining the charge provider's actual history and root policy.
   The magnetic moment responds to both source components.
3. Combine the accepted directional gradients with the tested force kernel,
   then run short impulse/refinement checks before any new bounded close pass.

This remains separate from magnetic self-radiation and the shell conservation
campaign. No energy gate is closed here.

## Evidence locations

Core tests: `/Users/benjaminfolsom/compute/lw-moment-force-prototype.cS99h4`,
using the shared `sardana-dev` Pixi environment. See the parent plan for its
environment command; add the four source/history/Hertz test modules listed
above to the previous three force-reference modules.

The study repository contains the two runnable diagnostics under
`studies/magnetic_dipole_electron_capture/diagnostics/`:
`probe_local_directional_gradient.py` and `inspect_local_directional_fit.py`.
Their reports are archived under the same study's
`planning/evidence/local_directional_gradient_2026-09-07/` and retain runner,
checkpoint, comparison and core-source hashes. The raw Mac reports are
`/Users/benjaminfolsom/compute/lw-magnetic-checkpoint-replay-20260907/`
`local_directional_gradient.json` and `local_directional_fit_inspection.json`.
