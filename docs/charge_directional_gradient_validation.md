# Charge contribution to the magnetic-force time correction

7 September 2026. Part of the [second-order moment-force plan](moment_force_second_order_plan.md),
alongside the [dipole-source derivative checks](causal_directional_gradient_validation.md).

## Result and scope

The charge solver can now optionally calculate how its field gradient changes
along the receiving particle's motion. A magnetic moment responds to this
charge-generated gradient as well as to the dipole-generated gradient.
Both contributions are needed to correct the force over a finite timestep.
This change supplies the charge contribution; **it does not advance a
trajectory or change the live force update**.

The new optional argument is `observer_four_velocity_mm_ns` on
`evaluate_retarded_charge_response_gradient_native`. The returned
`partial_antisymmetric_response_along_velocity` contains
$u^k\partial_k\partial_l F_p$ as a 4-by-6 array. The six columns store pairs
01, 02, 03, 12, 13 and 23 of the antisymmetric response; coordinates are
$(ct,x,y,z)$. No separate electric or magnetic three-vector calculation is
introduced.

## Reuse and availability

The calculation takes the exact scalar retarded times, segment indices and
prepared quintic coefficients used by the ordinary compiled charge response.
It performs no additional scalar root solve and retains the caller's choice
of reconstructed versus supplied instantaneous acceleration. Analytical
differentiation propagates through the implicit retarded-time equation.

The potential arithmetic is shared with the existing broader directional
reference, but needs third rather than fourth order and returns only the
24 consumed values. Its internal Taylor arithmetic is still transparent
Python, not a new compiled performance improvement. With the option absent,
none of this extra calculation runs.

If the ordinary provider needs its maintained finite-difference fallback,
the new derivative is `None` with `directional_unavailable_reason`. It is
never silently zero-filled. Fully excluded sources contribute genuine zero;
missing required history remains an error. Returned derivative arrays are
read-only. The per-source `directional_jet_residual` is an unscaled residual
of the differentiated light-cone equation; its coefficients have different
derivative units, so it is not an energy-error bound.

## Validation

Mac export: `/Users/benjaminfolsom/compute/lw-charge-directional-prototype.3rWBZS`,
base `7e04b4b` plus this change. Shared Pixi `sardana-dev`, explicit `PYTHONPATH`.
The 11 focused core modules pass **213 tests** (5.93 s with warm caches).
Black and Ruff pass for all three changed Python files.

The 21 new checks include an exact static Coulomb derivative, independent
five-point differences of the existing stable charge response at Lorentz
factors 1, 2, 10, 100 and 1000, accelerating and inertial sources, both
acceleration conventions, sums and exclusions, unchanged ordinary outputs,
root reuse, boundaries and invalid inputs. The high-gamma cases use an
off-axis observation direction; they are not an exhaustive beaming-cone,
boost-covariance or high-gamma trajectory benchmark. Agreement with the
broader potential reference is an additional consistency check, not an
independent derivation, because that reference shares the algebra.

The study diagnostic `diagnostics/probe_charge_directional_gradient.py`
uses the actual charge-provider settings and past-only source prefixes from
the previously reproduced flyby checkpoint. It compares inbound 12 pm,
closest approach and outbound 12 pm, for both receiving particles:

- All six cases return the derivative without fallback.
- Ordinary outputs, scalar roots and segment indices remain bitwise equal
  with versus without the optional calculation.
- Across all numerical comparisons staying on the same source segments,
  the largest response-norm relative discrepancy is **1.471e-9**.
- Larger displacements often cross segments; their discrepancies reach
  **7.65%**. Those comparisons are recorded, not counted as a local pass.

The read-only report, including exact source/checkpoint/helper hashes, is
`/Users/benjaminfolsom/compute/lw-magnetic-checkpoint-replay-20260907/charge_directional_gradient.json`.
SHA-256: `9db560aaed5b8f18099665419f004194a4a6eb9bd5dd23e9bae93703e0a97669`.
An archival copy and reproduction command are in the study's
`planning/charge_directional_gradient_2026-09-07.md` and linked evidence.

## Next decision

Combine the tested charge and dipole derivatives with physical observer
acceleration and spin evolution in a short diagnostic force-impulse check.
Use the explicit degree-six dipole candidate, keeping its acceptance checks.
Then compare timestep refinement with radiation reaction off and on, with
matched force-memory initialization. Test segment boundaries and fit changes
explicitly before authorizing unrestricted live steps. An accurate derivative
inside one segment does not guarantee a valid Taylor update across a join.

The old flyby energy limits remain failed. No new full flyby, global
conservation claim, default fit change or magnetic self-reaction change is
part of this milestone. The shell campaign's clean `dbe2a21` reference stays
unchanged.
