# Review: `review/kick-knots` (f842f40) against `development`

Scope: one commit, *Record stochastic kicks as split velocity knots on the exact
path; lift the guard*. Diff base is merge-base `2b6f8bf`. `development` has
since gained only release/changelog commits (`3032cb4`).
Touched: `core/exact_visibility_history.py`, `core/retarded_fields.py`,
`core/stochastic_emission.py`, `core/integration_runner.py` (guard removed),
docs, CHANGELOG, and `tests/unit/test_stochastic_emission.py`.

**Recommendation: merge after a small fix for P2-1.** P2-2 (cost) should be
tracked as a follow-up and stated in the docs before anyone runs production
exact-path emission studies. I found no P0 or P1 defects. The central physics
claim reproduces exactly on all three charge backends.

Line numbers below refer to f842f40.

---

## Validation performed

All runs used a fresh venv with `pip install -e ".[dev]"` and Python 3.13.

| Check | Result |
|---|---|
| `pytest -p no:anyio tests/unit/test_stochastic_emission.py test_exact_visibility_gates.py test_inertial_prehistory.py test_light_cone_history.py test_resolved_knot_checkpoint.py test_exact_retarded_charge_numba.py` | 85 passed, 11 deselected (slow) |
| `pytest -p no:anyio tests/unit/test_retarded_fields.py test_exact_same_bunch_fields.py test_accepted_pair_checkpoint.py test_retarded_dipole_source_integration.py` | 49 passed |
| `black --check` on the touched files | clean |
| `mypy core/exact_visibility_history.py core/stochastic_emission.py` vs dev | **4 new errors** (see P3-1) |
| Probes (`review_probes/kick-knots/`) | see the findings and the evidence section |

### Evidence for the physics claim (passes)

`probe_synthetic.py` builds a coasting source with β = (0, 0, 0.8) before
t = 1 ns and (0, 0.02, 0.7) after it. The kick is published exactly as
`commit_general_state` publishes it. A passive observer at (30, 0, 120) mm sees
241 events within ±0.3 ns of the kick's light cone, including the event on the
cone. Each field is compared with the field of the matching unkicked ballistic
branch.

| dt (ns) | with knots: max ‖ΔE‖/‖E‖ | with knots: Σ‖ΔE‖² dt | metadata stripped (pre-fix): max rel | stripped Σ‖ΔE‖² dt |
|---|---|---|---|---|
| 0.1 | 0 | 0 | 1.14 | 5.9e-12 |
| 0.05 | 0 | 0 | 2.27 | 9.8e-12 |
| 0.025 | 0 | 0 | 4.52 | 1.8e-11 |

The same exact zeros were measured with `BACKEND=numba_full_strict_serial` and
`numba_analytic_charge_response_serial`. The stripped column reproduces the
original bug: peak and fluence roughly double with each halving of dt. On the
cone itself the root is assigned to the outgoing branch, consistently.

`probe_accel.py` places a kick between two smoothly accelerated branches with a
constant lab force. It compares the kicked history with each analytically
extended branch at the same dt. The residual converges, with no pulse:

| dt | max rel ‖ΔE‖ | fluence |
|---|---|---|
| 0.1 | 5.5e-5 | 9.0e-21 |
| 0.05 | 1.0e-5 | 2.1e-22 |
| 0.025 | 1.8e-6 | 4.8e-24 |

On the non-light-cone quintic path used by dipole sources
(`light_cone_enabled=False`), `_quintic_worldline_sample` returns exactly the
incoming velocity at the end of segment k−1 and exactly the outgoing velocity at
the start of segment k. Both one-sided accelerations converge.

`bystander.py` puts an unkicked source in the same history as a kicked one. Its
prepared coefficients, primes and light-cone samples are bit-identical to those
from a history without kick metadata.

### Default-behaviour parity (passes)

`parity.py` was run on dev (`2b6f8bf`) and on this branch and every numeric
trajectory array was compared bytewise:

- An exact-path gate switch with no emission (`crossing_run(8, gate(radius=1))`,
  default `light_cone_quintic`): **bit-identical**, 1571 arrays. This covers the
  change from `beta_primes[sl]` to `primes` in `constrain_proper_velocity_segment`
  (`retarded_fields.py:518`). Gate splitting refuses `proper_velocity` histories
  (`NotImplementedError`), so that change is reachable only through kick knots.
- The sampled path (COLD_START) with one forced photon: all numeric arrays are
  **bit-identical**. The only difference is that the kicked row's
  `potential_inclusion_state` now carries `exact_gate_history` (see P3-4).

---

## P0: wrong default behaviour

None found.

## P1: wrong physics when enabled

None found. The checks listed in the brief are below.

- **Two-branch reconstruction.** `_branch_kinematics` swaps in the incoming
  β and resolved data only at the right end of a branch that ends on a kick.
  Branches start from the stored outgoing knot. Position is continuous because
  the kick row's x/t are shared. The quintic pieces (`retarded_fields.py:1343-1352`)
  and the light-cone segments (`_build_null_segments`) both use the per-branch
  velocities. One-sided primes are written back only where the gate slot is NaN
  (`:1323-1331`), so existing gate one-sided data is preserved.
- **Light-cone segment selection and root search.** The light-cone residual is
  monotone for subluminal sources. `kick_boundary` is accepted only when the
  ballistic root is at or after the newest kick knot and within
  `extrapolate_ns`. Because of monotonicity, that test agrees with the root of
  the real worldline. Otherwise control falls through to the segment solver
  (`:2569-2597`). The ±0.3 ns sweep, including the event on the cone, matched
  the branch fields exactly.
- **Derivative samples at the knot and JSON null.** Partially non-finite
  primes serialize as `null` (`exact_visibility_history.py:65`). They read back
  as NaN, which means "reconstruct", and are never treated as zero acceleration.
- **Medina restart.** `_clear_medina_force_history` was already present on dev,
  and the commit does not change it. The parametrized restart test covers python
  quintic, `proper_velocity` and compiled runs bit-identically through a
  post-kick checkpoint.
- **Checkpoint compatibility.** Old payloads, which always store lists for
  `before`/`after`, still load. See P3-5 for the forward direction.

## P2: robustness

### P2-1: per-particle merge can misalign a bunch's knot lists → `ValueError` crash
`core/exact_visibility_history.py:73`

`save_visibility_knots(..., append=True)` decides **per particle** whether the
kick entry merges into the last existing gate knot or is appended. The rule is
`entries[-1]["t"] == entry["t"]`. `_expand` requires every particle in a bunch
to have the same number of knots (`:145`).

Failure scenario: an exact run with particle loss, beamline gates and
stochastic emission. A dead particle copies its previous state
(`core/equations.py:2383`), so its time is frozen. In a step where an interior
gate switch happens and an alive particle in the same bunch emits:

- for the alive particles, the gate knot is interior and the kick entry is
  **appended** (n+1 knots);
- for the dead particle, gate-knot t equals its frozen endpoint t, so the entry
  is **merged** (n knots).

The next field preparation then raises
`ValueError: gate source knots must be aligned across a bunch`.

Evidence: `review_probes/kick-knots/align.py` builds a two-particle endpoint
whose gate knot coincides with the endpoint for one particle only. It applies the
kick save, and `_expand` raises exactly this error. I did not reproduce it in a
full runner session; the trigger is a narrow combination.

Suggested fix: make the coincidence decision once per bunch, using non-dead
particles, and apply it to every particle. Alternatively, merge only when the
gate history's last knot *is* the endpoint knot for every particle that is not
dead.

### P2-2: one kick makes the rest of the run rebuild the full history every step (O(N²))
`core/retarded_fields.py:1704-1715`, and the dipole equivalent at
`core/retarded_dipole_fields.py:744-767`

Kick metadata routes `_prepare_history` into the `gate_knots` variant. Its
`append` callback calls `_prepare_history_uncached(expanded)`, and `_expand`
re-decodes the JSON of every row on every append. Gate knots were rare, so
this was tolerable. Every stochastic photon now leaves a permanent knot, so
after the first emission the incremental append cache is never used again.

Measured with `perf.py`: the exact-path python backend, one forced kick at
step 1, then otherwise the same run as the control.

| steps | no emission | one kick |
|---|---|---|
| 160 | 6.5 s | 19.0 s |
| 320 | 12.2 s | 60.0 s |

The control scales linearly; the kicked run is quadratic. At the default
~1200-step test length this is roughly 10–15× slower. Every particle in the
bunch also gets a full knot entry for each kick step (about 30 floats plus 14
`resolved_before` values), which grows checkpoint metadata.

This does not block correctness. Before production use, give kick-only metadata
an incremental append path: only the kick's neighbouring segments change. Until
then, note the cost in `stochastic_quantum_emission.rst`.

## P3: nits

1. **mypy regressions (4 new errors).** The errors are at
   `exact_visibility_history.py:65` (`None` assigned to a `float`-typed dict
   value) and `retarded_fields.py:545`, `:1335`, `:1368` (Optional arrays
   indexed without narrowing). `:1368` also indexes `history.resolved_knots`
   with no `None` guard. Every current kick history comes through
   `_extract_history`, which always sets it, so this is latent. An `assert` or
   an explicit fallback would fix both the typing and the hazard.
2. **The newest kick knot's outgoing acceleration is 0 until the next row**
   (`retarded_fields.py:1332-1336`). Evaluations that extrapolate past the
   newest knot use a ballistic outgoing branch. These are the endpoint-potential
   and trial evaluations, and they go through `kick_boundary`. When the next
   row arrives, the outgoing prime is reconstructed retroactively. This matches
   the "neighbouring one-sided derivatives may change" contract and is
   consistent with Medina. A one-line note in the docs would help anyone
   comparing provisional and final fields.
3. **Instantaneous-semantics scope widening**
   (`retarded_fields.py:1286-1289`). If any source in the history has a kick,
   the branch reconstruction runs for **every** source. Under
   `source_acceleration_semantics="instantaneous"`, that replaces the stored
   exact `bdot` of unkicked sources with reconstructed derivatives. No current
   instantaneous caller sees kick metadata (`finite_magnetic_source_force` uses
   analytic histories), so this is latent. Consider gating per source on
   `kicks.any()`.
4. **The sampled path now stores kick knots too**
   (`stochastic_emission.py:689` is unconditional). Trajectories are unchanged.
   Post-run exact diagnostics on a sampled trajectory would now see the split,
   which is arguably better, but this behaviour should be stated or gated
   explicitly.
5. **Forward compatibility.** `null` primes and the
   `velocity_before`/`resolved_before` keys cannot be read by pre-branch code:
   `np.array` over `None` produces an object array. That is acceptable for an
   additive feature, but a schema bump or changelog line would make it explicit.
6. **Gate and kick at the same endpoint drop the gate's exact `after` prime**
   (`exact_visibility_history.py:79`). This is defensible, because the gate's
   post-switch force was computed at the pre-kick velocity. A comment saying so
   would stop a future reader from "fixing" it.
7. **Per-source copies of the full `[steps, sources, 3]` gate-prime arrays**
   (`retarded_fields.py:1302-1307`) cost O(N·S²). At N = 200 and S ≤ 100 they
   add about 30% to preparation (`scale.py`: 6.2 s → 8.2 s at S = 100);
   it grows with bunch size. Copying only the one source's column would avoid
   this.
8. **Test structure.** The general-exact restart checks were folded into
   `test_pair_checkpoint_resume_is_bit_identical_after_forced_photon_kick` with
   an early return (`tests/unit/test_stochastic_emission.py:174`), so the pair
   test's name now covers unrelated runners. The headline claim, zero
   additional classical field across the kick cone under refinement, has no
   field-level regression test; the test checks only knot velocities and
   nonzero accelerations. Under the owner's rule on new tests, I think one
   strongly-justified addition is warranted: a coarsened `probe_synthetic.py`
   asserting `E == E_branch` exactly at two dt values. It takes under a
   second.

---

## Probe scripts

These live in `review_probes/kick-knots/` and run from a repo checkout with the
working directory at the repo root.

- `probe_synthetic.py [--strip]`, with `BACKEND=` env: coasting kick, with and
  without kick metadata.
- `probe_accel.py`: accelerated branches, plus the quintic (dipole) path.
- `bystander.py`: unkicked source parity inside a kicked history.
- `align.py`: P2-1 misalignment reproducer.
- `perf.py`: P2-2 timing.
- `scale.py`: preparation cost with and without kick metadata (P3-7).
- `parity.py <out.npz>`: run in the dev and branch trees and compare npz
  arrays bytewise.
