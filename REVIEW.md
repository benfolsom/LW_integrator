# Review: `review/photon-transport` (1d580d4) against `development` (faad1ae)

Independent review of the opt-in photon transport added on top of stochastic
quantum emission. Scope: `core/photon_transport.py`,
`core/photon_transport_adapters.py`, and the runner, CLI and testbed wiring.
Line numbers refer to commit 1d580d4.

## Merge recommendation

**Do not merge as-is.** The kernels are correct. The physics-off default is
byte-identical. Restarts with live photons are identical. Two geometry
defects in how targets are presented to the transport loop make enabled
scattering rates wrong, though:

- **P1-A:** the general (sampled) runner compares photons and targets at
  different lab times. The error is unbounded and grows with the run.
- **P1-B:** in both runners, targets are frozen within a charge step. Rates
  then depend on the step size, by O(1) factors once `|v|·dt ≳ R`.

Minimum to merge:

- Fix P1-A, or refuse `photon_transport` on the general runner until it is
  fixed. The pair runner is not affected by P1-A.
- Fix P1-B (interpolate target positions), or refuse runs with
  `max |v_target|·dt > R/4`.
- Add the regression tests that the AGENTS.md bug-fix rule then requires.

P2 performance makes the general path impractical for photons above
2 m c² with pair conversion on. Fix it or document it before anyone relies
on that path.

No P0 findings.

## What was verified (evidence)

All checks ran on an archive of 1d580d4 (and of faad1ae for parity), Python
3 venv, `pip install -e ".[dev]"`. Scripts are in `review_checks/`.

| Claim | Result |
|---|---|
| KN total cross-section | Matches direct quadrature of dσ/dΩ to ≤3e-10 for x = 1e-5…1e6, including both sides of the x = 1e-3 series switch. r_e = 2.81794e-12 mm and σ_T = 6.6525e-23 mm² are correct; 1/α = 137.036. |
| KN sampling | Rest-frame energy-ratio KS tests at x = 0.05, 0.99, 1, 3, 30, 300, 1e4 give p = 0.06–0.85 (40k samples each; the Geant4 mixture branch is correct). An early p = 0.002 at x = 300 was an artefact of my μ-grid reference; the ε-grid reference passes. The x < 1 uniform-μ rejection has acceptance ≤ 1. |
| Compton kinematics | The lab invariant k·P = k′·P + k·k′ is implemented exactly. Worst relative four-momentum residual over 1500 random events (γ up to 1e5, near-collinear included) is 8.3e-16. Head-on γ = 10 maximum E′/E = 382.7 matches (γ(1+β))²/(1+2Eγ(1+β)/mc²). |
| Rate formula | `W = σ_KN(E*) c (1 − β·n̂)/V` with `E* = E γ(1 − β·n̂)` is the correct lab collision rate for a lab-frame density. Static targets are exactly step-independent (table below). |
| χ_γ | `(E/mc²)|E⊥ + n̂×B|/E_S` with Gaussian `E_S = m²c³/(eħ)`: χ = 1 at E = E_S and E_γ = mc². Co-propagating plane wave → 0; longitudinal E → 0; counter-propagating crossed field → 2×. |
| BW kernel | The bracket and prefactor agree with the SMILEI/Baier–Katkov form after the substitution x^{3/2} = 1/(χ f(1−f)). T(χ) agrees with an independent `quad` of the full kernel to ≤1e-6 for χ ≥ 1. The small-χ ratio to (3/16)√(3/2)e^{−8/3χ} → 0.992 at χ = 0.05; T·χ^{1/3} → 0.373 at χ = 1e3 (asymptote ≈ 0.38). Split sampling KS p = 0.59, ⟨f⟩ = 0.5007. |
| Ledger (runtime level) | Six photons with pair, scattering and escape outcomes. Σ target E + Σ photon E (escaped/alive) + Σ (E₋ + E₊ − Q₀) closes to 0.0 relative. Momentum closes to 1.4e-16. |
| Default-off parity | General CLI run (e⁻ 5 GeV vs protons, emission on, 120 steps): output JSON is byte-identical between faad1ae and 1d580d4, with the key absent, with `{"enabled": false}`, and with emission off. Pair runner: identical SHA-256 of the `advance_pair` payload and records on both commits. With transport on and no interactions, the charged outputs are identical to transport off (both runners). |
| Restart | Runtime JSON round-trip mid-run with RNG-consuming pair and scattering channels: identical. Pair runner, 4 steps vs 2 + JSON + 2 with a live photon: identical. General runner: simulated crash at step 75, resume from the step-60 checkpoint with 2 live photons, final report identical to the uninterrupted run. |
| Existing tests | 240 passed, covering `test_stochastic_emission`, `test_accepted_pair_checkpoint`, `test_momentum_center_pair`, `test_nonlinear_pair_cli`, `test_full_spin_initialization`, `test_simulation_options`, `test_cli`, `test_testbed_config_loader` and `test_config` (`pytest -p no:anyio`). I did not rerun the full 291. |

## Findings

### P0 — wrong default behaviour

None found. The parity evidence is above.

### P1 — wrong physics when enabled

**P1-A. The general runner pairs photons with targets at mismatched lab times**
(`core/photon_transport_adapters.py:162-200`).

`commit_general_transport` takes every target from `history[-1]` at that
particle's own lab time. It then advances all photons to
`endpoint = max(t)` over both bunches (line 199). The sampled general
integrator steps a common *proper* time, so rider and driver lab clocks
diverge by about (γ_r − γ_d)·h per step.

Evidence, from `review_checks/e2e` configurations:

- e⁻ rider at γ = 9786 with a proton driver at T = 1 GeV (γ = 2.07),
  120 steps.
- Final rider t = 0.0845 ns; final driver t = 2.95e-5 ns.
- The photons sit at t ≈ 0.0845 ns, but the driver targets are frozen at
  z = 5.008 mm. At the photon's time the driver would be at z ≈ 27.2 mm.
  That is a 22 mm offset against R = 1 mm.

So scattering with any moving target of different γ is computed against the
wrong geometry. Whether a photon hits or misses is essentially arbitrary,
and the error grows linearly with run length. Recoil is applied at the
target's own time, but the event is stamped with the photon's time.

The same mismatch makes `_sampled_photon_fields` (lines 114-153) return the
latest available row when the source history lags the true retarded time.
See P2-C.

Fix: evaluate each target at the photon substep time from its own row
history, interpolating between bracketing rows. Advance photons only to the
causally safe barrier `min(t)`, not `max(t)`. Until then, refuse general-path
transport unless all targets share a lab clock, or restrict transport to
the pair runner.

**P1-B. Targets frozen within a charge step make the scattering probability
step-size dependent** (`core/photon_transport.py:400-421`, docstring 371-375;
adapters 163/224).

Targets are held at the accepted *endpoint* position for the whole
transport interval. The flux factor `(1 − β·n̂)` assumes the target density
moves with the target. With a frozen sphere, the expected count becomes
σ(1 − β·n̂)∫ρ ds instead of σ c(1 − β·n̂)/|c n̂ − v| ∫ρ ds_rel. The ratio is
|n̂ − β| once the target moves more than about R per interval.

Deterministic probe (`review_checks/frozen2.py`) records every Bernoulli
W·dt through the real `advance`. Values are expected count / exact answer:

| geometry | c·dt = 20 mm | 4 mm | 1 mm | 0.25 mm | 0.05 mm |
|---|---|---|---|---|---|
| static | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| head-on, γ = 10 | **1.995** | 1.097 | 0.997 | 0.997 | 0.997 |
| transverse, γ = 10 (exact 0.704) | **0.000** | **0.000** | 0.500 | 0.725 | 0.700 |
| co-moving, γ = 3 | **0.226** | 0.789 | 0.992 | 1.001 | 1.001 |

The `max_step_mm` and R/4 caps bound only the photon substeps. They do not
bound the charge interval over which targets are frozen.

- **Pair runner:** steps are short (for example, 1e-4 ns gives c·dt =
  0.03 mm), so it is in the converged regime.
- **General runner:** the AGENTS default (1000 mm, 1200 steps) gives
  c·dt ≈ 0.4–0.8 mm, which is borderline. Larger steps are badly off.

The docs say "Targets are frozen … converge the charge timestep", but there
is no guard. The claimed KS p = 0.19 Compton check cannot detect this if it
used static targets.

Fix: linear interpolation of target positions between the previous and
current accepted rows at each substep start. Both rows are available in
both runners. Alternatively, refuse `max|v|·Δt_charge > R/4`.

### P2 — robustness and usability

**P2-A. General-path field queries cost O(history × particles) in Python on
every photon substep** (`core/photon_transport_adapters.py:114-153`).

One query with 2×20 particles and 1200 rows takes 0.44 s. A 1000 mm
crossing at `max_step_mm = 0.1` is about 1e4 queries, so about 1.2 h per
photon. Each query also allocates a `_dead_particles` default per row, and
the runner copies `trajectory[: i + 1]` every step
(`core/integration_runner.py:5427`). This path is effectively unusable for
photons above 2 m c² with `pair_conversion` on.

Fix: cache per-source time and position arrays once per step, and bisect
for the null-time sample. Also skip field queries when no plausible field
can reach χ above a floor.

**P2-B. `breit_wheeler_spectrum` recomputes 4097 Bessel evaluations
(about 2.8 ms) on almost every substep** (`core/photon_transport.py:227-241`).

The `lru_cache` is keyed on exact float (χ, E), so it essentially never
hits. Only the total rate is needed per substep; the CDF is needed only on
an accepted conversion. Use a tabulated T(χ) for the rate (the existing
`bessel_integral` table pattern), and build the CDF only when a conversion
is sampled.

**P2-C. A stale field sample is silently accepted** (adapters:137).

`min(residual)` over causal rows picks the latest row even when that row is
far inside the past light cone, because the source history has not yet
reached the true retarded time. See P1-A. Refuse samples whose residual
exceeds the source's row spacing, or report them, rather than silently
using a stale field.

**P2-D. The run-level ledger is incomplete** (adapters:89-111;
`lw_integrator/testbed_runner.py:2192`).

Per-event four-momenta are exact, and escaped photons keep their final
energy, so a post-hoc ledger closes (verified to 1e-16). However:

- Compton recoil on the general path is not written to
  `radiation_energy_applied` or any per-step field. Emission recoil is
  (`core/stochastic_emission.py:663`).
- No summary of photon energy by status, Σ target ΔE from scattering,
  pair energy, or Σ Q is reported.

So an energy-ledger series shows unexplained kinetic jumps. Add a compact
`photon_transport.ledger` block, plus a per-step target field for
scattering energy exchanged.

### P3 — nits

1. **Bernoulli first-order bias.** `core/photon_transport.py:475` is
   documented, but it biases the optical depth by about `max_probability/2`
   (0.5% at the default). Exact optical-depth or exponential sampling costs
   the same and removes the bias, along with the end-of-substep event
   placement.
2. **Self-scattering not excluded.** The emitter is not excluded from its
   own photon's targets (lines 400-421). The probability is about
   σ_T/R² ≈ 1e-22 for physical electrons, so negligible, but it is
   conceptually a double count.
3. **Photon order within an interval.** Photons are advanced one at a time
   to the endpoint (line 380). A target recoiled by photon A at time t_A is
   seen with its post-recoil momentum by photon B at times before t_A. This
   is a second-order effect at physical rates.
4. **Low-χ BW accuracy.** The claimed "BW rate error < 8e-7" holds for
   χ ≥ 1. At χ = 0.05 it is 2.3e-4, from the `bessel_integral` table tail
   and the grid; the rate there is about 1e-24 of the χ = 1 value.
5. **Spurious resume error.** `configure_checkpoint`
   (`lw_integrator/nonlinear_pair.py:165-172`) raises "Resume retains photon
   transport settings and RNG state" when a user passes
   `{"enabled": false}` for an evolved checkpoint that never had transport.
   Reproduced on the forced-kick fixture with accepted_steps = 12. Treat a
   disabled config with `previous is None` as a no-op.
6. **Local path in the docs.** `docs/source/photon_transport.rst:183` cites
   `local/task_av/report.md`, which is not in the repository. The parent
   commit faad1ae ("drop local note paths") removed exactly this kind of
   reference.
7. **Default seed.** `PhotonTransportConfig.seed` defaults to 0, so every
   run shares one interaction stream unless a seed is set. Consider
   requiring an explicit seed when `enabled`, or deriving one from the
   emission seed.
8. **Testbed payload.** `SimulationOptions` keeps an explicit
   `{"enabled": false}` dict in its serialized payload
   (`lw_integrator/testbed_runner.py:923`). It is harmless to physics,
   but it differs from the no-key form in saved configs.

## Tests (owner rule: new tests only when strongly justified)

1. **KN/LCFA kernel regression: justified, keep it to one file, under 1 s.**
   - The commit adds about 840 lines of new physics with zero tests.
   - The kernels are pure and deterministic.
   - BW silently depends on `stochastic_emission.bessel_integral`, so a
     change to that shared table would alter pair rates without any failing
     test.

   Assert KN σ against quadrature at a few x, including both sides of
   1e-3. Assert per-event Compton four-momentum conservation at large γ.
   Assert T(χ) against the small-χ asymptote and an independent `quad` at
   χ = 1 and 10. Leave KS statistics out of the suite.
2. **Live-photon checkpoint regression: justified only as an extension, not
   a new test.**
   - The runtime round-trip is a deepcopy and low-risk.
   - The real risk is runner wiring: where the payload is saved and
     restored.
   - Adding `photon_transport` to the existing
     `test_pair_checkpoint_resume_is_bit_identical_after_forced_photon_kick`
     fixture costs about 1 s and covers the pair runner.
   - A general-runner crash/resume test (about 10 s) is not strongly
     justified while P1-A stands.
3. **Bug-fix regression tests are required by AGENTS.md:** the fixes for
   P1-A and P1-B each need a regression test. A small deterministic W·dt
   probe like `review_checks/frozen2.py`, plus a two-clock general-runner
   geometry check, would cover both.

## Reproduction

`review_checks/` contains:

| Script | What it checks |
|---|---|
| `kernels.py` | KN, Compton, BW and χ |
| `kn_hi.py` | High-x KN sampling |
| `frozen2.py` | P1-B table; `L=300` for the co-moving row |
| `restart.py` | Runtime restart and ledger |
| `pair.py` | Pair parity and restart; run in both trees |
| `crash.py` | General-runner resume |
| `chi_probe.py` | Emitter self-field χ, which is 0.0 as expected |
| `e2e/*.json` | CLI configurations |

Run from the root of a 1d580d4 checkout with the venv Python.
