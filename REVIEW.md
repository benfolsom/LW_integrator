# Review: `review/stochastic-quantum-emission` vs `development`

Reviewed commits: `afab457` (opt-in stochastic emission) and `e053153` (LCFA
unification), `git diff development...review/stochastic-quantum-emission`
(13 files, +1480/-16). No test files are touched by the branch.

Environment used for reproduction (Dell, local): a throwaway venv
(`/tmp/lwrev`, Python 3.12, numpy 2.5.3, scipy 1.18.1, numba 0.68) with the
worktree on `PYTHONPATH`; scripts referred to below lived under `/tmp/lwrev_*.py`
and are quoted inline where load-bearing. Nothing was pushed to `development`
or to the review branch.

## Summary

The physics module (`core/stochastic_emission.py`) is correct as far as I can
check numerically: alpha, chi, the Ritus/Baier-Katkov spin-averaged LCFA
number spectrum, its classical and high-chi limits, the Erber Gaunt factor,
the recoil ledger and the mass shell all reproduce textbook values to the
precision expected. The default-off path is byte-identical to `development`
(182 trajectory arrays compared bitwise on a sampled B2B run with an external
field and Medina RR). Seeded runs are reproducible, checkpoint restart
reproduces an uninterrupted run exactly in both runners, and a seed change on
resume is rejected by the checkpoint fingerprint.

The weaknesses are (a) the hand-off rule between Medina/LAD and the stochastic
model introduces a deliberate but undocumented ~35 % step in mean radiated
power at the default threshold, (b) a run dies with `ValueError` rather than
sub-stepping when `W dt` exceeds the cap, (c) the branch ships ~1500 lines of
new physics with zero test coverage, so "426 tests pass" says nothing about
the enabled path, and (d) a few robustness nits. No P0 found.

## P0 (crash or wrong physics on the default path)

None found.

Evidence for the default path:

* `/tmp/lwrev_e2e.py parity`: electron (gamma 1e4) in a 100 T uniform field,
  neutron driver, COLD_START sampled B2B, `radiation_reaction_mode="medina_lad"`,
  120 steps, run on the branch with no `stochastic_emission` argument, on the
  branch with `stochastic_emission=dict(enabled=False)`, and on a clean
  `git archive development` tree with the same venv. All 182 rider+driver SOA
  arrays compared with `np.array_equal` (NaN-aware): `differing: []` in both
  comparisons.
* Code paths read for the disabled case: `core/equations.py:2112-2124` (no
  config key, `emission_config=None`, `particle_radiation_mode = radiation_mode`),
  `core/integration_runner.py:3457-3500` (`emission_runtime=None`),
  `core/full_dipole_history.py:146-170, 517-574, 826-834` (empty
  `velocity_jumps` leaves segment fitting and payload format untouched),
  `core/momentum_center_pair.py:651-656, 930-949` (runtime only when enabled).
* `tests/unit` on the branch in this environment: 727 passed, 1 skipped,
  1 failed (`test_full_dipole_radiation.py::test_rest_electric_or_magnetic_dipole_has_expected_quadratic_radiation`).
  That failure is **pre-existing**: the identical test fails identically on
  the `development` tree in the same venv (numpy 2.5 / scipy 1.18), so it is
  not attributable to this branch. `tests/test_cli_gui_parity.py`,
  `tests/unit/test_full_dipole_history.py`, `tests/unit/test_momentum_center_pair.py`
  and `tests/unit/test_full_dipole_inertial_prehistory.py`: 50 passed.

## P1 (wrong physics when enabled)

None found. Items checked and confirmed, with numbers:

* Fine-structure constant from native units, `core/stochastic_emission.py:271`:
  `charge**2/(HBAR_NATIVE*C)` = 1/137.0359990747.
* chi definition, `core/stochastic_emission.py:92-93, 104-119, 129-139`: the
  rest-frame Lorentz force is `F_par + gamma*F_perp`, which is the correct
  boost of the three-force (`K'_par = F_par`, `K'_perp = gamma F_perp`), so
  `chi = hbar|F_rest|/(m^2 c^3) = E_rest/E_S`. Checks: gamma=1000 with
  `gamma*E_perp = 1e-3 E_S` gives chi = 1.0000e-3; 1 T transverse at gamma=1000
  gives chi = 2.2655e-7 = gamma*B/B_S. Against the integrator's own force
  (`/tmp/lwrev_coulomb2.py`, electron beta 0.5 passing a static proton at
  R = 1.18 mm): reported chi / instantaneous-Coulomb chi = 1.030, and the
  solver's own `dP/dt` over the same step is 3 % off Coulomb for the same
  retardation/start-of-step reason, so the indicator is consistent with the
  applied force. `sampled_lw_fields` (`core/stochastic_emission.py:142-165`)
  reproduces `q (n-beta)/(gamma^2 kappa^3 R^2)` for a moving source exactly
  (spy output `E=(4.42e-6,-8.84e-6,0)` for beta=0.5, n=(0,1,0),
  q=-1.1787e-5); the integrator's `bdot` is `d beta/d(ct)`
  (`core/equations.py:5028`), matching the docstring assumption.
* Rate normalisation, `core/stochastic_emission.py:251-274, 294-332`: the
  module's integrated rate agrees with a direct `scipy.integrate.quad` of the
  Ritus bracket with the same prefactor to 5e-8. Limits (gamma=1e5):

  | chi | W/W_cl | W_cl,num/W_cl | P/P_cl | g_Erber(chi) | W hbar gamma/(alpha m c^2 chi^(2/3)) |
  |---|---|---|---|---|---|
  | 1e-4 | 0.9999 | 1.0000 | 0.9994 | 0.9995 | 0.067 |
  | 1e-2 | 0.9911 | 1.0000 | 0.9448 | 0.9488 | 0.308 |
  | 0.1 | 0.9303 | 0.9999 | 0.6550 | 0.6628 | 0.623 |
  | 1 | 0.7163 | 0.8565 | 0.1821 | 0.1811 | 1.034 |
  | 10 | 0.4236 | 0.4763 | 0.0187 | 0.0185 | 1.317 |
  | 100 | 0.2126 | 0.2295 | 0.0011 | 0.0011 | 1.424 |

  with `W_cl = 5 alpha m c^2 chi/(2 sqrt3 hbar gamma)` and
  `P_cl = (2/3) alpha (m c^2)^2 chi^2/hbar`. The classical bracket reproduces
  `W_cl` to 1e-4 below chi=0.1 (the deficit at chi>=1 is the `k < T_kin`
  cap); the quantum power follows Erber's `g(chi)` to <1 %; the high-chi
  coefficient approaches Ritus' 1.46 from below. The `density[0]` closed form
  (`:328`) is the correct `t -> 0` limit of the `t = z^(1/3)` quadrature.
  `lcfa_spectral_brackets` (`:214-248`) is algebraically identical to thread
  C's `spectral_brackets` since `1 - s + 1/(1-s) = 2 + s^2/(1-s)`.
  Sampled mean photon fraction (20k draws, chi=1) 0.11733 vs spectrum mean
  0.11740.
* Exact recoil, `core/stochastic_emission.py:335-377`: on a run with 5 photons
  (`/tmp/lwrev_e2e.py enabled`) `sum E_gamma = 273.3242800132` native vs
  electron `Delta(Pt) c = 273.3242800131` (rel. 5e-13);
  `radiation_energy_applied` sums to the same; final mass shell
  `Pt^2 - P^2 = (mc)^2` to 4.6e-8 relative (the integrator's own level,
  unchanged from the disabled run); `p + Q = p' + k` to 3e-14; collinear
  `|Q|/|p| <= 1e-12`; photons are placed collinear for gamma>=10 and the
  gamma<10 aberrated dipole pattern records the non-collinear remainder as
  `background_impulse` (`|Q|/|p| ~ 4e-2` at beta=0.5), which is the honest
  bookkeeping the docs describe.
* Medina/LAD interaction, `core/equations.py:5056-5110, 5193` and
  `core/momentum_center_pair.py:1078-1101, 1354-1372`: the rule is "once a
  particle's indicators cross, continuous RR is switched off for that particle
  for the rest of the run and only discrete photons remove energy". With RR
  on and emission active the classical `radiation_energy` diagnostic column is
  still filled (505.9 native in the run above) while `radiation_energy_applied`
  equals the photon sum (273.3), so nothing is double-applied; the diagnostic
  column is correctly labelled "classical field diagnostics" in the docs.
  Stochastic / classical mean energy loss on that run: 0.960 (5 photons drawn
  against an expectation of 5.55; consistent).
* Pair runner kick bookkeeping (`/tmp/lwrev_pair.py`, electron beta 0.9 +
  proton, `charge_ll`, one recoil forced through the module's own `recoil()`):
  history becomes `full-dipole-history-v9` with one `velocity_jumps` record
  (|v_in|/c = 0.9000, |v_out|/c = 0.8710), the v9 payload restores, the
  impulsive Hermite segments on both sides of the kick are position-continuous
  to 0.0, and a 6+6-step run through a `json.loads(json.dumps(...))` round
  trip equals the 12-step one-shot run exactly.

## P2 (robustness / modelling choices that need a documented decision)

1. **Mean-power discontinuity at the hand-off threshold** —
   `core/stochastic_emission.py:122-126`, `core/equations.py:5103-5105`,
   `docs/source/stochastic_quantum_emission.rst:14-17`.
   Below threshold the particle radiates with Medina/LAD, i.e. classical power
   `P_cl`; at the first crossing it switches to the quantum rate with mean power
   `g(chi) P_cl`. At the default `chi_threshold = 0.1` that is a 34.5 % drop in
   the mean energy-loss rate at a single step (table above: `P/P_cl = 0.655`).
   For a particle that ramps slowly through chi ~ 0.1 (undulator entry, beam-
   beam) the energy loss before the switch is overestimated by up to 1/g and
   the trajectory after it is biased relative to a run with
   `chi_threshold = 0.01` (5 % step). Not wrong, but the docs should state the
   size of the step and recommend a threshold of order 0.01 when the pre-
   crossing energy loss matters; alternatively multiply the Medina power by
   `g(chi)` below threshold. The activation is one-way (no hysteresis needed
   because there is no deactivation) and that is documented.

2. **Probability cap raises instead of sub-stepping** —
   `core/stochastic_emission.py:434-440`. When `W dt > max_probability`
   (0.05 default, 0.1 max) the run terminates with `ValueError` at whatever
   step that happens, discarding the run. For chi=1, gamma=1e4 the limit is
   `dt <= 8.5e-8 ns` (formation time ~1.3e-8 ns) and for chi=0.1 it is
   `6.6e-7 ns`; a particle that drifts into a stronger field late in a run
   kills it. Since the RNG is only consumed at the acceptance barrier, an
   emission sub-loop (split the accepted interval into `ceil(W dt / p_max)`
   Bernoulli trials with the same indicators) would be cheap and would also
   remove the first-order bias of a single Bernoulli trial
   (`1-exp(-W dt)` vs `W dt`, 2.5 % at the cap). At minimum the error should
   say at which step and particle it happened and the checkpoint should be
   flushed first so the run can be resumed with a smaller step.

3. **No test coverage for ~1500 lines of new physics** — no file under
   `tests/` is touched. "426 tests pass" only confirms the disabled path; the
   enabled path, the v9 history format, `EmissionRuntime` checkpoint payloads,
   and the CLI flag plumbing are unexercised by the suite, and
   `AGENTS.md` asks for regression coverage of checkpoint/restart behaviour.
   Given the owner's rule (new tests only when strongly justified), I think
   two are justified and sufficient, because nothing else covers them:
   (a) a module-level check that `photon_spectrum` reproduces `W_cl` at
   chi=1e-4 to 1e-3 and the Ritus quadrature at chi=1 to 1e-6 (guards the
   prefactor, Bessel table and `density[0]` limit against refactors), and
   (b) the pair-runner split-resume equality with one forced kick (guards the
   v9 history, `velocity_jumps` validation and RNG payload round trip). The
   general-runner restart equality could be folded into (b) or into the
   existing checkpoint restart test by parametrising it with
   `stochastic_emission` on; I would not add more than that.

4. **Stale per-row emission payloads** — `core/integration_runner.py:4645-4648`.
   Each checkpoint write stores `emission_runtime.to_payload()` on
   `trajectory[-1]` *at that time*, so after a run with periodic checkpoints
   the legacy trajectory list carries several rows with partial
   `_stochastic_emission` dicts (events up to that step). `build_report` and
   the testbed only read the final row, so the output is right, but any
   consumer iterating rows will find inconsistent event logs. Store it once at
   the end, or pop the stale key when the next checkpoint is written.

5. **Disabled-but-present emission payload still forbids passive pair
   updates** — `core/momentum_center_pair.py:931-949`. `nonlinear_pair.configure_checkpoint`
   writes a `stochastic_emission` payload whenever any emission flag is given,
   including `--emission-seed 5` without `--stochastic-emission` (`enabled`
   stays False). `_advance_particles` then raises "does not support passive
   pair updates" before checking `enabled`. Move the `passive_selector` check
   under `if emission_config.enabled`.

## P3 (nits)

* `lw_integrator/testbed_runner.py:565-568, 916`: `SimulationOptions.to_dict()`
  now always emits a `stochastic_emission` block, so GUI-saved configs and
  anything diffing saved configs change even when the feature is off. Harmless
  for simulation output (verified), but worth noting in the changelog.
* `core/stochastic_emission.py:63-68`: `is_electron` silently excludes muons,
  for which the same formulas hold with `m_mu`; a positive-mass check plus an
  explicit species whitelist would be clearer than a silent no-op.
* `core/stochastic_emission.py:431-433`: `photon_spectrum` rebuilds a 2049-point
  spectrum (two Bessel evaluations per point) for every active electron on
  every step. Fine for pairs, ~ms per call; a cache keyed on rounded
  `(chi, cap)` is deliberately avoided per the docstring, but a cheaper
  rate-only pre-check (`W dt` from the total rate before building the CDF)
  would halve the cost for the common no-emission step.
* `core/full_dipole_history.py:519-574`: intervals touching a kick fall back to
  cubic Hermite reconstruction, so the partner's radiation field near a kick is
  C1 only and the velocity jump itself radiates classically into the partner's
  LW field. This is the physical field of an impulsive velocity change and is
  partly what the photon already carried; it is a small effect for
  `s ~ 1e-4` but should be listed under the documented limitations.
* `core/stochastic_emission.py` ruff: one import-order and four `dict()` call
  style hits (`C408`), one `TRY004`; the touched legacy files have the same
  pre-existing ruff counts as on `development`; `black --check` passes on all
  touched files.
* Docs (`stochastic_quantum_emission.rst`) are thorough; the "Pairs evaluate the
  field at the interval start" statement should add that the photon energy is
  sampled from the end-of-interval momentum with start-of-interval chi, i.e.
  the scheme is first order in `dt` regardless of RK4.

## Merge recommendation

**Merge after addressing P2.1 (document the hand-off step and suggest a
threshold), P2.2 (sub-step or at least fail gracefully with step/particle and a
flushed checkpoint), P2.5 (one-line fix) and adding the two tests in P2.3.**
The physics, unit handling, recoil ledger, RNG/checkpoint reproducibility and
default-off parity all hold up under direct numerical checks; nothing here
requires a redesign.
