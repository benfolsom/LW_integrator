# Review: `review/trim-config` → `development`

- Reviewed commit: `7659e6a` "Consolidate run and sweep option schemas into one config layer (audit rank 9)".
- Base: `development` @ `3032cb4` (merge-base, v0.11.0).
- Scope: 11 files, +1531 / −2890 (net about −1,360 lines; the description's "about 2,800 net lines removed" is the gross deletion count).
- Reviewer: independent review session, 2026-10-11. Nothing was pushed to `development` or `review/trim-config`.

## Verdict

**Merge after one requested change (P2-1). No P0 or P1 findings.**

Parsing, precedence, aliases, nested blocks, defaults, coercion, error behaviour,
GUI/CLI/optimisation entry points and the single-integration hand-off are
behaviourally identical to `development` on every input I tested (≈58k
differential cases, below). The only observable regression is **key order in
saved JSON files**. Run configs (`save_config`, `to_dict`) and GUI sweep saves
(`build_saved_config_payload`) are no longer written in the historical order.
No loader cares about the order, but every regenerated or archived config will
show a large text diff. The CHANGELOG says the commit preserves "the existing
GUI save format", which is not literally true. The fix is small, and a golden
test should be added with it.

## Findings

### P0: wrong default behaviour or lost coverage

None found.

- Field sets, annotation strings, defaults (including default factories), and
  `inspect.signature` of `SimulationOptions` (220 parameters) and
  `OptimizationConfig` (247) are identical to `development`. Checked with
  `review_notes/trim-config_harness/fields_dump.py`.
- Coverage: the replaced `test_config_serialization` (a plain `json.dumps`
  smoke test) is now a stronger parametrised round-trip over every
  `configs/**/*.json`. `test_config_snapshot_structure` now derives from
  `SimulationOptions().to_dict()` instead of a literal dict. That is slightly
  more tautological but loses no assertions. See P2-2 for what is still not
  pinned.

### P1: wrong physics

None found. All physics-relevant values reach the runtime unchanged in every
tested case: self-consistency, chrono aliases, gamma reconciliation, RR mode,
RFS-dependent defaults (`radiation_reaction_mode="off"` and
`adaptive_timestep_enabled=False` for `rfs_minimal_2021`), particle loss,
pseudo-grid, driver train, smearing, external fields and the exact backend.
That covers the `SimulationOptions` built by `build_single_integration_setup`,
`from_simulation_options`, the GUI `apply_persisted_config_overrides` path and
the CLI sweep path. AGENTS.md numerical defaults (`fixed_geometry`, 2
iterations, `DISABLED` gamma reconciliation, `chrono_interpolate=false`,
`medina_lad`) are unchanged.

### P2: robustness

**P2-1: Saved JSON key order changed for run configs and GUI sweep configs.**

- `input_output/config.py:222-239`: `serialize_run_options` iterates
  `fields(options)`. Because `SimulationOptions` now inherits from
  `SharedOptions`, the inherited fields come first, so a saved run config
  starts with `seed`, `image_subcharge_count`, … instead of `steps`, `seed`,
  `simulation_type`, …. Inside the nested blocks, `pseudo_grid` and
  `driver_train` also change order. The other nested blocks keep their order.
- `input_output/config.py:719-773`: `build_saved_config_payload` builds the
  dict from `_SWEEP_SAVED_FIELDS` and then `update()`s `simulation_type`,
  `aperture_min/max`, `energy_min/max`, … at the end. The old payload started
  with `simulation_type`, `mode`, `aperture_min`, ….
- Failure scenario: open and re-save any config in the GUI, or run with
  `save_config` (also `gui_runtime_mixins.py:169` `run_config.json`). The file
  keeps the same keys and values but is reordered. This produces large diffs in
  checked-in configs (here and in LW_feasibility_studies) and makes
  old-vs-new run snapshots hard to compare by eye.
- Evidence: `run_to_dict_order` / `run_save_text` differ in 42,217 of 42,217
  successfully parsed run cases, and `gui_save_text` differs in 58,116 of
  58,116 GUI cases. In the same cases the order-insensitive content is
  identical, and the key sets are identical (117 top-level run keys, 136 GUI
  keys).
- A quick fix is not enough: emitting keys in `RUN_OPTION_ORDER` still differs.
  The old hand-written `to_dict` placed `stochastic_emission` and the inner
  `pseudo_grid` keys in their own order. Suggested fix: record the
  `development` key order (top-level and per nested block) as an explicit
  saved-order table for run and GUI payloads, and add a golden byte-for-byte
  test (P2-2).

**P2-2: The new round-trip test checks only that new code agrees with itself.**

- `tests/test_integrator_testbed.py:256` checks
  `from_dict(to_dict(x)) == to_dict(x)` and document read/write idempotence.
  That is why P2-1 passed CI. Suggest adding one golden fixture each for
  `SimulationOptions().to_dict()` text, a GUI `build_saved_config_payload` text
  and `convert_sweep_options` on a sweep fixture, all captured from
  `development`.

**P2-3: Coercion policy is dispatched on annotation strings and name prefixes.**

- `input_output/config.py:373-435` (`parse_run_options`) chooses coercion by
  `item.type == "bool"`, `"Optional[float]"`, `"Tuple[float, float, float]"`,
  etc., and chooses strictness by name prefix (`checkpoint_`, `adaptive_pair_`,
  `magnetic_dipole_`, `*_polarization`).
- This is correct today: identical results on all fuzzed types per key. But it
  is latent. A future field annotated `float | None`, `Optional[Tuple[...]]`
  with different spacing, or a non-future-annotations module would silently
  skip coercion. A new `magnetic_dipole_*` field would silently become strict.
  Suggest asserting at import time that every field's annotation is in the
  handled set (or is explicitly listed as passthrough), so drift fails loudly.

### P3: nits

1. **Import layout (ruff I001), new in this branch:**
   `lw_integrator/testbed_runner.py:26`, `optimization/config.py:3`,
   `optimization/plugin_config_mixins.py:3` (stray blank line after
   `from __future__`), and `optimization/single_integration_helpers.py:3`.
   There are three separate `from input_output.config import` statements in
   `optimization/config.py:11-14` and `testbed_runner.py:34-42`.
2. **Extraneous parentheses (UP034):** `optimization/config.py:398,406,411`,
   e.g. `tuple((float(v) for v in ...))`.
3. **Constructor `TypeError` messages lost the class prefix when positional
   args are used.** `preserve_option_constructor` (`input_output/config.py:1400`)
   binds through `inspect.Signature`. For example, the old message was
   `SimulationOptions.__init__() got multiple values for argument 'steps'`; the
   new one is `multiple values for argument 'steps'`. "Too many positional"
   messages and "unexpected keyword" with positional args change the same way.
   Pure keyword construction is unchanged.
4. Dataclass field order changed, so `repr()`, `asdict()` and `fields()` order
   for both option classes differ. Nothing in-repo persists those (checked
   `asdict`/`astuple`/`fields` call sites), but `repr` in logs reorders.
5. `read_config_document` / `write_config_document` import `json` inside the
   function (`input_output/config.py:1088,1098`). Move it to module level.
6. `input_output/config.py:1419` uses `setattr(fn, "__signature__", ...)`
   (B010).
7. The commit description overstates the reduction (net −1,359, not −2,800).

## Deprecated keys and warnings

- Load-time: neither branch emits Python warnings or prints while parsing any
  of the ≈58k cases. Captured `warnings` and stdout/stderr per entry point were
  identical.
- Runtime warnings are unchanged because their inputs are unchanged.
  `ChronoHighPrecisionDeprecationWarning`, `SelfConsistencyOptionWarning`
  (inert `chrono_*`, gamma reconciliation, exact-path chrono) come from
  `core/self_consistency.py`, which was not touched, and they receive identical
  option values. Legacy `self_consistency_chrono_*` aliases still resolve with
  canonical-wins precedence on file load and OR/non-default merging on direct
  construction. All 40 file-load `chrono_*` × alias conflict combinations (5 settings × 2×2 values × 2 bases)
  match, as do 20 direct-constructor combinations per class.
  `mass_shell_only` / `full_iteration` still canonicalise.
  `tests/test_self_consistency_options.py` passes.

## Evidence and method

### 1. Targeted tests

These were run with `pytest -p no:anyio -m "not slow"` under `xvfb-run` on
Python 3.12, so tkinter GUI tests ran:

```
tests/physics/test_free_crossing_null.py tests/test_adaptive_energy_increments.py
tests/test_cli.py tests/test_cli_gui_parity.py tests/test_cli_per_run_timeout.py
tests/test_energy_aware_timestep.py tests/test_failed_run_retry.py tests/test_gui.py
tests/test_gui_config_magnetic_dipole.py tests/test_gui_config_smearing.py
tests/test_integrator_testbed.py tests/test_optimization.py
tests/test_optimization_config_helpers.py tests/test_optimization_plugin.py
tests/test_optimization_run_parameter_helpers.py tests/test_self_consistency_options.py
tests/test_simulation_options.py tests/test_simulation_options_magnetic_dipole.py
tests/test_single_integration_helpers.py tests/test_single_run_logging.py
tests/test_sweep_config_conversion.py tests/unit/test_adaptive_bunch.py
tests/unit/test_instantaneous_clock_config.py tests/unit/test_rfs_integration_benchmark.py
tests/unit/test_source_history_representation_config.py
tests/unit/test_testbed_config_loader.py
```

- `review/trim-config`: 537 passed, 1 skipped, 17 deselected.
- `development`: 520 passed, 1 skipped, 17 deselected.
- The extra 17 are the new parametrised round-trip tests. The skip is a
  missing reference config on both branches.

### 2. Key-access parity

The old and new loaders were given recording dicts, recursively over `{}`, a
full nested `to_dict()` payload and a flattened payload. Results:

- `SimulationOptions.from_dict`: both read exactly the same 335 key paths.
- `apply_persisted_config_overrides`: both read exactly the same 113 paths.

The old and new code therefore look at the same keys, nested blocks and flat
compatibility spellings in the same situations. See `record.py`.

### 3. Differential fuzzing: 58,152 cases, each through both branches

Script: `gen.py` → `probe2.py` / `probe3.py` → `compare.py`.

Corpus and mutations:

- Corpus: all 27 JSON objects under `configs/`, `examples/`,
  `tests/fixtures/` and `tests/unit/fixtures/`. This includes all 6 run
  configs, the 8 sweep configs plus `manual_smoke`, and the legacy sweep
  fixtures.
- Single-key mutations of 5 bases: empty, full nested `to_dict()`, flattened,
  the counter-propagating B2B run config, and `demo_parameter_sweep.json`.
- Mutated keys: every recorded key path, every `OptimizationConfig` field,
  sweep JSON aliases (`aperture_min/max`, `energy_min/max`, `rider_offset_*`,
  `auto_steps_distance`, …), every leaf of the four sweep nested blocks, and
  `sweep_parameters/*/{enabled,min,max,points,log,fixed_value}`.
- Mutation values (22): `None`, `""`, garbage strings, numeric strings, ints,
  floats, bools, a list, a 3×3 matrix, `{}`, legacy mode names, `cpp`,
  `CONDUCTING_WALL`, `sweep`, `first-exit`.
- Chrono canonical/alias conflict matrices.

Entry points compared. Each comparison records the full canonicalised object
state, or the exception type and message, plus captured warnings and stdout:

| Entry point | Compared | Semantic diffs |
|---|---:|---:|
| `SimulationOptions.from_dict` (run/GUI/`load_config`) | 58,152 | 0 |
| `to_dict` content (order-insensitive) | 42,217 | 0 |
| `save_config` → `load_config` reload | 42,217 | 0 |
| `OptimizationConfig.from_simulation_options` (GUI → sweep) | 42,217 | 0 |
| `_convert_json_config_to_dataclass` (CLI sweep) | 58,152 | 0 |
| CLI-filtered `OptimizationConfig(**converted)` | 57,143 | 0 |
| `apply_persisted_config_overrides` (GUI sweep load) | 58,152 | 0 |
| `build_single_integration_setup(...).options` (sweep → run) | 14,274 | 0 |
| Input payload mutated by loader | 58,152 | 0 |
| **Saved run JSON text / key order** | 42,217 | **42,217 (order only)** |
| **Saved GUI sweep JSON text** | 58,116 | **58,116 (order only)** |
| `convert_sweep_options` dict insertion order | 58,152 | 11,397 (order only; no effect, consumed as kwargs) |

### 4. Constructors

`ctor.py` exercises positional construction (`SimulationOptions(17, 42, ...)`,
`OptimizationConfig(SimulationType, "optimization", ...)`), chrono alias
merging and error cases. Results match except for the error message text in
P3-3.

### 5. Static checks

- `black --check` is clean on all changed files.
- mypy (changed modules) reports 1028 errors vs 1043 on `development`, with no
  new error classes.
- ruff: the new findings are listed in P3.

## Limits

- **LW_feasibility_studies was not reachable.** This session has no GitHub
  credentials, and that repository is private (`git ls-remote` asks for
  credentials), so I could not round-trip its JSONs. The mutation corpus
  covers every key path either loader reads. Still, re-running
  `review_notes/trim-config_harness/` over that repo's configs before merge is
  recommended, especially for archived GUI sweep saves and `run_config.json`
  snapshots.
- GUI widget-level load/save (`OptimizationPluginConfigMixin._load_config_from_path`
  / `_save_config`) was exercised through the existing tkinter tests and
  through the helper functions those methods call, not by hand in a live GUI.
- AGENTS.md asks for new worktrees to be added to Zed; `zed` is unavailable in
  this cloud container.

## Reproducing

Run from a worktree of each branch:

```
PYTHONPATH=$PWD python review_notes/trim-config_harness/gen.py $PWD   # writes cases.json next to the script (≈300 MB)
PYTHONPATH=$PWD python review_notes/trim-config_harness/probe2.py $PWD 0 58152 out_<branch>_0.json
```

`gen.py` reads `acc_run.json` / `acc_gui.json` produced by `record.py run|gui`
on `development`. Then adapt `compare.py` to the output file names.
