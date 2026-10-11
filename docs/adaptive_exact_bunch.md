# Adaptive exact runs with same-bunch charge fields

`adaptive_pair_return` can advance exact RFS runs with multiple particles in
both bunches. Enable `INERTIAL_PREHISTORY`,
`second_order_start_taylor_endpoint`, and checkpointing. Enable retarded
space-charge fields to include interactions within each bunch. Dipole source
fields and intrinsic spin self-reaction currently require one particle per
bunch. Particle-loss scheduling, driver trains, pseudo-grid schedules, and
cavity-exit tails retain their existing restrictions.

The initial `time_step` is a proper-time guess and sets the scale for the
minimum and maximum step factors. `target_lab_time_ns` is the common final
lab time. Each particle gets its own proper-time increment so particles with
different Lorentz factors reach the same lab-time boundary. The controller
compares a full step against two half steps across every particle's position,
mechanical momentum, spin, and energy diagnostics. Bunch vector errors use
per-particle Euclidean magnitudes, so acceptance is independent of coordinate
rotation. The existing one-particle pair norm remains unchanged. It commits the two half
steps jointly for both bunches after acceptance. Rejected trials leave
accepted histories unchanged.

Every accepted midpoint and endpoint stays in the source history and in the
returned arrays. Output sampling does not remove retarded source knots.
Interpolation uses each segment's actual duration, so unequal intervals are
supported. The same-bunch predictor continues only the last accepted quintic
segment within the current trial horizon. Endpoint reconstruction uses both
provisional bunch histories and excludes each observer's own source index.
This predictor is bounded by the trial's error estimate; it is not an
independent guarantee of accuracy at a singular collision.

A nested CLI/testbed configuration can contain:

```json
{
  "adaptive_pair_return": {
    "enabled": true,
    "target_lab_time_ns": 1e-9,
    "tolerance_scale": 1000000,
    "minimum_step_factor": 1e-6,
    "maximum_step_factor": 64,
    "shared_time_absolute_tolerance_ns": 1e-27,
    "diagnostics_absolute_tolerance_native": 1e-4
  },
  "checkpoint": {
    "enabled": true,
    "directory": "/tmp/adaptive-charge-bunch",
    "interval_steps": 1000,
    "interval_seconds": 300
  }
}
```

These values are an explicit alpha–mu benchmark budget, not new defaults.
`tolerance_scale` multiplies the existing absolute and relative error scales.
`diagnostics_absolute_tolerance_native`, when supplied, replaces only the
absolute energy-diagnostic scale; its relative scale remains unchanged.
Choose this budget in the integrator's native energy units, amu mm²/ns², from
the physical energy accuracy required. The original scale can be dominated by
roundoff in subtracting large rest energies or by projection bookkeeping at
very small orbit scales. Position and mechanical-momentum tests still have
to pass independently. Omitting this field preserves the original tolerance.
The option is available through the Python configuration, nested JSON CLI
input, and testbed configuration; the GUI has no separate budget control.

Fresh prehistory input must supply mechanical momentum. The runner adds the
exact canonical potential once. An older helper that pre-adds approximate
$qA/c$ must be adapted before using this mode.

For the reproducible CTMC benchmark, run from the worktree root:

```sh
PYTHONPATH="$PWD" NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /Users/benjaminfolsom/work/extracurr/LW_integrator/.venv/bin/python \
  scripts/benchmark_adaptive_bunch.py 200 767058261 \
  --out /tmp/adaptive-bunch-200
```

The helper reads the feasibility study's seeded initial conditions and DOP853
reference, forces imports to this worktree, supplies mechanical input, and
writes trajectories, checkpoints, and `summary.json` under the output path.
Radiation reaction is explicitly off for comparison with the conservative
classical orbit reference. This does not validate a quantum bound-state model.
