"""Reproduce the alpha–mu CTMC cases using optional adaptive exact steps.

Run from this worktree with its source on PYTHONPATH. The reference study is
read only; all trajectory output and checkpoints go to the supplied directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core

os.environ["LW_INTEGRATOR_DIR"] = str(Path(__file__).resolve().parents[1])

STUDY = (
    Path.home()
    / "work/extracurr/LW_feasibility_studies/studies/ring_counterpropagating_survey/scripts"
)
sys.path.insert(0, str(STUDY))
import numpy as np
import r2_ctmc_lw as reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("b", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("--z0", type=float, default=40.0)
    parser.add_argument("--spu", type=float, default=100.0)
    parser.add_argument("--tolerance", type=float, default=1000000.0)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--diagnostics", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ct = reference.ct
    speed = math.sqrt(2 * 14.0 / (ct.M_ION * ct.MMU)) / ct.ALPHA
    build_config = reference.lb.build_config

    def adaptive_config(**kwargs):
        if args.fixed:
            return build_config(**kwargs)
        duration = 2 * args.z0 / speed * reference.T0_NS
        config = build_config(
            **kwargs,
            pair_return_target_ns=duration,
            pair_return_tolerance_scale=args.tolerance,
        )
        config["adaptive_pair_return"][
            "diagnostics_absolute_tolerance_native"
        ] = args.diagnostics
        config["adaptive_pair_return"]["minimum_step_factor"] = 1.0e-6
        config["adaptive_pair_return"]["shared_time_absolute_tolerance_ns"] = 1e-27
        config["checkpoint"].update(interval_steps=1000, interval_seconds=300)
        return config

    reference.lb.build_config = adaptive_config
    set_particles = reference.set_particles

    def mechanical_input(state, masses, charges, positions, velocities, phi, a=None):
        # Current INERTIAL_PREHISTORY runner adds exact canonical potentials.
        # The older reference helper supplied approximate canonical input too.
        set_particles(state, masses, charges, positions, velocities, np.zeros_like(phi))

    reference.set_particles = mechanical_input
    m, q, pos, vel, fpos, fvel, t_lab, result = reference.run_lw(
        "Au_bare",
        speed,
        args.b,
        args.seed,
        args.z0,
        args.spu,
        args.out,
        8,
        "INERTIAL_PREHISTORY",
    )
    cpos, cvel = reference.coulomb_final(m, q, pos, vel, t_lab)
    energy = reference.pair_energy
    row = {
        "b": args.b,
        "seed": args.seed,
        "z0": args.z0,
        "initial_spu": args.spu,
        "adaptive": not args.fixed,
        "tolerance_scale": args.tolerance,
        "t_lab": t_lab,
        "e0": energy(m, q, pos, vel, 0, 1),
        "e_c": energy(m, q, cpos, cvel, 0, 1),
        "e_lw": energy(m, q, fpos, fvel, 0, 1),
        "duration_s": result.duration_s,
        "accepted_knots": len(result.rider["t"]),
        "diagnostics_absolute_tolerance_native": args.diagnostics,
        "worktree": str(Path(core.__file__).resolve().parents[1]),
    }
    row["relative_energy_error"] = abs(row["e_lw"] / row["e_c"] - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
