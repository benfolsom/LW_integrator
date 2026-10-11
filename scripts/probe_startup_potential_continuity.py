"""Reproduce the part-2 ledger convergence evidence without changing kinematics."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json

import numpy as np
from scipy.integrate import solve_ivp

from core.constants import C_MMNS


def main():
    spec = importlib.util.spec_from_file_location(
        "ledger", "tests/physics/test_self_space_charge_energy.py"
    )
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    namespace = dict(vars(ledger))
    source = inspect.getsource(ledger._run_short_self_space_charge)
    source = source.replace("    steps = 40", "    steps = probe_steps")
    exec(compile(source, "<ledger-step-study>", "exec"), namespace)
    run = namespace["_run_short_self_space_charge"]
    population = 554800806.6187345
    results = []
    for intervals in (39, 78, 156, 312, 624):
        namespace["probe_steps"] = intervals + 1
        for mode in ("diagnostic_only", "medina_lad"):
            with contextlib.redirect_stdout(io.StringIO()):
                trajectory = run(
                    charge_scale=1,
                    space_charge_enabled=True,
                    radiation_reaction_mode=mode,
                )
            energy = (
                np.array([ledger._physical_kinetic_mev(s) for s in trajectory])
                * 1e6
                / population
            )
            potential = (
                np.array(
                    [
                        ledger._pair_potential_mev(s, 0.1)
                        for s in (trajectory[0], trajectory[-1])
                    ]
                )
                * 1e6
                / population
            )
            gamma = float(trajectory[0]["gamma"][0])
            beta = abs(float(trajectory[0]["bz"][0]))
            result = dict(
                mode=mode,
                intervals=intervals,
                h_ns=10 / (gamma * beta * C_MMNS * intervals),
                initial_kinetic_eV=float(energy[0]),
                final_kinetic_eV=float(energy[-1]),
                step1_eV=float(energy[1] - energy[0]),
                final_eV=float(energy[-1] - energy[0]),
                potential_change_eV=float(potential[1] - potential[0]),
                total_change_eV=float(
                    energy[-1] - energy[0] + potential[1] - potential[0]
                ),
                rr_native_sum=float(
                    sum(np.sum(s.get("radiation_reaction_work", 0)) for s in trajectory)
                ),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    rider, _ = ledger._make_bunch()
    n = len(rider["x"])
    position = np.stack([rider[a] for a in "xyz"], axis=-1)
    momentum = np.stack([rider["P" + a] for a in "xyz"], axis=-1)
    mass = rider["m_species"]
    observer_charge = rider["q_observer"]
    source_charge = rider["q_source"]
    y0 = np.concatenate((position.ravel(), momentum.ravel()))
    time_end = 10 / (abs(float(rider["bz"][0])) * C_MMNS)

    def rhs(t, y):
        r = y[: 3 * n].reshape(n, 3)
        p = y[3 * n :].reshape(n, 3)
        gamma = np.sqrt(1 + np.sum(p * p, axis=1) / (mass * C_MMNS) ** 2)
        dr = p / (mass * gamma)[:, None]
        d = r[:, None, :] - r[None, :, :]
        r3 = (np.sum(d * d, axis=-1) + 0.1**2) ** 1.5
        np.fill_diagonal(r3, np.inf)
        force = observer_charge[:, None] * np.sum(
            source_charge[None, :, None] * d / r3[:, :, None], axis=1
        )
        return np.concatenate((dr.ravel(), force.ravel()))

    references = []
    for tolerance in (1e-10, 1e-12):
        solution = solve_ivp(
            rhs,
            (0, time_end),
            y0,
            method="DOP853",
            rtol=tolerance,
            atol=tolerance * 1e-2,
        )
        y = solution.y[:, -1]
        final = dict(rider)
        p = y[3 * n :].reshape(n, 3)
        final["gamma"] = np.sqrt(1 + np.sum(p * p, axis=1) / (mass * C_MMNS) ** 2)
        for k, a in enumerate("xyz"):
            final[a] = y[: 3 * n].reshape(n, 3)[:, k]
        kinetic_change = (
            (ledger._physical_kinetic_mev(final) - ledger._physical_kinetic_mev(rider))
            * 1e6
            / population
        )
        potential_change = (
            (
                ledger._pair_potential_mev(final, 0.1)
                - ledger._pair_potential_mev(rider, 0.1)
            )
            * 1e6
            / population
        )
        reference = dict(
            reference="independent instantaneous softened Coulomb, coordinate time",
            rtol=tolerance,
            success=solution.success,
            nfev=solution.nfev,
            final_eV=kinetic_change,
            potential_change_eV=potential_change,
            total_change_eV=kinetic_change + potential_change,
        )
        references.append(reference)
        print(json.dumps(reference), flush=True)
    from tests.physics.test_startup_inclusion_continuity import (
        run_gate,
        gate_measurements,
    )

    gates = []
    for mode in ("diagnostic_only", "medina_lad"):
        with contextlib.redirect_stdout(io.StringIO()):
            trajectory, driver, *_ = run_gate(mode)
        gate = dict(mode=mode, **gate_measurements(trajectory, driver))
        gates.append(gate)
        print(json.dumps(gate), flush=True)
    with open("/private/tmp/lw_continuity_convergence.json", "w") as f:
        json.dump(dict(runs=results, references=references, gates=gates), f, indent=2)


if __name__ == "__main__":
    main()
