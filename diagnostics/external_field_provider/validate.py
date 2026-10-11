"""Short, single-process external-provider diagnostics; run from repository root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from core.constants import C_MMNS, ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from core.integration_runner import retarded_integrator
from core.self_consistency import SelfConsistencyConfig
from core.types import ExternalFieldConfig, SimulationType, StartupMode


def particle(gamma=2.0, x=0.0, charge=ELEMENTARY_CHARGE, mass=ELECTRON_MASS_AMU):
    beta = np.sqrt(1 - gamma**-2)
    values = dict(
        x=x,
        y=0.0,
        z=0.0,
        t=0.0,
        Px=0.0,
        Py=0.0,
        Pz=gamma * mass * C_MMNS * beta,
        Pt=gamma * mass * C_MMNS,
        gamma=gamma,
        bx=0.0,
        by=0.0,
        bz=beta,
        bdotx=0.0,
        bdoty=0.0,
        bdotz=0.0,
        q=charge,
        m=mass,
        char_time=2 * charge**2 / (3 * mass * C_MMNS**3),
    )
    return {key: np.array([value]) for key, value in values.items()}


def run(field=None, steps=1200, h=1e-6, state=None, mode="medina_lad", exact=False):
    state = particle() if state is None else state
    empty = {key: np.array([], dtype=float) for key in particle()}
    exact_options = {}
    if exact:
        from core.types import MagneticDipoleConfig, MagneticDipoleParticleConfig

        neutral_spin = MagneticDipoleParticleConfig(
            species="custom",
            magnetic_moment_j_per_t=0.0,
            spin_quantum_number=0.5,
            polarization=0.0,
        )
        exact_options["magnetic_dipole"] = MagneticDipoleConfig(
            enabled=True,
            spin_precession_enabled=True,
            stern_gerlach_force_enabled=False,
            exact_retarded_update="second_order_start_taylor_endpoint",
            rider=neutral_spin,
            driver=neutral_spin,
        )
    result = retarded_integrator(
        steps=steps,
        h_step=h,
        wall_z=0.0,
        aperture_radius=1e9,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=state,
        init_driver=empty,
        mean=0.0,
        cav_spacing=0.0,
        z_cutoff=1e9,
        startup_mode=(
            StartupMode.INERTIAL_PREHISTORY if exact else StartupMode.COLD_START
        ),
        use_numba=True,
        self_consistency=SelfConsistencyConfig(),
        external_field=field,
        radiation_reaction_mode=mode,
        **exact_options,
    )
    traj = result[0]
    assert len(traj) == steps
    assert all(
        np.all(np.isfinite(s[a])) and len(s[a]) == 1
        for s in traj
        for a in ("x", "y", "z", "gamma")
    )
    assert all(s["gamma"][0] >= 1 for s in traj)
    return {
        key: np.array([s.get(key, np.array([0.0]))[0] for s in traj])
        for key in traj[-1]
        if isinstance(traj[-1][key], np.ndarray) and traj[-1][key].shape == (1,)
    }


def snapshot(path):
    data = run(steps=32)
    np.savez(path, **data)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_difference(provider, positions, epsilon):
    potential_errors, field_errors = [], []
    for pos in positions:
        pos = np.array(pos, dtype=float)
        sample = provider(position_mm=tuple(pos), time_ns=0.0)
        grad = np.zeros(3)
        jac = np.zeros((3, 3))
        for j in range(3):
            offset = np.eye(3)[j] * epsilon
            plus = provider(position_mm=tuple(pos + offset), time_ns=0.0)
            minus = provider(position_mm=tuple(pos - offset), time_ns=0.0)
            grad[j] = (plus.phi - minus.phi) / (2 * epsilon)
            jac[:, j] = (plus.electric - minus.electric) / (2 * epsilon)
        e_scale = max(np.max(np.abs(sample.electric)), 1e-100)
        j_scale = max(np.max(np.abs(sample.partial_e)), 1e-100)
        potential_errors.append(float(np.max(np.abs(grad + sample.electric)) / e_scale))
        field_errors.append(
            float(np.max(np.abs(jac - sample.partial_e[1:].T)) / j_scale)
        )
    return dict(
        max_gradient_relative_error=max(potential_errors),
        max_jacobian_relative_error=max(field_errors),
    )


def interface_checks():
    from core.external_field_provider import (
        ExternalFieldSample,
        provider_force_derivative,
    )
    from core.external_fields import (
        electric_field_v_per_m_to_native,
        magnetic_field_tesla_to_native,
    )

    # Differentiate a general time-varying Lorentz force along a moving event,
    # including beta_dot cross B. This exercises all four derivative rows.
    e0 = np.array([2.0, -3.0, 1.0])
    b0 = np.array([0.3, 2.0, -0.5])
    edot = np.array([0.7, -0.2, 0.9])
    bdot = np.array([-0.1, 0.5, 0.2])
    je = np.array([[2.0, 1.0, 3.0], [-1.0, -0.4, 0.1], [0.2, 4.0, -0.5]])
    jb = np.array([[1.0, 2.0, 3.0], [-2.0, -0.6, 0.2], [0.1, -0.3, -0.4]])

    def sample(position, time):
        return ExternalFieldSample(
            0.0,
            np.zeros(3),
            e0 + je @ position + edot * time,
            b0 + jb @ position + bdot * time,
            np.vstack([edot / C_MMNS, je.T]),
            np.vstack([bdot / C_MMNS, jb.T]),
        )

    position = np.array([0.2, -0.3, 0.4])
    beta = np.array([0.1, 0.2, -0.3])
    beta_dot = np.array([0.4, -0.6, 0.2])
    eps = 1e-7

    def force(sign):
        s = sample(position + sign * eps * C_MMNS * beta, 0.7 + sign * eps)
        return s.electric + np.cross(beta + sign * eps * beta_dot, s.magnetic)

    analytic = provider_force_derivative(
        sample(position, 0.7), charge=1.0, beta=beta, beta_dot=beta_dot
    )
    numeric = (force(1) - force(-1)) / (2 * eps)
    relative = float(np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic)))
    assert relative < 1e-8
    converted = ExternalFieldSample.from_si(
        phi_volts=2.0,
        vector_potential_t_m=[1.0, 2.0, 3.0],
        electric_v_m=[1.0, 2.0, 3.0],
        magnetic_t=[1.0, 2.0, 3.0],
        partial_e_si=np.ones((4, 3)),
        partial_b_si=np.ones((4, 3)),
    )
    np.testing.assert_allclose(
        converted.phi, 2000 * electric_field_v_per_m_to_native(1.0)
    )
    np.testing.assert_allclose(
        converted.partial_e[0],
        electric_field_v_per_m_to_native(1.0) / (1000 * 299792458.0),
    )
    np.testing.assert_allclose(
        converted.partial_b[1:], magnetic_field_tesla_to_native(1.0) / 1000
    )
    assert not converted.electric.flags.writeable
    from core.types import ExternalFieldConfig

    for kwargs in ({"z_min": 0.0}, {"electric_field_native": (1.0, 0.0, 0.0)}):
        try:
            ExternalFieldConfig(provider=lambda **_: converted, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("provider must own its smooth field region")

    def disabled_callable(**_):
        raise AssertionError("disabled provider was evaluated")

    disabled = run(
        ExternalFieldConfig(enabled=False, provider=disabled_callable), steps=32
    )
    default = run(steps=32)
    assert disabled.keys() == default.keys()
    assert all(disabled[k].tobytes() == default[k].tobytes() for k in default)
    return dict(
        disabled_callable_parity_arrays=len(default),
        lorentz_force_derivative_relative_error=relative,
        si_conversion_and_readonly_sample="passed",
        hard_windows_and_base_field_conflicts="rejected",
    )


def validate(output, baseline=None, reuse_harmonic=False):
    from diagnostics.external_field_provider.providers import (
        HarmonicChannel,
        SmoothTube,
    )
    from core.external_field_provider import ExternalFieldSample

    output.mkdir(parents=True, exist_ok=True)
    summary = {"interface": interface_checks()}
    if baseline is not None:
        before = np.load(baseline)
        after = run(steps=32)
        changed = [
            key for key in before.files if before[key].tobytes() != after[key].tobytes()
        ]
        summary["default_parity"] = dict(arrays=len(before.files), changed=changed)
        assert not changed
    harmonic = HarmonicChannel(3.0)
    tube = SmoothTube(
        depth=7.0, radius_mm=0.2, z_min_mm=-1.0, z_max_mm=1.0, ramp_mm=0.3
    )
    summary["finite_differences"] = dict(
        harmonic=finite_difference(harmonic, [(0.1, 0.2, 0.0), (-0.3, 0.4, 2.0)], 1e-6),
        tube=finite_difference(
            tube,
            [
                (0.0, 0.0, 0.0),
                (0.05, -0.07, -0.9),
                (0.12, 0.03, 0.9),
                (0.2, 0.1, 0.0),
                (0.2, 0.1, -1.1),
            ],
            1e-6,
        ),
    )
    for check in summary["finite_differences"].values():
        assert check["max_gradient_relative_error"] < 1e-7
        assert check["max_jacobian_relative_error"] < 1e-7
    zero = run(ExternalFieldConfig(provider=HarmonicChannel(0.0)))
    drift = run()
    drift_error = max(
        float(np.max(np.abs(zero[a] - drift[a])))
        for a in ("x", "y", "z", "t", "Px", "Py", "Pz", "gamma")
    )
    summary["zero_field"] = dict(max_absolute_drift_difference=drift_error)
    assert drift_error == 0.0

    electric = np.array([2e7, -1e7, -1e8])
    magnetic = np.array([0.0, 3e7, 0.0])

    def uniform(*, position_mm, time_ns):
        pos = np.asarray(position_mm)
        return ExternalFieldSample(
            -electric @ pos,
            0.5 * np.cross(magnetic, pos),
            electric,
            magnetic,
            np.zeros((4, 3)),
            np.zeros((4, 3)),
        )

    reference = run(
        ExternalFieldConfig(
            electric_field_native=tuple(electric), magnetic_field_native=tuple(magnetic)
        ),
        state=particle(gamma=20.0),
        steps=1200,
        h=1e-6,
    )
    provided = run(
        ExternalFieldConfig(provider=uniform),
        state=particle(gamma=20.0),
        steps=1200,
        h=1e-6,
    )
    summary["uniform"] = dict(
        max_momentum_relative_difference=float(
            np.max(
                np.linalg.norm(
                    np.array([provided[a] - reference[a] for a in ("Px", "Py", "Pz")]),
                    axis=0,
                )
            )
            / np.max(
                np.linalg.norm(
                    np.array([reference[a] for a in ("Px", "Py", "Pz")]), axis=0
                )
            )
        ),
        terminal_gamma_difference=float(provided["gamma"][-1] - reference["gamma"][-1]),
        provider_caps=int(np.sum(provided["medina_impulse_capped"])),
        reference_caps=int(np.sum(reference["medina_impulse_capped"])),
    )
    assert summary["uniform"]["max_momentum_relative_difference"] < 2e-7
    assert (
        summary["uniform"]["provider_caps"] == summary["uniform"]["reference_caps"] == 0
    )

    omega = 1e11  # ns^-1; deliberately rapid toy channel makes damping measurable.
    tau = 2 * ELEMENTARY_CHARGE**2 / (3 * ELECTRON_MASS_AMU * C_MMNS**3)
    curvature = ELECTRON_MASS_AMU * omega**2 / ELEMENTARY_CHARGE
    field = ExternalFieldConfig(provider=HarmonicChannel(curvature))
    amplitude = 1e-4 * C_MMNS / omega  # nonrelativistic transverse motion.
    duration = 20 * np.pi / omega
    summary["harmonic"] = dict(
        omega_per_ns=omega,
        tau_ns=tau,
        tau_omega=tau * omega,
        analytic_action_rate_per_ns=tau * omega**2,
        duration_ns=duration,
        refinements=[],
    )
    terminal = []
    for steps in (601, 1201, 2401):
        h = duration / (steps - 1)
        artifact = output / f"harmonic_{steps}.npz"
        if reuse_harmonic:
            cached = np.load(artifact)
            off = {k[4:]: cached[k] for k in cached.files if k.startswith("off_")}
            rr = {k[3:]: cached[k] for k in cached.files if k.startswith("rr_")}
            assert len(off["x"]) == len(rr["x"]) == steps
            np.testing.assert_allclose(off["x"][0], amplitude, rtol=1e-15, atol=0.0)
            np.testing.assert_allclose(off["t"][-1], duration, rtol=1e-8, atol=0.0)
        else:
            off = run(
                field,
                steps=steps,
                h=h,
                state=particle(gamma=1.0, x=amplitude),
                mode="off",
            )
            rr = run(field, steps=steps, h=h, state=particle(gamma=1.0, x=amplitude))

        def energy(data):
            momentum2 = sum(data[a] ** 2 for a in ("Px", "Py", "Pz"))
            kinetic = momentum2 / (ELECTRON_MASS_AMU * (data["gamma"] + 1))
            return kinetic + ELEMENTARY_CHARGE * 0.5 * curvature * (
                data["x"] ** 2 + data["y"] ** 2
            )

        e_off, e_rr = energy(off), energy(rr)
        off_drift = float(e_off[-1] - e_off[0])
        rr_change = float(e_rr[-1] - e_rr[0])
        rr_work = float(np.sum(rr["radiation_reaction_work"]))
        ratio = e_rr / e_off
        select = rr["t"] > 2 * np.pi / omega
        rate = -np.polyfit(rr["t"][select], np.log(ratio[select]), 1)[0]
        entry = dict(
            steps=steps,
            h_ns=h,
            rr_off_max_energy_relative_error=float(
                np.max(np.abs(e_off / e_off[0] - 1))
            ),
            rr_off_max_abs_delta_h=float(np.max(np.abs(e_off - e_off[0]))),
            rr_off_terminal_delta_h=off_drift,
            rr_on_terminal_delta_h=rr_change,
            rr_on_summed_reaction_work=rr_work,
            rr_on_delta_h_minus_work=rr_change - rr_work,
            rr_budget_residual_relative_to_initial_energy=float(
                (rr_change - rr_work - off_drift) / e_off[0]
            ),
            fitted_action_rate_per_ns=float(rate),
            action_rate_relative_error=float(abs(rate / (tau * omega**2) - 1)),
            terminal_action_ratio=float(ratio[-1]),
            terminal_x_over_amplitude=float(rr["x"][-1] / amplitude),
            caps=int(np.sum(rr["medina_impulse_capped"])),
        )
        summary["harmonic"]["refinements"].append(entry)
        terminal.append(
            np.array(
                [
                    rr["x"][-1] / amplitude,
                    rr["Px"][-1] / (ELECTRON_MASS_AMU * omega * amplitude),
                ]
            )
        )
        if not reuse_harmonic:
            np.savez_compressed(
                output / f"harmonic_{steps}.npz",
                **{f"off_{k}": v for k, v in off.items()},
                **{f"rr_{k}": v for k, v in rr.items()},
            )
        assert entry["caps"] == 0
    refinements = summary["harmonic"]["refinements"]
    errors = [e["rr_off_max_energy_relative_error"] for e in refinements]
    assert errors[2] < errors[1] < errors[0]
    summary["harmonic"]["rr_off_max_energy_observed_orders"] = [
        float(np.log2(errors[i] / errors[i + 1])) for i in range(2)
    ]
    # Damping ratios cancel a shared integration error. Check the RR-off
    # error's size and first-order refinement separately from that ratio.
    assert errors[-1] < 0.015
    assert all(
        0.85 < order < 1.2
        for order in summary["harmonic"]["rr_off_max_energy_observed_orders"]
    )
    assert abs(refinements[-1]["rr_budget_residual_relative_to_initial_energy"]) < 0.005
    assert refinements[-1]["action_rate_relative_error"] < 0.03
    summary["harmonic"]["terminal_refinement_differences"] = [
        float(np.linalg.norm(terminal[i + 1] - terminal[i])) for i in range(2)
    ]
    assert (
        summary["harmonic"]["terminal_refinement_differences"][1]
        < summary["harmonic"]["terminal_refinement_differences"][0]
    )
    # A finite region exposes permanent exit-energy errors that a harmonic
    # damping ratio can hide. RR is explicitly off for this integration control.
    exit_tube = SmoothTube(
        depth=7e7, radius_mm=0.2, z_min_mm=0.05, z_max_mm=0.55, ramp_mm=0.1
    )
    exit_gamma = 2.0
    exit_beta = np.sqrt(1 - exit_gamma**-2)
    summary["tube_exit"] = {}
    for exact in (False, True):
        exit_refinements = []
        for steps in (151, 301, 601):
            h = 0.7 / (exit_gamma * exit_beta * C_MMNS * (steps - 1))
            data = run(
                ExternalFieldConfig(provider=exit_tube),
                steps=steps,
                h=h,
                state=particle(gamma=exit_gamma, x=0.1),
                mode="off",
                exact=exact,
            )
            kinetic = sum(data[a] ** 2 for a in ("Px", "Py", "Pz")) / (
                ELECTRON_MASS_AMU * (data["gamma"] + 1)
            )
            assert data["z"][-1] < exit_tube.z_min_mm
            assert data["Pz"][-1] < 0.0
            exit_refinements.append(
                {
                    "steps": steps,
                    "h_ns": h,
                    "terminal_z_mm": float(data["z"][-1]),
                    "relative_exit_energy_error": float(
                        (kinetic[-1] - kinetic[0]) / kinetic[0]
                    ),
                }
            )
        errors = [abs(e["relative_exit_energy_error"]) for e in exit_refinements]
        orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(2)]
        summary["tube_exit"]["exact_second_order" if exact else "default"] = {
            "refinements": exit_refinements,
            "exit_energy_observed_orders": orders,
        }
        assert errors[2] < errors[1] < errors[0]
        assert all(order > (1.5 if exact else 0.8) for order in orders)
    # Species-neutral hook and exact second-order endpoint route smoke probes.
    ion = run(
        ExternalFieldConfig(provider=HarmonicChannel(3.0)),
        steps=64,
        state=particle(gamma=2.0, x=0.01, charge=6 * ELEMENTARY_CHARGE, mass=12.0),
    )
    exact_data = run(
        field,
        steps=64,
        h=duration / 1200,
        state=particle(gamma=1.0, x=amplitude),
        exact=True,
    )
    np.savez_compressed(output / "exact_route_smoke.npz", **exact_data)
    summary["route_smoke"] = dict(
        ion_terminal_gamma=float(ion["gamma"][-1]),
        ion_radiated_energy=float(np.sum(ion["radiation_energy"])),
        exact_terminal_x_over_amplitude=float(exact_data["x"][-1] / amplitude),
        exact_caps=int(np.sum(exact_data["medina_impulse_capped"])),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--reuse-harmonic", action="store_true")
    args = parser.parse_args()
    if args.interface_only:
        print(json.dumps(interface_checks(), indent=2))
    if args.snapshot:
        print(snapshot(args.snapshot))
    if args.validate:
        validate(args.validate, args.baseline, args.reuse_harmonic)
