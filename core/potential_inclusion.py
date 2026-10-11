"""Canonical bookkeeping for sampled source inclusion at an accepted event.

The ledger stores raw four-potential changes in native units, in x, y, z, t
order (A_t = Phi). It is never a force, an impulse from a field, or work.
Source indices remain stable in the full sampled histories. Reduced pseudo-grid
source identities and refreshed source representations require a separate design.
"""

from __future__ import annotations

import base64
import copy
import json
import zlib
from typing import Callable

import numpy as np

from .constants import C_MMNS
from .types import ChronoMatchingMode, SimulationType, StartupMode
from .vectorized_interactions import (
    ExternalSampleBatch,
    compute_vectorized_contributions,
    gather_external_samples,
    gather_external_samples_soa,
)

LEDGER_FIELDS = tuple(f"potential_inclusion_delta_A_{axis}" for axis in "xyzt")
STATE_FIELDS = LEDGER_FIELDS + (
    "sampled_source_canonical_ready",
    "potential_inclusion_state",
)


def decode_inclusion_state(encoded: str) -> dict:
    """Read detached accepted metadata; compact strings avoid dense JSON storage."""
    if not encoded:
        return {}
    return json.loads(zlib.decompress(base64.b85decode(encoded.encode("ascii"))))


def encode_inclusion_state(inclusion: dict) -> str:
    payload = json.dumps(inclusion, separators=(",", ":"), allow_nan=False)
    return base64.b85encode(zlib.compress(payload.encode("ascii"))).decode("ascii")


def copy_inclusion_state(current: dict, result: dict) -> None:
    count = len(current["x"])
    for name in LEDGER_FIELDS:
        result[name] = np.array(current.get(name, np.zeros(count)), copy=True)
    result["sampled_source_canonical_ready"] = np.array(
        current.get("sampled_source_canonical_ready", np.zeros(count)),
        dtype=bool,
        copy=True,
    )
    # Lists avoid NumPy fixed-width string truncation while assembling a trial.
    result["potential_inclusion_state"] = list(
        current.get("potential_inclusion_state", [""] * count)
    )


def _gather(history, soa, indices, chrono, high_precision):
    common = {"include_positions": True}
    if chrono is not None:
        common.update(indices_next=chrono.indices_next, weights=chrono.weights)
    if soa is not None:
        if chrono is not None:
            common["needs_interpolation"] = chrono.needs_interpolation
        samples = gather_external_samples_soa(soa, indices, **common)
        # Keep raw kinematics for the counterfactual old set even if this source
        # died at the sampled event. The production gather masks it out.
        missing = ~samples.valid_mask
        if np.any(missing):
            raw = gather_external_samples(
                history,
                indices,
                **{k: v for k, v in common.items() if k != "needs_interpolation"},
            )
            for name in ("bx", "by", "bz", "gamma", "x", "y", "z"):
                getattr(samples, name)[missing] = getattr(raw, name)[missing]
    else:
        if chrono is not None:
            common.update(
                indices_prev=chrono.indices_prev,
                indices_next2=chrono.indices_next2,
                use_cubic=chrono.use_cubic,
                interpolate_positions=high_precision,
            )
        samples = gather_external_samples(history, indices, **common)
    return samples


def _potential(
    samples: ExternalSampleBatch, nhat: dict, charges: np.ndarray
) -> np.ndarray:
    """Evaluate only selected charges; no force computation enters the ledger."""
    samples = copy.copy(samples)
    samples.charge = np.asarray(charges, dtype=float)
    samples.valid_mask = samples.charge != 0.0
    values = compute_vectorized_contributions(
        h=0.0,
        charge_i=C_MMNS,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.asarray(nhat["nx"]),
        nhat_ny=np.asarray(nhat["ny"]),
        nhat_nz=np.asarray(nhat["nz"]),
        R_separation=np.asarray(nhat["R"]),
        samples=samples,
        apply_external=True,
        verbosity=0,
    )
    return np.asarray(values[4:8], dtype=float)


def sector_change(
    old: dict,
    model: str,
    charges: np.ndarray,
    evaluate: Callable[[str, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Compare inclusion at one event, cancelling unchanged sources exactly."""
    old_charges = np.asarray(old.get("charges", np.zeros_like(charges)), dtype=float)
    if "ids" in old:
        dense = np.zeros_like(charges, dtype=float)
        ids = np.asarray(old["ids"], dtype=int)
        if np.any(ids >= len(dense)):
            raise ValueError(
                "reduced source identity is outside the full source history"
            )
        dense[ids] = old_charges
        old_charges = dense
    if old_charges.shape != charges.shape:
        raise ValueError("sampled source identities changed; use a full stable history")
    old_model = old.get("model", model)
    if old_model == model:
        delta = charges - old_charges
        return evaluate(model, delta) if np.any(delta) else np.zeros(4)
    return evaluate(model, charges) - evaluate(old_model, old_charges)


def sampled_inclusion_change(
    *,
    trajectory,
    trajectory_ext,
    index,
    particle,
    h,
    startup_mode,
    sim_type,
    chrono_mode,
    self_consistency,
    space_charge,
    beamline_geometry,
    traj_soa,
    traj_ext_soa,
    macroparticle_smearing,
    ordinary_gate_vector_change=None,
):
    """Return one trial's offset and detached next inclusion state.

    Every evaluation uses the accepted observer position and time. Nonlinear
    refinements reuse this offset. Only publishing the returned trial advances
    readiness, the ledger, and source-set metadata.
    """
    # Lazy import avoids an equations/module import cycle.
    from . import equations as eq
    from .distances import compute_retarded_distance, compute_retarded_distance_soa
    from .macroparticle_smearing import fixed_cloud_offsets, smear_source_samples
    from .beamline_geometry import compute_directional_visibility_mask

    state = trajectory[index]
    ready = bool(
        state.get("sampled_source_canonical_ready", np.zeros(len(state["x"])))[particle]
    )
    encoded = state.get("potential_inclusion_state", [""] * len(state["x"]))[particle]
    previous = decode_inclusion_state(encoded) if ready else {}
    # Match the existing observer guards at startup. Changes in numerical
    # guards themselves are deliberately outside the inclusion-event design.
    observer_charge = float(eq._get_particle_observer_charge(state, particle))
    if abs(observer_charge) < 1e-20 or state["gamma"][particle] > 1e6:
        return np.zeros(4), encoded if ready else encode_inclusion_state({})
    next_sets = {}
    delta_A = np.zeros(4)
    position = tuple(float(state[axis][particle]) for axis in "xyz")
    beta = tuple(float(state["b" + axis][particle]) for axis in "xyz")
    high_precision = bool(getattr(self_consistency, "chrono_high_precision", False))

    external_last = min(index, len(trajectory_ext) - 1)
    while external_last >= 0 and "x" not in trajectory_ext[external_last]:
        external_last -= 1
    if external_last >= 0 and len(trajectory_ext[external_last]["x"]):
        old = previous.get("external", {})
        skip = False
        if startup_mode is StartupMode.COLD_START:
            latest = trajectory_ext[external_last]
            max_R = float(
                np.max(
                    np.linalg.norm(
                        np.stack([latest[a] for a in "xyz"], axis=-1) - position,
                        axis=-1,
                    )
                )
            )
            beta_avg = np.linalg.norm([state["beta_avg_" + a][particle] for a in "xyz"])
            travel = np.linalg.norm(
                np.asarray(position) - [state["origin_" + a][particle] for a in "xyz"]
            )
            skip = travel < beta_avg * max_R / (1.0 + beta_avg)
        # Closed gates with no previously included sources need no sampling.
        if skip and not np.any(old.get("charges", [])):
            # Even a closed gate uses the same source identities as its open
            # representation. Smearing expands every centroid into subcharges.
            source_count = len(trajectory_ext[external_last]["x"])
            if (
                macroparticle_smearing is not None
                and macroparticle_smearing.enabled
                and (
                    macroparticle_smearing.apply_to_active_sources
                    or macroparticle_smearing.apply_to_passive_sources
                )
            ):
                source_count *= max(1, int(macroparticle_smearing.subcharge_count))
            charges = np.zeros(source_count)
            next_sets["external"] = {
                "model": "retarded",
                "charges": charges.tolist(),
                "gate_open": False,
            }
        else:
            chrono = None
            if startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY:
                source_index = external_last
                nhat, indices = eq._compute_approximate_retarded_distance(
                    state, trajectory_ext[source_index], particle, source_index
                )
            else:
                nhat, indices, chrono = eq._compute_full_retarded_distance(
                    trajectory,
                    trajectory_ext,
                    index,
                    particle,
                    chrono_mode,
                    self_consistency,
                    timestep_h=h,
                    **({"traj_soa": traj_soa} if traj_soa is not None else {}),
                    **(
                        {"traj_ext_soa": traj_ext_soa}
                        if traj_ext_soa is not None
                        else {}
                    ),
                )
            samples = _gather(
                trajectory_ext, traj_ext_soa, indices, chrono, high_precision
            )
            samples, smeared = smear_source_samples(
                samples=samples,
                observer_position=position,
                config=macroparticle_smearing,
                step_index=index,
                fixed_offsets=fixed_cloud_offsets(
                    trajectory_ext, macroparticle_smearing
                ),
            )
            if smeared:
                nhat = smeared
            gate_open = not skip and eq._should_apply_external_forces(
                startup_mode, sim_type, nhat, state, particle
            )
            included = samples.valid_mask.copy()
            included &= gate_open
            if beamline_geometry is not None and beamline_geometry.enabled:
                included &= compute_directional_visibility_mask(
                    np.stack([samples.x, samples.y, samples.z], axis=-1),
                    beamline_geometry,
                    observer_direction=beta,
                )
            charges = np.where(included, samples.charge, 0.0)
            if not ready or sim_type is SimulationType.BUNCH_TO_BUNCH:
                delta_A += sector_change(
                    old,
                    "retarded",
                    charges,
                    lambda model, q: _potential(samples, nhat, q),
                )
            elif gate_open and not old.get("gate_open", False):
                # Image evolution is outside this task, but admitting the
                # entire image class through COLD_START is the same gate event.
                delta_A += _potential(samples, nhat, charges)
            elif not gate_open and old.get("gate_open", False):
                old_charges = np.asarray(old.get("charges", []), dtype=float)
                if old_charges.shape == samples.charge.shape:
                    # Current image weights belong to ordinary evolution.
                    # Remove only the currently represented, previously included
                    # image set; never subtract yesterday's image potential.
                    counterfactual = np.where(old_charges != 0, samples.charge, 0)
                    delta_A -= _potential(samples, nhat, counterfactual)
            # Canonical P retains the preceding step's sampled vector
            # potential. Established sources therefore subtract its ordinary
            # event-to-event change during mechanical reconstruction. On the
            # first open-gate step that preceding sample is absent: inclusion
            # initializes A at this event, leaving a full canonical impulse
            # where subsequent steps receive the mechanical impulse. Prime
            # the missing ordinary change using the same source set at both
            # events. This is separate from the event-local inclusion ledger.
            if (
                ordinary_gate_vector_change is not None
                and ready
                and index > 0
                and startup_mode is StartupMode.COLD_START
                and sim_type is SimulationType.BUNCH_TO_BUNCH
                and gate_open
                and not old.get("gate_open", False)
            ):
                old_charges = np.asarray(old.get("charges", np.zeros_like(charges)))
                admitted = np.where(old_charges == 0.0, charges, 0.0)
                if np.any(admitted):
                    prior_nhat, prior_indices, prior_chrono = (
                        eq._compute_full_retarded_distance(
                            trajectory,
                            trajectory_ext,
                            index - 1,
                            particle,
                            chrono_mode,
                            self_consistency,
                            timestep_h=h,
                            traj_soa=traj_soa,
                            traj_ext_soa=traj_ext_soa,
                        )
                    )
                    prior_samples = _gather(
                        trajectory_ext,
                        traj_ext_soa,
                        prior_indices,
                        prior_chrono,
                        high_precision,
                    )
                    prior_position = (
                        float(trajectory[index - 1]["x"][particle]),
                        float(trajectory[index - 1]["y"][particle]),
                        float(trajectory[index - 1]["z"][particle]),
                    )
                    prior_samples, prior_smeared = smear_source_samples(
                        samples=prior_samples,
                        observer_position=prior_position,
                        config=macroparticle_smearing,
                        step_index=index - 1,
                        fixed_offsets=fixed_cloud_offsets(
                            trajectory_ext, macroparticle_smearing
                        ),
                    )
                    if prior_smeared:
                        prior_nhat = prior_smeared
                    ordinary_gate_vector_change[:] = (
                        _potential(samples, nhat, admitted)[:3]
                        - _potential(prior_samples, prior_nhat, admitted)[:3]
                    )

            next_sets["external"] = {
                "model": "retarded",
                "charges": charges.tolist(),
                "gate_open": bool(gate_open),
            }

    if space_charge is not None and space_charge.enabled and len(state["x"]) > 1:
        old = previous.get("same_bunch", {})
        model = (
            "retarded"
            if len(trajectory) > space_charge.resolve_min_retarded_steps(h)
            else "stationary"
        )
        cache = {}

        def same_bunch_event(representation):
            if representation in cache:
                return cache[representation]
            if representation == "retarded":
                nhat, indices, chrono = eq._compute_full_retarded_distance(
                    trajectory,
                    trajectory,
                    index,
                    particle,
                    ChronoMatchingMode.FAST,
                    self_consistency,
                    timestep_h=h,
                    traj_soa=traj_soa,
                    traj_ext_soa=traj_soa,
                )
            else:
                indices = np.full(len(state["x"]), index, dtype=int)
                chrono = None
                if traj_soa is not None:
                    nhat = compute_retarded_distance_soa(
                        traj_soa, traj_soa, index, particle, indices
                    )
                else:
                    nhat = compute_retarded_distance(
                        trajectory, trajectory, index, particle, indices
                    )
            nhat = dict(nhat)
            softening = float(space_charge.softening_mm)
            if softening > 0.0:
                nhat["R"] = np.sqrt(np.asarray(nhat["R"]) ** 2 + softening**2)
            samples = _gather(trajectory, traj_soa, indices, chrono, high_precision)
            if representation == "stationary":
                for axis in "xyz":
                    setattr(samples, "b" + axis, np.zeros_like(samples.charge))
                    setattr(samples, "bdot" + axis, np.zeros_like(samples.charge))
                samples.gamma = np.ones_like(samples.charge)
            samples.charge[particle] = 0.0
            samples.valid_mask[particle] = False
            samples, smeared = smear_source_samples(
                samples=samples,
                observer_position=position,
                config=macroparticle_smearing,
                step_index=index,
                fixed_offsets=fixed_cloud_offsets(trajectory, macroparticle_smearing),
            )
            if smeared:
                nhat = dict(smeared)
                if softening > 0.0:
                    nhat["R"] = np.sqrt(np.asarray(nhat["R"]) ** 2 + softening**2)
            cache[representation] = (samples, nhat)
            return samples, nhat

        samples, _ = same_bunch_event(model)
        charges = np.where(samples.valid_mask, samples.charge, 0.0)

        def same_potential(rep: str, q: np.ndarray) -> np.ndarray:
            event_samples, event_nhat = same_bunch_event(rep)
            cloud = fixed_cloud_offsets(trajectory, macroparticle_smearing)
            if rep != "stationary" or cloud is None:
                return _potential(event_samples, event_nhat, q)
            from .instantaneous_space_charge import plummer_force_potential

            sources = np.column_stack([getattr(event_samples, a) for a in "xyz"])
            phi = 0.0
            for offset in cloud[particle]:
                _, value = plummer_force_potential(
                    np.asarray(position) + offset - sources,
                    q,
                    observer_charge,
                    float(space_charge.softening_mm) ** 2,
                )
                phi += value / len(cloud[particle])
            return np.asarray((0.0, 0.0, 0.0, phi))

        delta_A += sector_change(old, model, charges, same_potential)
        next_sets["same_bunch"] = {"model": model, "charges": charges.tolist()}

    # A source class cannot silently disappear: disabled classes with prior
    # sources must still supply a history for the counterfactual evaluation.
    missing = previous.keys() - next_sets.keys()
    if any(np.any(previous[k].get("charges", [])) for k in missing):
        raise ValueError(
            "included source history disappeared before continuity evaluation"
        )
    return delta_A, (
        encoded
        if ready and next_sets == previous
        else encode_inclusion_state(next_sets)
    )
