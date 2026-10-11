"""Endpoint potentials and accepted inclusion metadata for reduced source solves."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .distances import compute_retarded_distance, compute_retarded_distance_soa
from .instantaneous_space_charge import plummer_force_potential
from .macroparticle_smearing import fixed_cloud_offsets, smear_source_samples
from .potential_inclusion import (
    _gather,
    _potential,
    decode_inclusion_state,
    encode_inclusion_state,
)
from .pseudo_grid import ActiveTrajectoryView, slice_trajectory_particle_history
from .types import ChronoMatchingMode, IndexedTrajectoryArrays, StartupMode


class ReducedPotentialEvaluator:
    """Use original particle identities for current and counterfactual source sets.

    Only nonzero representatives are sampled. Removed representatives are sampled
    at the current event for the inclusion difference, never at yesterday's event.
    Potentials and ledger entries use x, y, z, t order internally.
    """

    def __init__(
        self,
        *,
        observer_history,
        source_history,
        observer_soa,
        source_soa,
        source_indices,
        source_charges,
        same_indices,
        same_charges,
        same_radii,
        h,
        startup_mode,
        sim_type,
        chrono_mode,
        self_consistency,
        space_charge,
        macroparticle_smearing,
        beamline_geometry,
    ):
        self.observer_history = observer_history
        self.source_history = source_history
        self.observer_soa = observer_soa
        self.source_soa = source_soa
        self.h = h
        self.startup_mode = startup_mode
        self.sim_type = sim_type
        self.chrono_mode = chrono_mode
        self.self_consistency = self_consistency
        self.space_charge = space_charge
        self.smearing = macroparticle_smearing
        self.geometry = beamline_geometry
        self.source_indices = source_indices
        self.source_charges = source_charges
        self.same_indices = same_indices
        self.same_charges = same_charges
        self.same_radii = same_radii
        self.observer_cloud = fixed_cloud_offsets(observer_history, self.smearing)
        self.source_cloud = fixed_cloud_offsets(source_history, self.smearing)
        self._histories = {}

    def _sources(self, sector, ids):
        key = (sector, tuple(ids))
        if key not in self._histories:
            history, soa = (
                (self.source_history, self.source_soa)
                if sector == "external"
                else (self.observer_history, self.observer_soa)
            )
            if soa is not None:
                if isinstance(soa, IndexedTrajectoryArrays):
                    subset_soa = IndexedTrajectoryArrays(
                        soa.base,
                        soa.particle_indices[np.asarray(ids, dtype=int)],
                        start_step=soa.start_step,
                    )
                else:
                    subset_soa = IndexedTrajectoryArrays(
                        soa, np.asarray(ids, dtype=int)
                    )
                subset = ActiveTrajectoryView(subset_soa)
            else:
                subset_soa = None
                subset = slice_trajectory_particle_history(
                    history, np.asarray(ids, dtype=int)
                )
            self._histories[key] = subset, subset_soa
        return self._histories[key]

    def _event(
        self,
        observer_history,
        particle,
        sector,
        ids,
        charges,
        model,
        radii,
        *,
        gate=True,
        visibility_mask=None,
    ):
        from . import equations as eq
        from .beamline_geometry import compute_directional_visibility_mask

        if not len(ids):
            cloud_count = (
                max(1, int(self.smearing.subcharge_count))
                if self.smearing is not None
                and self.smearing.enabled
                and (
                    self.smearing.apply_to_active_sources
                    or self.smearing.apply_to_passive_sources
                )
                else 1
            )
            return (
                np.zeros(4),
                np.zeros(0),
                False,
                np.zeros((0, cloud_count), dtype=bool),
            )
        sources, source_soa = self._sources(sector, ids)
        index = len(observer_history) - 1
        state = observer_history[index]
        position = np.array([state[a][particle] for a in "xyz"])
        beta = tuple(state["b" + a][particle] for a in "xyz")
        source_index = min(index, len(sources) - 1)
        # Chronology and distance read only the current observer row. Broadcast
        # that event without materializing earlier observer/source dictionaries.
        event_soa = None
        if source_soa is not None:
            event_soa = SimpleNamespace(
                n_steps=index + 1,
                n_particles=len(state["x"]),
                **{
                    name: np.broadcast_to(
                        np.asarray(state[name]), (index + 1, len(state["x"]))
                    )
                    for name in ("x", "y", "z", "t")
                },
            )
        skip = False
        if sector == "external" and self.startup_mode is StartupMode.COLD_START:
            latest = sources[source_index]
            max_r = np.max(
                np.linalg.norm(
                    np.column_stack([latest[a] for a in "xyz"]) - position, axis=1
                )
            )
            beta_avg = np.linalg.norm([state["beta_avg_" + a][particle] for a in "xyz"])
            travel = np.linalg.norm(
                position - [state["origin_" + a][particle] for a in "xyz"]
            )
            skip = travel < beta_avg * max_r / (1 + beta_avg)
        chrono = None
        if model == "stationary":
            indices = np.full(len(ids), source_index, dtype=int)
            nhat = (
                compute_retarded_distance_soa(
                    event_soa, source_soa, index, particle, indices
                )
                if event_soa is not None
                else compute_retarded_distance(
                    observer_history, sources, index, particle, indices
                )
            )
        elif (
            sector == "external"
            and self.startup_mode is StartupMode.APPROXIMATE_BACK_HISTORY
        ):
            nhat, indices = eq._compute_approximate_retarded_distance(
                state, sources[source_index], particle, source_index
            )
        else:
            nhat, indices, chrono = eq._compute_full_retarded_distance(
                observer_history,
                sources,
                index,
                particle,
                self.chrono_mode if sector == "external" else ChronoMatchingMode.FAST,
                self.self_consistency,
                timestep_h=self.h,
                traj_soa=event_soa,
                traj_ext_soa=source_soa,
            )
        samples = _gather(
            sources,
            source_soa,
            indices,
            chrono,
            bool(self.self_consistency and self.self_consistency.chrono_high_precision),
        )
        # Caller supplies effective charges; current deaths remain excluded.
        included = samples.valid_mask.copy() if gate else np.ones(len(ids), dtype=bool)
        if sector == "external" and gate:
            included &= not skip and eq._should_apply_external_forces(
                self.startup_mode, self.sim_type, nhat, state, particle
            )
        admitted = np.where(included, charges, 0.0)
        samples.charge = admitted.copy()
        samples.valid_mask = admitted != 0
        if model == "stationary":
            for a in "xyz":
                setattr(samples, "b" + a, np.zeros(len(ids)))
                setattr(samples, "bdot" + a, np.zeros(len(ids)))
            samples.gamma = np.ones(len(ids))
        cloud = self.source_cloud if sector == "external" else self.observer_cloud
        samples, smeared = smear_source_samples(
            samples=samples,
            observer_position=tuple(position),
            config=self.smearing,
            step_index=index,
            fixed_offsets=None if cloud is None else cloud[ids],
        )
        # Visibility is evaluated on the same expanded source samples as
        # the active force solve. Retain each subcharge's admission separately.
        subcharges = len(samples.charge) // len(ids)
        visible = np.ones((len(ids), subcharges), dtype=bool)
        if visibility_mask is not None:
            visible = np.asarray(visibility_mask, dtype=bool)
            if visible.shape != (len(ids), subcharges):
                raise ValueError("source visibility mask must match expanded cloud")
        elif (
            sector == "external"
            and gate
            and self.geometry is not None
            and self.geometry.enabled
        ):
            visible = compute_directional_visibility_mask(
                np.column_stack([getattr(samples, a) for a in "xyz"]),
                self.geometry,
                observer_direction=beta,
            ).reshape(len(ids), subcharges)
        samples.valid_mask &= visible.ravel()
        softening = (
            float(self.space_charge.softening_mm) if sector == "same_bunch" else 0.0
        )
        if model == "stationary":
            subcharges = len(samples.charge) // len(ids)
            eps2 = softening**2 + np.repeat(np.asarray(radii) ** 2, subcharges)
            displacement = position - np.column_stack(
                [getattr(samples, a) for a in "xyz"]
            )
            q = np.where(samples.valid_mask, samples.charge, 0)
            if self.observer_cloud is not None and smeared:
                offsets = self.observer_cloud[particle]
                displacement = (displacement[None, :, :] + offsets[:, None, :]).reshape(
                    -1, 3
                )
                q = np.tile(q / len(offsets), len(offsets))
                eps2 = np.tile(eps2, len(offsets))
            _, phi = plummer_force_potential(displacement, q, 1.0, eps2)
            value = np.array([0.0, 0.0, 0.0, phi])
        else:
            nhat = dict(smeared or nhat)
            if sector == "same_bunch":
                # Match the active path: smeared distances retain softening;
                # unsmeared distances also include representative source radii.
                radius2 = 0 if smeared else np.asarray(radii) ** 2
                nhat["R"] = np.sqrt(np.asarray(nhat["R"]) ** 2 + softening**2 + radius2)
            value = _potential(
                samples, nhat, np.where(samples.valid_mask, samples.charge, 0.0)
            )
        return value, admitted, bool(np.any(included)), visible

    def evaluate(self, particle, *, endpoint=None, previous=None, ledger=True):
        history = (
            self.observer_history
            if endpoint is None
            else list(self.observer_history) + [endpoint]
        )
        state = history[-1]
        previous = self.observer_history[-1] if previous is None else previous
        old = decode_inclusion_state(
            previous.get("potential_inclusion_state", [""] * len(previous["x"]))[
                particle
            ]
        )
        total = np.zeros(4)
        delta = np.zeros(4)
        next_sets = {}
        sectors = [
            (
                "external",
                self.source_indices,
                self.source_charges,
                "retarded",
                np.zeros(len(self.source_indices)),
            )
        ]
        if (
            self.space_charge is not None
            and self.space_charge.enabled
            and len(state["x"]) > 1
        ):
            # Freeze the active step's startup model also at its endpoints.
            model = (
                "retarded"
                if len(self.observer_history)
                > self.space_charge.resolve_min_retarded_steps(self.h)
                else "stationary"
            )
            sectors.append(
                (
                    "same_bunch",
                    self.same_indices,
                    self.same_charges[particle],
                    model,
                    self.same_radii,
                )
            )
        for sector, ids, charges, model, radii in sectors:
            selected = np.asarray(charges) != 0
            ids = np.asarray(ids, dtype=int)[selected]
            charges = np.asarray(charges)[selected]
            radii = np.asarray(radii)[selected]
            value, admitted, opened, visible = self._event(
                history, particle, sector, ids, charges, model, radii
            )
            total += value
            kept = admitted != 0
            # Sparse original identities avoid storing dense bunch charge rows.
            next_sets[sector] = dict(
                model=model,
                ids=ids[kept].tolist(),
                charges=admitted[kept].tolist(),
                radii=radii[kept].tolist(),
                gate_open=opened,
            )
            if (
                sector == "external"
                and self.geometry is not None
                and self.geometry.enabled
            ):
                next_sets[sector]["visibility_mask"] = visible[kept].tolist()
            if ledger:
                prior = old.get(sector, {})
                old_q = np.asarray(prior.get("charges", []), dtype=float)
                old_ids = np.asarray(prior.get("ids", np.arange(len(old_q))), dtype=int)
                old_r = np.asarray(
                    prior.get("radii", np.zeros(len(old_q))), dtype=float
                )
                old_visible = np.asarray(
                    prior.get(
                        "visibility_mask",
                        np.ones((len(old_q), visible.shape[1]), dtype=bool),
                    ),
                    dtype=bool,
                ).reshape(len(old_q), visible.shape[1])
                selected_old = old_q != 0
                old_visible = old_visible[selected_old]
                old_ids, old_q, old_r = (
                    old_ids[selected_old],
                    old_q[selected_old],
                    old_r[selected_old],
                )
                # Cancel unchanged representations exactly, including ordinary
                # motion. Counterfactual removed sources bypass current masks.
                if next_sets[sector] == prior:
                    continue
                union = np.union1d(ids, old_ids)
                new_q = np.zeros(len(union))
                prior_q = np.zeros(len(union))
                new_r = np.zeros(len(union))
                prior_r = np.zeros(len(union))
                new_pos = np.searchsorted(union, ids)
                old_pos = np.searchsorted(union, old_ids)
                new_q[new_pos] = admitted
                prior_q[old_pos] = old_q
                new_r[new_pos] = radii
                prior_r[old_pos] = old_r
                old_model = prior.get("model", model)
                common = (new_q != 0) & (prior_q != 0)
                new_masks = np.ones((len(union), visible.shape[1]), dtype=bool)
                prior_masks = new_masks.copy()
                new_masks[new_pos] = visible
                prior_masks[old_pos] = old_visible
                if (
                    model == old_model
                    and np.array_equal(new_r[common], prior_r[common])
                    and np.array_equal(new_masks[common], prior_masks[common])
                ):
                    dq = new_q - prior_q
                    chosen = dq != 0
                    r = np.where(new_q != 0, new_r, prior_r)
                    delta += self._event(
                        history,
                        particle,
                        sector,
                        union[chosen],
                        dq[chosen],
                        model,
                        r[chosen],
                        gate=False,
                        visibility_mask=np.where(
                            (new_q != 0)[:, None], new_masks, prior_masks
                        )[chosen],
                    )[0]
                else:
                    prior_value = self._event(
                        history,
                        particle,
                        sector,
                        old_ids,
                        old_q,
                        old_model,
                        old_r,
                        gate=False,
                        visibility_mask=old_visible,
                    )[0]
                    delta += value - prior_value
        if endpoint is not None:
            next_sets["external"]["passive_endpoint"] = True
        if not np.all(np.isfinite(total)) or not np.all(np.isfinite(delta)):
            raise ValueError("reduced endpoint potential must be finite")
        return total, delta, encode_inclusion_state(next_sets)
