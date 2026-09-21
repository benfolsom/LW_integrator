"""Coupled full-spin radiative self-field and local source motion.

Working full-spin reaction model. The auxiliary self-field retains its value
and first spacetime derivatives; higher spacetime derivatives are omitted.
Full particle-plus-field conservation and strong-domain coverage remain open.
No state, history, spin magnitude or velocity-domain threshold is projected.
"""

import numpy as np
from scipy.optimize import least_squares

from . import momentum_center as m
from .full_dipole_reaction import (
    ordinary_source_coefficients,
    evaluate_with_self_field,
    inherit_reaction_domain,
    with_local_self_field,
)
from .full_dipole_self_field import regular_self_field

ROOT_TOLERANCE = 1e-5
APPROXIMATION = (
    "Coupled full-spin radiative self-field; local first spacetime derivatives "
    "retained, higher spacetime derivatives omitted; not validated total balance"
)


class BoostedPotential:
    """Transform the complete external potential response to a boosted frame."""

    def __init__(self, provider, boost):
        self.provider = provider
        self.boost = boost
        self.inverse = m.METRIC[:, None] * boost.T * m.METRIC[None, :]
        domain = getattr(provider, "reaction_fit_domain", None)
        if domain is not None:
            self.reaction_fit_domain = lambda event, velocity: domain(
                self.inverse @ event, self.inverse @ velocity
            )
        taylor = getattr(provider, "taylor_response", None)
        if taylor is not None:

            def expanded(events):
                a, da, f, df = taylor(events @ self.inverse.T)
                return (
                    a @ self.boost.T,
                    np.einsum("ka,nkl,bl->nab", self.inverse, da, self.boost),
                    np.einsum("ai,nij,bj->nab", self.boost, f, self.boost),
                    np.einsum(
                        "ka,bi,cj,nkij->nabc", self.inverse, self.boost, self.boost, df
                    ),
                )

            self.taylor_response = expanded

    def __call__(self, event):
        a, da, f, df = self.provider(self.inverse @ event)
        return (
            self.boost @ a,
            self.inverse.T @ da @ self.boost.T,
            self.boost @ f @ self.boost.T,
            np.einsum("ka,bi,cj,kij->abc", self.inverse, self.boost, self.boost, df),
        )


def packed(field: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    return np.r_[m.pack(field), np.concatenate([m.pack(row) for row in gradient])]


def unpacked(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return m.unpack(vector[:6]), np.array(
        [m.unpack(row) for row in vector[6:].reshape(4, 6)]
    )


def transformed_pair(
    field: np.ndarray, gradient: np.ndarray, boost: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inverse = m.METRIC[:, None] * boost.T * m.METRIC[None, :]
    return boost @ field @ boost.T, np.einsum(
        "ka,bi,cj,kij->abc", inverse, boost, boost, gradient
    )


def solve_point_rest_frame(
    state, particle, provider, *, window, guess, scale, max_nfev=20, method="centered"
):
    """Condition the same local root in the kinetic momentum rest frame.

    Independent laboratory component perturbations can be enormous physical
    perturbations after a large boost. No timelike threshold or force is changed.
    """
    a = provider(state[:4])[0]
    p = state[4:8] - particle.charge * a
    if m.dot(p, p) <= 0 or p[0] <= 0:
        raise m.VelocityDomainError(
            "Future timelike momentum required for rest-frame solve"
        )
    direction = p / np.sqrt(m.dot(p, p))
    boost = np.eye(4)
    boost[0, 0] = direction[0]
    boost[0, 1:] = boost[1:, 0] = -direction[1:]
    boost[1:, 1:] += np.outer(direction[1:], direction[1:]) / (1 + direction[0])
    inverse = m.METRIC[:, None] * boost.T * m.METRIC[None, :]
    rest_state = np.r_[
        boost @ state[:4],
        boost @ state[4:8],
        m.pack(boost @ m.unpack(state[8:]) @ boost.T),
    ]
    rest_field, rest_gradient = transformed_pair(*unpacked(scale * guess), boost)
    vector = packed(rest_field, rest_gradient)
    rest_scale = np.r_[
        np.full(6, np.linalg.norm(vector[:6]) / np.sqrt(6)),
        np.full(24, np.linalg.norm(vector[6:]) / np.sqrt(24)),
    ]
    if np.any(rest_scale == 0):
        raise ValueError(
            "Nonzero field and gradient seed required by this research root"
        )
    result, _, response = solve_point(
        rest_state,
        particle,
        BoostedPotential(provider, boost),
        window=window,
        guess=vector / rest_scale,
        scale=rest_scale,
        max_nfev=max_nfev,
        method=method,
    )
    result["residual_frame"] = "kinetic_momentum_rest"
    if not result["converged"]:
        return result, guess, None
    field, gradient = transformed_pair(
        np.array(result["field"]), np.array(result["gradient"]), inverse
    )
    result["field"], result["gradient"] = field.tolist(), gradient.tolist()
    transformed = dict(response)
    for prefix in ("", "charge_", "dipole_"):
        transformed[prefix + "field"], transformed[prefix + "gradient"] = (
            transformed_pair(
                response[prefix + "field"], response[prefix + "gradient"], inverse
            )
        )
    transformed["source_balance"] = {
        sector: {key: inverse @ value for key, value in data.items()}
        for sector, data in response["source_balance"].items()
    }
    return result, packed(field, gradient) / scale, transformed


def solve_point(
    state,
    particle,
    provider,
    *,
    window,
    guess,
    scale,
    strength=1.0,
    max_nfev=20,
    method="centered",
):
    """Return a checked local root and its generated full self-field response.

    Component scaling conditions the numerical solve; final residuals are
    normalized separately for field and gradient. The local field expansion
    remains an approximation, regardless of how small this residual becomes.
    """
    evaluations = 0
    generated = None

    def residual(coordinates):
        nonlocal evaluations, generated
        evaluations += 1
        fself, dfself = unpacked(scale * coordinates)

        supplied = with_local_self_field(provider, fself, dfself, state[:4])

        u, d, _ = ordinary_source_coefficients(
            state,
            particle,
            inherit_reaction_domain(provider, supplied),
            window=window,
            method=method,
        )
        generated = regular_self_field(u, d, charge=particle.charge)
        return (
            coordinates
            - strength * packed(generated["field"], generated["gradient"]) / scale
        )

    solution = least_squares(
        residual,
        guess,
        jac="3-point",
        diff_step=1e-3,
        max_nfev=max_nfev,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    check = residual(solution.x)
    new = solution.x - check
    errors = [
        float(np.linalg.norm(check[part]) / np.linalg.norm(new[part]))
        for part in (slice(0, 6), slice(6, 30))
    ]
    record = dict(
        strength=strength,
        converged=max(errors) < 1e-5,
        relative_residuals=errors,
        evaluations=evaluations,
        message=solution.message,
    )
    if record["converged"]:
        fself, dfself = unpacked(scale * solution.x)
        a, _, f, df = provider(state[:4])
        p = state[4:8] - particle.charge * a
        w = m.velocity_direction(
            p, m.unpack(state[8:]), f + fself, df + dfself, particle
        )[0]
        record["normalized_timelike_margin"] = m.dot(p, p) * m.dot(w, w)
        # Check the full normalization, not only the algebraic residual.
        m.velocity(p, m.unpack(state[8:]), f + fself, df + dfself, particle)
        record["field"] = fself.tolist()
        record["gradient"] = dfself.tolist()
    return record, solution.x, generated


def residuals(first, second):
    """Frobenius differences / generated field and gradient norms separately."""
    values = []
    for key in ("field", "gradient"):
        delta = float(np.linalg.norm(first[key] - second[key]))
        reference = float(np.linalg.norm(second[key]))
        values.append(
            delta / reference if reference else (0.0 if not delta else np.inf)
        )
    return values


def evaluate_coupled_reaction(state, particle, provider, *, window, method="centered"):
    """Deterministic seed per state: no step-order dependent warm-start cache."""
    if method not in ("analytic", "centered", "boundary_aware"):
        raise ValueError(
            "Coupled reaction requires analytic, centered or boundary_aware source derivatives"
        )
    u, d, initial = ordinary_source_coefficients(
        state, particle, provider, window=window, method=method
    )
    response = regular_self_field(u, d, charge=particle.charge)
    evaluations = 0
    errors = [0.0, 0.0]
    solver = "exact_coasting"
    if not initial.get("coasting_control", False):
        # Cheap full-strength substitution is accepted only after the same
        # residual check as the coupled root, never merely after a fixed count.
        for _ in range(4):
            seed = response

            supplied = with_local_self_field(
                provider, seed["field"], seed["gradient"], state[:4]
            )

            u, d, _ = ordinary_source_coefficients(
                state,
                particle,
                inherit_reaction_domain(provider, supplied),
                window=window,
                method=method,
            )
            response = regular_self_field(u, d, charge=particle.charge)
            evaluations += 1
            errors = residuals(seed, response)
            if max(errors) < ROOT_TOLERANCE:
                solver = "checked_substitution"
                break
        else:
            vector = packed(response["field"], response["gradient"])
            magnitudes = [np.linalg.norm(vector[:6]), np.linalg.norm(vector[6:])]
            floor = max(max(magnitudes) * np.finfo(float).eps, np.finfo(float).tiny)
            scale = np.r_[
                np.full(6, max(magnitudes[0], floor) / np.sqrt(6)),
                np.full(24, max(magnitudes[1], floor) / np.sqrt(24)),
            ]
            root, _, response = solve_point(
                state,
                particle,
                provider,
                window=window,
                guess=vector / scale,
                scale=scale,
                max_nfev=20,
                method=method,
            )
            evaluations += root["evaluations"]
            errors = root["relative_residuals"]
            solver = "coupled_least_squares"
            if not root["converged"]:
                raise ValueError(f"Coupled self-field residual failed: {errors}")
    rate, data = evaluate_with_self_field(state, particle, provider, response)
    data["reaction"]["mode"] = "full_dipole_coupled"
    data["reaction"]["approximation"] = APPROXIMATION
    data["reaction"]["coupled_root"] = dict(
        solver=solver,
        relative_residuals=errors,
        evaluations=evaluations,
        norm="Frobenius difference / generated field or gradient norm; exact zero control",
        derivative_method=method,
    )
    return rate, data
