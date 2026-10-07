"""CPU float64 causal material-history kernels, ported without study imports.

completion_step2k.py: ballistic completion; field_step2m.py: C2 field and
complete trajectory derivative, including smooth physical endpoint curvature.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False)
def arrival(u, r, z, beta, inverse_gamma2, z_at_zero):
    c = z_at_zero - z
    b = u + beta * c
    d = u * u - r * r - c * c
    root = np.sqrt(max(0.0, b * b - inverse_gamma2 * d))
    return (b + root) / inverse_gamma2 if b >= 0 else d / (b - root)


@njit(cache=False)
def replay(
    times,
    p,
    j,
    r,
    z,
    vol,
    cells,
    components,
    inactive,
    gamma,
    q,
    queries,
    nodes,
    qw,
    z_at_zero,
):
    beta = np.sqrt(1.0 - gamma**-2)
    aa = gamma**-2
    curves = np.zeros(len(queries))
    material_work = 0.0
    terminal = 0.0
    # Each quadrature point has its own retarded upper integration limit.
    for n in range(len(r)):
        cell = cells[n]
        end = min(times[-1], inactive[n])
        limits = np.empty(len(queries))
        for h in range(len(queries)):
            t = queries[h]
            dz = z_at_zero + beta * t - z[n]
            distance = np.hypot(r[n], dz)
            # Rationalized t-R avoids cancellation in late, forward queries.
            u = (
                aa * t * t
                + 2 * beta * (z[n] - z_at_zero) * t
                - (z[n] - z_at_zero) ** 2
                - r[n] ** 2
            ) / (t + distance)
            limits[h] = min(end, max(times[0], u))
        running = 0.0
        k = 0
        previous = times[0]
        for h in range(len(queries) + 1):
            upper = end if h == len(queries) else limits[h]
            while previous < upper - 1e-14:
                while k + 1 < len(times) - 1 and times[k + 1] <= previous + 1e-14:
                    k += 1
                right = min(upper, times[k + 1])
                dt = times[k + 1] - times[k]
                p0, p1 = p[k, cell], p[k + 1, cell]
                j0, j1 = j[k, cell], j[k + 1, cell]
                c2 = (3 * (p1 - p0) / dt - 2 * j0 - j1) / dt
                c3 = (2 * (p0 - p1) / dt + j0 + j1) / (dt * dt)
                for x, w in zip(nodes, qw):
                    u = (previous + right) / 2 + (right - previous) * x / 2
                    s = u - times[k]
                    current = j0 + s * (2 * c2 + 3 * s * c3)
                    gap = z[n] - z_at_zero - beta * u
                    incident = (
                        q
                        * gamma
                        * (r[n] if components[n] == 0 else gap)
                        / (4 * np.pi * (r[n] ** 2 + (gamma * gap) ** 2) ** 1.5)
                    )
                    running += vol[n] * (right - previous) / 2 * w * incident * current
                previous = right
            if h == len(queries):
                material_work += running
                if np.isfinite(inactive[n]):
                    continue
                t = arrival(end, r[n], z[n], beta, aa, z_at_zero)
                distance = t - end
                nr, nz = -r[n] / distance, (z_at_zero + beta * t - z[n]) / distance
                up = (aa * distance * distance + beta * beta * r[n] * r[n]) / (
                    distance * (distance + beta * (z_at_zero + beta * t - z[n]))
                )
                far = nz * nr if components[n] == 0 else -nr * nr
                terminal += (
                    vol[n]
                    * (
                        -q * beta * far * j[-1, cell] / up / distance
                        + q
                        * p[-1, cell]
                        * (nr if components[n] == 0 else nz)
                        / distance**2
                    )
                    / (4 * np.pi)
                )
                continue
            t = queries[h]
            dz = z_at_zero + beta * t - z[n]
            distance = np.hypot(r[n], dz)
            u = limits[h]
            # No material existed before t=0.
            if u <= times[0]:
                continue
            pp = jj = 0.0
            if u < end - 1e-13:
                kk = min(np.searchsorted(times, u, side="right") - 1, len(times) - 2)
                dt = times[kk + 1] - times[kk]
                s = u - times[kk]
                c2 = (
                    3 * (p[kk + 1, cell] - p[kk, cell]) / dt
                    - 2 * j[kk, cell]
                    - j[kk + 1, cell]
                ) / dt
                c3 = (
                    2 * (p[kk, cell] - p[kk + 1, cell]) / dt
                    + j[kk, cell]
                    + j[kk + 1, cell]
                ) / (dt * dt)
                pp = p[kk, cell] + s * (j[kk, cell] + s * (c2 + s * c3))
                jj = j[kk, cell] + s * (2 * c2 + 3 * s * c3)
            elif not np.isfinite(inactive[n]):
                pp = p[-1, cell]  # Explicit frozen-P, stopped-J continuation.
            nr, nz = -r[n] / distance, dz / distance
            up = (aa * distance * distance + beta * beta * r[n] * r[n]) / (
                distance * (distance + beta * dz)
            )
            far = nz * nr if components[n] == 0 else -nr * nr
            endpoint = q * beta * far * jj / (4 * np.pi * distance * up) - q * pp * (
                nr if components[n] == 0 else nz
            ) / (4 * np.pi * distance**2)
            curves[h] += -running + vol[n] * endpoint
    return curves, material_work, terminal


@njit(cache=False)
def field(
    t,
    delta_z,
    velocity,
    beta,
    times,
    p,
    j,
    r,
    z,
    vol,
    cells,
    comp,
    inactive,
    taper,
    z_at_zero,
    smooth_endpoints,
):
    value = derivative = 0.0
    for n in range(len(r)):
        dz = z_at_zero + beta * t + delta_z - z[n]
        distance = np.hypot(r[n], dz)
        # Rationalized subtraction is essential for t ~ gamma^2 and beyond.
        centre = z[n] - z_at_zero - delta_z
        u = (
            (1 - beta) * (1 + beta) * t * t
            + 2 * beta * centre * t
            - centre * centre
            - r[n] * r[n]
        ) / (t + distance)
        if u < times[0] or u >= inactive[n]:
            continue
        cell = cells[n]
        if u >= times[-1] + taper:
            aend = (j[-1, cell] - j[-2, cell]) / (times[-1] - times[-2])
            pp = p[-1, cell] + j[-1, cell] * taper / 2 + aend * taper * taper / 12
            jj = aa = jerk = 0.0
        else:
            if u >= times[-1]:
                # Smoothly bring J to zero, preserving P and J at the join.
                dt = taper
                s = (u - times[-1]) / dt
                p0 = p[-1, cell]
                j0 = j[-1, cell]
                a0 = (j[-1, cell] - j[-2, cell]) / (times[-1] - times[-2])
                p1 = p0 + j0 * dt / 2 + a0 * dt * dt / 12
                j1 = a1 = 0.0
            else:
                k = min(np.searchsorted(times, u, side="right") - 1, len(times) - 2)
                dt = times[k + 1] - times[k]
                s = (u - times[k]) / dt
                p0 = p[k, cell]
                p1 = p[k + 1, cell]
                j0 = j[k, cell]
                j1 = j[k + 1, cell]
                a0 = (j[min(k + 1, len(times) - 1), cell] - j[max(k - 1, 0), cell]) / (
                    times[min(k + 1, len(times) - 1)] - times[max(k - 1, 0)]
                )
                a1 = (j[min(k + 2, len(times) - 1), cell] - j[k, cell]) / (
                    times[min(k + 2, len(times) - 1)] - times[k]
                )
                # Physical C3 preparation and opening have zero P'' at
                # their joins. A one-sided current difference would invent
                # a finite acceleration jump even with a smooth coupling.
                if smooth_endpoints and k == 0:
                    a0 = 0.0
                if smooth_endpoints and np.isfinite(inactive[n]):
                    if abs(times[k] - inactive[n]) < 1e-10:
                        a0 = 0.0
                    if abs(times[k + 1] - inactive[n]) < 1e-10:
                        a1 = 0.0
            rp = p1 - p0 - dt * j0 - dt * dt * a0 / 2
            rj = dt * (j1 - j0) - dt * dt * a0
            ra = dt * dt * (a1 - a0)
            c3 = 10 * rp - 4 * rj + ra / 2
            c4 = -15 * rp + 7 * rj - ra
            c5 = 6 * rp - 3 * rj + ra / 2
            pp = (
                p0
                + dt * j0 * s
                + dt * dt * a0 * s * s / 2
                + s**3 * (c3 + s * (c4 + s * c5))
            )
            jj = j0 + dt * a0 * s + s * s * (3 * c3 + s * (4 * c4 + 5 * c5 * s)) / dt
            aa = a0 + s * (6 * c3 + s * (12 * c4 + 20 * c5 * s)) / (dt * dt)
            jerk = (6 * c3 + s * (24 * c4 + 60 * c5 * s)) / (dt**3)
        nr, nz = -r[n] / distance, dz / distance
        rd = velocity * nz
        npz = velocity * nr * nr / distance
        npr = -nr * rd / distance
        up = (
            (1 - velocity) * (1 + velocity) * distance * distance
            + velocity * velocity * r[n] * r[n]
        ) / (distance * (distance + velocity * dz))
        if comp[n] == 0:
            near, far = 3 * nz * nr, nz * nr
            nearp, farp = 3 * (npz * nr + nz * npr), npz * nr + nz * npr
        else:
            near, far = 2 - 3 * nr * nr, -nr * nr
            nearp, farp = -6 * nr * npr, -2 * nr * npr
        induction = pp / distance**3 + jj / distance**2
        ef = near * induction + far * aa / distance
        df = (
            nearp * induction
            + near
            * (
                jj * up / distance**3
                - 3 * pp * rd / distance**4
                + aa * up / distance**2
                - 2 * jj * rd / distance**3
            )
            + farp * aa / distance
            + far * (jerk * up / distance - aa * rd / distance**2)
        )
        value += vol[n] * ef / (4 * np.pi)
        derivative += vol[n] * df / (4 * np.pi)
    return value, derivative
