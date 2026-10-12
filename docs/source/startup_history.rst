Startup-history preflight check
===============================

``COLD_START`` and ``APPROXIMATE_BACK_HISTORY`` begin with no source states
before the initial time ``t0``. At ``t0`` an observer at distance ``R`` needs
the source at about ``t0 - R/c``. During the first ``tau_c`` of observer time
(the root of ``|x_obs(t) - x_src(t0)| = c (t - t0)``) that part of the source
worldline does not exist, so the field is suppressed or approximated. If a
significant part of the interaction falls in that window, for example a close
approach or a visibility gate that opens soon after ``t0``, the result is wrong
without any error. ``COLD_START`` is valid only where no strongly affecting
field acts for long enough to build a history.

What is checked
---------------

Before integration, ``retarded_integrator`` evaluates every ``BUNCH_TO_BUNCH``
rider/driver observer-source pair in both directions, for every start-up mode
other than ``INERTIAL_PREHISTORY``. Large bunches use a deterministic subset:
at most 24 observers, each against up to 24 evenly strided sources plus its 8
nearest sources. Same-bunch space charge (explicit start-up handling) and
conducting-wall image sources are not point worldlines and are not estimated.
A checkpoint resume skips the check because history already exists.

Each source worldline is the straight-line extrapolation of its initial state,
and each observer is extrapolated the same way over the nominal run,
``(steps - 1) * h_step * gamma`` of lab time. The force is the
acceleration-free Lienard-Wiechert field at the analytic retarded point. For
each observer the check forms

.. math::

   \varepsilon = \frac{\sum_j |q_j| \int_{t_0}^{t_c} |F_j|\,dt}
                      {\sum_j |q_j| \int_{t_0}^{t_0+T} |F_j|\,dt},

the fraction of the time integral of the force magnitude (a conservative
impulse measure that cannot cancel by sign) that falls in the missing-history
interval. The worst observer is reported with its largest contributing pair.
Integration-time effects such as adaptive stepping and radiation reaction are
not modelled, so the estimate is a screen, not a correction.

If a beamline geometry is enabled, the directional visibility mask is applied
at the retarded source point, as in the integrator. Any visible, non-zero field
at a retarded time before the source history exists means a gate window
intersects the missing region and is an error regardless of ``epsilon``.

Failure and override
--------------------

By default the run is rejected with ``StartupHistoryError`` (a ``ValueError``)
when ``epsilon > 1e-3`` or a gate window intersects the missing region. The
message names the pair, the missing interval, the run length and the missing
fraction, and recommends ``INERTIAL_PREHISTORY`` (exact path) or a larger
starting separation. For a deliberate diagnostic set
``allow_incomplete_startup_history`` (testbed and run-config JSON key,
``--allow-incomplete-startup-history`` on the CLI, ``IntegratorConfig`` field,
or ``retarded_integrator`` keyword). The same message is then issued as a
``RuntimeWarning`` and logged as ``STARTUP-HISTORY OVERRIDE``. The tolerance is
``startup_history_tolerance`` (``--startup-history-tolerance``).

A run that passes is unchanged: the check reads the initial states and does not
modify them.

Sampled path and INERTIAL_PREHISTORY
------------------------------------

``INERTIAL_PREHISTORY`` is an exact-path mode: it builds an analytic
straight-line source history for the light-cone root solver. The sampled path
(default, chrono matching) is not given the same mode. Its source histories are
index-aligned trajectory lists, so a prehistory would have to be prepended to
``trajectory_ext`` and every consumer of the step index (chrono matching, the
``COLD_START`` origin gating, ``APPROXIMATE_BACK_HISTORY`` lookups, checkpoint
and output trajectories, energy ledgers) would have to skip or offset it. That
is not a cheap change, and the sampled path is a labelled diagnostic that
failed timestep, source-resolution and zero-chirp convergence in the October
2026 benchmarks. Use the exact path with ``INERTIAL_PREHISTORY`` for physical
claims; use the check above to decide whether a sampled ``COLD_START`` run is
admissible at all.
