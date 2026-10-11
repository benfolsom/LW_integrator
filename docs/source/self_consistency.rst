.. _self_consistency:

Self-consistency and chrono settings
======================================

The maintained solver keeps self-consistency enabled with two fixed-geometry
force trials by default. Spatial mechanical momentum determines on-shell gamma
and beta in every radiation mode. The stopping criterion checks the mass shell
from the second trial onward; it does not compare successive forces.
A small mass-shell residual therefore does not certify trajectory accuracy or
energy conservation.

Defaults and configuration
----------------------------

API, CLI, GUI/testbed, and sweep entry points use these defaults when keys are
missing. This is a settings fragment, not a complete run configuration:

.. code-block:: json

   {
     "self_consistency_enabled": true,
     "self_consistency_convergence_mode": "fixed_geometry",
     "self_consistency_target_ms_tolerance": 1e-6,
     "self_consistency_max_iterations": 2,
     "self_consistency_mass_shell_tolerance": 0.01,
     "self_consistency_verbosity": 0,
     "self_consistency_gamma_reconciliation_method": "DISABLED",
     "chrono_interpolate": false,
     "chrono_tolerance": 0.001,
     "chrono_high_precision": false,
     "chrono_adaptive_tolerance": false,
     "radiation_reaction_mode": "medina_lad"
   }

The Python ``SelfConsistencyConfig`` uses the corresponding unprefixed field
names. Radiation reaction is a separate runner setting. The maintained
retardation-mode default is ``FAST``; the direct CLI uses ``chrono_mode`` and
GUI/testbed settings use ``chrono_matching_mode``.

``fixed_geometry`` retains geometry between trials. ``variable_geometry``
recomputes it, but is unsupported for exact inertial endpoint reconstruction.
Historical ``mass_shell_only`` and ``full_iteration`` names normalize to those
two modes. Unknown modes and fewer than two maximum iterations with
self-consistency enabled raise during configuration.

If the mass-shell check fails through the iteration limit, the step raises
``SelfConsistencyNonConvergenceError``. A pseudo-grid active solve may instead
record a lost particle within its numerical-failure budget. Nonconvergence is
not silently accepted. With the maintained tolerances, the later
``mass_shell_tolerance`` fallback cannot rescue a step that already exhausted
the loop.

``mass_shell_relaxation`` is retained for compatibility but unused on the
corrected path; omit it from new settings. Gamma reconciliation is a diagnostic
option that can reseed an iteration on supported paths. It is not an energy
conservation correction and should remain ``DISABLED``. The runner warns when
it cannot act, including the exact RFS/dipole route, Medina/LAD, or disabled
self-consistency.

What to validate
------------------

Keep the two-trial default unless an independent comparison validates another
choice. Disabling self-consistency changes velocity seeding and some blowup
checks. Increasing the maximum to four does not necessarily perform four
trials: the mass-shell criterion can stop at two without measuring force
convergence.

Compare trajectories under timestep refinement, check finite states and particle
losses, and use population-weighted conservation ledgers. Separate mechanical
work, radiation, field energy, boundary heat, and external work as applicable.
The ``mass_shell_projection_energy`` diagnostic is a temporal-predictor
residual, not radiation-reaction work; see :doc:`same_bunch_energy`.

Chrono matching is separate
-----------------------------

Chrono matching selects source samples on the non-exact retarded route.
``FAST`` uses the delay :math:`R/[c(1-\boldsymbol\beta\cdot\hat{\mathbf n})]`,
where the unit direction points from source to observer. ``AVERAGED`` is a
diagnostic alternative. ``APPROXIMATE_BACK_HISTORY`` uses its analytic
back-history model instead of sampled chrono matching.

``chrono_interpolate`` is off by default. When enabled, the maintained
structure-of-arrays path uses linear interpolation with
``chrono_tolerance`` in nanoseconds. ``chrono_adaptive_tolerance`` instead
scales that tolerance with the timestep and is also off by default.
``chrono_high_precision`` is deprecated and has no effect in the maintained
runner; setting it emits ``ChronoHighPrecisionDeprecationWarning``. The old
non-array fallback is not the maintained run path.

The exact route solves light cones in explicit source history and does not use
these sampled chrono options. The runner warns for ignored exact-path options
and for chrono sub-options set without interpolation. See :doc:`exact_path`.
The sampled retarded route remains a diagnostic: timestep, source-resolution,
and zero-chirp controls have failed in bounded moderate-speed studies. Enabling
interpolation alone does not validate it for physics claims.

Migration from older configurations
-------------------------------------

Older entry points supplied different missing-key defaults. In particular,
a CLI configuration without self-consistency keys now enables the two-trial
path. Explicit saved values remain explicit; inspect old settings before
comparing results. The switches require JSON booleans, not strings such as
``"false"``. Unknown gamma-reconciliation names raise.

Canonical chrono keys take precedence over legacy
``self_consistency_chrono_*`` aliases. The direct CLI also accepts
``chrono_matching_mode`` and ``self_consistency_chrono_matching_mode`` as mode
aliases, but rejects conflicting values. The GUI retains a loaded non-FAST
mode through runs and saves. See ``docs/self_consistency_option_audit.md`` for
the earlier option audit, and :doc:`validation` for maintained checks.
