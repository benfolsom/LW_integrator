Radiation Reaction Plan
=======================

This plan tracks the remaining work needed to make radiation reaction in the
integrator physically auditable. The current code should be treated as having
three separate concepts:

* passive Lienard radiated-power diagnostics,
* an optional power-matched damping approximation, and
* future candidate self-force models that act on mechanical momentum before
  recomposing canonical momentum.

The legacy ``bdot`` correction has been removed. It was a post-step edit to a
stored acceleration field rather than a real force contribution, and it was not
a reliable radiation-reaction implementation.

Reference Model From The Paper
------------------------------

The foundational paper discusses a Medina-style radiation-reaction force based
on the Lorentz-Abraham-Dirac equation:

.. math::

   F^{RAD} =
   {2 \over 3}{e^2 \over mc^3}
   \left[
     {d\gamma \over dt}F_{ext}
     - {\gamma^3 \over c^2}(F_{ext}\cdot a)v
   \right].

This remains a reasonable candidate model, but it should not be added directly
to the existing canonical momentum accumulator until the variable mapping is
explicit:

* ``F_ext`` is a mechanical external force, not the canonical
  ``delta_Px/Py/Pz/Pt`` accumulator.
* ``v`` and ``a`` are coordinate-time mechanical quantities.
* the integrator evolves proper-time steps and canonical momentum, so the model
  needs a conversion layer: compute the mechanical impulse, update mechanical
  momentum, then recompose canonical momentum from the potentials.
* the paper notes a dressed mass caveat. The first implementation should use
  rest mass only, document that choice, and leave dressed-mass handling behind
  an explicit option.

Ordered Tasks
-------------

1. Finish terminology cleanup.

   Rename diagnostics that historically referred to "radiation reaction
   activation" when they only detect large changes in stored acceleration.
   Keep compatibility aliases until downstream scripts are updated.

2. Lock down passive radiation bookkeeping.

   Add regression tests for ``radiation_power``, ``radiation_energy``, and
   ``radiation_energy_applied`` in ``off`` and ``power_matched_damping`` modes.
   Verify that diagnostic-only radiation never changes trajectory state.

3. Validate the Lienard power helper in isolation.

   Required checks:

   * zero power for unaccelerated motion,
   * parallel and transverse acceleration terms,
   * circular-motion/synchrotron scaling,
   * coordinate-time ``bdot`` conversion from stored integrator values, and
   * timestep convergence of integrated radiated energy.

4. Make the provisional damping model deliberately narrow.

   The existing ``power_matched_damping`` mode should remain labeled as an
   energy-bookkeeping approximation. It should have tests showing that it
   removes at most the requested radiated energy, never crosses rest energy, and
   preserves momentum direction by construction. It should not be presented as
   a LAD, Landau-Lifshitz, or Medina self-force model.

5. Derive the Medina implementation in native units.

   Produce a short derivation note before coding. It should define native units
   for charge, mass, force, acceleration, time, and energy; state whether the
   force is integrated over coordinate time or proper time; and include a
   dimension check for the final impulse.

6. Implement the Medina candidate behind an explicit mode.

   The first code path should be opt-in, probably
   ``radiation_reaction_mode="medina_lad"``. It should:

   * compute the external mechanical force represented by the current step,
   * compute ``dgamma/dt`` from coordinate-time quantities,
   * compute coordinate-time acceleration from beta-dot,
   * apply the radiation-reaction impulse to mechanical momentum,
   * cap the impulse only as a numerical guard with diagnostics, and
   * recompose canonical momentum using the current potentials.

7. Benchmark self-force candidates against controlled systems.

   Start with tests that do not require wall images or retarded multi-particle
   edge cases:

   * prescribed straight-line acceleration,
   * uniform circular motion in an imposed magnetic field,
   * ultra-relativistic transverse acceleration,
   * a low-energy case where reaction is negligible, and
   * a high-gamma case where the reaction term is visible but stable.

8. Revisit conducting-surface collision cases.

   The paper motivates radiation reaction mostly as a way to prevent runaway
   behavior near image-charge collisions. After isolated benchmarks pass,
   compare ``off``, ``power_matched_damping``, and ``medina_lad`` on the
   original aperture/image-charge scenarios. The acceptance criterion should be
   energy bookkeeping and timestep convergence, not just smoother trajectories.

9. Evaluate alternatives only after the Medina track is measurable.

   Landau-Lifshitz reduced-order radiation reaction is likely the best
   conventional comparison model. Eliezer-Ford-O'Connell can remain a research
   track unless the Medina candidate shows instability or poor convergence in
   the near-surface regime.

Low-Hanging Work
----------------

The immediate low-risk work is documentation and naming:

* add this roadmap to the docs,
* rename the old activation diagnostic to ``find_large_bdot_changes``,
* retain ``find_radiation_reaction_activations`` as a compatibility alias, and
* update tests so old scripts continue to work while new code uses the clearer
  name.

