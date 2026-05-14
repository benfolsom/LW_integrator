Radiation Reaction Plan
=======================

.. note::

   This is a development-branch planning document. It can be committed on the
   ``development`` branch, but should not be promoted to ``main`` or release
   documentation until the implementation and validation tasks below are
   complete.

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

1. Finish terminology cleanup. **Status: complete.**

   Rename diagnostics that historically referred to "radiation reaction
   activation" when they only detect large changes in stored acceleration.
   Keep compatibility aliases until downstream scripts are updated.

   Current implementation:

   * ``find_large_bdot_changes`` is the maintained diagnostic name.
   * ``find_radiation_reaction_activations`` remains as a compatibility alias.
   * legacy ``bdot`` radiation-reaction mode names are rejected.

2. Lock down passive radiation bookkeeping. **Status: in progress.**

   Add regression tests for ``radiation_power``, ``radiation_energy``, and
   ``radiation_energy_applied`` in ``off`` and ``power_matched_damping`` modes.
   Verify that diagnostic-only radiation never changes trajectory state. These
   tests should cover both legacy list trajectories and ``TrajectoryArrays`` so
   the bookkeeping remains compatible with the current SoA optimization path.

   Current coverage:

   * ``off`` and ``diagnostic_only`` produce identical trajectory state.
   * ``power_matched_damping`` records applied radiated energy separately.
   * startup and SoA paths zero-fill radiation bookkeeping fields.

3. Validate the Lienard power helper in isolation. **Status: in progress.**

   Required checks:

   * zero power for unaccelerated motion,
   * parallel and transverse acceleration terms,
   * circular-motion/synchrotron scaling,
   * coordinate-time ``bdot`` conversion from stored integrator values, and
   * timestep convergence of integrated radiated energy.

   Current coverage:

   * zero acceleration,
   * coordinate-time ``d beta / dt`` input,
   * parallel and transverse acceleration scaling,
   * synchrotron-like gamma scaling for transverse acceleration, and
   * prescribed magnetic-bend timestep convergence.

4. Make the provisional damping model deliberately narrow. **Status: in progress.**

   The existing ``power_matched_damping`` mode should remain labeled as an
   energy-bookkeeping approximation. It should have tests showing that it
   removes at most the requested radiated energy, never crosses rest energy, and
   preserves momentum direction by construction. It should not be presented as
   a LAD, Landau-Lifshitz, or Medina self-force model.

   Current coverage:

   * isolated damping helper preserves momentum direction by scaling mechanical
     momentum magnitude,
   * the helper cannot remove more than kinetic energy above rest energy, and
   * the magnetic-bend integration test checks that applied damping energy
     matches the observed gamma reduction.

5. Derive the Medina implementation in native units. **Status: not started.**

   Produce a short derivation note before coding. It should define native units
   for charge, mass, force, acceleration, time, and energy; state whether the
   force is integrated over coordinate time or proper time; and include a
   dimension check for the final impulse.

6. Implement the Medina candidate behind an explicit mode. **Status: blocked by
   Task 5.**

   The first code path should be opt-in, probably
   ``radiation_reaction_mode="medina_lad"``. It should:

   * compute the external mechanical force represented by the current step,
   * compute ``dgamma/dt`` from coordinate-time quantities,
   * compute coordinate-time acceleration from beta-dot,
   * apply the radiation-reaction impulse to mechanical momentum,
   * cap the impulse only as a numerical guard with diagnostics, and
   * recompose canonical momentum using the current potentials.

7. Add prescribed external-field support for controlled benchmarks. **Status:
   complete for uniform fields.**

   Radiation-reaction validation needs clean external accelerators that do not
   depend on image-charge or point-source artifacts. Start with uniform fields
   in native solver units, including the static longitudinal ``E_z`` case
   discussed around Eq. 21 of the foundational paper. Required guardrails:

   * provide explicit SI-to-native conversion helpers for electric fields,
   * keep field configs compatible with the canonical SoA/Numba integrator path,
   * support simple spatial and temporal windows, and
   * document that the first implementation is a mechanical Lorentz-force
     impulse, not yet a full external potential/map system.

   Later extensions can add field maps, callable fields, and explicit
   vector/scalar-potential providers for fully covariant canonical updates.

   Time-dependent fields need their own model boundary. The static Eq. 21
   longitudinal field used in the paper can be written as
   ``E_z = partial^z A^0 - partial^0 A^z`` with the time derivative set to
   zero. For any time-dependent extension, do not implement this as a scalar
   field amplitude toggle alone. Add a potential-provider interface that can
   evaluate ``A^0``, ``A^i``, and their derivatives at ``(x, y, z, t)`` so the
   ``-partial^0 A^i`` contribution is represented explicitly and canonical
   momentum can be recomposed from the same potentials used to compute the
   mechanical Lorentz-force impulse.

   Current implementation:

   * uniform electric and magnetic fields in native solver units,
   * SI-to-native conversion helper for electric fields,
   * spatial and temporal field windows,
   * canonical integrator, SoA/Numba, CLI, GUI, and config plumbing, and
   * explicit documentation that this is not yet a full external-potential or
     time-dependent field-map system.

8. Benchmark self-force candidates against controlled systems. **Status: not
   started.**

   Start with tests that do not require wall images or retarded multi-particle
   edge cases:

   * prescribed straight-line acceleration,
   * uniform circular motion in an imposed magnetic field,
   * ultra-relativistic transverse acceleration,
   * a low-energy case where reaction is negligible, and
   * a high-gamma case where the reaction term is visible but stable.

9. Revisit conducting-surface collision cases. **Status: not started.**

   The paper motivates radiation reaction mostly as a way to prevent runaway
   behavior near image-charge collisions. After isolated benchmarks pass,
   compare ``off``, ``power_matched_damping``, and ``medina_lad`` on the
   original aperture/image-charge scenarios. The acceptance criterion should be
   energy bookkeeping and timestep convergence, not just smoother trajectories.

10. Evaluate alternatives only after the Medina track is measurable. **Status:
    not started.**

   Landau-Lifshitz reduced-order radiation reaction is likely the best
   conventional comparison model. Eliezer-Ford-O'Connell can remain a research
   track unless the Medina candidate shows instability or poor convergence in
   the near-surface regime.

Next Best Steps
---------------

The immediate low-risk work is now validation rather than new physics:

* add one more integration-level damping bound check:
  ``sum(radiation_energy_applied) <= sum(radiation_energy)`` for a controlled
  prescribed-field run,
* add a native-units derivation note for the Medina candidate,
* define the mechanical-force extraction API that Medina/LAD will consume, and
* only then add ``radiation_reaction_mode="medina_lad"`` behind an explicit
  opt-in.
