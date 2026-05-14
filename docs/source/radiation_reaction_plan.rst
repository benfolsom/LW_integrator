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

Medina Native-Units Derivation Draft
------------------------------------

The maintained solver uses the historical native system:

* length: ``mm``,
* time: ``ns``,
* velocity: ``mm / ns``,
* mass: ``amu``,
* mechanical momentum: ``amu mm / ns``,
* mechanical force: ``amu mm / ns^2``, and
* mechanical energy: ``amu mm^2 / ns^2``.

Charge uses the solver-native value ``ELEMENTARY_CHARGE = 1.178734e-5``. A
native electric field is a force-per-charge quantity, so ``q E`` has units of
``amu mm / ns^2``. The current prescribed-field implementation converts SI
``V/m`` values by matching the SI force on one elementary charge to one native
force unit.

The integrator step variable ``h`` is a proper-time increment ``d tau``. The
coordinate-time increment for a particle is therefore

.. math::

   dt = \gamma h.

The mechanical momentum is

.. math::

   p = \gamma m v = \gamma m c \beta,

with units ``amu mm / ns``. The current prescribed external-field impulse is

.. math::

   \Delta p_{ext} = h q \gamma (E + \beta \times B)
                  = dt\, F_{ext},

where

.. math::

   F_{ext} = q (E + \beta \times B).

For a general Medina implementation, the first required helper should expose
the mechanical non-radiation force represented by a step:

.. math::

   F_{ext} \approx {\Delta p_{mech, nonrad} \over dt}.

For prescribed fields this can be computed directly from ``q(E + beta x B)``.
For retarded image/source forces it must be derived from the mechanical
momentum increment after subtracting the vector-potential contribution, not
from the canonical ``delta_P`` accumulator alone.

The Medina force can be written in native variables as

.. math::

   F^{RAD} = \tau_0
   \left[
     {d\gamma \over dt}F_{ext}
     - {\gamma^3 \over c^2}(F_{ext}\cdot a)v
   \right],

where

.. math::

   \tau_0 = {2 \over 3}{q^2 \over m c^3}.

This is the same characteristic time already stored on particle states as
``char_time`` when native charge and rest mass are used. Here
``v = c beta`` and ``a = dv/dt = c d beta/dt`` are coordinate-time mechanical
quantities. Both bracketed terms have units of force per time:

* ``(d gamma / dt) F_ext`` gives ``amu mm / ns^3``;
* ``(gamma^3 / c^2)(F_ext dot a)v`` also gives ``amu mm / ns^3``.

Multiplying by ``tau_0`` gives a mechanical force in ``amu mm / ns^2``. The
mechanical radiation-reaction impulse for one integrator step is therefore

.. math::

   \Delta p_{RAD} = F^{RAD} dt = F^{RAD} \gamma h.

The candidate implementation should apply this impulse to mechanical momentum,
then recompose canonical momentum from the same potentials used by the normal
step:

.. math::

   p_{new} = p_{nonrad} + \Delta p_{RAD},

.. math::

   \gamma_{new} = \sqrt{1 + {|p_{new}|^2 \over (m c)^2}},

.. math::

   P_{i,new} = p_{i,new} + m A_{i,solver},

.. math::

   P_{t,new} = \gamma_{new} m c + q \Phi.

The first Medina mode should use rest mass in ``tau_0`` and explicitly state
that the paper's dressed-mass caveat is not yet modeled. It should also start
with prescribed-field benchmarks before being applied to retarded image-charge
collision cases.

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

5. Derive the Medina implementation in native units. **Status: first draft
   complete; needs review before coding.**

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

* review the Medina native-units derivation above,
* define the mechanical-force extraction API that Medina/LAD will consume,
* add a prescribed-field-only Medina prototype once that API is explicit, and
* only then add ``radiation_reaction_mode="medina_lad"`` behind an explicit
  opt-in.
