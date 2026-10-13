Radiation Reaction Plan
=======================

.. note::

   This page retains the model derivation and remaining validation plan.
   Medina/LAD and smooth external providers are implemented. Higher-level
   ``SimulationOptions`` and ``IntegratorConfig`` default to ``medina_lad``;
   low-level ``retarded_integrator`` calls default to ``off``. Implementation
   does not close the outstanding field-energy or boundary accounting.

This plan tracks the remaining work needed to make radiation reaction in the
integrator physically auditable. The current code should be treated as having
three separate concepts:

* passive Lienard radiated-power diagnostics,
* an optional power-matched damping approximation, and
* explicit candidate self-force models that act on mechanical momentum before
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
     {d \over dt}\left(\gamma F_{ext}\right)
     - {\gamma^3 \over c^2}(F_{ext}\cdot a)v
   \right].

The first term is the complete derivative

.. math::

   {d \over dt}\left(\gamma F_{ext}\right)
   =\gamma {dF_{ext}\over dt}+{d\gamma\over dt}F_{ext}.

Dropping :math:`\gamma\,dF_{ext}/dt` is not the Medina equation and severely
underestimates reaction during a Coulomb flyby.  The maintained implementation
uses accepted force samples on sampled-force routes. Smooth external providers
supply analytic field derivatives for the complete force derivative; see
:doc:`external_fields`.  The
model is applied to mechanical momentum rather than being added directly to
the canonical accumulator:

* ``F_ext`` is a mechanical external force, not the canonical
  ``delta_Px/Py/Pz/Pt`` accumulator.
* ``v`` and ``a`` are coordinate-time mechanical quantities.
* the integrator evolves proper-time steps and canonical momentum, so the model
  needs a conversion layer: compute the mechanical impulse, update mechanical
  momentum, then recompose canonical momentum from the potentials.
* the paper notes a dressed mass caveat. The first implementation should use
  rest mass only, document that choice, and leave dressed-mass handling behind
  an explicit option.

Medina Impulse Guard Modes
--------------------------

The existing config key ``radiation_reaction_mode`` selects both modes. The
default, ``medina_lad``, retains the legacy cap at 25% of the non-RR external
impulse. This preserves existing results, but can suppress physical radiation
loss when the lab-frame RR force dominates the Lorentz force.

Opt into ``medina_lad_validity`` in a run or sweep config, the GUI mode selector,
or with ``--radiation-reaction-mode medina_lad_validity``. This mode keeps the
complete reduced-order impulse. It screens the instantaneous rest-frame
effective field relative to the classical field scale: with native Gaussian
units, ``r_q = q^2/(m c^2)``, the criterion is
``r_q |F_ext_rest|/(m c^2) <= 0.01``. It also requires
``|F_RR_rest| <= 0.01 |F_ext_rest|``, including the sampled force derivative.
Mechanical three-forces transform with unchanged parallel components and
transverse components multiplied by gamma. Thus, a large lab-frame RR/Lorentz
ratio alone does not activate the guard. The effective field is inferred from
the total non-RR mechanical force, using physical particle charge and rest mass.

The step criterion is ``|delta_p_RR| <= 0.01 max(|p_start|, m c)``; the rest
momentum scale covers particles initially at rest. Any failed criterion sets
the existing ``medina_impulse_capped`` flag, which now means that either guard
acted. Adaptive exact-pair and exact-bunch trials reject flagged states and
shrink without publishing trial history. At the minimum step, an unresolved
failure ends the run. Other stepping paths apply the full impulse and flag the
step for review; they never clip in this mode. Refining a step cannot cure a
physical validity failure. Accepted force-history priming remains unchanged.

These 1% thresholds are conservative engineering screens motivated by the
small-force and slow-variation assumptions of
`Medina's reduced-order approximation <https://arxiv.org/abs/physics/0508031>`_.
They do not certify quantum validity, resolve field variation absent from the
accepted force samples, or test an independent rest-frame magnetic field.
Radiated energy and signed reaction work continue to describe the full force
in the opt-in mode, including flagged fixed-step states.

Medina Native-Units Derivation Draft
------------------------------------

A standalone TeX supplement with the same derivation is available as
:download:`Radiation Reaction Model Supplement <medina_native_units_analysis.tex>`.


The maintained solver uses the historical native system:

* length: ``mm``,
* time: ``ns``,
* velocity: ``mm / ns``,
* mass: ``amu``,
* mechanical momentum: ``amu mm / ns``,
* mechanical force: ``amu mm / ns^2``, and
* mechanical energy: ``amu mm^2 / ns^2``.

Charge uses the derived solver-native value
``ELEMENTARY_CHARGE = ELEMENTARY_CHARGE_STATC * STATCOULOMB_TO_NATIVE_CHARGE``
(approximately ``1.1787088598e-5``). A native electric field is a
force-per-charge quantity, so ``q E`` has units of ``amu mm / ns^2``. The
current prescribed-field implementation converts SI ``V/m`` values by
matching the SI force on one elementary charge to one native force unit.

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
     {d \over dt}\left(\gamma F_{ext}\right)
     - {\gamma^3 \over c^2}(F_{ext}\cdot a)v
   \right],

where

.. math::

   \tau_0 = {2 \over 3}{q^2 \over m c^3}.

This is the same characteristic time already stored on particle states as
``char_time`` when native charge and rest mass are used. Here
``v = c beta`` and ``a = dv/dt = c d beta/dt`` are coordinate-time mechanical
quantities. Both bracketed terms have units of force per time:

* ``d(gamma F_ext)/dt`` gives ``amu mm / ns^3``;
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

   Remaining coverage gap:

   * timestep convergence of active ``medina_lad`` energy/momentum changes,
     separately from passive Lienard power integration.

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

5. Derive the Medina implementation in native units. **Status: corrected first
   implementation complete; needs continued review.**

   Produce a short derivation note before coding. It should define native units
   for charge, mass, force, acceleration, time, and energy; state whether the
   force is integrated over coordinate time or proper time; and include a
   dimension check for the final impulse.

6. Implement the Medina candidate behind an explicit mode. **Status: complete
   for the charge-only candidate path.**

   The first code path should be opt-in, probably
   ``radiation_reaction_mode="medina_lad"``. It should:

   * compute the external mechanical force represented by the current step,
   * compute the complete ``d(gamma F_ext)/dt`` from coordinate-time
     quantities,
   * compute coordinate-time acceleration from beta-dot,
   * apply the radiation-reaction impulse to mechanical momentum,
   * cap the impulse only as a numerical guard with diagnostics, and
   * recompose canonical momentum using the current potentials.

   Current implementation:

   * ``radiation_reaction_mode="medina_lad"`` is accepted as an explicit mode,
   * the non-radiation mechanical force is estimated from the mass-shell
     projected mechanical momentum increment over coordinate time,
   * accepted interval-midpoint force samples provide a first-order backward
     estimate of the complete ``d(gamma F_ext)/dt``; an unprimed first sample
     records far radiation but applies no incomplete reaction impulse,
   * ``d beta/dt`` is derived from the mechanical force estimate, keeping the
     recoil term consistent with the accepted non-radiation update,
   * the impulse is applied to mechanical momentum and canonical momentum is
     recomposed with the existing vector/scalar potential terms, and
   * the impulse cap, derivative readiness, signed reaction work, far-radiated
     energy, and cross-field energy are exposed as trajectory diagnostics.

   Remaining work before treating this as physics evidence:

   * extend the magnetic-bend refinement control in
     ``tests/unit/test_external_fields.py`` from ``diagnostic_only`` radiated
     energy to active ``medina_lad`` trajectories, including prescribed-field,
     retarded-source, and boundary encounters,
   * evaluate whether the mechanical-force estimator is adequate for retarded
     image/source forces or whether those contributions need a more explicit
     force-decomposition API.

7. Add prescribed external-field support for controlled benchmarks. **Status:
   implemented for uniform fields and smooth Python providers.**

   ``ExternalFieldConfig`` accepts uniform native fields, SI conversion helpers,
   and simple windows, or a pure provider returning potentials, fields, and
   analytic spacetime derivatives. The provider implements its own smooth
   entrances and exits and cannot be combined with hard config windows.
   ``tests/unit/test_external_fields.py`` covers provider parity and derivatives.
   See :doc:`external_fields` for a construction example and route guards.

   These fields drive mechanical momentum. Provider potentials are diagnostic;
   they do not add an external canonical gauge offset. Arbitrary providers have
   no JSON factory, and checkpoint fingerprinting requires a serializable
   dataclass callable. A general external canonical-potential coupling remains
   separate work.

8. Benchmark self-force candidates against controlled systems. **Status: in
   progress.**

   Start with tests that do not require wall images or retarded multi-particle
   edge cases:

   * prescribed straight-line acceleration,
   * uniform circular motion in an imposed magnetic field,
   * ultra-relativistic transverse acceleration,
   * a low-energy case where reaction is negligible, and
   * a high-gamma case where the reaction term is visible but stable.

   Current coverage:

   * direct Medina helper test for longitudinal cancellation,
   * direct Medina helper test for transverse/synchrotron-style damping,
   * prescribed longitudinal electric acceleration with nonzero
     ``dgamma/dt``, verifying diagnostic-only neutrality and small Medina
     longitudinal applied energy,
   * short high-gamma prescribed magnetic bend, and
   * several-hundred-mm prescribed transverse magnetic bend.

   Remaining coverage gap:

   * active-mode trajectory and reaction-work timestep convergence for
     prescribed fields, retarded sources, and boundaries; the existing
     magnetic-bend refinement test measures ``diagnostic_only`` radiation,
   * mixed longitudinal/transverse prescribed fields, and
   * low-energy negligible-reaction and ultra-relativistic visible-reaction
     brackets with explicit acceptance thresholds.

9. Revisit conducting-surface collision cases. **Status: not started.**

   The paper motivates radiation reaction mostly as a way to prevent runaway
   behavior near image-charge collisions. After isolated benchmarks pass,
   compare ``off``, ``power_matched_damping``, and ``medina_lad`` on the
   original aperture/image-charge scenarios. The acceptance criterion should be
   energy bookkeeping and timestep convergence, not just smoother trajectories.

10. Evaluate alternatives only after the Medina track is measurable. **Status:
    not started.**

   Alternative reduced-order formalisms should remain literature cross-checks,
   not the production target. In particular, Landau-Lifshitz is less aligned
   with the maintained solver because it is normally written in terms of local
   field-solution quantities, while this code's primary representation is
   covariant Lienard-Wiechert potentials and canonical momentum. Keep any LL or
   Eliezer-Ford-O'Connell work behind explicit comparison modes and use them
   only if they clarify Medina stability, convergence, or near-surface behavior.

Next Best Steps
---------------

The immediate low-risk work is now validation rather than another force model:

* keep the corrected complete force derivative and native-unit mapping under
  review,
* extend the existing diagnostic-only magnetic-bend timestep control to active
  Medina trajectory and reaction-work checks before generalizing to coupled
  retarded sources and boundary encounters,
* add mixed-field and low/high-energy benchmark brackets with explicit
  tolerances, and
* refine the mechanical-force extraction API before using ``medina_lad`` as
  evidence in conducting-wall collision cases.
