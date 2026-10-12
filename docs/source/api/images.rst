Image charge generation
=======================

``core.images.generate_conducting_image`` constructs the maintained wall-image
approximation: mirrored subcharges with geometric suppression, radial weighting,
and optional stochastic transverse displacements. It does not solve the full
conducting circular-aperture boundary-value problem. See :doc:`../boundary`
for the separate experimental boundary solver.

Charge and spread parameters
----------------------------

``macroparticle_charge_multiplier`` multiplies generated image charges only;
it does not mutate the input particle's observer charge. In the representative
particle initializer, species charge $q_s$, species mass $m_s$, and population
$N$ remain separate: source charge is $Nq_s$, observer charge is $q_s$,
inertia is $m_s$, and diagnostic weight is $N$. The wall-image multiplier is a
legacy source option, distinct from :doc:`../exact_macroparticle_sources`.

``bunch_transv_dist`` is a transverse half-width in mm, used here as the base
Gaussian position spread. ``bunch_transv_mom`` is a momentum spread in
amu mm/ns. ``macroparticle_sigma_multiplier`` scales both spreads (default
1.0). With ``macroparticle_use_momentum_errors=True`` (the default), the
image displacement at step $n$ uses
$\sigma_{\rm total}=\sqrt{\sigma_x^2+(\sigma_p h n/m)^2}$, where $h$ is
proper time in ns. Both bunch spreads default to zero.

The displacements are drawn independently in x and y before radial weighting;
the initial solid-angle suppression is computed earlier. This prescription
does not evolve the full position–angle covariance needed to establish
normalized emittance evolution. Such a claim requires a separate beam
phase-space comparison.

Usage example
-------------

Given an initialized ``rider_state`` in native units:

.. code-block:: python

   from core.images import generate_conducting_image

   image_state = generate_conducting_image(
       vector=rider_state,
       wall_z=100.0,                         # mm
       aperture_radius=0.001,                # 1 micrometre
       subcharge_count=12,
       use_weighting=True,
       macroparticle_charge_multiplier=10.0,
       macroparticle_sigma_multiplier=1.0,
       macroparticle_use_momentum_errors=True,
       bunch_transv_dist=1e-5,                # 0.01 micrometre
       bunch_transv_mom=1e-6,                 # amu mm/ns
       timestep=3e-7,                        # proper ns; illustrative only
       step_number=500,
   )

The result is a new ``ParticleState`` containing image arrays. Supply
``timestep`` and ``step_number`` when ``bunch_transv_mom > 0`` and momentum
errors are enabled; their zero defaults produce no cumulative displacement.
For an unspread image, omit the spread arguments. Timestep and source-model
convergence remain necessary for any physical claim.

Module reference
----------------

The source docstring's claim that the multiplier also scales the test particle
is stale: the implementation only scales the returned image charges. The
reference below exposes signatures without repeating that description.

.. py:currentmodule:: core.images

.. py:function:: generate_conducting_image(vector, wall_z, aperture_radius, subcharge_count=12, *, use_weighting=True, macroparticle_charge_multiplier=1.0, macroparticle_sigma_multiplier=1.0, macroparticle_use_momentum_errors=True, bunch_transv_dist=0.0, bunch_transv_mom=0.0, timestep=0.0, step_number=0)

   Return the image ``ParticleState`` described above. The input state is not
   modified. Distances are mm, masses are amu, and the timestep is proper ns.

.. autofunction:: core.images.generate_switching_image

See also
--------

* :doc:`../theory` for the field and stepping conventions.
* :doc:`core` for integrator APIs consuming image states.
