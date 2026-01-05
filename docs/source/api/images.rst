Image charge generation
=======================

The ``core.images`` module provides utilities for generating mirror-image
charges representing conducting-wall boundaries in electromagnetic simulations.
The primary function ``generate_conducting_image`` creates virtual particle
bunches that reproduce the boundary conditions of a conducting surface with a
circular aperture.

As of January 2025, the module includes macroparticle simulation support,
allowing stochastic position and momentum spread errors to be applied to image
subcharges for realistic beam emittance modeling.

Module reference
----------------

.. automodule:: core.images
   :members:
   :undoc-members:
   :show-inheritance:

Macroparticle simulation
-------------------------

The ``generate_conducting_image`` function accepts optional macroparticle
parameters that enable realistic modeling of beam collective effects:

**Charge scaling**
   The ``macroparticle_charge_multiplier`` parameter scales both the test
   particle charge and all image subcharge magnitudes by a constant factor.
   This is used to represent macroparticles (charge bundles) rather than
   individual particles. Default: 1.0 (no scaling).

**Position spread**
   The ``macroparticle_position_spread`` parameter (σ_x in mm) applies
   independent Gaussian errors to the x and y positions of each image
   subcharge. These errors are drawn from N(0, σ_x²) and represent the
   transverse spatial extent of the beam. Default: 0.0 (no spread).

**Momentum spread**
   The ``macroparticle_momentum_spread`` parameter (σ_p) creates cumulative
   transverse displacement that grows with each integration timestep. At
   step number *n*, the total spread is:

   .. math::

      \sigma_{\text{total}}(n) = \sqrt{\sigma_x^2 + \left(\frac{\sigma_p}{m} \times h \times n\right)^2}

   where *h* is the timestep size and *m* is the particle mass. This models
   the divergence of particles with different transverse momenta as they
   propagate. Default: 0.0 (no momentum spread).

**Error application order**
   Position and momentum errors are applied to subcharge coordinates **before**
   the radial weighting and charge attenuation calculations. This ensures that
   the geometric suppression of charges near the aperture edge accounts for the
   stochastic displacement of the beam distribution.

Usage example
-------------

Basic conducting-wall image generation:

.. code-block:: python

   from core.images import generate_conducting_image
   from core.types import ParticleState

   # Generate image charges for a particle near a conducting wall
   image_state = generate_conducting_image(
       vector=rider_state,
       wall_z=100.0,              # Wall at z = 100 mm
       aperture_radius=0.001,     # 1 mm aperture
       subcharge_count=12,        # 12 subcharges around aperture rim
       use_weighting=True,        # Apply radial attenuation
   )

Macroparticle mode with beam emittance:

.. code-block:: python

   # Simulate 10× charge scaling with position and momentum spread
   image_state = generate_conducting_image(
       vector=rider_state,
       wall_z=100.0,
       aperture_radius=0.001,
       subcharge_count=12,
       use_weighting=True,
       macroparticle_charge_multiplier=10.0,     # 10× charge
       macroparticle_position_spread=1e-5,       # 10 μm position σ
       macroparticle_momentum_spread=1e-6,       # Momentum spread
       timestep=3e-7,                            # Integration timestep (ns)
       step_number=500,                          # Current step number
   )

The ``timestep`` and ``step_number`` parameters are required when
``macroparticle_momentum_spread > 0`` to calculate the cumulative momentum-
driven displacement.

Physical interpretation
------------------------

The macroparticle simulation parameters correspond to physical beam properties:

**Charge multiplier**
   Represents the effective number of real particles per simulation macroparticle.
   For example, a value of 100 means each simulated particle represents 100
   actual particles with the total charge scaled accordingly.

**Position spread**
   Models the RMS beam size (geometric emittance) in the transverse plane. For
   a Gaussian beam with RMS size σ_x, this parameter should be set to σ_x.

**Momentum spread**
   Models the angular divergence of the beam. For a beam with RMS divergence
   σ_x' (radians), the momentum spread is approximately σ_p ≈ p × σ_x' where
   p is the characteristic momentum scale. The cumulative growth with timesteps
   reproduces the natural divergence of particles with different angles as they
   propagate.

The combined effect of position and momentum spread reproduces the normalized
emittance evolution:

.. math::

   \epsilon_n = \beta \gamma \sqrt{\langle x^2 \rangle \langle x'^2 \rangle - \langle x x' \rangle^2}

where β and γ are the relativistic velocity and Lorentz factor.

See also
--------

* :doc:`../recent_changes` for implementation details and recent enhancements
* :doc:`../theory` for the theoretical foundation of image charge methods
* :doc:`core` for the main integrator functions that consume image states
