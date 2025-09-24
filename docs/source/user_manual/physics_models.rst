Physics Models and Mathematical Foundations
==========================================

The LW Integrator implements a covariant electromagnetic particle tracking system based on the Liénard-Wiechert formulation of electromagnetic fields. This document provides detailed mathematical foundations and implementation details.

Covariant Formulation
---------------------

The integrator uses a fully covariant approach to electromagnetic field calculations, ensuring relativistic consistency across all energy regimes.

Four-Vector Formalism
~~~~~~~~~~~~~~~~~~~~~

Particle states are represented using four-vectors in Minkowski spacetime:

* **Position four-vector:** :math:`x^\mu = (ct, \mathbf{x})`
* **Four-velocity:** :math:`u^\mu = \gamma (c, \mathbf{v})`
* **Four-momentum:** :math:`p^\mu = m u^\mu = (\gamma mc, \gamma m\mathbf{v})`

where :math:`\gamma = (1 - v^2/c^2)^{-1/2}` is the Lorentz factor.

Conjugate Momentum Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Implementation Detail:** The integrator uses the **canonical (conjugate) momentum** rather than kinetic momentum:

.. math::
   \mathbf{P} = \gamma m \mathbf{v} + q \mathbf{A}

where:
- :math:`\gamma m \mathbf{v}` is the kinetic momentum
- :math:`q \mathbf{A}` is the electromagnetic field contribution
- :math:`\mathbf{A}` is the vector potential

This formulation is essential for:

1. **Gauge Invariance:** The conjugate momentum automatically incorporates gauge transformations
2. **Hamiltonian Mechanics:** Enables symplectic integration methods
3. **Electromagnetic Consistency:** Properly couples particle motion to field dynamics

**Energy-Momentum Relation:**

.. math::
   E^2 = (pc)^2 + (mc^2)^2

where :math:`p` is the magnitude of the kinetic momentum :math:`\gamma m v`.

Explicit Integration Scheme
---------------------------

The integrator employs an **explicit predictor-corrector scheme** optimized for electromagnetic field evolution.

Integration Algorithm
~~~~~~~~~~~~~~~~~~~~

The explicit integrator advances particle states using:

1. **Field Evaluation:** Calculate electromagnetic fields at current positions
2. **Force Calculation:** Compute Lorentz forces including retardation effects
3. **Momentum Update:** Advance conjugate momentum using force
4. **Position Update:** Advance position using updated velocity
5. **Consistency Check:** Verify four-momentum constraint

**Mathematical Scheme:**

.. math::
   \begin{align}
   \mathbf{F}^n &= q(\mathbf{E}^n + \mathbf{v}^n \times \mathbf{B}^n) \\
   \mathbf{P}^{n+1} &= \mathbf{P}^n + \Delta t \cdot \mathbf{F}^n \\
   \mathbf{v}^{n+1} &= \frac{\mathbf{P}^{n+1} - q\mathbf{A}^{n+1}}{\gamma^{n+1} m} \\
   \mathbf{x}^{n+1} &= \mathbf{x}^n + \Delta t \cdot \mathbf{v}^{n+1}
   \end{align}

**Advantages of Explicit Method:**

* **Computational Efficiency:** No matrix inversions required
* **Parallelization Friendly:** Independent particle updates
* **Memory Efficient:** Minimal storage requirements
* **Stability Control:** Adaptive time stepping prevents instabilities

**Stability Considerations:**

The explicit scheme requires careful time step selection:

.. math::
   \Delta t < \min\left(\frac{1}{\omega_c}, \frac{\lambda}{c}\right)

where:
- :math:`\omega_c = qB/(\gamma m)` is the cyclotron frequency
- :math:`\lambda` is the shortest electromagnetic wavelength in the system

Liénard-Wiechert Electromagnetic Fields
---------------------------------------

The integrator computes electromagnetic fields using the exact Liénard-Wiechert potentials, accounting for finite propagation speed and relativistic effects.

Retarded Potentials
~~~~~~~~~~~~~~~~~~

The scalar and vector potentials at observation point :math:`\mathbf{r}` and time :math:`t` due to a charge at retarded position :math:`\mathbf{r}'(t_{ret})` are:

.. math::
   \begin{align}
   \Phi(\mathbf{r}, t) &= \frac{q}{4\pi\epsilon_0} \frac{1}{(1 - \mathbf{n} \cdot \boldsymbol{\beta})R} \bigg|_{ret} \\
   \mathbf{A}(\mathbf{r}, t) &= \frac{\boldsymbol{\beta}}{c} \Phi(\mathbf{r}, t)
   \end{align}

where:
- :math:`t_{ret} = t - R(t_{ret})/c` is the retarded time
- :math:`\mathbf{R} = \mathbf{r} - \mathbf{r}'(t_{ret})` is the retarded separation vector
- :math:`R = |\mathbf{R}|` is the retarded distance
- :math:`\mathbf{n} = \mathbf{R}/R` is the unit vector from source to observer
- :math:`\boldsymbol{\beta} = \mathbf{v}/c` is the normalized velocity

Electric and Magnetic Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The electric and magnetic fields are derived from the potentials:

.. math::
   \begin{align}
   \mathbf{E} &= \frac{q}{4\pi\epsilon_0} \left[ \frac{\mathbf{n} - \boldsymbol{\beta}}{(1 - \mathbf{n} \cdot \boldsymbol{\beta})^3 R^2} + \frac{\mathbf{n} \times ((\mathbf{n} - \boldsymbol{\beta}) \times \dot{\boldsymbol{\beta}})}{c(1 - \mathbf{n} \cdot \boldsymbol{\beta})^3 R} \right]_{ret} \\
   \mathbf{B} &= \frac{1}{c} \mathbf{n} \times \mathbf{E}
   \end{align}

**Field Components:**

1. **Coulomb Term:** :math:`\propto R^{-2}` - Static-like fields
2. **Radiation Term:** :math:`\propto R^{-1}` - Acceleration-dependent fields

Radiation Reaction
-----------------

The integrator includes self-consistent radiation reaction through the Abraham-Lorentz-Dirac equation.

Abraham-Lorentz-Dirac Force
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The radiation reaction force on a relativistic charged particle is:

.. math::
   f^\mu_{rad} = \frac{q^2}{6\pi\epsilon_0 mc^3} \left( \frac{d^2u^\mu}{d\tau^2} + \frac{1}{c^2} u^\mu u_\nu \frac{d^2u^\nu}{d\tau^2} \right)

where :math:`\tau` is the proper time and :math:`u^\mu` is the four-velocity.

**In the particle rest frame:**

.. math::
   \mathbf{f}_{rad} = \frac{q^2}{6\pi\epsilon_0 c^3} \dot{\mathbf{a}}

where :math:`\mathbf{a}` is the three-acceleration.

Synchrotron Power Loss
~~~~~~~~~~~~~~~~~~~~~

The radiated power (Larmor formula) is:

.. math::
   P = \frac{q^2}{6\pi\epsilon_0 c^3} \gamma^2 \left( \dot{\mathbf{v}}^2 - \frac{1}{c^2}(\mathbf{v} \times \dot{\mathbf{v}})^2 \right)

**For circular motion:**

.. math::
   P = \frac{q^2}{6\pi\epsilon_0 c^3} \frac{\gamma^4 v^4}{R^2}

Numerical Implementation Details
-------------------------------

Adaptive Time Stepping
~~~~~~~~~~~~~~~~~~~~~~

The integrator employs adaptive time stepping based on multiple criteria:

1. **Force Gradient:** :math:`\Delta t \propto |\mathbf{F}|/|\nabla \mathbf{F}|`
2. **Cyclotron Frequency:** :math:`\Delta t < 2\pi/(10\omega_c)`
3. **Energy Conservation:** Reject steps with :math:`|\Delta E/E| > \epsilon_{tol}`
4. **Position Accuracy:** Maintain spatial resolution requirements

Energy Conservation Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrator continuously monitors energy conservation:

.. math::
   E_{total} = \sum_i \sqrt{(p_i c)^2 + (m_i c^2)^2} + \frac{1}{2}\epsilon_0 \int (\mathbf{E}^2 + c^2\mathbf{B}^2) d^3x

**Conservation Tolerance:** Simulations maintain :math:`|\Delta E/E| < 10^{-6}` for stable integration.

Stability Considerations
~~~~~~~~~~~~~~~~~~~~~~~

**Critical Stability Requirements:**

1. **Minimal Transverse Momentum:** Initialize with :math:`p_\perp/p_\parallel < 10^{-6}` to prevent numerical instabilities
2. **Early Cutoff:** Terminate integration after particle crossing events to avoid runaway behavior
3. **Energy-Dependent Stepping:** Use smaller time steps for lower energy particles
4. **Field Gradient Monitoring:** Adapt steps based on electromagnetic field variations

**Typical Stable Parameters:**

- **High Energy (>10 GeV):** :math:`\Delta t \sim 1-5` ps
- **Medium Energy (100 MeV - 10 GeV):** :math:`\Delta t \sim 0.1-1` ps
- **Low Energy (<100 MeV):** :math:`\Delta t \sim 0.01-0.1` ps

Boundary Conditions and Apertures
---------------------------------

Conducting Apertures
~~~~~~~~~~~~~~~~~~~

Conducting boundaries are modeled using the method of images:

1. **Image Charges:** Mirror charges are placed to satisfy boundary conditions
2. **Image Currents:** Mirror currents maintain current continuity
3. **Surface Fields:** Tangential :math:`\mathbf{E} = 0`, normal :math:`\mathbf{B} = 0`

**Implementation:** The image system is updated dynamically as particles move, maintaining exact boundary conditions throughout the simulation.

Validation and Testing
---------------------

The integrator has been validated against:

1. **Analytic Solutions:** Comparison with known electromagnetic trajectories
2. **Energy Conservation:** Long-term stability in isolated systems
3. **Cyclotron Motion:** Exact circular orbits in uniform magnetic fields
4. **Synchrotron Radiation:** Agreement with classical radiation formulas
5. **Relativistic Effects:** Proper Lorentz transformation behavior

**Test Coverage:** The validation suite includes scenarios from 1 keV to 100 GeV particle energies with electromagnetic fields spanning 12 orders of magnitude.
