Core Integrators
===============

This section documents the core electromagnetic integration classes and their mathematical foundations.

LienardWiechertIntegrator
------------------------

The primary integration class implementing covariant electromagnetic field calculations with conjugate momentum formulation.

.. autoclass:: LW_integrator.covariant_integrator_library_heavyion.LienardWiechertIntegrator
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
~~~~~~~~~~~

.. automethod:: LW_integrator.covariant_integrator_library_heavyion.LienardWiechertIntegrator.integrate_retarded_fields

   **Mathematical Foundation:**

   Implements explicit integration scheme using conjugate momentum:

   .. math::
      \\mathbf{P} = \\gamma m \\mathbf{v} + q \\mathbf{A}

   **Integration Algorithm:**

   1. **Field Evaluation**: Calculate Liénard-Wiechert fields at current positions
   2. **Force Calculation**: Compute Lorentz forces including retardation effects
   3. **Momentum Update**: Advance conjugate momentum using explicit scheme
   4. **Position Update**: Update positions using corrected velocities
   5. **Energy Monitoring**: Validate energy conservation throughout integration

   **Critical Parameters:**

   * ``h_step``: Integration time step (ps) - must satisfy stability criteria
   * ``z_cutoff``: Early termination position (mm) - prevents runaway behavior
   * ``px_initial_fraction``: Minimal transverse momentum fraction - critical for stability
   * ``py_initial_fraction``: Minimal transverse momentum fraction - critical for stability

.. automethod:: LW_integrator.covariant_integrator_library_heavyion.LienardWiechertIntegrator.calculate_retarded_fields

   **Liénard-Wiechert Implementation:**

   Computes exact electromagnetic fields accounting for finite propagation speed:

   .. math::
      \\mathbf{E} = \\frac{q}{4\\pi\\epsilon_0} \\left[ \\frac{\\mathbf{n} - \\boldsymbol{\\beta}}{(1 - \\mathbf{n} \\cdot \\boldsymbol{\\beta})^3 R^2} + \\frac{\\mathbf{n} \\times ((\\mathbf{n} - \\boldsymbol{\\beta}) \\times \\dot{\\boldsymbol{\\beta}})}{c(1 - \\mathbf{n} \\cdot \\boldsymbol{\\beta})^3 R} \\right]_{ret}

Particle State Representation
----------------------------

Particle states use four-vector formalism with conjugate momentum:

**State Dictionary Structure:**

.. code-block:: python

   state = {
       "x": np.array([x_positions]),       # mm
       "y": np.array([y_positions]),       # mm
       "z": np.array([z_positions]),       # mm
       "t": np.array([times]),             # ns
       "Px": np.array([x_momenta]),        # amu·mm/ns (conjugate)
       "Py": np.array([y_momenta]),        # amu·mm/ns (conjugate)
       "Pz": np.array([z_momenta]),        # amu·mm/ns (conjugate)
       "Pt": np.array([time_momenta]),     # amu·mm/ns (energy/c)
       "gamma": np.array([lorentz_factors]),
       "bx": np.array([beta_x]),           # vx/c
       "by": np.array([beta_y]),           # vy/c
       "bz": np.array([beta_z]),           # vz/c
       "q": np.array([charges]),           # amu·mm/ns (Gaussian units)
       "m": np.array([masses]),            # amu
       "char_time": float                  # Characteristic time scale
   }

**Conjugate Momentum Relations:**

.. math::
   \\begin{align}
   \\mathbf{P}_{conjugate} &= \\gamma m \\mathbf{v} + q \\mathbf{A} \\\\
   \\mathbf{v} &= \\frac{\\mathbf{P}_{conjugate} - q \\mathbf{A}}{\\gamma m} \\\\
   E &= P_t c = \\gamma m c^2
   \\end{align}

Physical Constants
-----------------

The integrator uses Gaussian CGS units with the following key constants:

.. code-block:: python

   C_MMNS = 299.792458                    # Speed of light (mm/ns)
   AMU_TO_MEV = 931.494102                # amu to MeV conversion
   ELEMENTARY_CHARGE_GAUSSIAN = 1.178734e-5  # Elementary charge (amu·mm/ns)
   PROTON_MASS_AMU = 1.007276466812       # Proton mass (amu)
   ELECTRON_MASS_AMU = 5.48579909070e-4   # Electron mass (amu)

Stability and Convergence
------------------------

Numerical Stability Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Stability Conditions:**

1. **Minimal Transverse Momentum**: Initialize with ``px_fraction, py_fraction ≤ 1e-6``
2. **Energy-Dependent Time Stepping**:

   * High Energy (>10 GeV): Δt ≤ 5.0 ps
   * Medium Energy (100 MeV - 10 GeV): Δt ≤ 1.0 ps
   * Low Energy (<100 MeV): Δt ≤ 0.1 ps

3. **Early Cutoff**: Terminate integration after physical events (e.g., z = +25mm after crossing)
4. **Energy Conservation**: Monitor |ΔE/E| < 10⁻⁶ for stable integration

**Convergence Criteria:**

.. math::
   \\begin{align}
   \\Delta t &< \\min\\left(\\frac{1}{\\omega_c}, \\frac{\\lambda}{c}\\right) \\\\
   \\omega_c &= \\frac{qB}{\\gamma m} \\\\
   \\lambda &= \\text{shortest EM wavelength}
   \\end{align}

Error Diagnostics
~~~~~~~~~~~~~~~~

**Common Integration Failures:**

1. **Energy Blowup**: Excessive initial transverse momentum
2. **Field Singularities**: Insufficient time step resolution near collision points
3. **Numerical Overflow**: Integration time exceeds stable regime
4. **Conservation Violations**: Indicates fundamental integration problems

**Diagnostic Tools:**

.. code-block:: python

   # Energy conservation monitoring
   analyze_trajectory_endpoints(trajectory, "Particle")

   # Systematic stability study
   energy_conservation_study()

   # Interactive parameter testing
   run_current_implementation(params)

Usage Examples
--------------

**Basic Two-Particle Collision:**

.. code-block:: python

   from LW_integrator.covariant_integrator_library_heavyion import LienardWiechertIntegrator

   # Initialize integrator
   integrator = LienardWiechertIntegrator()

   # Create particle states with minimal transverse momentum
   proton_state = create_particle_state(
       mass_amu=1.007276466812,
       charge_sign=+1,
       kinetic_energy_mev=500.0,
       position_z=-250.0,
       px_fraction=1e-6,  # Critical: minimal Px
       py_fraction=1e-6,  # Critical: minimal Py
       moving_direction=+1
   )

   antiproton_state = create_particle_state(
       mass_amu=1.007276466812,
       charge_sign=-1,
       kinetic_energy_mev=500.0,
       position_z=+250.0,
       px_fraction=1e-6,
       py_fraction=1e-6,
       moving_direction=-1
   )

   # Run integration with early cutoff
   rider_traj, driver_traj = integrator.integrate_retarded_fields(
       static_steps=100,
       ret_steps=50000,
       h_step=1e-3,           # 1 ps time step
       wall_Z=0.0,
       apt_R=0.0,             # No aperture
       sim_type=2,            # Two-particle collision
       init_rider=proton_state,
       init_driver=antiproton_state,
       bunch_dist=0.0,
       z_cutoff=25.0,         # Early cutoff at z=+25mm
       Ez_field=0.0
   )

**Conducting Aperture Study:**

.. code-block:: python

   # Single particle vs conducting aperture
   rider_traj, _ = integrator.integrate_retarded_fields(
       static_steps=100,
       ret_steps=30000,
       h_step=0.2e-3,         # 0.2 ps for aperture interaction
       wall_Z=0.0,
       apt_R=0.5,             # 0.5mm radius aperture
       sim_type=1,            # Single particle
       init_rider=proton_state,
       init_driver=None,
       bunch_dist=0.0,
       z_cutoff=25.0,
       Ez_field=0.0
   )
