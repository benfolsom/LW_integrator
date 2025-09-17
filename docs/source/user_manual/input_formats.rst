Input Formats
=============

The LW Integrator supports multiple input formats to ensure compatibility with existing accelerator physics workflows and provide flexibility for different use cases.

Supported Formats
-----------------

* **JSON** - Primary format with full feature support
* **YAML** - Human-readable format ideal for documentation and manual editing
* **MAD-X** - Basic compatibility for importing lattice definitions
* **ELEGANT** - Basic compatibility for beam distributions
* **HDF5** - Binary format for large datasets and programmatic generation

Standard JSON Format
--------------------

The JSON format provides complete access to all LW Integrator features:

.. code-block:: json

   {
     "description": "Complete simulation specification",
     "version": "1.0",
     "beam": {
       "particle_type": "electron",
       "total_energy_mev": 10000.0,
       "n_particles": 1000,
       "particles_per_macroparticle": 1000000,
       "emit_x": 1e-9,
       "emit_y": 1e-11,
       "beta_x": 10.0,
       "beta_y": 5.0,
       "alpha_x": 0.0,
       "alpha_y": 0.0,
       "sigma_z": 1e-4,
       "sigma_dp": 1e-4,
       "distribution_x": "gaussian",
       "distribution_y": "gaussian", 
       "distribution_z": "gaussian"
     },
     "lattice": [
       {
         "name": "quad1",
         "type": "quadrupole",
         "length": 0.5,
         "strength": 2.5,
         "aperture": {
           "aperture_type": "circular",
           "aperture_limits": [0.005],
           "material": "copper",
           "thickness": 0.001,
           "position": 0.25
         }
       }
     ],
     "simulation": {
       "n_turns": 1,
       "n_steps_per_element": 500,
       "integration_method": "adaptive",
       "radiation_reaction": true,
       "space_charge": false,
       "wake_fields": false,
       "force_threshold": 1e-8,
       "energy_threshold": 1e-8,
       "field_gradient_threshold": 1e-6,
       "output_frequency": 10,
       "save_trajectories": true,
       "save_distributions": true
     }
   }

Beam Parameters
~~~~~~~~~~~~~~~

The beam section defines the particle species and distribution:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Type
     - Description
   * - ``particle_type``
     - string
     - Particle species: "electron", "proton", "muon", etc.
   * - ``total_energy_mev``
     - float
     - Total energy in MeV (rest mass + kinetic)
   * - ``n_particles``
     - integer
     - Number of macroparticles to track
   * - ``particles_per_macroparticle``
     - integer
     - Real particles per macroparticle (for space charge)
   * - ``emit_x``, ``emit_y``
     - float
     - Normalized emittances in m⋅rad
   * - ``beta_x``, ``beta_y``
     - float
     - Beta functions in meters
   * - ``alpha_x``, ``alpha_y``
     - float
     - Alpha parameters (dimensionless)
   * - ``sigma_z``
     - float
     - RMS bunch length in meters
   * - ``sigma_dp``
     - float
     - RMS momentum spread (relative)
   * - ``distribution_x``, ``distribution_y``, ``distribution_z``
     - string
     - Distribution types: "gaussian", "uniform", "waterbag"

Lattice Elements
~~~~~~~~~~~~~~~~

Supported lattice element types:

**Drift Spaces**

.. code-block:: json

   {
     "name": "drift1",
     "type": "drift", 
     "length": 1.0,
     "aperture": {
       "aperture_type": "circular",
       "aperture_limits": [0.01]
     }
   }

**Quadrupoles**

.. code-block:: json

   {
     "name": "quad1",
     "type": "quadrupole",
     "length": 0.5,
     "strength": 2.5,
     "aperture": {
       "aperture_type": "rectangular", 
       "aperture_limits": [0.02, 0.015]
     }
   }

**Dipoles**

.. code-block:: json

   {
     "name": "bend1",
     "type": "dipole",
     "length": 2.0,
     "angle": 0.1,
     "field": 1.5,
     "edge_angle": [0.05, 0.05]
   }

**Sextupoles**

.. code-block:: json

   {
     "name": "sext1",
     "type": "sextupole", 
     "length": 0.2,
     "strength": 100.0
   }

Aperture Definitions
~~~~~~~~~~~~~~~~~~~~

Apertures define physical boundaries and material interactions:

.. code-block:: json

   {
     "aperture_type": "circular",
     "aperture_limits": [0.005],
     "material": "copper",
     "thickness": 0.001,
     "position": 0.0,
     "roughness": 1e-6,
     "conductivity": 5.96e7
   }

Aperture types:

* ``"circular"``: Single radius limit
* ``"rectangular"``: [half_width, half_height]  
* ``"elliptical"``: [semi_major, semi_minor]
* ``"custom"``: User-defined boundary function

Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~

Control integration and physics models:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Type  
     - Description
   * - ``integration_method``
     - string
     - "basic", "optimized", "adaptive", "self_consistent"
   * - ``radiation_reaction``
     - boolean
     - Include synchrotron radiation and reaction forces
   * - ``space_charge``
     - boolean
     - Include self-field effects between macroparticles
   * - ``wake_fields``
     - boolean
     - Include resistive wall and geometric wake fields
   * - ``force_threshold``
     - float
     - Adaptive integration force magnitude threshold
   * - ``energy_threshold``
     - float
     - Energy change threshold for step size control
   * - ``output_frequency``
     - integer
     - Save particle data every N integration steps

YAML Format
-----------

YAML provides the same functionality as JSON with improved readability:

.. code-block:: yaml

   description: "10 GeV electron aperture study"
   version: "1.0"
   
   beam:
     particle_type: electron
     total_energy_mev: 10000.0
     n_particles: 500
     particles_per_macroparticle: 1000000
     
     # Transverse parameters
     emit_x: 1.0e-9    # 1 nm⋅rad
     emit_y: 1.0e-11   # 10 pm⋅rad
     beta_x: 12.0      # m
     beta_y: 8.0       # m
     
     # Longitudinal parameters  
     sigma_z: 1.0e-4   # 0.1 mm
     sigma_dp: 1.0e-4  # 0.01%
     
     # Distribution types
     distribution_x: gaussian
     distribution_y: gaussian
     distribution_z: gaussian
   
   lattice:
     - name: entrance
       type: drift
       length: 0.5
       aperture:
         aperture_type: circular
         aperture_limits: [0.01]
         material: copper
     
     - name: focus_quad
       type: quadrupole
       length: 0.3
       strength: 5.0
       aperture:
         aperture_type: circular
         aperture_limits: [0.005]
   
   simulation:
     n_turns: 0
     n_steps_per_element: 1000
     integration_method: adaptive
     radiation_reaction: true
     space_charge: false
     force_threshold: 1.0e-8
     energy_threshold: 1.0e-8
     save_trajectories: true

MAD-X Compatibility
-------------------

Basic MAD-X lattice definitions can be imported:

.. code-block:: text

   ! MAD-X input file
   BEAM, PARTICLE=ELECTRON, ENERGY=10.0;
   
   QF: QUADRUPOLE, L=0.5, K1=2.0;
   QD: QUADRUPOLE, L=0.5, K1=-2.0;
   DRIFT1: DRIFT, L=1.0;
   
   LATTICE: LINE=(QF, DRIFT1, QD);

Import using:

.. code-block:: python

   from lw_integrator.io import StandardInputFormat
   
   config = StandardInputFormat()
   config.load_from_file('lattice.madx')

Note: MAD-X parsing is simplified and may not support all MAD-X features. For complex lattices, use the native JSON/YAML formats.

ELEGANT Compatibility  
---------------------

Import beam distributions from ELEGANT:

.. code-block:: text

   &run_setup
     lattice = lattice.lte,
     p_central_mev = 10000,
     default_order = 2
   &end
   
   &bunched_beam
     n_particles_per_bunch = 1000,
     emit_x = 1e-9, emit_y = 1e-11,
     beta_x = 10, beta_y = 5,
     sigma_dp = 1e-4, sigma_s = 1e-4
   &end

Programmatic Generation
-----------------------

For complex setups, generate input files programmatically:

.. code-block:: python

   from lw_integrator.io import StandardInputFormat, BeamParameters, LatticeElement
   from lw_integrator.io import ApertureDefinition, SimulationParameters
   from lw_integrator.io import ParticleType, DistributionType
   
   # Create configuration
   config = StandardInputFormat()
   
   # Define beam
   config.beam_parameters = BeamParameters.from_energy(
       particle_type=ParticleType.ELECTRON,
       energy_mev=10000.0,
       n_particles=1000,
       emit_x=1e-9,
       emit_y=1e-11,
       beta_x=10.0,
       beta_y=5.0
   )
   
   # Build lattice
   aperture = ApertureDefinition(
       aperture_type="circular",
       aperture_limits=[0.005],
       material="copper"
   )
   
   quad = LatticeElement(
       name="focusing_quad",
       element_type="quadrupole", 
       length=0.5,
       aperture=aperture
   )
   
   config.lattice_elements = [quad]
   
   # Set simulation parameters
   config.simulation_parameters = SimulationParameters(
       integration_method='adaptive',
       radiation_reaction=True,
       n_steps_per_element=500
   )
   
   # Save configuration
   config.save_to_file('generated_config.yaml')

Validation and Error Checking
------------------------------

The input parser performs comprehensive validation:

.. code-block:: python

   try:
       config = StandardInputFormat()
       config.load_from_file('my_input.yaml')
   except ValueError as e:
       print(f"Input validation error: {e}")
   except FileNotFoundError:
       print("Input file not found")

Common validation errors:

* Invalid particle types or energies
* Negative aperture dimensions
* Inconsistent beam parameters
* Missing required fields
* Incompatible physics flags

For detailed validation messages, enable debug logging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   config = StandardInputFormat()
   config.load_from_file('input.yaml')  # Detailed validation output

Best Practices
--------------

1. **Start Simple**: Begin with YAML format for readability
2. **Validate Early**: Test input files with small particle counts first
3. **Document Settings**: Include descriptions for non-obvious parameters
4. **Version Control**: Track input file changes with your simulation results
5. **Modular Design**: Separate beam, lattice, and simulation configurations for reuse
6. **Units**: Always specify units explicitly in comments
7. **Cross-Check**: Compare with analytical estimates for simple cases