Archive Documentation
===================

This document provides information about archived files and the archive management system for the LW Integrator project.

Archive Structure
----------------

Location: ``/home/benfol/work/LW_windows/archive/``

The archive maintains historical development files that have been superseded by more comprehensive implementations:

Directory Organization
~~~~~~~~~~~~~~~~~~~~~

``archive/debug_files_legacy/``
  Contains outdated debugging and test files from electromagnetic integrator development.

  **Key archived files:**

  * ``corrected_crossing_test.py`` - Early crossing detection implementation
  * ``debug_*.py`` files - Various debugging approaches for force calculations and integration
  * ``*_test.py`` files - Multiple test implementations across different energy ranges
  * ``extensive_stability_test.py`` - Early stability analysis framework
  * ``high_energy_acceleration_test.py`` - Superseded by current high-energy testing

``archive/outdated_documentation/``
  Reserved for documentation files that become outdated as the codebase evolves.

Archive Policy
--------------

Files are Archived When
~~~~~~~~~~~~~~~~~~~~~~~

1. **Superseded by Better Implementations**: Functionality consolidated into more comprehensive tools
2. **Experimental Conclusion**: Research phase completed with results integrated elsewhere
3. **Duplicate Functionality**: Multiple implementations consolidated into single authoritative version
4. **Outdated Approaches**: Methods replaced by more stable or efficient alternatives

**Archival Date**: September 24, 2025

Current Active vs Archived Files
--------------------------------

Electromagnetic Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~

**Archived (Legacy):**

.. code-block:: text

   debug_files_legacy/
   ├── corrected_crossing_test.py          # Early crossing detection
   ├── crossing_comparison_test.py         # Comparison framework prototype
   ├── debug_aperture_movement.py          # Aperture interaction debugging
   ├── debug_force_accumulation.py         # Force calculation analysis
   ├── debug_forces.py                     # Force component debugging
   ├── debug_image_charges.py              # Image charge implementation tests
   ├── debug_integration.py                # Integration step debugging
   ├── debug_retarded_step.py              # Retarded field step analysis
   ├── debug_static_issues.py              # Static field debugging
   ├── debug_step_by_step.py               # Detailed step progression
   ├── demo_extreme_acceleration.py        # Extreme acceleration scenarios
   ├── double_precision_test.py            # Precision testing
   ├── efficient_crossing_test.py          # Crossing efficiency optimization
   ├── extensive_stability_test.py         # Comprehensive stability analysis
   ├── high_energy_acceleration_test.py    # High-energy testing framework
   ├── high_precision_test.py              # Precision validation
   ├── large_aperture_test.py              # Large aperture interactions
   ├── legacy_static_test.py               # Static field legacy comparison
   ├── minimal_aperture_debug.py           # Small aperture debugging
   ├── mixed_static_retarded_test.py       # Mixed field regime testing
   ├── realistic_step_crossing_test.py     # Realistic crossing scenarios
   ├── simple_double_precision_test.py     # Basic precision testing
   ├── simple_high_energy_test.py          # Simple high-energy scenarios
   ├── static_only_test.py                 # Static-only field testing
   ├── test_extreme_acceleration.py        # Extreme acceleration validation
   ├── test_force_direction.py             # Force direction validation
   ├── test_low_energy.py                  # Low-energy regime testing
   ├── two_particle_stability_test.py      # Two-particle stability analysis
   └── verify_static_legacy.py             # Static field legacy verification

**Current (Active):**

.. code-block:: text

   debug_files/
   ├── electromagnetic_debugging_notebook.ipynb    # Primary interactive debugging environment
   ├── fast_high_energy_test.py                   # Current high-energy testing framework
   └── PERSISTENT_NOTE_TRAJECTORY_ANALYSIS.py     # User requirements documentation

Replacement Mapping
-------------------

Specific File Replacements
~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-Energy Testing:**

* **Archived**: ``high_energy_acceleration_test.py``, ``simple_high_energy_test.py``
* **Current**: ``fast_high_energy_test.py`` with comprehensive trajectory analysis
* **Improvement**: Energy conservation monitoring, minimal momentum initialization, early cutoff mechanisms

**Stability Analysis:**

* **Archived**: ``extensive_stability_test.py``, ``two_particle_stability_test.py``
* **Current**: ``electromagnetic_debugging_notebook.ipynb`` with systematic parameter studies
* **Improvement**: Interactive parameter modification, real-time visualization, automated parameter sweeps

**Debugging Framework:**

* **Archived**: Multiple ``debug_*.py`` files with specific focus areas
* **Current**: ``electromagnetic_debugging_notebook.ipynb`` with comprehensive debugging suite
* **Improvement**: Unified debugging environment, interactive plotting, configurable test scenarios

**Force and Field Analysis:**

* **Archived**: ``debug_forces.py``, ``debug_force_accumulation.py``, ``debug_retarded_step.py``
* **Current**: Integrated into ``electromagnetic_debugging_notebook.ipynb`` with trajectory analysis
* **Improvement**: Real-time force monitoring, energy conservation tracking, comprehensive state analysis

Archive Retrieval Guidelines
----------------------------

If Archived Code is Needed
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Check Current Implementation First**: Functionality may exist in active files
2. **Verify Compatibility**: Archived files may not work with current codebase structure
3. **Update Dependencies**: File paths and import statements may need modification
4. **Consider Architecture Changes**: Conjugate momentum formulation may require code updates

**File Path Updates Required:**

.. code-block:: python

   # Archived files may have outdated imports
   from LW_integrator.old_module import OldClass  # ❌ Outdated

   # Update to current structure
   from LW_integrator.covariant_integrator_library_heavyion import LienardWiechertIntegrator  # ✅ Current

**Parameter Updates Required:**

.. code-block:: python

   # Archived files may not include stability requirements
   params = {
       'step_size': 1e-3,
       'max_steps': 10000
   }  # ❌ Missing critical stability parameters

   # Update with current stability requirements
   params = SimulationParams(
       step_size_ps=1.0,
       max_steps=10000,
       px_initial_fraction=1e-6,  # ✅ Critical for stability
       py_initial_fraction=1e-6,  # ✅ Critical for stability
       cutoff_z_mm=25.0          # ✅ Early cutoff mechanism
   )

Archive Maintenance
------------------

Regular Archive Review
~~~~~~~~~~~~~~~~~~~~~~

**Quarterly Review Process:**

1. **Identify Redundant Files**: Check for multiple implementations of same functionality
2. **Consolidate Documentation**: Update archive README with new archival decisions
3. **Verify Current Alternatives**: Ensure archived functionality exists in active codebase
4. **Clean Obsolete Dependencies**: Remove files that reference non-existent modules

**Archive Expansion Criteria:**

Files should be archived when:

* **Functionality Superseded**: Better implementation available in active codebase
* **Research Phase Complete**: Experimental work concluded with results integrated
* **Maintenance Burden**: File requires updates that don't provide proportional value
* **Code Quality Issues**: Implementation doesn't meet current standards

**Retention Policy:**

* **Keep All Archived Files**: Historical context valuable for understanding development decisions
* **Maintain Archive Documentation**: Keep detailed records of archival reasons and replacements
* **Preserve Git History**: Archived files remain accessible in version control
* **Update Cross-References**: Ensure documentation points to current implementations

Integration with Documentation
-----------------------------

The archive is integrated with the main documentation system:

**Developer Guide References:**

* Archive policy and procedures documented in developer guide
* Replacement mapping provided for developers seeking archived functionality
* Integration guidelines for retrieving and updating archived code

**API Documentation Updates:**

* Deprecated functions marked with archive references
* Current alternatives highlighted in API documentation
* Migration guides provided for major architectural changes

**Example Documentation Pattern:**

.. code-block:: python

   def old_integration_method():
       """
       Legacy integration method - ARCHIVED

       .. deprecated:: 2025.09
          This method has been archived. Use
          :meth:`LienardWiechertIntegrator.integrate_retarded_fields` instead.

       **Archive Location**: ``archive/debug_files_legacy/old_integration_test.py``

       **Replacement**:
          - Current: ``electromagnetic_debugging_notebook.ipynb``
          - Improvements: Energy conservation monitoring, stability controls
       """
       pass

This archive system maintains development history while keeping the active codebase focused and maintainable.
