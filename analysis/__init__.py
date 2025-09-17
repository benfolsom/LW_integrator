"""
Enhanced Aperture Analysis Module

This module provides comprehensive aperture verification and analysis tools
with enhanced energy tracking, trajectory visualization, and optimized performance.

Key Components:
- aperture_verification: Enhanced physics tracking with energy evolution
- interactive_analysis: Fast parameter exploration and testing
- Optimized for computational efficiency while maintaining physics accuracy

Author: LW Integrator Development Team
Date: 2025-09-15
"""

# Import main analysis functions with error handling for missing dependencies
try:
    from .aperture_verification import (
        enhanced_beam_initialization,
        run_enhanced_simulation,
        run_optimized_test_suite,
        save_results,
        EnhancedMacroParticle,
        TrajectoryData
    )
    APERTURE_VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: aperture_verification module not fully available: {e}")
    APERTURE_VERIFICATION_AVAILABLE = False

try:
    from .interactive_analysis import (
        quick_test,
        run_fast_simulation,
        run_realistic_test_suite,
        get_realistic_configs,
        FastSimConfig,
        SimpleParticle
    )
    INTERACTIVE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: interactive_analysis module not fully available: {e}")
    INTERACTIVE_ANALYSIS_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "LW Integrator Development Team"

# Dynamic __all__ based on available modules
__all__ = []

if APERTURE_VERIFICATION_AVAILABLE:
    __all__.extend([
        'enhanced_beam_initialization',
        'run_enhanced_simulation', 
        'run_optimized_test_suite',
        'save_results',
        'EnhancedMacroParticle',
        'TrajectoryData'
    ])

if INTERACTIVE_ANALYSIS_AVAILABLE:
    __all__.extend([
        'quick_test',
        'run_fast_simulation',
        'run_realistic_test_suite',
        'get_realistic_configs',
        'FastSimConfig',
        'SimpleParticle'
    ])

from .aperture_verification import (
    TrajectoryData,
    EnhancedMacroParticle,
    enhanced_beam_initialization,
    run_enhanced_simulation,
    run_optimized_test_suite,
    save_results
)

from .interactive_analysis import (
    FastSimConfig,
    SimpleParticle,
    initialize_beam,
    run_fast_simulation,
    quick_test,
    get_realistic_configs,
    run_realistic_test_suite
)

__all__ = [
    # aperture_verification
    'TrajectoryData',
    'EnhancedMacroParticle', 
    'enhanced_beam_initialization',
    'run_enhanced_simulation',
    'run_optimized_test_suite',
    'save_results',
    # interactive_analysis
    'FastSimConfig',
    'SimpleParticle',
    'initialize_beam',
    'run_fast_simulation',
    'quick_test',
    'get_realistic_configs',
    'run_realistic_test_suite'
]