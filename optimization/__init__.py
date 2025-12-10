"""Optimization module for LW integrator parameter tuning.

This module provides tools for finding optimal configurations to maximize
energy gain in laser-wakefield acceleration simulations.

Main components:
- metrics: Compute performance metrics from trajectories
- parameter_sweep: Run systematic parameter space exploration
- optimizer: Gradient-free optimization algorithms
- visualization: Create heatmaps and analysis plots

Example Usage
-------------
Basic parameter sweep:

    >>> from optimization import create_energy_aperture_grid, run_parameter_sweep
    >>> from optimization.visualization import plot_energy_heatmap
    >>>
    >>> # Define parameter grid
    >>> grid = create_energy_aperture_grid(
    ...     aperture_sizes_mm=[0.001, 0.01, 0.1, 1.0],
    ...     energies_gev=[1.0, 10.0, 100.0]
    ... )
    >>>
    >>> # Run sweep
    >>> results = run_parameter_sweep(base_config, grid)
    >>>
    >>> # Visualize
    >>> plot_energy_heatmap(
    ...     results['arrays']['max_energy_gain_gev'],
    ...     save_path='heatmap.png'
    ... )

Optimization:

    >>> from optimization import optimize_parameters
    >>>
    >>> result = optimize_parameters(
    ...     config_template=base_config,
    ...     parameter_names=['aperture_radius', 'initial_energy_gev'],
    ...     parameter_bounds=[(1e-6, 1.0), (1.0, 100.0)],
    ...     metric_name='max_energy_gain_gev',
    ...     method='differential_evolution',
    ...     maximize=True
    ... )
    >>>
    >>> print(f"Best parameters: {result.best_params_dict}")
    >>> print(f"Best energy gain: {result.objective_function.best_value} GeV")
"""

from optimization.metrics import (
    compute_energy_at_position,
    compute_energy_gain_near_aperture,
    compute_max_energy_gain,
    compute_percent_energy_gain,
    compute_relative_energy_gain,
    compute_trajectory_metrics,
    detect_transverse_deflection,
)
from optimization.optimizer import (
    ObjectiveFunction,
    adaptive_grid_search,
    multi_start_optimize,
    optimize_parameters,
)
from optimization.parameter_sweep import (
    ParameterGrid,
    create_energy_aperture_grid,
    load_sweep_results,
    run_parameter_sweep,
)
from optimization.visualization import (
    create_interactive_plot,
    plot_dual_energy_curves,
    plot_energy_heatmap,
    plot_optimization_summary,
    plot_parameter_slice,
)

__all__ = [
    # Metrics
    "compute_max_energy_gain",
    "compute_energy_gain_near_aperture",
    "compute_relative_energy_gain",
    "compute_percent_energy_gain",
    "compute_energy_at_position",
    "detect_transverse_deflection",
    "compute_trajectory_metrics",
    # Parameter sweeps
    "ParameterGrid",
    "create_energy_aperture_grid",
    "run_parameter_sweep",
    "load_sweep_results",
    # Optimization
    "ObjectiveFunction",
    "optimize_parameters",
    "multi_start_optimize",
    "adaptive_grid_search",
    # Visualization
    "plot_energy_heatmap",
    "plot_dual_energy_curves",
    "plot_parameter_slice",
    "plot_optimization_summary",
    "create_interactive_plot",
]

__version__ = "0.1.0"
