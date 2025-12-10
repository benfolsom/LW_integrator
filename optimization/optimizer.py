"""Optimization algorithms for finding optimal LW integrator configurations.

This module provides gradient-free optimization methods to find parameter
configurations that maximize energy gain or other metrics. Uses scipy.optimize
for robust optimization algorithms.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize

from core.integration_runner import run_integrator
from optimization.metrics import compute_trajectory_metrics

logger = logging.getLogger(__name__)


class ObjectiveFunction:
    """Wrapper for objective functions to be minimized.

    Parameters
    ----------
    config_template : Dict[str, Any]
        Base configuration template
    parameter_names : List[str]
        Names of parameters to optimize
    parameter_bounds : List[Tuple[float, float]]
        Bounds for each parameter (min, max)
    metric_name : str, optional
        Name of metric to optimize (default: 'max_energy_gain_gev')
    maximize : bool, optional
        If True, maximizes metric; if False, minimizes (default: True)
    """

    def __init__(
        self,
        config_template: Dict[str, Any],
        parameter_names: List[str],
        parameter_bounds: List[Tuple[float, float]],
        metric_name: str = "max_energy_gain_gev",
        maximize: bool = True,
    ):
        self.config_template = config_template
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.metric_name = metric_name
        self.maximize = maximize

        self.n_calls = 0
        self.best_value = -np.inf if maximize else np.inf
        self.best_params = None
        self.history = []

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate objective function at parameter vector x.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector to evaluate

        Returns
        -------
        float
            Objective function value (negated if maximizing)
        """
        self.n_calls += 1

        # Create config with these parameters
        config = self._create_config(x)

        try:
            # Run simulation
            trajectory = self._run_simulation(config)

            # Compute metrics
            metrics = self._compute_metrics(trajectory, config)

            # Get objective value
            value = metrics.get(self.metric_name, np.nan)

            if np.isnan(value):
                logger.warning(f"NaN metric value for params {x}")
                return np.inf if not self.maximize else -np.inf

            # Track best
            if self.maximize:
                if value > self.best_value:
                    self.best_value = value
                    self.best_params = x.copy()
                objective_value = -value  # Minimize negative
            else:
                if value < self.best_value:
                    self.best_value = value
                    self.best_params = x.copy()
                objective_value = value

            # Record history
            self.history.append(
                {
                    "params": x.copy(),
                    "value": value,
                    "objective": objective_value,
                    "metrics": metrics,
                }
            )

            logger.info(
                f"Evaluation {self.n_calls}: {self.metric_name} = {value:.6f} "
                f"(params: {dict(zip(self.parameter_names, x))})"
            )

            return objective_value

        except Exception as e:
            logger.error(f"Simulation failed for params {x}: {e}")
            return np.inf if not self.maximize else -np.inf

    def _create_config(self, x: np.ndarray) -> Dict[str, Any]:
        """Create configuration from parameter vector."""
        config = self.config_template.copy()

        for param_name, param_value in zip(self.parameter_names, x):
            # Map parameter names to config structure
            if param_name == "aperture_radius":
                if "aperture" not in config:
                    config["aperture"] = {}
                config["aperture"]["radius"] = float(param_value)
            elif param_name == "initial_energy_gev":
                # Convert to gamma
                rest_energy_mev = config.get("rest_energy_mev", 0.511)
                gamma = float(param_value) * 1e3 / rest_energy_mev
                config["initial_gamma"] = gamma
            elif param_name == "timestep":
                config["timestep"] = float(param_value)
            elif param_name == "start_z":
                config["initial_z"] = float(param_value)
            elif param_name == "transverse_momentum":
                config["initial_transverse_momentum"] = float(param_value)
            elif param_name == "position_spread":
                config["initial_position_spread"] = float(param_value)
            else:
                # Direct mapping
                config[param_name] = float(param_value)

        return config

    def _run_simulation(self, config: Dict[str, Any]) -> List:
        """Run simulation with given config."""
        # NOTE: This is a placeholder implementation
        # The actual run_integrator function requires IntegratorConfig and ParticleState objects
        # This would need to be adapted based on your specific configuration structure
        # For now, this serves as a template that users can customize

        raise NotImplementedError(
            "run_simulation needs to be adapted to your specific configuration structure. "
            "Please see the example script for how to properly call run_integrator with "
            "IntegratorConfig and ParticleState objects."
        )

    def _compute_metrics(
        self, trajectory: List, config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute metrics from trajectory."""
        if len(trajectory) == 0:
            return {self.metric_name: np.nan}

        initial_state = trajectory[0]
        rest_energy_mev = config.get("rest_energy_mev", 0.511)

        aperture_z = None
        if "aperture" in config and "z" in config["aperture"]:
            aperture_z = config["aperture"]["z"]

        return compute_trajectory_metrics(
            trajectory, initial_state, rest_energy_mev, aperture_z=aperture_z
        )


def optimize_parameters(
    config_template: Dict[str, Any],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
    metric_name: str = "max_energy_gain_gev",
    method: str = "differential_evolution",
    maximize: bool = True,
    maxiter: int = 100,
    **optimizer_kwargs,
) -> OptimizeResult:
    """Optimize parameters to maximize/minimize a metric.

    Parameters
    ----------
    config_template : Dict[str, Any]
        Base configuration template
    parameter_names : List[str]
        Names of parameters to optimize (e.g., ['aperture_radius', 'initial_energy_gev'])
    parameter_bounds : List[Tuple[float, float]]
        Bounds for each parameter as (min, max) tuples
    metric_name : str, optional
        Name of metric to optimize (default: 'max_energy_gain_gev')
    method : str, optional
        Optimization method: 'differential_evolution', 'nelder_mead', 'powell', etc.
        (default: 'differential_evolution')
    maximize : bool, optional
        If True, maximizes metric; if False, minimizes (default: True)
    maxiter : int, optional
        Maximum number of iterations (default: 100)
    **optimizer_kwargs
        Additional keyword arguments passed to optimizer

    Returns
    -------
    OptimizeResult
        Scipy optimization result with additional attributes:
        - best_params_dict: Dictionary of best parameters
        - objective_function: ObjectiveFunction instance with history
    """
    logger.info(f"Starting optimization of {metric_name}")
    logger.info(f"Parameters: {parameter_names}")
    logger.info(f"Bounds: {parameter_bounds}")
    logger.info(f"Method: {method}")

    # Create objective function
    objective = ObjectiveFunction(
        config_template=config_template,
        parameter_names=parameter_names,
        parameter_bounds=parameter_bounds,
        metric_name=metric_name,
        maximize=maximize,
    )

    # Run optimization
    if method == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds=parameter_bounds,
            maxiter=maxiter,
            **optimizer_kwargs,
        )
    elif method in ["nelder_mead", "powell", "cobyla", "slsqp"]:
        # Need initial guess
        x0 = optimizer_kwargs.pop("x0", None)
        if x0 is None:
            # Use midpoint of bounds
            x0 = [(b[0] + b[1]) / 2 for b in parameter_bounds]

        result = minimize(
            objective,
            x0=x0,
            bounds=parameter_bounds,
            method=method,
            options={"maxiter": maxiter, **optimizer_kwargs},
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")

    # Add custom attributes
    result.best_params_dict = dict(zip(parameter_names, result.x))
    result.objective_function = objective

    logger.info(
        f"Optimization complete. Best {metric_name}: {objective.best_value:.6f}"
    )
    logger.info(f"Best parameters: {result.best_params_dict}")

    return result


def multi_start_optimize(
    config_template: Dict[str, Any],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
    metric_name: str = "max_energy_gain_gev",
    n_starts: int = 5,
    method: str = "nelder_mead",
    maximize: bool = True,
    maxiter: int = 50,
    **optimizer_kwargs,
) -> OptimizeResult:
    """Run multiple optimization attempts with random starting points.

    Useful for finding global optima when landscape has multiple local optima.

    Parameters
    ----------
    config_template : Dict[str, Any]
        Base configuration template
    parameter_names : List[str]
        Names of parameters to optimize
    parameter_bounds : List[Tuple[float, float]]
        Bounds for each parameter
    metric_name : str, optional
        Name of metric to optimize
    n_starts : int, optional
        Number of random starts (default: 5)
    method : str, optional
        Optimization method (default: 'nelder_mead')
    maximize : bool, optional
        If True, maximizes metric (default: True)
    maxiter : int, optional
        Maximum iterations per start (default: 50)
    **optimizer_kwargs
        Additional kwargs for optimizer

    Returns
    -------
    OptimizeResult
        Best result from all starts
    """
    logger.info(f"Running multi-start optimization with {n_starts} starts")

    best_result = None
    best_value = -np.inf if maximize else np.inf
    all_results = []

    for i in range(n_starts):
        logger.info(f"Starting optimization attempt {i + 1}/{n_starts}")

        # Generate random initial point
        x0 = [np.random.uniform(bounds[0], bounds[1]) for bounds in parameter_bounds]

        # Run optimization
        result = optimize_parameters(
            config_template=config_template,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            metric_name=metric_name,
            method=method,
            maximize=maximize,
            maxiter=maxiter,
            x0=x0,
            **optimizer_kwargs,
        )

        all_results.append(result)

        # Check if this is best
        objective_value = result.objective_function.best_value
        if maximize:
            if objective_value > best_value:
                best_value = objective_value
                best_result = result
        else:
            if objective_value < best_value:
                best_value = objective_value
                best_result = result

    # Add all results to best result
    best_result.all_starts = all_results

    logger.info(f"Multi-start optimization complete.")
    logger.info(f"Best {metric_name}: {best_value:.6f}")
    logger.info(f"Best parameters: {best_result.best_params_dict}")

    return best_result


def adaptive_grid_search(
    config_template: Dict[str, Any],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
    metric_name: str = "max_energy_gain_gev",
    maximize: bool = True,
    initial_points_per_dim: int = 5,
    refinement_levels: int = 2,
    refinement_factor: float = 0.2,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Adaptive grid search that refines around promising regions.

    Starts with coarse grid, identifies best regions, then refines.

    Parameters
    ----------
    config_template : Dict[str, Any]
        Base configuration template
    parameter_names : List[str]
        Names of parameters to optimize
    parameter_bounds : List[Tuple[float, float]]
        Bounds for each parameter
    metric_name : str, optional
        Name of metric to optimize
    maximize : bool, optional
        If True, maximizes metric (default: True)
    initial_points_per_dim : int, optional
        Number of points per dimension in initial grid (default: 5)
    refinement_levels : int, optional
        Number of refinement iterations (default: 2)
    refinement_factor : float, optional
        Fraction of parameter range to search in refinement (default: 0.2)

    Returns
    -------
    Tuple[np.ndarray, float, Dict[str, Any]]
        (best_params, best_value, history)
    """
    logger.info("Starting adaptive grid search")

    current_bounds = list(parameter_bounds)
    history = {"levels": []}

    for level in range(refinement_levels + 1):
        logger.info(f"Grid search level {level}")
        logger.info(f"Current bounds: {current_bounds}")

        # Create grid for this level
        grids = [
            np.linspace(bounds[0], bounds[1], initial_points_per_dim)
            for bounds in current_bounds
        ]

        # Evaluate all grid points
        best_value_level = -np.inf if maximize else np.inf
        best_params_level = None

        level_results = []

        for params in np.array(np.meshgrid(*grids)).T.reshape(-1, len(parameter_names)):
            # Create objective
            objective = ObjectiveFunction(
                config_template=config_template,
                parameter_names=parameter_names,
                parameter_bounds=parameter_bounds,
                metric_name=metric_name,
                maximize=maximize,
            )

            # Evaluate
            obj_value = objective(params)
            metric_value = -obj_value if maximize else obj_value

            level_results.append(
                {
                    "params": params,
                    "metric_value": metric_value,
                }
            )

            # Update best for this level
            if maximize:
                if metric_value > best_value_level:
                    best_value_level = metric_value
                    best_params_level = params
            else:
                if metric_value < best_value_level:
                    best_value_level = metric_value
                    best_params_level = params

        history["levels"].append(
            {
                "bounds": current_bounds,
                "best_value": best_value_level,
                "best_params": best_params_level,
                "results": level_results,
            }
        )

        logger.info(
            f"Level {level} best: {best_value_level:.6f} at {best_params_level}"
        )

        # Refine bounds around best point for next level
        if level < refinement_levels:
            new_bounds = []
            for i, (bounds, best_val) in enumerate(
                zip(current_bounds, best_params_level)
            ):
                range_size = bounds[1] - bounds[0]
                new_range = range_size * refinement_factor

                new_min = max(parameter_bounds[i][0], best_val - new_range / 2)
                new_max = min(parameter_bounds[i][1], best_val + new_range / 2)

                new_bounds.append((new_min, new_max))

            current_bounds = new_bounds

    # Return final best
    final_best = history["levels"][-1]
    logger.info(
        f"Adaptive grid search complete. Best {metric_name}: {final_best['best_value']:.6f}"
    )

    return final_best["best_params"], final_best["best_value"], history
