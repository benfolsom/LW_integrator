"""Optimization algorithms for finding optimal LW integrator configurations.

This module provides gradient-free optimization methods to find parameter
configurations that maximize energy gain or other metrics. Uses scipy.optimize
for robust optimization algorithms.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

# Genetic algorithm utilities
from typing import Tuple as TypingTuple

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
    objective_function: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
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
    objective_function : Callable, optional
        Custom objective function that takes parameter array and returns scalar
        to minimize. If None, uses default ObjectiveFunction class.
    progress_callback : Callable, optional
        Callback function called during optimization progress.
        For differential_evolution: called with (xk, convergence) after each generation.
        For other methods: called with iteration info when available.
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

    # Create or use provided objective function
    if objective_function is None:
        objective = ObjectiveFunction(
            config_template=config_template,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            metric_name=metric_name,
            maximize=maximize,
        )
    else:
        objective = objective_function

    # Run optimization
    if method == "differential_evolution":
        # Wrap progress callback for scipy's differential_evolution format
        scipy_callback = None
        if progress_callback:
            iteration_counter = [0]

            def scipy_callback(xk, convergence=0.0):
                iteration_counter[0] += 1
                best_value = objective(xk)
                # Convert back to maximization if needed
                if maximize:
                    best_value = -best_value
                progress_callback(
                    generation=iteration_counter[0],
                    best_value=best_value,
                    improvement=convergence,
                    tolerance=0.01,  # scipy's default
                    patience_remaining=maxiter - iteration_counter[0],
                    converged=False,
                )
                return False  # Don't stop early

        result = differential_evolution(
            objective,
            bounds=parameter_bounds,
            maxiter=maxiter,
            callback=scipy_callback,
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
    n_starts: int = 10,
    maximize: bool = True,
    maxiter: int = 100,
    method: str = "nelder_mead",
    objective_function: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
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
        Number of random starts (default: 10)
    maximize : bool, optional
        If True, maximizes metric (default: True)
    maxiter : int, optional
        Maximum iterations per start (default: 100)
    method : str, optional
        Optimization method (default: 'nelder_mead')
    objective_function : Callable, optional
        Custom objective function that takes parameter array and returns scalar
        to minimize. If None, uses default ObjectiveFunction class.
    progress_callback : Callable, optional
        Callback function called after each refinement level completes.
    progress_callback : Callable, optional
        Callback function called after each start completes.
    **optimizer_kwargs
        Additional keyword arguments passed to the local optimizer
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
            progress_callback=progress_callback,
            objective_function=objective_function,
            x0=x0,
            **optimizer_kwargs,
        )

        all_results.append(result)

        # Check if this is best
        # Handle both ObjectiveFunction and custom objective functions
        if hasattr(result, "objective_function") and hasattr(
            result.objective_function, "best_value"
        ):
            objective_value = result.objective_function.best_value
        else:
            # For custom objective functions, use result.fun (negated if maximizing)
            objective_value = -result.fun if maximize else result.fun

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
    refinement_factor: float = 0.5,
    objective_function: Optional[Callable] = None,
    progress_callback: Optional[Callable] = None,
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
            # Create or use provided objective function
            if objective_function is None:
                objective = ObjectiveFunction(
                    config_template=config_template,
                    parameter_names=parameter_names,
                    parameter_bounds=parameter_bounds,
                    metric_name=metric_name,
                    maximize=maximize,
                )
                # Evaluate
                obj_value = objective(params)
            else:
                # Use provided objective function directly
                obj_value = objective_function(params)
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

        logger.info(f"Level {level} complete. Best: {best_value_level:.6f}")

        # Call progress callback if provided
        if progress_callback:
            progress_callback(
                generation=level + 1,
                best_value=best_value_level,
                improvement=0.0,  # Grid search doesn't have improvement metric
                tolerance=0.0,
                patience_remaining=refinement_levels - level,
                converged=(level == refinement_levels),
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


def genetic_algorithm(
    config_template: Dict[str, Any],
    parameter_names: List[str],
    parameter_bounds: List[Tuple[float, float]],
    metric_name: str = "max_energy_gain_gev",
    maximize: bool = True,
    population_size: int = 20,
    n_generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    elite_fraction: float = 0.1,
    tournament_size: int = 3,
    seed: Optional[int] = None,
    objective_function: Optional[Callable] = None,
    convergence_tol: float = 1e-6,
    convergence_patience: int = 10,
    progress_callback: Optional[Callable] = None,
) -> OptimizeResult:
    """Genetic algorithm optimization.

    Uses evolutionary approach with selection, crossover, and mutation to find
    optimal parameter configurations.

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
    population_size : int, optional
        Size of population (default: 20)
    n_generations : int, optional
        Number of generations to evolve (default: 50)
    mutation_rate : float, optional
        Probability of mutation per gene (default: 0.1)
    crossover_rate : float, optional
        Probability of crossover (default: 0.7)
    elite_fraction : float, optional
        Fraction of population to preserve as elite (default: 0.1)
    tournament_size : int, optional
        Size of tournament for selection (default: 3)
    seed : int, optional
        Random seed for reproducibility
    objective_function : Callable, optional
        Custom objective function that takes parameter array and returns scalar
        to minimize. If None, uses default ObjectiveFunction class.
    convergence_tol : float, optional
        Relative tolerance for convergence detection (default: 1e-6).
        Stops if improvement over last `convergence_patience` generations
        is less than this tolerance.
    convergence_patience : int, optional
        Number of generations to look back for convergence check (default: 10).
        Early stopping triggers if best fitness doesn't improve by
        `convergence_tol` over this many generations.
    progress_callback : Callable, optional
        Callback function called after each generation with convergence info.
        Signature: callback(generation, best_value, improvement, tolerance, patience_remaining)

    Returns
    -------
    OptimizeResult
        Optimization result with best individual and convergence history
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Starting genetic algorithm optimization of {metric_name}")
    logger.info(f"Population: {population_size}, Generations: {n_generations}")
    logger.info(f"Parameters: {parameter_names}")
    logger.info(f"Bounds: {parameter_bounds}")

    n_params = len(parameter_names)
    n_elite = max(1, int(population_size * elite_fraction))

    # Create or use provided objective function
    if objective_function is None:
        objective = ObjectiveFunction(
            config_template=config_template,
            parameter_names=parameter_names,
            parameter_bounds=parameter_bounds,
            metric_name=metric_name,
            maximize=maximize,
        )
    else:
        objective = objective_function

    # Initialize population randomly within bounds
    population = np.random.uniform(
        low=[b[0] for b in parameter_bounds],
        high=[b[1] for b in parameter_bounds],
        size=(population_size, n_params),
    )

    # Evaluate initial population
    fitness = np.array([objective(ind) for ind in population])

    # Track best individual (remember: objective returns value to minimize)
    if maximize:
        best_idx = np.argmin(fitness)  # Most negative = best for maximization
    else:
        best_idx = np.argmin(fitness)  # Most positive = best for minimization

    best_individual = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    convergence_history = []

    # Evolution loop
    for generation in range(n_generations):
        # Sort population by fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        # Track convergence
        current_best_fitness = fitness[0]
        convergence_history.append(
            {
                "generation": generation,
                "best_fitness": current_best_fitness,
                "mean_fitness": np.mean(fitness),
                "std_fitness": np.std(fitness),
            }
        )

        # Update global best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[0].copy()

        logger.info(
            f"Generation {generation + 1}/{n_generations}: "
            f"Best = {-best_fitness if maximize else best_fitness:.6f}, "
            f"Mean = {-np.mean(fitness) if maximize else np.mean(fitness):.6f}"
        )

        # Check for early stopping (fitness plateau detection)
        improvement = 0.0
        tolerance = convergence_tol
        patience_remaining = convergence_patience - generation

        if generation >= convergence_patience:
            recent_best = [
                h["best_fitness"] for h in convergence_history[-convergence_patience:]
            ]
            improvement = abs(recent_best[0] - recent_best[-1])
            # Use relative tolerance with fallback to absolute for near-zero values
            tolerance = (
                convergence_tol * abs(recent_best[-1])
                if abs(recent_best[-1]) > 1e-10
                else convergence_tol * 1e-2
            )

            if improvement < tolerance:
                if progress_callback:
                    progress_callback(
                        generation=generation + 1,
                        best_value=-best_fitness if maximize else best_fitness,
                        improvement=improvement,
                        tolerance=tolerance,
                        patience_remaining=0,
                        converged=True,
                    )
                logger.info(
                    f"Early stopping at generation {generation + 1}: "
                    f"fitness plateau detected (improvement={improvement:.2e} < tolerance={tolerance:.2e})"
                )
                logger.info(
                    f"Best fitness converged to: {-best_fitness if maximize else best_fitness:.6f}"
                )
                logger.info(
                    f"Convergence achieved after {generation + 1}/{n_generations} generations"
                )
                # Break out of evolution loop
                break
            else:
                patience_remaining = convergence_patience - (
                    generation - convergence_patience
                )

        # Call progress callback with convergence info
        if progress_callback:
            progress_callback(
                generation=generation + 1,
                best_value=-best_fitness if maximize else best_fitness,
                improvement=improvement,
                tolerance=tolerance,
                patience_remaining=patience_remaining,
                converged=False,
            )

        # Create new population
        new_population = []

        # Elitism: preserve best individuals
        new_population.extend(population[:n_elite])

        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = _tournament_selection(population, fitness, tournament_size)
            parent2 = _tournament_selection(population, fitness, tournament_size)

            # Crossover
            if np.random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = _mutate(child1, parameter_bounds, mutation_rate)
            child2 = _mutate(child2, parameter_bounds, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = np.array(new_population)

        # Evaluate new population
        fitness = np.array([objective(ind) for ind in population])

        # Check for cancellation (all fitness values are inf)
        if np.all(np.isinf(fitness)):
            logger.info(
                f"Optimization cancelled at generation {generation + 1}: "
                f"all evaluations returned inf (likely user cancellation)"
            )
            # Break out of evolution loop
            break

    # Final evaluation
    sorted_indices = np.argsort(fitness)
    population = population[sorted_indices]
    fitness = fitness[sorted_indices]

    if fitness[0] < best_fitness:
        best_fitness = fitness[0]
        best_individual = population[0].copy()

    # Create result object
    result = OptimizeResult()
    result.x = best_individual
    result.fun = best_fitness
    # Handle both ObjectiveFunction and custom objective functions
    if hasattr(objective, "n_calls"):
        result.nfev = objective.n_calls
    else:
        result.nfev = population_size * n_generations  # Approximate
    result.nit = n_generations
    result.success = True
    result.message = f"Genetic algorithm completed {n_generations} generations"
    result.best_params_dict = dict(zip(parameter_names, best_individual))
    result.objective_function = objective
    result.convergence_history = convergence_history
    result.final_population = population
    result.final_fitness = fitness

    # Log completion with appropriate metric value
    if hasattr(objective, "best_value"):
        logger.info(
            f"Genetic algorithm complete. Best {metric_name}: {objective.best_value:.6f}"
        )
    else:
        # For custom objective functions, use the best fitness (negated if maximizing)
        best_metric_value = -best_fitness if maximize else best_fitness
        logger.info(
            f"Genetic algorithm complete. Best {metric_name}: {best_metric_value:.6f}"
        )
    logger.info(f"Best parameters: {result.best_params_dict}")

    return result


def _tournament_selection(
    population: np.ndarray, fitness: np.ndarray, tournament_size: int
) -> np.ndarray:
    """Select individual using tournament selection.

    Parameters
    ----------
    population : np.ndarray
        Population array (n_individuals, n_params)
    fitness : np.ndarray
        Fitness values (lower is better)
    tournament_size : int
        Number of individuals to compete

    Returns
    -------
    np.ndarray
        Selected individual
    """
    # Randomly select tournament_size individuals
    tournament_indices = np.random.choice(
        len(population), size=tournament_size, replace=False
    )
    tournament_fitness = fitness[tournament_indices]

    # Select best from tournament (lowest fitness)
    winner_idx = tournament_indices[np.argmin(tournament_fitness)]

    return population[winner_idx].copy()


def _crossover(
    parent1: np.ndarray, parent2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform uniform crossover between two parents.

    Parameters
    ----------
    parent1 : np.ndarray
        First parent
    parent2 : np.ndarray
        Second parent

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two offspring
    """
    n_params = len(parent1)

    # Uniform crossover: randomly choose genes from each parent
    mask = np.random.random(n_params) < 0.5

    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)

    return child1, child2


def _mutate(
    individual: np.ndarray,
    bounds: List[Tuple[float, float]],
    mutation_rate: float,
) -> np.ndarray:
    """Apply mutation to individual.

    Uses Gaussian mutation with adaptive step size based on parameter range.

    Parameters
    ----------
    individual : np.ndarray
        Individual to mutate
    bounds : List[Tuple[float, float]]
        Parameter bounds
    mutation_rate : float
        Probability of mutation per gene

    Returns
    -------
    np.ndarray
        Mutated individual
    """
    mutated = individual.copy()

    for i in range(len(individual)):
        if np.random.random() < mutation_rate:
            # Gaussian mutation with range-based step size
            param_range = bounds[i][1] - bounds[i][0]
            step_size = 0.1 * param_range  # 10% of range

            mutated[i] += np.random.normal(0, step_size)

            # Clip to bounds
            mutated[i] = np.clip(mutated[i], bounds[i][0], bounds[i][1])

    return mutated
