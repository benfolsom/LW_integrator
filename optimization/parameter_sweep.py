"""Parameter sweep functionality for LW integrator optimization.

This module provides tools to run parameter sweeps over configuration spaces,
enabling systematic exploration of aperture sizes, energies, and other parameters
to find optimal configurations for maximum energy gain.
"""

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.integration_runner import run_integrator
from core.types import ParticleState
from optimization.metrics import compute_trajectory_metrics

logger = logging.getLogger(__name__)


class ParameterGrid:
    """Define a grid of parameters to sweep over.

    Parameters
    ----------
    parameters : Dict[str, List[Any]]
        Dictionary mapping parameter names to lists of values to sweep
    """

    def __init__(self, parameters: Dict[str, List[Any]]):
        self.parameters = parameters
        self.param_names = list(parameters.keys())
        self.param_values = [parameters[name] for name in self.param_names]

    def __iter__(self):
        """Iterate over all parameter combinations."""
        for values in itertools.product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __len__(self) -> int:
        """Return total number of parameter combinations."""
        length = 1
        for values in self.param_values:
            length *= len(values)
        return length

    def get_grid_shape(self) -> Tuple[int, ...]:
        """Return shape of the parameter grid."""
        return tuple(len(values) for values in self.param_values)


def create_energy_aperture_grid(
    aperture_sizes_mm: Optional[List[float]] = None,
    energies_gev: Optional[List[float]] = None,
) -> ParameterGrid:
    """Create a standard grid for aperture size vs energy sweeps.

    Parameters
    ----------
    aperture_sizes_mm : List[float], optional
        List of aperture sizes in mm. If None, uses default range.
    energies_gev : List[float], optional
        List of energies in GeV. If None, uses default range.

    Returns
    -------
    ParameterGrid
        Grid for sweeping aperture and energy
    """
    if aperture_sizes_mm is None:
        # Default: log-spaced from 1nm to 1mm
        aperture_sizes_mm = np.logspace(-6, 0, 20).tolist()  # 1e-6 mm = 1 nm

    if energies_gev is None:
        # Default: log-spaced from 1 MeV to 500 GeV
        energies_gev = np.logspace(-3, 2.7, 20).tolist()  # 1e-3 GeV = 1 MeV

    return ParameterGrid(
        {
            "aperture_radius": aperture_sizes_mm,
            "initial_energy_gev": energies_gev,
        }
    )


def run_parameter_sweep(
    base_config: Dict[str, Any],
    parameter_grid: ParameterGrid,
    output_dir: Optional[Path] = None,
    metric_function: Optional[Callable] = None,
    save_trajectories: bool = False,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """Run a parameter sweep over a configuration grid.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration dictionary. Parameters from grid will override.
    parameter_grid : ParameterGrid
        Grid of parameters to sweep over
    output_dir : Path, optional
        Directory to save results. If None, doesn't save to disk.
    metric_function : Callable, optional
        Function to compute metrics from trajectory. If None, uses default.
    save_trajectories : bool, optional
        Whether to save full trajectories (default: False, saves space)
    max_workers : int, optional
        Number of parallel workers (default: 1, sequential)

    Returns
    -------
    Dict[str, Any]
        Results dictionary containing:
        - 'parameters': List of parameter dictionaries
        - 'metrics': List of metric dictionaries
        - 'grid_shape': Shape of parameter grid
        - 'param_names': Names of swept parameters
    """
    logger.info(f"Starting parameter sweep over {len(parameter_grid)} configurations")

    results = {
        "parameters": [],
        "metrics": [],
        "grid_shape": parameter_grid.get_grid_shape(),
        "param_names": parameter_grid.param_names,
    }

    if save_trajectories:
        results["trajectories"] = []

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run sweep
    for i, params in enumerate(parameter_grid):
        logger.info(f"Running configuration {i + 1}/{len(parameter_grid)}: {params}")

        # Create config for this run
        config = base_config.copy()
        config.update(params)

        # Handle special parameter mappings
        config = _map_parameters_to_config(config, params)

        try:
            # Run integration
            trajectory = _run_single_config(config)

            # Compute metrics
            if metric_function is not None:
                metrics = metric_function(trajectory, config)
            else:
                metrics = _default_metrics(trajectory, config)

            # Store results
            results["parameters"].append(params)
            results["metrics"].append(metrics)

            if save_trajectories:
                results["trajectories"].append(trajectory)

        except Exception as e:
            logger.error(f"Failed for config {params}: {e}")
            results["parameters"].append(params)
            results["metrics"].append({"error": str(e)})
            if save_trajectories:
                results["trajectories"].append(None)

    # Save results
    if output_dir is not None:
        _save_sweep_results(results, output_dir)

        # Move to archive/incomplete if below minimum run threshold
        from optimization.result_io import relocate_incomplete_sweep

        relocate_incomplete_sweep(output_dir, min_runs=100)

    logger.info(f"Parameter sweep complete. {len(results['metrics'])} runs finished.")

    return results


def _map_parameters_to_config(
    config: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Map parameter names to appropriate config fields.

    Handles special cases like:
    - initial_energy_gev -> initial gamma
    - aperture_radius -> aperture config
    """
    mapped_config = config.copy()

    # Map energy to gamma
    if "initial_energy_gev" in params:
        energy_gev = params["initial_energy_gev"]
        # For electrons: E = γ * m_e * c^2, m_e*c^2 = 0.511 MeV
        rest_energy_mev = config.get("rest_energy_mev", 0.511)
        gamma = (
            energy_gev * 1e3 / rest_energy_mev
        )  # GeV to MeV, then divide by rest energy
        mapped_config["initial_gamma"] = gamma

    # Map aperture radius
    if "aperture_radius" in params:
        if "aperture" not in mapped_config:
            mapped_config["aperture"] = {}
        mapped_config["aperture"]["radius"] = params["aperture_radius"]

    # Map transverse momentum
    if "transverse_momentum" in params:
        mapped_config["initial_transverse_momentum"] = params["transverse_momentum"]

    # Map position spread
    if "position_spread" in params:
        mapped_config["initial_position_spread"] = params["position_spread"]

    # Map timestep
    if "timestep" in params:
        mapped_config["timestep"] = params["timestep"]

    # Map starting z position
    if "start_z" in params:
        mapped_config["initial_z"] = params["start_z"]

    return mapped_config


def _run_single_config(config: Dict[str, Any]) -> List[ParticleState]:
    """Run integration for a single configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    List[ParticleState]
        Trajectory of particle states
    """
    # NOTE: This is a placeholder implementation
    # The actual run_integrator function requires IntegratorConfig and ParticleState objects
    # This would need to be adapted based on your specific configuration structure
    # For now, this serves as a template that users can customize

    raise NotImplementedError(
        "run_single_config needs to be adapted to your specific configuration structure. "
        "Please see the example script for how to properly call run_integrator with "
        "IntegratorConfig and ParticleState objects."
    )


def _default_metrics(
    trajectory: List[ParticleState], config: Dict[str, Any]
) -> Dict[str, float]:
    """Compute default metrics for a trajectory.

    Parameters
    ----------
    trajectory : List[ParticleState]
        Trajectory to analyze
    config : Dict[str, Any]
        Configuration used for this run

    Returns
    -------
    Dict[str, float]
        Computed metrics
    """
    if len(trajectory) == 0:
        return {"error": "Empty trajectory"}

    initial_state = trajectory[0]
    rest_energy_mev = config.get("rest_energy_mev", 0.511)

    # Get aperture z position if available
    aperture_z = None
    if "aperture" in config and "z" in config["aperture"]:
        aperture_z = config["aperture"]["z"]

    metrics = compute_trajectory_metrics(
        trajectory,
        initial_state,
        rest_energy_mev,
        aperture_z=aperture_z,
    )

    return metrics


def _save_sweep_results(results: Dict[str, Any], output_dir: Path):
    """Save sweep results to disk.

    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
    output_dir : Path
        Output directory
    """
    # Save parameters and metrics as JSON
    output_data = {
        "parameters": results["parameters"],
        "metrics": results["metrics"],
        "grid_shape": results["grid_shape"],
        "param_names": results["param_names"],
    }

    json_path = output_dir / "sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved results to {json_path}")

    # Save as numpy arrays for easier analysis
    try:
        import numpy as np

        # Create arrays for each metric
        if len(results["metrics"]) > 0 and "error" not in results["metrics"][0]:
            metric_names = list(results["metrics"][0].keys())

            for metric_name in metric_names:
                values = [m.get(metric_name, np.nan) for m in results["metrics"]]
                array = np.array(values).reshape(results["grid_shape"])

                np_path = output_dir / f"{metric_name}.npy"
                np.save(np_path, array)

            logger.info(f"Saved numpy arrays to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not save numpy arrays: {e}")


def load_sweep_results(output_dir: Path) -> Dict[str, Any]:
    """Load previously saved sweep results.

    Parameters
    ----------
    output_dir : Path
        Directory containing sweep results

    Returns
    -------
    Dict[str, Any]
        Loaded results dictionary
    """
    json_path = Path(output_dir) / "sweep_results.json"

    with open(json_path, "r") as f:
        results = json.load(f)

    # Load numpy arrays if available
    try:
        import numpy as np

        results["arrays"] = {}
        for npy_file in Path(output_dir).glob("*.npy"):
            metric_name = npy_file.stem
            results["arrays"][metric_name] = np.load(npy_file)
    except Exception as e:
        logger.warning(f"Could not load numpy arrays: {e}")

    return results
