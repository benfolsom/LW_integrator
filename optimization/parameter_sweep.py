"""Parameter sweep functionality for LW integrator optimization.

This module provides tools to run parameter sweeps over configuration spaces,
enabling systematic exploration of aperture sizes, energies, and other parameters
to find optimal configurations for maximum energy gain.
"""

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
