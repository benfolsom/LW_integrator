"""Headless sweep runner for CLI execution of parameter sweeps.

This module provides a standalone interface to run parameter sweeps without
requiring the GUI. It can be invoked from the command-line interface or
used programmatically.

Output Locations
----------------
When running sweeps via CLI, output is written to two locations:

1. **Results directory** (e.g., results/sweeps/YYYYMMDD_HHMMSS_configname/)
   - results.json: Parameter combinations and metrics
   - sweep.log: High-level progress summary

2. **logcache/** directory (same as GUI sweeps)
   - YYYYMMDD_HHMMSS_sweep_cli.log: Detailed debug output
   - Includes SC iterations, adaptive timestep details, etc.
   - Automatically rotated when files exceed 50 MB
   - Old logs purged when cache exceeds 500 MB

This matches the behavior of GUI sweeps, ensuring consistency in logging
and debugging workflows.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.constants import C_MMNS
from core.debug_logger import initialize_debug_logging, set_logging_context
from core.integration_runner import retarded_integrator
from core.smoothness_analyzer import SmoothnessConfig, analyze_trajectory_smoothness
from core.types import (
    ChronoMatchingMode,
    SimulationType,
    StartupMode,
)
from input_output.bunch_initialization import create_bunch_from_energy
from optimization.config import (
    OptimizationConfig,
    calculate_auto_steps,
    calculate_auto_timestep,
)
from optimization.metrics import compute_trajectory_metrics


class SweepRunner:
    """Execute parameter sweeps from configuration files without GUI."""

    def __init__(
        self, config: OptimizationConfig, output_dir: Path, verbose: bool = True
    ):
        """Initialize sweep runner.

        Parameters
        ----------
        config : OptimizationConfig
            Sweep configuration
        output_dir : Path
            Directory for results output
        verbose : bool, optional
            Whether to print progress messages, by default True
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        self.log_file = None

    def _log(self, message: str) -> None:
        """Log a message to stdout and log file with [OPTIMIZATION] prefix."""
        # Always print to stdout with [OPTIMIZATION] prefix (captured by debug_logger to logcache)
        print(f"[OPTIMIZATION] {message}", flush=True)

        # Also write to the sweep.log file in the results directory
        if self.log_file is not None:
            self.log_file.write(f"[OPTIMIZATION] {message}\n")
            self.log_file.flush()

    @staticmethod
    def _driver_ke_from_params(driver_params: Dict[str, Any]) -> float:
        """Derive driver kinetic energy in GeV from a driver_params dict.

        Uses the inverse of the kinetic-energy Pz formula:
            Pz = gamma * m * c * beta
            gamma*beta = |Pz| / (m * c)
            gamma = sqrt((gamma*beta)^2 + 1)
            KE = (gamma - 1) * m * c^2

        Returns
        -------
        float
            Kinetic energy in GeV (always positive).
        """
        m = driver_params["m_particle"]  # amu
        pz = abs(driver_params["starting_Pz"])  # amu·mm/ns
        AMU_TO_MEV = 931.494
        gamma_beta = pz / (m * C_MMNS) if m > 0 else 0.0
        gamma = np.sqrt(gamma_beta**2 + 1.0)
        ke_mev = (gamma - 1.0) * m * AMU_TO_MEV
        return ke_mev / 1e3  # GeV

    @staticmethod
    def _make_range(
        min_val: float, max_val: float, points: int, log_scale: bool = False
    ) -> List[float]:
        """Return a list of *points* values between *min_val* and *max_val*."""
        if points <= 1:
            return [(min_val + max_val) / 2.0]
        if log_scale and min_val > 0 and max_val > 0:
            return np.logspace(np.log10(min_val), np.log10(max_val), points).tolist()
        return np.linspace(min_val, max_val, points).tolist()

    def _generate_parameter_grids(self) -> Dict[str, List[float]]:
        """Generate parameter grids for sweep.

        In addition to the legacy aperture / energy / start_z / transv_offset
        grids, this now also generates grids for any BUNCH_TO_BUNCH sweep
        parameters whose ``*_range`` fields are populated on the config
        (e.g. ``driver_energy_range``, ``driver_starting_distance_range``).
        """
        grids = {}

        # ── Aperture grid (not used for BUNCH_TO_BUNCH, but kept for compat) ──
        if self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH:
            if self.config.aperture_points > 1:
                aper_min, aper_max = self.config.aperture_range
                grids["aperture"] = self._make_range(
                    aper_min,
                    aper_max,
                    self.config.aperture_points,
                    self.config.aperture_log_scale,
                )
            else:
                grids["aperture"] = [self.config.aperture_range[0]]

        # ── Energy grid (rider kinetic energy) ──
        if self.config.energy_points > 1:
            e_min, e_max = self.config.energy_range
            grids["energy"] = self._make_range(
                e_min,
                e_max,
                self.config.energy_points,
                self.config.energy_log_scale,
            )
        else:
            grids["energy"] = [self.config.energy_range[0]]

        # ── Starting z positions ──
        if (
            self.config.starting_z_positions
            and len(self.config.starting_z_positions) >= 1
        ):
            grids["start_z"] = self.config.starting_z_positions
        elif (
            self.config.starting_z_range is not None
            and self.config.starting_z_points > 1
        ):
            grids["start_z"] = np.linspace(
                self.config.starting_z_range[0],
                self.config.starting_z_range[1],
                self.config.starting_z_points,
            ).tolist()
        else:
            # Default: particle starts before wall
            grids["start_z"] = [self.config.wall_z - 100.0]

        # ── Transverse offsets ──
        if (
            self.config.transverse_offset_fractions
            and len(self.config.transverse_offset_fractions) >= 1
        ):
            # Use first value as the single scalar offset for this sweep
            grids["transv_offset_frac"] = [self.config.transverse_offset_fractions[0]]
        else:
            grids["transv_offset_frac"] = [0.0]

        # ── Wall-z sweep (optional) ──
        if self.config.wall_z_range is not None and self.config.wall_z_points > 1:
            grids["wall_z"] = self._make_range(
                self.config.wall_z_range[0],
                self.config.wall_z_range[1],
                self.config.wall_z_points,
                False,
            )

        # ── BUNCH_TO_BUNCH driver/rider sweep parameters ──
        # Each (range_attr, points_attr, grid_key, log_attr) tuple describes a
        # sweepable parameter.  If the range is populated with more than 1
        # point we add it to the grid; otherwise the fixed scalar value on the
        # config is used inside ``_run_single_integration``.
        _sweep_param_defs = [
            # rider params
            ("particle_mass_range", "particle_mass_points", "rider_m_particle", None),
            (
                "particle_charge_range",
                "particle_charge_points",
                "rider_charge_sign",
                None,
            ),
            ("particle_count_range", "particle_count_points", "rider_pcount", None),
            (
                "transverse_momentum_range",
                "transverse_momentum_points",
                "rider_transv_mom",
                None,
            ),
            (
                "transverse_spread_range",
                "transverse_spread_points",
                "rider_transv_dist",
                None,
            ),
            (
                "rider_stripped_ions_range",
                "rider_stripped_ions_points",
                "rider_stripped_ions",
                None,
            ),
            (
                "macroparticle_charge_range",
                "macroparticle_charge_points",
                "macroparticle_charge_multiplier",
                None,
            ),
            (
                "macroparticle_sigma_range",
                "macroparticle_sigma_points",
                "macroparticle_sigma_multiplier",
                None,
            ),
            # driver params
            ("driver_mass_range", "driver_mass_points", "driver_m_particle", None),
            (
                "driver_charge_sign_range",
                "driver_charge_sign_points",
                "driver_charge_sign",
                None,
            ),
            ("driver_pcount_range", "driver_pcount_points", "driver_pcount", None),
            (
                "driver_transv_mom_range",
                "driver_transv_mom_points",
                "driver_transv_mom",
                None,
            ),
            (
                "driver_transv_dist_range",
                "driver_transv_dist_points",
                "driver_transv_dist",
                None,
            ),
            (
                "driver_starting_distance_range",
                "driver_starting_distance_points",
                "driver_starting_distance",
                None,
            ),
            (
                "driver_energy_range",
                "driver_energy_points",
                "driver_energy_gev",
                "driver_energy_log_scale",
            ),
            (
                "driver_stripped_ions_range",
                "driver_stripped_ions_points",
                "driver_stripped_ions",
                None,
            ),
        ]

        for range_attr, points_attr, grid_key, log_attr in _sweep_param_defs:
            rng = getattr(self.config, range_attr, None)
            if rng is None:
                continue
            pts = getattr(self.config, points_attr, 1)
            if pts <= 1:
                continue
            # Skip driver params if not BUNCH_TO_BUNCH
            if (
                grid_key.startswith("driver_")
                and self.config.simulation_type != SimulationType.BUNCH_TO_BUNCH
            ):
                continue
            log_scale = getattr(self.config, log_attr, False) if log_attr else False
            min_val, max_val = float(rng[0]), float(rng[1])
            grids[grid_key] = self._make_range(min_val, max_val, pts, log_scale)

        return grids

    def _run_single_integration(
        self,
        aperture: float,
        energy_gev: float,
        start_z: float,
        transv_offset_frac: float,
        run_num: int,
        total_runs: int = 1,
        sweep_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run a single integration with given parameters.

        Parameters
        ----------
        aperture : float
            Aperture radius in mm
        energy_gev : float
            Particle energy in GeV
        start_z : float
            Starting z position in mm
        transv_offset_frac : float
            Transverse offset as fraction of aperture
        run_num : int
            Run number for tracking
        sweep_overrides : dict, optional
            Per-run overrides for rider/driver sweep parameters.  Keys are
            the grid names produced by ``_generate_parameter_grids`` (e.g.
            ``"driver_energy_gev"``, ``"driver_starting_distance"``, etc.).

        Returns
        -------
        Dict[str, Any]
            Result dictionary with metrics and trajectory info
        """
        if sweep_overrides is None:
            sweep_overrides = {}

        # ── Resolve rider parameters (sweep overrides > config) FIRST ──
        # These must be resolved before timestep calculation so that the
        # correct particle mass is used.
        rider_m_particle = sweep_overrides.get(
            "rider_m_particle", self.config.m_particle
        )
        rider_charge_sign = sweep_overrides.get(
            "rider_charge_sign", self.config.charge_sign
        )
        rider_pcount = int(sweep_overrides.get("rider_pcount", self.config.pcount))
        rider_transv_mom = sweep_overrides.get(
            "rider_transv_mom", self.config.transv_mom
        )
        rider_transv_dist = sweep_overrides.get(
            "rider_transv_dist", self.config.transv_dist
        )
        rider_stripped_ions = sweep_overrides.get(
            "rider_stripped_ions", self.config.stripped_ions
        )
        macro_charge_mult = sweep_overrides.get(
            "macroparticle_charge_multiplier",
            self.config.macroparticle_charge_multiplier,
        )
        macro_sigma_mult = sweep_overrides.get(
            "macroparticle_sigma_multiplier", self.config.macroparticle_sigma_multiplier
        )

        # Calculate transverse offset
        transv_offset = transv_offset_frac * aperture

        # Calculate timestep based on strategy (using resolved rider mass)
        if self.config.timestep_strategy == "auto_distance":
            timestep = calculate_auto_timestep(
                start_z=start_z,
                wall_z=self.config.wall_z,
                distance_past_wall=self.config.auto_steps_distance_past_wall,
                particle_energy_gev=energy_gev,
                particle_mass_amu=rider_m_particle,
                target_steps=self.config.auto_steps_target,
            )
        else:
            timestep = self.config.timestep

        # Calculate steps if auto mode enabled
        if self.config.auto_steps:
            steps = calculate_auto_steps(
                start_z=start_z,
                wall_z=self.config.wall_z,
                distance_past_wall=self.config.auto_steps_distance_past_wall,
                timestep=timestep,
                particle_energy_gev=energy_gev,
                particle_mass_amu=rider_m_particle,
            )
        else:
            steps = self.config.steps

        # Log timestep calculation details
        AMU_TO_MEV = 931.494
        rest_energy_mev = rider_m_particle * AMU_TO_MEV
        # Kinetic energy convention: γ = KE / E_rest + 1
        gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
        if gamma < 1.0:
            gamma = 1.0
        beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0

        print(
            f"[OPTIMIZATION]   [TIMESTEP] Run {run_num} strategy '{self.config.timestep_strategy}':",
            flush=True,
        )
        print(
            f"[OPTIMIZATION]     E={energy_gev:.4f} GeV, m={rider_m_particle:.4e} amu",
            flush=True,
        )
        print(f"[OPTIMIZATION]     gamma={gamma:.2f}, beta={beta:.8f}", flush=True)
        print(
            f"[OPTIMIZATION]     timestep h={timestep:.4e} ns (proper time = dt/gamma)",
            flush=True,
        )
        print(f"[OPTIMIZATION]     steps={steps}", flush=True)

        if self.config.timestep_strategy == "auto_distance":
            distance_per_step = beta * gamma * C_MMNS * timestep
            expected_total = distance_per_step * steps
            print(
                f"[OPTIMIZATION]     distance_per_step = β·γ·c·h = {distance_per_step:.4f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     expected_total_distance = {expected_total:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     wall_z={self.config.wall_z:.2f} mm, start_z={start_z:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     distance_to_wall = {abs(self.config.wall_z - start_z):.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     distance_past_wall={self.config.auto_steps_distance_past_wall:.2f} mm",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     target_steps={self.config.auto_steps_target}",
                flush=True,
            )

        # Log [START] line in format expected by plotting script
        # Use appropriate precision based on aperture magnitude
        if aperture >= 1.0:
            aperture_str = f"{aperture:.1f}"
        elif aperture >= 0.01:
            aperture_str = f"{aperture:.4f}"
        else:
            aperture_str = f"{aperture:.6f}"

        print(
            f"[OPTIMIZATION] [START] Run {run_num}/{total_runs}: a={aperture_str}mm, E={energy_gev:.2f}GeV",
            flush=True,
        )
        # Log additional details on separate line
        print(
            f"[OPTIMIZATION]   [PARAMS] z={start_z:.2f}mm, h={timestep:.4e}ns, N={steps}",
            flush=True,
        )
        print(
            f"[OPTIMIZATION]   [CONFIG] Run {run_num} stability settings:", flush=True
        )
        print(
            f"[OPTIMIZATION]     smoothness_enabled: {self.config.smoothness_enabled}",
            flush=True,
        )
        if self.config.smoothness_enabled:
            print(
                f"[OPTIMIZATION]     smoothness_window_size: {self.config.smoothness_window_size}",
                flush=True,
            )
            print(
                f"[OPTIMIZATION]     smoothness_reject_on_violation: {self.config.smoothness_reject_on_violation}",
                flush=True,
            )

        if aperture < 0.1:
            print(
                f"[OPTIMIZATION]   [DIAGNOSTIC] Run {run_num}: Small aperture detected ({aperture:.6f} mm)",
                flush=True,
            )

        print(
            f"[OPTIMIZATION]   [DEBUG] Calling run_testbed for Run {run_num}...",
            flush=True,
        )

        # Build rider params (using resolved overrides)
        # Kinetic energy convention: γ = KE / E_rest + 1
        AMU_TO_MEV = 931.494
        rest_energy_mev = rider_m_particle * AMU_TO_MEV
        gamma = (energy_gev * 1e3) / rest_energy_mev + 1.0
        if gamma < 1.0:
            gamma = 1.0
        rider_beta = np.sqrt(1.0 - 1.0 / (gamma * gamma)) if gamma > 1.0 else 0.0
        rider_pz = gamma * rider_m_particle * C_MMNS * rider_beta

        rider_params = {
            "starting_distance": start_z,
            "transv_mom": rider_transv_mom,
            "transv_dist": rider_transv_dist,
            "m_particle": rider_m_particle,
            "charge_sign": rider_charge_sign,
            "pcount": rider_pcount,
            "stripped_ions": rider_stripped_ions,
            "starting_Pz": rider_pz,
        }

        # Core params
        core_params = {
            "time_step": timestep,
            "wall_z": self.config.wall_z,
            "aperture_radius": aperture,
            "mean": 1.0e5,
            "cav_spacing": self.config.cavity_spacing,
            "z_cutoff": 0.0,
            "z_cutoff_mode": self.config.z_cutoff_mode,
        }

        # ── Resolve driver parameters (sweep overrides > config) ──
        driver_params = None
        driver_transv_offset = 0.0
        if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
            d_m = sweep_overrides.get(
                "driver_m_particle", self.config.driver_m_particle
            )
            d_charge = sweep_overrides.get(
                "driver_charge_sign", self.config.driver_charge_sign
            )
            d_pcount = int(
                sweep_overrides.get("driver_pcount", self.config.driver_pcount)
            )
            d_transv_mom = sweep_overrides.get(
                "driver_transv_mom", self.config.driver_transv_mom
            )
            d_transv_dist = sweep_overrides.get(
                "driver_transv_dist", self.config.driver_transv_dist
            )
            d_start_dist = sweep_overrides.get(
                "driver_starting_distance", self.config.driver_starting_distance
            )
            d_stripped = sweep_overrides.get(
                "driver_stripped_ions", self.config.driver_stripped_ions
            )
            d_energy_gev = sweep_overrides.get(
                "driver_energy_gev", self.config.driver_energy_gev
            )

            # Determine Pz sign from config direction setting
            # "-z" → negative Pz (driver moves toward rider, conventional)
            # "+z" → positive Pz (driver moves away from rider)
            driver_negative = getattr(self.config, "driver_direction", "-z") == "-z"
            pz_sign = -1.0 if driver_negative else 1.0

            # Kinetic energy convention: gamma = KE / E_rest + 1
            driver_gamma = (abs(d_energy_gev) * 1e3) / (d_m * AMU_TO_MEV) + 1.0
            if driver_gamma < 1.0:
                driver_gamma = 1.0
            driver_beta = (
                np.sqrt(1.0 - 1.0 / (driver_gamma * driver_gamma))
                if driver_gamma > 1.0
                else 0.0
            )
            driver_pz_mag = driver_gamma * d_m * C_MMNS * driver_beta
            driver_params = {
                "starting_distance": d_start_dist,
                "transv_mom": d_transv_mom,
                "transv_dist": d_transv_dist,
                "m_particle": d_m,
                "charge_sign": d_charge,
                "pcount": d_pcount,
                "stripped_ions": d_stripped,
                "starting_Pz": pz_sign * driver_pz_mag,
            }

            dir_label = "\u2212\u1e91" if driver_negative else "+\u1e91"
            print(
                f"[OPTIMIZATION]   [DRIVER] energy={d_energy_gev:.4f} GeV, "
                f"m={d_m:.4e} amu, gamma={driver_gamma:.4f}, "
                f"Pz={driver_params['starting_Pz']:.4e} ({dir_label}), "
                f"stripped={d_stripped:.2e}, pcount={d_pcount}",
                flush=True,
            )

        # Create particle states
        try:
            # Create rider bunch - use transverse_spread instead of transverse_radius
            rider_state, rest_energy_mev_rider = create_bunch_from_energy(
                kinetic_energy_mev=energy_gev * 1e3,
                mass_amu=rider_params["m_particle"],
                charge_sign=rider_params["charge_sign"],
                position_z=rider_params["starting_distance"],
                particle_count=rider_params["pcount"],
                transverse_spread=rider_params["transv_dist"],
                transverse_momentum=rider_params["transv_mom"],
                transverse_offset_x=transv_offset,
                transverse_offset_y=0.0,
            )

            # Set stripped ions
            rider_state["stripped_ions"] = np.full(
                rider_params["pcount"], rider_params["stripped_ions"]
            )

            # Create driver bunch if needed
            driver_state = None
            if driver_params is not None:
                # Recover energy magnitude and direction sign from the
                # already-built driver_params dict so these variables are
                # always in scope (avoids "possibly unbound" warnings).
                _driver_pz = driver_params["starting_Pz"]
                _driver_pz_sign = -1.0 if _driver_pz < 0 else 1.0
                _driver_ke_gev = self._driver_ke_from_params(driver_params)

                driver_state, _ = create_bunch_from_energy(
                    kinetic_energy_mev=_driver_ke_gev * 1e3,
                    mass_amu=driver_params["m_particle"],
                    charge_sign=driver_params["charge_sign"],
                    position_z=driver_params["starting_distance"],
                    particle_count=driver_params["pcount"],
                    transverse_spread=abs(driver_params["transv_dist"]),
                    transverse_momentum=driver_params["transv_mom"],
                    transverse_offset_x=driver_transv_offset,
                    transverse_offset_y=0.0,
                )
                # create_bunch_from_energy always produces positive Pz.
                # Apply direction derived from the driver_params starting_Pz sign.
                driver_state["Pz"] = _driver_pz_sign * np.abs(driver_state["Pz"])
                driver_state["bz"] = _driver_pz_sign * np.abs(driver_state["bz"])
                # Recompute Pt (magnitude unchanged, but recalc for consistency)
                driver_state["Pt"] = np.sqrt(
                    driver_state["Px"] ** 2
                    + driver_state["Py"] ** 2
                    + driver_state["Pz"] ** 2
                    + (driver_params["m_particle"] * C_MMNS) ** 2
                )
                driver_state["stripped_ions"] = np.full(
                    driver_params["pcount"], driver_params["stripped_ions"]
                )

        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"Failed to create particle states: {e}\n{traceback.format_exc()}",
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                },
            }

        # Run core integration
        try:
            # Create progress callback for step-by-step logging
            def progress_callback(current_step: int, total_steps: int):
                if current_step % 100 == 0:
                    progress_pct = (current_step / total_steps) * 100
                    print(
                        f"[OPTIMIZATION]     [PROGRESS] Run {run_num}: step {current_step}/{total_steps} ({progress_pct:.0f}%)",
                        flush=True,
                    )

            # Create logger callback for verbose messages (only if adaptive debug enabled)
            def logger_callback(message: str):
                if self.config.adaptive_timestep_debug:
                    print(f"[OPTIMIZATION]     [VERBOSE] {message}", flush=True)

            # Build self-consistency config
            from core.self_consistency import SelfConsistencyConfig

            sc_config = None
            if self.config.self_consistency_enabled:
                sc_config = SelfConsistencyConfig(
                    enabled=True,
                    target_ms_tolerance=self.config.self_consistency_tolerance,
                    max_iterations=self.config.self_consistency_max_iterations,
                    verbosity=self.config.self_consistency_verbosity,
                    chrono_interpolate=self.config.self_consistency_chrono_interpolate,
                    chrono_tolerance=self.config.self_consistency_chrono_tolerance,
                    chrono_high_precision=self.config.self_consistency_chrono_high_precision,
                    chrono_adaptive_tolerance=self.config.self_consistency_chrono_adaptive_tolerance,
                )

            # Build adaptive timestep config
            from core.integration_runner import AdaptiveTimestepConfig

            adaptive_config = None
            if self.config.adaptive_timestep_enabled:
                adaptive_config = AdaptiveTimestepConfig(
                    enabled=True,
                    energy_jump_threshold=self.config.adaptive_timestep_threshold,
                    timestep_reduction_factor=self.config.adaptive_timestep_reduction_factor,
                    min_timestep_factor=self.config.adaptive_timestep_min_factor,
                )

            rider_trajectory, driver_trajectory = retarded_integrator(
                steps=steps,
                h_step=timestep,
                wall_z=core_params["wall_z"],
                aperture_radius=core_params["aperture_radius"],
                sim_type=self.config.simulation_type,
                init_rider=rider_state,
                init_driver=driver_state,
                mean=core_params["mean"],
                cav_spacing=core_params["cav_spacing"],
                z_cutoff=core_params["z_cutoff"],
                chrono_mode=ChronoMatchingMode.AVERAGED,
                startup_mode=StartupMode.COLD_START,
                image_subcharge_count=self.config.image_subcharge_count,
                use_conducting_image_weighting=self.config.use_image_weighting,
                self_consistency=sc_config,
                adaptive_timestep=adaptive_config,
                macroparticle_charge_multiplier=macro_charge_mult,
                macroparticle_sigma_multiplier=macro_sigma_mult,
                macroparticle_use_momentum_errors=self.config.macroparticle_use_momentum_errors,
                bunch_transv_dist=rider_params["transv_dist"],
                bunch_transv_mom=rider_params["transv_mom"],
                progress_callback=progress_callback
                if self.config.log_verbosity == "full"
                else None,
                logger=logger_callback if self.config.log_verbosity == "full" else None,
            )

            print(
                f"[OPTIMIZATION]   [DEBUG] run_testbed completed for Run {run_num}",
                flush=True,
            )

            # Check if trajectory is valid
            if rider_trajectory is None or len(rider_trajectory) == 0:
                return {
                    "success": False,
                    "error": "Empty trajectory",
                    "parameters": {
                        "aperture": aperture,
                        "energy_gev": energy_gev,
                        "start_z": start_z,
                        "transv_offset": transv_offset,
                    },
                }

            # Compute metrics
            try:
                metrics = compute_trajectory_metrics(
                    trajectory=rider_trajectory,
                    initial_state=rider_trajectory[0],
                    rest_energy_mev=rest_energy_mev,
                    aperture_z=self.config.wall_z,
                )
            except (KeyError, IndexError) as e:
                import traceback

                return {
                    "success": False,
                    "error": f"Failed to compute metrics: {e}. Trajectory length: {len(rider_trajectory)}, First state keys: {list(rider_trajectory[0].keys()) if rider_trajectory else 'empty'}\n{traceback.format_exc()}",
                    "parameters": {
                        "aperture": aperture,
                        "energy_gev": energy_gev,
                        "start_z": start_z,
                        "transv_offset": transv_offset,
                    },
                }

            # Check smoothness if enabled
            if self.config.smoothness_enabled:
                # Convert trajectory list to dict format for smoothness analysis
                try:
                    trajectory_dict = {
                        "z": np.array([s["z"][0] for s in rider_trajectory]),
                        "gamma": np.array([s["gamma"][0] for s in rider_trajectory]),
                        "t": np.array([s["t"][0] for s in rider_trajectory]),
                    }
                except (KeyError, IndexError) as e:
                    return {
                        "success": False,
                        "error": f"Failed to extract trajectory data for smoothness analysis: {e}. Trajectory keys: {rider_trajectory[0].keys() if rider_trajectory else 'empty'}",
                        "parameters": {
                            "aperture": aperture,
                            "energy_gev": energy_gev,
                            "start_z": start_z,
                            "transv_offset": transv_offset,
                        },
                    }

                smoothness_config = SmoothnessConfig(
                    window_size=self.config.smoothness_window_size,
                    oscillation_threshold=self.config.smoothness_oscillation_threshold,
                    trend_smoothness_threshold=self.config.smoothness_trend_threshold,
                    max_allowed_violations=self.config.smoothness_max_violations,
                )
                smoothness_result = analyze_trajectory_smoothness(
                    trajectory_dict,
                    smoothness_config,
                    particle_mass_amu=self.config.m_particle,
                )
                metrics["smoothness_passed"] = smoothness_result.passed
                metrics["smoothness_violations"] = len(smoothness_result.violations)

                if (
                    self.config.smoothness_reject_on_violation
                    and not smoothness_result.passed
                ):
                    return {
                        "success": False,
                        "error": f"Smoothness violation: {len(smoothness_result.violations)} violations",
                        "parameters": {
                            "aperture": aperture,
                            "energy_gev": energy_gev,
                            "start_z": start_z,
                            "transv_offset": transv_offset,
                        },
                        "metrics": metrics,
                    }

            return {
                "success": True,
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                    "timestep": timestep,
                    "steps": steps,
                },
                "metrics": metrics,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "parameters": {
                    "aperture": aperture,
                    "energy_gev": energy_gev,
                    "start_z": start_z,
                    "transv_offset": transv_offset,
                },
            }

    def run(self) -> bool:
        """Execute the parameter sweep.

        Returns
        -------
        bool
            True if sweep completed successfully, False otherwise
        """
        import itertools

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open log file
        log_path = self.output_dir / "sweep.log"
        self.log_file = open(log_path, "w")

        # Initialize debug logging to logcache (like GUI sweeps)
        initialize_debug_logging(context="sweep_cli")
        set_logging_context("sweep_cli")

        # Save original verbosity settings before any overrides
        original_sc_verbosity = self.config.self_consistency_verbosity
        original_adaptive_debug = self.config.adaptive_timestep_debug

        try:
            # Apply log verbosity settings (like GUI does)

            if (
                self.config.log_verbosity == "none"
                or self.config.log_verbosity == "truncated"
            ):
                # Suppress detailed logging for non-full modes
                self.config.self_consistency_verbosity = 0
                self.config.adaptive_timestep_debug = False
            # else: "full" mode inherits settings from config (don't override)

            self._log("")
            self._log(f"Log verbosity: {self.config.log_verbosity}")
            if self.config.log_verbosity == "full":
                self._log("  Full debug logging enabled (inherits config settings)")
                self._log(f"    SC verbosity: {self.config.self_consistency_verbosity}")
                self._log(
                    f"    Adaptive timestep debug: {self.config.adaptive_timestep_debug}"
                )
            elif self.config.log_verbosity == "truncated":
                self._log("  Truncated logging (parameters + metrics + errors only)")
                self._log("    SC verbosity: 0 (overridden)")
                self._log("    Adaptive timestep debug: False (overridden)")
            elif self.config.log_verbosity == "none":
                self._log("  Debug logging disabled")
                self._log("    SC verbosity: 0 (overridden)")
                self._log("    Adaptive timestep debug: False (overridden)")
            self._log(
                f"Trajectory saving: Top N={self.config.save_top_n_trajectories}, All={self.config.save_all_trajectories}, Failed={self.config.save_failed_trajectories}"
            )

            if self.config.mode == "optimization":
                self._log("[ERROR] Optimization mode not yet supported in headless CLI")
                self._log("Please use the GUI for optimization runs")
                return False

            # Generate parameter grids
            param_grids = self._generate_parameter_grids()

            # Calculate total runs
            total_runs = 1
            for key, values in param_grids.items():
                total_runs *= len(values)

            # Log sweep start with total runs
            self._log(f"Starting BLIND SWEEP (Grid Search): {total_runs} total runs")
            self._log(f"  Simulation type: {self.config.simulation_type}")

            # Log each grid dimension
            for grid_key, grid_vals in param_grids.items():
                if len(grid_vals) > 1:
                    self._log(
                        f"  {grid_key}: {len(grid_vals)} points from {min(grid_vals):.4e} to {max(grid_vals):.4e}"
                    )
                else:
                    self._log(f"  {grid_key}: {grid_vals[0]:.4e} (fixed)")

            self._log(f"  Timestep strategy: {self.config.timestep_strategy}")
            if self.config.timestep_strategy == "auto_distance":
                self._log(
                    f"    Distance past wall: {self.config.auto_steps_distance_past_wall} mm"
                )
                self._log(
                    f"    Target steps for timestep calculation: {self.config.auto_steps_target}"
                )
                self._log(
                    "    All particles will travel to consistent z regardless of energy"
                )
            self._log(f"  z_cutoff_mode: {self.config.z_cutoff_mode}")

            # Log fixed particle parameters
            if self.config.simulation_type == SimulationType.BUNCH_TO_BUNCH:
                self._log("")
                self._log("  Fixed rider parameters:")
                self._log(f"    m_particle: {self.config.m_particle:.4e} amu")
                self._log(f"    charge_sign: {self.config.charge_sign}")
                self._log(f"    pcount: {self.config.pcount}")
                self._log(f"    stripped_ions: {self.config.stripped_ions:.2e}")
                self._log(f"    transv_mom: {self.config.transv_mom:.4e}")
                self._log(f"    transv_dist: {self.config.transv_dist:.4e}")
                self._log("  Fixed driver parameters:")
                self._log(f"    m_particle: {self.config.driver_m_particle:.4e} amu")
                self._log(f"    charge_sign: {self.config.driver_charge_sign}")
                self._log(f"    pcount: {self.config.driver_pcount}")
                self._log(f"    stripped_ions: {self.config.driver_stripped_ions:.2e}")
                self._log(f"    energy_gev: {self.config.driver_energy_gev:.4f}")
                self._log(
                    f"    starting_distance: {self.config.driver_starting_distance:.2f}"
                )

            self._log("")
            self._log(f"Output directory: {self.output_dir}")
            self._log("")

            # ── Build iteration over all grid dimensions ──
            param_names = list(param_grids.keys())
            param_values_lists = [param_grids[name] for name in param_names]

            # Names that are handled as positional args to _run_single_integration
            _positional_keys = {"aperture", "energy", "start_z", "transv_offset_frac"}

            # Run sweep
            start_time = time.time()
            run_num = 0
            failed_count = 0
            result = None  # Initialize result variable

            for param_combo in itertools.product(*param_values_lists):
                run_num += 1
                params_dict = dict(zip(param_names, param_combo))

                # Extract positional grid values
                aperture = params_dict.get("aperture", 1.0e-4)
                energy = params_dict["energy"]
                start_z = params_dict["start_z"]
                transv_offset_frac = params_dict.get("transv_offset_frac", 0.0)

                # Everything else → sweep_overrides passed to _run_single_integration
                sweep_overrides = {
                    k: v for k, v in params_dict.items() if k not in _positional_keys
                }

                # Log parameter values
                self._log(f"  [PARAMS] Run {run_num}/{total_runs} - All parameters:")
                for pname in param_names:
                    pval = params_dict[pname]
                    if isinstance(pval, float):
                        self._log(f"    {pname}: {pval:.6e}")
                    else:
                        self._log(f"    {pname}: {pval}")

                if self.config.macroparticle_enabled:
                    self._log(
                        f"    macroparticle_enabled: {self.config.macroparticle_enabled}"
                    )

                try:
                    result = self._run_single_integration(
                        aperture=aperture,
                        energy_gev=energy,
                        start_z=start_z,
                        transv_offset_frac=transv_offset_frac,
                        run_num=run_num,
                        total_runs=total_runs,
                        sweep_overrides=sweep_overrides,
                    )

                    # Attach run number and full parameter snapshot to result
                    result["run_number"] = run_num
                    if result.get("parameters") is None:
                        result["parameters"] = {}
                    result["parameters"].update(params_dict)

                    self.results.append(result)

                    if not result["success"]:
                        failed_count += 1
                        error_msg = result.get("error", "Unknown error")
                        self._log(f"  [FAILED] Run {run_num}/{total_runs}: {error_msg}")
                except Exception as e:
                    failed_count += 1
                    import traceback

                    error_detail = traceback.format_exc()
                    self._log(f"  [EXCEPTION] Run {run_num}/{total_runs}: {e}")
                    self._log("  Traceback:")
                    for line in error_detail.split("\n"):
                        if line:
                            self._log(f"    {line}")
                    self.results.append(
                        {
                            "run_number": run_num,
                            "success": False,
                            "error": f"{e}\n{error_detail}",
                            "parameters": dict(params_dict),
                        }
                    )
                    result = self.results[-1]

                if result.get("success"):
                    metrics = result.get("metrics", {})
                    # Log metrics in format compatible with plotting script
                    print(
                        f"[OPTIMIZATION] max_percent_energy_gain: {metrics.get('max_percent_energy_gain', 0):.6f}%",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.6e} GeV",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] max_relative_gain: {metrics.get('max_relative_gain', 0):.6e}",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] final_gamma: {metrics.get('final_gamma_mean', 1):.6f}",
                        flush=True,
                    )
                    print(
                        f"[OPTIMIZATION] initial_gamma: {metrics.get('initial_gamma_mean', 1):.6f}",
                        flush=True,
                    )

                    # Also log to sweep.log
                    self._log(f"  [RESULT] Run {run_num}/{total_runs}:")
                    self._log(
                        f"    max_energy_gain: {metrics.get('max_energy_gain_gev', 0):.6e} GeV"
                    )
                    self._log(
                        f"    max_percent_energy_gain: {metrics.get('max_percent_energy_gain', 0):.6f}%"
                    )
                    self._log(
                        f"    max_relative_gain: {metrics.get('max_relative_gain', 0):.6e}"
                    )
                    self._log(
                        f"    final_gamma: {metrics.get('final_gamma_mean', 1):.6f}"
                    )
                    self._log(
                        f"    initial_gamma: {metrics.get('initial_gamma_mean', 1):.6f}"
                    )

            # Save results
            elapsed_time = time.time() - start_time

            self._log("")
            self._log("=" * 80)
            self._log("SWEEP COMPLETE")
            self._log("=" * 80)
            self._log(f"Total runs: {total_runs}")
            self._log(f"Successful: {total_runs - failed_count}")
            self._log(f"Failed: {failed_count}")
            self._log(f"Elapsed time: {elapsed_time:.1f}s ({elapsed_time / 60:.1f}min)")
            self._log("=" * 80)

            # Save results to JSON
            results_path = self.output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(
                    {
                        "config": {
                            "simulation_type": self.config.simulation_type.name
                            if hasattr(self.config.simulation_type, "name")
                            else str(self.config.simulation_type),
                            "aperture_range": list(self.config.aperture_range),
                            "aperture_points": self.config.aperture_points,
                            "energy_range": list(self.config.energy_range),
                            "energy_points": self.config.energy_points,
                            "param_grids": {k: v for k, v in param_grids.items()},
                        },
                        "total_runs": total_runs,
                        "successful": total_runs - failed_count,
                        "failed": failed_count,
                        "elapsed_time_seconds": elapsed_time,
                        "results": self.results,
                    },
                    f,
                    indent=2,
                )

            self._log("")
            self._log(f"Results saved to: {results_path}")

            return True

        except KeyboardInterrupt:
            self._log("")
            self._log("")
            self._log("[INFO] Sweep interrupted by user")
            return False
        except Exception as e:
            self._log("")
            self._log("")
            self._log(f"[ERROR] {e}")
            import traceback

            for line in traceback.format_exc().split("\n"):
                if line:
                    self._log(f"  {line}")
            return False
        finally:
            # Restore original verbosity settings
            self.config.self_consistency_verbosity = original_sc_verbosity
            self.config.adaptive_timestep_debug = original_adaptive_debug

            if self.log_file is not None:
                self.log_file.close()


def _convert_json_config_to_dataclass(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON config format to OptimizationConfig dataclass format.

    JSON uses separate min/max/points fields (e.g., aperture_min, aperture_max),
    while OptimizationConfig uses tuple ranges (e.g., aperture_range).

    This also maps *fixed* (non-swept) sweep_parameters to the corresponding
    OptimizationConfig scalar fields so that the CLI sweep runner uses the
    correct particle parameters instead of hard-coded defaults.
    """
    converted = dict(config_dict)

    # Convert simulation_type string to SimulationType enum
    if "simulation_type" in converted and isinstance(converted["simulation_type"], str):
        sim_type_str = converted["simulation_type"].upper()
        if sim_type_str == "CONDUCTING_WALL":
            converted["simulation_type"] = SimulationType.CONDUCTING_WALL
        elif sim_type_str == "SWITCHING_WALL":
            converted["simulation_type"] = SimulationType.SWITCHING_WALL
        elif sim_type_str == "BUNCH_TO_BUNCH":
            converted["simulation_type"] = SimulationType.BUNCH_TO_BUNCH

    # Convert aperture fields
    if "aperture_min" in converted and "aperture_max" in converted:
        converted["aperture_range"] = (
            converted.pop("aperture_min"),
            converted.pop("aperture_max"),
        )

    # Convert energy fields
    if "energy_min" in converted and "energy_max" in converted:
        converted["energy_range"] = (
            converted.pop("energy_min"),
            converted.pop("energy_max"),
        )

    # Convert wall_z sweep fields
    if "wall_z_range" in converted and converted["wall_z_range"] is not None:
        wall_z_range = converted["wall_z_range"]
        if isinstance(wall_z_range, list) and len(wall_z_range) == 2:
            converted["wall_z_range"] = tuple(wall_z_range)

    # Map rider_stripped_ions top-level key -> stripped_ions (OptimizationConfig name)
    if "rider_stripped_ions" in converted and "stripped_ions" not in converted:
        converted["stripped_ions"] = float(converted.pop("rider_stripped_ions"))
    elif "rider_stripped_ions" in converted:
        converted.pop("rider_stripped_ions")

    # Map rider_offset / driver_offset top-level keys
    if "rider_offset_x" in converted:
        converted["transv_offset_x"] = float(converted.pop("rider_offset_x"))
    if "rider_offset_y" in converted:
        converted["transv_offset_y"] = float(converted.pop("rider_offset_y"))
    if "driver_offset_x" in converted:
        converted["driver_transv_offset_x"] = float(converted.pop("driver_offset_x"))
    if "driver_offset_y" in converted:
        converted["driver_transv_offset_y"] = float(converted.pop("driver_offset_y"))

    # Convert sweep_parameters to appropriate ranges and fixed values
    sweep_params = converted.get("sweep_parameters", {})

    # ── Mapping from sweep_parameter names to OptimizationConfig scalar fields ──
    _fixed_field_map_rider = {
        "rider_m_particle": "m_particle",
        "rider_charge_sign": "charge_sign",
        "rider_pcount": "pcount",
        "rider_transv_mom": "transv_mom",
        "rider_transv_dist": "transv_dist",
        "rider_stripped_ions": "stripped_ions",
        "macroparticle_charge_multiplier": "macroparticle_charge_multiplier",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_multiplier",
    }
    _fixed_field_map_driver = {
        "driver_m_particle": "driver_m_particle",
        "driver_charge_sign": "driver_charge_sign",
        "driver_pcount": "driver_pcount",
        "driver_transv_mom": "driver_transv_mom",
        "driver_transv_dist": "driver_transv_dist",
        "driver_starting_distance": "driver_starting_distance",
        "driver_energy_gev": "driver_energy_gev",
        "driver_stripped_ions": "driver_stripped_ions",
    }

    # ── Mapping from sweep_parameter names to OptimizationConfig range fields ──
    _range_field_map_rider = {
        "rider_m_particle": "particle_mass_range",
        "rider_charge_sign": "particle_charge_range",
        "rider_pcount": "particle_count_range",
        "rider_transv_mom": "transverse_momentum_range",
        "rider_transv_dist": "transverse_spread_range",
        "rider_stripped_ions": "rider_stripped_ions_range",
        "macroparticle_charge_multiplier": "macroparticle_charge_range",
        "macroparticle_sigma_multiplier": "macroparticle_sigma_range",
    }
    _range_field_map_driver = {
        "driver_m_particle": "driver_mass_range",
        "driver_charge_sign": "driver_charge_sign_range",
        "driver_pcount": "driver_pcount_range",
        "driver_transv_mom": "driver_transv_mom_range",
        "driver_transv_dist": "driver_transv_dist_range",
        "driver_starting_distance": "driver_starting_distance_range",
        "driver_energy_gev": "driver_energy_range",
        "driver_stripped_ions": "driver_stripped_ions_range",
    }

    # Process ALL sweep_parameters (both enabled=swept and disabled=fixed)
    all_param_names = list(_fixed_field_map_rider.keys()) + list(
        _fixed_field_map_driver.keys()
    )
    all_range_maps = {**_range_field_map_rider, **_range_field_map_driver}
    all_fixed_maps = {**_fixed_field_map_rider, **_fixed_field_map_driver}

    for param_name in all_param_names:
        if param_name not in sweep_params:
            continue
        param_config = sweep_params[param_name]

        if (
            param_config.get("enabled")
            and "min" in param_config
            and "max" in param_config
        ):
            # ── Swept parameter → range + points ──
            if param_name in all_range_maps:
                field_name = all_range_maps[param_name]
                min_val = float(param_config["min"])
                max_val = float(param_config["max"])
                # Energy magnitude is always positive
                if param_name == "driver_energy_gev":
                    min_val = abs(min_val)
                    max_val = abs(max_val)
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                converted[field_name] = (min_val, max_val)
                points_field = field_name.replace("_range", "_points")
                if "points" in param_config:
                    converted[points_field] = int(param_config["points"])
                # Also store log-scale flag for this sweep param
                if "log" in param_config:
                    log_field = field_name.replace("_range", "_log_scale")
                    converted[log_field] = bool(param_config["log"])
        else:
            # ── Fixed (disabled) parameter → scalar field ──
            if "fixed_value" in param_config and param_name in all_fixed_maps:
                scalar_field = all_fixed_maps[param_name]
                raw_val = param_config["fixed_value"]
                # pcount fields must be int
                if param_name in ("rider_pcount", "driver_pcount"):
                    converted[scalar_field] = int(float(raw_val))
                else:
                    converted[scalar_field] = float(raw_val)

    # Remove sweep_parameters from converted dict as it's been processed
    converted.pop("sweep_parameters", None)

    return converted


def run_sweep_from_config(
    config_path: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    verbosity_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run a parameter sweep from a configuration file.

    Parameters
    ----------
    config_path : Path
        Path to sweep configuration JSON file
    output_dir : Path, optional
        Output directory. If None, auto-generated from config name and timestamp
    verbose : bool, optional
        Whether to print progress messages, by default True
    verbosity_overrides : Dict[str, Any], optional
        Dictionary of verbosity settings to override config values.
        Supported keys: 'log_verbosity', 'self_consistency_verbosity', 'adaptive_timestep_debug'

    Returns
    -------
    bool
        True if sweep completed successfully, False otherwise
    """
    # Load config
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Convert JSON format to dataclass format
    converted_dict = _convert_json_config_to_dataclass(config_dict)

    # Filter to only include valid OptimizationConfig fields
    from dataclasses import fields

    valid_fields = {f.name for f in fields(OptimizationConfig)}
    filtered_dict = {k: v for k, v in converted_dict.items() if k in valid_fields}

    # Create OptimizationConfig
    config = OptimizationConfig(**filtered_dict)

    # Apply verbosity overrides from CLI arguments
    if verbosity_overrides:
        for key, value in verbosity_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(
                    f"[INFO] Overriding {key} from CLI: {value}",
                    flush=True,
                )

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config_path.stem
        output_dir = Path(config.output_dir) / f"{timestamp}_{config_name}"

    # Create and run sweep
    runner = SweepRunner(config, output_dir, verbose=verbose)
    return runner.run()
