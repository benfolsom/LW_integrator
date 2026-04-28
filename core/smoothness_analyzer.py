"""Trajectory Smoothness Analysis Module

This module provides tools to assess whether particle trajectories are being
integrated smoothly by the stability mechanisms (adaptive timestep, self-consistency).

PHILOSOPHY:
We EXPECT large jumps in physical systems (radiation reaction, image charges, etc.).
The question is not "are there jumps?" but "are the jumps being handled smoothly
across multiple timesteps by our integrator?"

This analyzer checks for:
1. Multi-step trend stability (smooth evolution over windows)
2. Oscillatory instabilities (back-and-forth jumps)
3. Convergence when viewed at coarser resolution
4. Consistency of adaptive timestep response

Author: LW Integrator Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


class StabilityViolationType(Enum):
    """Types of numerical stability violations (NOT physical jumps)."""

    OSCILLATORY_INSTABILITY = "oscillatory_instability"
    TREND_DIVERGENCE = "trend_divergence"
    MULTI_SCALE_INCONSISTENCY = "multi_scale_inconsistency"
    TIMESTEP_INEFFECTIVENESS = "timestep_ineffectiveness"


@dataclass
class SmoothnessConfig:
    """Configuration for trajectory stability analysis.

    Parameters
    ----------
    enabled : bool
        Enable smoothness checking (default: True)
    window_size : int
        Number of steps for moving-window analysis (default: 20)
        Larger = more tolerant of short-term variations
    oscillation_threshold : float
        Maximum variance in sign changes (default: 0.5)
        Detects back-and-forth oscillations in energy/gamma
    trend_smoothness_threshold : float
        Maximum relative residual from polynomial fit (default: 0.30)
        How well does trajectory fit smooth trend?
    downsample_factor : int
        Factor for multi-scale check (default: 10)
        Trajectory should still be smooth when downsampled
    min_steps_for_analysis : int
        Minimum trajectory length (default: 50)
        Need enough steps for statistical analysis
    reject_on_violation : bool
        If True, reject runs with violations (default: True)
    max_allowed_violations : int
        Maximum violations before rejection (default: 3)
        Allows occasional outliers without rejecting entire run
    """

    enabled: bool = True
    window_size: int = 20
    oscillation_threshold: float = 0.5
    trend_smoothness_threshold: float = 0.30
    downsample_factor: int = 10
    min_steps_for_analysis: int = 50
    reject_on_violation: bool = True
    max_allowed_violations: int = 3

    @classmethod
    def strict(cls) -> "SmoothnessConfig":
        """Return strict configuration for critical simulations."""
        return cls(
            enabled=True,
            window_size=30,
            oscillation_threshold=0.3,
            trend_smoothness_threshold=0.20,
            reject_on_violation=True,
            max_allowed_violations=1,
        )

    @classmethod
    def permissive(cls) -> "SmoothnessConfig":
        """Return permissive configuration for exploratory runs."""
        return cls(
            enabled=True,
            window_size=15,
            oscillation_threshold=0.7,
            trend_smoothness_threshold=0.50,
            reject_on_violation=False,
            max_allowed_violations=10,
        )


@dataclass
class StabilityViolation:
    """Record of a numerical stability violation.

    Attributes
    ----------
    violation_type : StabilityViolationType
        Type of violation detected
    window_start : int
        Starting index of problematic window
    window_end : int
        Ending index of problematic window
    value : float
        Magnitude of the violation metric
    threshold : float
        Threshold that was exceeded
    severity : float
        How many times threshold was exceeded (value / threshold)
    description : str
        Human-readable description
    """

    violation_type: StabilityViolationType
    window_start: int
    window_end: int
    value: float
    threshold: float
    severity: float
    description: str


@dataclass
class SmoothnessAnalysisResult:
    """Result of trajectory stability analysis.

    Attributes
    ----------
    passed : bool
        True if trajectory is numerically stable
    violations : List[StabilityViolation]
        List of detected violations (empty if passed)
    num_steps_analyzed : int
        Number of trajectory steps analyzed
    oscillation_score : float
        Measure of oscillatory behavior (0=none, 1=severe)
    trend_smoothness_score : float
        How well trajectory fits smooth trend (lower=smoother)
    multi_scale_consistency : float
        Consistency between full and downsampled trajectory (lower=better)
    quality_summary : str
        Overall quality assessment
    """

    passed: bool
    violations: List[StabilityViolation]
    num_steps_analyzed: int
    oscillation_score: float
    trend_smoothness_score: float
    multi_scale_consistency: float
    quality_summary: str

    def __str__(self) -> str:
        """Format analysis result as string."""
        if self.passed:
            return (
                f"Stability check PASSED ({self.num_steps_analyzed} steps)\n"
                f"  Oscillation score: {self.oscillation_score:.3f}\n"
                f"  Trend smoothness: {self.trend_smoothness_score:.3f}\n"
                f"  Multi-scale consistency: {self.multi_scale_consistency:.3f}\n"
                f"  Quality: {self.quality_summary}"
            )
        else:
            lines = [
                f"Stability check FAILED ({len(self.violations)} violations):",
                f"  Oscillation score: {self.oscillation_score:.3f}",
                f"  Trend smoothness: {self.trend_smoothness_score:.3f}",
                f"  Quality: {self.quality_summary}",
                "",
                "Violations:",
            ]
            for v in self.violations:
                lines.append(f"  - {v.description}")
                lines.append(
                    f"    Steps {v.window_start}-{v.window_end}: {v.value:.3g} "
                    f"(threshold: {v.threshold:.3g}, severity: {v.severity:.1f}x)"
                )
            return "\n".join(lines)


def analyze_trajectory_smoothness(
    trajectory: Dict[str, np.ndarray],
    config: SmoothnessConfig,
    particle_mass_amu: float = None,
) -> SmoothnessAnalysisResult:
    """Analyze trajectory for numerical stability (not physical jumps).

    This function checks whether the integrator is handling the particle trajectory
    smoothly across multiple timesteps, even in regions where large physical forces
    cause rapid changes. It focuses on multi-step behavior rather than single-step jumps.

    Parameters
    ----------
    trajectory : Dict[str, np.ndarray]
        Trajectory dictionary with keys 'z', 'r', 'pz', 'pr', 't', 'gamma', etc.
    config : SmoothnessConfig
        Configuration for stability analysis
    particle_mass_amu : float, optional
        Particle mass in AMU (for energy calculations if needed)

    Returns
    -------
    SmoothnessAnalysisResult
        Analysis result with pass/fail status and quality metrics

    Notes
    -----
    Analysis focuses on:

    1. **Oscillatory instability**: Are there back-and-forth oscillations in energy/gamma?
       - Computed via moving-window sign-change analysis
       - Indicates numerical instability, not physical behavior

    2. **Trend smoothness**: Does trajectory fit a smooth polynomial over windows?
       - Uses quadratic fit to detect erratic behavior
       - Large residuals indicate poor numerical resolution

    3. **Multi-scale consistency**: Is downsampled trajectory still smooth?
       - Trajectory should be smooth at multiple time resolutions
       - Inconsistency suggests timestep is marginal

    All checks use **windowed statistics** to distinguish physical jumps (localized,
    smooth across windows) from numerical issues (oscillatory, erratic).

    Examples
    --------
    >>> from core.smoothness_analyzer import analyze_trajectory_smoothness, SmoothnessConfig
    >>> config = SmoothnessConfig()
    >>> result = analyze_trajectory_smoothness(trajectory, config)
    >>> if not result.passed:
    ...     print(f"Numerical instability detected: {result.quality_summary}")
    """
    if not config.enabled:
        return SmoothnessAnalysisResult(
            passed=True,
            violations=[],
            num_steps_analyzed=0,
            oscillation_score=0.0,
            trend_smoothness_score=0.0,
            multi_scale_consistency=0.0,
            quality_summary="Analysis disabled",
        )

    # Extract arrays
    try:
        z = np.asarray(trajectory.get("z", []))
        # Handle gamma carefully - don't convert None to array yet
        gamma_raw = trajectory.get("gamma", None)
        if gamma_raw is not None:
            gamma = np.asarray(gamma_raw)
        else:
            gamma = None
    except Exception as e:
        raise ValueError(f"Invalid trajectory format: {e}")

    # Early exit if gamma values are extreme (prevents hangs in polyfit)
    # Gamma > 1e9 indicates severe numerical instability
    if gamma is not None and len(gamma) > 0:
        max_gamma = np.max(np.abs(gamma))
        if max_gamma > 1e9 or np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
            return SmoothnessAnalysisResult(
                passed=False,
                violations=[
                    StabilityViolation(
                        violation_type=StabilityViolationType.TREND_DIVERGENCE,
                        window_start=0,
                        window_end=len(gamma),
                        value=max_gamma,
                        threshold=1e9,
                        severity=10.0,
                        description=f"Extreme gamma values detected (max: {max_gamma:.2e}) - severe numerical instability",
                    )
                ],
                num_steps_analyzed=len(gamma),
                oscillation_score=1.0,
                trend_smoothness_score=1.0,
                multi_scale_consistency=0.0,
                quality_summary="REJECTED - Extreme values (numerical breakdown)",
            )

    n_steps = len(z)

    # Check minimum length
    if n_steps < config.min_steps_for_analysis:
        return SmoothnessAnalysisResult(
            passed=True,
            violations=[],
            num_steps_analyzed=n_steps,
            oscillation_score=0.0,
            trend_smoothness_score=0.0,
            multi_scale_consistency=0.0,
            quality_summary=f"Too short for analysis (< {config.min_steps_for_analysis} steps)",
        )

    violations = []

    # Use gamma as primary energy proxy
    if gamma is None or gamma.ndim == 0 or len(gamma) != n_steps:
        # Fall back to position magnitude as crude proxy
        gamma = np.sqrt(z**2 + trajectory.get("r", np.zeros_like(z)) ** 2)

    # === 1. Oscillatory Instability Detection ===
    # Compute energy changes
    gamma_changes = np.diff(gamma) / (gamma[:-1] + 1e-100)

    # Moving window analysis of sign changes (oscillations)
    window = config.window_size
    oscillation_scores = []

    for i in range(0, len(gamma_changes) - window, window // 2):
        window_changes = gamma_changes[i : i + window]
        if len(window_changes) > 1:
            # Count sign changes (oscillations)
            signs = np.sign(window_changes)
            sign_changes = np.sum(np.abs(np.diff(signs))) / 2.0
            # Normalize by window size
            oscillation_score = sign_changes / len(window_changes)
            oscillation_scores.append(oscillation_score)

            if oscillation_score > config.oscillation_threshold:
                violations.append(
                    StabilityViolation(
                        violation_type=StabilityViolationType.OSCILLATORY_INSTABILITY,
                        window_start=i,
                        window_end=min(i + window, n_steps),
                        value=oscillation_score,
                        threshold=config.oscillation_threshold,
                        severity=oscillation_score / config.oscillation_threshold,
                        description=f"Oscillatory instability (sign-change rate: {oscillation_score:.2f})",
                    )
                )

    max_oscillation = max(oscillation_scores) if oscillation_scores else 0.0

    # === 2. Trend Smoothness Analysis ===
    # Fit polynomial to windowed segments and check residuals
    trend_residuals = []

    for i in range(0, n_steps - window, window // 2):
        window_indices = np.arange(i, min(i + window, n_steps))
        window_gamma = gamma[window_indices]

        if len(window_gamma) >= 3:
            # Fit quadratic polynomial (allows for acceleration)
            x = np.arange(len(window_gamma))
            try:
                # Check for extreme values before polyfit (can hang with large numbers)
                if np.max(np.abs(window_gamma)) > 1e8:
                    # Skip polyfit for extreme values, mark as unstable
                    trend_residuals.append(1.0)
                    violations.append(
                        StabilityViolation(
                            violation_type=StabilityViolationType.TREND_DIVERGENCE,
                            window_start=i,
                            window_end=min(i + window, n_steps),
                            value=1.0,
                            threshold=config.trend_smoothness_threshold,
                            severity=10.0,
                            description=f"Extreme values in window (max: {np.max(np.abs(window_gamma)):.2e})",
                        )
                    )
                    continue

                coeffs = np.polyfit(x, window_gamma, deg=2)
                fit = np.polyval(coeffs, x)
                residual = np.sqrt(np.mean((window_gamma - fit) ** 2))
                relative_residual = residual / (np.mean(np.abs(window_gamma)) + 1e-100)
                trend_residuals.append(relative_residual)

                if relative_residual > config.trend_smoothness_threshold:
                    violations.append(
                        StabilityViolation(
                            violation_type=StabilityViolationType.TREND_DIVERGENCE,
                            window_start=i,
                            window_end=min(i + window, n_steps),
                            value=relative_residual,
                            threshold=config.trend_smoothness_threshold,
                            severity=relative_residual
                            / config.trend_smoothness_threshold,
                            description=f"Erratic trend (residual: {relative_residual:.2f})",
                        )
                    )
            except (np.linalg.LinAlgError, ValueError, np.RankWarning):
                # Polynomial fit failed (degenerate data)
                pass

    max_trend_residual = max(trend_residuals) if trend_residuals else 0.0

    # === 3. Multi-Scale Consistency ===
    # Check that downsampled trajectory is still smooth
    stride = config.downsample_factor
    if n_steps >= stride * 5:  # Need at least 5 points after downsampling
        gamma_downsampled = gamma[::stride]
        gamma_changes_full = np.abs(np.diff(gamma)) / (gamma[:-1] + 1e-100)
        gamma_changes_down = np.abs(np.diff(gamma_downsampled)) / (
            gamma_downsampled[:-1] + 1e-100
        )

        # Compare statistics
        full_std = np.std(gamma_changes_full)
        down_std = np.std(gamma_changes_down)

        # Downsampled should have similar or lower variance (smoother)
        # If downsampled has HIGHER variance, suggests instability at fine scale
        if down_std > 0:
            consistency_ratio = full_std / down_std
            # Ratio < 1 is bad (downsampled is rougher than full resolution)
            if consistency_ratio < 0.7:
                multi_scale_score = 1.0 / consistency_ratio
                violations.append(
                    StabilityViolation(
                        violation_type=StabilityViolationType.MULTI_SCALE_INCONSISTENCY,
                        window_start=0,
                        window_end=n_steps,
                        value=multi_scale_score,
                        threshold=1.43,  # 1/0.7
                        severity=multi_scale_score / 1.43,
                        description=f"Multi-scale inconsistency (ratio: {consistency_ratio:.2f})",
                    )
                )
            multi_scale_consistency = abs(1.0 - consistency_ratio)
        else:
            multi_scale_consistency = 0.0
    else:
        multi_scale_consistency = 0.0

    # === Determine Overall Quality ===
    # Count violations by type
    violation_counts = {vtype: 0 for vtype in StabilityViolationType}
    for v in violations:
        violation_counts[v.violation_type] += 1

    # Quality assessment
    if max_oscillation > 0.7:
        quality = "Poor - severe oscillations detected"
    elif max_trend_residual > 0.5:
        quality = "Poor - highly erratic evolution"
    elif len(violations) > config.max_allowed_violations:
        quality = f"Marginal - {len(violations)} stability issues"
    elif max_oscillation > 0.3 or max_trend_residual > 0.2:
        quality = "Acceptable - minor numerical artifacts"
    else:
        quality = "Good - smooth integration"

    # Determine pass/fail
    passed = len(violations) <= config.max_allowed_violations

    if config.reject_on_violation and not passed:
        quality = f"REJECTED - {quality}"

    return SmoothnessAnalysisResult(
        passed=passed,
        violations=violations,
        num_steps_analyzed=n_steps,
        oscillation_score=max_oscillation,
        trend_smoothness_score=max_trend_residual,
        multi_scale_consistency=multi_scale_consistency,
        quality_summary=quality,
    )


def filter_stable_trajectories(
    results: List[Dict[str, Any]],
    config: SmoothnessConfig,
    particle_mass_amu: float = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter simulation results, keeping only numerically stable trajectories.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of simulation result dictionaries, each containing 'trajectory' key
    config : SmoothnessConfig
        Configuration for stability analysis
    particle_mass_amu : float, optional
        Particle mass in AMU
    verbose : bool
        Print filtering progress (default: True)

    Returns
    -------
    stable_results : List[Dict[str, Any]]
        Results that passed stability checks
    rejected_results : List[Dict[str, Any]]
        Results that failed stability checks (includes analysis result)

    Examples
    --------
    >>> from core.smoothness_analyzer import filter_stable_trajectories, SmoothnessConfig
    >>> config = SmoothnessConfig()
    >>> stable, rejected = filter_stable_trajectories(sweep_results, config)
    >>> print(f"Kept {len(stable)}/{len(sweep_results)} stable trajectories")
    """
    stable = []
    rejected = []

    for i, result in enumerate(results):
        if "trajectory" not in result:
            if verbose:
                print(f"Result {i}: No trajectory data, skipping")
            continue

        analysis = analyze_trajectory_smoothness(
            result["trajectory"],
            config,
            particle_mass_amu=particle_mass_amu,
        )

        if analysis.passed:
            stable.append(result)
            if verbose:
                print(f"Result {i}: PASSED - {analysis.quality_summary}")
        else:
            result["smoothness_analysis"] = analysis
            rejected.append(result)
            if verbose:
                print(f"Result {i}: FAILED - {analysis.quality_summary}")
                if config.reject_on_violation:
                    print("  -> REJECTED")

    return stable, rejected
