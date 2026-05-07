"""Helpers for parsing and summarizing optimization monitor log files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


MONITORED_INSIGHT_PARAMETERS = (
    "initial_energy_gev",
    "transverse_momentum",
    "rider_transv_dist",
    "driver_stripped_ions",
    "driver_transv_mom",
    "driver_transv_dist",
)

_EVALUATION_PATTERN = re.compile(r"Evaluation (\d+): (.+)")
_PARAMETER_PATTERN = re.compile(r"(\w+)=([\d.e+-]+)")
_GAIN_PATTERN = re.compile(r"max_percent_energy_gain:\s*([-\d.e+-]+)%")


def _parse_log_parameters(params_str: str) -> Dict[str, Any]:
    """Parse key=value parameter assignments from one evaluation log line."""
    params: Dict[str, Any] = {}
    for param_match in _PARAMETER_PATTERN.finditer(params_str):
        key = param_match.group(1)
        value_str = param_match.group(2)
        try:
            params[key] = float(value_str)
        except ValueError:
            params[key] = value_str
    return params


def parse_optimization_log(log_path: Path) -> list[Dict[str, Any]]:
    """Parse one optimization log file into evaluation result dictionaries."""
    results = []
    current_eval: Dict[str, Any] | None = None

    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                eval_match = _EVALUATION_PATTERN.search(line)
                if eval_match:
                    current_eval = {
                        "eval_num": int(eval_match.group(1)),
                        "params": _parse_log_parameters(eval_match.group(2)),
                        "log_file": log_path.name,
                        "timestamp": log_path.stat().st_mtime,
                    }
                    continue

                gain_match = _GAIN_PATTERN.search(line)
                if gain_match and current_eval:
                    current_eval["energy_gain"] = float(gain_match.group(1))
                    results.append(current_eval)
                    current_eval = None
    except (OSError, UnicodeDecodeError):
        return []

    return results


def select_optimization_log_files(
    logcache_dir: Path | str,
    *,
    latest_only: bool = False,
    specific_run: str | None = None,
) -> list[Path]:
    """Select optimization log files from a logcache directory."""
    log_dir = Path(logcache_dir)
    log_files = sorted(log_dir.glob("*optimization*.log"))

    if specific_run:
        return [log_file for log_file in log_files if specific_run in log_file.name]

    if latest_only and log_files:
        return [max(log_files, key=lambda log_file: log_file.stat().st_mtime)]

    return log_files


def analyze_optimization_logs(
    logcache_dir: Path | str,
    *,
    latest_only: bool = False,
    specific_run: str | None = None,
) -> tuple[list[Dict[str, Any]], list[Path]]:
    """Parse and rank optimization evaluations from matching log files."""
    log_files = select_optimization_log_files(
        logcache_dir,
        latest_only=latest_only,
        specific_run=specific_run,
    )

    all_results = []
    for log_file in log_files:
        all_results.extend(parse_optimization_log(log_file))

    all_results.sort(
        key=lambda result: result.get("energy_gain", float("-inf")),
        reverse=True,
    )
    return all_results, log_files


def collect_varied_parameters(results: Sequence[Dict[str, Any]]) -> list[str]:
    """Collect numeric parameter names that vary across result entries."""
    all_param_keys = set()
    for result in results:
        all_param_keys.update(result.get("params", {}).keys())

    varied_params = []
    for param in sorted(all_param_keys):
        values = [
            result.get("params", {}).get(param)
            for result in results
            if param in result.get("params", {})
        ]
        numeric_values = [value for value in values if isinstance(value, (int, float))]
        if len(numeric_values) > 1 and min(numeric_values) != max(numeric_values):
            varied_params.append(param)

    return varied_params


def summarize_parameter_ranges(
    results: Sequence[Dict[str, Any]],
    *,
    parameter_names: Iterable[str] = MONITORED_INSIGHT_PARAMETERS,
    top_fraction: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Summarize parameter ranges for the top-performing fraction of results."""
    if not results:
        return {}

    top_count = max(1, int(len(results) * top_fraction))
    top_performers = results[:top_count]
    summaries: Dict[str, Dict[str, float]] = {}

    for parameter in parameter_names:
        values = [
            result.get("params", {}).get(parameter)
            for result in top_performers
            if isinstance(result.get("params", {}).get(parameter), (int, float))
        ]
        if not values:
            continue
        summaries[parameter] = {
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return summaries


__all__ = [
    "MONITORED_INSIGHT_PARAMETERS",
    "analyze_optimization_logs",
    "collect_varied_parameters",
    "parse_optimization_log",
    "select_optimization_log_files",
    "summarize_parameter_ranges",
]
