"""Repository-surface checks for maintained APIs and archived reference code."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_validation_examples_do_not_reintroduce_legacy_comparison_scripts():
    validation_dir = PROJECT_ROOT / "examples" / "validation"
    active_scripts = sorted(
        path.name for path in validation_dir.glob("*.py") if path.name != "__init__.py"
    )

    assert active_scripts == []


def test_legacy_plotting_helpers_are_not_active_python_modules():
    assert not (PROJECT_ROOT / "legacy" / "plotting_variables.py").exists()


def test_legacy_reference_tree_keeps_notebooks_only():
    legacy_dir = PROJECT_ROOT / "legacy"
    active_python_modules = sorted(path.name for path in legacy_dir.glob("*.py"))

    assert active_python_modules == []


def test_example_configs_do_not_use_removed_comparison_keys():
    removed_keys = {
        "legacy_enabled",
        "overlay_display",
        "overlay_save",
        "difference_display",
        "difference_save",
        "metrics_save",
    }
    config_paths = sorted((PROJECT_ROOT / "configs").glob("**/*.json"))
    config_paths.extend(sorted((PROJECT_ROOT / "examples").glob("**/*.json")))

    offenders = {}
    for path in config_paths:
        payload = json.loads(path.read_text())
        stale_keys = sorted(removed_keys.intersection(payload))
        if stale_keys:
            offenders[str(path.relative_to(PROJECT_ROOT))] = stale_keys

    assert offenders == {}


def test_tracked_docs_do_not_reference_local_worktree_notes():
    doc_paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CHANGELOG.md"]
    doc_paths.extend(sorted((PROJECT_ROOT / "docs" / "source").glob("**/*.rst")))

    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in doc_paths
        if "local/" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
