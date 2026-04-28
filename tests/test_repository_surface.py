"""Repository-surface checks for maintained APIs and archived reference code."""

from __future__ import annotations

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
