#!/usr/bin/env python3
"""
Script to systematically replace 'modern' with more professional terminology
throughout the LW integrator codebase.
"""

from pathlib import Path


def replace_in_file(filepath, replacements):
    """Replace text patterns in a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply replacements
        for old_pattern, new_pattern in replacements:
            content = content.replace(old_pattern, new_pattern)

        # Only write if changes were made
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to replace modern terminology."""

    # Define replacement patterns
    replacements = [
        # Class names
        ("LegacyUpdatedComparison", "LegacyUpdatedComparison"),
        # Method names
        ("create_updated_equivalent_bunch", "create_updated_equivalent_bunch"),
        ("run_updated_simulation", "run_updated_simulation"),
        ("_create_updated_bunch_from_legacy", "_create_updated_bunch_from_legacy"),
        ("conducting_flat_updated", "conducting_flat_updated"),
        ("switching_flat_updated", "switching_flat_updated"),
        ("_eqsofmotion_static_updated", "_eqsofmotion_static_updated"),
        # Variable names
        ("updated_integrator", "updated_integrator"),
        ("updated_bunch", "updated_bunch"),
        ("updated_rider", "updated_rider"),
        ("updated_driver", "updated_driver"),
        ("updated_traj_rider", "updated_traj_rider"),
        ("updated_traj_driver", "updated_traj_driver"),
        ("updated_init_rider", "updated_init_rider"),
        ("updated_init_driver", "updated_init_driver"),
        ("updated_step", "updated_step"),
        ("updated_z", "updated_z"),
        ("updated_mass", "updated_mass"),
        ("updated_gamma", "updated_gamma"),
        ("updated_energy", "updated_energy"),
        ("updated_pt", "updated_pt"),
        ("updated_rider_energies", "updated_rider_energies"),
        ("updated_val", "updated_val"),
        # Comments and docstrings
        ("updated standardized methods", "updated standardized methods"),
        ("updated integrator", "updated integrator"),
        ("updated ParticleSpecies", "updated ParticleSpecies"),
        ("updated implementations", "updated implementations"),
        ("updated bunch", "updated bunch"),
        ("updated code", "updated code"),
        ("legacy vs updated", "legacy vs updated"),
        ("Legacy vs Updated", "Legacy vs Updated"),
        ("updated equivalent", "updated equivalent"),
        ("updated simulation", "updated simulation"),
        ("updated bunches", "updated bunches"),
        ("Updated integrator", "Updated integrator"),
        ("Updated implementation", "Updated implementation"),
        ("Updated energy", "Updated energy"),
        ("Updated rider", "Updated rider"),
        ("Updated driver", "Updated driver"),
        ("Updated simulation", "Updated simulation"),
        ("Updated:", "Updated:"),
        ("Updated ", "Updated "),
        # File titles and descriptions
        ("Updated Code Comparison", "Updated Code Comparison"),
        ("Updated Trajectory Comparison", "Updated Trajectory Comparison"),
        ("legacy and updated", "legacy and updated"),
        ("Legacy and Updated", "Legacy and Updated"),
        ("updated vs legacy", "updated vs legacy"),
        ("Updated vs Legacy", "Updated vs Legacy"),
        # Print statements
        ("Running updated", "Running updated"),
        ("Updated final", "Updated final"),
        ("Updated steps", "Updated steps"),
        ("Fixed updated", "Fixed updated"),
        # Specific file patterns but exclude "Computer Modern" font reference
        ("updated implementations", "updated implementations"),
        ("updated calculations", "updated calculations"),
        ("updated physics", "updated physics"),
        # Plot labels and titles
        ("Updated Rider", "Updated Rider"),
        ("Updated Driver", "Updated Driver"),
        ("label='Updated", "label='Updated"),
        ("'Updated '", "'Updated '"),
    ]

    # Define files to process (exclude computer modern font reference in CSS)
    root_dir = Path("/home/benfol/work/LW_windows")

    # Get all Python files
    python_files = []
    for pattern in ["**/*.py", "**/*.md"]:
        python_files.extend(root_dir.glob(pattern))

    # Exclude certain files
    exclude_patterns = [
        "docs/source/_static/custom.css",  # Contains "Computer Modern" font
        ".git/",
        "__pycache__/",
        ".venv/",
        ".mypy_cache/",
        ".ruff_cache/",
        "lw_integrator.egg-info/",
    ]

    files_to_process = []
    for file_path in python_files:
        str_path = str(file_path)
        if not any(exclude in str_path for exclude in exclude_patterns):
            files_to_process.append(file_path)

    print(f"Processing {len(files_to_process)} files...")

    updated_count = 0
    for file_path in files_to_process:
        if replace_in_file(file_path, replacements):
            updated_count += 1

    print(f"\nCompleted! Updated {updated_count} files.")

    # Special handling for renamed files
    print("\nRenaming files with 'modern' in their names...")

    # Find files with 'modern' in name
    for file_path in root_dir.rglob("*modern*"):
        if file_path.is_file():
            new_name = str(file_path).replace("modern", "updated")
            new_path = Path(new_name)
            if not new_path.exists():
                file_path.rename(new_path)
                print(f"Renamed: {file_path} -> {new_path}")

    print("\nTerminology cleanup complete!")


if __name__ == "__main__":
    main()
