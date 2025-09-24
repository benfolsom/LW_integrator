#!/usr/bin/env python3
"""
Plot Output Manager

Centralized module for managing test plot outputs with timestamp organization.
All test files should use this module to avoid redundant code and ensure
consistent output organization.

Author: AI Assistant
Date: 2025-09-20
"""

import datetime
from pathlib import Path
from typing import Optional


class PlotOutputManager:
    """Manages plot outputs with timestamped directory organization."""

    def __init__(self, base_output_dir: str = "test_outputs"):
        """
        Initialize the plot output manager.

        Args:
            base_output_dir: Base directory for all test outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the date-stamped directory
        self.date_dir = self.base_output_dir / self.current_date
        self.timestamp_dir = self.date_dir / self.current_timestamp

        # Ensure directories exist
        self.timestamp_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, filename: str, subfolder: Optional[str] = None) -> Path:
        """
        Get the full output path for a plot file.

        Args:
            filename: Name of the plot file (should include extension)
            subfolder: Optional subfolder within the timestamp directory

        Returns:
            Full path where the plot should be saved
        """
        if subfolder:
            output_dir = self.timestamp_dir / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.timestamp_dir

        return output_dir / filename

    def save_plot(
        self,
        plt_object,
        filename: str,
        subfolder: Optional[str] = None,
        dpi: int = 150,
        bbox_inches: str = "tight",
    ) -> Path:
        """
        Save a matplotlib plot to the managed output directory.

        Args:
            plt_object: Matplotlib pyplot object or figure
            filename: Name of the plot file (should include extension)
            subfolder: Optional subfolder within the timestamp directory
            dpi: DPI for the saved image
            bbox_inches: Bounding box setting for saved image

        Returns:
            Path where the plot was saved
        """
        output_path = self.get_output_path(filename, subfolder)

        # Handle both plt and figure objects
        if hasattr(plt_object, "savefig"):
            plt_object.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        else:
            # Assume it's a figure object
            plt_object.figure.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)

        print(f"📊 Plot saved: {output_path}")
        return output_path

    def create_summary_file(
        self, content: str, filename: str = "test_summary.md"
    ) -> Path:
        """
        Create a summary file in the timestamp directory.

        Args:
            content: Content to write to the summary file
            filename: Name of the summary file

        Returns:
            Path where the summary was saved
        """
        summary_path = self.timestamp_dir / filename

        with open(summary_path, "w") as f:
            f.write(content)

        print(f"📄 Summary saved: {summary_path}")
        return summary_path

    def get_timestamp_dir(self) -> Path:
        """Get the current timestamp directory path."""
        return self.timestamp_dir

    def get_date_dir(self) -> Path:
        """Get the current date directory path."""
        return self.date_dir


# Convenience function for quick usage
def create_plot_manager(base_dir: str = "test_outputs") -> PlotOutputManager:
    """
    Create a PlotOutputManager instance.

    Args:
        base_dir: Base directory for test outputs

    Returns:
        Configured PlotOutputManager instance
    """
    return PlotOutputManager(base_dir)


# Example usage for tests:
"""
# At the top of your test file:
from utils.plot_output_manager import create_plot_manager

# Create the manager
plot_mgr = create_plot_manager()

# Save plots
plot_mgr.save_plot(plt, "my_test_result.png", subfolder="validation_tests")

# Create summary
summary = '''
# Test Results
- Test passed: ✅
- Energy gain: 5.2 MeV
- Validation: Complete
'''
plot_mgr.create_summary_file(summary, "validation_summary.md")
"""
