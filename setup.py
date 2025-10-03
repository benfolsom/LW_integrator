from pathlib import Path
from typing import Any, Dict, cast

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent


def load_version() -> str:
    version_file = PROJECT_ROOT / "core" / "_version.py"
    namespace: Dict[str, Any] = {}
    with version_file.open("r", encoding="utf-8") as handle:
        exec(handle.read(), namespace)
    return cast(str, namespace["__version__"])


def read_long_description() -> str:
    primary = PROJECT_ROOT / "README_PRODUCTION.md"
    fallback = PROJECT_ROOT / "README.md"

    if primary.is_file():
        return primary.read_text(encoding="utf-8")
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8")
    return "LW Integrator"


package_version = load_version()
long_description = read_long_description()

setup(
    name="lw-integrator",
    version=package_version,
    author="Ben Folsom",
    author_email="",
    description="Production-ready Lienard-Wiechert electromagnetic field simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/LW_integrator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "numba>=0.50.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "docs": [
            "sphinx>=6.2,<7.0",
            "sphinx-rtd-theme>=1.3",
            "nbsphinx>=0.9",
            "ipykernel>=6.0",
            "sphinx-autobuild>=2021.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "lw-simulate=lw_integrator.cli:main",
        ],
    },
    keywords="physics electromagnetic simulation lienard-wiechert relativity",
    project_urls={
        "Bug Reports": "https://github.com/username/LW_integrator/issues",
        "Source": "https://github.com/username/LW_integrator",
        "Documentation": "https://lw-integrator.readthedocs.io/",
    },
)
