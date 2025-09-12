from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

from setuptools import setup, find_packages

# Read README for long description
with open("README_PRODUCTION.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lw-integrator",
    version="1.0.0",
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
