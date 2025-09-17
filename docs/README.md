# LW Integrator Documentation

This directory contains the comprehensive documentation for the LW (Liénard-Wiechert) Integrator, a covariant electromagnetic particle tracking code for accelerator physics applications.

## Documentation Structure

```
docs/
├── source/           # Sphinx source files
├── build/           # Generated documentation (HTML, PDF)
├── user_manual/     # User manual chapters
├── developer_guide/ # Developer documentation
├── api_reference/   # API documentation
├── examples/        # Example notebooks and scripts
└── figures/         # Documentation figures and diagrams
```

## Building Documentation

### Prerequisites

Install the required packages:
```bash
pip install sphinx sphinx-rtd-theme numpydoc matplotlib jupyter nbsphinx
```

### Build Commands

Generate HTML documentation:
```bash
sphinx-build -b html source build/html
```

Generate PDF documentation:
```bash
sphinx-build -b latex source build/latex
cd build/latex && make
```

Auto-build with live reload during development:
```bash
sphinx-autobuild source build/html
```

## Documentation Standards

- Use reStructuredText (.rst) format for main documentation
- Include Jupyter notebooks (.ipynb) for examples and tutorials
- Follow NumPy docstring conventions for API documentation
- Include mathematical equations using LaTeX/MathJax
- Provide cross-references and links between sections
- Include code examples and verification scripts

## Contributing

When adding new features or modules:
1. Update relevant user manual sections
2. Add API documentation with docstrings
3. Include example usage in tutorials
4. Update this README if structure changes

For detailed contribution guidelines, see `developer_guide/contributing.rst`.