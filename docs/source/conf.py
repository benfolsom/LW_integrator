# Configuration file for Sphinx documentation builder
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
from typing import Any, Dict


def _load_version() -> str:
    version_file = project_root / "core" / "_version.py"
    namespace: Dict[str, Any] = {}
    with version_file.open("r", encoding="utf-8") as handle:
        exec(handle.read(), namespace)
    return namespace["__version__"]


# Add the project root to Python path
docs_dir = Path(__file__).parent.parent
project_root = docs_dir.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------

project = "LW Integrator"
copyright = "2025, LW Integrator Development Team"
author = "LW Integrator Development Team"
release = _load_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Automatic documentation from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.intersphinx",  # Link to other documentation
    "sphinx.ext.mathjax",  # Mathematics support
    "sphinx.ext.todo",  # Todo extension
    "sphinx.ext.githubpages",  # GitHub Pages support
    "nbsphinx",  # Jupyter notebook support
    "IPython.sphinxext.ipython_console_highlighting",  # IPython syntax highlighting
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__, steps, time_step, wall_position, aperture_radius, simulation_type, bunch_mean, cavity_spacing, z_cutoff",
}

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = None
html_favicon = None

# Custom CSS
html_css_files = ["custom.css"]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
        \usepackage{physics}
        \usepackage{siunitx}
        \DeclareMathOperator{\Tr}{Tr}
    """,
    "fncychap": "\\usepackage[Bjornstrup]{fncychap}",
    "printindex": "\\footnotesize\\raggedright\\printindex",
}

latex_documents = [
    (
        "index",
        "lw_integrator.tex",
        "LW Integrator Documentation",
        "LW Integrator Development Team",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------

man_pages = [("index", "lw_integrator", "LW Integrator Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        "index",
        "lw_integrator",
        "LW Integrator Documentation",
        author,
        "LW_Integrator",
        "Covariant electromagnetic particle tracking for accelerator physics.",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 60

# Math settings
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "macros": {
            "vec": ["\\boldsymbol{#1}", 1],
            "mat": ["\\boldsymbol{#1}", 1],
            "tensor": ["\\boldsymbol{#1}", 1],
            "unit": ["\\,\\mathrm{#1}", 1],
            "dd": "\\mathrm{d}",
            "pp": "\\partial",
            "cc": "\\mathrm{c}",
            "ee": "\\mathrm{e}",
        },
    }
}

# Todo configuration
todo_include_todos = True

# Source file suffixes
source_suffix = ".rst"

# Master document
master_doc = "index"

# Language
language = "en"

# Pygments style
pygments_style = "sphinx"

# -- Custom setup function ---------------------------------------------------


def setup(app):
    """Custom setup function for additional configuration"""
    app.add_css_file("custom.css")
