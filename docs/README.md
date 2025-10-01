# LW Integrator Documentation (2025 refresh)

This directory hosts the refreshed documentation set for the LW (Liénard–Wiechert) Integrator.  The previous “user manual / developer guide / API reference” split had drifted out of sync with the codebase, so the docs now focus on the workflows that exist in the repository today.

## What’s here now?

```
docs/
├── README.md            # This guide
├── build_docs.sh        # Convenience wrapper around sphinx-build
└── source/              # Authoritative documentation sources
	 ├── index.rst        # Root table of contents
	 ├── overview.rst     # High-level tour of the project
	 ├── quickstart.rst   # Environment setup + first run
	 ├── validation.rst   # Regression and comparison workflows
	 ├── notebooks.rst    # How to use the interactive assets
	 ├── api/             # Lightweight API reference shells
	 └── development/     # Contribution & maintenance notes
```

Legacy pages that are no longer linked remain in `docs/source/` for historical reference; they can be deleted once their content has been migrated or confirmed obsolete.

## Building the docs

1. Ensure the documentation dependencies are available.  From the repository root and inside the project’s virtual environment, run:
	```bash
	pip install -e ".[docs]"
	```
	This installs Sphinx, `sphinx-autobuild`, and the supporting extensions declared in `setup.py`.  The `pyproject.toml` already provides the runtime libraries (NumPy, SciPy, etc.) that Sphinx imports while building modules.

2. From `docs/`, generate HTML output:
	```bash
	./build_docs.sh --clean --type html
	```
	The rendered site lives in `docs/build/html/index.html`.

3. For iterative writing, enable live reload:
	```bash
	./build_docs.sh --watch
	```

4. Other builders (`--type latex`, `--type epub`, …) are also wired through `build_docs.sh`.  The script wraps `sphinx-build -W --keep-going` so warnings break the build by default.

## Content principles

- **Reality first** – every page documents code that currently ships in `core/`, `input_output/`, or `examples/`.  
- **Link to source** – favour short narratives that send readers to modules, scripts, or notebooks in the repository.  The API pages use `automodule` blocks to pull docstrings straight from the code so they stay current.
- **Workflow oriented** – the main sections map to real tasks: setting up an environment, reproducing validation plots, or extending the integrator.  Keep tutorials in sync with the scripts and notebooks that they reference.

## Updating the docs alongside code changes

- Extend the relevant `.rst` page (or create a new one) when you add a publicly visible feature.
- Prefer `.. literalinclude::` or `:mod:` cross references to manual code snippets so refactors are harder to break.
- Review `docs/source/conf.py` if you introduce new optional dependencies or relocate packages.
- Run `./build_docs.sh --clean` before sending a PR; CI treats warnings as errors.

Questions or gaps?  Open an issue or drop a note in the repository’s documentation channel so we can keep the refreshed structure healthy.
