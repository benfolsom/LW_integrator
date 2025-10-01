# LW Integrator

The LW Integrator is a covariant charged-particle tracking code that evaluates
retarded Liénard–Wiechert potentials to obtain first-principles beam dynamics.
The repository contains a modernised ``core`` implementation that mirrors the
validated legacy solver, an updated Sphinx documentation set, and a collection
of validation scripts and notebooks.

---

## Contents

1. [Project overview](#project-overview)
2. [Repository layout](#repository-layout)
3. [Environment setup](#environment-setup)
4. [Running validation workloads](#running-validation-workloads)
5. [Documentation workflow](#documentation-workflow)
6. [Versioning and release notes](#versioning-and-release-notes)
7. [Development guidelines](#development-guidelines)
8. [Support](#support)

---

## Project overview

* **Physics focus.**  The code integrates particle trajectories using
  retarded-vector potentials and conjugate-momentum dynamics.  The ``core``
  package is a faithful transcription of the proven legacy solver and is kept in
  numerical lockstep by an integration test suite.
* **Documentation.**  The refreshed Sphinx site under ``docs/`` explains the
  theoretical background, quick-start workflows, validation procedures, and
  contributor guidance.  A new ``theory`` page summarises the covariant
  derivations drawn from the in-repo technical note.
* **Validation assets.**  The ``examples/validation`` tree provides both Python
  scripts and notebooks for reproducing benchmark comparisons between the
  modern and legacy implementations.

---

## Repository layout

```
LW_windows/
├── core/                 # Maintained integrator implementation and helpers
├── docs/                 # Sphinx configuration, sources, and build script
├── examples/
│   └── validation/       # CLI and notebook-based comparison studies
├── input_output/         # Particle bunch initialisation utilities
├── legacy/               # Archived original solver and notebooks
├── tests/                # Pytest suite covering physics and helper modules
├── .github/workflows/    # Continuous-integration pipelines (docs publishing)
├── core/_version.py      # Single source of truth for the project version
└── README.md             # You are here
```


---

## Environment setup

1. **Create and activate a virtual environment** (Python 3.8–3.13 are supported).

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the package in editable mode** with commonly used extras.

   ```bash
   pip install -e ".[dev,examples,docs]"
   ```

   * ``dev`` adds the lint/test toolchain.
   * ``examples`` installs notebook dependencies.
   * ``docs`` brings in Sphinx, ``sphinx-autobuild``, and related extensions.

3. **(Optional) register the kernel for Jupyter usage.**

   ```bash
   python -m ipykernel install --user --name lw-integrator --display-name "LW Integrator (.venv)"
   ```

---

## Running validation workloads

The canonical comparison between the core and legacy solvers lives in
``examples/validation/core_vs_legacy_benchmark.py``.  Execute it directly:

```bash
python examples/validation/core_vs_legacy_benchmark.py --seeds 0 1 2 --steps 5000 --plot
```

The script accepts additional options for output paths, DPI control, and plot
styling.  Consult ``--help`` for the full list.  Companion notebooks in the same
directory expose an interactive widget-driven interface for exploratory work.

The ``tests/`` directory contains deterministic Pytest suites that ensure
physics parity across configurations:

```bash
pytest tests
```

---

## Documentation workflow

All documentation sources are under ``docs/source/``.  The helper script
``docs/build_docs.sh`` wraps ``sphinx-build`` and ``sphinx-autobuild``.

* **One-off build** (HTML):

  ```bash
  cd docs
  ./build_docs.sh --clean --type html
  ```

* **Live preview with automatic reload** (requires ``sphinx-autobuild``):

  ```bash
  cd docs
  ./build_docs.sh --clean --watch
  ```

  The preview runs at ``http://localhost:8000`` as long as the process remains
  active.

GitHub Actions publishes the rendered site to GitHub Pages whenever the ``main``
branch is updated.  Every build also uploads the HTML artefact so intermediate
branches can download the output for review.

---

## Versioning and release notes

The project version is defined exactly once in ``core/_version.py``.  Both
``setup.py`` and ``docs/source/conf.py`` import that value, ensuring the wheel
metadata and Sphinx footer remain consistent.  To cut a new release:

1. Update ``__version__`` in ``core/_version.py``.
2. Commit the change alongside relevant release notes or change logs.
3. Tag and publish as needed; the packaging metadata is already aligned.

---

## Development guidelines

* Maintain parity between the ``core`` and ``legacy`` solvers when modifying
  physics-critical code.  New behaviours should be backed by updated validation
  plots and regression tests.
* Prefer the helper utilities in ``input_output/`` when constructing particle
  bunches.  They guarantee the integrator receives correctly shaped state
  dictionaries.
* Run the Pytest suite and build the documentation before submitting changes.
  The repository treats Sphinx warnings as errors to keep the rendered site
  trustworthy.
* The console entry point ``lw-simulate`` currently points to
  ``lw_integrator.cli:main``.  Implement ``lw_integrator/cli.py`` before relying
  on this executable in production scripts.

---

## Support

Discussion of new physics scenarios, validation additions, or documentation
improvements is welcome via GitHub issues.  When reporting a problem, please
include:

* the observed behaviour and expected outcome,
* the relevant configuration (energy range, simulation type, etc.), and
* reproduction steps or sample notebooks.

For background reading on the theoretical model, consult ``docs/source/theory``
and the accompanying technical note under ``LW_local_refs/main.tex``.
