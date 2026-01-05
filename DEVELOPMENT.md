# Development Guide

## Version Management

This project uses [bump2version](https://github.com/c4urself/bump2version) for automated version management.

### Prerequisites

Install bump2version:
```bash
pip install bump2version
```

### Releasing a New Version

The version number follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`

- **PATCH**: Bug fixes, small improvements (0.4.1 → 0.4.2)
- **MINOR**: New features, backward-compatible changes (0.4.2 → 0.5.0)
- **MAJOR**: Breaking changes (0.5.0 → 1.0.0)

#### Bump Patch Version (0.4.1 → 0.4.2)
```bash
bump2version patch
```

#### Bump Minor Version (0.4.1 → 0.5.0)
```bash
bump2version minor
```

#### Bump Major Version (0.4.1 → 1.0.0)
```bash
bump2version major
```

### What `bump2version` Does Automatically

1. Updates `core/_version.py` with the new version
2. Updates `.bumpversion.cfg` with the current version
3. Creates a Git commit with message: "Bump version to X.Y.Z"
4. Creates a Git tag: `vX.Y.Z`

### Pushing the Release

After running `bump2version`, push both the commit and tag:

```bash
# Push to both master and development
git push origin master development --tags
```

### Manual Version Check

Check current version:
```bash
python -c "from core._version import __version__; print(__version__)"
```

List all tags:
```bash
git tag --list
```

### Dry Run (Test Before Committing)

Preview what will change without making any commits:
```bash
bump2version --dry-run --verbose patch
```

---

## Development Workflow

### Branch Strategy

- **`master`**: Production-ready releases only
- **`development`**: Active development, integration testing

### Making Changes

1. Work on `development` branch
2. Test thoroughly
3. When ready to release:
   ```bash
   # On development branch
   git checkout development
   bump2version patch  # or minor/major
   
   # Switch to master and cherry-pick
   git checkout master
   git cherry-pick <commit-hash>
   
   # Push both branches
   git push origin master development --tags
   ```

### Reverting a Bad Release

If you need to remove a tag:
```bash
# Delete local tag
git tag -d v0.4.2

# Delete remote tag
git push origin :refs/tags/v0.4.2
```

---

## Code Quality

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
# Format with black
black .

# Check with ruff
ruff check .
```

### Type Checking
```bash
mypy .
```

---

## Documentation

Documentation is built with Sphinx and hosted on Read the Docs.

### Build Docs Locally
```bash
cd docs
make html
```

View at `docs/_build/html/index.html`

---

## Common Tasks

### Adding a New Feature

1. Create feature on `development` branch
2. Add tests in `lw_integrator/tests/`
3. Update relevant docstrings
4. Run tests and linting
5. Commit with descriptive message
6. When ready: bump version and release

### Fixing a Bug

1. Fix on `development` branch
2. Add regression test
3. Commit with message: "Fix: description of bug"
4. Bump patch version
5. Cherry-pick to `master`

### Updating Dependencies

Edit `setup.py` install_requires or extras_require sections, then:
```bash
pip install -e .
```

---

## Configuration Files

- `.bumpversion.cfg`: Version bumping configuration
- `pyproject.toml`: Build system, code formatting (black, ruff)
- `setup.py`: Package metadata and dependencies
- `.gitignore`: Files to exclude from version control
- `pytest.ini`: Test configuration

---

## Questions?

Check the main [README.md](README.md) or open an issue on GitHub.