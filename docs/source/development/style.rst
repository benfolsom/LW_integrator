Style notes
===========

* Follow ``black``'s formatting and use type hints for all public
  functions.
* Keep imports explicit; relative imports inside packages are acceptable when
  referencing sibling modules.
* Avoid reformatting legacy files wholesale unless you are migrating them into
  the ``core/`` namespace.  Targeted edits help reviewers track physics changes.
* Matplotlib figures should use the colourblind-friendly palette introduced in
  ``core_vs_legacy_benchmark.py``.  When in doubt, import the palette from
  ``examples/validation/core_vs_legacy_benchmark.py`` to ensure consistency.
* Tests should prefer ``np.testing`` helpers for numerical comparisons and
  include explanatory comments when using loose tolerances.
