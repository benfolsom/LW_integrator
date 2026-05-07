Style notes
===========

* Follow ``black``'s formatting and use type hints for all public
  functions.
* Keep imports explicit; relative imports inside packages are acceptable when
  referencing sibling modules.
* Avoid reformatting archived reference files wholesale unless you are migrating
  them into the ``core/`` namespace.  Targeted edits help reviewers track
  physics changes.
* Matplotlib figures should use colourblind-friendly palettes and consistent
  labels.  Prefer shared plotting helpers in maintained modules over copying
  notebook-specific styling.
* Tests should prefer ``np.testing`` helpers for numerical comparisons and
  include explanatory comments when using loose tolerances.
