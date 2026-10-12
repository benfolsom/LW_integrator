Historical evidence and reproduction limits
===========================================

This index distinguishes inspectable source and tests from historical numerical
results whose complete artifacts are missing. The source baseline checked for
this documentation correction is commit
``3032cb4906e05da72c6f0f31dcf7267a5c5c0eb1``. Tracked test definitions and
scripts specify bounded checks; their presence is not evidence that a quoted
historical campaign passed, nor that every solver route converges physically.

High-gamma Stage 2 results cite ``codex_report_stage2.md``, which is absent.
The tracked ``tests/unit/test_high_gamma_precise_charge.py`` and
``tests/unit/test_high_gamma_stored_history.py`` use
``tests/unit/data/high_gamma_uniform_charge.json``. These supply inspectable
fixture-level controls, but not the missing measured-order run report.

PIC correction results cite ``codex_report_pic_correction_k_7_convergence.md``
and ``codex_report_pic_correction_k_8_coupling.md``, neither distributed here.
The tracked ``scripts/study_pic_correction_k_convergence.py``,
``scripts/summarize_pic_correction_k_convergence.py``, and
``scripts/study_pic_correction_k_coupling.py`` describe reproduction workflows.
Their historical arrays and comparison reports are not available through this
site. Preserve the reported close-crossing convergence failures and the
near-field reaction restrictions; algebraic kick closure is insufficient.

RFS M5 and Metal study pages retain hardware, workload, revision, and hash
qualifiers. Temporary filenames and hashes identify earlier runs but do not
provide downloadable evidence. The exact CPU speedup report and harness are
likewise located in a separate checkout, not this revision. Backend arithmetic
parity is distinct from independent trajectory and physical convergence.

The classical-limit studies name a feasibility-repository branch and directory,
but do not distribute a commit-pinned mapping from inputs and reference data to
solver revisions, convergence controls, and output tables. Their percentages,
endpoint comparisons, and 165 fm failure remain historical reports. The cited
papers do not validate these individual implementation runs.

Stochastic-emission study counts include pre-guard exact runs that are no longer
supported. External spectrum comparisons, detector-fit statistics, and the
failed magnetic proxy lack their complete versioned inputs and outputs here.
``tests/unit/test_stochastic_emission.py`` supplies inspectable code checks,
not material-transport or detector-spectrum validation. The proxy remains a
failure; no missing evidence is treated as a passing result.

The README ideal-screening figure also lacks its generating calculation,
exact-field comparison inputs, revision, and result table. Its former precise
gain and agreement claims have been removed. Historical sweep configurations
are not a substitute for those artifacts.

Closing these gaps requires a durable archive with exact source revisions,
input configurations, reference datasets and licenses, reproduction commands,
convergence settings, output arrays, checksums, and the associated report.
No such archive is published by this documentation correction. The remaining
historical claims are explicitly qualified rather than supplied with invented
links or reconstructed results.
