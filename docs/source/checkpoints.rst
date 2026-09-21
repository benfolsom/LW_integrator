Resumable checkpoints
=====================

Long fixed-step bunch-to-bunch runs can write restart data after accepted
steps.  A checkpoint contains both rider and driver histories, including any
hidden inertial-prehistory rows, spin and magnetic diagnostics, particle-loss
masks, canonical momentum, and the Medina force-derivative history.  Restart
therefore continues from the same physical and numerical state instead of
starting a new trajectory from the last visible position.

The format is an append-only directory.  Immutable NumPy chunk files are
written first; ``manifest.json`` is replaced atomically only after the chunk is
flushed.  A process or machine loss can leave an unreferenced chunk, but the
manifest continues to name the last complete accepted step.

CLI use
-------

Start a testbed/GUI configuration with both step and wall-clock triggers:

.. code-block:: bash

   lw-simulate --testbed-config capture.json \
     --checkpoint-dir results/capture.checkpoint \
     --checkpoint-every-steps 1000 \
     --checkpoint-every-seconds 900

Resume the same configuration later:

.. code-block:: bash

   lw-simulate --testbed-config capture.json \
     --resume-from results/capture.checkpoint

The checkpoint records a fingerprint of the physics inputs and maintained
``core`` Python sources.  Changing the step count, timestep, initial bunches,
field/RFS configuration, or core implementation causes restart to fail before
integration.  This strict check is intentional: a restart must not silently
join trajectories produced by different equations or settings.

The native direct-integrator CLI uses the same flags with ``--config``.

Saved configuration and GUI
---------------------------

The full testbed JSON schema stores the same controls:

.. code-block:: json

   {
     "checkpoint": {
       "enabled": true,
       "directory": "results/capture.checkpoint",
       "resume_from": null,
       "interval_steps": 1000,
       "interval_seconds": 900.0
     }
   }

The main GUI exposes **Write resumable checkpoints**, checkpoint and resume
directory pickers, and both intervals in **Single Run Configuration**.  When no
checkpoint directory is entered, a new checkpoint directory is created inside
the run's timestamped output directory.  Selecting **Resume from** reuses that
checkpoint while final plots and trajectory exports go to the new GUI run
directory.

Current boundary
----------------

The first maintained format supports fixed-step ``BUNCH_TO_BUNCH`` runs.  It
rejects adaptive timestep, pseudo-grid, driver-train, and cavity-exit-tail
configurations before starting because those modes carry additional scheduler
state that is not yet serialized.  Ordinary fixed-step self-consistency,
particle-loss masks, exact inertial history, RFS spin, and Medina/LAD history
are included.

Checkpoint intervals limit lost work; they do not make the inner accepted step
interruptible.  The GUI **Cancel** action flushes the latest complete joint
rider/driver step before stopping.  A terminal interrupt, hard process kill,
or machine loss during one expensive step returns to the most recent committed
manifest boundary.

Variable-length exact-pair checkpoint
-------------------------------------

The exact-retarded adaptive pair path has a second checkpoint format for a
variable number of accepted electron--proton history knots. It
stores equal rider and driver chunks together with the adaptive-controller
state and the sparse public-output cursor. A focused interrupted/resumed run
reproduces the uninterrupted accepted histories, controller, and output-row
selection bit-for-bit.

Accepted-pair checkpoint schema 3 also reserves an optional JSON state for the
causal intrinsic-spin reduction history.  Second-order exact-retarded adaptive
runs record it automatically; its rider and driver histories advance in the
same joint acceptance transaction and are restored exactly on resume.
First-order runs write ``null`` and follow the unchanged trajectory path.
When intrinsic-spin self-reaction diagnostics are explicitly selected, the
same state includes bounded recent analytical/causal route records and
lifetime route counters.  Schema-1 and schema-2 accepted-pair checkpoints were
alpha-development artifacts and must be restarted rather than silently
interpreted without these state boundaries.

Enable it in a testbed configuration with ``adaptive_pair_return.enabled`` or
on the direct CLI with ``--adaptive-pair-return``. A direct CLI launch also
needs ``--adaptive-pair-target-time-ns`` and ``--checkpoint-dir``. The GUI
control **Adaptive exact pair return** enables checkpointing and exposes the
target time, tolerance scale, step-factor bounds, and shared-time tolerances.

For this format, ``checkpoint.interval_steps`` counts accepted history knots,
including refined midpoints and endpoints, rather than fixed outer-loop steps.
The wall-clock interval retains its ordinary meaning. GUI/CLI cancellation
flushes the latest jointly accepted pair before raising the cancellation
signal; rejected trials are never checkpointed.

The fixed-step options documented above continue to use the original
maintained format. See :doc:`multirate_return` for the strict mode guards and
the separation between accepted source history and public output.
