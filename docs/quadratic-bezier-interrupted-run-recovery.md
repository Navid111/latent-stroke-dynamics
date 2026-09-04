# Interrupted quadratic-Bezier comparison: preservation and inspection

## Event

The first authorized comparison attempt was manually interrupted after approximately 12 minutes when the user's internet connection dropped and Colab displayed a reconnecting state. The notebook heartbeat continued because the Colab kernel and child process were still active. Pressing stop raised `KeyboardInterrupt` in the parent heartbeat loop; the handler then terminated the comparison process and preserved the `.incomplete` output directory.

This is an interrupted attempt, not a completed execution and not a scientific result. It does not consume the authorization for one completed comparison. It must not be deleted, renamed, treated as final, or restarted from scratch before inspection.

## Immediate boundary

- Do not rerun Cell 5.
- Do not delete or rename `quadratic-bezier-fixed-comparison-v1.incomplete`.
- Do not open generated images, numerical summaries, plots, or method mappings.
- Do not infer an outcome from partial run order or file count.
- Preserve any completed output directory too, if one unexpectedly exists.

## Read-only inspection

Open `notebooks/quadratic_bezier_incomplete_run_inspection.ipynb` and run all four code cells. It mounts Drive and creates only `/content/quadratic_bezier_incomplete_diagnostic.json`. It does not modify the Drive output.

The diagnostic reports whether a completed or incomplete directory exists, how many run summaries are complete, whether those summaries and their artifact hashes pass, whether one partial run directory exists, and whether an aggregate or failure record was written. It deliberately withholds numerical metrics and does not open images.

## Recovery decision

Design a recovery only after the diagnostic is returned. A valid recovery must preserve completed run directories and hashes, quarantine rather than overwrite any partial run, execute only missing units under the same frozen source and authorization, and still produce exactly one completed aggregate result. No target, seed, condition, proposal, threshold, or source setting may change.
