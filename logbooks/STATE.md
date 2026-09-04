# Current State

**Last updated:** 2026-09-04  
**Branch:** `quadratic-bezier-extension`  
**Current stage:** authorized comparison interrupted; preserved-state inspection pending  
**Status:** no completed comparative result; reruns and recovery suspended

## Closed evidence

The formal ranking-aware one-step comparison remains a strong positive result at 74.44% retrieval versus 31.44% for MSE-only.

The controlled long-horizon evidence remains closed. Latent ranking improved over random but missed the exact-pixel ratio criterion. The multi-scale prediction-only model achieved 97.61% four-way retrieval, while qualitative 128-candidate exact top-1 was 14% and the trajectory overpainted after step 50.

The exact-pixel RGB baseline and resolution-by-budget ablation are closed. The quality-priority 128x128/420 setting improved mean common-resolution MSE by 17.97% across the fixed five-target set at a 3.56x compute proxy.

## Manuscript state

All six ULAB-aligned chapters have complete v0.1 drafts in Notion. The pre-extension snapshot is dated 2026-09-03. Closed experiment conclusions cannot be rewritten by the new renderer study.

## Active bounded extension

The fixed comparison uses six deterministic procedural rights-safe targets, 128x128 planning, 512x512 evaluation, 420 accepted strokes, 64 candidates per pool, and seeds 73/137/211. Target hashes, source, tests, environment, and decision rules remain frozen.

Fail-closed runner source commit `398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b` passed all 210 tests and the unauthorized-output probe. Authorization commit `cc857407ed431c5583fd9e1c02a0ba619a8c187a` permits one completed comparison.

## Interrupted attempt

The first authorized attempt was manually interrupted after approximately 12 minutes when the user's internet connection dropped and Colab showed a reconnecting state. The heartbeat continued because the Colab kernel and comparison child process remained active. Pressing stop raised `KeyboardInterrupt` in the heartbeat sleep; the notebook handler then called `process.terminate()` and waited for the child process to stop.

The `.incomplete` directory must be preserved. The interruption is not a completed execution and provides no scientific result. No metrics or images should be opened, and no fresh execution should begin.

## Next action

Open `notebooks/quadratic_bezier_incomplete_run_inspection.ipynb`, run all four code cells, and return `quadratic_bezier_incomplete_diagnostic.json`. The notebook only hashes and inventories the saved state. It does not reveal metrics, open images, mutate Drive, resume execution, or start a new run. A recovery plan may be written only after this diagnostic is reviewed.
